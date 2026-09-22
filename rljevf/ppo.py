"""A compact PPO loop whose critic is a swappable component.

Written because (a) TRL 1.13 no longer ships a PPOTrainer, and (b) the paper's
central comparison needs the critic to be the *only* thing that changes between
two runs. With one implementation and two critic objects, the learned-value
arm and the Jev arm share every other line of code, so any difference between
them is attributable to the critic and not to the harness.

    critic = LearnedValueCritic(backbone)   -> textbook PPO
    critic = JevCritic(tokenizer)           -> frozen zero-shot external critic,
                                               no value head, no value loss
"""
from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F

from .critic.jev_critic import JevCritic, LearnedValueCritic


# --------------------------------------------------------------------------- #
def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    if dim is None:
        return (x * mask).sum() / mask.sum().clamp_min(1.0)
    return (x * mask).sum(dim) / mask.sum(dim).clamp_min(1.0)


def masked_whiten(x: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
    m = masked_mean(x, mask)
    var = masked_mean((x - m) ** 2, mask)
    out = (x - m) * torch.rsqrt(var + 1e-8)
    return out if shift_mean else out + m


def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """log p(label) computed as z_label - logsumexp(z).

    The obvious `log_softmax(...).gather(...)` allocates a second tensor the
    size of the logits, which for a 150k vocabulary is gigabytes per batch and
    was enough on its own to OOM a 16GB card.
    """
    logits = logits.float()
    chosen = torch.gather(logits, 2, labels.unsqueeze(-1)).squeeze(-1)
    return chosen - torch.logsumexp(logits, dim=-1)


def entropy_from_logits(logits: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    """Entropy over the vocabulary, chunked along time so peak memory stays
    proportional to `chunk` rather than to the full sequence."""
    outs = []
    for i in range(0, logits.shape[1], chunk):
        z = logits[:, i : i + chunk, :].float()
        lse = torch.logsumexp(z, dim=-1, keepdim=True)
        p = torch.exp(z - lse)
        outs.append((lse.squeeze(-1) - (p * z).sum(-1)))
    return torch.cat(outs, dim=1)


# --------------------------------------------------------------------------- #
@dataclass
class PPOArgs:
    learning_rate: float = 1e-6
    value_learning_rate: float = 1e-5
    kl_coef: float = 0.05
    gamma: float = 1.0
    lam: float = 0.95
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.5
    ppo_epochs: int = 2
    generation_batch_size: int = 8  # rollout chunk that fits in VRAM
    forward_chunk: int = 4          # no-grad scoring chunk (logits are B x T x |V|)
    batch_prompts: int = 16      # rollout prompts per optimisation step
    micro_batch_size: int = 4   # backward-pass chunk
    max_grad_norm: float = 1.0
    whiten_rewards: bool = True
    whiten_advantages: bool = True
    max_completion_length: int = 384
    max_prompt_length: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    max_steps: int = 250
    seed: int = 0
    log_every: int = 1
    save_every: int = 50
    reward_call_budget: int | None = None


class PPOTrainer:
    def __init__(
        self,
        policy,
        ref_policy,
        tokenizer,
        reward_fn: Callable[..., Sequence[float]],
        critic: JevCritic | LearnedValueCritic,
        args: PPOArgs,
        train_dataset,
        out_dir: Path,
        device: str | None = None,
        callbacks: Sequence[Callable[[dict], None]] = (),
    ):
        self.args = args
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.policy = policy.to(self.device)
        # A LoRA policy is its own reference: disabling the adapters recovers
        # the base model, which removes a full second copy from VRAM.
        self.peft_reference = ref_policy is None and hasattr(policy, "disable_adapter")
        if self.peft_reference:
            self.ref = None
        else:
            self.ref = ref_policy.to(self.device).eval()
            for p in self.ref.parameters():
                p.requires_grad_(False)
        self.tok = tokenizer
        self.reward_fn = reward_fn
        self.critic = critic
        self.external_critic = isinstance(critic, JevCritic)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.callbacks = list(callbacks)

        trainable = [p for p in self.policy.parameters() if p.requires_grad]
        params = [{"params": trainable, "lr": args.learning_rate}]
        if not self.external_critic:
            self.critic = critic.to(self.device)
            params.append(
                {"params": self.critic.v_head.parameters(), "lr": args.value_learning_rate}
            )
        self.opt = torch.optim.AdamW(params)

        # A fixed permutation indexed by step, rather than a DataLoader
        # iterator: the batch for step k is a pure function of (seed, k), so a
        # resumed run sees exactly the data an uninterrupted one would, and
        # every arm sees the same prompts in the same order.
        self.dataset = train_dataset
        self.prompt_order = torch.randperm(
            len(train_dataset), generator=torch.Generator().manual_seed(args.seed)
        ).tolist()
        self.history: list[dict] = []
        self.reward_calls = 0
        self.step_idx = 0
        self._interrupted = False

    def batch_for_step(self, step: int) -> list[str]:
        n = len(self.prompt_order)
        bs = self.args.batch_prompts
        start = (step * bs) % n
        idx = [self.prompt_order[(start + i) % n] for i in range(bs)]
        return [self.dataset[i]["prompt"] for i in idx]

    # -- rollout -------------------------------------------------------------
    @torch.no_grad()
    def generate(self, prompts: list[str]):
        """Rollout in generation-sized chunks, then pad the chunks to a common
        width. A full PPO batch of prompts does not fit in 16GB at once."""
        gb = max(1, self.args.generation_batch_size)
        if len(prompts) > gb:
            parts = [self._generate_chunk(prompts[i : i + gb]) for i in range(0, len(prompts), gb)]
            return _merge_rollouts(parts, self.tok.pad_token_id, self.device)
        return self._generate_chunk(prompts)

    @torch.no_grad()
    def _generate_chunk(self, prompts: list[str]):
        chats = [
            self.tok.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            for p in prompts
        ]
        self.tok.padding_side = "left"
        enc = self.tok(
            chats, return_tensors="pt", padding=True, truncation=True,
            max_length=self.args.max_prompt_length, add_special_tokens=False,
        ).to(self.device)
        self.policy.eval()
        out = self.policy.generate(
            **enc,
            max_new_tokens=self.args.max_completion_length,
            do_sample=True,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            pad_token_id=self.tok.pad_token_id,
        )
        self.policy.train()
        q_len = enc["input_ids"].shape[1]
        resp = out[:, q_len:]
        # mask everything at and after the first EOS
        eos = self.tok.eos_token_id
        is_eos = resp == eos
        lengths = torch.where(
            is_eos.any(1), is_eos.float().argmax(1) + 1,
            torch.full((resp.size(0),), resp.size(1), device=resp.device),
        ).long()
        idx = torch.arange(resp.size(1), device=resp.device)[None, :]
        mask = (idx < lengths[:, None]).long()
        return enc["input_ids"], enc["attention_mask"], resp, mask, lengths

    def _forward_logprobs(self, model, q_ids, q_mask, r_ids, r_mask):
        ids = torch.cat([q_ids, r_ids], dim=1)
        att = torch.cat([q_mask, r_mask], dim=1)
        logits = model(input_ids=ids, attention_mask=att, use_cache=False).logits
        logits = logits[:, q_ids.shape[1] - 1 : -1, :] / self.args.temperature
        return logprobs_from_logits(logits, r_ids), logits

    # -- one optimisation step ----------------------------------------------
    def step(self, prompts: list[str]) -> dict[str, Any]:
        """Retry once at half the chunk sizes if the card runs out of memory.

        VRAM headroom depends on the sampled completion lengths, so a batch
        that fit for 200 steps can still OOM at step 201. Shrinking and
        continuing is far better than losing the run."""
        for attempt in range(3):
            try:
                return self._step(prompts)
            except torch.cuda.OutOfMemoryError:
                if attempt == 2:
                    raise
                # A traceback keeps the failed step's tensors alive, so
                # without collecting first the retry would start with less
                # memory than the attempt that just failed.
                import gc

                gc.collect()
                torch.cuda.empty_cache()
                self.args.forward_chunk = max(1, self.args.forward_chunk // 2)
                self.args.micro_batch_size = max(1, self.args.micro_batch_size // 2)
                self.args.generation_batch_size = max(1, self.args.generation_batch_size // 2)
                print(f"CUDA OOM: retrying with forward_chunk="
                      f"{self.args.forward_chunk}, micro_batch_size="
                      f"{self.args.micro_batch_size}, generation_batch_size="
                      f"{self.args.generation_batch_size}", flush=True)
        raise RuntimeError("unreachable")

    def _step(self, prompts: list[str]) -> dict[str, Any]:
        t0 = time.perf_counter()
        a = self.args
        q_ids, q_mask, r_ids, r_mask, lengths = self.generate(prompts)
        completions = self.tok.batch_decode(
            [r[:l] for r, l in zip(r_ids, lengths)], skip_special_tokens=True
        )

        with torch.no_grad():
            old_logp = self._logprobs_chunked(self.policy, q_ids, q_mask, r_ids, r_mask)
            if self.peft_reference:
                with self.policy.disable_adapter():
                    ref_logp = self._logprobs_chunked(
                        self.policy, q_ids, q_mask, r_ids, r_mask)
            else:
                ref_logp = self._logprobs_chunked(self.ref, q_ids, q_mask, r_ids, r_mask)

        t_rew = time.perf_counter()
        scores = torch.tensor(
            _maybe_sync(self.reward_fn(prompts=prompts, completions=completions)),
            dtype=torch.float32, device=self.device,
        )
        self.reward_calls += len(prompts)
        reward_time = time.perf_counter() - t_rew

        # per-token KL shaping + terminal score (the textbook PPO-for-LM reward)
        kl = old_logp - ref_logp
        rewards = -a.kl_coef * kl * r_mask
        last = (lengths - 1).clamp_min(0)
        rewards[torch.arange(len(prompts), device=self.device), last] += scores
        if a.whiten_rewards:
            rewards = masked_whiten(rewards, r_mask, shift_mean=False)

        # -- values ----------------------------------------------------------
        t_val = time.perf_counter()
        diag: dict[str, Any] = {}
        if self.external_critic:
            values, diag = self.critic.values(
                prompts, [r.tolist() for r in r_ids], lengths.tolist()
            )
            values = values.to(self.device) * r_mask
        else:
            with torch.no_grad():
                values = self._critic_values_chunked(q_ids, q_mask, r_ids, r_mask) * r_mask
        value_time = time.perf_counter() - t_val

        advantages, returns = self._gae(rewards, values, r_mask, lengths)
        if a.whiten_advantages:
            advantages = masked_whiten(advantages, r_mask)

        # -- PPO epochs --------------------------------------------------------
        stats = {"pg_loss": [], "vf_loss": [], "clipfrac": [], "entropy": []}
        B = len(prompts)
        for _ in range(a.ppo_epochs):
            perm = torch.randperm(B)
            for i in range(0, B, a.micro_batch_size):
                sel = perm[i : i + a.micro_batch_size]
                mb = lambda t: t[sel]  # noqa: E731
                new_logp, logits = self._forward_logprobs(
                    self.policy, mb(q_ids), mb(q_mask), mb(r_ids), mb(r_mask)
                )
                m = mb(r_mask).float()
                ratio = torch.exp(new_logp - mb(old_logp))
                adv = mb(advantages)
                pg = torch.max(
                    -adv * ratio,
                    -adv * ratio.clamp(1 - a.cliprange, 1 + a.cliprange),
                )
                pg_loss = masked_mean(pg, m)
                loss = pg_loss
                vf_loss = torch.zeros((), device=self.device)
                if not self.external_critic:
                    v = self._critic_values(mb(q_ids), mb(q_mask), mb(r_ids), mb(r_mask))
                    v_clip = mb(values) + (v - mb(values)).clamp(
                        -a.cliprange_value, a.cliprange_value
                    )
                    ret = mb(returns)
                    vf_loss = 0.5 * masked_mean(
                        torch.max((v - ret) ** 2, (v_clip - ret) ** 2), m
                    )
                    loss = loss + a.vf_coef * vf_loss

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), a.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    stats["pg_loss"].append(float(pg_loss))
                    stats["vf_loss"].append(float(vf_loss))
                    stats["clipfrac"].append(
                        float(masked_mean(((ratio - 1).abs() > a.cliprange).float(), m))
                    )
                    stats["entropy"].append(float(masked_mean(entropy_from_logits(logits), m)))

        rec = {
            "reward_mean": float(scores.mean()),
            "reward_std": float(scores.std()) if len(scores) > 1 else 0.0,
            "kl_mean": float(masked_mean(kl, r_mask)),
            "value_mean": float(masked_mean(values, r_mask)),
            "adv_std": float(masked_whiten(advantages, r_mask).std()),
            "length_mean": float(lengths.float().mean()),
            "reward_time_s": round(reward_time, 3),
            "value_time_s": round(value_time, 3),
            "step_time_s": round(time.perf_counter() - t0, 3),
            **{k: (sum(v) / len(v) if v else 0.0) for k, v in stats.items()},
            **diag,
        }
        return rec

    @torch.no_grad()
    def _logprobs_chunked(self, model, q_ids, q_mask, r_ids, r_mask, chunk=None):
        chunk = chunk or max(1, self.args.forward_chunk)
        outs = []
        for i in range(0, q_ids.shape[0], chunk):
            sl = slice(i, i + chunk)
            lp, _ = self._forward_logprobs(model, q_ids[sl], q_mask[sl],
                                           r_ids[sl], r_mask[sl])
            outs.append(lp)
            del _
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def _critic_values_chunked(self, q_ids, q_mask, r_ids, r_mask, chunk=None):
        chunk = chunk or max(1, self.args.forward_chunk)
        outs = []
        for i in range(0, q_ids.shape[0], chunk):
            sl = slice(i, i + chunk)
            outs.append(self._critic_values(q_ids[sl], q_mask[sl], r_ids[sl], r_mask[sl]))
        return torch.cat(outs, dim=0)

    def _critic_values(self, q_ids, q_mask, r_ids, r_mask) -> torch.Tensor:
        ids = torch.cat([q_ids, r_ids], dim=1)
        att = torch.cat([q_mask, r_mask], dim=1)
        v = self.critic(ids, att)
        return v[:, q_ids.shape[1] - 1 : -1]

    def _gae(self, rewards, values, mask, lengths):
        a = self.args
        T = rewards.shape[1]
        adv = torch.zeros_like(rewards)
        last = torch.zeros(rewards.shape[0], device=rewards.device)
        for t in reversed(range(T)):
            nextv = values[:, t + 1] if t + 1 < T else torch.zeros_like(values[:, 0])
            delta = rewards[:, t] + a.gamma * nextv - values[:, t]
            last = delta + a.gamma * a.lam * last
            adv[:, t] = last
        adv = adv * mask
        return adv, (adv + values) * mask

    # -- loop ----------------------------------------------------------------
    def train(self) -> list[dict]:
        a = self.args
        self._install_signal_handlers()
        while self.step_idx < a.max_steps:
            step = self.step_idx
            prompts = self.batch_for_step(step)
            rec = self.step(prompts)
            rec["step"] = step
            self.history.append(rec)
            for cb in self.callbacks:
                cb(rec)
            if step % a.log_every == 0:
                print(
                    f"[{step:4d}] R={rec['reward_mean']:+.3f} KL={rec['kl_mean']:+.4f} "
                    f"len={rec['length_mean']:.0f} pg={rec['pg_loss']:+.4f} "
                    f"vf={rec['vf_loss']:.4f} t={rec['step_time_s']:.1f}s",
                    flush=True,
                )
            self.step_idx = step + 1
            if a.save_every and self.step_idx % a.save_every == 0:
                self.save_checkpoint()
            if self._interrupted:
                print("interrupted -- checkpointing and stopping", flush=True)
                self.save_checkpoint()
                return self.history
            if a.reward_call_budget and self.reward_calls >= a.reward_call_budget:
                print(f"reward-call budget reached ({self.reward_calls})", flush=True)
                break
        self.save("final")
        self.save_checkpoint()
        return self.history

    # -- interruption --------------------------------------------------------
    def _install_signal_handlers(self) -> None:
        """The guard sends SIGTERM and waits; use that window to checkpoint
        rather than losing the run."""
        import signal

        def on_term(signum, _frame):
            print(f"\nreceived signal {signum}: will checkpoint after this step",
                  flush=True)
            self._interrupted = True

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, on_term)
            except ValueError:
                pass  # not on the main thread

    # -- full state ----------------------------------------------------------
    def _rng_state(self) -> dict:
        import random

        st = {"python": random.getstate(), "torch": torch.get_rng_state()}
        if torch.cuda.is_available():
            st["cuda"] = torch.cuda.get_rng_state_all()
        try:
            import numpy as np

            st["numpy"] = np.random.get_state()
        except ImportError:
            pass
        return st

    def _load_rng(self, st: dict) -> None:
        import random

        try:
            random.setstate(st["python"])
            torch.set_rng_state(st["torch"].cpu() if hasattr(st["torch"], "cpu") else st["torch"])
            if torch.cuda.is_available() and "cuda" in st:
                torch.cuda.set_rng_state_all([x.cpu() for x in st["cuda"]])
            if "numpy" in st:
                import numpy as np

                np.random.set_state(st["numpy"])
        except Exception as e:  # noqa: BLE001
            print(f"warning: could not restore RNG state ({e}); "
                  f"the resumed run is not bit-identical", flush=True)

    def save_checkpoint(self, name: str = "resume_state") -> Path:
        """Everything needed to continue: weights, optimiser, step counter,
        RNG, the reward-call meter and the history.

        Deliberately NOT called `checkpoint-*`: the Kaggle runner deletes
        directories matching that glob to keep the session output small, and
        this is the one thing that must survive.
        """
        d = self.out_dir / name
        tmp = self.out_dir / (name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)

        self.policy.save_pretrained(tmp / "policy")
        self.tok.save_pretrained(tmp / "policy")
        state = {
            "step_idx": self.step_idx,
            "reward_calls": self.reward_calls,
            "optimizer": self.opt.state_dict(),
            "rng": self._rng_state(),
            "prompt_order": self.prompt_order,
            "args": asdict(self.args),
            "external_critic": self.external_critic,
        }
        if not self.external_critic:
            state["v_head"] = self.critic.v_head.state_dict()
        else:
            state["calibration"] = self.critic.calibration.to_dict()
        torch.save(state, tmp / "trainer_state.pt")
        (tmp / "history.json").write_text(json.dumps(self.history, indent=1))
        # atomic-ish swap so a kill mid-write cannot leave a corrupt checkpoint
        if d.exists():
            shutil.rmtree(d)
        tmp.rename(d)
        print(f"checkpoint saved: step {self.step_idx} -> {d}", flush=True)
        return d

    def load_checkpoint(self, path: str | Path) -> bool:
        d = Path(path)
        sf = d / "trainer_state.pt"
        if not sf.exists():
            return False
        state = torch.load(sf, map_location="cpu", weights_only=False)
        from transformers import AutoModelForCausalLM

        loaded = AutoModelForCausalLM.from_pretrained(
            d / "policy", dtype=next(self.policy.parameters()).dtype
        )
        self.policy.load_state_dict(loaded.state_dict())
        self.policy.to(self.device)
        del loaded
        self.opt.load_state_dict(state["optimizer"])
        if not self.external_critic and "v_head" in state:
            self.critic.v_head.load_state_dict(state["v_head"])
        if self.external_critic and "calibration" in state:
            from .critic.jev_critic import Calibration

            self.critic.calibration = Calibration(**state["calibration"])
        self.step_idx = int(state["step_idx"])
        self.reward_calls = int(state.get("reward_calls", 0))
        self.prompt_order = state.get("prompt_order", self.prompt_order)
        self._load_rng(state.get("rng", {}))
        hj = d / "history.json"
        if hj.exists():
            self.history = json.loads(hj.read_text())
        print(f"resumed from {d} at step {self.step_idx} "
              f"({self.reward_calls} reward calls already spent)", flush=True)
        return True

    def save(self, name: str) -> Path:
        d = self.out_dir / name
        d.mkdir(parents=True, exist_ok=True)
        self.policy.save_pretrained(d)
        self.tok.save_pretrained(d)
        (self.out_dir / "history.json").write_text(json.dumps(self.history, indent=1))
        return d



def _merge_rollouts(parts, pad_id: int, device):
    """Concatenate rollout chunks whose prompt and response widths differ.
    Prompts are left-padded, responses right-padded, so each side is padded on
    the side that keeps its alignment."""
    qw = max(p[0].shape[1] for p in parts)
    rw = max(p[2].shape[1] for p in parts)

    def padl(t, w, v):
        return torch.nn.functional.pad(t, (w - t.shape[1], 0), value=v)

    def padr(t, w, v):
        return torch.nn.functional.pad(t, (0, w - t.shape[1]), value=v)

    q_ids = torch.cat([padl(p[0], qw, pad_id) for p in parts])
    q_mask = torch.cat([padl(p[1], qw, 0) for p in parts])
    r_ids = torch.cat([padr(p[2], rw, pad_id) for p in parts])
    r_mask = torch.cat([padr(p[3], rw, 0) for p in parts])
    lengths = torch.cat([p[4] for p in parts])
    return q_ids.to(device), q_mask.to(device), r_ids.to(device), r_mask.to(device), lengths.to(device)


def _maybe_sync(x):
    """reward_fn may be async (TRL-style) or plain."""
    if hasattr(x, "__await__"):
        from .jevclient import _run_sync

        return _run_sync(x)
    return x
