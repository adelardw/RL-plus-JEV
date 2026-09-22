"""Resume must restore *everything*, not just the weights.

A resumed run that silently restarts the step counter would double-spend the
reward budget and quietly invalidate the matched-budget comparison the paper
rests on, so this checks the parts that fail silently.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from rljevf.critic.jev_critic import LearnedValueCritic
from rljevf.ppo import PPOArgs, PPOTrainer
from rljevf.resume import find_resume, resume_step

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def _mk(tmp_path: Path, seed: int = 0, steps: int = 2):
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ds = Dataset.from_dict({"prompt": [f"Question number {i}?" for i in range(32)]})
    policy = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    ref = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    policy.config.use_cache = False
    critic = LearnedValueCritic(policy)

    def rew(prompts, completions, **kw):
        return [float(len(c)) / 100 for c in completions]

    args = PPOArgs(max_steps=steps, batch_prompts=2, micro_batch_size=1,
                   generation_batch_size=2, max_completion_length=8,
                   save_every=0, log_every=99, seed=seed, ppo_epochs=1)
    return PPOTrainer(policy=policy, ref_policy=ref, tokenizer=tok, reward_fn=rew,
                      critic=critic, args=args, train_dataset=ds,
                      out_dir=tmp_path, device="cpu")


def test_prompt_schedule_is_a_pure_function_of_seed_and_step(tmp_path):
    a = _mk(tmp_path / "a", seed=7)
    b = _mk(tmp_path / "b", seed=7)
    assert [a.batch_for_step(k) for k in range(5)] == [b.batch_for_step(k) for k in range(5)]
    c = _mk(tmp_path / "c", seed=8)
    assert a.batch_for_step(0) != c.batch_for_step(0)


def test_checkpoint_roundtrip_restores_step_optimizer_and_weights(tmp_path):
    t = _mk(tmp_path / "run", steps=2)
    t.train()
    assert t.step_idx == 2
    d = t.out_dir / "resume_state"
    assert d.is_dir(), "training must leave a resume_state directory"
    assert (d / "trainer_state.pt").exists() and (d / "policy").is_dir()

    before = {k: v.clone() for k, v in t.policy.state_dict().items()}
    opt_before = t.opt.state_dict()["state"]

    t2 = _mk(tmp_path / "run2", steps=4)
    assert t2.step_idx == 0
    assert t2.load_checkpoint(d)

    # the counter continues rather than restarting -- otherwise the resumed run
    # would re-spend the whole reward budget
    assert t2.step_idx == 2
    assert t2.reward_calls == t.reward_calls
    assert len(t2.history) == 2

    after = t2.policy.state_dict()
    for k, v in before.items():
        assert torch.allclose(v, after[k], atol=1e-5), f"weight {k} not restored"

    # optimiser moments must come back, or the resumed run is effectively
    # restarting Adam with a cold state
    opt_after = t2.opt.state_dict()["state"]
    assert set(opt_before) == set(opt_after)
    assert opt_after, "optimiser state is empty after resume"
    k0 = next(iter(opt_before))
    assert torch.allclose(opt_before[k0]["exp_avg"], opt_after[k0]["exp_avg"], atol=1e-6)
    assert opt_before[k0]["step"] == opt_after[k0]["step"]


def test_resumed_run_continues_the_same_data_order(tmp_path):
    t = _mk(tmp_path / "r", steps=2)
    t.train()
    t2 = _mk(tmp_path / "r2", steps=4)
    t2.load_checkpoint(t.out_dir / "resume_state")
    # step 2 in the resumed trainer is step 2 in an uninterrupted one
    assert t2.batch_for_step(t2.step_idx) == t.batch_for_step(2)


def test_find_resume_locates_and_dates_the_state(tmp_path):
    t = _mk(tmp_path / "R9-test", steps=1)
    t.train()
    found = find_resume("R9-test", tmp_path / "R9-test")
    assert found is not None and found.name == "resume_state"
    assert resume_step(found) == 1


def test_half_written_checkpoint_is_rejected(tmp_path):
    d = tmp_path / "run" / "resume_state"
    d.mkdir(parents=True)
    (d / "history.json").write_text("[]")      # no trainer_state.pt, no policy/
    assert find_resume("run", tmp_path / "run") is None
