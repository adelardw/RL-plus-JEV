"""Central configuration. Every experimental knob lives here so that runs differ
only in the fields the paper claims they differ in."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = Path(os.environ.get("RLJEVF_RUN_ROOT", PROJECT_ROOT / "runs"))
CACHE_ROOT = Path(os.environ.get("RLJEVF_CACHE_ROOT", PROJECT_ROOT / ".cache"))

# --- models -----------------------------------------------------------------
POLICY_MODEL = os.environ.get("RLJEVF_POLICY", "Qwen/Qwen2.5-0.5B-Instruct")
# RLAIF judge: a hosted instruct model read from logits (parse-free, like Jev).
# Chosen over a local judge because the sizes that fit on 2x T4 measure at or
# near chance on held-out preference pairs (paper, Experiment 0).
API_JUDGE_MODEL = os.environ.get("RLJEVF_API_JUDGE", "deepseek/deepseek-v4-flash-0731")
# Local judge, still used for the self-judge arm and as a local-cost reference.
JUDGE_MODEL = os.environ.get("RLJEVF_JUDGE", "Qwen/Qwen2.5-3B-Instruct")
BERT_RM_MODEL = os.environ.get(
    "RLJEVF_BERT_RM", "OpenAssistant/reward-model-deberta-v3-large-v2"
)
JEV_MODEL = os.environ.get("RLJEVF_JEV", "typesafe/jev-1.13")
# Eval judge must share no family with ANY training reward source (plan 8.1).
# Must share no family with any training reward source (Qwen, DeepSeek, Jev).
EVAL_JUDGE_MODEL = os.environ.get("RLJEVF_EVAL_JUDGE", "openai/gpt-5-mini")

# --- Jev / OpenRouter -------------------------------------------------------
JEV_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
OPENROUTER_CHAT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
JEV_MAX_CONCURRENCY = int(os.environ.get("RLJEVF_JEV_CONCURRENCY", "48"))
# Hard stop. The whole study is budgeted at $30; a bug must not spend it.
BUDGET_USD = float(os.environ.get("RLJEVF_BUDGET_USD", "30.0"))


@dataclass
class RunConfig:
    """One row of the results table."""

    run_id: str                      # "R3-jev-rf"
    reward_source: str               # bert_rm | llm_judge | api_judge | jev | self_certainty | none
    algo: str                        # grpo | ppo | none
    critic: str = "none"             # none | learned | jev   (PPO only)
    rubric_source: str = "static"    # static | generated | policy  (SG2JEV ablation)
    seed: int = 0

    # --- controlled variables: identical across every run (plan section 4) ---
    policy_model: str = POLICY_MODEL
    sft_checkpoint: str | None = None
    max_prompt_length: int = 512
    max_completion_length: int = 384
    temperature: float = 1.0
    top_p: float = 1.0
    num_generations: int = 8         # G, GRPO
    prompts_per_step: int = 16
    learning_rate: float = 1e-6
    beta: float = 0.04               # KL coefficient, set explicitly (TRL default drifted)
    max_steps: int = 250

    # --- PPO-only ---
    ppo_lr_value: float = 1e-5
    ppo_kl_coef: float = 0.05
    ppo_gamma: float = 1.0
    ppo_lam: float = 0.95
    ppo_cliprange: float = 0.2
    ppo_cliprange_value: float = 0.2
    ppo_epochs: int = 2
    ppo_whiten_rewards: bool = True
    jev_critic_num_prefixes: int = 6  # V(s_t) probes per completion

    # --- accounting ---
    reward_call_budget: int | None = None  # equalise cost across sources, not steps

    @property
    def out_dir(self) -> Path:
        return RUN_ROOT / f"{self.run_id}_seed{self.seed}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["out_dir"] = str(self.out_dir)
        return d


def seeds(n: int = 3) -> list[int]:
    return [0, 1, 2][:n]


def supports_bf16() -> bool:
    """torch reports bf16 "supported" on Turing (T4) via emulation, which is
    much slower than fp16. Gate on real hardware support instead."""
    import torch

    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8
