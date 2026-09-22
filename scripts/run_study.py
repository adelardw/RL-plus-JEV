#!/usr/bin/env python
"""The experiment matrix, in one place.

Every arm of the paper is declared here with the exact command that produces it,
so the study is reproducible from a single file and the cost can be estimated
before anything is spent.

  python scripts/run_study.py --list           # the matrix and its cost estimate
  python scripts/run_study.py --launch R4-jev  # push one arm to Kaggle
  python scripts/run_study.py --launch-all     # push everything, respecting
                                               # Kaggle's concurrent-session cap
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

# Measured unit costs (see results/jev_properties.json), in USD per call.
COST_RUBRIC_CALL = 3.0e-5     # 5-question rubric over prompt + full completion
COST_PREFIX_CALL = 2.0e-5     # 1 question over prompt + partial completion
COST_JUDGE_PAIR = 5.1e-4      # one A/B verdict from the external eval judge
COST_RUBRIC_GEN = 7.2e-4      # one SG2JEV question-set generation
COST_API_JUDGE = 1.45e-4      # one completion scored on all 5 rubric questions

CORE_SEEDS = (0, 1, 2)
ABLATION_SEEDS = (0,)

SFT = "/kaggle/working/runs/R0-sft"   # produced by the R0 arm, then reused


@dataclass
class Arm:
    name: str
    command: str
    seeds: tuple[int, ...] = CORE_SEEDS
    gpu: bool = True
    usd_per_seed: float = 0.0
    note: str = ""
    depends_on: str | None = None

    def commands(self) -> list[tuple[str, str]]:
        out = []
        for s in self.seeds:
            slug = f"rljevf-{self.name.lower()}-s{s}"
            out.append((slug, self.command.format(seed=s, name=self.name, sft=SFT)))
        return out

    @property
    def total_usd(self) -> float:
        return self.usd_per_seed * len(self.seeds)


# --- shared scale ---------------------------------------------------------- #
# The reward-call budget is a controlled variable shared by every arm, so
# shrinking it rescales all arms equally and leaves every comparison intact.
# 180 keeps three seeds per arm inside the $30 budget with room for reruns,
# which two seeds at 250 steps would not have bought.
GRPO_STEPS = 180
GRPO_PROMPTS = 16
GRPO_G = 8
GRPO_CALLS = GRPO_STEPS * GRPO_PROMPTS * GRPO_G          # 32,000 reward calls

PPO_STEPS = 110
PPO_PROMPTS = 64
PPO_CALLS = PPO_STEPS * PPO_PROMPTS                      # 9,600 reward calls
PPO_PREFIXES = 6

# Memory-derived settings (scripts/memory_budget.py, validated against an
# observed OOM to within 0.5%): full fine-tuning of the 0.5B policy does not fit
# on a 14.6GB T4 at ANY micro-batch size, because gradients, optimiser state and
# the separate reference model together exceed the card before the logits are
# allocated. LoRA removes all three at once -- TRL sets ref_model = None under
# PEFT and takes the reference by disabling adapters -- and the same adapter
# configuration is used by every arm, so the comparison is unaffected.
LORA_R = 16
MICRO_BS = 8


def grpo_cmd(reward: str) -> str:
    """The GRPO command line. {seed}/{name}/{sft} stay as placeholders for Arm."""
    return (
        "scripts/train_grpo.py --run-id {name} --reward " + reward + " --sft {sft} "
        "--seed {seed} "
        f"--steps {GRPO_STEPS} --num-generations {GRPO_G} "
        f"--prompts-per-step {GRPO_PROMPTS} --micro-bs {MICRO_BS} "
        f"--max-completion-length 384 --lora-r {LORA_R}"
    )


def ppo_cmd(critic: str) -> str:
    return (
        "scripts/train_ppo.py --run-id {name} --reward jev --critic " + critic + " --sft {sft} "
        "--seed {seed} "
        f"--steps {PPO_STEPS} --batch-prompts {PPO_PROMPTS} "
        "--gen-bs 8 --micro-bs 4 --max-completion-length 384 "
        f"--num-prefixes {PPO_PREFIXES} --calibration-n 96 --lora-r {LORA_R}"
    )


ARMS: list[Arm] = [
    Arm("R0-sft", "scripts/train_sft.py --seed {seed} --out " + SFT,
        seeds=(0,), note="shared start point for every arm"),

    # --- Phase 1: GRPO, reward source is the only variable ---
    Arm("R1-bert", grpo_cmd("bert_rm"),
        usd_per_seed=0.0, note="DeBERTa Bradley-Terry RM, local", depends_on="R0-sft"),
    Arm("R2-self", grpo_cmd("self_judge"),
        usd_per_seed=0.0, note="frozen SFT copy judges itself, local", depends_on="R0-sft"),
    Arm("R3-rlaif", grpo_cmd("api_judge"),
        usd_per_seed=GRPO_STEPS * GRPO_PROMPTS * GRPO_G * COST_API_JUDGE,
        note="hosted judge read from logits (parse-free), DeepSeek V4 Flash",
        depends_on="R0-sft"),
    Arm("R4-jev", grpo_cmd("jev"),
        usd_per_seed=GRPO_CALLS * COST_RUBRIC_CALL, note="Jev as reward function",
        depends_on="R0-sft"),

    # --- Phase 2: PPO, critic is the only variable ---
    Arm("R5-ppo-learned", ppo_cmd("learned"),
        usd_per_seed=PPO_CALLS * COST_RUBRIC_CALL,
        note="textbook PPO: learned value head, Jev reward", depends_on="R0-sft"),
    Arm("R6-ppo-jevcritic", ppo_cmd("jev"),
        usd_per_seed=PPO_CALLS * COST_RUBRIC_CALL
        + PPO_STEPS * PPO_PROMPTS * PPO_PREFIXES * COST_PREFIX_CALL,
        note="no value network: Jev values prefixes", depends_on="R0-sft"),

    # --- Ablations ---
    Arm("A1-sg2jev-gen",
        grpo_cmd("jev") + " --rubric-source generated --n-prompts 1000",
        seeds=ABLATION_SEEDS,
        usd_per_seed=GRPO_CALLS * COST_RUBRIC_CALL + 1000 * COST_RUBRIC_GEN,
        note="SG2JEV: questions written per prompt by a frozen LLM", depends_on="R0-sft"),
    Arm("A2-sg2jev-policy",
        grpo_cmd("jev") + " --rubric-source policy",
        seeds=ABLATION_SEEDS, usd_per_seed=GRPO_CALLS * COST_RUBRIC_CALL,
        note="SG2JEV: the policy writes its own grading criteria (hacking probe)",
        depends_on="R0-sft"),
    Arm("A3-selfcert", grpo_cmd("self_certainty"),
        seeds=ABLATION_SEEDS, usd_per_seed=0.0,
        note="no external judge at all (degenerate control)", depends_on="R0-sft"),
]

N_EVAL_SYSTEMS = sum(len(a.seeds) for a in ARMS if a.name != "R0-sft")
EVAL_USD = N_EVAL_SYSTEMS * 200 * 2 * COST_JUDGE_PAIR
CROSS_USD = (N_EVAL_SYSTEMS + 1) * 200 * COST_RUBRIC_CALL


def show() -> None:
    print(f"{'arm':20s} {'seeds':>5s} {'USD/seed':>9s} {'USD tot':>8s}  note")
    print("-" * 96)
    total = 0.0
    for a in ARMS:
        total += a.total_usd
        print(f"{a.name:20s} {len(a.seeds):5d} {a.usd_per_seed:9.2f} {a.total_usd:8.2f}  {a.note}")
    print("-" * 96)
    print(f"{'training subtotal':20s} {'':5s} {'':9s} {total:8.2f}")
    print(f"{'win-rate judging':20s} {'':5s} {'':9s} {EVAL_USD:8.2f}  "
          f"{N_EVAL_SYSTEMS} systems x 200 prompts x 2 orders")
    print(f"{'cross-reward matrix':20s} {'':5s} {'':9s} {CROSS_USD:8.2f}")
    print(f"{'TOTAL':20s} {'':5s} {'':9s} {total + EVAL_USD + CROSS_USD:8.2f}")
    print(f"\nreward-call budget: GRPO {GRPO_CALLS:,} per run; "
          f"PPO {PPO_CALLS:,} reward + {PPO_STEPS*PPO_PROMPTS*PPO_PREFIXES:,} critic")
    print(f"training runs to schedule: {sum(len(a.seeds) for a in ARMS)}")


def launch(arm: Arm, wait: bool, dry: bool) -> None:
    for slug, cmd in arm.commands():
        argv = [sys.executable, str(PROJECT / "scripts" / "kaggle_run.py"),
                "--kernel", slug, "--command", cmd]
        if wait:
            argv += ["--wait", "--fetch-to", str(PROJECT / "runs" / slug)]
        print("+", " ".join(argv))
        if not dry:
            subprocess.run(argv, check=False)
            time.sleep(5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--launch", default=None)
    ap.add_argument("--launch-all", action="store_true")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.launch_all:
        for a in ARMS:
            launch(a, args.wait, args.dry_run)
    elif args.launch:
        matches = [a for a in ARMS if a.name == args.launch]
        if not matches:
            raise SystemExit(f"unknown arm {args.launch!r}; known: {[a.name for a in ARMS]}")
        launch(matches[0], args.wait, args.dry_run)
    else:
        show()


if __name__ == "__main__":
    main()
