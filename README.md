# RL fine-tuning with Jev as a reward source **and** as a critic

Research code for a comparison of reward sources for RL fine-tuning of a small
policy LLM, with [Jev](https://docs.typesafe.ai/api) (TypeSafe's *System One*
model, which returns typed calibrated values instead of text) used in two
distinct roles.

## The two roles

| Role | Question asked of Jev | What it replaces |
|---|---|---|
| **Reward function** | 5-question rubric over a **finished** response, one request | a Bradley–Terry RM / an LLM judge |
| **Critic** | one forward-looking question over a **partial** response | PPO's learned value network |

The critic role rests on the observation that in a language-model MDP with a
single terminal reward, `V*(s_t) = E[R | prompt, tokens_<t]` — the expected
quality of the *finished* response given the prefix. That is a yes/no question
about a partial state, which is exactly Jev's input format. Measured behaviour
on a partial response (full study in `results/calibration_study.json`):

```
V(prefix) = "will the finished response be correct and helpful?"
  good answer:  0%=0.67  15%=0.84  30%=0.86  50%=0.87  75%=0.91  100%=0.92   ↗
  bad answer:   0%=0.66  15%=0.51  30%=0.47  50%=0.30  75%=0.31  100%=0.18   ↘
```

### Why there is no "Jev as evaluator" arm
Scoring a finished response and using the scalar as the learning signal **is**
using Jev as a reward function. A separate "evaluator" arm would be the same
column twice, so it was dropped and the budget spent on the critic role, which
queries states the reward function never sees.

## Experiment matrix

```
python scripts/run_study.py --list
```

| Arm | Optimiser | Manipulated variable |
|---|---|---|
| `R0-sft` | — | shared start checkpoint |
| `R1-bert` | GRPO | DeBERTa Bradley–Terry RM |
| `R2-self` | GRPO | frozen SFT copy judges itself |
| `R3-rlaif` | GRPO | LLM judge, read from logits |
| `R4-jev` | GRPO | **Jev as reward function** |
| `R5-ppo-learned` | PPO | learned value head (control) |
| `R6-ppo-jevcritic` | PPO | **Jev as critic, no value network** |
| `A1-sg2jev-gen` | GRPO | rubric generated per prompt by an LLM |
| `A2-sg2jev-policy` | GRPO | rubric proposed by the policy (hacking probe) |
| `A3-selfcert` | GRPO | self-certainty, no external judge |

Held fixed across all arms: start checkpoint, prompt set **and order** (verified
by a logged fingerprint), decoding parameters, KL coefficient, `G`, and the
**budget of reward calls** — not the step count. Three seeds per core arm.

## Fairness protocol

Two things that would otherwise hand Jev an unearned win:

1. **No text parsing anywhere.** The LLM judge and self-judge are read from
   their logits — `P(yes)/(P(yes)+P(no))` for a yes/no item, a softmax over
   level-index tokens for a scale — so every judge returns the same kind of
   object Jev does. See `rljevf/rewards/llm_judge.py`.
2. **Judge competence is measured, not assumed.** `scripts/judge_benchmark.py`
   scores every source on held-out human preference pairs before any RL runs, so
   the RLAIF arm uses a judge that actually works rather than a straw man.

## Layout

```
rljevf/
  config.py          every experimental knob, one dataclass per run
  jevclient.py       async Jev client: hard budget cap, disk cache, cost/latency meter
  orclient.py        OpenRouter chat client (rubric generation, eval judge)
  rubric.py          the shared rubric, rendered into each judge's native format
  sg2jev.py          StructuredGeneration2JEV: static | generated | policy
  data.py            UltraFeedback / GSM8K, with dataset fingerprints
  registry.py        RunConfig -> reward source / critic
  ppo.py             PPO with a swappable critic (TRL 1.13 dropped PPOTrainer)
  rewards/           bert_rm, llm_judge, self_judge, jev_rf, self_certainty
  critic/            jev_critic (prefix values + calibration), learned value head
  evaluate/          win-rate (length-controlled), GSM8K, hacking metrics, cross-matrix
scripts/
  train_sft.py       R0
  train_grpo.py      phase 1
  train_ppo.py       phase 2
  judge_benchmark.py experiment 0: how good is each reward source?
  calibration_study.py experiment 1: is the prefix probability a value function?
  evaluate_run.py    one checkpoint -> eval.json
  collect_results.py run artefacts -> results/*.json
  make_tables.py     results/*.json -> paper/tables/*.tex
  run_study.py       the experiment matrix + cost estimate
  kaggle_run.py      ship a run to Kaggle, poll, fetch
  kaggle_selftest.py pre-flight for a Kaggle session
paper/               main.tex, refs.bib, generated tables
```

## Setup

```bash
uv sync
echo "OPEN_ROUTER_API_KEY=..." >> .env      # local
```

**For Kaggle runs the key must be added once in the UI**: open any notebook →
*Add-ons → Secrets* → add `OPEN_ROUTER_API_KEY`. Kaggle does not expose secret
attachment over the API, so this is the one manual step.

## Running

```bash
# pre-flight on Kaggle (CPU, does not consume GPU quota)
python scripts/kaggle_run.py --kernel rljevf-smoke-cpu --no-gpu \
    --command "scripts/kaggle_selftest.py" --wait

# cost estimate for the whole study
python scripts/run_study.py --list

# one arm, or everything
python scripts/run_study.py --launch R4-jev --wait
python scripts/run_study.py --launch-all

# analysis
python scripts/collect_results.py
python scripts/make_tables.py
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Cost control

`rljevf/jevclient.py` enforces a hard spend cap (`RLJEVF_BUDGET_USD`, default
$30) that is **persisted to disk**, so it survives the Kaggle session
pre-emptions that this project runs into routinely. Every response's reported
cost is accumulated; exceeding the cap raises `BudgetExceeded` rather than
continuing. All Jev and chat calls are cached in SQLite keyed by
`(model, state, questions)`, so re-analysis and resumed runs are free.

## Environment notes

Pinned because they cost debugging time:

- **TRL 1.13 removed `PPOTrainer`** — hence `rljevf/ppo.py`. This turned out to
  be convenient: one PPO implementation with a swappable critic makes the
  learned-vs-Jev comparison exact.
- **TRL 1.13 removed `GRPOConfig.max_prompt_length`** — prompt-length control
  moved into the dataset (`data.filter_by_prompt_tokens`).
- **transformers 5.17 segfaults on macOS** when a second model is loaded into a
  busy process (threaded weight materialisation), which is exactly what
  `GRPOTrainer` does when it builds the reference model. `rljevf/__init__.py`
  sets `HF_DEACTIVATE_ASYNC_LOAD=1` on Darwin only.
- **`gradient_checkpointing=True` is CUDA-only here** — it also segfaults on MPS.
- TRL detects async reward functions with `inspect.iscoroutinefunction`, which
  is `False` for an instance with an `async __call__`. `registry.py` wraps the
  Jev source in a real coroutine function so it runs on TRL's async loop.

## Session robustness

A Kaggle session is killed at 12 hours and can be pre-empted before that, so
the study is built to survive interruption rather than to hope against it.

**Session budget.** The plan's `session_seconds` (10.5h) stops the runner
before the 12h wall, and `reserve_seconds` (20 min) is held back so artifacts
are always packaged. This matters because Kaggle commits `/kaggle/working`
only when the kernel exits cleanly -- a killed session loses its output.

**Per-job guard** (`scripts/guard.py`, wrapped around every job). Enforces a
hard timeout, aborts a job that has printed nothing for too long, aborts
before the working disk fills, and logs elapsed time / RSS / VRAM / free disk
on a heartbeat. On any of these it sends SIGTERM and gives the child a grace
period to checkpoint before SIGKILL. Exit codes are distinguishable:
124 timeout, 125 silence, 123 disk, 126 signal.

**Checkpoint and resume.** Both trainers save *complete* state -- weights,
optimiser moments, step counter, LR schedule, RNG states, the reward-call
meter and the run history -- and both install a SIGTERM handler so the guard's
grace period is used to checkpoint. The directory is called `resume_state`,
never `checkpoint-*`, because the runner deletes that glob. On start,
`rljevf/resume.py` searches the working directory and every mounted dataset
for a usable state; a half-written one is rejected rather than loaded.
The prompt schedule is a pure function of `(seed, step)`, so a resumed run
sees exactly the data an uninterrupted one would.

**Verified download.** `scripts/package_artifacts.py` writes MANIFEST.json
with a sha256 per file; `scripts/fetch_artifacts.py` re-downloads and checks
every file against it, retrying, and fails loudly rather than leaving a
silently truncated file to corrupt the analysis.

**Across sessions.** `scripts/orchestrate.py` waits for the session, fetches
and verifies, lifts `resume_state` into `state/` (shipped back in the code
dataset), and writes the next plan with finished jobs marked `done` so they
are skipped. A failed job is retried in the next batch by default.

**Live monitoring.** `scripts/watch_kaggle.py` streams the session log and
emits one line per event worth acting on (JOB / GUARD / ERROR / SPEND /
STATUS), so a stuck job surfaces while the GPU quota can still be saved.

### Credentials, and why the study runs unattended

Kaggle cannot attach a notebook secret over the API, and pushing a kernel
version drops an attachment made in the UI (`ApiSaveKernelRequest` has no
field for one) -- verified by pushing an unchanged kernel and watching
`secret: loaded` become `secret: FAILED`. That makes a secret-based setup
incompatible with unattended batches, because starting a run *is* a push.

So the key is mounted instead, as a **private** Kaggle dataset
(`scripts/kaggle_run.py push-secrets`). The runner prefers a notebook secret
when one is attached and falls back to the mounted file, so both modes work.

This is a real trade-off: the key now sits in Kaggle storage rather than in
Kaggle's secret store. Two things keep it contained, and a third is worth
doing:

  * the dataset is created with `public=False`, and the uploader **verifies**
    it afterwards -- it must appear private in the owned listing *and* be
    absent from public search, or the command aborts and tells you to delete
    it;
  * the key is never printed, and the staging copy is deleted after upload;
    `rljevf_secrets.json` is in `.gitignore`;
  * use a **dedicated OpenRouter key with a spend cap** for this project, so
    the blast radius of a mistake is bounded by that cap rather than by the
    account balance.

With the key mounted, `python scripts/kaggle_run.py run` starts a batch and
`scripts/orchestrate.py --auto` chains batches until the plan finishes or the
weekly GPU quota runs low. Nobody needs to click anything.
