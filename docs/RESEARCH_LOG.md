# Research log

Decisions, measurements and open problems, in the order they were settled.
Everything here is measured unless marked otherwise. Numbers quoted from
documentation rather than from our own runs are labelled as such.

---

## Target and constraints

- **Venue:** NeurIPS/ICLR. Workshop is realistic; the main track is a stretch
  and the reason is stated under *Open problems*.
- **Compute:** Kaggle only, no rented GPU (user's decision, 2026-09-22).
  2x Tesla T4, 14.6 GB each, **30 GPU-h/week** (not 6 -- see *Traps*).
- **API budget:** $30 on OpenRouter. Spent so far: ~$0.10.
- **Language:** paper in English, working discussion in Russian.

## Design decisions

| Decision | Why |
|---|---|
| Dropped "Jev as evaluator" as a separate arm | It *is* "Jev as reward function": score a finished response, take the scalar. It would have been the same column twice. The freed slot went to the critic role, which queries states the reward function never sees. |
| RLAIF judge is hosted, not local | Every local judge that fits on 2x T4 measures at or near chance (Exp. 0). Training against one would be beating a straw man. |
| Hosted judge is read from logprobs, not parsed | Keeps the fairness protocol: every judge returns the same kind of object Jev does. Requires `reasoning: {enabled: false}` and `provider: {require_parameters: true}`. |
| Prompt order is state-then-question | The five rubric questions about one completion share the long prefix and differ in a short suffix. This is the reverse of "instructions first" advice, which optimises the wrong axis here. |
| Reward-call budget cut 250 -> 180 steps | It is a controlled variable shared by every arm, so cutting it rescales all arms equally. Bought 3 seeds per arm inside budget instead of 2. |
| SG2JEV is an ablation, not the pipeline | Static rubric is the only setting in which Jev, the RLAIF judge and the self-judge are comparable. Generated and policy-proposed rubrics are measured as an axis. |

## Measured results

### Experiment 0 -- can each reward source recognise a human preference?
300 held-out UltraFeedback pairs, on a T4.

| Source | Accuracy | Cohen's d | Wall |
|---|---|---|---|
| Jev | 0.650 | +0.394 | 7 s |
| BERT RM (DeBERTa BT) | 0.597 | +0.191 | 85 s |
| Qwen2.5-3B (logits) | 0.577 | +0.252 | 975 s |
| Qwen2.5-1.5B (logits) | 0.507 | -0.067 | 486 s |
| Qwen2.5-7B | OOM on one T4 | | |

**This n is too small to support the ranking.** Resolving 0.650 vs 0.597 needs
n >= 1308 at zero judge correlation, n >= 655 at rho = 0.5 (`rljevf/stats.py`).
The 5.3-point gap sits entirely inside its own confidence interval at n = 300.
To be re-run at n = 2000 with McNemar and paired bootstrap.

### Experiment 1 -- is the prefix probability a value function?
n = 256 completions from the base policy, scored by Jev.

| Prefix seen | V(good) | V(bad) | AUC | rho |
|---|---|---|---|---|
| 0% | 0.539 | 0.506 | 0.608 | +0.222 |
| 25% | 0.574 | 0.347 | 0.754 | +0.487 |
| 50% | 0.498 | 0.239 | 0.784 | +0.556 |
| 100% | 0.452 | 0.143 | 0.802 | +0.635 |

Calibration `V = 2.272 p - 0.970`, r = 0.721, R^2 = 0.520. Replicates an
earlier n = 128 run on MPS (R^2 = 0.559).

Two honest caveats. AUC at 0% is 0.608, not 0.5: part of the signal comes from
the prompt alone, since a hard request predicts a worse answer. And both curves
fall, because the base policy writes badly (`padded` = 0.68, mean reward
-0.30), so Jev grows pessimistic about everything -- faster about the bad half.
Calibration is therefore re-fit per run on fresh samples rather than fixed once.

### The mechanism transfers, and calibration is what differs
Same prefix question, same prompt, three readers:

```
                   0%    15%    30%    50%    75%   100%
jev      good :  0.59   0.84   0.89   0.88   0.92   0.93   rising
         bad  :  0.59   0.55   0.34   0.13   0.18   0.06   falling
api      good :  0.22   1.00   1.00   1.00   1.00   1.00   rising
         bad  :  0.22   0.08   0.44   0.15   0.05   0.00   falling
```

Both separate eventually-good from eventually-bad prefixes. But the hosted
judge **saturates** at 0.00/1.00 while Jev spreads over [0,1]. Saturation is
harmless for a terminal reward and damaging for a critic: TD errors are
differences between neighbouring values, so a saturated curve carries no signal
in the middle of a response, which is where credit assignment is needed.

This is the paper's sharpest claim, because it is the one argument that cannot
be answered with "use a bigger LLM judge".

### Sparse-probe bias (Propositions 1-2)
n = 15 completions, probed every 4 tokens, sparse estimates rebuilt from the
same probes so judge noise is excluded.

| K | Mean error | Max error |
|---|---|---|
| 2 | 0.1220 | 0.2731 |
| 4 | 0.0481 | 0.2251 |
| 8 | 0.0264 | 0.1782 |
| 16 | 0.0128 | 0.1248 |

Mean error decays as **K^-1.03** -- the Lipschitz rate of Proposition 2 almost
exactly, and not the K^-2 of bounded curvature. Worst case decays as
**K^-0.35**. The gap says the value curve is Lipschitz but not smooth: a
response can commit to being bad at one token, and expected quality steps down
there. Raising K buys mean accuracy and barely touches the worst case, which
points at adaptive probe placement rather than larger K.

**n = 15 is too small to publish.** Re-run at n >= 64, stride 2 (~$0.10).

### Measured properties of Jev 1.13
- `POST https://openrouter.ai/api/alpha/decisions`, 32k context.
- $0.0299 per 1,000 five-question rubric calls; ~713 input tokens per call.
- Latency p50 0.57 s, p95 1.28 s; 23-37 calls/s at concurrency 32-48.
- Question types: `noul` (probability), `choice` (distribution + confidence),
  `score` (expected level + distribution + confidence). `score` returns the
  expected level in [0, L-1], so it is divided by L-1.

## Traps, each of which cost time

| Trap | Reality |
|---|---|
| Kaggle GPU quota "6h" | The SDK's JSON serialiser drops the `days` component of a Duration: `timedelta(days=1, seconds=21600)` printed as `21600s`. The real quota is **30h/week**. |
| Kaggle secrets over the API | Cannot be attached, and a kernel push drops an attachment made in the UI. Verified by pushing an unchanged kernel and watching `secret: loaded` become `secret: FAILED`. The key is mounted as a private dataset instead, which is what makes unattended runs possible. |
| `torch.cuda.is_bf16_supported()` on a T4 | Returns True via emulation, which is slow. Gate on compute capability >= 8 instead. |
| TRL 1.13 | `PPOTrainer` and `GRPOConfig.max_prompt_length` are both gone. Hence `rljevf/ppo.py` and `data.filter_by_prompt_tokens`. |
| TRL async reward functions | Detected with `inspect.iscoroutinefunction`, which is False for an instance with an async `__call__`. Wrap in a real coroutine function. |
| transformers 5.17 on macOS | Segfaults when a second model is loaded into a busy process -- exactly what GRPOTrainer does for the reference model. `HF_DEACTIVATE_ASYNC_LOAD=1` on Darwin only. |
| `gradient_checkpointing` on MPS | Also segfaults. CUDA-only here. |
| OpenRouter logprobs | Listed in `supported_parameters` but only some providers return them; without `require_parameters` half the calls silently fall back to parsing. |
| `kernels_logs_stream` | An unbounded live tail that restarts from the beginning after a dropped connection. Consume incrementally, de-duplicate by timestamp. |
| Prices from a model-recommendation agent | Quoted `:batch` tier prices as standard. gpt-oss-120b is $0.15/$0.60, not $0.04/$0.18. Always confirm against `/api/v1/models`. |

## Open problems

1. **Policy scale is 0.5B.** The first question any reviewer asks. Not fixable
   on Kaggle: 2x T4 at 30 h/week is roughly 19 A100-hours per week, against the
   60-80 needed for 1.5B across six arms and three seeds. Removing this costs
   about $150 of rented A100 time; the user has declined for now.
2. **PPO and GRPO both OOM at the study's own settings** on a T4. The benchmark
   now searches downward for the largest micro-batch that fits and also varies
   the optimiser, since Adam's state is 4 GB of 14.6 for a 0.5B model in fp32.
   Unresolved until that benchmark completes.
3. **No training run has happened yet.** All 22 are ahead.
4. **Closed judge.** Jev's weights are unpublished and its behaviour may change
   between versions. Mitigated by pinning `jev-1.13`, recording the dated model
   string returned with every response, and caching every call so the analysis
   is reproducible from the artifact even if the API moves.
5. **No human evaluation.** Everything is model-judged.
