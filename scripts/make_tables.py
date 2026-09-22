#!/usr/bin/env python
"""Render every LaTeX table in the paper from the JSON artefacts.

The paper never hard-codes a number. Re-running this after a new arm lands
updates the manuscript, and a missing artefact renders as a visible placeholder
rather than a stale value.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
TABLES = PROJECT / "paper" / "tables"
RESULTS = PROJECT / "results"


def load(name: str):
    p = RESULTS / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def write(name: str, body: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / name).write_text(body.rstrip() + "\n")
    print(f"wrote paper/tables/{name}")


def placeholder(caption: str, label: str) -> str:
    return (
        "\\begin{table}[t]\\centering\n"
        f"\\caption{{{caption}}}\\label{{{label}}}\n"
        "\\textcolor{red}{[pending: run not yet complete]}\n"
        "\\end{table}"
    )


def fmt(x, d=3):
    if x is None:
        return "--"
    if isinstance(x, float):
        if x != x:  # NaN
            return "--"
        return f"{x:.{d}f}"
    return str(x)


# --------------------------------------------------------------------------- #
def t_jev_properties() -> None:
    d = load("jev_properties.json")
    if d is None:
        write("jev_properties.tex", placeholder("Measured properties of \\jev{} 1.13.", "tab:jevprops"))
        return
    rows = "\n".join(
        f"{k} & {v} \\\\" for k, v in d["rows"]
    )
    write("jev_properties.tex", f"""\\begin{{table}}[t]\\centering\\small
\\caption{{Measured properties of \\jev{{}} 1.13 accessed through OpenRouter's
decisions endpoint. Latency and cost are measured from our own calls, not quoted
from documentation; they include network round-trip from the client region and
so upper-bound the provider-side figures.}}
\\label{{tab:jevprops}}
\\begin{{tabular}}{{ll}}
\\toprule
Property & Value \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}""")



def t_judge_benchmark() -> None:
    d = load("judge_benchmark.json")
    if d is None:
        write("judge_benchmark.tex",
              placeholder("Reward-source agreement with human preferences.", "tab:judges"))
        return
    order = sorted(d["sources"].items(),
                   key=lambda kv: -(kv[1].get("accuracy") or 0))
    rows = []
    for name, v in order:
        if "accuracy" not in v:
            continue
        label = (name.replace("llm_judge:", "LLM judge ")
                     .replace("bert_rm", "BERT RM (DeBERTa BT)")
                     .replace("jev", "\\jev{}")
                     .replace("_", "\\_"))
        rows.append(f"{label} & {100*v['accuracy']:.1f} & {v['cohens_d']:+.2f} & "
                    f"{v['mean_margin']:+.3f} & {v['wall_s']:.0f} \\\\")
    body = "\n".join(rows)
    write("judge_benchmark.tex", f"""\\begin{{table}}[t]\\centering\\small
\\caption{{How well each reward source recognises a human preference, measured
before any RL. Each source scores the chosen and the rejected response of
$n={d['n']}$ held-out UltraFeedback pairs; accuracy is how often it ranks the
chosen one higher, and Cohen's $d$ is the separation in units of the margin's
own standard deviation. A source that cannot do this will not teach a policy
anything, so this is what fixes the RLAIF judge size rather than an assumption.
Wall-clock is for all $2n$ scorings on the same hardware.}}
\\label{{tab:judges}}
\\begin{{tabular}}{{lcccc}}
\\toprule
Reward source & Accuracy (\\%) & Cohen's $d$ & Mean margin & Wall-clock (s) \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}""")


def t_calibration() -> None:
    d = load("calibration_study.json")
    if d is None:
        write("calibration.tex", placeholder("Prefix value vs.\\ final quality.", "tab:calib"))
        write("calib_inline.tex", "\\todo{calibration pending}")
        return
    cal = d["calibration_at_full"]
    frac_rows = []
    for f, v in d["by_fraction"].items():
        frac_rows.append(
            f"{float(f):.0%} & {fmt(d['mean_value_curve_good'][f])} & "
            f"{fmt(d['mean_value_curve_bad'][f])} & {fmt(v['auc_vs_final_quality'])} & "
            f"{fmt(v['spearman_vs_final_reward'])} & {fmt(v['calibration']['r2'])} \\\\"
        )
    frac_rows = "\n".join(frac_rows).replace("%", "\\%")
    write("calibration.tex", f"""\\begin{{table}}[t]\\centering\\small
\\caption{{\\jev{{}}'s forward-looking probability on a \\emph{{partial}} response,
as a function of how much of the response it has seen, measured on
$n={d['n']}$ completions sampled from the SFT policy. ``Good'' and ``bad'' are
the above- and below-median halves by \\emph{{final}} rubric reward, which is not
observable at the time of the query. AUC is the probability that the prefix
value ranks an eventually-good response above an eventually-bad one; $R^2$ is
for the affine fit of final reward on prefix value.}}
\\label{{tab:calib}}
\\begin{{tabular}}{{lccccc}}
\\toprule
Prefix seen & $\\bar{{p}}$ (good) & $\\bar{{p}}$ (bad) & AUC & Spearman $\\rho$ & $R^2$ \\\\
\\midrule
{frac_rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}""")
    write("calib_inline.tex",
          f"AUC $={fmt(d['by_fraction']['1.00']['auc_vs_final_quality'],2)}$ at the full response "
          f"and $={fmt(d['by_fraction']['0.25']['auc_vs_final_quality'],2)}$ after only a quarter of it; "
          f"$R^2={fmt(cal['r2'],2)}$")


def t_setup() -> None:
    d = load("setup.json")
    if d is None:
        write("setup.tex", placeholder("Experimental configuration.", "tab:setup"))
        return
    rows = "\n".join(f"{k} & {v} \\\\" for k, v in d["rows"])
    write("setup.tex", f"""\\begin{{table}}[t]\\centering\\small
\\caption{{Configuration. Every entry is held fixed across all arms except the
reward source (GRPO phase) and the critic (PPO phase).}}
\\label{{tab:setup}}
\\begin{{tabular}}{{ll}}
\\toprule
Component & Setting \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}""")


def _agg(runs: list[dict], key: str, sub: str | None = None):
    vals = []
    for r in runs:
        v = r.get(key)
        if sub and isinstance(v, dict):
            v = v.get(sub)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return m, 0.0
    sd = (sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5
    return m, sd


def _ms(m, sd, d=1, pct=False):
    if m is None:
        return "--"
    if pct:
        return f"{100*m:.{d}f} $\\pm$ {100*(sd or 0):.{d}f}"
    return f"{m:.{d}f} $\\pm$ {(sd or 0):.{d}f}"


def _arm_table(d, caption, label, first_col):
    rows = []
    for arm in d["arms"]:
        runs = arm["seeds"]
        wr_m, wr_s = _agg(runs, "winrate", "lc_win_rate")
        raw_m, raw_s = _agg(runs, "winrate", "raw_win_rate")
        gs_m, gs_s = _agg(runs, "gsm8k_acc")
        ln_m, ln_s = _agg(runs, "length", "tokens_mean")
        sp_m, sp_s = _agg(runs, "self_praise_rate")
        kl_m, _ = _agg(runs, "kl_per_token_vs_sft")
        rows.append(
            f"{arm['label']} & {_ms(wr_m, wr_s, 1, pct=True)} & {_ms(raw_m, raw_s, 1, pct=True)} & "
            f"{_ms(gs_m, gs_s, 1, pct=True)} & {_ms(ln_m, ln_s, 0)} & "
            f"{_ms(sp_m, sp_s, 1, pct=True)} & {fmt(kl_m, 3)} \\\\"
        )
    body = "\n".join(rows)
    return f"""\\begin{{table}}[t]\\centering\\small
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{lcccccc}}
\\toprule
{first_col} & LC win-rate (\\%) & Raw (\\%) & GSM8K (\\%) & Len (tok) & Self-praise (\\%) & KL/tok \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


def t_grpo() -> None:
    d = load("grpo_main.json")
    if d is None:
        write("grpo_main.tex", placeholder("GRPO: reward sources.", "tab:grpo"))
        return
    write("grpo_main.tex", _arm_table(
        d,
        "GRPO with the reward source as the only manipulated variable, at a "
        "matched budget of reward calls. Win-rate is against the SFT checkpoint, "
        "length-controlled, judged by a model unrelated to every training judge. "
        "Mean $\\pm$ s.d.\\ over three seeds.",
        "tab:grpo", "Reward source"))


def t_ppo() -> None:
    d = load("ppo_main.json")
    if d is None:
        write("ppo_main.tex", placeholder("PPO: critics.", "tab:ppo"))
        return
    write("ppo_main.tex", _arm_table(
        d,
        "PPO with the critic as the only manipulated variable: a learned value "
        "head versus \\jev{} valuing token prefixes zero-shot. Both arms use the "
        "same reward source and the same PPO implementation.",
        "tab:ppo", "Critic"))


def t_cross() -> None:
    d = load("cross_matrix.json")
    if d is None:
        write("cross_matrix.tex", placeholder("Cross-reward matrix.", "tab:cross"))
        return
    srcs = d["sources"]
    head = " & ".join(srcs)
    rows = []
    for run, row in d["matrix"].items():
        cells = " & ".join(fmt(row.get(s), 3) for s in srcs)
        gap = d.get("overoptimisation", {}).get(run)
        rows.append(f"{run} & {cells} & {fmt(gap, 3)} \\\\")
    body = "\n".join(rows)
    write("cross_matrix.tex", f"""\\begin{{table}}[t]\\centering\\small
\\caption{{Cross-reward matrix, in units of change from the SFT baseline. Each
row is a trained checkpoint; each column is a reward source scoring it. The
diagonal is the source that checkpoint was trained on. The final column is
$\\mathrm{{diag}} - \\overline{{\\mathrm{{off\\text{{-}}diag}}}}$: large values mean the
policy moved towards its own judge rather than towards quality the other judges
also recognise.}}
\\label{{tab:cross}}
\\begin{{tabular}}{{l{'c' * (len(srcs) + 1)}}}
\\toprule
Checkpoint & {head} & Over-opt.\\ gap \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}""")


def t_sg2jev() -> None:
    d = load("sg2jev.json")
    if d is None:
        write("sg2jev.tex", placeholder("Rubric source ablation.", "tab:sg2jev"))
        return
    write("sg2jev.tex", _arm_table(
        d,
        "Where the \\jev{} question set comes from (\\textsc{StructuredGeneration2Jev}). "
        "All three arms are otherwise identical GRPO runs with \\jev{} as the reward "
        "source. The \\emph{policy} arm lets the model being trained write its own "
        "grading criteria and is included as a reward-hacking probe.",
        "tab:sg2jev", "Rubric source"))


def t_cost() -> None:
    d = load("cost.json")
    if d is None:
        write("cost.tex", placeholder("Cost accounting.", "tab:cost"))
        return
    rows = "\n".join(
        f"{r['arm']} & {r.get('calls','--')} & {fmt(r.get('usd'),3)} & "
        f"{fmt(r.get('p50_s'),2)} & {fmt(r.get('p95_s'),2)} & {fmt(r.get('gpu_hours'),2)} \\\\"
        for r in d["rows"]
    )
    write("cost.tex", f"""\\begin{{table}}[t]\\centering\\small
\\caption{{Measured cost of one training run per reward source. Local judges cost
GPU-hours rather than dollars; API judges cost both. Latency is per reward call
as observed by the trainer, including network round-trip.}}
\\label{{tab:cost}}
\\begin{{tabular}}{{lccccc}}
\\toprule
Arm & Reward calls & USD & p50 (s) & p95 (s) & GPU-h \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}""")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    for f in (t_jev_properties, t_judge_benchmark, t_calibration, t_setup,
              t_grpo, t_ppo, t_cross, t_sg2jev, t_cost):
        f()


if __name__ == "__main__":
    main()
