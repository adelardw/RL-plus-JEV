#!/usr/bin/env python
"""Render the paper's figures from the result JSON.

Rendering, not experimenting: this reads numbers computed on Kaggle and writes
PDFs, so it runs anywhere.

Palette and mark choices follow the project's visualisation rules. The three
series use the first three categorical slots, validated all-pairs in both
modes (worst CVD dE 9.2, worst normal-vision dE 24.0); aqua falls below 3:1
against the surface, so every series is also direct-labelled and given its own
marker -- identity never rests on colour alone.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

PROJECT = Path(__file__).resolve().parent.parent

SERIES = {                      # slot order is fixed, never cycled
    "jev": ("#2a78d6", "o", "Jev (typed)"),
    "api": ("#eb6834", "s", "Hosted judge (logprobs)"),
    "local": ("#1baf7a", "^", "Local judge (logits)"),
}
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e6e5e1"


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.size": 8.5, "font.family": "serif",
        "axes.edgecolor": GRID, "axes.labelcolor": INK,
        "axes.titlesize": 9, "axes.titleweight": "regular",
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "legend.frameon": False, "lines.linewidth": 1.8,
        "lines.markersize": 4.5, "savefig.bbox": "tight",
    })


def _clean(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)


def fig_prefix_value(cal: dict, out: Path) -> None:
    """Left: how far apart the critic holds eventually-good and eventually-bad
    prefixes. Right: how much of [0,1] it actually uses at the full response."""
    critics = cal.get("critics") or {"jev": cal}
    # Only draw the saturation panel if there is something to put in it; an
    # empty axis with default 0-1 ticks reads as a measurement of zero.
    sat = {k: (critics.get(k) or {}).get("spread_at_full")
           for k in SERIES if (critics.get(k) or {}).get("spread_at_full")}
    if sat:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    else:
        fig, ax1 = plt.subplots(figsize=(3.9, 2.7))
        ax2 = None

    for key, (colour, marker, label) in SERIES.items():
        c = critics.get(key)
        if not c or "mean_value_curve_good" in c is None:
            continue
        good, bad = c.get("mean_value_curve_good"), c.get("mean_value_curve_bad")
        if not good or not bad:
            continue
        fr = sorted(good, key=float)
        xs = [float(f) for f in fr]
        sep = [good[f] - bad[f] for f in fr]
        ax1.plot(xs, sep, color=colour, marker=marker, label=label, zorder=3)
        ax1.annotate(label.split(" (")[0], (xs[-1], sep[-1]),
                     textcoords="offset points", xytext=(6, 0),
                     color=colour, fontsize=7.5, va="center")

    ax1.set_xlabel("fraction of the response seen")
    ax1.set_ylabel(r"$V(\mathrm{good}) - V(\mathrm{bad})$")
    ax1.set_title("Separation grows as the response unfolds", loc="left")
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax1.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])   # no tick past the data
    ax1.axhline(0, color=GRID, linewidth=1)
    ax1.set_xlim(-0.03, 1.30)
    # A single series is named by its direct label; two or more always carry a
    # legend as well, so identity never rests on colour alone.
    if len(ax1.lines) > 1:
        ax1.legend(loc="lower right", fontsize=7.5, labelcolor=INK_MUTED)
    _clean(ax1)

    # saturation: a critic pinned at 0 or 1 carries no TD signal mid-response
    labels, values, colours = [], [], []
    for key, (colour, _m, label) in SERIES.items():
        spread = sat.get(key) or {}
        if "saturated_frac" not in spread:
            continue
        labels.append(label.split(" (")[0])
        values.append(100 * spread["saturated_frac"])
        colours.append(colour)
    if ax2 is not None and values:
        bars = ax2.bar(labels, values, color=colours, width=0.55, zorder=3)
        for b, v in zip(bars, values):
            ax2.annotate(f"{v:.0f}%", (b.get_x() + b.get_width() / 2, v),
                         textcoords="offset points", xytext=(0, 3),
                         ha="center", color=INK, fontsize=7.5)
        ax2.set_ylabel("responses valued at 0 or 1 (%)")
        ax2.set_title("Saturation: no gradient left to give", loc="left")
        ax2.set_ylim(0, max(values) * 1.25 + 1)
    if ax2 is not None:
        _clean(ax2)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_probe_density(dens: dict, out: Path) -> None:
    """Interpolation error against the number of probes, on log-log, with the
    two rates the analysis predicts."""
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ks = sorted((int(k) for k in dens["by_k"]))
    mean = [dens["by_k"][str(k)]["mean_abs_error"] for k in ks]
    mx = [dens["by_k"][str(k)]["max_abs_error"] for k in ks]

    c1, c2 = SERIES["jev"][0], SERIES["api"][0]
    ax.plot(ks, mean, color=c1, marker="o", label="mean", zorder=3)
    ax.plot(ks, mx, color=c2, marker="s", label="worst case", zorder=3)
    ax.annotate("mean", (ks[-1], mean[-1]), textcoords="offset points",
                xytext=(5, -2), color=c1, fontsize=7.5)
    ax.annotate("worst case", (ks[-1], mx[-1]), textcoords="offset points",
                xytext=(5, -2), color=c2, fontsize=7.5)

    # reference slopes, anchored at the first point so only the slope is compared
    for slope, dash, name in ((-1, (4, 2), r"$K^{-1}$ (Lipschitz)"),
                              (-2, (1, 2), r"$K^{-2}$ (bounded curvature)")):
        ref = [mean[0] * (k / ks[0]) ** slope for k in ks]
        ax.plot(ks, ref, color=INK_MUTED, linewidth=0.9, dashes=dash, zorder=1)
        ax.annotate(name, (ks[-1], ref[-1]), textcoords="offset points",
                    xytext=(5, -2), color=INK_MUTED, fontsize=7)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("probes per completion, $K$")
    ax.set_ylabel(r"$|\hat V_K - V^\ast|$")
    ax.set_title(f"Mean bias follows the Lipschitz rate "
                 f"($K^{{{dens['fitted_exponent_mean']:.2f}}}$)", loc="left")
    ax.set_xlim(ks[0] * 0.9, ks[-1] * 2.6)
    ax.legend(loc="lower left", fontsize=7.5, labelcolor=INK_MUTED)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(PROJECT / "results"))
    ap.add_argument("--out", default=str(PROJECT / "paper" / "figures"))
    args = ap.parse_args()
    _style()
    res, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    made = 0
    cal = res / "calibration_study.json"
    if cal.exists():
        fig_prefix_value(json.loads(cal.read_text()), out / "prefix_value.pdf")
        made += 1
    else:
        print(f"skipping prefix-value figure: {cal.name} not present")

    dens = res / "probe_density.json"
    if dens.exists():
        fig_probe_density(json.loads(dens.read_text()), out / "probe_density.pdf")
        made += 1
    else:
        print(f"skipping probe-density figure: {dens.name} not present")
    print(f"{made} figure(s) written to {out}")


if __name__ == "__main__":
    main()
