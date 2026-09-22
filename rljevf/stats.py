"""Paired statistics for judge comparisons.

Every reward source scores the *same* preference pairs, so comparing them as
two independent proportions throws away the pairing and most of the power. On
300 pairs at ~0.65 accuracy an unpaired 95% interval is about +/-5.4 points,
which is wider than the gap we are trying to resolve; the paired tests below
use the per-pair agreement structure instead.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict
from typing import Sequence


@dataclass
class PairedComparison:
    n: int
    acc_a: float
    acc_b: float
    diff: float
    # McNemar on the discordant pairs
    b: int                    # a right, b wrong
    c: int                    # a wrong, b right
    mcnemar_stat: float
    p_value: float
    ci_low: float
    ci_high: float

    def to_dict(self) -> dict:
        return asdict(self)

    def verdict(self, alpha: float = 0.05) -> str:
        if self.p_value < alpha:
            better = "A" if self.diff > 0 else "B"
            return f"{better} better (p={self.p_value:.4f})"
        return f"not separable at n={self.n} (p={self.p_value:.3f})"


def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def paired_compare(
    correct_a: Sequence[bool], correct_b: Sequence[bool],
    n_boot: int = 10000, seed: int = 0,
) -> PairedComparison:
    """McNemar's exact-ish test plus a bootstrap CI on the paired difference."""
    a = [bool(x) for x in correct_a]
    b = [bool(x) for x in correct_b]
    assert len(a) == len(b) and a, "need equal, non-empty sequences"
    n = len(a)
    nb = sum(1 for x, y in zip(a, b) if x and not y)
    nc = sum(1 for x, y in zip(a, b) if y and not x)

    if nb + nc == 0:
        stat, p = 0.0, 1.0
    else:
        # continuity-corrected McNemar; for small discordant counts fall back
        # to the exact binomial two-sided test
        if nb + nc < 25:
            k = min(nb, nc)
            tail = sum(math.comb(nb + nc, i) for i in range(k + 1)) / 2 ** (nb + nc)
            p = min(1.0, 2 * tail)
            stat = float(nb - nc)
        else:
            stat = (abs(nb - nc) - 1) ** 2 / (nb + nc)
            # two-sided p for a chi-square with one degree of freedom
            p = math.erfc(math.sqrt(stat / 2))

    rng = random.Random(seed)
    diffs = []
    idx = range(n)
    for _ in range(n_boot):
        s = [rng.choice(idx) for _ in idx]
        diffs.append(sum(a[i] for i in s) / n - sum(b[i] for i in s) / n)
    diffs.sort()

    return PairedComparison(
        n=n,
        acc_a=sum(a) / n,
        acc_b=sum(b) / n,
        diff=sum(a) / n - sum(b) / n,
        b=nb, c=nc,
        mcnemar_stat=float(stat),
        p_value=float(p),
        ci_low=diffs[int(0.025 * n_boot)],
        ci_high=diffs[int(0.975 * n_boot) - 1],
    )


def required_n(p1: float, p2: float, rho: float = 0.5,
               alpha: float = 0.05, power: float = 0.8) -> int:
    """Roughly how many paired items are needed to resolve p1 vs p2.

    `rho` is the correlation between the two judges' per-item correctness;
    higher correlation means fewer items are needed, which is why the paired
    design matters here.
    """
    z_a, z_b = 1.959964, 0.8416212
    var = p1 * (1 - p1) + p2 * (1 - p2) - 2 * rho * math.sqrt(
        p1 * (1 - p1) * p2 * (1 - p2))
    var = max(var, 1e-9)
    return int(math.ceil(((z_a + z_b) ** 2 * var) / (p1 - p2) ** 2))
