Measurements too small to support the claim they appear to make.

`judge_benchmark_n300.json` — 300 preference pairs. Resolving the gaps it shows
needs n >= 1308 at zero judge correlation (rljevf/stats.py::required_n), so every
accuracy difference in it sits inside its own confidence interval. Kept only as
a cross-check against the full-size re-run.

`make_tables.py` reads `results/`, not this directory, so moving a file here
reverts its table to a visible placeholder rather than leaving an underpowered
number in the manuscript.
