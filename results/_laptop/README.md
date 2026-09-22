Results computed on the laptop (MPS), kept only for reference.

They must not reach the paper: MPS and CUDA are not comparable on throughput,
and two of these were run at sample sizes too small to support their claim
(probe density at n=15, against n>=64 needed). `scripts/make_tables.py` reads
`results/`, not this directory, so quarantining a file reverts its table to a
visible placeholder rather than leaving a laptop number in the manuscript.

- `probe_density.json` -- n=15, MPS sampling. Re-run on Kaggle at n=64, stride 2.
- `jev_properties.json` -- latency measured from a laptop's network, which
  upper-bounds the provider figure by an unknown amount. Re-measure in-session.
