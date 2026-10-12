# Verification boundary — 12 October 2026

The implementation was checked in an independent local CPU environment. No
VLA_WS environment or source file was modified.

- Thirteen tests passed: trajectory isolation, causal raw-HDF5 alignment and
  dataset hashes, native action-token/label contracts through a CPU interface
  fixture, frozen base/adapter gating, output-factor penalty gradients,
  calibration ties, gate reset, current-request-only sampling, matched stage 1,
  exact interrupted optimizer resume, complete pipeline/resume/identity guard,
  missing-episode detection, and honest native setup-failure reporting.
- The actual CLI smoke command completed all six stages with six adapter
  checkpoints, 360 offline rows and 60 paired synthetic episodes.
- Ruff checks and formatting passed. The local installed core dependencies
  passed their dependency consistency check.
- PNG outputs were visually inspected; heatmap scales are shared between arms,
  coincident trade-off points use a legend, and physical displacement is plotted
  separately from simulator command magnitudes. PDF counterparts are exported.
- The proposed repository tree preserves the original VLA_WS tree object
  `1b652f841b3f6b8a062246737fb6cb08b123200a`.

This is **not** native research validation. The 7B CUDA/NF4 backend, Blackwell
slice throughput, FlashAttention-2, EGL rendering, cluster job submission and
real LIBERO suppression were not executed locally. Windows Bash could not
start in this environment, so the cluster scripts were reviewed but not
executed. On Linux, check them individually before submission:
`for script in cluster/*.sh; do bash -n "$script"; done`.
Run native `prepare` and `benchmark` first; a failed native setup
must remain an incomplete engineering report.

The CPU environment used Python 3.12 and Torch 2.14.1+cpu. Native setup targets
a separate Linux Python 3.10/3.11 CUDA environment documented in README.md.
Synthetic scores are deliberately labeled plumbing results and must never be
used as evidence for any research hypothesis.
