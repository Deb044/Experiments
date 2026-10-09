# Predictable deletion changes in VLA weight space

This repository implements the focused pilot:

> **Do deletion requests produce predictable changes in VLA weight space?**

It trains paired full-data and retained-only SmolVLA adapters, measures the
effective-weight change `delta = (BA)_clean - (BA)_full`, and predicts that change
on held-out requests and tasks. It does **not** implement continual learning,
claim certified unlearning, or treat loss of robot competence as successful deletion.

## Protocol

`configs/pilot.json` fixes three tasks: LIBERO Goal-0 (middle drawer), Object-0
(alphabet soup into basket), and Spatial-0 (bowl between plate and ramekin).
Only these tasks are selected by exact normalized language instruction.

Each task has a trajectory-level 70/10/20 adaptation/calibration/held-out split,
two training seeds and three independently sampled 20% deletion requests. The
same deletion subsets are used across seeds. The resulting collection has
**24 training runs and 18 paired changes**, each trained for 2,000 optimizer
updates. Sampling is uniform over trajectories, then over frames. Clean runs
never sample deleted trajectories. Calibration episodes are never deleted or
used for optimizer updates; their normalization is shared by each paired run.

The foundation model is frozen. Rank-16, alpha=rank LoRA is applied to q/v
projections in language and action transformers (not the vision tower).
Only adapter parameters are optimized. Frozen-model dropout is disabled in
both paired runs. AdamW uses lr=1e-4, betas=(0.9,0.95), weight decay=1e-10,
gradient clipping=10, effective batch=16 and native SmolVLA flow matching.

Predictors: zero change, mean change, scaled forget-gradient, and a small ridge
predictor using current-weight and forget/retain-gradient summaries. A shuffled
conditioning control is included. PCA, feature scaling, and regularization are
fit on training folds only. PCA uses a sample-sized Gram matrix; dimensions are
bounded by available training pairs (8 requested in the pilot), rather than
pretending 18 models support a 64-dimensional learned representation.

Outer leave-one-request-out and leave-one-task-out tests are separate. Inner
validation groups entire checkpoints (request test) or entire tasks (task test).
The audit records training/test pair IDs and selected ridge values for every
layer/fold. Target clean weights are never used as predictor inputs.

## Install

Use Python **3.12**. Linux is recommended for the research GPU.

```bash
python -m venv .venv
source .venv/bin/activate
export GIT_LFS_SKIP_SMUDGE=1
# Choose a CUDA build supported by the installed NVIDIA driver.
pip install 'torch>=2.7,<2.12' 'torchvision>=0.22,<0.27' \
  --index-url https://download.pytorch.org/whl/cu130
pip install -e '.[research,test]'
vla-weightspace doctor
```

LeRobot is pinned to commit
`b9cb121cb7d4e3e26ec5c906d08dda68d151ba52`; the research extra includes the
dataset and SmolVLA dependencies. `prepare` resolves dataset, policy, and nested
VLM/tokenizer revisions to immutable Hub commits and stores them in the manifest.
The latest refs in the config are used only for initial resolution, never as
mutable training references. A rerun under changed revisions/config fails rather
than mixing artifacts. Copy the original manifest/config for reproducibility.

For a CPU plumbing check without LeRobot or model/dataset downloads:

```bash
pip install -e '.[test]'
pytest -q
vla-weightspace --config configs/smoke.json run
```

The synthetic fixture uses a tiny flow model and explicitly labels every report
as **not VLA research results**. Its timings do not estimate GPU performance.
On Windows, activate with `.venv\Scripts\Activate.ps1` instead of `source`.

## Run the real pilot

```bash
vla-weightspace prepare
vla-weightspace benchmark
# Start with one paired run before committing the full GPU budget:
vla-weightspace train --run-id goal-0/seed-0/full
vla-weightspace train --run-id goal-0/seed-0/clean-101
# Inspect losses, timings, and availability of nontrivial learned updates.
vla-weightspace train
vla-weightspace extract
vla-weightspace predict
vla-weightspace evaluate
vla-weightspace report
```

Or run all stages with `vla-weightspace run`. Preparation downloads metadata and
pins revisions. Training downloads the selected episodes and pretrained weights.
`benchmark` warms up 50 updates then measures 100 complete updates, including data
loading, backward, accumulation, and optimizer steps. Benchmark updates are not
saved as scientific training checkpoints.

`train --task goal-0` restricts training to a task. `train --run-id ...
--max-updates 100` executes a bounded slice and writes a resumable checkpoint.
Rerun the same command without the bound to finish. Resume restores optimizer,
sampler, CPU/CUDA RNG, and adapters; completed runs are integrity checked and skipped.
Do not run multiple workers against the same output directory.

Change device, microbatch, or output **before preparation**. With effective
batch=16, microbatch 1/2/4/8/16 are valid; adjust according to the measured memory.
Use separate configs/output directories for different memory/training protocols.
The code does not pretend a 24 GB VRAM allocation grants 25% of GPU compute.

## Results and interpretation

Outputs are ignored by Git under `artifacts/pilot/`:

| Artifact | Contents |
|---|---|
| `manifest.json` | Episode IDs, deletions, configuration, pinned revisions, fingerprint |
| `runs/<task>/seed-*/.../complete.json` | Loss history, timing, initialization/adapter hashes |
| `runs/.../resume.pt` | Trusted local resumable training state |
| `pairs/*.npz` | Per-layer observed changes, forget gradients and predictor features |
| `predictions/metrics.json` | Fold audits and per-pair/layer prediction metrics |
| `functional/metrics.json` | Matched-noise, held-out native-flow comparisons |
| `benchmark.json` | Actual update throughput, hardware/memory and projected collection time |
| `report/results.md` | Main prediction table and limitations |
| `report/*.csv` | Aggregate/layer/functional/request-versus-seed results |
| `report/*.png` | Magnitude, layer predictability and held-out-task plots |

The report measures squared prediction error, relative L2 error, direction
agreement, and gain over predicting zero. Near-zero target changes have undefined
normalized gains rather than an epsilon-generated success score. Functional
checks compare predicted effective updates with clean models on 128 fixed held-out
windows and two shared noise/time draws, masking padded actions. Predicted dense
updates are evaluated exactly, without a rank-truncation step that would change
the prediction under study. This is a weight-space probe, not an efficient
deployable unlearning algorithm.

Evidence for predictability requires gains over zero, mean change, and shuffled
conditioning on held-out data; direction agreement and functional checks are
supporting diagnostics. Report familiar-request and unseen-task results separately.
Compare same-request seed variation against different-request variation. Two seeds
and three tasks do not support broad statistical or privacy guarantees. Layers are
not independent model replicates. Poor policy learning or negligible full-clean
differences make the pilot inconclusive; low error alone is not evidence of structure.

## Compute planning

The exact reference-training budget is 48,000 updates. At measured 0.4/0.7/1.0/1.6
seconds per update it takes 5.3/9.3/13.3/21.3 hours. Add model loading, checkpoint
I/O, gradient extraction, prediction and functional checks; the report does not
hide those in the training-only estimate. The 1–2 day target requires a working
environment, cached data and adequate GPU access. No actual RTX PRO 6000 benchmark
is bundled or claimed. FlashAttention changes and simulation rollouts are outside
this focused study.

## Code structure

- `backends/`: explicit synthetic fixture and native SmolVLA integration.
- `manifest.py`, `prepare.py`: immutable provenance and trajectory splits.
- `lora.py`, `training.py`: identifiable updates and resumable paired adaptation.
- `extraction.py`, `features.py`: actual deletion changes and target-free features.
- `prediction.py`: train-only PCA/ridge, nested grouped folds and controls.
- `evaluation.py`, `report.py`: functional probes, tables and plots.
- `runtime.py`, `benchmark.py`, `cli.py`: timing and stage orchestration.
- `tests/`: scientific invariants, gradients, checkpoint resume and pipeline tests.

Only internally generated training checkpoints should be loaded; `resume.pt` is
not an interchange format for untrusted model files. NumPy artifacts disable pickle.
