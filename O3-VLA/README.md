# O3-VLA: continual robot behavior suppression

The second experiment in this repository extends the sequential-request and
orthogonal-adapter ideas of [O3](https://github.com/GCYZSL/O3-LLM-UNLEARNING) to
OpenVLA. It tests whether suppressing a new robot task restores earlier
suppressed tasks, and whether orthogonal regularization reduces that interference.
The base policy is frozen and still accessible when adapters are disabled:
**this is conditional behavior suppression, not certified data deletion**.

Read [EXPERIMENTS.md](EXPERIMENTS.md) for the questions, comparisons, metrics,
and interpretation; [SOURCE_ALIGNMENT.md](SOURCE_ALIGNMENT.md) for native model
contracts and differences from O3. `VLA_WS` is independent and untouched.

## Installation on the GPU cluster

Use an independent Python 3.10 or 3.11 environment. Do not reuse or modify the
working VLA_WS environment. Install the CUDA Torch/torchvision pair appropriate
to Blackwell first; the previous SmolVLA timing does not predict OpenVLA timing.

```bash
cd "$HOME/VLA/Experiments/O3-VLA"
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e '.[openvla,test]'
```

Install LIBERO at the revision in `configs/pilot.json` into an O3-specific
scratch checkout and install its simulation dependencies. Do not install
LIBERO's historical full requirements over your working CUDA stack; many are
training dependencies not used by this experiment. See the setup instructions
below. The real CUDA/quantization/EGL combination must pass `prepare` and
`benchmark` on the cluster before treating it as usable.

```bash
export O3_SCRATCH=/scratch/$USER/VLA/O3-VLA
mkdir -p "$O3_SCRATCH"
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$O3_SCRATCH/LIBERO"
git -C "$O3_SCRATCH/LIBERO" checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
python -m pip install -e "$O3_SCRATCH/LIBERO"
python -m pip install robosuite==1.4.1 mujoco==2.3.7 bddl==1.0.1 gym==0.25.2 easydict pyyaml cloudpickle 'opencv-python-headless<4.12' 'numpy<2'
```

Obtain the official **LIBERO-Object HDF5 demonstrations** using LIBERO's dataset
download instructions. Place the suite under
`$O3_SCRATCH/datasets/libero_object/`; filenames must match the suite's
`get_task_demonstration()` paths. RLDS-converted data is not accepted by this
reader. Download/model setup time is included in the one-day feasibility budget.
LIBERO assets and initial-state files come from the pinned checkout. The runner
writes its own LIBERO configuration before importing the package, avoiding
interactive prompts and changes to another experiment's LIBERO configuration.

## Run and resume

```bash
cd "$HOME/VLA/Experiments/O3-VLA"
bash cluster/submit.sh medium prepare
bash cluster/submit.sh medium benchmark
# Submit subsequent stages after the preceding job has completed:
bash cluster/submit.sh medium run
```

`run` includes prepare, benchmark, features/detectors, matched training, offline
evaluation and paired rollouts. Completed artifacts are reused; partial training
resumes from the last atomic optimizer checkpoint. Never run two writers against
the same output folder. Logs and errors go to **`$HOME/VLA/Experiments/logs`**;
large outputs/caches go to **`/scratch/$USER/VLA/O3-VLA`**. Override paths using
`O3_OUTPUT`, `O3_DATA_ROOT`, `O3_LIBERO_ROOT`, `O3_PYTHON`, or `O3_LOG_ROOT`.

The approved medium template allows **six hours per job**, not 24 hours.
The one-day protocol can require several sequential allocations, excluding queue
waiting. A longer single job can be requested with `O3_WALLTIME=24:00:00` only
if permitted by the partition/QOS. Process termination can interrupt feature
caching or reporting; those stages restart safely, while completed episodes and
training checkpoints resume. The CLI's 24-hour limit is per invocation, so also
track cumulative job time when enforcing the study's one-day limit.

Commands can be submitted individually: `doctor`, `prepare`, `benchmark`,
`features`, `train`, `evaluate`, `rollout`, `report`, `run`. Set `O3_OUTPUT` to a
new directory for any scientific configuration change. Keep one immutable
configuration across stages.

## CPU verification

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python -m o3_vla.cli --config configs/smoke.json run
```

The synthetic fixture exercises splitting, training, resume, gating, simulator
bookkeeping and exported reports without downloading a VLA. Its outputs and
figures are prominently labeled **not research results**. Passing CPU tests does
not establish GPU feasibility or real robot suppression.

## Outputs and hardware benchmarking

Each run saves its configuration/source identity, trajectory manifest and file
hashes, adapter/optimizer checkpoints, raw offline CSV, paired episode JSON/CSV,
control-step traces, detector metrics, and per-phase GPU telemetry. `report/`
contains trajectory-averaged tables, recovery after subsequent requests,
suppression/utility comparisons, paired differences, and PNG/PDF figures.

`benchmark.json` measures actual optimizer updates on the selected slice and
freezes the common 200/100/50 update budget for both arms. It estimates training
time only. Telemetry separately records phase time, host RSS, allocator peaks,
allocated-device VRAM and utilization when available. Unsupported MIG utilization
is null; a node-wide reading is never substituted for your slice.

Medium (48GB) is the target. Small (24GB) is a benchmark option, **not a claim
that the native experiment fits**. Large (96GB) is optional. Compare actual
update latency, available memory and end-to-end phase times across slices using
new output directories; do not reuse another slice's timing estimate.

SDPA is the default attention implementation. `flash_attention_2` is an optional
configuration requiring a compatible separately installed `flash-attn` build.
It is not automatically enabled or silently substituted. Blackwell support and
speed must be measured on your installed stack; an FA2 comparison needs a new
configuration/output directory and the same data, precision and update budget.
