# IIT Guwahati GPU jobs

The three launchers use the resource settings in the supplied facility templates.
They run one Python process on one MIG slice. Code/environment stay in home;
data and large outputs stay in scratch.

| Level | Partition / QoS | GRES | CPUs | Host RAM | Advertised VRAM | Default / maximum time |
|---|---|---|---:|---:|---:|---|
| small | `gpu_small` | `gpu:1g.24gb:1` | 6 | 24 GB | 24 GB | 4 / 24 hours |
| medium | `gpu_medium` | `gpu:2g.48gb:1` | 12 | 48 GB | 48 GB | 6 / 48 hours |
| large | `gpu_large` | `gpu:4g.96gb:1` | 24 | 96 GB | 96 GB | 8 / 48 hours |

Measured usable VRAM can differ from these advertised capacities. Small permits
two MIGs in the source template, but this implementation requests one: two
24 GB slices do not give this single-device model a 48 GB memory pool.

## First submission

From the login node, with the research environment already installed:

```bash
cd "$HOME/VLA/Experiments"
# Install into the same environment that has the compatible CUDA Torch/LeRobot.
.venv/bin/python -m pip install -e '.[cluster]'
bash cluster/submit.sh medium benchmark
```

If the repo itself is at `$HOME/VLA` instead, run from that directory. The submit
wrapper discovers the repo from its own location. Logs still go to the explicitly
requested `$HOME/VLA/Experiments/logs`. Set `VLA_PYTHON` to your environment's
absolute Python path if you use Conda or another venv; load the required modules
and activate the environment before submitting. No module names are guessed.

Use `submit.sh`, which creates the log directory **before** `sbatch` opens its
files. Slurm does not expand `$HOME` in `#SBATCH` directives; the wrapper supplies
absolute paths. Do not submit the templates directly unless you also supply
absolute `--chdir`, `--output` and `--error` paths and create the log directory.

Every command first prepares/reuses its immutable manifest. A benchmark job runs
three independent model-loading/warmup/measurement repetitions by default
(`VLA_BENCHMARK_REPEATS=1` for a quick check). Each repeat measures 100 complete
updates after 50 warmup updates. Cold startup/download time is visible in job
logs but is excluded from measured step throughput. Use the same config and
immutable model/dataset revisions for cross-level comparisons.

## Files and live logs

```text
$HOME/VLA/Experiments/logs/
  vla-medium-<jobid>.out             stdout and flushed progress
  vla-medium-<jobid>.err             stderr, warnings and tracebacks
  jobs/<jobid>-<restart>/
    config.json, git-commit.txt, git-status.txt
    slurm-job.txt, slurm-accounting.txt, exit-code.txt
    node-gpu-inventory.txt           node-wide diagnostic, not job utilization
    prepare/gpu-*.jsonl, gpu-*-summary.json
    command-1/gpu-*.jsonl, gpu-*-summary.json
    benchmark-1.json                repeat archives (2 and 3 likewise)
/scratch/$USER/VLA/
  cache/                            Hugging Face, LeRobot, Torch and plot caches
  tmp/<jobid>/                      temporary files
  experiments/pilot-medium/
    config.json, manifest.json, benchmark.json
    benchmarks/<UTC timestamp>/    measured-window GPU stream + benchmark
    runs/, pairs/, predictions/, functional/, report/
```

```bash
tail -f "$HOME/VLA/Experiments/logs/vla-medium-<jobid>.out"
tail -f "$HOME/VLA/Experiments/logs/vla-medium-<jobid>.err"
squeue -u "$USER"
```

The script propagates the application's exit code to Slurm and records it locally.
Accounting can lag and may leave `MaxRSS` blank at job exit; obtain a final report
later with `sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,MaxRSS,AllocTRES`.
Live samples are flushed even if the process later fails. Summary files are written
on ordinary completion/Python exceptions, including CUDA OOM, but a hard kill or
node failure can prevent a final summary. Periodic training checkpoints remain
resumable; wall-time termination may lose updates since the last saved checkpoint.

## Selecting the job level

Run the same benchmark on each level, checking the small job for OOM before a full
pilot. Each level uses a separate experiment/output directory by default.

```bash
bash cluster/submit.sh small benchmark
bash cluster/submit.sh medium benchmark
bash cluster/submit.sh large benchmark

# Once those jobs finish; exclude failed jobs without a benchmark result.
.venv/bin/python -m vla_weightspace.cluster compare \
  "$HOME/VLA/Experiments/logs/jobs/<small-id>-0/"benchmark-*.json \
  "$HOME/VLA/Experiments/logs/jobs/<medium-id>-0/"benchmark-*.json \
  "$HOME/VLA/Experiments/logs/jobs/<large-id>-0/"benchmark-*.json \
  --csv "$HOME/VLA/Experiments/logs/gpu-level-comparison.csv"
```

The CSV gives usable capacity in GiB, peak reserved tensor memory, sampled total
device memory, mean/p95 update time, samples/second, training-hours estimate,
utilization sample count and a VRAM headroom flag. By default the observed peak
must fit with 20% free capacity (`--headroom 0.2`). This is a planning margin,
not a guarantee against OOM on longer sequences or a later pipeline stage.

Compare the distribution of the three repeats. Choose the smallest level with
adequate VRAM headroom and a training-time estimate that fits your access window;
use a larger level when its measured throughput makes the time budget feasible.
MIG profiles allocate different compute resources as well as memory: 48 GB does
not imply half the speed of 96 GB. CPU allocation also differs by tier. The table
measures complete job-level performance, not an isolated causal effect of VRAM.
Queue delay and allocation costs are not estimated.

The comparison rejects synthetic fixtures and mismatched workloads, including
microbatch, precision, revisions and benchmark update counts. To test a smaller
microbatch after an OOM, use a **new** experiment name (for example
`VLA_MICROBATCH=1 VLA_EXPERIMENT=pilot-small-mb1`) and compare against jobs using
that same microbatch. Do not reinterpret a successful small benchmark as proof
that extraction/evaluation fit: run one paired training run and the later stages,
then inspect their whole-command memory peaks before committing the full budget.

## MIG utilization

NVML is loaded from the optional `nvidia-ml-py` package. The sampler resolves the
CUDA device's UUID (or explicit MIG UUID in visibility settings), and refuses a
physical-index fallback or an enabled MIG parent. It records PyTorch allocation
and reservation peaks plus total device memory occupancy from CUDA/NVML. These
two memory measures have different scopes; sampled occupancy can miss short peaks.

NVIDIA documents that NVML/`nvidia-smi` utilization attribution can be unavailable
on MIG. In that case utilization is **null**, with an error reason; null means
unknown, not 0% busy. Memory-interface utilization, when supported, measures
memory activity rather than the percentage of VRAM occupied. The observed GPU
activity on a parent card may include other users and is never substituted.
See [NVIDIA's MIG monitoring documentation](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/getting-started-with-mig.html).

For actual MIG compute/SM/tensor activity, use the cluster's existing DCGM service
if administrators expose profiling. Have them provide a DCGM group containing
**only the MIG slice assigned to this job**, and export `VLA_DCGM_GROUP=<groupid>`
for that allocation. This opt-in starts a read-only `dcgmi dmon` stream for fields
1001/1002/1004/1005 (engine, SM, tensor and DRAM activity) under
`jobs/<jobid>-<restart>/dcgm-profiling.txt`, with a separate `.err` file and discovery
snapshot. No host engine is started and no device/group is reconfigured. Verify
the assigned-slice identity in the discovery snapshot. Group IDs must be resolved
for the actual allocation and should not be reused blindly across queued jobs.
These raw DCGM ratios are separate from the NVML percentage columns in the CSV;
they are not silently combined. An unavailable/unauthorized DCGM stream records
errors without turning its output into utilization measurements. See
[NVIDIA's DCGM profiling guide](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/profiling.html).

## Train and resume

### Recovering from `goal-0: need at least ten distinct complete trajectories`

The original selector assumed every episode metadata row contained `tasks`.
The LIBERO v3 export does not have that column, so it selected zero episodes.
This failure is in preparation, before model training; the Hugging Face
unauthenticated-download warning is unrelated. Updated preparation resolves
task indices from frame parquet files and preserves the trajectory requirement.

On the cluster, update the code and reinstall the editable package into the
**same environment used by the job**, without changing its research dependencies:

```bash
cd "$HOME/VLA/Experiments"
git pull --ff-only
cd VLA_WS
export VLA_PYTHON="${VLA_PYTHON:-$PWD/.venv/bin/python}"
"$VLA_PYTHON" -m pip install -e . --no-deps
"$VLA_PYTHON" -m vla_weightspace.cli doctor
VLA_BENCHMARK_REPEATS=1 bash cluster/submit.sh medium benchmark
```

If your environment lives elsewhere, set `VLA_PYTHON` to that environment's
absolute `bin/python` path before these commands. Doctor prints the interpreter,
environment prefix and imported code path; the code path should be inside
`VLA_WS/src`. CUDA availability on a login node can be false; check it in an
allocated GPU job. There is no reason to rebuild the environment solely for
this traceback. The failed preparation did not create a manifest/checkpoints,
so rerun using the same scratch experiment name/config; do not delete caches
or change the scientific split settings.

The first fixed preparation scans/cache-downloads the small frame parquet files
and logs `Task-label scan` followed by all three selected trajectory counts.
Subsequent preparation reuses the immutable manifest. If selection is still
insufficient, the error now prints matched counts and available task labels.

```bash
# Use the benchmarked medium protocol and its persistent output directory.
VLA_WALLTIME=01:00:00 bash cluster/submit.sh medium train \
  --run-id goal-0/seed-0/full --max-updates 100
VLA_WALLTIME=06:00:00 bash cluster/submit.sh medium train \
  --run-id goal-0/seed-0/full
# Request a longer permitted allocation once the measured budget is clear.
VLA_WALLTIME=1-00:00:00 bash cluster/submit.sh medium run
```

Reusing a level/experiment resumes the same protocol, pins and checkpoints.
Changing config or microbatch requires a new experiment name. A file lock prevents
two jobs from writing the same experiment simultaneously. Scratch retention is
site-dependent: archive needed final results/manifests outside scratch before
the facility purges it. Jobs never automatically delete scratch data.

Overrides: `VLA_REPO_ROOT`, `VLA_PYTHON`, `VLA_CONFIG`, `VLA_EXPERIMENT`,
`VLA_SCRATCH_ROOT`, `VLA_LOG_ROOT`, `VLA_MICROBATCH`, `VLA_WALLTIME`, `VLA_ACCOUNT`,
`VLA_BENCHMARK_REPEATS`. Defaults match the requested home/log and scratch locations.

The launchers have CPU fixture/invariant tests and a Linux launcher integration
test with simulated Slurm commands. Actual Slurm admission, CUDA/NVML/DCGM access
and GPU measurements must be checked on the facility; no cluster benchmark numbers
are fabricated here.
