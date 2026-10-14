#!/bin/bash
# Shared application body, sourced by the approved resource templates.
set -euo pipefail
: "${SLURM_JOB_ID:?Run through cluster/submit.sh on Slurm}"
repo=${VLA_REPO_ROOT:-$HOME/VLA/Experiments}
log_root=${VLA_LOG_ROOT:-$HOME/VLA/Experiments/logs}
scratch=${VLA_SCRATCH_ROOT:-/scratch/$USER/VLA}
experiment=${VLA_EXPERIMENT:-pilot-${VLA_JOB_LEVEL}}
if [[ ! $experiment =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then echo 'Invalid VLA_EXPERIMENT (use a simple directory name)' >&2; exit 2; fi
python=${VLA_PYTHON:-$repo/.venv/bin/python}
source_config=${VLA_CONFIG:-$repo/configs/pilot.json}
output=$scratch/experiments/$experiment
job_dir=$log_root/jobs/${SLURM_JOB_ID}-${SLURM_RESTART_COUNT:-0}
mkdir -p -- "$job_dir" "$output" "$scratch/cache" "$scratch/tmp/$SLURM_JOB_ID"
export HF_HOME=$scratch/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$scratch/cache/datasets
export HF_LEROBOT_HOME=$scratch/cache/lerobot
export XDG_CACHE_HOME=$scratch/cache/xdg
export TORCH_HOME=$scratch/cache/torch
export TMPDIR=$scratch/tmp/$SLURM_JOB_ID
export MPLCONFIGDIR=$scratch/cache/matplotlib
export PYTHONUNBUFFERED=1
cd -- "$repo"
if [[ ! -x $python ]]; then echo "Missing environment Python: $python (set VLA_PYTHON)" >&2; exit 2; fi
# Host RAM differs from VRAM. Keep Torch/BLAS within the allocated CPU budget.
case "$VLA_JOB_LEVEL" in small) cpus=6;; medium) cpus=12;; large) cpus=24;; esac
export OMP_NUM_THREADS=$cpus
export MKL_NUM_THREADS=$cpus
if (($# == 0)); then set -- benchmark; fi
printf 'Job=%s level=%s partition=%s CUDA_VISIBLE_DEVICES=%s\nRepo=%s\nOutput=%s\nLogs=%s\n' \
    "$SLURM_JOB_ID" "$VLA_JOB_LEVEL" "${SLURM_JOB_PARTITION:-unknown}" "${CUDA_VISIBLE_DEVICES:-unset}" "$repo" "$output" "$job_dir"
printf 'Command:'; printf ' %q' "$@"; printf '\n'
git rev-parse HEAD > "$job_dir/git-commit.txt"
git status --porcelain > "$job_dir/git-status.txt"
# These diagnostics may show the whole node; they are explicitly NOT job utilization.
if command -v nvidia-smi >/dev/null; then nvidia-smi -L > "$job_dir/node-gpu-inventory.txt" 2>&1 || true; fi
scontrol show job "$SLURM_JOB_ID" > "$job_dir/slurm-job.txt" 2>&1 || true
# One writer per experiment; the descriptors/locks persist across all srun steps. Training
# restricted to one task (optionally one seed) writes only that slice's runs, so such jobs
# hold the experiment lock shared and may run side by side once the manifest exists.
task= seed=
if [[ $1 == train ]]; then
    arguments=("${@:2}")
    for ((i = 0; i < ${#arguments[@]}; i += 1)); do
        case "${arguments[i]}" in
            --task) task=${arguments[i+1]:-};;
            --task=*) task=${arguments[i]#--task=};;
            --seed) seed=${arguments[i+1]:-};;
            --seed=*) seed=${arguments[i]#--seed=};;
        esac
    done
fi
if [[ -z $task ]]; then seed=; fi
if [[ -n $task && ! $task =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || [[ -n $seed && ! $seed =~ ^[0-9]+$ ]]; then
    echo 'Invalid --task or --seed for a training job' >&2; exit 2
fi
exec 9> "$output/.job.lock"
if [[ -z $task ]]; then
    if ! flock -n 9; then echo "Experiment already has a running job: $output" >&2; exit 2; fi
else
    if [[ ! -f $output/manifest.json ]]; then echo "Prepare the experiment once before per-task training jobs: $output" >&2; exit 2; fi
    if ! flock -n -s 9; then echo "Experiment has a running job that needs it exclusively: $output" >&2; exit 2; fi
    exec 8> "$output/.job-$task.lock"
    if [[ -z $seed ]]; then
        if ! flock -n 8; then echo "Task $task already has a running training job: $output" >&2; exit 2; fi
    else
        if ! flock -n -s 8; then echo "Task $task has a running job covering every seed: $output" >&2; exit 2; fi
        exec 7> "$output/.job-$task-seed-$seed.lock"
        if ! flock -n 7; then echo "Task $task seed $seed already has a running training job: $output" >&2; exit 2; fi
    fi
fi
dcgm_pid=
finish() {
    status=$?
    trap - EXIT
    if [[ -n $dcgm_pid ]]; then kill "$dcgm_pid" 2>/dev/null || true; wait "$dcgm_pid" 2>/dev/null || true; fi
    printf 'Exit code=%s completion=%s\n' "$status" "$(date -Is)"
    printf '%s\n' "$status" > "$job_dir/exit-code.txt"
    # Accounting can lag; this is best effort, not a substitute for gpu samples.
    sacct -j "$SLURM_JOB_ID" --parsable2 --format=JobID,JobName,Partition,State,ExitCode,Elapsed,AllocTRES,MaxRSS > "$job_dir/slurm-accounting.txt" 2>&1 || true
    exit "$status"
}
trap finish EXIT
config_args=(--source "$source_config" --destination "$output/config.json" --output "$output")
if [[ -n ${VLA_MICROBATCH:-} ]]; then config_args+=(--microbatch="$VLA_MICROBATCH"); fi
"$python" -u -m vla_weightspace.cluster config "${config_args[@]}"
cp -- "$output/config.json" "$job_dir/config.json"
# Optional MIG profiling through an administrator-provided DCGM group containing
# ONLY this job's slice. Never default to the all-GPU group or start hostengine.
if [[ -n ${VLA_DCGM_GROUP:-} ]]; then
    if [[ ! $VLA_DCGM_GROUP =~ ^[0-9]+$ ]] || ! command -v dcgmi >/dev/null; then
        echo 'VLA_DCGM_GROUP requires a numeric allocated-slice group and installed dcgmi' >&2; exit 2
    fi
    dcgmi discovery -c > "$job_dir/dcgm-discovery.txt" 2>&1 || true
    dcgmi dmon -e 1001,1002,1004,1005 -g "$VLA_DCGM_GROUP" -d 2000 > "$job_dir/dcgm-profiling.txt" 2> "$job_dir/dcgm-profiling.err" &
    dcgm_pid=$!
fi
if [[ $1 != doctor ]]; then
    srun "$python" -u -m vla_weightspace.cli --config "$output/config.json" --telemetry-dir "$job_dir/prepare" prepare
fi
repeats=1
if [[ $1 == benchmark ]]; then repeats=${VLA_BENCHMARK_REPEATS:-3}; fi
if [[ ! $repeats =~ ^[1-9][0-9]*$ ]]; then echo 'Invalid VLA_BENCHMARK_REPEATS' >&2; exit 2; fi
for ((attempt=1; attempt<=repeats; attempt++)); do
    srun "$python" -u -m vla_weightspace.cli --config "$output/config.json" --telemetry-dir "$job_dir/command-$attempt" "$@"
    if [[ $1 == benchmark || $1 == run ]]; then cp -- "$output/benchmark.json" "$job_dir/benchmark-$attempt.json"; fi
done
