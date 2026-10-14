#!/bin/bash
# Run on a login node; creates Slurm log directories BEFORE submitting.
set -euo pipefail
level=${1:-medium}
if (($#)); then shift; fi
case "$level" in small|medium|large) ;; *) echo 'Expected small, medium or large' >&2; exit 2;; esac
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export VLA_REPO_ROOT=${VLA_REPO_ROOT:-$(cd -- "$script_dir/.." && pwd)}
export VLA_LOG_ROOT=${VLA_LOG_ROOT:-$HOME/VLA/Experiments/logs}
export VLA_SCRATCH_ROOT=${VLA_SCRATCH_ROOT:-/scratch/$USER/VLA}
export VLA_EXPERIMENT=${VLA_EXPERIMENT:-pilot-$level}
mkdir -p -- "$VLA_LOG_ROOT"
options=(--chdir="$VLA_REPO_ROOT" --output="$VLA_LOG_ROOT/%x-%j.out" --error="$VLA_LOG_ROOT/%x-%j.err" --export=ALL)
if [[ -n ${VLA_WALLTIME:-} ]]; then options+=(--time="$VLA_WALLTIME"); fi
if [[ -n ${VLA_ACCOUNT:-} ]]; then options+=(--account="$VLA_ACCOUNT"); fi
if [[ -n ${VLA_DEPENDENCY:-} ]]; then options+=(--dependency="$VLA_DEPENDENCY"); fi
if (($# == 0)); then set -- benchmark; fi
sbatch "${options[@]}" "$script_dir/job_gpu_${level}.sh" "$@"
