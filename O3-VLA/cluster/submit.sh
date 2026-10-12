#!/usr/bin/env bash
set -euo pipefail
level=${1:-medium}
command=${2:-run}
[[ "$level" =~ ^(small|medium|large)$ ]] || { echo 'Choose small, medium or large' >&2; exit 2; }
[[ "$command" =~ ^(doctor|prepare|benchmark|features|train|evaluate|rollout|report|run)$ ]] || exit 2
export O3_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export O3_COMMAND=$command
export O3_LOG_ROOT=${O3_LOG_ROOT:-$HOME/VLA/Experiments/logs}
mkdir -p "$O3_LOG_ROOT"
options=(--chdir="$O3_ROOT" --output="$O3_LOG_ROOT/%x-%j.out" --error="$O3_LOG_ROOT/%x-%j.err")
# Override only if this duration is permitted by the cluster's partition/QOS policy.
if [[ -n "${O3_WALLTIME:-}" ]]; then options+=(--time="$O3_WALLTIME"); fi
sbatch "${options[@]}" "$O3_ROOT/cluster/job_gpu_${level}.sh"
