#!/usr/bin/env bash
#SBATCH --job-name=o3-vla-medium
#SBATCH --partition=gpu_medium
#SBATCH --qos=gpu_medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2g.48gb:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=48G
#SBATCH --time=06:00:00
set -euo pipefail
export O3_JOB_LEVEL=medium
source "${O3_ROOT:-$HOME/VLA/Experiments/O3-VLA}/cluster/run_job.sh"
