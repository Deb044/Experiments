#!/usr/bin/env bash
#SBATCH --job-name=o3-vla-small
#SBATCH --partition=gpu_small
#SBATCH --qos=gpu_small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=24G
#SBATCH --time=04:00:00
set -euo pipefail
export O3_JOB_LEVEL=small
source "${O3_ROOT:-$HOME/VLA/Experiments/O3-VLA}/cluster/run_job.sh"
