#!/usr/bin/env bash
#SBATCH --job-name=o3-vla-large
#SBATCH --partition=gpu_large
#SBATCH --qos=gpu_large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4g.96gb:1
#SBATCH --cpus-per-gpu=24
#SBATCH --mem-per-gpu=96G
#SBATCH --time=08:00:00
set -euo pipefail
export O3_JOB_LEVEL=large
source "${O3_ROOT:-$HOME/VLA/Experiments/O3-VLA}/cluster/run_job.sh"
