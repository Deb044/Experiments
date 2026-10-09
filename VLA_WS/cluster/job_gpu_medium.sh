#!/bin/bash
# IITG approved single-MIG medium allocation: 48 GB VRAM, 12 CPUs, 48 GB host RAM.
# Submit via: bash cluster/submit.sh medium benchmark
#SBATCH --job-name=vla-medium
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu_medium
#SBATCH --qos=gpu_medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2g.48gb:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=48G
#SBATCH --time=06:00:00
set -euo pipefail
export VLA_JOB_LEVEL=medium
source "${VLA_REPO_ROOT:-$HOME/VLA/Experiments}/cluster/run_job.sh"
