#!/bin/bash
# IITG approved single-MIG large allocation: 96 GB VRAM, 24 CPUs, 96 GB host RAM.
# Submit via: bash cluster/submit.sh large benchmark
#SBATCH --job-name=vla-large
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=gpu_large
#SBATCH --qos=gpu_large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4g.96gb:1
#SBATCH --cpus-per-gpu=24
#SBATCH --mem-per-gpu=96G
#SBATCH --time=08:00:00
set -euo pipefail
export VLA_JOB_LEVEL=large
source "${VLA_REPO_ROOT:-$HOME/VLA/Experiments}/cluster/run_job.sh"
