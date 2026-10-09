#!/bin/bash
# IITG approved single-MIG small allocation: 24 GB VRAM, 6 CPUs, 24 GB host RAM.
# Submit via: bash cluster/submit.sh small benchmark
#SBATCH --job-name=vla-small
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=gpu_small
#SBATCH --qos=gpu_small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=24G
#SBATCH --time=04:00:00
set -euo pipefail
export VLA_JOB_LEVEL=small
source "${VLA_REPO_ROOT:-$HOME/VLA/Experiments}/cluster/run_job.sh"
