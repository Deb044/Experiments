#!/usr/bin/env bash
set -euo pipefail
O3_ROOT=${O3_ROOT:-$HOME/VLA/Experiments/O3-VLA}
export O3_SCRATCH=${O3_SCRATCH:-/scratch/$USER/VLA/O3-VLA}
export HF_HOME=${O3_HF_HOME:-$O3_SCRATCH/cache/huggingface}
export XDG_CACHE_HOME=$O3_SCRATCH/cache
export TMPDIR=$O3_SCRATCH/tmp
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-12}
mkdir -p "$HF_HOME" "$TMPDIR"
cd "$O3_ROOT"
python=${O3_PYTHON:-$O3_ROOT/.venv/bin/python}
[[ -x "$python" ]] || { echo 'Missing O3 environment. See O3-VLA/README.md.' >&2; exit 2; }
output=${O3_OUTPUT:-$O3_SCRATCH/pilot-${O3_JOB_LEVEL:-medium}}
data=${O3_DATA_ROOT:-$O3_SCRATCH/datasets}
libero=${O3_LIBERO_ROOT:-$O3_SCRATCH/LIBERO}
config=${O3_CONFIG:-$O3_ROOT/configs/pilot.json}
echo "O3-VLA command=${O3_COMMAND:-run} level=${O3_JOB_LEVEL:-unknown} output=$output"
srun "$python" -u -m o3_vla.cli --config "$config" --output "$output" \
  --data-root "$data" --libero-root "$libero" "${O3_COMMAND:-run}"
