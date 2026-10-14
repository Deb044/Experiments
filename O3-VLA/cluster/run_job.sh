#!/usr/bin/env bash
set -euo pipefail
O3_ROOT=${O3_ROOT:-$HOME/VLA/Experiments/O3-VLA}
export O3_SCRATCH=${O3_SCRATCH:-/scratch/$USER/VLA/O3-VLA}
export HF_HOME=${O3_HF_HOME:-$O3_SCRATCH/cache/huggingface}
export XDG_CACHE_HOME=$O3_SCRATCH/cache
export TMPDIR=$O3_SCRATCH/tmp
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-12}
mkdir -p "$HF_HOME" "$TMPDIR"
cd "$O3_ROOT"
python=${O3_PYTHON:-$O3_ROOT/.venv/bin/python}
[[ -x "$python" ]] || { echo 'Missing O3 environment. See O3-VLA/README.md.' >&2; exit 2; }
# MIG slices have no NVIDIA graphics: mesa-egl renders in software. egl needs a whole GPU.
render=${O3_RENDER:-mesa-egl}
case "$render" in
  egl|osmesa) export MUJOCO_GL=$render PYOPENGL_PLATFORM=$render ;;
  mesa-egl)
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    export __EGL_VENDOR_LIBRARY_FILENAMES=${O3_MESA_EGL_VENDOR:-/usr/share/glvnd/egl_vendor.d/50_mesa.json}
    [[ -r "$__EGL_VENDOR_LIBRARY_FILENAMES" ]] || { echo "Missing Mesa EGL vendor file $__EGL_VENDOR_LIBRARY_FILENAMES" >&2; exit 2; }
    # Mesa lists unusable DRM cards first, so the software device must be selected explicitly.
    MUJOCO_EGL_DEVICE_ID=$("$python" -m o3_vla.render) || { echo 'mesa-egl: no working EGL device' >&2; exit 2; }
    export MUJOCO_EGL_DEVICE_ID ;;
  *) echo 'Choose O3_RENDER=egl, mesa-egl or osmesa' >&2; exit 2 ;;
esac
output=${O3_OUTPUT:-$O3_SCRATCH/pilot-${O3_JOB_LEVEL:-medium}}
data=${O3_DATA_ROOT:-$O3_SCRATCH/datasets}
libero=${O3_LIBERO_ROOT:-$O3_SCRATCH/LIBERO}
config=${O3_CONFIG:-$O3_ROOT/configs/pilot.json}
echo "O3-VLA command=${O3_COMMAND:-run} level=${O3_JOB_LEVEL:-unknown} output=$output render=$render egl_device=${MUJOCO_EGL_DEVICE_ID:-unset}"
srun "$python" -u -m o3_vla.cli --config "$config" --output "$output" \
  --data-root "$data" --libero-root "$libero" "${O3_COMMAND:-run}"
