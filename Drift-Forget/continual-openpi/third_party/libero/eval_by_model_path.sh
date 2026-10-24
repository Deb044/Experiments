#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Evaluate sequential checkpoints one by one
# ------------------------------------------------------------

EXPERIMENT_DIR="${1:?need EXPERIMENT_DIR}"
BENCHMARK="${2:?need BENCHMARK}"
ALGO="${3:?need ALGO}"
RUN_DIR="${4:?need RUN_DIR (dir containing task*_model*.pth)}"

SEED="${5:-0}"
EP="${6:-50}"
# We only use one GPU in sequential mode
GPU_ID="${7:-0}" 

POLICY="bc_transformer_policy"
SAVE_VIDEOS="--save-videos" 

# Use osmesa for headless rendering; change to egl if using NVIDIA hardware rendering
export MUJOCO_GL="osmesa"

# ---------- Helpers ----------
pick_model_for_task() {
  local i="$1"
  local direct="$RUN_DIR/task${i}_model.pth"
  if [[ -f "$direct" ]]; then
    echo "$direct"
    return 0
  fi

  shopt -s nullglob
  local candidates=( "$RUN_DIR/task${i}_model_epoch"*.pth )
  shopt -u nullglob
  if (( ${#candidates[@]} == 0 )); then
    return 1
  fi

  local best="" best_ep=-1
  for f in "${candidates[@]}"; do
    if [[ "$f" =~ _epoch([0-9]+)\.pth$ ]]; then
      local epnum="${BASH_REMATCH[1]}"
      if (( epnum > best_ep )); then
        best_ep="$epnum"
        best="$f"
      fi
    fi
  done

  if [[ -n "$best" ]]; then
    echo "$best"
    return 0
  fi
  return 1
}

# ---------- Main ----------
MAX_LOAD_TASK=9 

echo "[INFO] Running SEQUENTIAL evaluation"
echo "[INFO] RUN_DIR      = $RUN_DIR"
echo "[INFO] GPU ID       = $GPU_ID"

for ((LOAD_TASK=0; LOAD_TASK<=MAX_LOAD_TASK; LOAD_TASK++)); do
  
  # 1. Locate the model for the current stage of lifelong learning
  if ! MODEL_PATH="$(pick_model_for_task "$LOAD_TASK")"; then
    echo "[WARN] Skipping load_task=$LOAD_TASK (no model found)"
    continue
  fi

  # 2. Evaluate all tasks up to the current stage to check for forgetting
  for ((TASK_ID=0; TASK_ID<=LOAD_TASK; TASK_ID++)); do
    
    results_dir="$(dirname "$MODEL_PATH")"
    log_file="$results_dir/eval_load${LOAD_TASK}_task${TASK_ID}.txt"

    if [[ -f "$log_file" ]]; then
      echo "[INFO] Skipping task${TASK_ID} for load_task=${LOAD_TASK} (log file exists: $log_file)"
      continue
    fi

    echo "--------------------------------------------------------"
    echo "[EXEC] Model: task${LOAD_TASK} | Evaluating on: task${TASK_ID}"
    echo "[LOG]  $log_file"
    echo "model path $MODEL_PATH"

    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    python libero/lifelong/evaluate.py \
      --experiment_dir "$EXPERIMENT_DIR" \
      --benchmark "$BENCHMARK" \
      --task_id "$TASK_ID" \
      --algo "$ALGO" \
      --policy "$POLICY" \
      --seed "$SEED" \
      --ep "$EP" \
      --device_id 0 \
      $SAVE_VIDEOS \
      --load_task ${LOAD_TASK} \
      --model_path "$MODEL_PATH" \
      --results_dir "$results_dir" \
      2>&1 | tee "$log_file"

  done
done

echo "[INFO] Sequential evaluation complete."
