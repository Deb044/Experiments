#!/usr/bin/env bash

# Define the base NFS path
BASE_DIR="."
EXPERIMENT_DIR="$BASE_DIR/experiments"
MAX_JOBS=1
declare -A job_gpus

while true; do
  echo "Scanning for new runs in $EXPERIMENT_DIR..."
  
  # Read runs into an array first to avoid subshell issues with job control
  #mapfile -t RUNS < <(find -L "$EXPERIMENT_DIR" -mindepth 4 -maxdepth 4 -type d -name "run_*" | sort)

	mapfile -t RUNS < <(
	  find -L "$EXPERIMENT_DIR" -mindepth 4 -maxdepth 4 -type d -name "run_*" \
	  ! -path "*/libero_10/*" \
	  ! -path "*/LIBERO_10/*" \
	  | sort
	)

  for FULL_RUN_PATH in "${RUNS[@]}"; do
    
    # Check for completed jobs and release GPUs
    while true; do
      # Remove finished jobs from tracking
      for pid in "${!job_gpus[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
           unset "job_gpus[$pid]"
        fi
      done

      # Note: ${#job_gpus[@]} is the number of currently running tracked jobs
      if [[ "${#job_gpus[@]}" -lt "$MAX_JOBS" ]]; then
        break
      fi
      # Wait for any job to finish before checking again
      sleep 1
    done

    # Determine free GPU
    # We want to find a GPU ID (0 or 1) that is NOT in the values of job_gpus
    GPU_ID=-1
    # Check 0
    gpu0_busy=false
    for pid in "${!job_gpus[@]}"; do
      if [[ "${job_gpus[$pid]}" == "0" ]]; then
        gpu0_busy=true
        break
      fi
    done

    if [[ "$gpu0_busy" == "false" ]]; then
      GPU_ID=0
    else
      GPU_ID=1
    fi

    # Extract relative path for display/logging
    REL_PATH="${FULL_RUN_PATH#$EXPERIMENT_DIR/}"
    
    # Extract BENCHMARK and ALGO
    BENCHMARK=$(echo "$REL_PATH" | cut -d'/' -f1 | tr '[:upper:]' '[:lower:]')
    ALGO=$(echo "$REL_PATH" | cut -d'/' -f2 | tr '[:upper:]' '[:lower:]')
    
    echo "=========================================================="
    echo "STARTING: $BENCHMARK | $ALGO | $REL_PATH on GPU $GPU_ID"
    echo "=========================================================="
    
    # Execute in background
    ./eval_by_model_path.sh \
      "$EXPERIMENT_DIR" \
      "$BENCHMARK" \
      "$ALGO" \
      "$FULL_RUN_PATH" \
      0 \
      50 \
      "$GPU_ID" &
    
    # Track PID
    job_gpus[$!]="$GPU_ID"
      
  done
  
  # Wait for all jobs in this batch to finish before sleeping/rescan?
  # The user wants "process all... then loop". 
  # Ideally we just keep going if find returns new stuff, but find is a snapshot.
  # Let's wait for all to clear before rescan to be safe and simple.
  wait
  
  echo "Batch loop cycle complete. Sleeping for 60 seconds..."
  sleep 60
done
