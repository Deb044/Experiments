import os
import json
import argparse
import re
from pathlib import Path
import numpy as np

def compute_metrics(conf_mat: np.ndarray):
    """
    Parameters
    ----------
    conf_mat : (n_tasks, n_tasks) ndarray
        Row i holds the accuracies (or success rates) of the model that has
        been trained up through task i, evaluated on tasks 0…i.

    Returns
    -------
    dict with keys 'fwt', 'bwt', 'auc'
        * fwt : mean forward transfer (ignoring NaNs)
        * bwt : mean backward transfer (ignoring NaNs)
        * auc : mean of per‑task areas under the learning curve (ignoring NaNs)
    """
    n_tasks = conf_mat.shape[0]

    # ---------- Forward transfer -------------------------------------------
    # FWT: mean of diagonal, skipping NaNs
    fwt = float(np.nanmean(np.diag(conf_mat)))

    # ---------- Backward transfer ------------------------------------------
    # For every task k, compare its final accuracy (diag) with its accuracy
    # after learning each later task τ>k, then average.

    bwt_per_task = []
    for k in range(n_tasks):
        # We only look at tasks that have future tasks trained after them
        if k == n_tasks - 1:
            continue

        val_k = conf_mat[k, k]
        vals_future = conf_mat[k+1:, k]

        # Calculate deltas: (original_perf - current_perf)
        # Note: If val_k is NaN, all deltas are NaN
        deltas = val_k - vals_future

        # Mean of deltas for this task, ignoring NaNs
        mean_delta = np.nanmean(deltas)

        if not np.isnan(mean_delta):
            bwt_per_task.append(mean_delta)

    # Average across tasks that had valid BWT data
    bwt = float(np.nanmean(bwt_per_task)) if bwt_per_task else 0.0

    # ---------- AUC ---------------------------------------------------------
    # For each task k, mean of valid accuracies from row k onwards

    auc_per_task = []
    for k in range(n_tasks):
        col_vals = conf_mat[k:, k]
        mean_auc = np.nanmean(col_vals)
        if not np.isnan(mean_auc):
            auc_per_task.append(mean_auc)

    auc = float(np.nanmean(auc_per_task)) if auc_per_task else 0.0

    return {"fwt": fwt, "bwt": bwt, "auc": auc}

def parse_args():
    parser = argparse.ArgumentParser(description="Summarize confusion matrix from evaluation results")
    parser.add_argument("--dir", type=str, default="experiments", help="Base directory to search for runs")
    parser.add_argument("--clean", action="store_true", help="Delete log files for incomplete/missing evaluations")
    return parser.parse_args()

def get_run_info(run_path):
    config_path = os.path.join(run_path, "config.json")
    info = {"n_memories": -1, "benchmark_name": "Unknown"}
    if not os.path.exists(config_path):
        return info
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            info["n_memories"] = config.get("lifelong", {}).get("n_memories", -1)
            info["benchmark_name"] = config.get("benchmark_name", "Unknown")
            return info
    except:
        return info

def get_confusion_matrix(run_path):
    # Pattern: {BENCHMARK}_{MODEL_NAME}_task_{TASK_ID}_success_rate_{RATE}.txt
    # We need to extract:
    # 1. Model Stage (from model name task{N}_model)
    # 2. Eval Task (from task_{M})
    # 3. Success Rate

    # Regex to capture:
    # ...task(\d+)_model.*_task_(\d+)_success_rate_([0-9.]+).txt
    # Note: Model name might vary but usually contains "taskN_model" for sequential/lifelong

    matrix = {} # {model_stage: {eval_task: rate}}
    files = list(Path(run_path).glob("*_success_rate_*.txt"))

    for file_path in files:
        filename = file_path.name
        # Match pattern
        # Example: LIBERO_GOAL_task0_model_task_0_success_rate_1.0.txt
        # Capture groups:
        # 1. model_stage (e.g., 0 from task0_model)
        # 2. eval_task (e.g., 0 from task_0)
        # 3. rate (e.g., 1.0)

        match = re.search(r"task(\d+)_model.*_task_(\d+)_success_rate_([0-9.]+).txt", filename)
        if match:
            model_stage = int(match.group(1))
            eval_task = int(match.group(2))
            rate = float(match.group(3))

            if model_stage not in matrix:
                matrix[model_stage] = {}
            matrix[model_stage][eval_task] = rate

    return matrix

def print_matrix(matrix, run_name, n_memories, benchmark_name):
    print(f"Run: {run_name}")
    print(f"Benchmark: {benchmark_name}")
    print(f"n_memories: {n_memories}")
    print(f"{'-'*60}")

    if not matrix:
        print("No evaluation results found.")
        return

    stages = sorted(matrix.keys())
    all_tasks = set()
    for stage in stages:
        all_tasks.update(matrix[stage].keys())

    if not all_tasks:
        print("No tasks found.")
        return

    tasks = sorted(list(all_tasks))

    # Header
    header = "Model \ Task"
    print(f"{header:<15} |", end="")
    for t in tasks:
        print(f" {t:<6}|", end="")
    print("\n" + "-" * (15 + 9 * len(tasks)))

    # Rows
    for stage in stages:
        print(f"{f'Task {stage}':<15} |", end="")
        for t in tasks:
            val = matrix[stage].get(t, "N/A")
            if isinstance(val, float):
                print(f" {val:.2f}  |", end="")
            else:
                print(f" {val:<6}|", end="")
        print("")
    print("")

    # Compute and print metrics
    # Dimension is max task ID + 1 to accommodate 0-indexed tasks
    n_tasks_dim = max(tasks) + 1 if tasks else 0
    if n_tasks_dim > 0:
        conf_mat = np.full((n_tasks_dim, n_tasks_dim), np.nan)
        for stage in matrix:
            for t, val in matrix[stage].items():
                # Only fill if indices are within bounds (should be if n_tasks_dim is max+1)
                if stage < n_tasks_dim and t < n_tasks_dim:
                    conf_mat[stage, t] = val

        metrics = compute_metrics(conf_mat)
        print(f"FWT: {metrics['fwt']:.4f}")
        print(f"BWT: {metrics['bwt']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")
        print("")

def main():
    args = parse_args()
    base_dir = args.dir

    runs = []

    # Use os.walk or find equivalent to locate run directories
    # A run directory is valid if it has config.json (or we can valid by directory name)
    for root, dirs, files in os.walk(base_dir, followlinks=True):
        if "config.json" in files:
            # Check if it looks like a run directory (optional, but good for filtering)
            if "run_" in os.path.basename(root) or True: # Accept any dir with config for now
                run_info = get_run_info(root)
                matrix = get_confusion_matrix(root)
                if matrix: # Only collect if there are results
                     runs.append({
                        "path": root,
                        "n_memories": run_info["n_memories"],
                        "benchmark_name": run_info["benchmark_name"],
                        "matrix": matrix
                    })

    # Sort by benchmark_name (asc) then n_memories (desc)
    runs.sort(key=lambda x: (x["benchmark_name"], -x["n_memories"]))

    current_benchmark = None


    for run in runs:
        if current_benchmark != run["benchmark_name"]:
            current_benchmark = run["benchmark_name"]
            print(f"\n{'='*30}")
            print(f" BENCHMARK: {current_benchmark}")
            print(f"{'='*30}\n")

        print_matrix(run["matrix"], run["path"], run["n_memories"], run["benchmark_name"])

        # Clean logs if requested
        if args.clean:
            # Re-scan for missing entries based on matrix gaps
            # matrix keys are model_stage
            # We assume tasks 0..model_stage should be evaluated for each model_stage

            # Find the max stage to know how far to check
            max_stage = 0
            if run["matrix"]:
                max_stage = max(run["matrix"].keys())

            # Iterate through all possible stages found in the matrix
            # If a stage is completely missing from matrix but folder exists, we might miss it here
            # But usually we have at least one result.
            # Using max_stage from matrix is a safe lower bound.
            # Ideally we should check all potential stages 0..9?
            # Let's check 0..9 as that's the Libero standard

            MAX_STAGES = 10

            cleaned_count = 0
            run_path = run["path"]

            for stage in range(MAX_STAGES):
                # Check if model exists for this stage, otherwise no point checking logs
                # Actually, eval script loops 0..MAX_LOAD_TASK(9).
                # It tries to pick a model. If no model, it skips.
                # If model exists, it proceeds to check log/success file.

                # We only want to delete logs where evaluation FAILED (i.e. log exists but success file doesn't).

                for task in range(stage + 1):
                    # Check if result exists in matrix
                    if stage in run["matrix"] and task in run["matrix"][stage]:
                        continue # Result exists, all good

                    # Result missing (N/A)
                    # Construct log file path
                    log_file = os.path.join(run_path, f"eval_load{stage}_task{task}.txt")

                    if os.path.exists(log_file):
                        # Verify NO success file exists (double check)
                        # The matrix construction already looked for success files.
                        # If it's not in matrix, success file is likely missing.
                        # But strictly speaking, `get_confusion_matrix` uses glob.

                        try:
                            print(f"[CLEAN] Deleting incomplete log: {log_file}")
                            os.remove(log_file)
                            cleaned_count += 1
                        except OSError as e:
                            print(f"[ERROR] Failed to delete {log_file}: {e}")

            if cleaned_count > 0:
                print(f"Deleted {cleaned_count} incomplete log files for run {run_path}")

if __name__ == "__main__":
    main()
