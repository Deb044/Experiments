#!/bin/bash
# Run on a login node: one preparation job, then one training job per task (or per task and
# seed) on separate allocations, then the remaining stages. Every job is a submit.sh submission.
set -euo pipefail
level=${1:-medium}
split=${2:-task}
case "$split" in task|seed) ;; *) echo 'Expected task or seed as the second argument' >&2; exit 2;; esac
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=${VLA_REPO_ROOT:-$(cd -- "$script_dir/.." && pwd)}
python=${VLA_PYTHON:-$repo/.venv/bin/python}
config=${VLA_CONFIG:-$repo/configs/pilot.json}
if [[ ! -x $python ]]; then echo "Missing environment Python: $python (set VLA_PYTHON)" >&2; exit 2; fi
units=$("$python" -c '
import json, sys
config = json.load(open(sys.argv[1]))
for task in config["tasks"]:
    for seed in config["seeds"] if sys.argv[2] == "seed" else [""]:
        print(task["id"], seed)' "$config" "$split")
submit() {
    local reply
    reply=$(bash "$script_dir/submit.sh" "$level" "$@")
    echo "$reply" >&2
    if [[ ! $reply =~ ([0-9]+)[^0-9]*$ ]]; then echo 'Could not read the job id from sbatch' >&2; exit 2; fi
    echo "${BASH_REMATCH[1]}"
}
prepared=$(submit prepare)
training=
while read -r task seed; do
    arguments=(train --task "$task")
    if [[ -n $seed ]]; then arguments+=(--seed "$seed"); fi
    training+=:$(VLA_DEPENDENCY=afterok:$prepared submit "${arguments[@]}")
done <<< "$units"
# afterany: if a training job runs out of time, the last job resumes its remaining runs.
VLA_DEPENDENCY=afterany$training submit run > /dev/null
