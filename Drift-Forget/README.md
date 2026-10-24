# Drift-Forget

How do weight drift and feature drift across continual-learning checkpoints relate to forgetting in pretrained
vision-language-action models? This study builds on Liu et al., *Pretrained Vision-Language-Action Models are
Surprisingly Resistant to Forgetting in Continual Learning* (arXiv 2603.03818).

| File or folder | Contents |
|---|---|
| [EXPERIMENTS.md](EXPERIMENTS.md) | Protocol: hypotheses, conditions, outcomes, analysis plan |
| [VENDORED.md](VENDORED.md) | Upstream commits of the vendored code and how the copy was verified |
| `continual-openpi/` | The authors' π0 code (vendored; our edits are diffs against import commit `eb46140`) |
| `continual-GR00T/` | The authors' GR00T code (vendored; phase 2) |
| `src/drift_forget/` | Our analysis package |

## Environments

There are two separate environments, both defined by the vendored code:

1. **Training and analysis.** Python 3.11 with JAX, created by `uv` from `continual-openpi/pyproject.toml`.
   The lock file `uv.lock` is created at the first cluster setup and committed (see VENDORED.md).
2. **LIBERO simulator client.** Python 3.8 venv from `continual-openpi/examples/libero/requirements.txt`, as
   described in `continual-openpi/examples/libero/README.md`.

`drift_forget` is never installed. Doing so would make `uv sync` remove it. Instead it runs in environment 1
through `PYTHONPATH`:

```bash
cd Drift-Forget
PYTHONPATH="$PWD/src" uv run --project continual-openpi python -m drift_forget ...
```

## Data

The training data is the OpenVLA-filtered LIBERO set (`no_noops`) in RLDS format, from the Hugging Face dataset
`openvla/modified_libero_rlds`. It is downloaded and converted on the cluster, into scratch. Only LIBERO-Object
is needed, plus LIBERO-10 if Tier 4 runs.

```bash
export DF_SCRATCH=/scratch/$USER/VLA/Drift-Forget
export HF_LEROBOT_HOME=$DF_SCRATCH/data/lerobot      # converted datasets; training reads them here too
cd Drift-Forget/continual-openpi

# 1. Download the raw RLDS data into $DF_SCRATCH/data/rlds (exact command fixed at cluster setup).
# 2. Convert. --idx picks the suite: 0 = libero_10, 1 = goal, 2 = object, 3 = spatial.
#    TF_VERSION and TFDS_VERSION are chosen at cluster setup (must support numpy<2 and Python 3.11)
#    and recorded here.
uv run --with tensorflow==$TF_VERSION --with tensorflow-datasets==$TFDS_VERSION \
  examples/libero/convert_libero_data_to_lerobot.py --data_dir $DF_SCRATCH/data/rlds --idx 2
# Do not pass --output_dir: the converter ignores it (see VENDORED.md). Output goes to
# $HF_LEROBOT_HOME/huihanl/libero-libero_object_no_noops/<task_slug>/, one dataset per task.

# 3. Per-task normalization statistics, written to ./assets/<asset name>/huihanl/libero-libero_object_no_noops/<task_slug>/
uv run scripts/compute_norm_stats.py --config-name pi0_libero_low_mem_finetune-libero_object_sequential
```

The norm-stat files stay out of git: openpi's `.gitignore` ignores `assets/`. After they are computed, the
SHA-256 of every file goes into a manifest, and training, evaluation and analysis jobs refuse to run if a file
no longer matches. The manifest and the check are written in a later increment.

Checked against the code:
- The fork trains Object tasks in the order `_LIBERO_OBJECT_SEQ` (`src/openpi/training/config.py`).
- That is the order in the paper's Appendix B.2.
- It is also the order LIBERO's simulator uses for task indices (`third_party/libero/.../libero_suite_task_map.py`,
  default `task_order_index=0`).
- LIBERO-10 matches too, once LIBERO's scene prefixes are dropped from the task names.

So training task k and evaluation task k are the same task.

Cluster setup and job launchers come later (plan increment 16). Nothing in this study runs on the development
laptop.
