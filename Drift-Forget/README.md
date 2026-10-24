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

Cluster setup and job launchers come later (plan increment 16). Nothing in this study runs on the development
laptop.
