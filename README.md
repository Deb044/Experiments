# Experiments

A collection of research experiments. Each experiment lives in its own folder,
with its code, configuration, tests, documentation and supporting files together.
`VLA_WS` is the first experiment; future experiments will have separate folders.

## Current experiment

| Folder | Research question | Instructions |
|---|---|---|
| [VLA_WS](VLA_WS/) | Do deletion requests produce predictable changes in VLA weight space? | [Experiment README](VLA_WS/README.md) · [Cluster guide](VLA_WS/cluster/README.md) |

```text
Experiments/
├── README.md
└── VLA_WS/
    ├── README.md
    ├── pyproject.toml
    ├── src/
    ├── configs/
    ├── tests/
    ├── cluster/
    ├── .github/
    ├── .gitattributes
    └── .gitignore
```

Generated artifacts, caches and local environments also belong inside the
experiment directory and remain untracked. Git's own `.git` metadata remains
at the repository root.

## Running VLA_WS

Run installation, tests and experiment commands **from `VLA_WS`**:

```bash
cd "$HOME/VLA/Experiments/VLA_WS"
python -m venv .venv
source .venv/bin/activate
# Install the appropriate CUDA/research dependencies from the experiment README.
# For a CPU-only plumbing check:
pip install -e '.[test]'
pytest -q
vla-weightspace --config configs/smoke.json run
```

Recreate an existing virtual environment after moving it, and reinstall the
editable package from this directory; environments can contain absolute paths.
Existing local manifests/checkpoints can also contain the old absolute output
path. Keep those artifacts for reference and prepare a new output directory
after relocation rather than editing their scientific provenance.

## GPU cluster

The launchers discover the experiment root when submitted through `submit.sh`.
Use the environment installed inside `VLA_WS`, or set `VLA_PYTHON` to your
research environment's absolute Python path:

```bash
cd "$HOME/VLA/Experiments/VLA_WS"
bash cluster/submit.sh medium benchmark
```

The existing launchers still write logs to `$HOME/VLA/Experiments/logs` and large
outputs to `/scratch/$USER/VLA`. To keep generated home-directory logs inside the
experiment folder as well, set the launcher's existing override:

```bash
export VLA_LOG_ROOT="$HOME/VLA/Experiments/VLA_WS/logs"
```

In the original cluster guide, replace the old repository working directory
`$HOME/VLA/Experiments` with `$HOME/VLA/Experiments/VLA_WS` for setup and commands.
Use `cluster/submit.sh` rather than directly submitting a job template: its
automatic root discovery handles the new layout. No existing experiment code,
configuration or launcher contents were changed during this move.

## Workflow location

The original workflow is preserved at `VLA_WS/.github/workflows/tests.yml`.
GitHub only discovers workflows at the repository-root `.github/workflows`, so
automatic CI is inactive with this requested root layout. The unchanged workflow
remains available as a reference; run the documented tests locally or on the cluster.
