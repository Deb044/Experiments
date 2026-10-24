# Vendored upstream code

`continual-openpi/` and `continual-GR00T/` are exact, unedited copies of the Continual-VLAs research code at the
commits below, imported on 2026-10-24. Their `.git` directories are not included. All of our changes to them
are made in later commits. `git diff <import commit> -- Drift-Forget/continual-openpi Drift-Forget/continual-GR00T`
lists every change made after the import.

| Directory | Upstream | Commit | Commit date |
|---|---|---|---|
| `continual-openpi/` | https://github.com/Continual-VLAs/continual-openpi | `a3f8ca4c5abbf84248ce2046e248054fdaab5186` | 2026-03-03 |
| `continual-openpi/third_party/libero/` (submodule) | https://github.com/Continual-VLAs/libero | `1af3b848a5b5a1a403b1ff82aa6bbbb937d4fb7d` | pinned by openpi |
| `continual-openpi/third_party/aloha/` (submodule) | https://github.com/Physical-Intelligence/aloha | `d1dc83afd89ded4379851257fe5d85632d31d5ec` | pinned by openpi |
| `continual-GR00T/` | https://github.com/Continual-VLAs/continual-GR00T | `4bfd66ff44a7fbf17dffa86e7c26ae4e0a4565c0` | 2026-03-03 |

The paper is Liu et al., *Pretrained Vision-Language-Action Models are Surprisingly Resistant to Forgetting in
Continual Learning*, arXiv 2603.03818.

## How the copy was made

1. Each repository was cloned and checked out at the commit above. For openpi,
   `git submodule update --init --recursive` was run at the submodule commits openpi pins. openpi's
   `.gitmodules` gives the libero URL as SSH (`git@github.com:`); the clone rewrote it to https for that
   command only, with `git -c url.https://github.com/.insteadOf=git@github.com:`. No file was changed.
2. Only files tracked by git were exported (`git ls-files --recurse-submodules`). Untracked or build files
   were not copied. Submodule files sit at their usual paths, so the tree looks like a recursive checkout.

## Verification

- **Files:** 1479 in total. continual-openpi has 1295 (136 of its own, 1124 libero, 35 aloha);
  continual-GR00T has 184. The copy contains no files that are absent upstream.
- **Contents:** every file's `git hash-object --no-filters` hash equals the blob hash in the upstream index
  at the pinned commit, so the content is byte-identical.
- **Modes:** the executable bit matches upstream for every file. There are no symlinks.

Check it again from this directory, with the upstream clones in `<src>`:

```bash
(cd <src>/continual-openpi && git ls-files -s | grep -v '^160000') | while read -r mode sha _ path; do
  [ "$(git hash-object --no-filters "continual-openpi/$path")" = "$sha" ] || echo "MISMATCH $path"; done
# Repeat for third_party/libero, third_party/aloha (prefix the path) and continual-GR00T.
```

## Our edits to vendored files

Every change made after the import, in order. `git diff eb46140 -- continual-openpi continual-GR00T` shows them all.

| File | Change | Why |
|---|---|---|
| `continual-openpi/examples/libero/convert_libero_data_to_lerobot.py` | `REPO_NAME = "libero"` → `"huihanl/libero"` | The sequential training configs read `huihanl/libero-<suite>_no_noops/<task>` (`src/openpi/training/config.py`, `_LIBERO_OBJECT_SEQ` and others). The converter's `--output_dir` cannot fix the path, because it is never passed to `LeRobotDataset.create`; we leave that option unused rather than change it. |

## Git notes for this repository

- **Nested ignore rules.** The `.gitignore` files of the vendored repositories still apply inside
  Experiments. openpi's rules `assets/` and `data/` would hide 587 libero files: 585 under
  `third_party/libero/libero/libero/assets/` (simulator meshes and textures) and the 2 YAML files under
  `third_party/libero/libero/configs/data/`. Upstream these files live in a separate submodule repository,
  where openpi's rules do not reach. The import therefore stages the vendored tree with `git add -f`.
  Tracked files stay tracked, and the ignore rules still apply to new runtime files such as norm-stat
  `assets/` and `checkpoints/`.
- **Large file.** `continual-openpi/third_party/aloha/aloha2/workcell_v2/Aloha workcell v2.stl` is 97 MB.
  That is under GitHub's 100 MB limit, so pushes print a warning but succeed. The vendored code takes
  about 600 MB in total.
- **No lock file.** The fork does not commit openpi's `uv.lock`, so the authors' exact dependency versions are
  unknown. Only some pins in `continual-openpi/pyproject.toml` are exact (`jax[cuda12]==0.5.3`, `flax==0.10.2`);
  others, such as `torch>=2.7.0`, are open. We run `uv lock` at the first cluster setup, verify the
  environment, and commit the resulting `continual-openpi/uv.lock` as our own file, so every later install is
  identical.
- **Git LFS.** `continual-GR00T/.gitattributes` marks `media/robot-demo.gif` and `*.safetensors` for LFS.
  Upstream stores the GIF as a 132-byte LFS pointer, and that pointer is what was vendored (the real GIF
  is a README image, not needed to run anything). The repository contains no `.safetensors` files.
