# Personal ARENA workspace

This checkout uses a project-local uv environment, with Python **3.13.11** pinned
in `.python-version`, dependencies in `pyproject.toml`, and resolved versions in
`uv.lock`. The full curriculum environment targets **Linux x86_64 / CPython**
(including WSL), matching the installed CUDA 13 stack. It includes Jupyter and
the marimo tools used by the tensor lesson.

## Start working

```zsh
workon ARENA_materials
uv sync --locked
uv run jupyter lab
```

Open your saved notebooks under `chapter0_fundamentals/exercises/`:

- `part1_ray_tracing/0.1_Ray_Tracing_exercises.ipynb`
- `part2_cnns/0.2_CNNs_&_ResNets_exercises.ipynb`

Their setup cells find this checkout locally; Colab downloads and widget setup
only run in Colab. Your exercise implementations and saved outputs are retained.

For the marimo lesson:

```zsh
uv run marimo edit chapter0_fundamentals/pytorch_einops_60_min_lesson.py
```

`workon` activates this same `.venv`. After the one-time workon migration below,
`syncvenv ARENA_materials` and `workon --sync ARENA_materials` also use uv's
project synchronization. Use `uv add PACKAGE` for a persistent dependency and
`uv add --dev PACKAGE` for a development tool. Commit `pyproject.toml` and
`uv.lock` together after dependency changes.

## Jupyter kernel

Launch Jupyter from this environment. A kernel named **ARENA (uv)** is installed
inside `.venv`; its interpreter is this checkout's `.venv/bin/python`.
To register it again after recreating the environment:

```zsh
uv run python -m ipykernel install --sys-prefix \
  --name arena-env --display-name "ARENA (uv)"
```

For an editor, select `.venv/bin/python` as the Python interpreter / notebook
kernel. Check CUDA availability from your own terminal with:

```zsh
uv run python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

## Save progress and get updates

The local Git remotes are:

| Remote | Repository | Purpose |
| --- | --- | --- |
| `origin` | `yandahman/ARENA_materials` | Push your work to your fork |
| `upstream` | `ARENA-education/ARENA_materials` | Fetch curriculum updates |
| `solutions` | `yandahman/arena_ai` | Original saved solutions and history |

```zsh
git add path/to/notebook.ipynb
git commit -m "Complete ray tracing exercises"
git push
```

With your work committed, update from ARENA:

```zsh
git fetch upstream
git merge upstream/main
uv sync --locked
git push
```

Resolve overlapping notebook edits if Git reports conflicts. If upstream changes
`requirements.txt`, review those changes and update `pyproject.toml` and `uv.lock`
as needed; upstream's requirements file remains available for comparison. The
Git dependencies are pinned to commits in `pyproject.toml`. Liger-Kernel uses its
v0.8.5 source commit. Explicit pushes to `upstream` are disabled locally.

The fork is public. Keep credentials in an ignored `.env`, and keep datasets,
checkpoints, environment directories, and generated notebook caches out of Git.

## One-time workon migration on this machine

The existing project registration already points at this directory. Its old
requirements-mode marker must be moved aside once, so `syncvenv` uses
`pyproject.toml` and `uv.lock`:

```zsh
mv ~/.config/workon/requirements-projects/ARENA_materials \
  ~/coderz/.arena-migration-backup-20261024/workon-requirements-project
```

Run that command only if the marker still exists. The migration backup contains
the previous Git refs, configuration, Python package inventory, and local lesson
files. The `backup/pre-fork-20261024` branch also preserves the original checkout.
The saved solutions were imported with their original Git history; the
`arena_ai` repository and its separate `arena_colab` checkout remain available.
