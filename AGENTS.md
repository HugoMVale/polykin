# Agent guide (PolyKin)

This repository is a Python package (src-layout) built and tested via `uv`.
This document is written for automated coding agents and human maintainers who want
consistent, low-friction changes.

## Ground rules

- Keep changes minimal and scoped to the user request.
- Prefer repo-standard tooling and commands (see below) so CI matches local runs.
- Do not change `README.md` unless explicitly requested.
- Avoid editing generated content under `docs/generated/`.
- When making behavioral changes, add or update tests under `tests/`.

## Repo layout

- Source: `src/polykin/`
- Tests: `tests/` (mirrors package subdomains like `thermo/`, `kinetics/`, etc.)
- Docs (MkDocs Material): `docs/` (includes tutorial notebooks under `docs/tutorials/`)
- CI workflows: `.github/workflows/`

## Supported Python

- Python >= 3.10 (see `pyproject.toml`).

## Setup (recommended: uv)

CI uses `uv sync` with `uv.lock`.

### Windows (PowerShell)

```powershell
# From repo root
pip install uv
uv sync
```

### macOS/Linux (bash)

```bash
pip install uv
uv sync
```

## Run tests

### Full test suite

```powershell
uv run --frozen pytest
```

### Run a subset

```powershell
uv run --frozen pytest tests/thermo
uv run --frozen pytest -k "flash" -q
```

### Disable Numba JIT (useful for determinism / faster CI-style coverage)

CI runs a second pass with JIT disabled.

```powershell
$env:NUMBA_DISABLE_JIT = "1"
uv run --frozen pytest
```

To re-enable in the same shell:

```powershell
Remove-Item Env:NUMBA_DISABLE_JIT -ErrorAction SilentlyContinue
```

## Linting and formatting

Project config lives in `pyproject.toml`.

- Ruff (lint):

```powershell
uv run ruff check .
```

- Ruff (format):

```powershell
uv run ruff format .
```

If you change imports or add new modules, ensure Ruff import rules pass.

### (Optional) pre-commit hooks

This repo includes a `.pre-commit-config.yaml` that runs Ruff lint and format via `uv`.

```powershell
uv run --frozen pre-commit install
uv run --frozen pre-commit run --all-files
```

## Documentation build

Docs build in CI using the `docs` dependency group:

```powershell
uv sync --group docs
uv run mkdocs build -d _site
```

## Change workflow (agent-friendly)

1. Identify the smallest set of files needed.
2. Implement the change.
3. Run targeted tests first, then the full suite if the change is broad.
4. Run `ruff check .` and `ruff format .` if you changed Python files.
5. If docs are affected, run `mkdocs build`.

## Conventions

- Keep functions/classes type-annotated where practical.
- Docstrings follow NumPy-style conventions (Ruff pydocstyle is enabled).
- Respect the repo line length (90) when formatting.
- Avoid adding new heavy dependencies without a clear need.

## CI parity notes

CI runs:

- `uv sync` (both highest and lowest-direct resolution)
- `pytest` with and without Numba JIT
- docs build via `mkdocs build`

When you need to replicate a CI failure locally, prefer running the exact `uv run --frozen ...` command.
