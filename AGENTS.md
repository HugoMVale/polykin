# Agent guide (PolyKin)

PolyKin is an `src`-layout Python package. Use `uv`, keep changes scoped, and
match the existing patterns in nearby source and tests.

## Boundaries

- Source: `src/polykin/`; tests: `tests/`; docs: `docs/`; CI:
  `.github/workflows/`.
- Tests mirror package subdomains. Put tests for
  `src/polykin/<subdomain>/` under `tests/<subdomain>/`.
- Do not change `README.md` unless requested.
- Do not edit generated content under `docs/generated/`. If it is stale, report
  that; no generator is currently documented.
- Behavioral changes require focused tests.
- Do not add or update dependencies unless the task requires it. Preserve
  `uv.lock` for ordinary code changes.

## Environment

- Python requirement: `>=3.11` (see `pyproject.toml`).
- Set up the locked development environment from the repository root:

```powershell
uv sync --frozen
```

Install `uv` first only when it is unavailable.

## Development commands

Run the smallest relevant test set while iterating:

```powershell
uv run --frozen pytest tests/thermo
uv run --frozen pytest tests/thermo/test_flash.py -q
uv run --frozen pytest -k "flash" -q
```

Run the full suite for cross-package, shared-utility, public-signature, or
otherwise broad changes:

```powershell
uv run --frozen pytest
```

For code affected by Numba, also validate with JIT disabled:

```powershell
$env:NUMBA_DISABLE_JIT = "1"
uv run --frozen pytest <relevant-tests>
Remove-Item Env:NUMBA_DISABLE_JIT -ErrorAction SilentlyContinue
```

For Python changes, lint and check formatting only the changed files:

```powershell
uv run --frozen ruff check <changed-python-files>
uv run --frozen ruff format --check <changed-python-files>
```

Use the formatter without `--check` when changes are needed. Do not format the
entire repository for a scoped change.

Build documentation only when docs, docstrings, or public API documentation are
affected:

```powershell
uv sync --frozen --group docs
uv run --frozen mkdocs build -d _site
```

## Change workflow

1. Inspect nearby implementation and tests before editing.
2. Make the smallest complete change; avoid unrelated cleanup.
3. Add or update tests for new or changed behavior.
4. Run targeted tests and checks, then broaden validation only when the change's
   impact warrants it.
5. If a baseline test fails before your change is relevant, report it rather
   than repairing unrelated code without request.

## Conventions

- Use NumPy-style docstrings and keep lines at 90 characters or fewer.
- Annotate public functions and classes, including parameters and return types.
- Follow Ruff configuration in `pyproject.toml`; resolve new diagnostics without
  adding broad ignores.
- Prefer existing helpers and package-local patterns over duplicate logic.

## CI parity

CI tests Python 3.11-3.14 with highest dependency resolution and Python 3.11
with lowest-direct resolution. It runs tests with Numba both enabled and
disabled.

Reproduce dependency resolution with:

```powershell
uv sync --resolution lowest-direct
uv run --frozen pytest
```

This command can update `uv.lock`; restore the locked environment with
`uv sync --frozen` afterward and do not commit lock changes unless requested.
Use the CI workflow as the source of truth if these commands drift.
