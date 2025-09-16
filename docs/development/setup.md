# Development Setup (Stub)

- Python 3.11
- pip install -r requirements/base.txt  # runtime stack
- pip install -r requirements/dev.txt   # tooling + tests
- Configure .env with FIX credentials (OpenAPI disabled)

The development manifest pins mypy, Ruff, Black, pytest, coverage, pre-commit, import-linter, and the
core type stub packages so local runs mirror CI. Use `pip install -r requirements/dev.txt` after pulling
to pick up any version bumps.

## Scientific stack checklist

Keep the runtime guard in sync with the dependency manifests. The table below mirrors the
hard limits enforced by `src/system/requirements_check.py`:

| Library | Minimum version | Notes |
| --- | --- | --- |
| numpy | 1.26.0 | Base numerical stack used across ingest, analytics, and trading loops. |
| pandas | 1.5.0 | Dataframe operations throughout sensory and validation modules depend on new indexing fixes. |
| scipy | 1.11.0 | Signal processing helpers rely on optimizations introduced in the 1.11 series. |

Run the pre-flight validator before major deployments:

```bash
python -m src.system.requirements_check
```

The command exits non-zero if any library is missing or below the documented floor and prints the
detected versions so upgrades can be recorded in `requirements/base.txt`.
 codex/assess-technical-debt-in-codebase

## Formatting expectations

Ruff owns both linting and formatting. Follow the staged rollout captured in
[`formatter_rollout.md`](formatter_rollout.md):

1. Run `ruff format` on any file that already appears in the
   `config/formatter/ruff_format_allowlist.txt` allowlist before committing.
2. When you finish normalizing a new directory, add it to the allowlist and run
   `python scripts/check_formatter_allowlist.py` locally to mirror the CI guard.
3. Use `ruff check --select I` to tidy import ordering in directories that have
   not yet been fully formatted so upcoming rollouts generate cleaner diffs.

CI fails if any allowlisted path diverges from the formatter output, so keep the
allowlist and your local environment in sync.
 main
