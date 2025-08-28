Title: Fix A — environment regression remediation (py311, mypy 1.17.1)

Summary:
- Stabilizes typing environment to Python 3.11 and mypy==1.17.1; adds deterministic Docker runner.
- Applies minimal, behavior-preserving type fixes across high-impact modules (≤5 edits/file).
- Updates report with auditable snapshot history and deltas.

Snapshot history:
- Baseline (py311): “Found 125 errors in 31 files (checked 343 source files)”
- Fix A pass 1 (2025-08-28T03-03-45Z): “Found 120 errors in 31 files (checked 343 source files)”
- Fix A pass 2 partial (2025-08-28T08-56-04Z): “Found 92 errors in 28 files (checked 343 source files)”
- Interim regression (2025-08-28T09-10-50Z): “Found 99 errors in 28 files (checked 343 source files)”
- Fix A pass 2.1 (2025-08-28T11-31-46Z): “Found 79 errors in 27 files (checked 343 source files)”

Latest artifacts:
- Summary: [mypy_summary_py311_2025-08-28T11-31-46Z.txt](mypy_snapshots/mypy_summary_py311_2025-08-28T11-31-46Z.txt:1)
- Ranked: [mypy_ranked_offenders_py311_2025-08-28T11-31-46Z.csv](mypy_snapshots/mypy_ranked_offenders_py311_2025-08-28T11-31-46Z.csv:1)
- Env: [env_py311_2025-08-28T11-31-46Z.txt](mypy_snapshots/env_py311_2025-08-28T11-31-46Z.txt:1)

Implementation notes:
- Patterns: [typing.TYPE_CHECKING](python.TYPE_CHECKING:1), [TypeAlias](python.TypeAlias:1), [typing.cast()](python.cast():1), dict normalization to [dict[str, object]](python.dict():1), Mapping inputs.
- Import hygiene maintained; no runtime behavior changes.
- Deterministic mypy snapshots with MYPY_CACHE_DIR and --no-incremental in runner [scripts/mypy_py311.sh](scripts/mypy_py311.sh:1).

CI expectations:
- PR typing workflow [.github/workflows/typing.yml](.github/workflows/typing.yml:1) runs strict-on-touch and full-repo mypy (py311).
- Nightly typing workflow [.github/workflows/typing-nightly.yml](.github/workflows/typing-nightly.yml:1) enabled.
- This PR should reduce totals; gates fail only on newly introduced errors.

After merge checklist:
- Tag release “fixA-py311-typing-stabilization”.
- Enforce protected checks: typing, static-analysis, tests.
- Optional: schedule micro-pass for remaining hotspots.
