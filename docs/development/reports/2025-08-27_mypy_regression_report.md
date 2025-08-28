# Batch10 typing regression report

Context: No-code-change regression summary comparing the last clean baseline to the latest snapshot. This document only reports and plans diagnosis; it does not propose code changes.

- Baseline vs current
  - Baseline clean snapshot: [mypy_summary_2025-08-27T09-43-50Z.txt](mypy_snapshots/mypy_summary_2025-08-27T09-43-50Z.txt:1) — 0 errors in 0 files
  - Latest snapshot: [mypy_summary_2025-08-27T15-21-18Z.txt](mypy_snapshots/mypy_summary_2025-08-27T15-21-18Z.txt:1) — Found 124 errors in 28 files
  - Full outputs:
    - Baseline full: [mypy_snapshot_2025-08-27T09-43-50Z.txt](mypy_snapshots/mypy_snapshot_2025-08-27T09-43-50Z.txt:1)
    - Current full: [mypy_snapshot_2025-08-27T15-21-18Z.txt](mypy_snapshots/mypy_snapshot_2025-08-27T15-21-18Z.txt:1)

- Offender ranking
  - Ranked offenders CSV (latest run): [mypy_ranked_offenders_2025-08-27T15-21-18Z.csv](mypy_snapshots/mypy_ranked_offenders_2025-08-27T15-21-18Z.csv:1)
  - Note: Use this CSV to identify top modules for focused review; this report intentionally avoids code changes and relies on the generated ranking.

- Recent cleanup references
  - Unused ignores removal artifacts (pre/post):
    - [mypy_unused_ignores_2025-08-27T14-35-19Z.txt](mypy_snapshots/mypy_unused_ignores_2025-08-27T14-35-19Z.txt:1)
    - [mypy_unused_ignores_postfix_2025-08-27T14-35-19Z.txt](mypy_snapshots/mypy_unused_ignores_postfix_2025-08-27T14-35-19Z.txt:1)
  - Stubs audit decisions and summary:
    - [stubs_audit_decisions_2025-08-27T14-47-39Z.csv](mypy_snapshots/stubs_audit_decisions_2025-08-27T14-47-39Z.csv:1)
    - [stubs_audit_post_summary_2025-08-27T14-47-39Z.txt](mypy_snapshots/stubs_audit_post_summary_2025-08-27T14-47-39Z.txt:1)

- Hypotheses for the delta (candidates to test; not assertions)
  - Environment mismatch (Python or mypy versions). Verify Python 3.11 and mypy version match CI.
  - Path/config drift (e.g., packages=src, namespace packages, or stubs path handling) relative to [mypy.ini](mypy.ini:1).
  - Stubs trimming side-effects: units removed (core, market_intelligence, sensory) may have increased analysis surface; prior audit suggested delta_total ≤ 0, but cross-check is warranted.
  - Cache effects (incremental or stale state). Ensure clean runs without stale caches.

- Diagnosis plan (zero-edit posture; local commands for investigation)
  - Capture environment for reproducibility:
    - python -V
    - mypy --version
    - pip freeze > [mypy_snapshots/env_2025-08-27T15-21-18Z.txt](mypy_snapshots/env_2025-08-27T15-21-18Z.txt:1)
  - Re-run with more detail (non-incremental, local):
    - mypy --config-file [mypy.ini](mypy.ini:1) --show-error-codes --no-incremental | tee [mypy_snapshots/mypy_snapshot_detailed_2025-08-27T15-21-18Z.txt](mypy_snapshots/mypy_snapshot_detailed_2025-08-27T15-21-18Z.txt:1)
  - Focused review:
    - Use [mypy_ranked_offenders_2025-08-27T15-21-18Z.csv](mypy_snapshots/mypy_ranked_offenders_2025-08-27T15-21-18Z.csv:1) to inspect top modules and dominant error categories first.
  - Sanity checks:
    - Confirm [mypy.ini](mypy.ini:1) is the config in use; verify packages=src and stubs search path unchanged.
    - Temporarily restore one stub unit (e.g., “core”) from quarantine for a local experiment to observe impact; do not commit changes.
    - Clear caches (e.g., remove .mypy_cache) and re-run to rule out cache artifacts.

- Next steps (maintaining zero-edit posture here)
  - If the delta is environmental: pin and document versions (Python, mypy, and key deps), then re-run CI to confirm resolution.
  - If specific modules dominate: prepare a diagnostics-only plan listing the top ~10 modules and primary error categories to feed a future fix batch.

Artifacts referenced in this report:
- Baseline summary: [mypy_summary_2025-08-27T09-43-50Z.txt](mypy_snapshots/mypy_summary_2025-08-27T09-43-50Z.txt:1)
- Current summary: [mypy_summary_2025-08-27T15-21-18Z.txt](mypy_snapshots/mypy_summary_2025-08-27T15-21-18Z.txt:1)
- Baseline full: [mypy_snapshot_2025-08-27T09-43-50Z.txt](mypy_snapshots/mypy_snapshot_2025-08-27T09-43-50Z.txt:1)
- Current full: [mypy_snapshot_2025-08-27T15-21-18Z.txt](mypy_snapshots/mypy_snapshot_2025-08-27T15-21-18Z.txt:1)
- Ranked offenders: [mypy_ranked_offenders_2025-08-27T15-21-18Z.csv](mypy_snapshots/mypy_ranked_offenders_2025-08-27T15-21-18Z.csv:1)
- Unused ignores (pre/post): [mypy_unused_ignores_2025-08-27T14-35-19Z.txt](mypy_snapshots/mypy_unused_ignores_2025-08-27T14-35-19Z.txt:1), [mypy_unused_ignores_postfix_2025-08-27T14-35-19Z.txt](mypy_snapshots/mypy_unused_ignores_postfix_2025-08-27T14-35-19Z.txt:1)
- Stubs audit: [stubs_audit_decisions_2025-08-27T14-47-39Z.csv](mypy_snapshots/stubs_audit_decisions_2025-08-27T14-47-39Z.csv:1), [stubs_audit_post_summary_2025-08-27T14-47-39Z.txt](mypy_snapshots/stubs_audit_post_summary_2025-08-27T14-47-39Z.txt:1)
- Config: [mypy.ini](mypy.ini:1)