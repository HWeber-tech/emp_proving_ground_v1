# Stage 4 Formatting Briefing – Operational & Performance slices

This briefing aligns owners, reviewers, and validation steps for the remaining
Stage 4 formatter rollout across the operational and performance directories.
Share it with anyone preparing PRs so we avoid conflicting freeze windows and
keep CI green throughout the rollout.

> **Status update (2025-10-02):** Stage 4 wrapped and the repository now enforces
> `ruff format --check .` globally. The notes below capture the original rollout
> plan for historical reference; no further allowlist updates are required.

## Scope

- `src/operational/metrics.py`
- `src/operational/metrics_registry.py`
- `src/operational/state_store/`
- ~~`src/operational/event_bus.py`~~ (module deleted; canonical implementation lives in `src/core/event_bus.py`)
- `src/performance/vectorized_indicators.py`
- `src/performance/__init__.py`

## Owners and reviewers

| Slice | Primary owner | Reviewer rota | Notes |
| --- | --- | --- | --- |
| Operational metrics | Platform (A. Patel) | Trading (J. McKay), Observability (L. Chen) | Pair the formatting diff with a quick smoke run of `tests/current/test_operational_metrics_sanitization.py`. |
| Operational async core | Platform (M. Rivera) | Orchestration (H. Singh), Reliability (C. Gomez) | Schedule during a low-traffic window; async helpers are imported widely and can introduce flakes. |
| Performance analytics | Performance (S. Ibarra) | Data Foundation (E. Rossi) | Confirm with Market Intelligence before landing changes; they import the module for dashboards. |

## Freeze windows

| Window | Timeframe | Reason |
| --- | --- | --- |
| Metrics slice | Tuesdays 15:00–18:00 UTC | Avoids overlapping with observability deploys and CI config updates. |
| Async slice | Thursdays 13:00–16:00 UTC | Leaves room to roll back before Friday change freeze. |
| Performance slice | Wednesdays 10:00–12:00 UTC | Aligns with performance modeling sync to coordinate downstream consumers. |

## Validation checklist *(historical)*

*The steps below describe the retired allowlist workflow and remain for audit
purposes only.*

1. Run `ruff format <target>` and confirm the diff is mechanical.
2. Execute targeted tests:
   - `pytest tests/current/test_operational_metrics_sanitization.py`
   - `pytest tests/current/test_orchestration_compose.py`
   - `pytest tests/current/test_performance_metrics_module.py -k vectorized`
     (covers the vectorized indicator helpers)
3. Update `config/formatter/ruff_format_allowlist.txt` in the same commit as the
   formatting changes.
4. Record the slice completion in [`docs/development/formatter_rollout.md`](../development/formatter_rollout.md)
   and link back to the PR.
5. Refresh the formatter status row in [`docs/status/ci_health.md`](ci_health.md)
   once the PR merges.

## Communication

- Post the planned freeze window in `#eng-ops` at least 24 hours in advance.
- Flag any manual cleanups (beyond `ruff format`) in the PR description and open
  follow-up tickets if they are unrelated to formatting.
- Drop a short summary into the Friday modernization sync so downstream teams
  know which directories now enforce `ruff format`.
