# CI Health Snapshot

This dashboard summarizes the current state of the "CI" workflow and points to
where to find richer telemetry. Update the metrics after significant
infrastructure changes or once per sprint so stakeholders can confirm the
pipeline remains healthy at a glance.

## Latest pipeline status

| Signal | Value | Notes |
| --- | --- | --- |
| Last successful run | Refer to the GitHub Actions **CI** workflow page | Capture the run URL in team status updates. |
| Coverage (pytest `--cov`) | 76% (from [CI Baseline – 2025-09-16](../ci_baseline_report.md)) | Update when the coverage target moves. |
| Formatter rollout | Stage 4 enforces `src/sensory/`, all data foundation modules (`config/`, `ingest/`, `persist/`, `replay/`, `schemas.py`), and the `src/data_integration/`, `src/operational/`, and `src/performance/` directories; the briefing now tracks follow-up tooling work. | Guarded by `scripts/check_formatter_allowlist.py`; focus on formatting the helper scripts and planning the allowlist retirement. |
| Risk guardrail coverage | Drawdown throttling, Kelly sizing, and limit updates now exercised via `tests/current/test_risk_manager_impl.py` | Extend to FIX execution/risk integration paths next. |
| Execution engine coverage | Partial fills, retries, and reconciliation flows locked in by `tests/current/test_execution_engine.py` and exercised end-to-end via `tests/current/test_orchestration_execution_risk_integration.py`. | Track reconciliation snapshots in future regression runs. |
| Orchestration ⇄ risk ⇄ execution | `tests/current/test_orchestration_execution_risk_integration.py` runs the orchestrator stubs through risk sizing and the execution engine, emitting telemetry on the event bus. | Extend to include sensory fixtures once the WHY regression slice lands. |
| Data foundation config coverage | YAML loader fallbacks and overrides regression-tested via `tests/current/test_data_foundation_config_loading.py` | Keep expanding toward operational metrics and sensory signal hotspots. |
| Operational metrics coverage | FIX/WHY telemetry sanitization and guardrails enforced in `tests/current/test_operational_metrics_sanitization.py` | Fold orchestration smoke tests into the suite to cover adapter wiring. |
| Pytest flake telemetry | `tests/.telemetry/flake_runs.json` emitted each run (repository copy currently mirrors CI runs #482–#483 for historical context); summarise via `python tools/telemetry/summarize_flakes.py`. | Override with `PYTEST_FLAKE_LOG` or `--flake-log-file`; upload alongside `pytest.log` and follow the observability plan for drill cadence. |
| Open CI alerts | Check the automatically managed **CI failure alerts** issue | Created/closed by `.github/workflows/ci-failure-alerts.yml`. |

## Where to look when something fails

1. **CI failure alerts issue** – Any failing `CI` run adds a comment to the
   `CI failure alerts` issue with the run link, triggering actor, and failing
   branch. Resolve the failure and close the issue to clear the alert backlog.
2. **Pytest log artifact** – The `tests` job uploads `pytest.log` on every run
   (success or failure). Download it to inspect the full trace beyond the tail
   mirrored in the step summary.
3. **CI baseline report** – The baseline in [`docs/ci_baseline_report.md`](../ci_baseline_report.md)
   lists historical hotspots and still-relevant remediation tickets.

## Telemetry summary

| Metric | Value |
| --- | --- |
| Session start | 1726612800.0 |
| Session end | 1726612860.0 |
| Exit status | 1 (failure) |
| Recorded events | 2 |
| Failing tests | `tests/current/test_fix_manager_failures.py::test_recoverable_disconnect`, `tests/current/test_operational_metrics_logging.py::test_metric_payload_shape` |
| Recent runs | #482 failure, #483 success |

## Maintenance checklist

- Confirm the alert issue auto-closes after the next successful run.
- Refresh coverage and formatter metrics after sizable refactors.
- Add notes about recurring flakes directly to this page so trends remain
  discoverable without mining historical logs.
- Verify the flake telemetry JSON uploads with each run (or override the
  location locally to avoid committing artifacts).
