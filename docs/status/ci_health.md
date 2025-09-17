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
| Formatter rollout | Stages 0–2 complete, Stage 3 locked (`tests/current/`, `src/system/`, `src/core/configuration.py`, `src/trading/execution/`, `src/trading/models/`, and the full `src/sensory/organs/dimensions/` package, including `__init__.py`, `utils.py`, `what_organ.py`, `when_organ.py`, `why_organ.py`, and prior organs), Stage 4 now enforces the entire `src/sensory/` tree plus `src/data_foundation/config/` via the allowlist | Guarded by `scripts/check_formatter_allowlist.py`; dry-run `src/data_foundation/ingest/` and `src/data_foundation/persist/` next and keep pytest green after each slice. |
| Risk guardrail coverage | Portfolio cap clamps and USD beta orientation now exercised via `tests/current/test_portfolio_risk_caps.py` | Extend to FIX execution/risk integration paths next. |
| Data foundation config coverage | YAML loader fallbacks and overrides regression-tested via `tests/current/test_data_foundation_config_loading.py` | Keep expanding toward operational metrics and sensory signal hotspots. |
| Operational metrics coverage | FIX/WHY telemetry sanitization and guardrails enforced in `tests/current/test_operational_metrics_sanitization.py` | Fold orchestration smoke tests into the suite to cover adapter wiring. |
| Pytest flake telemetry | `tests/.telemetry/flake_runs.json` emitted each run | Override with `PYTEST_FLAKE_LOG` or `--flake-log-file`; upload alongside `pytest.log`. |
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

## Maintenance checklist

- Confirm the alert issue auto-closes after the next successful run.
- Refresh coverage and formatter metrics after sizable refactors.
- Add notes about recurring flakes directly to this page so trends remain
  discoverable without mining historical logs.
- Verify the flake telemetry JSON uploads with each run (or override the
  location locally to avoid committing artifacts).
