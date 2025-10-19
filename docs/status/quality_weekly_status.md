# CI Quality & Observability Weekly Status

This log captures the CI telemetry deltas that back the quality and
observability roadmap stream. Generate updates with:

```bash
python -m tools.telemetry.status_digest --mode weekly-status --metrics tests/.telemetry/ci_metrics.json
```

Add `--dashboard docs/status/ci_dashboard.json` (or another snapshot path) when
you want the observability panel summary in the digest. Run the command after
refreshing coverage reports and telemetry trendlines so the
roadmap, dashboards, and status updates stay in sync. Keep the most recent entry
at the top of the log.

## Entries

### 2025-10-07T05:57:01+00:00

- Coverage: 77.80% (Δ +1.40 pts) vs 2025-10-06T11:00:00+00:00 (source: coverage-reports/pytest-2025-10-07.xml)
- Lagging domains: 1 (Δ -1) — evolution 70.20% (threshold 80.0%) (source: coverage-reports/pytest-2025-10-07.xml)
- Remediation statuses:
  - coverage_threshold: 80
  - lagging_count: 1 (Δ -1.00)
  - overall_coverage: 77.8 (Δ +1.40)
  - worst_domain: evolution
- Note: Lagging domains: evolution (70.2%)
- Telemetry freshness (threshold 168h, evaluated 2025-10-07T05:57:01+00:00):
  - All telemetry fresh
  - Fresh feeds: coverage trend 0.5h; coverage domain trend 0.5h; formatter trend 1.4h; remediation trend 6.0h
- Alert response drill:
  - Label: ci-alert-2025-10-07 (drill)
- Acknowledged: 2025-10-07T12:03:00+00:00 (via slack, by oncall-analyst, evidence slack://incidents/ci-alert-2025-10-07)
- Resolved: 2025-10-07T12:18:30+00:00 (via github, by maintainer, evidence https://github.com/org/repo/issues/73)
  - Durations: MTTA 0:03:00; MTTR 0:18:30
  - Progress: CI status digests now render the alert-response telemetry into the dashboard table and weekly log, ignore `unknown` acknowledgement/resolution channels, and capture MTTA/MTTR with channel evidence so observability records the firing, acknowledgement, and recovery path.
- Evidence: tests/.telemetry/ci_metrics.json; coverage-reports/pytest-2025-10-07.xml

### 2025-10-06T18:13:30+00:00

- Coverage: 76.40% (Δ +1.30 pts) vs 2025-09-29T12:00:00+00:00 (source: coverage-reports/pytest-2025-10-06.xml)
- Lagging domains: 2 (Δ -1) — trading 79.60%, evolution 66.90% (threshold 80.0%) (source: coverage-reports/pytest-2025-10-06.xml)
- Remediation statuses:
  - coverage_threshold: 80
  - lagging_count: 2 (Δ -1.00)
  - overall_coverage: 76.4 (Δ +1.30)
  - worst_domain: evolution
- Note: Lagging domains: trading (79.6%), evolution (66.9%)
- Telemetry freshness (threshold 168h, evaluated 2025-10-06T18:13:30+00:00):
  - Stale feeds: formatter trend 849.2h
  - Fresh feeds: coverage trend 7.2h; coverage domain trend 7.2h; remediation trend 18.2h
- Evidence: tests/.telemetry/ci_metrics.json; coverage-reports/pytest-2025-10-06.xml

### 2025-10-06T11:00:00+00:00

- Coverage: 76.40% (Δ +1.30 pts) vs 2025-09-29T12:00:00+00:00 (source: coverage-reports/pytest-2025-10-06.xml)
- Lagging domains: 2 (Δ -1) — trading, evolution; worst: evolution (66.90%) (source: coverage-reports/pytest-2025-10-06.xml)
- Remediation statuses:
  - coverage_threshold: 80
  - lagging_count: 2 (Δ -1.00)
  - overall_coverage: 76.4 (Δ +1.30)
  - worst_domain: evolution
- Note: Lagging domains: trading (79.6%), evolution (66.9%)
- Evidence: coverage-reports/pytest-2025-10-06.xml
