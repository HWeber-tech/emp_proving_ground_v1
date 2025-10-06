# CI Quality & Observability Weekly Status

This log captures the CI telemetry deltas that back the quality and
observability roadmap stream. Generate updates with:

```bash
python -m tools.telemetry.ci_digest --mode weekly --metrics tests/.telemetry/ci_metrics.json
```

Run the command after refreshing coverage reports and telemetry trendlines so the
roadmap, dashboards, and status updates stay in sync. Keep the most recent entry
at the top of the log.

## Entries

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
