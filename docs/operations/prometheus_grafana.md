# EMP Prometheus & Grafana Monitoring

This setup provisions in-repo infrastructure-as-code for Prometheus alerts and Grafana dashboards so operations teams can run the roadmap-observable drills without bespoke configuration.

## Components

- `config/prometheus/prometheus.yml` ships with the EMP scrape targets and now loads `emp_rules.yml` for SLO alerting.
- `config/prometheus/emp_rules.yml` defines warning and critical alerts for understanding-loop latency, drift-alert freshness, and replay determinism using the SLO status gauges exported by `src/operations/slo.py`.
- `config/grafana/datasources/prometheus.yml` provisions a Grafana data source named `EMP Prometheus` that points at the docker compose Prometheus container.
- `config/grafana/dashboards/json/emp_observability.json` renders the core SLO metrics (latency, drift, replay) and their status gauges so operators can see both raw trends and breach severity.

## Local drill

1. `docker compose up -d prometheus grafana`
2. Start the EMP runtime (or fixtures) so `start_metrics_server()` exposes `/metrics` on the configured port.
3. Visit Grafana at `http://localhost:3000` (credentials admin/admin). The "EMP Observability SLOs" dashboard is auto-imported.
4. Prometheus at `http://localhost:9090` now evaluates the alert rules. Use **Alerts** in the UI to inspect breach simulations.

Alerts fire when the status gauges exported by the SLO helpers report warning (`1`) or breach (`2`). Adjust thresholds in the `LoopLatencyProbe`, `DriftAlertFreshnessProbe`, or `ReplayDeterminismProbe` dataclasses and the rules automatically follow.

## Validation

- `tests/config/test_prometheus_monitoring.py` ensures the rule file stays wired into Prometheus and that all expected SLO alerts are present.
- `tests/config/test_grafana_dashboard.py` guards the Grafana dashboard structure, data source UID, and metric queries so automated formatting does not drop the key panels.

This completes the roadmap checkpoint "Prometheus/Grafana (or cloud) monitoring; SLO alerting as code" without mutating the roadmap document itself.
