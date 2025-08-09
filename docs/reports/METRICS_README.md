### EMP FIX Metrics Quickstart

- The FIX manager now exposes Prometheus metrics on `/metrics` (default port 8081). Override with `EMP_METRICS_PORT`.
- Metrics include:
  - Counters: `fix_messages_total{session,msg_type}`, `fix_reconnect_attempts_total{session,outcome}`, `fix_business_rejects_total{ref_msg_type}`
  - Gauges: `fix_session_connected{session}`, `fix_md_staleness_seconds{symbol}`
  - Histograms: `fix_exec_report_latency_seconds`, `fix_cancel_latency_seconds`

Local run:
1. `docker compose up -d prometheus grafana`
2. Start EMP app (metrics auto-start). Visit `http://localhost:8081/metrics` to confirm.
3. In Grafana (`http://localhost:3000`), import `docs/reports/EMP_FIX_Grafana_Dashboard.json`.


