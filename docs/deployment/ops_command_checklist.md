# Ops Command daily checklist

This checklist operationalises the "Ops Command" expectations from the EMP
Encyclopedia and the High-Impact roadmap. Run it at the start and end of every
trading day to ensure institutional hygiene.

## Morning activation

- [ ] Review overnight ingest telemetry via `ProfessionalPredatorApp.summary()`.
- [ ] Confirm Timescale, Redis, and Kafka health using the observability
      dashboard (`operations.observability_dashboard.build_observability_dashboard`).
- [ ] Execute `scripts/run_disaster_recovery_drill.py --output artifacts/ops/drill.md`
      if the last drill is older than 7 days; attach the artefact to the ops log.
- [ ] Run `python -m tools.security.pip_audit_runner --format markdown --output artifacts/security/pip_audit.md`
      and file tickets for outstanding vulnerabilities.
- [ ] Verify compliance monitors (`compliance.trade_compliance.TradeComplianceMonitor`,
      `compliance.kyc.KycAmlMonitor`) report healthy status in the runtime summary.
- [ ] Announce activation in the #ops channel with a link to the latest
      `docs/status/high_impact_roadmap.md` snapshot.

## Intraday cadence

- [ ] Monitor order lifecycle metrics (`trading.order_management.lifecycle_processor.OrderLifecycleProcessor`)
      for anomalies; escalate when latency exceeds thresholds.
- [ ] Ensure risk guardrails remain within bounds by checking the automated
      risk report generated via `scripts/generate_risk_report.py`.
- [ ] Record any incident candidates in `docs/runbooks/templates/incident_postmortem_template.md`.

## Evening shutdown

- [ ] Capture closing PnL and exposure snapshots from the dashboard introduced in
      Workstream 1C; archive under `artifacts/ops/pnl/YYYY-MM-DD.md`.
- [ ] Trigger infrastructure drift detection (`make infra-plan ENV=staging`) and
      store the plan output with the deployment artefacts.
- [ ] Rotate application logs by invoking the observability pipeline configured in
      `config/observability/` and verify archive delivery.
- [ ] Log the day’s summary (ingest status, risk posture, incidents, actions) in
      `docs/operations/daily_logs/YYYY-MM-DD.md`.

Maintaining this cadence keeps the roadmap evidence fresh and ensures operators
inherit the same institutional muscle memory described in the encyclopedia.
