# Drift Sentry response

The drift sentry runbook activates when Page–Hinkley or rolling variance
thresholds breach the guardrails defined in `src/operations/sensory_drift.py`
for sensory dimensions or the belief/regime metrics processed by
`src/operations/drift_sentry.py`. The readiness stack now raises a
`drift_sentry` component inside the operational readiness snapshot and
dispatches alert categories `sensory.drift*`, `understanding.drift_sentry`, and
`operational.drift_sentry` whenever the severity reaches WARN or ALERT. Use this
playbook to decide whether to pause promotions, adjust thresholds, or escalate
to adaptation leads.

## Detector signals

- **Sensory drift snapshot** – Published via `telemetry.sensory.drift` with
  per-dimension metadata for Page–Hinkley statistics, variance ratios, and the
  triggered detector list. The snapshot metadata also exposes
  `severity_counts` and `detectors` maps for dashboards and alert payloads.【F:src/operations/sensory_drift.py†L1-L287】
- **Understanding drift snapshot** – Published via
  `telemetry.understanding.drift_sentry` with belief/regime detector summaries,
  baseline/evaluation statistics, and severity counts so operational dashboards
  can render AlphaTrade drift posture alongside sensory telemetry.【F:src/operations/drift_sentry.py†L1-L279】
- **Operational readiness component** – `evaluate_operational_readiness` now adds
  a `drift_sentry` component whose summary surfaces the degraded dimensions and
  detector reasons, including belief/regime metrics and the linked runbook path
  for AlphaTrade responders. Guardrail alerts route through the existing
  operations email and webhook channels.【F:src/operations/operational_readiness.py†L83-L347】
- **Alert policy** – `default_alert_policy_config` includes explicit routing for
  `sensory.drift`, `sensory.drift.<dimension>`, `understanding.drift_sentry`,
  and `operational.drift_sentry` categories so notifications reach the ops
  email, Slack, webhook, SMS, and GitHub issue transports with suppression
  windows tuned for incident response drills.【F:src/operations/alerts.py†L703-L820】

## Triage checklist

1. **Confirm telemetry** – Pull the latest payload from the Predator summary or
   the runtime event bus. Validate that the drift snapshot timestamp is recent
   and that the `detectors` array pinpoints the triggering dimensions.
2. **Check readiness dashboard** – Open the operational readiness panel and
   confirm the `drift_sentry` component status. The metadata block includes
   detector stats and WARN/FAIL counts for direct export into incident updates.
3. **Correlate with understanding loop** – Inspect the understanding dashboard
   tile for the affected dimensions. If ledger approvals, belief/regime scoring,
   or fast-weight experiments align with the drift spike, coordinate with the
   adaptation lead before re-enabling promotions. The drift sentry snapshot
   includes baseline/evaluation stats to speed the comparison.
4. **Escalate if ALERT** – On ALERT severity the runbook requires notifying the
   incident response lead, pausing paper-trade promotions, and logging a
   decision diary entry referencing the drift snapshot ID. Confirm the latest
   TradingManager event shows the drift gate `force_paper` flag and that release
   execution metadata records the forced reason before attempting any live
   fills.【F:src/trading/gating/drift_sentry_gate.py†L321】【F:src/trading/execution/release_router.py†L175】【F:tests/trading/test_trading_manager_execution.py†L533】
5. **Adjust guardrails (optional)** – If drift is expected (e.g. scheduled data
   migrations), update the configuration pack to widen thresholds and link the
   change ticket in the decision diary for audit completeness.

## Recovery actions

- **Stabilise inputs** – Verify ingest health and Timescale retention telemetry
  to ensure upstream data did not regress. Use the ingest trend and Kafka
  readiness runbooks when OTA feeds are involved.
- **Recompute baselines** – Trigger a manual recalibration by running the
  sensory snapshot CLI or restarting the sensory organ with updated baseline
  windows. Document the new parameters in the context pack.
- **Validate clearance** – Once detectors revert to `normal`, capture the
  markdown export from `SensoryDriftSnapshot.to_markdown()` and attach it to the
  incident or governance report. Ensure the operational readiness dashboard
  reflects `drift_sentry: ok` before resuming promotions.

Keep this runbook aligned with the AlphaTrade understanding loop brief so drift
responses stay synchronised with governance, decision diary expectations, and
promotion gates.
