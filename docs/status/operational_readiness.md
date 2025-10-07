# Operational readiness telemetry

The operational readiness snapshot fuses system validation, incident response,
and SLO telemetry into a single payload that dashboards can ingest directly.
Recent hardening work adds structured metadata so downstream consumers can track
how many components are warning or failing without re-implementing severity
logic.

## Snapshot contract

`evaluate_operational_readiness` now enriches the snapshot metadata with two
fields:

- `status_breakdown` – counts of components per readiness status, expressed as a
  mapping such as `{ "warn": 2, "fail": 1 }`.
- `component_statuses` – a mapping from component name to its status value so
  dashboards can highlight degraded services directly (for example,
  `{ "incident_response": "fail" }`).
- `issue_counts` – rolled-up WARN/FAIL totals sourced from each component’s
  issue metadata so responders can see the breadth of active incidents without
  drilling into every payload.
- `component_issue_details` – per-component issue catalogs (counts, highest
  severity, structured entries) that observability dashboards can surface when
  a component reports WARN/FAIL conditions.【F:src/operations/operational_readiness.py†L113-L184】
- `drift_sentry` component – aggregates the Page–Hinkley and variance detector
  outputs from both the sensory drift and understanding drift sentry snapshots,
  surfaces degraded dimensions/metrics with detector reasons, attaches the
  shared runbook path, and feeds WARN/FAIL counts plus rich issue details into
  the readiness metadata and alert contexts.【F:src/operations/operational_readiness.py†L83-L347】【F:src/operations/drift_sentry.py†L1-L279】
- `incident_response.reliability_metrics` – captures MTTA/MTTR samples,
  acknowledgement/resolution counts, and metric staleness so dashboards and
  alert policies can grade acknowledgement/resolution cadence without recomputing
  timelines.【F:src/operations/incident_response.py†L295-L640】【F:tests/operations/test_incident_response.py†L195-L248】
- `system_validation.reliability` – summarises recent validation runs with fail
  streaks, stale-age detection, and rolling success-rate thresholds, exposing
  structured issues plus alert contexts for reliability regressions.【F:src/operations/system_validation.py†L245-L618】【F:tests/operations/test_system_validation.py†L201-L342】

Both fields accompany the existing `component_count` and remain present in the
snapshot dictionary returned by `OperationalReadinessSnapshot.as_dict()` and the
payload attached to derived alert events. The regression suite
(`tests/operations/test_operational_readiness.py`) locks the contract so CI fails
if the breakdown or component mapping drift.

## Using the metadata

- Dashboards can render stacked bar charts or summary chips by reading the
  `status_breakdown` map instead of recomputing severities.
- Incident response drill-downs reuse `component_issue_details` to highlight
  missing runbooks, roster gaps, and backlog breaches with their recorded
  severities.
- Alerting policies receive the enriched metadata via the alert context, making
  it trivial to route failures for specific components.
- Additional metadata can still be supplied via the `metadata` parameter when
  evaluating readiness; custom keys override defaults if needed.

Combine the snapshot with the observability dashboard output to give operators a
clear view of which operational slices need attention each run.

## Alert routing and publishing

`derive_operational_alerts` translates the aggregated snapshot into component and
overall alert events with deterministic severity mapping, optional suppression of
the overall status, and tag inheritance so routing policies can dispatch the same
evidence that dashboards render.【F:src/operations/operational_readiness.py†L205-L292】【F:tests/operations/test_operational_readiness.py†L86-L183】
`route_operational_readiness_alerts` pipes those events through any
`AlertManager`, returning dispatch results for guardrail assertions and ensuring
context packs capture which transports emitted notifications.【F:src/operations/operational_readiness.py†L294-L317】【F:tests/operations/test_operational_readiness.py†L147-L184】
The default alert policy now includes dedicated routes for incident response
MTTA/MTTR breaches and system validation reliability regressions so SMS/webhook
escalations fire when acknowledgement cadence or validation quality drifts from
institutional targets.【F:src/operations/alerts.py†L158-L226】【F:tests/operations/test_alerts.py†L200-L240】

Incident response and system validation alert helpers now expose gate events via
`include_gate_event=True`, propagating gate metadata (blocking reasons, decision
status, and guardrail configuration) into alert contexts so routing policies can
escalate deployment blockers without re-running gate evaluation logic.【F:src/operations/incident_response.py†L826-L905】【F:tests/operations/test_incident_response.py†L306-L352】【F:src/operations/system_validation.py†L660-L742】【F:tests/operations/test_system_validation.py†L249-L309】

`publish_operational_readiness_snapshot` now reuses the shared failover helper to
log runtime bus failures, raise typed errors on unexpected exceptions, and fall
back to the global bus when necessary, with regression coverage locking the
runtime/global escalation paths.【F:src/operations/operational_readiness.py†L319-L373】【F:tests/operations/test_operational_readiness.py†L186-L221】

## Incident response issue catalog

Incident response telemetry now records a structured issue catalog alongside the
string summaries. `IncidentResponseSnapshot.metadata` exposes `issue_details`,
`issue_counts`, `issue_category_severity`, and the `highest_issue_severity`
driver so dashboards and alert policies can branch on the dominant failure mode
without re-parsing Markdown.【F:src/operations/incident_response.py†L242-L354】
`derive_incident_response_alerts` includes the structured detail in alert
contexts and tags category names onto issue events, enabling routing policies to
escalate missing runbooks or postmortem backlogs via dedicated channels under
pytest coverage.【F:tests/operations/test_incident_response.py†L132-L167】

## Feeder snapshots

- `evaluate_incident_response` turns policy/state mappings into a readiness
  snapshot, attaches missing-runbook, roster, backlog, and chatops context, and
  exposes Markdown plus alert derivation helpers so the operational readiness
  aggregate inherits actionable incident evidence under pytest coverage.【F:src/operations/incident_response.py†L249-L715】【F:tests/operations/test_incident_response.py†L1-L200】
- `evaluate_system_validation` ingests structured reports, computes success
  rates, annotates failing checks, and publishes via the shared failover helper
  so readiness retains validator metadata and degraded-check evidence even when
  the runtime bus falters, with regressions covering evaluation, alerting, and
  failover paths.【F:src/operations/system_validation.py†L470-L889】【F:tests/operations/test_system_validation.py†L1-L195】

## Gating deployments with system validation

Call `evaluate_system_validation_gate` on the snapshot to enforce deployment
requirements: FAIL always blocks, `block_on_warn=True` escalates WARN to a
blocking condition, `min_success_rate` guards aggregate pass thresholds, and
`required_checks` ensures critical checks are both present and successful. The
returned `SystemValidationGateResult` records blocking reasons and exposes an
`as_dict()` helper for dashboards and guardrails.【F:src/operations/system_validation.py†L724-L889】【F:tests/operations/test_system_validation.py†L189-L279】

## Dashboard integration

The observability dashboard now renders operational readiness as a dedicated
panel, summarising component severities, highlighting degraded services, and
embedding the full snapshot metadata in panel payloads. Regression coverage
asserts that the panel headlines, remediation summaries, and Markdown export all
surface the readiness status so responders inherit the enriched snapshot without
custom wiring.【F:src/operations/observability_dashboard.py†L443-L493】【F:tests/operations/test_observability_dashboard.py†L135-L236】
