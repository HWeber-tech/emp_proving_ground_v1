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

Both fields accompany the existing `component_count` and remain present in the
snapshot dictionary returned by `OperationalReadinessSnapshot.as_dict()` and the
payload attached to derived alert events. The regression suite
(`tests/operations/test_operational_readiness.py`) locks the contract so CI fails
if the breakdown or component mapping drift.

## Using the metadata

- Dashboards can render stacked bar charts or summary chips by reading the
  `status_breakdown` map instead of recomputing severities.
- Alerting policies receive the enriched metadata via the alert context, making
  it trivial to route failures for specific components.
- Additional metadata can still be supplied via the `metadata` parameter when
  evaluating readiness; custom keys override defaults if needed.

Combine the snapshot with the observability dashboard output to give operators a
clear view of which operational slices need attention each run.

## Feeder snapshots

- `evaluate_incident_response` turns policy/state mappings into a readiness
  snapshot, attaches missing-runbook, roster, backlog, and chatops context, and
  exposes Markdown plus alert derivation helpers so the operational readiness
  aggregate inherits actionable incident evidence under pytest coverage.【F:src/operations/incident_response.py†L249-L715】【F:tests/operations/test_incident_response.py†L1-L200】
- `evaluate_system_validation` ingests structured reports, computes success
  rates, annotates failing checks, and publishes via the shared failover helper
  so readiness retains validator metadata and degraded-check evidence even when
  the runtime bus falters, with regressions covering evaluation, alerting, and
  failover paths.【F:src/operations/system_validation.py†L1-L312】【F:tests/operations/test_system_validation.py†L1-L195】

## Dashboard integration

The observability dashboard now renders operational readiness as a dedicated
panel, summarising component severities, highlighting degraded services, and
embedding the full snapshot metadata in panel payloads. Regression coverage
asserts that the panel headlines, remediation summaries, and Markdown export all
surface the readiness status so responders inherit the enriched snapshot without
custom wiring.【F:src/operations/observability_dashboard.py†L443-L493】【F:tests/operations/test_observability_dashboard.py†L135-L236】
