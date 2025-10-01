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
