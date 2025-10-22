# Post-enable review log

Use this log to capture the outcomes of the two-week audit that follows every new data integration launch. The entry should summarise production evidence so reliability, compliance, and trading leads can review trends without replaying the full deployment history.

## Required evidence

- **Ingestion health** – Attach the latest runtime health snapshot that includes the new integration so reviewers can confirm gauges remain green after rollout.【F:src/runtime/healthcheck.py†L1-L258】
- **Quality validators** – Include the most recent ingest quality report for the integration together with any anomalies raised by coverage, drift, or staleness checks.【F:src/data_foundation/ingest/quality.py†L1-L200】
- **Anomaly telemetry** – Provide the feed anomaly detector export to show alert thresholds were exercised and tuned using production traffic.【F:src/data_foundation/monitoring/feed_anomaly.py†L1-L200】
- **Operational manifest** – Export the managed ingest connector manifest to document Timescale/Redis/Kafka posture at the time of the review.【F:tools/operations/managed_ingest_connectors.py†L92-L156】
- **Incident recap** – Summarise any incidents, alerts, or manual interventions recorded in the observability plan since the feature flag was enabled.【F:docs/operations/observability_plan.md†L100-L140】

## Template

```
## <Integration name> – Post-enable audit (<YYYY-MM-DD>)
- Health snapshot: <link>
- Quality report: <link>
- Anomaly telemetry: <link>
- Connector manifest: <link>
- Incidents & interventions: <summary>
- Follow-up actions: <list of tickets or owners>
```

Record each completed audit in this file and link to the relevant change ticket for traceability. Keep entries in chronological order with the newest review at the top.
