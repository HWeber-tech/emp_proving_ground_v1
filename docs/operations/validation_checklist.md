# Validation checklist: promote data integrations safely

This checklist enumerates the gating steps required before enabling a new market-data or fundamentals integration inside the Execution Management Platform (EMP). Complete every item during the release-readiness review that precedes flipping production feature flags.

## Scope

Applies to: new raw data connectors, schema upgrades that affect downstream analytics, and onboarding of new venues or instruments that flow through the ingestion and storage layers.

Excludes: cosmetic dashboard tweaks and research-only datasets that never touch production jobs.

## Checklist

### 1. Data contract readiness
- [ ] Confirm the vendor contract is archived in the compliance workspace and update the vendor summary in the data foundation runbook so operational owners and SLAs stay aligned.【F:docs/runbooks/data_foundation.md†L320-L362】
- [ ] Publish schema and payload documentation alongside the lineage graph so downstream teams can trace ownership, freshness, and retention expectations.【F:src/data_foundation/documentation/lineage.py†L1-L160】
- [ ] Record authentication requirements, token rotation cadence, and storage location in the security playbook to unblock environment provisioning.【F:docs/security/authentication_tokens.md†L1-L160】

### 2. Schema and parser validation
- [ ] Extend or introduce Pydantic models covering the new payloads and ensure field coercion rules are encoded in the canonical schemas.【F:src/data_foundation/schemas.py†L1-L88】
- [ ] Run the ingestion regression suites that exercise the new connector (for example `pytest tests/data_foundation/test_multi_source_aggregator.py -k <integration>`). Tests must assert timezone handling, null-safe parsing, and symbol normalisation.【F:tests/data_foundation/test_multi_source_aggregator.py†L1-L200】
- [ ] Wire the integration into the ingest quality validators so referential integrity, coverage, and drift checks fire before persistence.【F:src/data_foundation/ingest/quality.py†L1-L200】

### 3. Storage and retention controls
- [ ] Validate tiered storage policies cover the new dataset, including hot/cold retention windows and archive metadata expectations.【F:src/data_foundation/storage/tiered_storage.py†L1-L160】
- [ ] Confirm Timescale cache and journal writers can persist the dataset by running the relevant Timescale integration tests in staging.【F:tests/data_foundation/test_timescale_ingest.py†L1-L200】
- [ ] Capture retention and recovery notes in the operational backbone documentation so the cold archive procedure remains reproducible.【F:src/data_foundation/ingest/recovery.py†L1-L200】

### 4. Monitoring and alerting
- [ ] Extend ingestion health metrics with coverage for the new integration and ensure gauges surface through the runtime health server before go-live.【F:src/data_foundation/ingest/health.py†L1-L200】【F:src/runtime/healthcheck.py†L1-L258】
- [ ] Add structured logging fields (`source`, `venue`, `instrument`) to the ingestion path and verify they appear in JSON logs produced by the observability logging helpers.【F:src/observability/logging.py†L1-L120】
- [ ] Feed anomaly detection thresholds with representative data so alert tuning reflects real-world variance rather than lab samples.【F:src/data_foundation/monitoring/feed_anomaly.py†L1-L200】

### 5. Operational sign-off
- [ ] Document runbooks for authentication lapses, schema drift, and upstream outages under `docs/runbooks/<integration>_ingestion.md`, cross-linking escalation paths already defined for the data foundation team.【F:docs/runbooks/data_foundation.md†L1-L200】
- [ ] Secure approvals from Data Engineering, Reliability, and Compliance leads in the rollout ticket, attaching evidence links for every checklist item.
- [ ] Schedule a post-enable audit two weeks after launch to review incidents, anomaly metrics, and backlog items captured in the operational review log.【F:docs/operations/post_enable_reviews.md†L1-L25】

## Audit trail

Once all items are checked, export the completed checklist to the release artifact bundle and store it alongside the deployment ticket. This ensures auditability for regulators and simplifies disaster-recovery rehearsals.
