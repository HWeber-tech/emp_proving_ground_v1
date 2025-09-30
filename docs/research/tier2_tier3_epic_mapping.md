# Tier-2/Tier-3 Vision to Epic Mapping

## Purpose
Provide a transparent bridge between the encyclopedia's Tier‑2 and Tier‑3
capabilities and the actionable backlog maintained in the issue tracker. Each
entry below includes the epic title, description, sequencing notes, and
cross-references to supporting documentation.

## Epic catalogue

| Vision theme | Epic ID | Epic title | Description | Sequencing notes | Linked artefacts |
| --- | --- | --- | --- | --- | --- |
| Institutional data backbone | DATA-203 | Timescale & Redis federation | Extend Timescale ingest with Redis query caching, Kafka mirroring, and automated failover drills. | Requires completion of Timescale scheduler and Kafka consumer hardening (Streams A & C checkpoints). | `docs/roadmap.md`, `docs/research/future_ga_extensions.md`, `docs/operations/runbooks/redis_cache_outage.md` |
| Sensory cortex intelligence | SENSE-178 | NLP sentiment integration | Deliver sentiment ingestion and sensors per the dedicated roadmap. | Blocks on completion of data governance policy updates; should launch after DATA-203 to leverage Kafka mirror. | `docs/research/nlp_sentiment_ingestion_roadmap.md`, `docs/policies/data_governance.md` |
| Evolutionary learning | EVO-152 | Multi-objective GA release | Implement Pareto-front optimisation and live-evolution telemetry. | Must follow Redis/Kafka deployment to guarantee telemetry throughput; run after SENSE-178 for richer features. | `docs/research/future_ga_extensions.md`, `docs/research/frontier_research_brief_Q1_2025.md` |
| Risk & compliance | RISK-244 | Causal guard-rail toolkit | Ship causal inference success metrics, monitoring hooks, and compliance sign-off flow. | Depends on `EVO-152` outputs to reuse telemetry surfaces; coordinates with compliance governance updates. | `docs/research/causal_ml_success_metrics.md`, `docs/operations/regulatory_telemetry.md` |
| Operational readiness | OPS-310 | Disaster recovery & cross-region automation | Automate backup validation, cross-region failover, and ingestion recovery orchestration. | Parallel to EVO-152 but must complete before live evolution to ensure rollback capacity. | `docs/deployment/drills/`, `docs/operations/cross_region_failover.md` |

## Sequencing cadence
1. Execute DATA-203 to stabilise ingestion surfaces and caching.
2. Launch SENSE-178 to onboard textual signals leveraging Kafka throughput.
3. Deliver EVO-152 to unlock Pareto optimisation and streaming evolution.
4. Run OPS-310 in tandem with EVO-152 to secure rollback and audit surfaces.
5. Finalise RISK-244 once causal metrics are validated against evolution
   outputs.

Update this mapping quarterly or whenever new Tier‑2/Tier‑3 vision items land in
the encyclopedia addendum.
