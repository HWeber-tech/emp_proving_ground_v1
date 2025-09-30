# Alignment brief – Institutional data backbone

## Concept promise

- Professional tiers require TimescaleDB for time-series storage, Redis for hot
  caches, Kafka for streaming, and Spark for batch analytics as part of the
  layered data flow that feeds the sensory cortex and execution stack.【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L376-L436】
- The architecture overview confirms the layered runtime that should host these
  services once implemented.【F:docs/architecture/overview.md†L9-L37】

## Reality snapshot

- The development status report classifies ingest, evolution, execution, and
  strategy modules as mock frameworks with no production-grade market data or
  risk sizing pipelines.【F:docs/DEVELOPMENT_STATUS.md†L19-L35】
- Technical debt findings flag unsupervised async tasks, hollow risk checks, and
  namespace drift that block dependable runtime assembly.【F:docs/technical_debt_assessment.md†L33-L80】
- Deprecated configs remain in-tree, signalling unfinished de-shimming work that
  risks misconfiguration when real services arrive.【F:src/config/risk_config.py†L1-L8】【F:src/config/evolution_config.py†L1-L8】

## Gap themes

1. **Infrastructure reality** – Provision real Timescale/Redis/Kafka services,
   parameterise SQL queries, and capture operational runbooks.
2. **Runtime discipline** – Adopt the builder abstraction everywhere, supervise
   background jobs, and remove deprecated shims.
3. **Observability** – Extend CI and runtime telemetry to include ingest health,
   cache hit ratios, streaming lag, and failover drills beyond the current
   baseline.【F:docs/ci_baseline_report.md†L8-L27】

## Delivery plan

### Now (0–30 days)

- Complete the security remediation tranche for SQL construction and `eval`
  removal in ingest modules.【F:docs/development/remediation_plan.md†L34-L61】
- Wire all runtime entrypoints through `RuntimeApplication` and a task supervisor
  so ingest, cache, and stream jobs are supervised.【F:docs/technical_debt_assessment.md†L33-L56】
- ✅ Timescale ingest scheduler now delegates background task creation to the
  runtime `TaskSupervisor`, preventing unsupervised loops and keeping shutdown
  hooks observable.【F:src/data_foundation/ingest/scheduler.py†L18-L115】【F:src/runtime/runtime_builder.py†L2966-L2992】
- Document current gaps and expected telemetry in updated runbooks and status
  pages (this brief, roadmap, high-impact reports).

### Next (30–90 days)

- Stand up managed Timescale, Redis, and Kafka environments in staging, including
  schema migrations, connection pooling, and credential rotation procedures.
- Implement cache health, ingest quality, and Kafka lag probes with pytest
  coverage and CI export.
- Replace deprecated configuration imports in ingest and runtime modules with
  canonical equivalents to prevent namespace drift.

### Later (90+ days)

- Exercise cross-region failover and batch backfill drills; capture playbooks for
  operators and compliance reviewers.
- Integrate Spark export pipelines and storage retention audits aligned with the
  encyclopedia’s enterprise claims.【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L376-L395】
- Automate dead-code sweeps to delete obsolete ingest paths once new services are
  stable.【F:docs/reports/CLEANUP_REPORT.md†L71-L175】

## Dependencies & coordination

- Risk enforcement work must land concurrently so ingest telemetry feeds policy
  decisions safely.【F:docs/technical_debt_assessment.md†L58-L72】
- Operational readiness initiatives (alert routing, incident response) rely on
  accurate ingest telemetry; coordinate milestone sequencing accordingly.
