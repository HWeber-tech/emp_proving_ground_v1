# High-impact roadmap detail

This companion expands the summary stream table with the context needed for
backlog grooming, release readiness reviews, and post-mortems.

## Stream A – Institutional data backbone

**Status:** Attention

**Concept coverage:**
- Layered runtime architecture and canonical `SystemConfig` definitions align
  with the encyclopedia blueprint.【F:docs/architecture/overview.md†L9-L48】
- Governance docs track the remediation phases for SQL hardening, cache rollout,
  and telemetry guardrails.【F:docs/development/remediation_plan.md†L34-L141】

**Reality check:**
- Institutional ingest, Redis caching, and Kafka streaming remain mock
  frameworks; the development status report highlights the absence of production
  data services, risk sizing, and portfolio management.【F:docs/DEVELOPMENT_STATUS.md†L19-L35】
- SQL is still assembled dynamically in legacy helpers, and async ingestion tasks
  use unsupervised `create_task` calls, leaving failure recovery and shutdown
  undefined.【F:docs/technical_debt_assessment.md†L33-L56】
- Production ingest slice now coordinates the Timescale orchestrator, Redis cache
  invalidation, and supervised lifecycles behind a managed entrypoint, though
  the backing Timescale/Redis/Kafka services are still pending real provisioning.【F:src/data_foundation/ingest/production_slice.py†L1-L220】【F:tests/data_foundation/test_production_ingest_slice.py†L1-L220】

**Gaps to close:**
1. Provision real Timescale/Redis/Kafka services with parameterised access layers
   and acceptance tests that demonstrate failover and telemetry parity.
2. Adopt the runtime builder end to end so ingest, cache warmers, and streaming
   consumers register under supervised tasks instead of ad-hoc loops.
3. Finish removing deprecated config shims now that the canonical risk
   configuration lives in `src/config/risk/risk_config.py` and the evolution
   configuration has been consolidated into `src/core/evolution/engine.py`,
   keeping namespaces aligned as ingestion hardens.【F:src/config/risk/risk_config.py†L1-L72】【F:src/core/evolution/engine.py†L13-L43】

**Actionable checklist:**
- [ ] Timescale/Redis/Kafka services provisioned with supervised connectors and failover tests.
- [ ] Runtime builder adoption complete for ingest, cache warmers, and streaming consumers.
- [x] Deprecated config shims replaced with canonical modules across ingest surfaces; risk uses `src/config/risk/risk_config.py`, and evolution imports resolve through `src/core/evolution/engine.py`.【F:src/config/risk/risk_config.py†L1-L72】【F:src/core/evolution/engine.py†L13-L43】

**Validation hooks:**
- Builder-driven smoke test that loads institutional credentials and exercises
  ingest → cache → stream telemetry.
- Bandit scan proves SQL/eval remediation by driving B608/B307 counts to zero for
  ingest modules.【F:docs/development/remediation_plan.md†L34-L61】
- CI baseline extended with coverage for ingest orchestration, cache health, and
  streaming lag metrics beyond the current 76% plateau.【F:docs/ci_baseline_report.md†L8-L27】

## Stream B – Sensory cortex & evolution uplift

**Status:** Attention

**Concept coverage:**
- The encyclopedia describes the five-dimensional sensory cortex and evolutionary
  uplift journey that the alignment briefs must deliver.【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L360-L420】
- Remediation plan phases call for hygiene refactors and structural work in the
  sensory and evolution modules.【F:docs/development/remediation_plan.md†L92-L141】

**Reality check:**
- Evolution, intelligence, and strategy subsystems remain skeletal; multiple
  modules still raise `NotImplementedError` or operate as mock frameworks.【F:docs/DEVELOPMENT_STATUS.md†L19-L35】
- Dead-code audits flag sensory/evolution directories as dormant, reinforcing the
  lack of executable pipelines.【F:docs/reports/CLEANUP_REPORT.md†L71-L175】
- Real sensory organ now fuses WHY/WHAT/WHEN/HOW/ANOMALY signals, records
  lineage, and publishes telemetry snapshots so runtime summaries have a single
  executable sensory surface, though feeds remain synthetic pending the
  institutional backbone.【F:src/sensory/real_sensory_organ.py†L20-L208】【F:src/sensory/real_sensory_organ.py†L210-L336】【F:tests/sensory/test_real_sensory_organ.py†L1-L107】

**Gaps to close:**
1. Implement HOW and ANOMALY organs with deterministic inputs, documented drift
   metrics, and integration into the runtime summary.
2. Replace placeholder genomes with catalogue-backed evolution cycles and surface
   lineage telemetry for governance.
3. Expand regression suites beyond FIX mocks to cover sensory drift, catalogue
   seeding, and adaptive decision loops.

**Actionable checklist:**
- [ ] Executable HOW/ANOMALY organs integrated with runtime summary and drift telemetry.
- [ ] Catalogue-backed genomes with lineage telemetry captured for governance review.
- [ ] Regression suites covering sensory drift, catalogue seeding, and adaptive loops.【F:docs/reports/CLEANUP_REPORT.md†L71-L175】

**Validation hooks:**
- Drift telemetry published to the event bus with pytest coverage.
- Evolution lineage snapshots stored via canonical persistence and referenced in
  compliance workflows.
- Context pack updates in this directory kept in lockstep with the shipped
  sensors and evolution milestones.

## Stream C – Execution, risk, compliance, ops readiness

**Status:** Attention

**Concept coverage:**
- The order lifecycle and layered runtime match the encyclopedia chapters, and
  the remediation plan spells out security and structural refactors for the
  execution stack.【F:docs/architecture/overview.md†L9-L37】【F:docs/development/remediation_plan.md†L34-L167】

**Reality check:**
- Risk management is still hollow: `RiskManager` ignores many configuration
  fields, and trading modules bypass policy enforcement.【F:docs/technical_debt_assessment.md†L58-L72】
- Runtime entrypoints partially adopt the new builder while legacy modules keep
  spawning unsupervised async tasks, creating shutdown hazards.【F:docs/technical_debt_assessment.md†L33-L56】
- Legacy `src.core`/`src.trading` risk facades now raise guided module errors and
  regression tests enforce the canonical imports, though integration docs still
  mention the old helpers.【F:src/core/risk/manager.py†L1-L14】【F:src/trading/risk_management/__init__.py†L1-L8】【F:tests/current/test_risk_shims_retired.py†L1-L23】
- Risk gateway and trading manager now attach runbook-backed `risk_reference`
  payloads to decisions, limit snapshots, and interface summaries so operators
  inherit consistent escalation metadata even while broader risk enforcement
  remains under construction.【F:src/trading/risk/risk_gateway.py†L232-L430】【F:src/trading/trading_manager.py†L815-L968】【F:tests/current/test_risk_gateway_validation.py†L1-L213】【F:tests/trading/test_trading_manager_execution.py†L1157-L1240】
- Bootstrap runtime and Predator control surfaces now publish release execution
  routing snapshots with default stages, engine mappings, and forced paper
  reasons so governance reviews see the execution posture despite mock execution
  engines underneath.【F:src/runtime/bootstrap_runtime.py†L195-L428】【F:src/runtime/predator_app.py†L1001-L1141】【F:tests/current/test_bootstrap_runtime_integration.py†L238-L268】【F:tests/trading/test_trading_manager_execution.py†L960-L983】
- Regulatory telemetry publisher uses the shared failover helper, logging
  runtime failures and falling back to the global bus so compliance snapshots
  persist through outages.【F:src/operations/regulatory_telemetry.py†L11-L388】【F:tests/operations/test_regulatory_telemetry.py†L18-L160】

**Gaps to close:**
1. Finalise the runtime builder migration, introduce a `TaskSupervisor`, and
   retire ad-hoc event-loop orchestration.
2. Implement deterministic risk APIs, enforce policy thresholds, and document
   escalation paths before attempting additional broker pilots.
3. Purge deprecated shims and shrink the dead-code backlog so operators and CI
   rely on a single set of canonical modules.【F:docs/reports/CLEANUP_REPORT.md†L71-L175】

**Actionable checklist:**
- [ ] Runtime builder migration complete with supervised task orchestration.
- [ ] Deterministic risk APIs enforced with documented escalation paths.
- [ ] Deprecated shims removed and dead-code backlog triaged to zero.【F:docs/reports/CLEANUP_REPORT.md†L71-L175】

**Validation hooks:**
- System validation report trends upward from 0/10 by triaging each failing
  check into owned remediation tickets.
- Risk regression suite exercises policy breaches, leverage limits, and shutdown
  behaviour.
- Operational runbooks expand beyond the legacy FIX simulator to cover task
  supervision, alert routing, and rollback drills.
