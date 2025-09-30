# Alignment brief – Evolution engine & adaptive intelligence

## Concept promise

- The encyclopedia positions evolutionary intelligence as the engine that adapts
  strategies through genetic algorithms, meta-learning, and optimisation loops
  layered above the sensory cortex.【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L132-L299】
- Planned milestones call for a basic evolution engine in Weeks 7–8 with hooks
  into strategy management and optimisation workflows.【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L528-L704】

## Reality snapshot

- Evolution and intelligence subsystems remain skeletal with `pass` statements
  and no executable algorithms; strategy decisions stay static.【F:docs/DEVELOPMENT_STATUS.md†L19-L35】
- Dead-code reports list evolution modules as unused, confirming the absence of
  active integration paths.【F:docs/reports/CLEANUP_REPORT.md†L71-L175】
- Technical debt priorities emphasise hollow risk controls and async hazards,
  meaning adaptive behaviour cannot be trusted without foundational fixes.【F:docs/technical_debt_assessment.md†L33-L80】

## Gap themes

1. **Real genomes & populations** – Define canonical genome schemas, seed
   catalogue-backed populations, and wire mutation/crossover logic.
2. **Lifecycle integration** – Connect evolution outputs to strategy registries,
   runtime summaries, and compliance workflows with telemetry and audit trails.
3. **Governance & safety** – Enforce risk policies during adaptive runs, record
   lineage, and provide operator controls for promotions/rollbacks.

## Delivery plan

### Now (0–30 days)

- Document existing stubs, remove unused imports, and align module exports to
  prepare for incremental implementation.【F:docs/reports/CLEANUP_REPORT.md†L71-L175】
- Define telemetry contracts for population health, lineage snapshots, and
  experiment metadata; add placeholder tests to lock schemas.
- Progress: Portfolio evolution now logs optional dependency outages, falls back
  to deterministic clustering, and records the degraded path under pytest so
  adaptive runs continue producing reproducible recommendations in minimal
  environments.【F:src/intelligence/portfolio_evolution.py†L47-L142】【F:tests/intelligence/test_portfolio_evolution_security.py†L1-L169】
- Coordinate with risk and data-backbone tracks so adaptive loops inherit
  trustworthy inputs and enforcement.【F:docs/technical_debt_assessment.md†L58-L72】

### Next (30–90 days)

- Implement catalogue-backed genome seeding with configuration toggles and pytest
  coverage.
- Build population management routines (selection, mutation, evaluation) tied to
  recorded datasets; expose metrics via runtime summaries.
- Surface lineage telemetry and integrate with compliance workflow templates for
  strategy approvals.
- Progress: Evolution experiment telemetry now guards publish failures, records
  markdown fallbacks, and documents exception paths via pytest so dashboards and
  status packs retain reliable ROI and backlog metrics even when transports
  misbehave.【F:src/operations/evolution_experiments.py†L40-L196】【F:tests/operations/test_evolution_experiments.py†L1-L126】

### Later (90+ days)

- Enable live adaptive runs gated behind governance approvals; capture rollback
  workflows.
- Add reinforcement-learning or meta-learning extensions once genetic loops are
  stable.
- Remove legacy templates and unused evolution files after the new engine ships
  to reduce dead-code noise.【F:docs/reports/CLEANUP_REPORT.md†L71-L175】

## Dependencies & coordination

- Requires institutional ingest and sensory telemetry to provide reliable fitness
  signals.
- Risk enforcement must land before adaptive strategies can run unattended.【F:docs/technical_debt_assessment.md†L58-L72】
