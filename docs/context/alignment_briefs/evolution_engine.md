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
- Progress: Realistic genome seeding now rotates through catalogue templates,
  harvests recorded experiment manifests into additional templates, derives
  jitter/metrics/tags from those artifacts, and applies lineage/performance
  metadata to default populations so baseline genomes reflect the strategy
  library plus recent experiments. The sampler now enriches genomes with parent
  IDs, mutation histories, and performance fingerprints, doubles as the bootstrap
  path in the population manager, and surfaces parent/mutation counts via lineage
  telemetry so orchestrator dashboards inherit richer provenance under pytest
  coverage of sampler cycling, metadata propagation, and seeded context on the
  evolution engine.【F:src/core/evolution/seeding.py†L82-L140】【F:src/core/population_manager.py†L62-L383】【F:src/core/evolution/engine.py†L267-L335】【F:src/evolution/lineage_telemetry.py†L200-L228】【F:tests/evolution/test_realistic_seeding.py†L48-L88】【F:tests/current/test_population_manager_with_genome.py†L86-L108】
- Build population management routines (selection, mutation, evaluation) tied to
  recorded datasets; expose metrics via runtime summaries.
- Progress: Recorded sensory replay evaluator converts archived sensory
  snapshots into deterministic evaluation metrics and now emits a trade ledger
  with confidence/strength metadata plus a trade-count fitness field so replay
  audits surface actionable evidence under pytest coverage before live ingest
  arrives. A dedicated telemetry builder lifts those metrics into lineage-backed
  Markdown/JSON summaries, flags drawdown/return severities, and records best/
  worst trade diagnostics so governance reviewers inherit ready-to-publish
  replay dossiers.【F:src/evolution/evaluation/recorded_replay.py†L160-L389】【F:src/evolution/evaluation/telemetry.py†L1-L203】【F:tests/evolution/test_recorded_replay_evaluator.py†L37-L98】【F:tests/evolution/test_recorded_replay_telemetry.py†L1-L88】
- Progress: Recorded dataset helpers now dump and reload real sensory snapshots
  to JSONL with lineage metadata, strict/append guards, and replay integration
  tests so adaptive runs can hydrate governance evidence without bespoke
  capture scripts.【F:src/evolution/evaluation/datasets.py†L1-L171】【F:src/evolution/__init__.py†L21-L71】【F:tests/evolution/test_recorded_dataset.py†L1-L108】
- Surface lineage telemetry and integrate with compliance workflow templates for
  strategy approvals.
- Progress: Evolution experiment telemetry now guards publish failures, records
  markdown fallbacks, and documents exception paths via pytest so dashboards and
  status packs retain reliable ROI and backlog metrics even when transports
  misbehave.【F:src/operations/evolution_experiments.py†L40-L196】【F:tests/operations/test_evolution_experiments.py†L1-L126】
- Progress: Evolution readiness evaluator merges the adaptive-run feature flag,
  population seed metadata, and lineage telemetry into a governance snapshot,
  rendering Markdown/JSON summaries with champion provenance and blocking issues
  so reviewers can gate adaptive runs deterministically under pytest coverage.【F:src/operations/evolution_readiness.py†L1-L206】【F:tests/operations/test_evolution_readiness.py†L1-L118】
- Progress: PolicyRouter now prunes expired fast-weight experiments and exposes
  a reflection report helper that packages digest metadata and reviewer-ready
  artifacts so experimentation stays current while governance receives
  consumable summaries under regression coverage.【F:src/thinking/adaptation/policy_router.py†L175-L525】【F:tests/thinking/test_policy_router.py†L248-L308】
- Progress: AdversarialTrainer now logs generator signature mismatches and
  unexpected training failures while preserving heuristic fallbacks so
  experimentation surfaces actionable diagnostics without stalling adaptive
  loops.【F:src/thinking/adversarial/adversarial_trainer.py†L14-L140】
- Progress: Evolution engine now records seed provenance on population
  initialization and after each generation, summarising catalogue templates,
  seed tags, and totals for the population manager while lineage telemetry emits
  the enriched payload under pytest coverage so orchestrator dashboards expose
  deterministic seed metadata instead of opaque populations.【F:src/core/evolution/engine.py†L65-L336】【F:src/core/population_manager.py†L115-L183】【F:src/evolution/lineage_telemetry.py†L1-L200】【F:tests/current/test_evolution_orchestrator.py†L83-L120】【F:tests/evolution/test_lineage_snapshot.py†L8-L66】
- Progress: Adaptive runs remain gated behind the `EVOLUTION_ENABLE_ADAPTIVE_RUNS`
  flag, with orchestrator wiring skipping champion registration and telemetry
  when the flag is disabled and pytest coverage documenting the contract so
  governance can stage reviews before enabling live evolution loops.【F:src/evolution/feature_flags.py†L1-L44】【F:src/orchestration/evolution_cycle.py†L172-L340】【F:tests/current/test_evolution_orchestrator.py†L195-L240】【F:tests/evolution/test_feature_flags.py†L1-L27】

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
