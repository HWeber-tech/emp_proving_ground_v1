# Alignment brief – Evolution engine & catalogue seeding

**Why this brief exists:** The concept blueprint promises an adaptive evolution
loop seeded with calibrated institutional genomes instead of random parameter
clouds. This brief documents how the new catalogue-backed population seeding
closes that gap, what remains open, and how the roadmap checkpoints tie to
validation artefacts.

## Concept promise

- Tier‑1 evolution mode bootstraps populations from the institutional genome
  catalogue, preserving desk-proven parameter clusters and telemetry for
  governance review.【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L640-L832】
- Population management publishes species composition and provenance so risk and
  compliance flows can audit strategy drift.【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L820-L892】
- Feature flags guard new evolutionary behaviours so bootstrap and paper tiers
  can switch them on deliberately once validation passes.【F:docs/roadmap.md†L84-L118】

## Reality snapshot (September 2025)

- Evolution runs previously generated fully random genomes, offering no linkage
  to institutional desks or roadmap telemetry.【F:src/core/population_manager.py†L220-L340】
- Roadmap execution called for replacing the stub provider with a catalogue but
  lacked concrete data or feature gates to toggle the behaviour.【F:docs/roadmap.md†L96-L118】
- CI had no visibility into seeded populations or catalogue metadata, making it
  difficult to verify when institutional genomes were active.

## Gap response

- `GenomeCatalogue` now curates calibrated entries (trend, carry, volatility,
  liquidity, macro) with metadata, sampling helpers, and telemetry payloads so
  evolution runs inherit real desk knowledge.【F:src/genome/catalogue.py†L1-L221】
- `PopulationManager` seeds from the catalogue when
  `EvolutionConfig.use_catalogue` or `EVOLUTION_USE_CATALOGUE` is enabled,
  storing catalogue metadata alongside population statistics for downstream
  monitoring.【F:src/core/population_manager.py†L1-L420】
- `EvolutionEngine` threads the config flag into the manager so orchestration
  layers can opt-in per run or via environment toggles without code edits.【F:src/core/evolution/engine.py†L1-L120】
- Pytest coverage documents catalogue sampling and feature-flag behaviour so CI
    dashboards and reviewers see the seeded population evidence.
    【F:tests/evolution/test_genome_catalogue.py†L1-L27】【F:tests/current/test_population_manager_catalogue.py†L1-L39】

## Delivery plan

### Now (30-day outlook)

1. **Catalogue telemetry & documentation**
   - Surface catalogue metadata in runtime summaries and status endpoints so
     operators can confirm seeded populations during drills.
   - Publish the catalogue schema and desk provenance in the evolution runbook
     to align compliance and governance teams.
   - **Status:** `EvolutionCycleOrchestrator` now captures catalogue snapshots,
     exposes them via `catalogue_snapshot`, and publishes
     `telemetry.evolution.catalogue` events for the runtime bus while tests cover
     the payload shape and bus integration.

2. **Bootstrap compatibility**
   - Ensure bootstrap tiers remain deterministic by keeping catalogue seeding
     behind the feature flag and documenting how to disable it in local workflows.

### Next (60-day outlook)

3. **Population lifecycle management**
   - Track catalogue lineage through mutation/crossover steps and emit telemetry
     for champion registration so governance can audit how genomes evolve away
     from their initial desks.
   - Introduce species-level quotas and decay policies that respect the catalogue
     composition.
   - **Status:** `EvolutionLineageSnapshot` now merges champion metadata (parent IDs,
     mutation history, species, fitness) with population statistics, publishes
     `telemetry.evolution.lineage`, and exposes the snapshot via orchestrator
     telemetry so operators and CI dashboards can track lineage drift alongside
     catalogue snapshots.【F:src/evolution/lineage_telemetry.py†L1-L165】【F:src/orchestration/evolution_cycle.py†L120-L392】【F:tests/evolution/test_lineage_snapshot.py†L1-L74】【F:tests/current/test_evolution_orchestrator.py†L1-L190】

4. **Governance integration**
   - **Status:** `StrategyRegistry` now records catalogue provenance for every
     champion and the compliance workflow exposes a "Strategy governance"
     checklist that consumes the registry summary, keeping approvals tied to the
     seeded desk templates.【F:src/governance/strategy_registry.py†L1-L420】【F:src/compliance/workflow.py†L1-L760】【F:tests/governance/test_strategy_registry.py†L1-L60】【F:tests/compliance/test_compliance_workflow.py†L1-L140】

### Later (90-day+ considerations)

- Expand the catalogue with alternative asset classes (equities, rates) and tie
  entries to Spark-based validation notebooks once the data backbone matures.
- Stress test catalogue seeding under simulated degradation (missing entries,
  stale performance metrics) and document the failover procedure.
- `evaluate_evolution_experiments` now records paper-trading experiment health,
  publishes `telemetry.evolution.experiments`, and surfaces the latest snapshot
  inside the professional runtime so operators can audit experiment outcomes
  alongside ingest telemetry.【F:src/operations/evolution_experiments.py†L1-L248】【F:src/runtime/runtime_builder.py†L2059-L2116】【F:src/runtime/predator_app.py†L848-L867】【F:tests/operations/test_evolution_experiments.py†L1-L114】
- `evaluate_evolution_tuning` combines experiment telemetry with per-strategy
  performance to emit `telemetry.evolution.tuning` guidance, publishes the feed
  through the runtime builder, and records the markdown block in professional
  summaries so operators can review automated tuning recommendations during
  runbooks and postmortems.【F:src/operations/evolution_tuning.py†L1-L443】【F:src/runtime/runtime_builder.py†L2566-L2649】【F:src/runtime/predator_app.py†L229-L515】【F:src/runtime/predator_app.py†L1098-L1104】【F:tests/operations/test_evolution_tuning.py†L1-L172】【F:tests/runtime/test_professional_app_timescale.py†L1298-L1338】

## Validation hooks

- **Pytest coverage** – Maintain the catalogue sampling tests and extend them
  whenever new species land to keep CI telemetry aligned.
- **Population statistics** – Capture the catalogue metadata snapshot in the CI
  health report so regression reviewers see the seeded provenance alongside
  ingest telemetry.
- **Feature-flag drills** – Run automated checks ensuring the evolution engine
  respects `use_catalogue=False` and reverts to random seeding for bootstrap
  tiers.

## Open questions

1. How should we version catalogue entries as desks recalibrate strategies—do we
   preserve historical versions for replayability or always promote the latest?
2. What governance checkpoints are required before enabling catalogue seeding in
   Tier‑1 staging—e.g., does compliance need to sign off on every seeded species?
3. How do we surface catalogue drift over time so operators can see when evolved
   strategies diverge materially from their seeded playbooks?
