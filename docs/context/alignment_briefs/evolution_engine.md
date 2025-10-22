# Alignment brief – Evolution engine & adaptive intelligence

## Concept promise

- The encyclopedia positions evolutionary intelligence as the engine that adapts
  strategies through genetic algorithms, meta-learning, and optimisation loops
  layered above the sensory cortex.【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L132-L299】
- Planned milestones call for a basic evolution engine in Weeks 7–8 with hooks
  into strategy management and optimisation workflows.【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L528-L704】

## Reality snapshot

- Evolution orchestration still leans heavily on scaffolding—key planners such
  as episodic memory remain stubs—but the codebase now includes a working NSGA-II
  core and safety controller even though neither is yet integrated into the
  production adaptation loop.【F:docs/DEVELOPMENT_STATUS.md†L19-L35】【F:src/evolution/algorithms/nsga2.py†L1-L334】【F:src/evolution/safety/controls.py†L1-L260】
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
- Progress: Intelligence package now lazily exposes the canonical sentient adaptation engine from `src.sentient.adaptation`, keeps legacy imports working via a lightweight shim, and retires the old `specialized_predators` surface; regression coverage proves duplicate class definitions are gone and removed modules raise deterministically so callers migrate to the supported API.【F:src/intelligence/__init__.py†L23-L193】【F:src/sentient/adaptation/sentient_adaptation_engine.py†L1-L211】【F:src/intelligence/sentient_adaptation.py†L1-L22】【F:tests/current/test_public_api_intelligence.py†L1-L87】
- Establish canonical strategy contracts that describe genotypes (features, topology, tunables, risk templates) and realise phenotypes with validated overrides so mutation operators have a typed substrate.
- Progress: New `StrategyFeature`/`StrategyGenotype`/`StrategyPhenotype` dataclasses normalise identifiers, freeze metadata, and enforce override guards while tests cover duplicate detection, bound checks, and phenotype realisation so evolution workflows can consume deterministic contracts immediately.【F:src/thinking/adaptation/strategy_contracts.py†L1-L409】【F:tests/thinking/adaptation/test_strategy_contracts.py†L17-L119】
- Build operator constraint frameworks that gate adaptive operations (register, degrade, mutate) by stage, regime, and parameter bounds, emitting reviewer-ready violation data.
- Progress: EvolutionManager now resolves `OperatorConstraintSet` bundles, threads policy regime context into adaptive runs, and refuses constraint breaches while tests lock allow/deny flows and violation metadata so operators can review blocked mutations deterministically.【F:src/thinking/adaptation/operator_constraints.py†L1-L388】【F:src/thinking/adaptation/evolution_manager.py†L80-L412】【F:src/orchestration/alpha_trade_loop.py†L658-L699】【F:tests/thinking/test_operator_constraints.py†L1-L104】【F:tests/thinking/test_evolution_manager.py†L310-L455】
- Progress: Preference articulation framework now exposes objective profiles,
  interactive tuning, and articulator helpers that feed weighted scoring into
  evolution loops, with pytest locking clamp, normalisation, and CLI flows so
  operators can shift priorities without bespoke tooling.【F:src/evolution/optimization/preferences.py†L1-L219】【F:tests/evolution/test_preferences.py†L1-L79】
- Progress: Evolution safety controller packages deterministic guard rails for
  drawdown, VaR, latency, slippage, and data quality, issuing cooldown and
  lockdown decisions that orchestration layers can consume once they wire the
  helper in.【F:src/evolution/safety/controls.py†L1-L260】【F:tests/evolution/test_safety_controls.py†L1-L109】
- Progress: Ecosystem optimizer now hoists species/regime heuristics into cached
  maps, batch-evaluates genomes, and samples from a bounded selection pool so
  population churn favours top performers while shrinking the critical path for
  large species cohorts.【F:src/ecosystem/optimization/ecosystem_optimizer.py†L80-L304】
- Implement genotype mutation operators that add/drop features, swap execution topology, and tighten risk templates while preserving provenance for governance reviewers.
- Progress: `op_add_feature`, `op_drop_feature`, `op_swap_execution_topology`, and `op_tighten_risk` now clone immutable genotypes, normalise mapping inputs, and emit `GenotypeOperatorResult` payloads capturing change metadata; the adaptation package exports the helpers and pytest coverage locks duplicate, replacement, and risk-floor scenarios.【F:src/thinking/adaptation/operators.py†L1-L353】【F:src/thinking/adaptation/__init__.py†L28-L98】【F:tests/thinking/adaptation/test_genotype_operators.py†L1-L158】
- Progress: NSGA-II now powers multi-objective selection with non-dominated sorting, crowding distance, configurable crossover/mutation hooks, and export helpers so Pareto fronts can be inspected in dashboards once integrated into the adaptation loop.【F:src/evolution/algorithms/nsga2.py†L1-L334】【F:tests/evolution/test_nsga2.py†L1-L119】
- Define telemetry contracts for population health, lineage snapshots, and
  experiment metadata; add placeholder tests to lock schemas.
- Progress: Legacy portfolio evolution module now raises a descriptive
  `ModuleNotFoundError`, and guard tests assert the removal so adaptive callers pivot to
  the canonical ecosystem surfaces instead of reviving the stub.【F:src/intelligence/portfolio_evolution.py:1】【F:tests/thinking/test_portfolio_evolution_security.py:1】
- Coordinate with risk and data-backbone tracks so adaptive loops inherit
  trustworthy inputs and enforcement.【F:docs/technical_debt_assessment.md†L58-L72】
- Progress: Experimentation scheduler CLI now drains `stage=idea` rows through quick screening, UCB-lite promotion, and bounded quick/full budgets while stamping baselines and metadata so replay candidates progress without bespoke notebooks under pytest coverage.【F:emp/cli/emp_cycle_scheduler.py†L1-L200】【F:emp/cli/_emp_cycle_common.py†L1-L216】【F:tests/emp_cycle/test_cycle_scheduler.py†L1-L86】

### Next (30–90 days)

- Implement catalogue-backed genome seeding with configuration toggles and pytest
  coverage.
- Progress: Evolution manager now seeds managed tactics with catalogue-backed
  variants, injects lineage metadata into registered tactics, and resets loss
  windows once adaptations fire so adaptive trials rotate through live catalogue
  definitions under regression coverage of variant promotion, feature-flag
  gating, and paper-stage enforcement.【F:src/thinking/adaptation/evolution_manager.py†L70-L205】【F:tests/thinking/test_evolution_manager.py†L24-L150】
- Progress: When catalogue queues empty the manager now synthesises
  parameter-mutation variants, scaling or offsetting numeric fields and tagging
  the identifiers for governance reviewers while regression checks confirm the
  mutated tactics retain paper-only guardrails.【F:src/thinking/adaptation/evolution_manager.py†L207-L269】【F:tests/thinking/test_evolution_manager.py†L204-L252】
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
- Progress: Recorded replay CLI now ingests archived sensory datasets,
  normalises inline genome definitions or JSON files, and renders JSON/Markdown
  telemetry decks with drawdown thresholds so evolution reviews can publish
  deterministic evidence without bespoke notebooks.【F:tools/evolution/recorded_replay_cli.py†L1-L320】【F:tests/tools/test_recorded_replay_cli.py†L44-L148】
- Progress: Bootstrap runtime now builds the evolution orchestrator from system
  config extras, executes cycles on a configurable cadence, and surfaces
  evolution telemetry plus cadence metadata through `status()` so governance and
  runtime summaries observe adaptive readiness under integration coverage.【F:src/runtime/predator_app.py†L1992-L2124】【F:src/runtime/bootstrap_runtime.py†L310-L624】【F:tests/current/test_bootstrap_runtime_integration.py†L153-L169】【F:tests/runtime/test_bootstrap_runtime_sensory.py†L162-L194】
- Progress: Evolution guardrails now log structured warnings when genome
  normalisers, mutation metadata updates, or seed attribute mutations fail,
  capturing the genome identifier and action while continuing execution so
  adaptive runs surface unsafe integrations without crashing, with pytest
  coverage around the defensive helpers.【F:src/core/evolution/engine.py†L1-L342】【F:src/core/evolution/seeding.py†L1-L220】【F:tests/evolution/test_evolution_security.py†L1-L95】
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
- Progress: PolicyRouter now tracks regime fitness across tactics, runs tournament selection with composite bonuses once each regime accrues history, and exposes snapshots for dashboards so adaptive routing leans on historical performance instead of single-step scores under guardrail tests.【F:src/thinking/adaptation/regime_fitness.py†L17-L207】【F:src/thinking/adaptation/policy_router.py†L582-L756】【F:tests/thinking/test_policy_router.py†L530-L623】
- Progress: Fast-weight constraints now clamp multipliers to non-negative values, prune excess activations to keep the router sparse, and emit `%_active` metrics inside reflection summaries so reviewers see enforced BDH-style sparsity, with a dedicated fast-weight test suite covering the invariants.【F:src/thinking/adaptation/fast_weights.py†L1-L160】【F:src/thinking/adaptation/policy_router.py†L320-L402】【F:tests/thinking/test_fast_weights.py†L1-L49】【F:tests/thinking/test_policy_router.py†L121-L152】
- Progress: Policy decisions now embed fast-weight sparsity metrics and propagate them through the understanding router so diaries and governance reports inherit activation counts, percentages, and participant lists under regression coverage.【F:src/thinking/adaptation/policy_router.py†L323-L411】【F:src/understanding/router.py†L260-L263】【F:tests/thinking/test_policy_router.py†L123-L145】【F:tests/understanding/test_understanding_router.py†L99-L128】
- Progress: Reflection builder now highlights emerging tactics and experiments
  with first/last-seen timestamps, decision counts, share, and gating metadata,
  emitting reviewer insights that call out regime filters and confidence
  thresholds so governance can track new behaviour without replaying telemetry,
  backed by pytest coverage for timezone normalisation and Markdown exports.【F:src/thinking/adaptation/policy_reflection.py†L153-L213】【F:src/thinking/adaptation/policy_router.py†L977-L1033】【F:tests/thinking/test_policy_reflection_builder.py†L80-L113】【F:tests/thinking/test_policy_router.py†L243-L277】
- Progress: Reflection digest now tracks confidence curves, feature highlights,
  and weight-multiplier statistics while decision diaries include weight
  breakdowns so reviewers can trace how fast weights and experiments influenced
  each recommendation under expanded pytest coverage. The latest uplift also
  aggregates `weight_stats`—base-score, multiplier, and fast-weight summaries—so
  reviewers see scoring dynamics without replaying raw history payloads.【F:src/thinking/adaptation/policy_router.py†L493-L977】【F:src/thinking/adaptation/policy_reflection.py†L205-L334】【F:src/understanding/decision_diary.py†L52-L82】【F:tests/thinking/test_policy_router.py†L59-L333】【F:tests/understanding/test_decision_diary.py†L68-L118】
- Progress: AdversarialTrainer now logs generator signature mismatches and
  unexpected training failures while preserving heuristic fallbacks so
  experimentation surfaces actionable diagnostics without stalling adaptive
  loops.【F:src/thinking/adversarial/adversarial_trainer.py†L14-L140】
- Progress: Evolution engine now records seed provenance on population
  initialization and after each generation, summarising catalogue templates,
  seed tags, and totals for the population manager while lineage telemetry emits
  the enriched payload under pytest coverage so orchestrator dashboards expose
  deterministic seed metadata instead of opaque populations.【F:src/core/evolution/engine.py†L65-L336】【F:src/core/population_manager.py†L115-L183】【F:src/evolution/lineage_telemetry.py†L1-L200】【F:tests/current/test_evolution_orchestrator.py†L83-L120】【F:tests/evolution/test_lineage_snapshot.py†L8-L66】
- Progress: Paper-stage adaptive runs now emit an `EvolutionAdaptationResult`
  describing the base tactic, win-rate, observations, and actions, and the
  AlphaTrade loop persists that payload into iteration metadata and diaries so
  governance reviewers see catalogue registrations and degradations inline with
  each trade.【src/thinking/adaptation/evolution_manager.py:104】【src/orchestration/alpha_trade_loop.py:299】【tests/orchestration/test_alpha_trade_loop.py:485】
- Progress: Replay-driven adaptive evaluations now route through `TacticReplayHarness`, which normalises recorded sensory snapshots, compares outcomes to stage thresholds, and emits governance-ready promotion/demotion decisions that the `AdaptiveGovernanceGate` applies to the policy ledger with audit metadata and optional approvals; guardrail tests prove promotions, demotions, and maintain paths update ledger state deterministically.【F:src/thinking/adaptation/replay_harness.py†L1-L274】【F:src/governance/adaptive_gate.py†L15-L101】【F:tests/thinking/test_adaptive_replay_harness.py†L88-L173】
- Progress: Strategy mixing operator now blends scored tactics with switching friction, decay, and per-tactic bounds via `op_mix_strategies`, returning typed results and coverage for share limits so ensembles evolve safely under regression tests.【F:src/evolution/mutation/strategy_mixer.py†L1-L200】【F:tests/evolution/test_strategy_mix_operator.py†L1-L118】
- Progress: Evolution engine now supports `mu_plus_lambda` survivor selection with configurable offspring counts, guardrails in `EvolutionConfig`, and regression tests that lock survivor preservation and invalid configuration handling so search loops can alternate between tournament and steady-state selection.【F:src/core/evolution/engine.py†L45-L320】【F:tests/current/test_evolution_engine_basic.py†L1-L58】
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

## Definition of Done templates

**Fast-Weight Adaptation Tuning.** Consider this deliverable complete when the fast-weight module enforces non-negative, sparsity-aware updates (thresholded or ReLU-style) configurable via `SystemConfig`, telemetry exposes the `%_active_strategies` metric alongside router snapshots, guardrail tests in `tests/thinking/test_policy_router.py` and a dedicated fast-weight suite lock the non-negative constraint with and without the feature flag, and decision diary reflections record the fast-weight summary for each loop iteration.

**Seed the Evolution Engine (Basic).** The seed phase is done when the strategy catalogue contains at least two lineage-tagged variants, EvolutionManager triggers swaps or parameter perturbations after the configured loss streak using live performance telemetry, regression suites in `tests/current/test_population_manager_with_genome.py` and `tests/thinking/test_policy_router.py` assert the replacement path, and governance logs capture the spawned or demoted tactics while the feature flag keeps the behaviour in paper mode.

## Dependencies & coordination

- Requires institutional ingest and sensory telemetry to provide reliable fitness
  signals.
- Risk enforcement must land before adaptive strategies can run unattended.【F:docs/technical_debt_assessment.md†L58-L72】
