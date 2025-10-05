# Canonicalization Plan — Phase 1

Purpose
- Establish one canonical definition per concept and eliminate duplicate class/function declarations with minimal breakage.
- Use thin re-export shims in legacy locations to keep imports working while code is migrated.
- Adopt stable, domain-aligned import paths to prevent backsliding.

Inputs
- Duplicate scan artifacts:
  - [docs/reports/duplicate_map_classes.csv](docs/reports/duplicate_map_classes.csv)
  - [docs/reports/duplicate_map_functions.csv](docs/reports/duplicate_map_functions.csv)
  - [docs/reports/duplicate_map.json](docs/reports/duplicate_map.json)
- Summary snapshot (from [docs/reports/duplicate_map.json](docs/reports/duplicate_map.json)):
  - Classes duplicate groups: 56
  - Functions duplicate groups: 2
  - Top families (count ≥ 3): EcosystemOptimizer, FAISSPatternMemory, MarketDataGenerator, MemoryEntry, OrderBookSnapshot, PredictiveMarketModeler, RealTimeLearningEngine, RiskManager, StrategyTester, ValidationResult

Guiding principles
- Domain-ownership: the canonical lives in the domain where the type conceptually belongs (core, trading, ecosystem, sensory, validation, sentient, thinking, intelligence).
- Import stability: canonical paths become the only source of truth; legacy files re-export temporarily.
- __init__.py exports only: no type implementations in __init__.py modules.
- No behavior changes in Phase 1: this is a structural merge. Semantic changes (refactors) are out-of-scope for this phase.
- Roll-forward-first: prefer re-export shims over touching call sites immediately. Call sites migrate in batches later.

Conventions (canonical import roots)
- Core primitives/utilities: under [src/core/](src/core/)
- Trading models and order book types: under [src/trading/](src/trading/)
- Ecosystem orchestrators/optimizers: under [src/ecosystem/](src/ecosystem/)
- Validation models/utilities: under [src/validation/](src/validation/)
- Sensory models/dimensions: under [src/sensory/](src/sensory/)
- Learning/memory for agents: under [src/sentient/](src/sentient/)
- Predictive modeling and research prototypes: under [src/thinking/](src/thinking/)
- Intelligence orchestration glue: under [src/intelligence/](src/intelligence/)
- Simulation and generators: under [src/simulation/](src/simulation/)

Phase 1 scope — families and decisions

1) EventBus (2)
- Observed in:
  - [src/core/event_bus.py](src/core/event_bus.py)
  - [src/operational/event_bus.py](src/operational/event_bus.py)
- Canonical: [src/core/event_bus.py](src/core/event_bus.py)
- Legacy shim: keep [src/operational/event_bus.py](src/operational/event_bus.py) as a re-export only.
- Rationale: event bus is a core infrastructure primitive.

2) RiskManager (3) and RiskConfig (2)
- Observed in:
  - [src/core/risk/manager.py](src/core/risk/manager.py)
  - [src/core/risk_manager.py](src/core/risk_manager.py)
  - ~~src/risk.py~~ (removed; canonical import surface is the package module)
  - ~~src/core.py~~ (removed; package exports provide legacy accessors)
- Canonical: [src/core/risk/manager.py](src/core/risk/manager.py)
- RiskConfig canonical target: create [src/config/risk/risk_config.py](src/config/risk/risk_config.py) or reuse [src/config/risk/](src/config/risk/) if a module already fits.
- Legacy shims:
  - [src/core/risk_manager.py](src/core/risk_manager.py) re-exports from core.risk.manager
  - ~~src/risk.py~~ (removed; see above)
  - ~~src/core.py~~ (removed; see above)
- Rationale: consolidate under core risk package; config lives under config.

3) Instrument (2) and InstrumentProvider (2)
- Observed in:
  - [src/core/instrument.py](src/core/instrument.py)
  - ~~src/core.py~~ (removed; instruments now surface via the package exports)
  - [src/domain/models.py](src/domain/models.py)
- Canonical: [src/core/instrument.py](src/core/instrument.py)
- Provider canonical target: create [src/core/instrument_provider.py](src/core/instrument_provider.py) (move provider here).
- Legacy shims:
  - ~~src/core.py~~ (removed; see above)
  - [src/domain/models.py](src/domain/models.py) re-exports InstrumentProvider
- Rationale: instruments are core primitives; avoid multi-ownership.

4) ValidationResult (3)
- Observed in:
  - [src/validation/validation_framework.py](src/validation/validation_framework.py)
  - [src/data_integration/__init__.py](src/data_integration/__init__.py)
  - ~~src/risk.py~~ (removed; validation imports go through the package)
- Canonical: create [src/validation/models.py](src/validation/models.py) and define ValidationResult here, or move into [src/validation/validation_framework.py](src/validation/validation_framework.py) if it already hosts the model cleanly.
- Legacy shims:
  - [src/data_integration/__init__.py](src/data_integration/__init__.py) re-exports from validation.models
  - ~~src/risk.py~~ (removed; see above)
- Rationale: Validation types belong in validation.

5) OrderBookSnapshot (3) and OrderBookLevel (2)
- Observed in:
  - [src/trading/strategies/order_book_analyzer.py](src/trading/strategies/order_book_analyzer.py)
  - [src/sensory/organs/dimensions/base_organ.py](src/sensory/organs/dimensions/base_organ.py)
  - [src/sensory/organs/dimensions/data_integration.py](src/sensory/organs/dimensions/data_integration.py)
- Canonical: create [src/trading/order_management/order_book/snapshot.py](src/trading/order_management/order_book/snapshot.py) for types (including OrderBookLevel).
- Legacy shims:
  - [src/sensory/organs/dimensions/base_organ.py](src/sensory/organs/dimensions/base_organ.py) re-exports or imports from trading.order_management.order_book.snapshot
  - [src/trading/strategies/order_book_analyzer.py](src/trading/strategies/order_book_analyzer.py) imports from canonical
- Rationale: order book models are trading domain types.

6) Position (2)
- Observed in:
  - [src/trading/models.py](src/trading/models.py)
  - [src/trading/models/position.py](src/trading/models/position.py)
- Canonical: [src/trading/models/position.py](src/trading/models/position.py)
- Legacy shim: [src/trading/models.py](src/trading/models.py) re-exports Position only (drop class body).
- Rationale: avoid duplicate type declared in aggregator modules.

7) OrderStatus (2)
- Observed in:
  - [src/operational/icmarkets_robust_application.py](src/operational/icmarkets_robust_application.py)
  - [src/trading/models/order.py](src/trading/models/order.py)
- Canonical: [src/trading/models/order.py](src/trading/models/order.py)
- Legacy shim: operational site references import from trading; if inline enum/class exists, replace with alias.
- Rationale: trading owns order models.

8) PerformanceTracker (2) and PerformanceMetrics (2)
- Observed in:
  - [src/trading/monitoring/performance_tracker.py](src/trading/monitoring/performance_tracker.py)
  - [src/sensory/organs/dimensions/utils.py](src/sensory/organs/dimensions/utils.py)
  - [src/trading/portfolio/real_portfolio_monitor.py](src/trading/portfolio/real_portfolio_monitor.py)
- Canonical: trading monitoring modules
  - [src/trading/monitoring/performance_tracker.py](src/trading/monitoring/performance_tracker.py)
  - [src/trading/portfolio/real_portfolio_monitor.py](src/trading/portfolio/real_portfolio_monitor.py) owns metrics if it defines them
- Legacy shim: [src/sensory/organs/dimensions/utils.py](src/sensory/organs/dimensions/utils.py) re-exports or imports from trading.*
- Rationale: performance tracking is a trading concern, not sensory.

9) Sensory dimensions duplicated in __init__.py (WhatDimension, AnomalyDimension, ChaosDimension, etc.)
- Observed in:
  - [src/sensory/dimensions/__init__.py](src/sensory/dimensions/__init__.py)
  - [src/sensory/organs/dimensions/anomaly_dimension.py](src/sensory/organs/dimensions/anomaly_dimension.py)
  - [src/sensory/organs/dimensions/chaos_dimension.py](src/sensory/organs/dimensions/chaos_dimension.py)
  - [src/sensory/organs/dimensions/pattern_engine.py](src/sensory/organs/dimensions/pattern_engine.py)
- Canonical: concrete dimension modules (not __init__.py).
- Legacy shim: __init__.py re-exports only.
- Rationale: keep __init__ as namespace, not implementation.

10) FAISSPatternMemory (3) and MemoryEntry (3)
- Observed in:
  - [src/sentient/memory/faiss_pattern_memory.py](src/sentient/memory/faiss_pattern_memory.py)
  - [src/thinking/memory/faiss_memory.py](src/thinking/memory/faiss_memory.py)
  - [src/sentient/memory/faiss_pattern_memory.py](src/sentient/memory/faiss_pattern_memory.py) and [src/thinking/memory/pattern_memory.py](src/thinking/memory/pattern_memory.py)
- Canonical: sentient memory package
  - [src/sentient/memory/faiss_pattern_memory.py](src/sentient/memory/faiss_pattern_memory.py)
- Legacy shims:
  - [src/thinking/memory/faiss_memory.py](src/thinking/memory/faiss_memory.py) re-exports
  - [src/thinking/memory/pattern_memory.py](src/thinking/memory/pattern_memory.py) imports MemoryEntry from sentient.*
- Rationale: memory persistence is part of agent (sentient) domain.

11) RealTimeLearningEngine (3)
- Observed in:
  - [src/sentient/learning/real_time_learning_engine.py](src/sentient/learning/real_time_learning_engine.py)
  - [src/thinking/learning/real_time_learner.py](src/thinking/learning/real_time_learner.py)
  - [src/intelligence/sentient_adaptation.py](src/intelligence/sentient_adaptation.py)
- Canonical: [src/sentient/learning/real_time_learning_engine.py](src/sentient/learning/real_time_learning_engine.py)
- Legacy shims: imports in thinking/intelligence modules re-export class from sentient.*
- Rationale: run-time agent learning is sentient.

12) PredictiveMarketModeler (3), MarketScenario (2), MarketScenarioGenerator (2)
- Observed in:
  - [src/thinking/prediction/predictive_market_modeler.py](src/thinking/prediction/predictive_market_modeler.py)
  - [src/thinking/prediction/predictive_modeler.py](src/thinking/prediction/predictive_modeler.py)
  - [src/intelligence/predictive_modeling.py](src/intelligence/predictive_modeling.py)
- Canonical: thinking prediction package
  - [src/thinking/prediction/predictive_market_modeler.py](src/thinking/prediction/predictive_market_modeler.py)
  - Optional: move MarketScenario(+Generator) to [src/simulation/market/](src/simulation/market/) in Phase 2 to separate data generation from modeling.
- Legacy shims: intelligence module imports from thinking.* only
- Rationale: thinking hosts research/prototyping; simulation hosts generators.

13) StrategyTester (3) and AdversarialTrainer (2)
- Observed in:
  - [src/thinking/adversarial/market_gan.py](src/thinking/adversarial/market_gan.py)
  - [src/intelligence/adversarial_training.py](src/intelligence/adversarial_training.py)
- Canonical: create a test harness in trading strategy engine:
  - [src/trading/strategy_engine/testing/strategy_tester.py](src/trading/strategy_engine/testing/strategy_tester.py)
- Legacy shims: thinking/intelligence modules import from trading.strategy_engine.testing
- Rationale: strategy testing belongs to trading engine.

14) CoordinationEngine (2), EcosystemOptimizer (3), SpecializedPredatorEvolution (2), NicheDetector (2), SpeciesManager (2)
- Observed in:
  - [src/ecosystem/coordination/coordination_engine.py](src/ecosystem/coordination/coordination_engine.py)
  - [src/ecosystem/optimization/ecosystem_optimizer.py](src/ecosystem/optimization/ecosystem_optimizer.py)
  - Duplicates mirrored under [src/thinking/ecosystem/](src/thinking/ecosystem/) and [src/intelligence/specialized_predators.py](src/intelligence/specialized_predators.py)
- Canonical:
  - [src/ecosystem/coordination/coordination_engine.py](src/ecosystem/coordination/coordination_engine.py)
  - [src/ecosystem/optimization/ecosystem_optimizer.py](src/ecosystem/optimization/ecosystem_optimizer.py)
- Legacy shims: thinking/intelligence re-export from ecosystem.*
- Rationale: ecosystem orchestration belongs to ecosystem domain.

15) CurrencyConverter (2)
- Observed in:
  - ~~src/core.py~~ (removed; stale shim deleted)
  - [src/domain/models.py](src/domain/models.py)
- Canonical: create [src/core/finance/currency_converter.py](src/core/finance/currency_converter.py)
- Legacy shims:
  - ~~src/core.py~~ (removed)
  - [src/domain/models.py](src/domain/models.py) imports from core.finance
- Rationale: financial utilities are core-level concerns.

16) Phase2DIntegrationValidator (2)
- Observed in:
  - [src/validation/phase2d_integration_validator.py](src/validation/phase2d_integration_validator.py)
  - ~~src/phase2d_integration_validator.py~~ (removed; see cleanup report)
- Canonical: [src/validation/phase2d_integration_validator.py](src/validation/phase2d_integration_validator.py)
- Legacy shim: root module re-exports only (or deletes content and forwards).
- Rationale: validation resides under validation domain.

17) SensorSignal (2), IntegratedSignal (2)
- Observed in:
  - [src/sensory/signals.py](src/sensory/signals.py)
  - [src/sensory/__init__.py](src/sensory/__init__.py)
- Canonical: [src/sensory/signals.py](src/sensory/signals.py)
- Legacy shim: [src/sensory/__init__.py](src/sensory/__init__.py) re-exports only.
- Rationale: no implementations in package __init__.

Function duplicates

- main (7): keep per-entrypoint scripts; do not unify. These are script entry points by design. Documented exception.
- get_global_cache (2):
  - Observed in: [src/performance/__init__.py](src/performance/__init__.py), [src/core/performance/market_data_cache.py](src/core/performance/market_data_cache.py)
  - Canonical: [src/core/performance/market_data_cache.py](src/core/performance/market_data_cache.py)
  - Legacy shim: [src/performance/__init__.py](src/performance/__init__.py) re-exports only
  - Rationale: keep implementation under core performance.

Re-export shim policy (Phase 1)
- Re-export only; no additional logic in legacy modules.
- Single source-of-truth import path used by all new code.
- Shims live where current imports originate:
- Example targets to receive shims (non-exhaustive): [src/operational/event_bus.py](src/operational/event_bus.py), ~~src/core.py~~ (removed), ~~src/risk.py~~ (removed), [src/trading/models.py](src/trading/models.py), [src/sensory/__init__.py](src/sensory/__init__.py), [src/data_integration/__init__.py](src/data_integration/__init__.py).

Migration order (low risk to higher)
1) Canonicalize “infrastructure primitives”:
   - EventBus, get_global_cache
2) Core market models:
   - Instrument, Position, OrderStatus, OrderBookSnapshot/Level
3) Risk layer:
   - RiskManager, RiskConfig (move config to config/risk/)
4) Validation:
   - ValidationResult and Phase2DIntegrationValidator
5) Sensory tidy-up:
   - Remove type implementations from sensory __init__.py; re-export only
6) Learning/memory:
   - RealTimeLearningEngine, FAISSPatternMemory, MemoryEntry
7) Thinking/Intelligence/Ecosystem consolidation:
   - PredictiveMarketModeler(+Scenario/Generator), EcosystemOptimizer family
8) CurrencyConverter:
   - Move to core finance

Testing and verification
- After each batch:
  - Repo-wide import graph check (no circulars introduced)
  - Run unit/integration tests targeting touched domains
  - Grep-based audit for legacy imports to ensure shims working
- Artifacts to update:
  - [docs/reports/CLEANUP_REPORT.md](docs/reports/CLEANUP_REPORT.md) append a “Resolved Duplicates” table listing concept, canonical path, legacy shims removed/pending.

Rollback plan
- Shims allow quick rollback: if breakage occurs, revert changed call sites; keep canonical unchanged.
- Maintain a migration branch per batch; merge only with green tests.

Deliverables
- Phase 1 PR set produces:
  - Canonical modules created/moved (as above)
  - Legacy modules reduced to shims
  - Updated import paths in modified packages
  - Updated documentation:
    - This plan ([docs/reports/CANONICALIZATION_PLAN.md](docs/reports/CANONICALIZATION_PLAN.md))
    - Deprecation/compat policy ([docs/reports/DEPRECATION_POLICY.md](docs/reports/DEPRECATION_POLICY.md))
    - Migration steps ([docs/reports/MIGRATION_PLAN.md](docs/reports/MIGRATION_PLAN.md))

Open questions (to resolve before Phase 2)
- Whether to relocate MarketDataGenerator into [src/simulation/market/](src/simulation/market/).
- Whether to unify StrategyTester directly under trading or maintain a minimal adapter in thinking/intelligence.
- Exact location for RiskConfig: [src/config/risk/risk_config.py](src/config/risk/risk_config.py) vs re-using an existing config file in [src/config/risk/](src/config/risk/).

Appendix — families with count ≥ 3 (current scan)
- RiskManager → canonical [src/core/risk/manager.py](src/core/risk/manager.py)
- ValidationResult → canonical [src/validation/models.py](src/validation/models.py)
- EcosystemOptimizer → canonical [src/ecosystem/optimization/ecosystem_optimizer.py](src/ecosystem/optimization/ecosystem_optimizer.py)
- PredictiveMarketModeler → canonical [src/thinking/prediction/predictive_market_modeler.py](src/thinking/prediction/predictive_market_modeler.py)
- RealTimeLearningEngine → canonical [src/sentient/learning/real_time_learning_engine.py](src/sentient/learning/real_time_learning_engine.py)
- StrategyTester → canonical [src/trading/strategy_engine/testing/strategy_tester.py](src/trading/strategy_engine/testing/strategy_tester.py)
- OrderBookSnapshot → canonical [src/trading/order_management/order_book/snapshot.py](src/trading/order_management/order_book/snapshot.py)
- FAISSPatternMemory, MemoryEntry → canonical [src/sentient/memory/faiss_pattern_memory.py](src/sentient/memory/faiss_pattern_memory.py)
