# Migration Plan — Phase 1

Purpose
- Execute structural unification of duplicate definitions discovered by the scanner with minimal runtime risk.
- Apply canonicalization decisions and introduce temporary re-export shims, then migrate imports batch-by-batch.

Inputs
- Scanner and outputs:
  - [scripts/cleanup/duplicate_map.py](scripts/cleanup/duplicate_map.py)
  - [docs/reports/duplicate_map_classes.csv](docs/reports/duplicate_map_classes.csv)
  - [docs/reports/duplicate_map_functions.csv](docs/reports/duplicate_map_functions.csv)
  - [docs/reports/duplicate_map.json](docs/reports/duplicate_map.json)
- Canonical targets:
  - [docs/reports/CANONICALIZATION_PLAN.md](docs/reports/CANONICALIZATION_PLAN.md)
- Policy:
  - [docs/reports/DEPRECATION_POLICY.md](docs/reports/DEPRECATION_POLICY.md)

Pre-requisites
- CI passes on current main.
- Branch created: feature/canonicalization-phase1
- Agree on canonical module paths per family as documented in [docs/reports/CANONICALIZATION_PLAN.md](docs/reports/CANONICALIZATION_PLAN.md)

General procedure per batch
1) Create or confirm canonical module exists and contains the only implementation.
2) Convert each legacy module into a thin re-export shim (no logic).
3) Adjust new/modified call sites to import from the canonical path.
4) Run tests for touched domains; run the duplicate scanner and compare counts.
5) Commit with a scoped message and update [docs/reports/CLEANUP_REPORT.md](docs/reports/CLEANUP_REPORT.md).

Commands (reference)
- Generate duplicate map after each batch:
  - python [duplicate_map.py](../../scripts/cleanup/duplicate_map.py) --root src --out docs/reports --min-count 2
- Optional check before commit: run the same command and ensure counts do not regress.

Shim template (Python)
```
# Legacy module (shim-only)
from canonical.package.module import Thing as Thing
__all__ = ["Thing"]
```

Batch plan

Batch 1 — Infrastructure primitives
- Families:
  - EventBus → canonical [src/core/event_bus.py](src/core/event_bus.py)
  - get_global_cache → canonical [src/core/performance/market_data_cache.py](src/core/performance/market_data_cache.py)
- Actions:
  - Ensure canonical modules hold the only implementations.
  - Convert [src/operational/event_bus.py](src/operational/event_bus.py) to a re-export shim.
  - Convert [src/performance/__init__.py](src/performance/__init__.py) to a re-export for get_global_cache.
- Tests:
  - Run unit/integration for core and operational packages.
  - Run scanner; expect duplicate groups for EventBus and get_global_cache to drop to 0.

Batch 2 — Core market models
- Families:
  - Instrument → canonical [src/core/instrument.py](src/core/instrument.py)
  - InstrumentProvider → canonical [src/core/instrument_provider.py](src/core/instrument_provider.py)
  - Position → canonical [src/trading/models/position.py](src/trading/models/position.py)
  - OrderStatus → canonical [src/trading/models/order.py](src/trading/models/order.py)
  - OrderBookSnapshot/OrderBookLevel → canonical [src/trading/order_management/order_book/snapshot.py](src/trading/order_management/order_book/snapshot.py)
- Actions:
  - Create [src/core/instrument_provider.py](src/core/instrument_provider.py) if missing; move Provider implementation here.
  - Convert [src/trading/models.py](src/trading/models.py) to re-export Position only.
  - Replace inline OrderStatus in [src/operational/icmarkets_robust_application.py](src/operational/icmarkets_robust_application.py) with import from trading.models.order.
  - Move order book types into snapshot.py and rewire imports in trading and sensory.
- Tests:
  - Run trading model tests; run scanner; confirm the above families disappear from duplicates.

Batch 3 — Risk layer
- Families:
  - RiskManager → canonical [src/core/risk/manager.py](src/core/risk/manager.py)
  - RiskConfig → canonical [src/config/risk/risk_config.py](src/config/risk/risk_config.py)
- Actions:
  - Convert [src/core/risk_manager.py](src/core/risk_manager.py) to re-export from core.risk.manager.
  - Legacy [src/risk.py] shim removed; ensure imports reference the package entrypoint.
  - Legacy [src/core.py] shim removed after RiskConfig relocation to [src/config/risk/risk_config.py](src/config/risk/risk_config.py).
- Tests:
  - Run risk unit tests; run scanner; verify RiskManager and RiskConfig no longer duplicated.

Batch 4 — Validation layer
- Families:
  - ValidationResult → canonical [src/validation/models.py](src/validation/models.py)
  - Phase2DIntegrationValidator → canonical [src/validation/phase2d_integration_validator.py](src/validation/phase2d_integration_validator.py)
- Actions:
  - Create [src/validation/models.py](src/validation/models.py) and move ValidationResult here.
  - Convert [src/data_integration/__init__.py](src/data_integration/__init__.py) to re-export.
  - Legacy shims [src/risk.py] and [src/phase2d_integration_validator.py] removed; no additional action required.
- Tests:
  - Run validation suite; run scanner; verify duplicates removed for these families.

Batch 5 — Sensory package cleanup
- Families:
  - WhatDimension, AnomalyDimension, ChaosDimension, SensorSignal, IntegratedSignal
- Actions:
  - Ensure implementations live in concrete modules (e.g., [src/sensory/organs/dimensions/anomaly_dimension.py](src/sensory/organs/dimensions/anomaly_dimension.py)).
  - Reduce [src/sensory/dimensions/__init__.py](src/sensory/dimensions/__init__.py) and [src/sensory/__init__.py](src/sensory/__init__.py) to re-exports only.
- Tests:
  - Run sensory tests; run scanner; verify no implementations remain in __init__.py.

Batch 6 — Learning/memory
- Families:
  - RealTimeLearningEngine, FAISSPatternMemory, MemoryEntry
- Actions:
  - Canonicalize under sentient: [src/sentient/learning/real_time_learning_engine.py](src/sentient/learning/real_time_learning_engine.py), [src/sentient/memory/faiss_pattern_memory.py](src/sentient/memory/faiss_pattern_memory.py)
  - Convert [src/thinking/memory/faiss_memory.py](src/thinking/memory/faiss_memory.py) and [src/thinking/memory/pattern_memory.py](src/thinking/memory/pattern_memory.py) to re-exports.
  - Convert [src/intelligence/sentient_adaptation.py](src/intelligence/sentient_adaptation.py) to import from sentient.*
- Tests:
  - Run sentient/thinking/intelligence tests; run scanner; confirm families resolved.

Batch 7 — Thinking/Intelligence/Ecosystem
- Families:
  - PredictiveMarketModeler (+ MarketScenario, MarketScenarioGenerator)
  - EcosystemOptimizer, CoordinationEngine, SpecializedPredatorEvolution, NicheDetector, SpeciesManager
- Actions:
  - Use thinking prediction package as canonical for predictive modeling: [src/thinking/prediction/predictive_market_modeler.py](src/thinking/prediction/predictive_market_modeler.py)
  - Canonicalize ecosystem under [src/ecosystem/](src/ecosystem/); convert thinking/intelligence modules to re-exports.
- Tests:
  - Run thinking/intelligence/ecosystem tests; run scanner; confirm duplicate groups drop.

Batch 8 — Financial utilities
- Families:
  - CurrencyConverter
- Actions:
  - Create [src/core/finance/currency_converter.py](src/core/finance/currency_converter.py) and move implementation.
  - Convert [src/core.py](src/core.py) and [src/domain/models.py](src/domain/models.py) to re-export/import from core.finance.
- Tests:
  - Run core/domain tests; run scanner; ensure duplicates removed.

Code search helpers
- Grep for import sites (examples):
  - EventBus: "from operational.event_bus import EventBus" and "from core.event_bus import EventBus"
  - RiskManager: "from core.risk_manager import RiskManager" and "from core.risk.manager import RiskManager"
  - ValidationResult: "from validation.validation_framework import ValidationResult"
  - OrderBookSnapshot: "OrderBookSnapshot"
- Prefer updating the import source rather than renaming the type; semantic changes are deferred to Phase 2.

Acceptance criteria per batch
- Duplicate scanner shows 0 duplicate groups for the batch families.
- Tests green in affected domains.
- No implementations remain in legacy modules; only re-export shims.
- Documentation updated (CLEANUP_REPORT append “Resolved Duplicates” entries).

Rollback strategy
- Revert call-site import changes while keeping canonical modules; shims ensure minimal disruption.
- Maintain atomic commits per batch to simplify rollback.

Post-Phase 1
- Remove shims (T2) after repo-wide import migration is completed.
- Introduce CI guard to prevent new duplicates and implementations inside __init__.py for canonicalized families.

Appendix — Known syntax issues surfaced by scanner (non-blocking for planning)
- [src/validation/phase2_validation_suite.py](src/validation/phase2_validation_suite.py)
- [src/sensory/organs/dimensions/real_sensory_organ.py](src/sensory/organs/dimensions/real_sensory_organ.py)
- [src/genome/models/genome.py](src/genome/models/genome.py)
- [src/ecosystem/evaluation/niche_detector.py](src/ecosystem/evaluation/niche_detector.py)
- These files raised SyntaxErrors during parsing; address as part of the respective batch before running unit tests.
