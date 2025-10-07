# EMP Codebase Consolidation Plan (FIX‑only)

Date: 2025‑08‑11

Objective: Remove redundancies, unify scattered concepts, and lock the project to a single, coherent implementation per the FIX‑only policy.

Key signals (from latest cleanup report):
- EventBus appears in 7 files (multiple implementations)
- RiskConfig appears in 4 files; RiskManager in 3
- Strategy/engine duplicates across namespaces
- Large orphan/dead‑code surface (168 candidates)

Guiding constraints:
- FIX‑only broker connectivity; retire non‑FIX or OpenAPI shims
- Calculations belong in the sensory layer where applicable
- New reports saved to `docs/reports/` and mirrored to `reports/`

Target single sources of truth (SoT):
- Event system: keep `src/core/event_bus.py`; legacy `src/operational.event_bus` shim has been removed in favour of direct aliases that point to the canonical module.
- Trading models: keep `src/trading/models/order.py`, `position.py`, `trade.py`; deprecate `src/trading/models.py` duplicates.
- Execution: keep `src/trading/execution/execution_model.py` (slippage/fees) and integrate it into pre‑trade checks. Review `fix_executor.py`; either align with `FIXBrokerInterface` or move to legacy.
- FIX connectivity: keep `src/operational/fix_connection_manager.py` and `src/trading/integration/fix_broker_interface.py`. Legacy `icmarkets_robust_application.py` has already been removed.
- Risk: consolidate under `src/core/risk/` with one `RiskManager` and one `RiskConfig`. Retire `src/risk/risk_manager_impl.py` if not the SoT.
- Liquidity prober: refactor `src/trading/execution/liquidity_prober.py` to depend on `FIXBrokerInterface` (remove `mock_ctrader_interface` coupling) or move to `docs/legacy/` until refactored.
- Configs: prefer typed configs under `src/data_foundation/config/` and YAML under `config/`. Remove duplicate config modules under `src/config/`.

Phased execution (1‑week sweep):
1) Inventory and gates (Day 1)
   - Freeze baseline via `scripts/cleanup/generate_cleanup_report.py` and commit.
   - Add CI gate: fail if duplicates or dead‑code count grows.

2) EventBus unification (Day 1‑2)
   - Replace imports of non‑core EventBus with `src/core/event_bus.py`.
   - Delete/legacy‑move other EventBus implementations; add adapters if needed.

3) Trading models (Day 2)
   - Replace usage of `src/trading/models.py` with `src/trading/models/*` modules.
   - Remove `models.py` after references are gone.

4) FIX/Execution convergence (Day 2‑3)
   - Align `fix_executor.py` with `FIXBrokerInterface` or retire it.
   - Ensure `execution_model.py` is used in pre‑trade checks.

5) Risk consolidation (Day 3‑4)
   - Keep a single `RiskManager`+`RiskConfig` in `src/core/risk/`.
   - Migrate callers; remove duplicates elsewhere.

6) LiquidityProber refactor (Day 4‑5)
   - Replace `mock_ctrader_interface` dependency with a thin broker abstraction backed by `FIXBrokerInterface`.
   - Or move to `docs/legacy/` if deferred.

7) Config normalization (Day 5)
   - Remove duplicate config modules under `src/config/` after verifying no imports.

Acceptance criteria:
- No duplicate EventBus, RiskConfig/Manager, or Trading models in tree
- Dead‑code list reduced by ≥70%
- All tests remain green; new CI gate prevents regressions

Notes:
- This plan preserves the FIX‑only execution pathway and avoids introducing non‑FIX paths.
