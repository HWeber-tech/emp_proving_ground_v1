# Policy & Code Audit — Phase II Completion

## Scope
- Governance and risk subsystems with emphasis on runtime safety gates and telemetry.
- Focused review of `src/risk` (manager facade/impl, real risk engine, telemetry) and supporting governance kill-switch tooling.
- Validation covered regression tests (`pytest tests/governance tests/risk --cov=src/governance --cov=src/risk`) and branch-aware coverage analysis.

## Findings & Actions
1. **Fail-open position sizing bug** (`src/risk/risk_manager_impl.py:519`)
   - Observed that `calculate_position_size` returned the configured minimum size when an internal error occurred, which could schedule unintended trades under uncertain state.
   - **Action:** Updated the exception path to return `0.0`, ensuring the sizing routine fails closed and logs the error.
   - **Verification:** Added `tests/risk/test_risk_manager_impl_additional.py::test_calculate_position_size_exception_returns_zero`.

2. **Risk posture telemetry blind spots** (`src/risk/telemetry.py`)
   - Several helper paths (minimum threshold grading, exposure fallbacks, decision sanitisation) lacked regression coverage, risking undetected regressions in governance dashboards.
   - **Action:** Added comprehensive unit coverage for telemetry helpers and markdown generation in `tests/risk/test_risk_telemetry.py` to lock behaviour around minimum/maximum guards, exposure computation, decision extraction, and markdown serialization.

3. **Async validation guardrails** (`src/risk/risk_manager_impl.py`) 
   - Branch coverage gaps masked scenarios where `validate_position` could bypass limits (non-positive inputs, exhausted budgets, aggregate risk overrides).
   - **Action:** Added async regression tests validating each rejection path and the happy path, plus non-positive limit overrides in `tests/risk/test_risk_manager_impl_additional.py`.

4. **Real risk engine normalisation** (`src/risk/real_risk_manager.py`)
   - Normalisation branches for negative config values were untested, risking silent regressions in drawdown/leverage defaults.
   - **Action:** Introduced `tests/risk/test_real_risk_manager.py` covering config coercion, budget fallbacks, and resilience to non-finite exposures.

## Coverage Improvements
- Risk & governance coverage now 90.37% (branch-aware), up from 75.51% prior to remediation.
- `src/risk/risk_manager_impl.py`: 88.21% coverage (improved >18pp).
- `src/risk/telemetry.py`: 91.64% coverage (improved >13pp).
- `src/risk/real_risk_manager.py`: 94.39% coverage (improved >15pp).

## Outstanding Follow-ups
- `RiskManagerImpl` still contains several defensive logging branches (sector snapshot warning paths) that remain unexercised; consider targeted fault-injection tests if we introduce simulated sector breaches.
- Governance documentation references for telemetry consumers should be refreshed to point at the new regression suite.
- Recommend scheduling a separate pass on runtime orchestration supervisors once task builder refactor lands (out of current scope).

*Audit executed on 2025-10-12 by Codex CLI assistant.*
