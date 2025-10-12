# Policy & Code Audit — Phase II Completion (Fail-Closed Refresh)

## Scope
- Revalidated governance safety surfaces: `src/governance/safety_manager.py`, `src/governance/policy_ledger.py`, and promotion tooling for regression drift.
- Deep audit of risk orchestration: `RiskManagerImpl`, facade, and associated telemetry to ensure runtime checks fail closed under fault conditions.
- Verification by targeted regression suites plus full `pytest tests/governance tests/risk --cov=src/governance --cov=src/risk` coverage run.

## Findings & Remediation
1. **High — Portfolio risk fails open on sanitisation error** (`src/risk/risk_manager_impl.py:845`)
   - Impact: When callers supplied non-numeric position weights (or the underlying risk engine raised), `evaluate_portfolio_risk` returned `0.0`, signalling "no risk" to governance surfaces. The trading gateway relies on `>=1.0` to block orders, so the bug allowed intents to proceed even though the portfolio snapshot was invalid.
   - Fix: Hardened normalisation to log and return `1.0` on conversion failures or downstream engine exceptions, guaranteeing fail-closed behaviour and surfacing root cause telemetry. The defensive branch is covered by the new regression guard at `tests/risk/test_risk_manager_impl_additional.py:531`.
   - Validation: Added `tests/risk/test_risk_manager_impl_additional.py:539` to simulate a real risk engine crash and confirm the breach signal, and aligned the legacy regression in `tests/current/test_risk_manager_impl.py:95` to reflect the fail-closed contract.

2. **Medium — Kill-switch readability still treated as soft warning** (`src/governance/safety_manager.py:63`)
   - Observation: The safety manager logs and proceeds when the filesystem refuses kill-switch inspection. Given the latest fault-injection coverage, we kept the current behaviour but documented the residual risk here; recommend pairing deployments with infrastructure health checks to guarantee the path remains accessible.

## Validation & Coverage
- Commands executed:
  - `pytest tests/risk/test_risk_manager_impl_additional.py tests/current/test_risk_manager_impl.py`
  - `pytest tests/governance tests/risk --cov=src/governance --cov=src/risk --cov-report term-missing`
- Resulting coverage: 90.46% combined (risk) / governance suites unchanged; `RiskManagerImpl` branches covering the new fail-closed logic exercise both sanitisation and downstream-failure paths.

## Follow-ups
- Consider mirroring the fail-closed philosophy inside `RiskManagerImpl.assess_risk` for telemetry pathways once consumers confirm they never expect "best-effort" scoring.
- Expand operations runbook to include filesystem health probes for the governance kill-switch to mitigate the residual risk noted above.
