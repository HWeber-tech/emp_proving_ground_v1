# Risk & Governance Phase II Mini-Audit

**Date:** 2024-05-09
**Auditor:** Automation (AlphaTrade Roadmap Task)

## Scope

Focused review of the risk gateway liquidity checks and related governance telemetry paths to close out the Phase II roadmap requirement for a risk/governance audit. Primary artefacts inspected:

- `src/trading/risk/risk_gateway.py`
- `tests/current/test_risk_gateway_validation.py`
- `tests/trading/test_trading_manager_execution.py`

## Findings & Resolutions

### 1. Liquidity probe lost portfolio price context (Critical)
- **Observation:** `_run_liquidity_probe` pulled prices exclusively from the intent payload. When an intent lacked a price (common for router-originated intents) the helper defaulted to `0.0`, emitting liquidity probes at zero price levels. This silently weakened liquidity confidence scoring and could mask thin book conditions.
- **Resolution:** Thread the portfolio state and resolved trade price into `_run_liquidity_probe`. The helper now reuses `_resolve_trade_price`, ensuring probes inherit the latest portfolio snapshot pricing even when intents are sparse.
- **Regression Coverage:** Added `test_risk_gateway_liquidity_probe_uses_portfolio_price` to guarantee probes honour portfolio prices. Updated trading-manager integration coverage to exercise the new signature.

### 2. Telemetry completeness for injected probes (Improvement)
- **Observation:** Manual invocations of `_run_liquidity_probe` (used by trading-manager bootstraps) required explicit market context after the fix above.
- **Resolution:** Adjusted the integration regression in `tests/trading/test_trading_manager_execution.py` to provide deterministic prices, preserving existing governance telemetry assertions.

## Coverage Impact

- Risk gateway liquidity branch now covered by `tests/current/test_risk_gateway_validation.py::test_risk_gateway_liquidity_probe_uses_portfolio_price`.
- Trading manager liquidity telemetry regression updated to align with stricter risk gateway contract.

## Recommendations

- Extend the audit to portfolio risk manager concurrency in Phase III (no blocking issues detected here).
- Consider capturing probe price sources in telemetry payloads to aid future audits.

