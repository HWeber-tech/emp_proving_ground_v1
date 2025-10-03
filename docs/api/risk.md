# Risk API

The trading runtime resolves every `RiskConfig` through
`src/trading/risk/risk_api.py`.  The module exposes helpers that give
supervisors and orchestration code a deterministic surface over the trading
manager's risk posture:

* `resolve_trading_risk_config` – returns a fully validated `RiskConfig`
  instance and raises `RiskApiError` if the manager exposes malformed data.
* `resolve_trading_risk_interface` – wraps the resolved configuration together
  with any status payload exposed by `TradingManager.get_risk_status()`.
* `build_runtime_risk_metadata` – converts the interface into a serialisable
  summary used by `RuntimeApplication` metadata and telemetry publishers.

`RiskApiError` now carries structured metadata and a default runbook link at
`docs/operations/runbooks/risk_api_contract.md`.  Callers should surface the
`runbook` URL in error messages and log the accompanying details so operators
can triage missing or invalid contracts quickly.

`summarise_risk_config` also renders sector exposure limits, combined sector
budget totals, and instrument-sector mappings when present so downstream
telemetry can display the enforced allocation posture alongside the numerical
limits without rehydrating the full configuration.【F:src/trading/risk/risk_api.py†L104-L128】【F:tests/trading/test_risk_api.py†L100-L142】
