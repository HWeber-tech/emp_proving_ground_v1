# Risk API contract violation

The runtime builder and trading manager raise a `RiskApiError` when a trading
manager fails to expose a canonical `RiskConfig`.  Use this runbook to restore
compliance.

## Detection

* Runtime startup logs include an error similar to `Trading manager risk
  configuration is invalid`.  The message references this runbook.
* `RuntimeApplication.summary()` contains a `risk_error` section with the
  failure metadata.
* `TradingManager.describe_risk_interface()` returns an `error` payload instead
  of the normal configuration summary.
* Risk gateway limit payloads expose a `risk_config_summary` and `runbook`
  field; if those entries are missing or reference stale metadata, the gateway
  failed to hydrate the deterministic risk API contract and the downstream
  trading stack should escalate using this runbook.【F:src/trading/risk/risk_gateway.py†L371-L389】【F:tests/current/test_risk_gateway_validation.py†L326-L350】

## Immediate response

1. Identify the failing trading manager instance from the error details.
2. Confirm that the manager initialised `_risk_config` with a valid
   `RiskConfig` or exposes a compliant payload from `get_risk_status()`.
3. If initialisation relies on configuration files, validate the numeric ranges
   (`max_risk_per_trade_pct`, `max_total_exposure_pct`, and `mandatory_stop_loss`).
4. Restart the runtime once the configuration is corrected and verify that the
   summary payload returns without the `error` field.

## Escalation

* If the manager cannot provide a valid configuration, escalate to the Execution
  & Risk squad with the captured `risk_error` metadata.
* Record the incident in the operational readiness log and link this runbook to
  the remediation ticket.
