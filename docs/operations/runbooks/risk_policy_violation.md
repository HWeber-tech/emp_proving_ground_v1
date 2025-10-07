# Risk policy violation escalation runbook

This runbook explains how to triage and escalate trade intents that fail the
institutional risk policy enforced by the trading gateway.  It links the
runtime telemetry surfaces exposed by the trading manager to the recovery steps
operators must follow during a breach.

## Detection

1. Monitor the runtime event bus topic `telemetry.risk.policy_violation`.
   - Every rejection publishes a structured alert containing the serialized
     policy snapshot and a Markdown summary.【F:src/trading/risk/policy_telemetry.py†L214-L285】
   - The risk gateway emits the alert as soon as a policy decision is rejected,
     so manual workflows and FIX pilots inherit the same escalation signal as
     the trading manager.【F:src/trading/risk/risk_gateway.py†L231-L340】
   - The trading manager forwards alerts whenever a policy decision is rejected
     or recorded with outstanding violations.【F:src/trading/trading_manager.py†L920-L991】
2. Secondary confirmation is available on `telemetry.risk.policy`, which
   streams the latest decision snapshot for dashboards and notebooks.
3. The FIX broker adapter mirrors the last snapshot in the rejection payload
   so manual operators see the same telemetry surface as automated flows.【F:src/trading/integration/fix_broker_interface.py†L207-L254】

## Immediate actions

1. Pull the Markdown summary from the alert payload or the trading-manager
   logs to identify the primary violation (`policy.*` guardrail).
2. Inspect the runbook metadata embedded in the alert.  This file is the
   canonical escalation path—notify the governance desk and pause affected
   strategies if the violation is not a configuration error.
3. Review `risk_policy.limit_snapshot()` values in the payload to confirm the
   configured thresholds match the approved governance file.
4. If the alert originated from the FIX bridge, cross-check the `policy_snapshot`
   stored on the order record for parity with the runtime builder.

## Remediation steps

1. Correct invalid strategy configuration or position metadata if the
   violation highlights stale exposure data.  Re-run the policy evaluation via
   `TradingManager.on_trade_intent` in research mode to confirm the fix before
   resuming live trading.
2. If the policy violation is genuine (e.g., exposure or min size breach),
   escalate to the governance contact list and keep the offending strategy in
   a halted state until approvals arrive.
3. After remediation, verify that new policy snapshots publish with
   `approved=true` and no outstanding violations before clearing the alert.
4. File an incident report if the breach crossed governance thresholds or
   required manual intervention beyond configuration hygiene.

## References

- `RiskPolicy` evaluation logic and limit mapping: `src/trading/risk/risk_policy.py`
- Trading manager telemetry bridge: `src/trading/trading_manager.py`
- FIX bridge risk escalation: `src/trading/integration/fix_broker_interface.py`
- Policy telemetry publisher: `src/trading/risk/policy_telemetry.py`
