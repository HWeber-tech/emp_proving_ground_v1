# Manual FIX order risk block

Manual FIX pilots surface risk enforcement failures via `telemetry.risk.intent_rejected`
from the broker interface. Use this runbook to triage policy violations or missing risk
metadata when a manual order is blocked before reaching execution.

## Detection

* `FIXBrokerInterface` emits a payload with `reason`, `policy_snapshot`, and
  `policy_violation` flags whenever the risk gateway denies an order. The payload includes
  a `runbook` pointing here plus a `risk_reference` section with the deterministic risk API
  runbook and the currently enforced limits.【F:src/trading/integration/fix_broker_interface.py†L214-L275】
* Trading telemetry dashboards show a critical alert with the same metadata, including the
  latest risk configuration summary resolved from the risk gateway.【F:src/trading/integration/fix_broker_interface.py†L214-L275】
* Policy violation alerts reference the governance playbook at
  `docs/operations/runbooks/risk_policy_violation.md`. Follow that runbook if the snapshot
  indicates a broken guardrail rather than a missing configuration.【F:src/trading/integration/fix_broker_interface.py†L214-L275】

## Immediate response

1. Capture the emitted payload from the event bus and attach it to the incident ticket.
   Confirm the symbol, side, and quantity match the trader request.
2. Review `risk_reference.risk_config_summary` to ensure the mandatory limits (risk per
   trade, exposure caps, and stop-loss posture) align with the intended manual override.
3. If `policy_snapshot` lists violations, execute the remediation steps in the risk policy
   violation runbook before reattempting the order.
4. When the violation stems from configuration drift (e.g., disabled stop loss outside
   research mode), escalate through the deterministic risk API contract runbook referenced
   in `risk_reference.risk_api_runbook`.

## Escalation

* Escalate to the Execution & Risk squad when the deterministic risk summary does not match
  the approved posture or when governance approvals are missing. Provide the captured
  payload, including the `risk_reference` metadata and decision snapshot.
* Log the incident in the operational readiness tracker and link both this runbook and the
  risk API contract runbook to the remediation work so institutional pilots inherit a
  documented trail.
