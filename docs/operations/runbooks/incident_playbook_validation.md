# Incident playbook validation drill

This runbook documents the combined drill that proves the institutional incident
playbook covers three critical recovery paths: kill-switch engagement, nightly
replay evidence capture, and trade throttle rollback. The bundled CLI produces a
timestamped artifact pack so the operational readiness context packs can absorb
fresh evidence without chasing ad-hoc logs.

## Run the bundled validator

1. From the repository root execute `python -m tools.operations.incident_playbook_validation`.
   - Optional flags:
     - `--run-root` to direct artifacts somewhere other than `artifacts/incident_playbook`.
     - `--timestamp` (format `YYYYMMDDTHHMMSSZ`) to line the drill up with a
       broader incident rehearsal.
   - The CLI orchestrates each drill and emits JSON artifacts plus a consolidated
     summary.【F:tools/operations/incident_playbook_validation.py†L185】【F:tools/operations/incident_playbook_validation.py†L208】

2. Inspect the run directory (for example
   `artifacts/incident_playbook/20250102T030405Z/`). You should see:
   - `kill_switch.json`
   - `nightly_replay.json`
   - `trade_rollback.json`
   - `incident_playbook_summary.json`
   Each file records the drill status, key metrics, and any error context for audit trails.【F:tools/operations/incident_playbook_validation.py†L224】【F:tools/operations/incident_playbook_validation.py†L243】

## Drill details

### Kill-switch enforcement

- The validator touches a temporary sentinel file, reruns the `SafetyManager`
  guard, and expects a hard abort when the file is present, proving the
  kill-switch wiring is live.【F:tools/operations/incident_playbook_validation.py†L44】【F:tools/operations/incident_playbook_validation.py†L50】【F:tools/operations/incident_playbook_validation.py†L70】
- `SafetyManager.enforce` blocks live mode without confirmation and aborts when
  the kill-switch path exists, matching the institutional gate strategy.【F:src/governance/safety_manager.py†L21】【F:src/governance/safety_manager.py†L108】
- Regression coverage verifies both confirmation coercion and kill-switch file
  handling so the drill mirrors production safeguards.【F:tests/governance/test_safety_manager.py†L13】【F:tests/governance/test_safety_manager.py†L55】

### Nightly replay evidence bundle

- The validator delegates to the nightly replay job which generates a replay
  dataset, evaluation summary, drift report, decision diary, and policy
  ledger entry tied to the run ID.【F:tools/operations/incident_playbook_validation.py†L79】【F:tools/operations/incident_playbook_validation.py†L94】【F:tools/operations/incident_playbook_validation.py†L113】【F:tools/operations/nightly_replay_job.py†L1】【F:tools/operations/nightly_replay_job.py†L71】【F:tools/operations/nightly_replay_job.py†L155】
- Automated coverage asserts the replay job produces both the dataset and the
  linked governance artifacts that the incident review relies on when replaying a
  timeline.【F:tests/tools/test_nightly_replay_job.py†L9】【F:tests/tools/test_nightly_replay_job.py†L21】【F:tests/tools/test_nightly_replay_job.py†L30】

### Trade throttle rollback

- The validator exercises `TradeThrottle` by allowing two trades, observing the
  third get blocked, rolling back the second, and confirming capacity is freed
  for a retry.【F:tools/operations/incident_playbook_validation.py†L123】【F:tools/operations/incident_playbook_validation.py†L141】【F:tools/operations/incident_playbook_validation.py†L145】【F:tools/operations/incident_playbook_validation.py†L150】
- The throttle implementation tracks window counts, notional budgets, and
  rollback semantics for institutional compliance, so the drill validates the
  operational rollback path without touching live infrastructure.【F:src/trading/execution/trade_throttle.py†L1】【F:src/trading/execution/trade_throttle.py†L172】【F:src/trading/execution/trade_throttle.py†L381】
- Trade throttle tests document the rollback contract and ensure manager wiring
  restores capacity after failed executions.【F:tests/trading/test_trade_throttle.py†L160】【F:tests/trading/test_trade_throttle.py†L417】

## Evidence capture checklist

- Confirm `incident_playbook_summary.json` reports `status: "passed"` for each
  drill alongside the run identifier; attach the summary in the operational
  readiness context pack entry for the rehearsal.【F:tests/tools/test_incident_playbook_validation.py†L9】【F:tests/tools/test_incident_playbook_validation.py†L32】【F:tests/tools/test_incident_playbook_validation.py†L41】
- Archive the full run directory with the incident review packet so the
  governance ledger, drift report, and diary snapshots remain discoverable.
- If any drill fails, rerun with `--log-level DEBUG` and capture the JSON plus
  logs as part of the incident review before opening remediation tickets.
