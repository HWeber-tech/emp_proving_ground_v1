# Policy & Code Audit - Phase II Completion

## Scope
- src/risk: `risk_manager_impl`, `manager`, `telemetry`, `real_risk_manager`
- src/governance: `policy_ledger`, `safety_manager`, `strategy_registry`
- runtime supervision touchpoints exposed through `task_supervisor`

## Findings

### Critical Fixes
- **Short exposure risk capture:** `RiskManagerImpl` now measures absolute position size when aggregating exposure so short trades consume risk budget instead of silently falling to zero (see `src/risk/risk_manager_impl.py:236`).
- **Ledger payload validation:** `PolicyLedgerRecord.from_dict` now rejects payloads without identifiers and normalises reviewer approvals, preventing malformed ledger rows from bypassing governance checks (`src/governance/policy_ledger.py:217`).
- **Fail-closed market regime detection:** `RiskManagerImpl.update_market_regime` previously propagated detector exceptions, terminating the orchestration loop and leaving risk throttles unchanged. The handler now fails closed -- logging the fault, setting the multiplier to 0, marking telemetry as blocked, and surfacing the error for governance review (see `src/risk/risk_manager_impl.py:639`).

### Risks Accepted / Follow-ups
- **Telemetry debt:** Risk telemetry still exposes legacy fields without structured severities. Recommend migrating to structured `RiskTelemetrySnapshot` schemas so governance dashboards can differentiate soft vs hard failures.
- **Policy ledger metadata hygiene:** Promotions rely on operators supplying consistent `metadata` payloads; a schema-backed check (pydantic model or jsonschema) would prevent malformed entries. Tracked for Phase III backlog.
- **Coverage floor sensitivity:** Running the narrowed audit suite with `--cov` trips the global `fail-under` gate. Full-suite coverage still clears the bar, but the incident highlights that risk/governance modules remain sensitive to partial test runs. Recommend adding module-level coverage badges for quick visibility.

## Tests
- `pytest --maxfail=1 --disable-warnings tests/governance/test_policy_ledger.py tests/current/test_risk_manager_impl.py`
