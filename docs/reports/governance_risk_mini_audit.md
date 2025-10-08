# Governance & Risk Mini-Audit – 2025-10-08

## Quick Fixes
- Patched `SafetyManager.from_config` to honour mapping payloads so live-run safeguards obey configuration supplied by environment loaders without needing attribute adapters (see `src/governance/safety_manager.py`).
- Risk telemetry snapshots now stamp `generated_at` using `datetime.now(timezone.utc)` to avoid naive timestamps leaking into downstream markdown/JSON pipelines that expect timezone-aware values (`src/risk/telemetry.py`).

## Observations & Follow-Ups
- The policy ledger persists to JSON without file locking; simultaneous writers could clobber state. Consider lightweight file locking or transactional writes before enabling multi-process governance tooling.
- Risk telemetry coverage is strong, but follow-up integration tests exercising policy-ledger-driven throttle changes would improve confidence in promotion gating under live data loads.

## Validation
- `pytest tests/governance/test_security_phase0.py tests/risk/test_risk_telemetry.py -q`
