# Governance & Risk Mini-Audit – 2025-10-08

## Audit Snapshot – 2025-10-09

### Quick Fixes
- Hardened `PolicyLedgerStore` writes with filesystem locks and atomic temp-file swaps so concurrent promotion artifacts no longer clobber each other (`src/governance/policy_ledger.py`).
- Updated `RiskManagerImpl.calculate_position_size` to fall back to the configured minimum size instead of a hard-coded value, preserving operator safety when derivatives of the signal pipeline misbehave (`src/risk/risk_manager_impl.py`).

### Findings & Follow-ups
- Governance ledger concurrency: prior implementation rewrote the ledger directly and could lose records when multiple supervisors promoted tactics. The new lock/atomic-write path eliminates partial writes; a follow-up backlog item should add process-level telemetry so stale locks can be surfaced to operators rather than silently cleared.
- Risk facade coverage: synchronous `RiskManager` now has guardrail tests for every branch (budget depletion, aggregate risk, sector budgets). Coverage rose to 93% for `src/risk/manager.py` and 83% for `src/governance/policy_ledger.py`. The sprawling `RiskManagerImpl` remains at ~63% because VaR/ES and regime helpers still rely on complex fixtures; recommend carving out injectable seams so we can raise its coverage above 80% without brittle fixtures.
- Sector gating logic now exercised via facade tests, but the asynchronous pipeline still shares internal mutable state. Consider introducing defensive copies (or locks) if background tasks start mutating `positions` concurrently.

### Validation
- `pytest --maxfail=1 --disable-warnings --color=no --cov=src/governance --cov=src/risk tests/governance tests/risk`
- `coverage report -m --include=src/governance/policy_ledger.py`
- `coverage report -m | head` (risk module heat-map)

## Quick Fixes
- Patched `SafetyManager.from_config` to honour mapping payloads so live-run safeguards obey configuration supplied by environment loaders without needing attribute adapters (see `src/governance/safety_manager.py`).
- Risk telemetry snapshots now stamp `generated_at` using `datetime.now(timezone.utc)` to avoid naive timestamps leaking into downstream markdown/JSON pipelines that expect timezone-aware values (`src/risk/telemetry.py`).

## Observations & Follow-Ups
- The policy ledger persists to JSON without file locking; simultaneous writers could clobber state. Consider lightweight file locking or transactional writes before enabling multi-process governance tooling.
- Risk telemetry coverage is strong, but follow-up integration tests exercising policy-ledger-driven throttle changes would improve confidence in promotion gating under live data loads.

## Validation
- `pytest tests/governance/test_security_phase0.py tests/risk/test_risk_telemetry.py -q`
