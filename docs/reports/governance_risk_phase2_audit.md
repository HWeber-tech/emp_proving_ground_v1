# Governance & Risk Audit Snapshot – Phase II

## Scope
- Review critical governance modules (`policy_ledger`, `safety_manager`) for correctness and operational guardrails.
- Inspect risk gating flows in `RiskManagerImpl`/`RiskManager` for concurrency or data hygiene issues that could bypass limits.
- Provide quick fixes for any critical findings and document remediation status.

## Key Findings
1. **Risk symbol normalisation gap (critical)**  
   Risk positions were stored under whichever symbol casing the caller supplied. When downstream components updated prices or validated new trades using a different case (e.g., `eurusd` vs `EURUSD`), the manager treated them as independent exposures. This allowed stale position data to linger, skewing aggregate risk calculations and potentially bypassing sector throttles.  
   **Fix:** Centralised canonical symbol handling in `RiskManagerImpl`/`RiskManager` with uppercase keys, automatic migration of existing entries, and aggregation helpers that deduplicate symbol exposure prior to calling the core risk engine. Added regression tests covering live updates, legacy state migration, and risk validation to prevent regressions.

2. **Governance ledger & safety manager review (informational)**  
   The policy ledger already applies file locks with stale-lock eviction and enforces reviewer/evidence metadata during promotions. Safety manager kill-switch and live-mode checks behave as expected under simulated I/O failures. No critical issues identified; recommend retaining the current locking contract for Phase III and extending diary evidence validation when real data feeds arrive.

## Follow-up Recommendations
- Extend governance unit tests with a simulated stale-lock eviction path to ensure interoperability with shared storage in Phase III.
- Introduce integration coverage that exercises the new symbol-normalisation path end-to-end (router → execution → risk) once live feed plumbing is available, ensuring case hygiene is preserved across service boundaries.

## Test & Coverage Updates
- Added risk regression coverage in `tests/risk/test_risk_manager_impl_additional.py` to verify canonical symbol storage, backwards-compatible migrations, and diary-friendly metrics output for mixed-case inputs.
- Existing governance guardrails remain green under the current pytest suite; no changes required this pass.

