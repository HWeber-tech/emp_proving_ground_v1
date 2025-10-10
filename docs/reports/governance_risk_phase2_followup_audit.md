# Governance & Risk Audit – Phase II Follow-up

## Scope
- Re-run a focused review of the `risk.manager` facade to ensure configuration guards and sector budget enforcement are resilient.
- Spot-check `policy_ledger` and `safety_manager` governance pathways for regressions introduced after the previous Phase II audit.
- Capture coverage evidence so the roadmap objective (“≥85% coverage across governance/risk modules”) remains demonstrably satisfied.

## Findings & Actions
1. **Risk facade misconfiguration guardrails (high priority)**  
   Manual review confirmed the facade’s `_coerce_risk_config` helper was the final guard against malformed payloads delivered by external orchestration code. The branch raising `TypeError`/`ValueError` was previously untested, meaning accidental relaxation of these checks would go unnoticed. Added explicit regression coverage exercising all failure modes (`None`, non-mapping payloads, and invalid percentage values) so CI now fails if the facade stops surfacing these errors. 【F:tests/risk/test_risk_manager_facade.py†L1-L67】

2. **Sector budget enforcement (medium priority)**  
   Sector throttles depend on operators supplying both `instrument_sector_map` and matching `sector_exposure_limits`. Without direct tests, a refactor that skipped the budget check could silently re-open the risk of allocating beyond governance-approved limits. A new scenario forces a FX trade to exceed its sector budget and asserts the facade denies the order, preventing regressions in drawdown discipline. 【F:tests/risk/test_risk_manager_facade.py†L70-L87】

3. **Governance ledger & safety manager posture (informational)**  
   No new defects were observed in ledger staging or kill-switch handling. Existing suites (`tests/governance/test_policy_ledger*.py`, `tests/governance/test_security_phase0.py`) remain authoritative. Recommend continuing to run the shared governance suite in CI to guard the promotion/kill-switch workflow as the runtime evolves.

## Evidence & Coverage
- Pytest guardrail covering the new scenarios: `pytest tests/risk/test_risk_manager_facade.py -q` (pass). 【dc9c06†L1-L4】
- Line coverage snapshot for `risk.manager` captured via the standard library `trace` harness. Executed/total tracked lines: 72/72 → **100 %**. Raw counter artefact retained under `artifacts/audit_trace/risk.manager.cover` for future reviews. 【9d0175†L1-L16】

## Follow-up Recommendations
- Extend the coverage harness to aggregate `risk.*` and `governance.*` packages during nightly builds so the ≥85 % target is automatically enforced instead of ad-hoc spot checks.
- When the governance CLI gains additional live controls, replicate this audit pattern (misconfiguration tests + trace snapshot) to keep policy and safety modules equally protected.
