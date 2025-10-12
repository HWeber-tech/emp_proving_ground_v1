# Risk & Governance Phase II Code/Policy Audit

## Scope
- Modules reviewed: `src/trading/risk/risk_gateway.py`, `src/governance/safety_manager.py`, `src/trading/risk/risk_policy.py`, and supporting tests.
- Focus areas: enforcement logic, sector limit coverage, configuration hygiene, and telemetry provenance for governance evidence packs.

## Findings
1. **Sector Limit Enforcement Gap**  
   - **Observation:** Sector guardrails only applied when `RiskConfig.instrument_sector_map` contained an entry. Trade intents supplying sector metadata (common in catalogued tactics) bypassed the limit, and portfolio state holdings with sector annotations were ignored.  
   - **Resolution:** Risk gateway now normalises sector hints from trade intents and open-position metadata before applying sector caps. Missing sector hints are surfaced as informational checks so governance reviews can spot unmapped instruments. Regression coverage exercises both the new fallback and the informational path.

2. **Safety Manager Configuration Review**  
   - **Observation:** No blocking defects; normalisation helpers reject malformed confirmation payloads as expected. Added guidance to continue capturing run-mode telemetry during future phases.

3. **Policy Ledger Spot Check**  
   - **Observation:** Stage progression guards reject regressions and preserve evidence history. No remedial work required this phase.

## Follow-Up Recommendations
- Extend governance runbooks to include a sector-mapping validation checklist before enabling new instruments.
- Monitor the informational `risk.sector_limit.unmapped` check in dashboards; persistent entries should trigger either config updates or tactic suppression.

## Evidence
- Code changes: `src/trading/risk/risk_gateway.py` sector hint resolution, informational telemetry, and exposure aggregation updates.
- Tests: `tests/current/test_risk_gateway_validation.py` covers intent metadata fallback and unmapped sector audit trail.
