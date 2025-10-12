# Policy & Code Audit – Phase II Follow-up

## Scope
- `src/governance/policy_ledger.py`
- `src/risk/risk_manager_impl.py`
- Regression tests covering governance ledger promotion flows and risk flag overrides

## Findings & Fixes

### Duplicate Governance Approvals Persisted
- **Impact:** Policy ledger stage transitions accepted duplicate reviewer identifiers, inflating history audit entries and misreporting approval quorum. Downstream governance dashboards treat approval counts as quorum checks; duplicate values mask missing reviewers.
- **Fix:** Stage promotions now normalise reviewer approvals during `PolicyLedgerRecord.with_stage`, ensuring the persisted tuple and the history trail store de-duplicated, sorted values (`src/governance/policy_ledger.py:136`). Added regression coverage in `tests/governance/test_policy_ledger.py::test_policy_ledger_deduplicates_approvals_on_update`.

### Misinterpreted Risk Flag Overrides
- **Impact:** `RiskManagerImpl.update_limits` cast string overrides (e.g., `"false"`) to `bool`, which always evaluates truthy. Remote configuration payloads therefore re-enabled mandatory stop-loss enforcement and disabled research mode fail-safes unexpectedly, triggering unnecessary trade rejections.
- **Fix:** Added `_coerce_flag` helper to interpret heterogeneous boolean payloads with validation, logging, and graceful fallback. Risk overrides now respect string/int inputs (`src/risk/risk_manager_impl.py:27`, `src/risk/risk_manager_impl.py:952`). New behavioural coverage via `tests/current/test_risk_manager_impl.py::test_update_limits_coerces_boolean_strings` and `tests/current/test_risk_manager_impl.py::test_update_limits_enables_research_mode_via_string`.

## Tests
- `pytest tests/current/test_risk_manager_impl.py tests/governance/test_policy_ledger.py`
