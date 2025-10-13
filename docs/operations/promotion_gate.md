# Promotion Gate

Operators promote AlphaTrade policies only when DecisionDiary telemetry and the
policy ledger show the tactic is ready for the next capital tier. The promotion
gate combines diary quality bars, governance approvals, and artifact
production so compliance inherits a deterministic audit trail.【F:src/governance/policy_ledger.py†L55-L216】【F:src/governance/policy_graduation.py†L1-L238】

## Governance inputs

* **Ledger stage & history** – `PolicyLedgerRecord` captures the current stage,
  approvals, evidence ID, optional policy deltas, and prior stage transitions
  with timestamps. Audit gaps surface automatically when evidence or approvals
  are missing for paper, pilot, or limited live promotions.【F:src/governance/policy_ledger.py†L107-L216】
* **DecisionDiary telemetry** – `PolicyGraduationEvaluator` aggregates forced
  decisions, severity mix, stage-specific counts, and consecutive "normal"
  streaks from the diary store to decide whether the paper, pilot, or limited
  live thresholds are satisfied.【F:src/governance/policy_graduation.py†L167-L340】
* **Regime coverage (optional)** – `PromotionGuard` enforces required regime
  labels and minimum decision counts when `PROMOTION_REQUIRED_REGIMES` and
  `PROMOTION_MIN_REGIME_COUNT` are set, blocking strategy status changes that
  would bypass diary evidence.【F:src/governance/promotion_integrity.py†L47-L165】【F:src/governance/strategy_registry.py†L33-L358】

## Stage thresholds

The evaluator promotes sequentially; any blocker keeps the recommendation at
its current stage and reports the failing checks.【F:src/governance/policy_graduation.py†L344-L460】

| Gate | Decision sample | Quality bars | Additional blockers |
| --- | --- | --- | --- |
| Experiment → Paper | ≥ 20 experiment-stage decisions | No alerts, forced_ratio ≤ 0.35, warn_ratio ≤ 0.40, normal_ratio ≥ 0.50 | — |
| Paper → Pilot | ≥ 40 paper-stage decisions | No alerts, forced_ratio ≤ 0.25, warn_ratio ≤ 0.20, normal_ratio ≥ 0.65, ≥ 8 consecutive "normal" decisions if still in paper | — |
| Pilot → Limited Live | ≥ 60 pilot-stage decisions | No alerts, forced_ratio ≤ 0.10, warn_ratio ≤ 0.10, normal_ratio ≥ 0.80, ≥ 15 consecutive "normal" decisions if still in pilot | At least two distinct approvals and zero audit gaps in the ledger |

The evaluator also reports global metrics (total decisions, forced ratio,
latest stage) so governance reviewers can spot deteriorating quality before a
promotion attempt.【F:tools/governance/alpha_trade_graduation.py†L119-L163】

## Guard configuration and enforcement

`StrategyRegistry` bootstraps a `PromotionGuard` so runtime status changes
cannot outrun the governance posture.【F:src/governance/strategy_registry.py†L45-L224】【F:src/governance/promotion_integrity.py†L47-L165】

* **Default config** – `config/governance/promotion_guard.yaml` declares the
  canonical ledger (`artifacts/governance/policy_ledger.json`) and diary
  (`artifacts/governance/decision_diary.json`) locations, maps strategy status
  `approved` → ledger stage `paper` and `active` → `limited_live`, and requires
  regime coverage for (`balanced`, `bullish`, `bearish`, `dislocated`) with at
  least three decisions per regime before the status change succeeds.
* **Environment overrides** – Operators may point the guard at alternative
  artefacts or adjust coverage by setting `POLICY_LEDGER_PATH`,
  `DECISION_DIARY_PATH`, `PROMOTION_REQUIRED_REGIMES`,
  `PROMOTION_MIN_REGIME_COUNT`, or a bespoke `PROMOTION_GUARD_CONFIG`; the
  guard normalises casing and rejects empty overrides.
* **Status gating** – Any attempt to mark a strategy `approved`/`active` when
  ledger stages lag, audit gaps exist, diary evidence is missing, or regime
  coverage falls short raises `PromotionIntegrityError` with a descriptive
  blocker so automation and operators can remedy the gap before retrying.

## Required artifacts & deliverables

* Updated ledger record under `artifacts/governance/policy_ledger.json` with
  the target stage, approvals, evidence ID, and any policy delta payloads.
* DecisionDiary evidence (`evidence_id`) that resolves to replay artifacts,
  narration, and telemetry backing the promotion.
* Governance approvals (`risk`, `compliance`, etc.) recorded in the ledger.
* Promotion log entry appended to `artifacts/governance/policy_promotions.log`
  with the ledger posture, captured automatically by the CLIs.【F:tools/governance/_promotion_helpers.py†L13-L108】
* Optional Markdown summary (`--summary-path`) for release notebooks and the
  regenerated policy/guardrail bundle produced by `rebuild_policy.py`.

## Operating procedure

1. **Evaluate readiness** – Run the graduation assessor to surface blockers and
   confirm diary quality bars:
   ```bash
   poetry run python -m tools.governance.alpha_trade_graduation \
     --ledger artifacts/governance/policy_ledger.json \
     --diary artifacts/governance/decision_diary.json \
     --policy-id alpha --json
   ```
   The command emits blocker keys aligned with the table above; address them
   before promoting or use `--hours` to inspect recent windows.【F:tools/governance/alpha_trade_graduation.py†L34-L282】
2. **Stage the promotion** – Apply the promotion once blockers clear:
   ```bash
   poetry run python -m tools.governance.promote_policy \
     --policy-id alpha --stage pilot \
     --approval risk --approval compliance \
     --evidence-id dd-alpha-0425 \
     --summary-path artifacts/governance/promotions/alpha-pilot.md
   ```
   The CLI normalises approvals/evidence, updates the ledger, writes the
   promotion log, and can emit a Markdown summary for compliance records.【F:tools/governance/promote_policy.py†L26-L146】【F:tools/governance/_promotion_helpers.py†L13-L108】
3. **Regenerate artifacts** – Rebuild enforceable payloads and update the
   governance changelog when the promotion lands:
   ```bash
   poetry run python -m tools.governance.rebuild_policy \
     --ledger artifacts/governance/policy_ledger.json \
     --changelog artifacts/governance/policy_changelog.md
   ```
   This recreates risk/guardrail bundles and publishes an updated governance
   changelog linked back to the promotion runbook.【F:tools/governance/rebuild_policy.py†L1-L147】

## Automated enforcement

* `StrategyRegistry.update_strategy_status` blocks status changes to `approved`
  or `active` when the ledger stage, evidence, approvals, or regime coverage
  fall short of the configured gate, returning actionable error messages to the
  caller.【F:src/governance/strategy_registry.py†L347-L375】【F:src/governance/promotion_integrity.py†L107-L141】
* Governance automation can run `alpha_trade_graduation.py --apply` to advance
  stages when blockers are empty; each applied promotion still writes to the log
  and ledger history for audits.【F:tools/governance/alpha_trade_graduation.py†L167-L279】

## When thresholds fail

Blockers are emitted in the format `context_key:value` so the remediation is
obvious (e.g., `pilot_forced_ratio_exceeds:0.18>0.10`). Address quality issues
in simulation or paper trading, collect additional DecisionDiary evidence, and
re-run the assessor. Use the policy promotion governance runbook for escalation
steps when evidence or approvals cannot be produced in time.
