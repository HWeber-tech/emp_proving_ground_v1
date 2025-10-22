# Audit Documentation Playbook

## Purpose and Scope
This playbook consolidates the operational procedures, artefacts, and validation steps used to satisfy internal and external audits for the EMP Proving Ground governance stack. It links each control to the source-of-truth implementation so reviewers can trace evidence directly to the codebase.

## Authoritative Evidence Sources

### Decision Diary Store
- Recorded by `DecisionDiaryStore`, which normalises policy decisions, market regimes, outcomes, and probe activations before persisting them to JSON and emitting governance telemetry.【F:src/understanding/decision_diary.py†L534-L623】【F:src/operations/event_bus_failover.py†L1-L73】
- Every entry carries a deterministic signature derived from the canonical payload. Probe definitions in the bundled registry are validated on load so historical reviews reference the same schema as the runtime.【F:src/understanding/decision_diary.py†L592-L621】
- Diary snapshots are archived automatically under `artifacts/diaries/<YYYY>/<MM>/<DD>/` via `archive_artifact`, ensuring auditors can replay prior states even after the live store rotates.【F:src/understanding/decision_diary.py†L604-L621】【F:src/artifacts/archive.py†L18-L94】

### Policy Ledger
- `PolicyLedgerRecord` tracks the governance posture for each tactic, including approvals, promotion history, human sign-offs, and an immutable signature for tamper detection.【F:src/governance/policy_ledger.py†L440-L519】
- `PolicyLedgerStore` enforces atomic writes with file locking, re-computes signatures on load, and archives each save under `artifacts/ledger_exports/` for traceable release snapshots.【F:src/governance/policy_ledger.py†L720-L815】
- `LedgerReleaseManager` exposes runtime checks for missing evidence or incomplete promotion checklists so auditors can confirm the execution layer respects governance gates.【F:src/governance/policy_ledger.py†L1040-L1109】

### Immutable Audit Signatures
- `compute_audit_signature` and `normalise_payload` hash the material fields of diary and ledger entries, optionally using an HMAC when `EMP_AUDIT_SIGNING_KEY` is configured. Auditors can regenerate signatures to prove artefacts are unchanged.【F:src/observability/immutable_audit.py†L1-L78】【F:src/observability/immutable_audit.py†L80-L115】

### Compliance Workflow Snapshots
- `ComplianceWorkflowSnapshot` aggregates per-regulation task checklists, highlighting outstanding remediation steps and providing Markdown exports for evidence packs.【F:src/compliance/workflow.py†L1-L108】【F:src/compliance/workflow.py†L132-L196】

## Pre-Audit Checklist
1. Confirm the latest decision diary JSON and policy ledger JSON are present under `artifacts/` (the nightly replay job publishes both when missing).【F:tools/operations/nightly_replay_job.py†L164-L200】【F:tools/operations/nightly_replay_job.py†L539-L596】
2. Ensure the compliance workflow snapshot for the audit period is captured (stored alongside nightly replay artefacts).
3. Retrieve the runtime configuration of enabled tactics or experiments referenced in the audit scope.

## Internal Audit Procedure
1. **Export decision evidence**
   - Run `poetry run python tools/understanding/decision_diary_cli.py export-diary --diary artifacts/understanding/decision_diary.json --format markdown --output artifacts/reports/decision_diary.md` to materialise reviewer-friendly entries. The CLI rehydrates probe metadata before export for context-rich evidence.【F:tools/understanding/decision_diary_cli.py†L1-L119】【F:tools/understanding/decision_diary_cli.py†L120-L199】
   - Optional: `poetry run python tools/understanding/decision_diary_cli.py summarise-reflection --diary ...` to generate tactic-level reflection digests used in governance meetings.【F:tools/understanding/decision_diary_cli.py†L159-L214】
2. **Validate diary integrity**
   - For each exported entry, recompute the signature via `compute_audit_signature(kind="decision_diary_entry", payload=entry_payload_without_signature)` and compare it to the stored value to confirm immutability.【F:src/understanding/decision_diary.py†L560-L595】【F:src/observability/immutable_audit.py†L80-L115】
   - Verify archived snapshots exist for the audit window; the presence of dated folders under `artifacts/diaries/` proves retention coverage.【F:src/artifacts/archive.py†L40-L94】
3. **Cross-check policy promotions**
   - Load the ledger with `poetry run python tools/governance/rebuild_policy.py --ledger artifacts/governance/policy_ledger.json --output artifacts/reports/policy_changelog.md` to regenerate a Markdown changelog for review.【F:tools/governance/rebuild_policy.py†L1-L199】【F:src/governance/policy_ledger.py†L720-L815】
   - Inspect each `PolicyLedgerRecord.audit_gaps()` to ensure required evidence IDs, approvals, and checklist completions exist for the target stage.【F:src/governance/policy_ledger.py†L520-L639】
4. **Verify compliance workflows**
   - Render the latest `ComplianceWorkflowSnapshot` to Markdown (via the nightly replay job or dedicated CLI) and confirm blocked items link to decision diary entry IDs for traceability.【F:src/compliance/workflow.py†L77-L138】【F:tools/operations/nightly_replay_job.py†L260-L340】
5. **Archive audit pack**
   - Collect exported Markdown, JSON, and signature validation reports into `artifacts/reports/<date>/`. Use `archive_artifact("reports", path)` for any locally generated documents to keep storage conventions consistent.【F:src/artifacts/archive.py†L18-L94】

## External Audit Support
- Provide auditors with the decision diary export, the ledger changelog, and the compliance workflow snapshot for the requested window.
- Supply signature verification scripts or outputs demonstrating integrity checks ran successfully.
- Share the promotion checklist status (from `PolicyLedgerRecord.promotion_checklist_status()`) to evidence risk governance coverage.【F:src/governance/policy_ledger.py†L440-L639】

## Evidence Examples
| Evidence | Source | Command | Notes |
| --- | --- | --- | --- |
| Decision diary (Markdown) | `artifacts/understanding/decision_diary.json` | `poetry run python tools/understanding/decision_diary_cli.py export-diary --diary <path> --format markdown` | Includes regime state, policy rationale, probe activations, and immutable signatures.【F:tools/understanding/decision_diary_cli.py†L63-L149】【F:src/understanding/decision_diary.py†L534-L623】 |
| Policy ledger changelog | `artifacts/governance/policy_ledger.json` | `poetry run python tools/governance/rebuild_policy.py --ledger <path> --output ledger.md` | Lists promotion history, approvals, checklist completion, and evidence IDs with tamper-proof signatures.【F:tools/governance/rebuild_policy.py†L1-L199】【F:src/governance/policy_ledger.py†L440-L639】 |
| Compliance workflow snapshot | Nightly replay artefacts | `poetry run python tools/operations/nightly_replay_job.py --output-dir artifacts/reports/nightly` | Summarises outstanding regulatory tasks and links to supporting artefacts (diary IDs, ledger evidence).【F:tools/operations/nightly_replay_job.py†L164-L340】【F:src/compliance/workflow.py†L77-L138】 |
| Signature validation report | Any diary/ledger entry | Custom Python invoking `compute_audit_signature` | Demonstrates cryptographic integrity of artefacts under review.【F:src/observability/immutable_audit.py†L80-L115】【F:src/governance/policy_ledger.py†L440-L519】 |

All procedures above must be captured in the audit ticket or runbook so reviewers can trace each step to the originating evidence.
