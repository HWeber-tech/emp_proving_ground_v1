# Reflection TRM Troubleshooting Runbook

## Shadow Mode Bring-up

1. Ensure the Decision Diary source directory exists (default `artifacts/diaries/`) and contains files matching `diary_glob` (`diaries-*.jsonl`). If absent, generate sample diaries via `make sim-diaries` (or copy fixtures into place).
2. Copy `config/reflection/rim.config.example.yml` to `config/reflection/rim.config.yml` (the tooling will fall back to the example if the real config is missing) and adjust parameters including `diaries_dir`, `diary_glob`, window, and caps.
3. Run `make rim-shadow` to execute `tools/rim_shadow_run.py`.
4. Inspect emitted artifact under `artifacts/rim_suggestions/` and confirm governance gate remains enabled.

## Sanity Checks

- **Encoder Output:** Enable debug logging (`RIM_DEBUG=1 make rim-shadow`) to log encoded feature shapes and batch counts.
- **Suggestion Count:** Verify suggestions do not exceed `suggestion_cap` and confidence remains above `confidence_floor`.
- **Telemetry:** Review `artifacts/rim_logs/rim-<date>.log` for runtime (`p50/p95`) and acceptance placeholders.
- **Schema Validation:** Run `make rim-validate` to validate JSONL artifacts (docs/examples + latest emissions).

## Retention & Housekeeping

- Run `make rim-prune` (or `python tools/rim_prune.py --dry-run`) to enforce the 30-day retention window on `artifacts/rim_suggestions/`.
- CI executes the prune step on every docs-related change; schedule an ops cron if additional pruning is required.

## Reading Artifacts

- Suggestions: newline-delimited JSON stored at `artifacts/rim_suggestions/rim-suggestions-UTC-<timestamp>.jsonl`.
- Each line contains schema metadata (version, hashes) and payload fields. Use `jq '.' <file>` to pretty-print.
- Governance reviewers should cross-reference `suggestion_id` with Decision Diary `audit_ids` for context.

## Common Failure Modes

| Symptom | Diagnosis | Remediation |
| --- | --- | --- |
| No output file generated | `kill_switch` may be `true` or diary directory empty. | Set kill switch to `false` and ensure diaries exist. |
| Validation errors during `make rim-validate` | Schema mismatch or missing metadata fields. | Regenerate artifacts with correct schema; consult `interfaces/rim_types.json`. |
| Low confidence suggestions | Encoder or runner misconfiguration. | Inspect `config_hash`, adjust feature scaling, retrain TRM. |
| Telemetry missing | Log directory misconfigured or lacks write permissions. | Confirm `telemetry.log_dir` path and permissions; rerun. |

## Escalation

- For governance ingestion issues, contact the Policy Ops team with artifact path and log excerpts.
- For schema updates, raise a PR updating `interfaces/rim_types.json`, bump `schema_version`, and document in the API CHANGELOG.
- For persistent runtime regressions, engage MLOps to profile TRM runner and adjust recursion parameters (`K_outer`, `n_inner`).
