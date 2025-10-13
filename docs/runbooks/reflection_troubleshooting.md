# Reflection TRM Troubleshooting Runbook

## Shadow Mode Bring-up

1. Ensure the Decision Diary source directory exists (default `artifacts/diaries/`) and contains files matching `diary_glob` (`diaries-*.jsonl`). If absent, generate sample diaries via `make sim-diaries` (or copy fixtures into place).
2. Copy `config/reflection/rim.config.example.yml` to `config/reflection/rim.config.yml` (the tooling will fall back to the example if the real config is missing) and adjust parameters including `diaries_dir`, `diary_glob`, window, and caps.
3. Run `make rim-shadow` to execute `tools/rim_shadow_run.py`.
4. Inspect emitted artifact under `artifacts/rim_suggestions/` and confirm governance gate remains enabled.

## Production Runner (non-shadow execution)

1. Use `tools/rim_run_trm.py --config config/reflection/rim.config.yml` to execute the production runner against the latest diaries. Provide `--model` if you need to override the weights path packaged in the config file.
2. The runner guards execution with a lockfile (`config.lock_path`) so concurrent cron jobs exit cleanly; expect `skipped_reason=lock_active` if a prior run is still active or the TTL has not expired.【F:src/reflection/trm/runner.py†L51-L167】
3. Artefacts default to `artifacts/rim_suggestions/` unless `publish_channel` is set (e.g. `file:///srv/rim/suggestions`). Each JSONL line includes the config hash, model hash, and run identifier so governance reviewers can trace provenance. Enable `--debug` to print the suggestion count, runtime, and output path to stdout.【F:tools/rim_run_trm.py†L14-L69】
4. With `enable_governance_gate: true` the runner also appends suggestions (with queue metadata) to `governance.queue_path` and writes JSON/Markdown digests under the configured governance paths, making the queue ready for policy reviewers without a shadow-mode handoff.【F:src/reflection/trm/governance.py†L17-L218】【F:src/reflection/trm/runner.py†L72-L153】
5. Telemetry logs append to `TelemetryConfig.log_dir`; confirm a new `rim-<date>.log` entry is written and contains runtime milliseconds, entry counts, and the resolved hashes. The regression suite validates encoder feature sets and schema compliance—run `pytest tests/reflection/test_trm_runner.py -q` after config updates to ensure outputs stay aligned.【F:tests/reflection/test_trm_runner.py†L24-L91】

## Sanity Checks

- **Encoder Output:** Enable debug logging (`RIM_DEBUG=1 make rim-shadow`) to log encoded feature shapes and batch counts.
- **Suggestion Count:** Verify suggestions do not exceed `suggestion_cap` and confidence remains above `confidence_floor`.
- **Telemetry:** Review `artifacts/rim_logs/rim-<date>.log` for runtime percentiles (`p50_ms`, `p95_ms`), throughput (`windows_processed`, `windows_halted_early_%`), and suggestion mix (`suggestions_emitted`, `suggestions_dropped_low_confidence`).
- **Schema Validation:** Run `make rim-validate` to validate JSONL artifacts (docs/examples + latest emissions).

## Lock & idempotency

- Lockfile: `artifacts/locks/rim.lock`. If it exists and is <2h old, exit 0.
- Output filename: `rim-suggestions-UTC-<ISO>-<RUN_ID>.jsonl`.
- RUN_ID = `<yyyyMMddHHmmssZ>-<hostname>-<pid>`. Replays set `rerun_of: <RUN_ID>`.

## Retention & Housekeeping

- Run `make rim-prune` (or `python tools/rim_prune.py --dry-run`) to enforce the 30-day retention window on `artifacts/rim_suggestions/`.
- CI executes the prune step on every docs-related change; schedule an ops cron if additional pruning is required.
- Decision Diaries are retained for 90 days (or the stricter data platform policy) to support backfills; purge or archive per compliance before expiry.

## Reading Artifacts

- Suggestions: newline-delimited JSON stored at `artifacts/rim_suggestions/rim-suggestions-UTC-<timestamp>.jsonl`.
- Each line contains schema metadata (version, hashes) and payload fields. Use `jq '.' <file>` to pretty-print.
- Governance reviewers should cross-reference `suggestion_id` with Decision Diary `audit_ids` for context.

## Time & data quality rules

- All diary timestamps and window boundaries must be normalized to UTC ISO-8601 before ingestion; runbooks computing windows should do so in UTC.
- On corrupt JSON lines: skip the line, increment `skipped_lines`, and continue. Log `{"skipped_lines": n, "total_lines": m}` for observability—never crash the run.
- When optional fields are missing, apply documented defaults (e.g., `belief_state_summary.vector = zeros(32)`).
- After loading, sort diary entries by timestamp; when timestamps tie, preserve original file order to keep hashes deterministic.

## Privacy & secrets hygiene

- Configure `redact.fields` and `redact.mode` to hash or drop identifiers such as `account_id` and `order_id` before artifacts leave the runner.
- Never persist secrets, API tokens, or credential material in artifacts or telemetry. Scrub configs and logs before sharing externally.

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

> RIM cannot mutate live weights. Only the Governance process may apply suggestions after approval.
