# Final Dry Run Harness

This runbook describes how to execute the AlphaTrade final dry run in support of Phase II sign-off. The harness coordinates a supervised runtime execution, captures structured evidence, and evaluates sign-off readiness using the roadmap criteria (≥72h duration, ≥98% uptime, decision diary + performance telemetry).

## Prerequisites
- A production-parity runtime configuration (Timescale/Kafka/Redis online, governance feature flags enabled).
- Decision diary and performance telemetry collectors wired to write JSON artifacts.
- Sufficient disk space for multi-day structured logs (estimate ~1GB/day at INFO level).

## Launching the harness
1. Pick an empty evidence directory, e.g. `artifacts/final_dry_run/2025-10-12`.
2. Run the harness CLI, providing the runtime command after `--`:
   ```sh
   tools/operations/final_dry_run.py \
     --log-dir artifacts/final_dry_run/2025-10-12 \
     --duration-hours 72 \
     --minimum-uptime-ratio 0.98 \
     --diary data/diaries/final_dry_run.jsonl \
     --performance artifacts/performance/final_dry_run.json \
     --metadata sprint=phase2 --metadata run_id=final-dry-run-2025-10-12 \
     -- python3 main.py --symbols EURUSD,GBPUSD
   ```
   The harness performs live log monitoring: stderr lines and structured logs
   with `level=error`/`critical`/`fatal` are captured immediately as harness
   incidents so operators can react before the post-run audit finishes. Pass
   `--no-log-monitor` if the runtime emits noisy warnings that should only be
   handled during the evidence review.
3. Monitor stdout for harness incidents; failures will surface immediately and exit with a non-zero code.

## Evidence bundle
- Structured logs: `<log-dir>/final_dry_run_<timestamp>.jsonl`
- Raw runtime logs: `<log-dir>/final_dry_run_<timestamp>.log`
- JSON summary (optional): provide `--json-report summary.json`
- Markdown summary (optional): provide `--markdown-report summary.md`
- Evidence packet (optional): pass `--packet-dir artifacts/final_dry_run/2025-10-12/packet --packet-archive artifacts/final_dry_run/2025-10-12/packet.tar.gz` to bundle summaries plus raw artefacts in a review-ready archive.
- Review brief (optional): add `--review-output review.md` (or `--review-output -` for stdout) to emit the meeting brief generated from the audit and sign-off records.

The JSON summary contains the dry run audit (`summary`) plus the sign-off report. Harness incidents (unexpected exits, duration shortfalls) are embedded into `summary.metadata.harness_incidents`.

## Sign-off criteria mapping
- **Duration**: enforced via `--duration-hours` / `--required-duration-hours` (default 72h)
- **Uptime ratio**: calculated from structured logs; set with `--minimum-uptime-ratio`
- **Decision diary / performance telemetry**: paths must be supplied (`--diary`, `--performance`) unless explicitly waived (`--no-diary-required`, `--no-performance-required`)
- **Sharpe threshold**: `--minimum-sharpe-ratio` ensures risk boards can demand risk-adjusted performance evidence.

## Post-run review
1. Upload the JSON summary, raw logs, decision diary, and performance report to the governance evidence bucket.
2. If the CLI exits with WARN or FAIL, inspect `harness_incidents` and the log summary sections to identify gaps (duration, uptime, log errors).
3. For WARN status accepted under `--allow-warnings`, file follow-up tickets and document risk sign-offs per governance policy.

## Regression coverage
Automated tests under `tests/operations/test_final_dry_run.py` simulate both a successful multi-heartbeat run and an early termination to guard against regressions in the harness or audit workflow.
