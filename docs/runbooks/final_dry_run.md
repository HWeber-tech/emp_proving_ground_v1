# Final Dry Run Harness

This runbook describes how to execute the AlphaTrade final dry run in support of Phase II sign-off. The harness coordinates a supervised runtime execution, captures structured evidence, and evaluates sign-off readiness using the roadmap criteria (≥72h duration, ≥98% uptime, decision diary + performance telemetry).

## Prerequisites
- A production-parity runtime configuration (Timescale/Kafka/Redis online, governance feature flags enabled).
- Decision diary and performance telemetry collectors wired to write JSON artifacts.
- Sufficient disk space for multi-day structured logs (estimate ~1GB/day at INFO level).

## Launching the harness
1. Pick an empty evidence directory, e.g. `artifacts/final_dry_run/2025-10-12`.
2. Use the turnkey orchestrator when you want the tool to provision timestamped
   run directories, wire evidence paths into the runtime environment, and emit
   review packets automatically:
   ```sh
   tools/operations/final_dry_run_orchestrator.py \
     --run-label "Phase II UAT" \
     --objective governance=pass:Controls-enforced \
     --attendee "Ops Lead" \
     --note "Verify throttles" \
     -- python3 main.py --symbols EURUSD,GBPUSD
   ```
   The orchestrator creates `<output-root>/<timestamp>/` (default
   `artifacts/final_dry_run/<slug>`), seeds `logs/`, `progress.json`,
   `summary.json`, `summary.md`, and review artefacts, and injects
   `FINAL_DRY_RUN_LOG_DIR`, `FINAL_DRY_RUN_DIARY_PATH`, and
   `FINAL_DRY_RUN_PERFORMANCE_PATH` into the child process unless you override
   them via `--env`. Metadata, objectives, attendees, notes (including
   `--notes-file`) flow into the generated review brief.
   The runtime now mirrors these extras onto the canonical
   `DECISION_DIARY_PATH` / `PERFORMANCE_METRICS_PATH` keys during
   `SystemConfig` construction and installs a supervised performance writer when
   `FINAL_DRY_RUN_PERFORMANCE_PATH` is present, so decision diary consumers and
   the auto-generated performance snapshot stay aligned without extra wiring.
3. When you need to reuse an existing directory or integrate with external
   wrappers you can still call the harness CLI directly—available both through
   the repo utility script and the consolidated EMP CLI:
   ```sh
   tools/operations/final_dry_run.py \
     --log-dir artifacts/final_dry_run/2025-10-12 \
     --progress-path artifacts/final_dry_run/2025-10-12/progress.json \
     --duration-hours 72 \
     --minimum-uptime-ratio 0.98 \
     --diary data/diaries/final_dry_run.jsonl \
     --performance artifacts/performance/final_dry_run.json \
     --compress-logs \
     --metadata sprint=phase2 --metadata run_id=final-dry-run-2025-10-12 \
     -- python3 main.py --symbols EURUSD,GBPUSD

   # or the equivalent EMP CLI entry point
   emp final-dry-run \
     --log-dir artifacts/final_dry_run/2025-10-12 \
     --progress-path artifacts/final_dry_run/2025-10-12/progress.json \
     --duration-hours 72 \
     --minimum-uptime-ratio 0.98 \
     --diary data/diaries/final_dry_run.jsonl \
     --performance artifacts/performance/final_dry_run.json \
     --compress-logs \
     --metadata sprint=phase2 --metadata run_id=final-dry-run-2025-10-12 \
     -- python3 main.py --symbols EURUSD,GBPUSD
   ```
   When the diary or performance artefacts are expected to update continuously,
   use `--diary-stale-warn-minutes` / `--diary-stale-fail-minutes` (and the
   equivalent `--performance-*` switches) to surface live incidents if evidence
   stops refreshing mid-run. The optional `--evidence-check-interval-minutes`
   flag tightens or relaxes the polling cadence, while
   `--evidence-initial-grace-minutes` controls how long the harness waits after
   startup before enforcing freshness thresholds.
   The harness performs live log monitoring: stderr lines and structured logs
   with `level=error`/`critical`/`fatal` are captured immediately as harness
   incidents so operators can react before the post-run audit finishes. Pass
   `--no-log-monitor` if the runtime emits noisy warnings that should only be
   handled during the evidence review. Add `--compress-logs` to keep multi-day
   runs from ballooning disk usage; the structured (`.jsonl.gz`) and raw (`.log.gz`)
   artefacts remain compatible with the audit tooling. The same log pump feeds cumulative
   per-stream and per-level counters that surface in the progress snapshot
   JSON for at-a-glance health monitoring. Stack traces printed to stdout or
   embedded in structured payloads are now auto-classified as FAIL incidents so
   latent exceptions surface even when emitted at INFO level.【F:src/operations/final_dry_run.py†L1206-L1321】【F:tests/operations/test_final_dry_run.py†L383-L401】
   Add `--live-gap-alert-minutes N` to surface a WARN (or FAIL with
   `--live-gap-alert-severity fail`) incident whenever the runtime becomes
   silent for `N` minutes. Tighten post-run expectations by supplying
   `--warn-gap-minutes` / `--fail-gap-minutes` when the 72 hour window requires
   explicit log coverage guarantees.
   Multi-day rehearsals can optionally rotate log files with
   `--log-rotate-hours 12` (or any positive interval), generating sequential
   `final_dry_run_<timestamp>_pNNN.jsonl[.gz]` segments so evidence remains
   reviewable without multi-gigabyte files.【F:src/operations/final_dry_run.py†L329-L414】【F:tests/operations/test_final_dry_run.py†L78-L115】
4. Monitor stdout for harness incidents; failures will surface immediately and exit with a non-zero code.

## Smoke testing
Use the bundled simulated runtime to verify harness wiring before committing to
a multi-day rehearsal:

```sh
tools/operations/final_dry_run_smoke.py \
  --duration-seconds 30 \
  --tick-interval 1.0 \
  --review-markdown artifacts/final_dry_run_smoke/review.md
```

The smoke wrapper launches `src.operations.final_dry_run_simulated_runtime`,
which emits structured heartbeats, diary entries, and performance snapshots in
real time. The short run exercises progress telemetry, evidence freshness
monitors, review briefs, and packet assembly without waiting 72 hours. Run the
smoke command after dependency upgrades or harness changes to catch regressions
quickly.

## Evidence bundle
- Structured logs: `<log-dir>/final_dry_run_<timestamp>.jsonl[.gz]`
- Raw runtime logs: `<log-dir>/final_dry_run_<timestamp>.log[.gz]`
- Progress telemetry: `<log-dir>/final_dry_run_<timestamp>_progress.json` (status, phase, log stream/level counts, incidents, exit code, sign-off verdict)
- JSON summary (optional): provide `--json-report summary.json`
- Markdown summary (optional): provide `--markdown-report summary.md`
- Evidence packet (optional): pass `--packet-dir artifacts/final_dry_run/2025-10-12/packet --packet-archive artifacts/final_dry_run/2025-10-12/packet.tar.gz` to bundle summaries plus raw artefacts in a review-ready archive.
- Review brief (optional): add `--review-output review.md` (or `--review-output -` for stdout) to emit the meeting brief generated from the audit and sign-off records.
- Acceptance objectives (optional): capture roadmap readiness checks with `--objective NAME=STATUS[:NOTE]` flags, e.g. `--objective governance=pass:Risk controls enforced`. Supply multiple flags to cover data backbone, understanding, governance, and launch readiness; they surface in the review brief and JSON reports.

The JSON summary contains the dry run audit (`summary`) plus the sign-off report. Harness incidents (unexpected exits, duration shortfalls) are embedded into `summary.metadata.harness_incidents`.

Progress snapshots default to every 5 minutes; adjust with `--progress-interval-minutes` or disable by setting it to `0` when invoking the CLI. Each snapshot includes cumulative log stats (per-stream and per-level counts, first/last timestamps), the harness status/phase, elapsed runtime, optional incidents, plus any computed summary or sign-off verdict once available. On shutdown the harness writes a final snapshot that locks in the exit code and sign-off decision so reviewers can audit the full run without tailing live logs.

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
