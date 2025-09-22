# Backup and restore readiness plan

The modernization roadmap calls for documented backup and restore drills for the
Timescale-backed institutional ingest slice. This plan summarises how the new
`evaluate_backup_readiness` helpers and runtime wiring satisfy that outcome and
what operators should configure or observe during drills.【F:docs/roadmap.md†L118-L135】【F:src/operations/backup.py†L1-L206】

## Policy inputs

Backups are configured via `SystemConfig` extras resolved by
`build_institutional_ingest_config` into a `TimescaleBackupSettings` payload.
Set the following environment variables to tune the policy for your deployment:

- `TIMESCALE_BACKUP_FREQUENCY_SECONDS` – expected interval between successful
  backups (defaults to 86 400 s).【F:src/data_foundation/ingest/configuration.py†L120-L210】
- `TIMESCALE_BACKUP_RETENTION_DAYS` / `TIMESCALE_BACKUP_MIN_RETENTION_DAYS` –
  actual vs. minimum retention window in days.
- `TIMESCALE_BACKUP_RESTORE_INTERVAL_DAYS` – maximum days between successful
  restore drills (0 disables restore monitoring).
- `TIMESCALE_BACKUP_LAST_SUCCESS` and `TIMESCALE_BACKUP_LAST_RESTORE` – ISO
  timestamps recorded by your backup tooling; the readiness evaluator parses
  them to determine staleness.
- `TIMESCALE_BACKUP_PROVIDERS` / `TIMESCALE_BACKUP_STORAGE` – descriptive
  metadata surfaced in telemetry so dashboards show storage targets.

The resolved settings are embedded inside runtime metadata and exposed through
`ProfessionalPredatorApp.summary()` for quick inspection.【F:src/data_foundation/ingest/configuration.py†L200-L320】【F:src/runtime/predator_app.py†L260-L380】

## Telemetry

`evaluate_backup_readiness` merges policy inputs with ingest health/quality and
failover findings to produce a `BackupReadinessSnapshot` that is:

1. Logged as markdown during ingest runs (`💾` block).
2. Published on `telemetry.operational.backups` via the runtime event bus.
3. Stored on the runtime so the `summary()` API exposes the latest snapshot for
   dashboards and operators.【F:src/runtime/runtime_builder.py†L300-L360】【F:src/runtime/predator_app.py†L260-L380】

Subscribe to the telemetry feed (event bus or Kafka bridge) or inspect the
`backups` section of the runtime summary when rehearsing disaster-recovery
scenarios. CI coverage in `tests/operations/test_backup.py` exercises policy
escalation, restore thresholds, and serialization to keep the contract stable.

## Drill guidance

1. **Verify telemetry:** Trigger a manual backup and restore; confirm the
   snapshot status returns to `ok` and the Markdown block lists the new
   timestamps.
2. **Fail the policy intentionally:** Edit `TIMESCALE_BACKUP_LAST_SUCCESS` to an
   old timestamp and rerun ingest. The readiness snapshot should escalate to
   `warn`/`fail` and emit an issue describing the stale backup.
3. **Restore rehearsal:** Set `TIMESCALE_BACKUP_RESTORE_INTERVAL_DAYS` to a low
   value (e.g. `1`) and confirm readiness escalates when restore drills are not
   executed within the window.

Record drill outcomes (timestamps, status, follow-up actions) alongside the
snapshot payload in your operational log so the roadmap’s backup milestone stays
verifiable.【F:docs/status/ci_health.md†L24-L36】
