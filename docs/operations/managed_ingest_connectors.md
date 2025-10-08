# Managed ingest connector report

Use the `tools/operations/managed_ingest_connectors.py` CLI to summarise the
managed Timescale, Redis, and Kafka connectors that the institutional ingest
vertical provisions.  The command evaluates the current `SystemConfig`
(extracted from environment variables or a YAML file), renders a redaction-safe
manifest for readiness dashboards, and optionally executes lightweight
connectivity probes under the same supervision scaffolding used in runtime.

## Quick start

```bash
python -m tools.operations.managed_ingest_connectors --format markdown
```

The output lists whether the ingest slice should run, the active schedule, and
redacted connector metadata (Timescale application name + URL, Redis cache
summary, Kafka bootstrap servers/topics).  Add the `--connectivity` flag to run
managed probes that issue `SELECT 1` against Timescale, ping the Redis cache via
an in-memory stub, and confirm Kafka topic visibility through the supervised
consumer bridge.

## Configuration options

- `--config path/to/config.yaml` – load settings from a legacy YAML file.
- `--env-file path/to/.env` – seed `SystemConfig` from a dotenv template without exporting values.
- `--extra KEY=VALUE` – inject or override `SystemConfig` extras without editing
  environment variables.
- `--format json|markdown` – switch between machine-readable output (JSON) and a
  runbook-ready Markdown summary.
- `--output report.json` – persist the manifest to disk for attachment to
  incident reviews or context packs.
- `--connectivity --timeout 2.0` – evaluate connector health with jitter-aware
  probes, mirroring the supervised scheduler contract used in production.
- `--ensure-topics`/`--topics-dry-run` – ensure Kafka ingest topics exist (or
  dry-run the provisioning step) using the same provisioner that runtime
  deployments rely on, returning the broker responses alongside the manifest.【F:tools/operations/managed_ingest_connectors.py†L92-L104】【F:tools/operations/managed_ingest_connectors.py†L293-L393】

The CLI leans on `plan_managed_manifest()` so the manifest and the runtime
provisioner share the same metadata contract.  When connectivity checks are
requested the tool instantiates the provisioner with a `TaskSupervisor`, using
an in-memory Redis client and the Kafka bridge stubs already covered by the
runtime guardrails.  As a result the evidence produced by the CLI matches what
operators see once the ingest slice is promoted.

Pair the connector report with the `tools.operations.institutional_ingest_readiness`
CLI when you need a single artefact that combines the manifest, optional
connectivity probes, and a failover drill snapshot for audits or promotion
reviews.【F:tools/operations/institutional_ingest_readiness.py†L1-L246】【F:tests/tools/test_institutional_ingest_readiness_cli.py†L30-L95】
