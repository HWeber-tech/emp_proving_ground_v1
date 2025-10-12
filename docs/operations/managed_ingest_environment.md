# Managed ingest environment quickstart

Use the `docker/institutional-ingest/docker-compose.yml` file to stand up the
managed Timescale, Redis, and Kafka services that the institutional ingest
vertical expects.  The compose stack exposes:

- TimescaleDB on `localhost:5433` with the `emp_market` database.
- Redis on `localhost:6380` with persistence enabled.
- Kafka (KRaft) on `localhost:9094` with external access for the ingest bridge.

## 1. Start the services

```bash
docker compose -f docker/institutional-ingest/docker-compose.yml up -d
```

Verify containers are healthy before wiring the runtime.  Timescale exposes a
`pg_isready` health-check, Redis responds to `redis-cli ping`, and Kafka is
polled through `kafka-topics.sh --list` in the compose health probe.

## 2. Configure SystemConfig extras

Copy `env_templates/institutional_ingest.env` to a safe location, update
credentials, and source it before running the runtime or the managed connectors
CLI:

```bash
cp env_templates/institutional_ingest.env .env.institutional
# edit the file, then
export $(grep -v '^#' .env.institutional | xargs)
```

The template seeds:

- Timescale connection URL, ingest dimensions, and intraday options.
- Redis defaults (client name, TTL, invalidation prefixes).
- Kafka bootstrap servers and ingest topics.
- Failover drill expectations for `run_failover_drill.py`.

## 3. Validate connectors

Run the managed connector CLI and point it at the environment file to confirm
configuration and collect connectivity evidence:

```bash
python -m tools.operations.managed_ingest_connectors \
  --env-file .env.institutional \
  --connectivity \
  --format markdown
```

The report mirrors the manifest exposed by the runtime builder, including the
managed connector snapshot and probe health.  Connectivity probes now emit
status strings (`ok`, `degraded`, `off`, `error`) together with latency and the
sanitised endpoint so operations can confirm a full Timescale→Redis→Kafka
handshake at a glance.  The `--env-file` flag injects the dotenv entries before
extras overrides, so the CLI can run without mutating the process environment.

## 3b. Combined readiness check

Generate a single report that includes the managed connector summary and an
optional failover drill by using the readiness CLI:

```bash
python -m tools.operations.institutional_ingest_readiness \
  --env-file .env.institutional \
  --connectivity \
  --ingest-results /tmp/ingest_results.json \
  --format markdown
```

When `--ingest-results` is supplied the command executes the same
`InstitutionalIngestProvisioner.run_failover_drill()` path used in runtime,
capturing the managed manifest, connectivity health, and failover snapshot in a
single artifact for reviews.

## 4. Exercise failover drills

Feed recent Timescale ingest results into the failover drill CLI while reusing
the same environment file:

```bash
python -m tools.operations.run_failover_drill \
  --env-file .env.institutional \
  --results /tmp/ingest_results.json \
  --format markdown
```

Add `--extra KEY=VALUE` overrides when you need to temporarily swap connectors
or credentials without editing the dotenv file. The drill snapshot includes the
managed manifest and the configured failover scenario, matching the metadata
recorded by `InstitutionalIngestServices`.

## 5. Teardown

```bash
docker compose -f docker/institutional-ingest/docker-compose.yml down -v
```

Removing the stack deletes the local volumes (Timescale, Redis, Kafka).  For
staging or production deployments replace the compose file with managed service
coordinates and rotate secrets according to the institutional data backbone
alignment brief.【F:docs/context/alignment_briefs/institutional_data_backbone.md†L139-L151】
