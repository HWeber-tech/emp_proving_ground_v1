# Development Setup (Stub)

- Python 3.11
- pip install -r requirements/base.txt  # runtime stack
- pip install -r requirements/dev.txt   # tooling + tests
- Configure .env with FIX credentials (OpenAPI disabled)

The development manifest pins mypy, Ruff, Black, pytest, coverage, pre-commit, import-linter, and the
core type stub packages so local runs mirror CI. Use `pip install -r requirements/dev.txt` after pulling
to pick up any version bumps.

## Scientific stack checklist

Keep the runtime guard in sync with the dependency manifests. The table below mirrors the
hard limits enforced by `src/system/requirements_check.py`:

| Library | Minimum version | Notes |
| --- | --- | --- |
| numpy | 1.26.0 | Base numerical stack used across ingest, analytics, and trading loops. |
| pandas | 1.5.0 | Dataframe operations throughout sensory and validation modules depend on new indexing fixes. |
| scipy | 1.11.0 | Signal processing helpers rely on optimizations introduced in the 1.11 series. |

Run the pre-flight validator before major deployments:

```bash
python -m src.system.requirements_check
```

The command exits non-zero if any library is missing or below the documented floor and prints the
detected versions so upgrades can be recorded in `requirements/base.txt`.

## Developer data backbone services

The default `docker-compose.yml` now provisions TimescaleDB, Redis, and Kafka so
the runtime can exercise the real data backbone locally. Bring the stack up and
export the matching connection settings before running institutional workflows:

```bash
# start the data services (TimescaleDB, Redis, Kafka)
docker compose up -d redis postgres kafka

# seed SystemConfig extras with local endpoints
cp env_templates/dev_data_services.env .env.dev-data
export $(grep -v '^#' .env.dev-data | xargs)

# optional: verify the managed connector manifest using the preset config
python -m tools.operations.managed_ingest_connectors --config config/system/dev_data_backbone.yaml --connectivity
```

Validate connectivity with the smoke probes shipped in
`tests/operations/test_dev_data_services.py`:

```bash
pytest tests/operations/test_dev_data_services.py -m integration
```

The tests skip automatically when the services or client libraries are not
available. When the stack is running they perform a Timescale table round-trip,
a Redis ping/set/get cycle, and a Kafka produce/consume loop against
`localhost:9094`.

## Containerised runtime profiles

The roadmap's container milestone ships with two docker-compose profiles in
`docker/runtime/docker-compose.dev.yml`. They build the production runtime image
and wire it to the Timescale, Redis, and Kafka services above with health
checks mapped to the runtime's `/health` endpoint.

```bash
cd docker/runtime
# bring up the paper profile (ports: 18081->8000)
docker compose --profile paper up runtime-paper
# or launch the developer mock profile (ports: 18080->8000)
docker compose --profile dev up runtime-dev
```

Both services read shared defaults from `docker/runtime/env.common` and overlay
profile-specific overrides from either `env.dev` or `env.paper`. The resolved
settings are mirrored in the SystemConfig presets under
`config/deployment/runtime_dev.yaml` and `config/deployment/runtime_paper.yaml`
so operators can load the same profile outside of Docker.

Once the stack is healthy, hit the containerised health endpoint to confirm the
runtime builder started the `RuntimeHealthServer` correctly:

```bash
curl -fsS http://localhost:18081/health | jq '.status'
```

Switch the port to `18080` when checking the developer profile. The JSON payload
includes FIX connectivity, market-data freshness, and telemetry exporter status,
staying consistent with the runtime health tests in
`tests/runtime/test_healthcheck.py`.

## Formatting expectations

Ruff owns both linting and formatting. The formatter now runs repo-wide, so run
`ruff format` before committing and double-check with `ruff format --check .`
locally if you are uncertain. CI enforces the same guard during the lint job.
Use `ruff check --select I` to tidy imports after formatting if needed.
