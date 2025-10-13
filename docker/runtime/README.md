# Runtime Docker profiles

This directory contains compose assets for running the Professional Predator
runtime in containers. The `docker-compose.dev.yml` file defines two profiles:

- `dev` – mock connection protocol for local smoke testing, publishes
  `runtime-dev` on port 18080.
- `paper` – paper-trading wiring with structured logging enabled, publishes
  `runtime-paper` on port 18081.

Both profiles rely on the shared environment defaults in `env.common` and the
profile-specific overrides in `env.dev` / `env.paper`. The same values are
available as SystemConfig presets under `config/deployment/runtime_dev.yaml` and
`config/deployment/runtime_paper.yaml` when operators need to run outside of
Docker.

Bring up a profile with:

```bash
cd docker/runtime
docker compose --profile paper up runtime-paper
```

Each service waits for TimescaleDB, Redis, and Kafka health checks and exposes
`/health` on the published port so deployment pipelines can monitor the runtime
via `RuntimeHealthServer`.
