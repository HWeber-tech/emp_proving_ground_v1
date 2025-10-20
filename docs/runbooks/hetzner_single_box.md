# Hetzner Single-Box Stack

Phase J.2 requires Docker installation plus a compose stack that runs the data
services (TimescaleDB, Redis, Kafka) alongside the trading engine. This runbook
assumes Ubuntu 22.04 on a dedicated Hetzner host with a sudo-capable operator.
Provision the host with the Terraform module under `infra/hetzner` before
continuing.

## 1. Prepare the host

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl git gnupg lsb-release
```

Clone or sync the EMP repository to the target path (e.g. `/opt/emp`) and
switch to the project root before proceeding.

## 2. Bootstrap Docker and the stack

The helper script installs Docker Engine, the compose plugin, and then brings up
TimescaleDB, Redis, Kafka, and the engine service using the production image.
Run it from the repository root:

```bash
./scripts/deployment/bootstrap_hetzner_stack.sh
```

The script:
- Configures Docker's apt repository if missing.
- Installs `docker-ce`, `docker-compose-plugin`, and enables the Docker daemon.
- Adds the current user to the `docker` group when needed.
- Builds the engine image and launches the compose stack defined in
  `docker/hetzner/docker-compose.yml`.

Log out/in after the first run if the script adds your account to the `docker`
group so `docker compose` works without `sudo`.

## 3. Managing the compose stack

Once the bootstrap completes, the stack can be managed with standard compose
commands:

```bash
# view status
docker compose -f docker/hetzner/docker-compose.yml ps

# follow engine logs
docker compose -f docker/hetzner/docker-compose.yml logs -f engine

# restart services if you deploy a new image
docker compose -f docker/hetzner/docker-compose.yml up -d --build engine
```

Services publish the following ports on the host. Adjust the compose file before
running the stack if you need different bindings or external Kafka advertising.

| Service | Host port | Container | Notes |
| --- | --- | --- | --- |
| TimescaleDB | 5432 | `timescale:5432` | Init script at `config/database/init.sql` seeds schemas. |
| Redis | 6379 | `redis:6379` | AOF persistence enabled. |
| Kafka | 9094 | `kafka:9092/9094` | External listener defaults to `localhost`; update `KAFKA_CFG_ADVERTISED_LISTENERS` for remote clients. |
| Engine | 8000 | `engine:8000` | Health endpoint at `/health`. |

Persistent data for TimescaleDB, Redis, and Kafka is stored in the docker
volumes declared inside the compose file. Engine logs and reports are bind
mounted from the repository (`data/`, `logs/`, `artifacts/`).

## 4. Health checks

Each container publishes a health check. Quick verification commands:

```bash
curl -fsS http://localhost:8000/health | jq '.status'
PGPASSWORD=emp_password psql -h localhost -U emp_user -d emp_db -c 'SELECT 1'
redis-cli -h localhost -p 6379 ping
/opt/bitnami/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9094 --list
```

With all checks green, the Hetzner single-box deployment satisfies J.2 and is
ready for runtime workloads.
