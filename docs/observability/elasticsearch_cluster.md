# Elasticsearch Cluster Deployment Guide

This guide explains how to bootstrap the development Elasticsearch cluster that
backs the observability roadmap deliverables. The helper script orchestrates the
Docker Compose bundle, waits for the service to become healthy, and pushes a
verification log document to confirm ingestion is working.

## Prerequisites

- Docker Engine with Compose Plugin (`docker compose`) or Docker Compose v1
- Python 3.11+
- Network access to the host running the containers (defaults to `localhost`)

## Quick start

```bash
# Launch the cluster and ingest a verification log
python tools/observability/deploy_elasticsearch_cluster.py
```

The script performs the following steps:

1. Starts the container defined in `docker/elasticsearch/docker-compose.yaml`
   using Docker Compose. Use `--skip-compose` to reuse an existing deployment.
2. Polls `/_cluster/health` until Elasticsearch reports a `yellow` or `green`
   status (defaults to a five-minute timeout).
3. Publishes a bootstrap log entry into the `logs-emp-bootstrap` index and
   confirms it is searchable.

Expect JSON output similar to the following when the deployment succeeds:

```json
{
  "cluster_name": "docker-cluster",
  "status": "yellow",
  "active_primary_shards": 1,
  "active_shards": 1,
  "delayed_unassigned_shards": 0
}
```

A second payload summarises the ingested document, including a unique marker
that can be used to trace the verification event.

## Customisation

- `--compose-file`: Point to an alternative Compose file (e.g. remote host).
- `--project-name`: Override the Compose project name.
- `--elastic-url`: Target a non-default Elasticsearch endpoint.
- `--skip-ingest`: Perform health checks without writing the verification log.

All options can be listed with `python tools/observability/deploy_elasticsearch_cluster.py --help`.

## Clean-up

Use Docker Compose to stop and remove the containers when they are no longer
needed:

```bash
cd docker/elasticsearch
docker compose down -v
```

This removes the persistent `esdata` volume as well as the container.
