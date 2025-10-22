# Local Elasticsearch Cluster

This Compose bundle provisions a single-node Elasticsearch 8.x cluster suitable
for development and integration testing. It is intentionally configured without
security to minimise setup friction and to make it easy to interact with the
cluster from the host machine. Persistent data is stored in the `esdata`
volume.

## Usage

```bash
cd docker/elasticsearch
# Start or update the cluster
docker compose up -d

# Tear it down when finished
docker compose down
```

By default the HTTP API is available on `http://localhost:9200` and transport on
`localhost:9300`. The accompanying deployment helper in
`tools/observability/deploy_elasticsearch_cluster.py` can bootstrap the
cluster, wait for it to become healthy, and ingest a verification log entry.
