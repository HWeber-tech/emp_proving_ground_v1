# Runbook – Redis cache outage

Institutional deployments rely on managed Redis caches to serve Timescale query
results and trading telemetry.  The runtime builder already records cache
telemetry and surfaces it through `telemetry.cache.health`; this runbook ties the
signals to concrete actions when Redis becomes unavailable.

## 1. Detect the incident

1. Watch the professional runtime summary or the event bus for the
   `cache_health` block.  A `fail` status with the issue *"Redis cache expected
   but not configured"* indicates that the runtime fell back to the in-memory
   client because Redis could not be reached.【F:src/operations/cache_health.py†L60-L187】【F:src/runtime/runtime_builder.py†L1770-L1793】
2. Confirm the hit/miss counters.  If the snapshot shows zero traffic or a sharp
   miss spike, Redis may be serving traffic but is not retaining items due to
   evictions or TTL expiry; note the metrics for later analysis.【F:src/operations/cache_health.py†L138-L183】
3. Review runtime logs for the `Redis cache configured` and `Redis ping failed`
   messages.  The cache wrapper logs instantiation failures and ping responses at
   startup, giving immediate clues about credential or network problems.【F:src/data_foundation/cache/redis_cache.py†L488-L509】

## 2. Stabilize the platform

1. Validate that the runtime is using the bootstrap fallback by checking the
   `backing` field recorded by the ingest loop.  When Redis is unreachable the
   builder downgrades to `InMemoryRedis`, which keeps the platform alive while
   disabling cross-process caching.【F:src/runtime/runtime_builder.py†L1770-L1794】【F:src/data_foundation/cache/redis_cache.py†L294-L400】
2. If the workload cannot tolerate stale reads, reduce batch sizes or pause
   downstream analytics until Redis is restored.  The in-memory cache is scoped
   to a single process and will not share state across replicas.
3. Capture the `ManagedRedisCache.metrics()` snapshot for the incident report so
   the operations team can analyse hit/miss ratios, evictions, and invalidations
   once Redis is back online.【F:src/data_foundation/cache/redis_cache.py†L316-L461】

## 3. Restore Redis connectivity

1. Verify that the Redis endpoint is reachable from the deployment environment
   (network ACLs, security groups, or managed service health page).
2. Rotate credentials if the incident stemmed from authentication failures; the
   runtime reads them from `SystemConfig`/environment variables and reconnects on
   the next ingest cycle.【F:src/data_foundation/cache/redis_cache.py†L82-L183】【F:src/runtime/runtime_builder.py†L750-L809】
3. Once Redis is reachable, restart the runtime or trigger the Redis client
   provisioning task so the builder recreates the managed cache wrapper with the
   institutional policy (15-minute TTL, 1 024 hot keys).【F:src/data_foundation/cache/redis_cache.py†L239-L279】【F:src/runtime/runtime_builder.py†L1770-L1794】

## 4. Validate recovery

1. Confirm that `evaluate_cache_health` reports `status: ok`, the namespace is
   correct, and hit/miss counters start to climb with traffic.【F:src/operations/cache_health.py†L138-L199】
2. Inspect runtime logs for the `Redis cache configured` info line, which is
   emitted after a successful ping, and ensure no new warnings are present.【F:src/data_foundation/cache/redis_cache.py†L488-L509】
3. Record the recovery in the on-call log along with the metrics snapshot so the
   Redis outage history remains linked to the institutional data backbone
   alignment brief.
