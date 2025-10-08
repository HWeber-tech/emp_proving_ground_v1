# Timescale failover drills

The institutional ingest vertical now provisions managed Timescale, Redis, and
Kafka services through `InstitutionalIngestProvisioner`.  Operators can call the
provisioner when constructing runtime applications to ensure the ingest
scheduler and Kafka bridge run under a `TaskSupervisor`, while the Redis cache is
configured with the institutional policy.  The professional runtime builder now
invokes the provisioner automatically, so Tier‑1 deployments inherit supervised
connectors without custom wiring.  The `InstitutionalIngestServices`
bundle exposes a `failover_metadata()` helper that returns the serialized
`TimescaleFailoverDrillSettings`, making it trivial to surface drill
requirements in dashboards or to kick off `execute_failover_drill()` workflows.

Typical flow:

1. Resolve ingest configuration via `build_institutional_ingest_config()` and
   instantiate an `InstitutionalIngestProvisioner` with the resulting settings
   plus any Redis extras required for managed caches.
2. Call `provision()` with the ingest coroutine, runtime event bus, and a
   `TaskSupervisor`.  The provisioner returns an
   `InstitutionalIngestServices` bundle containing the scheduler, Kafka bridge,
   and optional Redis cache.
3. Start the services inside the runtime bootstrap.  The scheduler registers a
   supervised task and any Kafka bridge is tracked by the same supervisor.
4. When running recovery exercises, call `failover_metadata()` to retrieve the
   active drill configuration or invoke `run_failover_drill()` directly to
   execute the simulated outage with the managed connector metadata attached.

The services bundle now exposes three additional helpers for operations teams:

- `managed_manifest()` returns a redaction-safe summary of the Timescale, Redis,
  and Kafka connectors, including supervision state and configured topics.  The
  manifest can be rendered directly in readiness dashboards without replicating
  redaction logic.
- `connectivity_report(probes=...)` runs optional synchronous or asynchronous
  probes against the managed connectors, returning health-enriched manifest
  snapshots.  Operators can inject bespoke Timescale, Redis, or Kafka pings and
  publish the resulting manifest to observability streams alongside failover
  metadata.
- `run_failover_drill(results, fail_dimensions=...)` wraps
  `operations.failover_drill.execute_failover_drill()` so ingest teams can feed
  recorded Timescale results into a supervised drill.  The helper automatically
  applies the configured failover dimensions (falling back to the ingest plan or
  observed results), merges the managed connector manifest, and honours the
  `run_fallback` setting when executing optional fallback callables.

By routing failover drills through the new helper, Tier-1 ingest deployments
retain a consistent picture of which Timescale dimensions must participate in
disaster-recovery simulations while ensuring all background workloads remain
observable under the shared task supervisor.

## CLI workflow

The `tools/operations/run_failover_drill.py` helper executes the drill outside
of the runtime.  Provide recent ingest results (for example, from an export or
from the runtime summary) as JSON and the CLI will replay the failover using the
configured scenario and dimensions:

```bash
python -m tools.operations.run_failover_drill \
  --results /tmp/ingest_results.json \
  --scenario staged_drill \
  --format markdown
```

The command honours any failover drill configuration in `config.yaml` or the
environment.  If no dimensions are supplied it falls back to the configured
drill dimensions (or the dimensions present in the results).  JSON output
carries the aggregated snapshot from `execute_failover_drill()`, while Markdown
renders a human-friendly table for runbooks and dashboards.
