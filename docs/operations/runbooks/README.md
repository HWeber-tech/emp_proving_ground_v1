# Operational runbooks

These runbooks translate the institutional data backbone roadmap into
step-by-step procedures for on-call engineers.  Each playbook references the
telemetry surfaces that the runtime builder and professional summary expose so
operators can react with the same context that appears in the modernization
briefs and CI health snapshot.

## Available runbooks

- [Redis cache outage](redis_cache_outage.md) – Detects degraded cache health,
  captures metrics from `ManagedRedisCache`, and walks through switching to the
  bootstrap fallback while institutional Redis is restored.
- [Kafka ingest offset recovery](kafka_ingest_offset_recovery.md) – Guides the
  ingest event bridge through consumer lag diagnosis, manual offset commits, and
  replay validation using the Kafka telemetry helpers.
- [Risk policy violation escalation](risk_policy_violation.md) – Surfaces the
  policy snapshot, Markdown alert, and governance escalation path for trade
  intents rejected by the deterministic risk gateway.
- [Manual FIX order risk block](manual_fix_order_risk_block.md) – Documents the
  telemetry contract emitted by `FIXBrokerInterface` when manual orders are
  denied and links to the deterministic risk API remediation path.
- [Drift sentry response](drift_sentry_response.md) – Maps Page–Hinkley and
  variance detector severities to alert policies, readiness dashboards, and
  remediation steps so operators can halt promotions when sensory drift spikes.
- [Incident playbook validation](incident_playbook_validation.md) – Bundles the
  kill-switch, nightly replay, and trade rollback drills into a repeatable
  evidence pack for the incident response programme.
- [Validation framework failure response](validation_framework.md) – Explains the composite validation suite, success thresholds, and how to triage failing checks before promoting builds.

Keep these documents close to the institutional data backbone alignment brief so
runbooks, roadmap status, and acceptance hooks evolve together.
