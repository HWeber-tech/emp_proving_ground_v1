# Compliance starter kits

These starter kits translate the compliance roadmap bullet into executable checklists
that ride on the same telemetry fabric as the other readiness feeds. Pair this document
with `docs/roadmap.md` and `docs/status/ci_health.md` when grooming tickets or reviewing
runtime output so the context stays anchored to the concept blueprint.

## Workflow inventory

- **MiFID II controls** – Transaction reporting (Article 26) and recordkeeping
  (Article 16) driven by trade compliance telemetry. The workflow surfaces failed
  rule checks, critical severities, and Timescale journaling status so desks can
  prove reporting readiness without manual spreadsheets.【F:src/compliance/workflow.py†L120-L226】
- **Dodd-Frank controls** – Large-trader threshold monitoring and swap data
  repository audit trails backed by trade history and per-symbol totals. The
  workflow highlights when policy failures block reporting or when history depth
  is insufficient for regulatory audits.【F:src/compliance/workflow.py†L228-L312】
- **KYC / AML workflows** – Customer due diligence, watchlist screening, and
  ongoing monitoring sourced from the KYC monitor summaries. Escalations and
  outstanding checklist items promote blocked/in-progress states so onboarding
  reviews stay actionable.【F:src/compliance/workflow.py†L314-L386】
- **Audit trail readiness** – Timescale-backed journaling for both trade compliance
  and KYC monitors, exposing gaps when journaling is disabled or monitors are
  offline. Operators can verify retention posture directly from runtime summaries
  before institutional audits.【F:src/compliance/workflow.py†L388-L468】

## Telemetry surfaces

- Runtime ingestion evaluates the workflows after each institutional ingest run and
  publishes `telemetry.compliance.workflow` alongside the readiness snapshot.
  The professional runtime records the latest payload and exposes Markdown tables
  via `ProfessionalPredatorApp.summary()` for dashboards and runbooks.【F:src/runtime/runtime_builder.py†L1200-L1336】【F:src/runtime/predator_app.py†L1-L520】
- Pytest coverage captures the workflow evaluator, publisher contract, and runtime
  integration so CI guards the new starter kits end-to-end.【F:tests/compliance/test_compliance_workflow.py†L1-L98】【F:tests/runtime/test_runtime_builder.py†L160-L240】【F:tests/runtime/test_professional_app_timescale.py†L200-L320】

## Operational notes

- Subscribe to `telemetry.compliance.workflow` on the event bus (or Kafka bridge)
  to ingest the checklists into dashboards or audit archives.
- The workflow metadata mirrors ingest success and readiness status so operators
  can correlate regulatory blockers with underlying data backbone issues.
- Extend the default tasks with desk-specific controls by feeding additional
  metadata into the trade or KYC summaries; the workflow evaluator automatically
  downgrades statuses when monitors go offline.
