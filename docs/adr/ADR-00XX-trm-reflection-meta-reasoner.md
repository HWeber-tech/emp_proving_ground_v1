# ADR-00XX: Reflection Intelligence Module (TRM Meta-Reasoner)

- **Status:** Proposed
- **Date:** 2024-XX-XX
- **Authors:** Reflection Working Group

## Context

Reflection operations require a lightweight meta-reasoner that can synthesize Decision Diary data without introducing latency or directly steering live execution. The Reflection Intelligence Module (RIM) adopts a Tiny Recursive Model (TRM) pattern to asynchronously refine assessments of strategy health and surface advisory actions for governance review. The design must coexist with existing observability, avoid new heavy ML dependencies, and respect current safety controls.

## Decision

Adopt a TRM-style architecture for RIM that:

1. Runs asynchronously against Decision Diaries and derived aggregates.
2. Utilizes a TRM loop to iteratively improve suggestion candidates before publishing JSONL artifacts.
3. Publishes advisory outputs only after passing through a governance gate, never executing trades automatically.
4. Provides CLI tooling, schemas, and telemetry necessary for shadow mode, gated rollout, and future automation.
5. Enforces schema_version/input_hash/model_hash/config_hash on every published object for auditability.

## Consequences

- Enables iterative reasoning on small hardware footprint, with clear integration points for future model upgrades.
- Introduces configuration, tooling, and monitoring overhead dedicated to the reflection lane.
- Requires ongoing governance reviews and artifact retention (30 days) to sustain traceability.
- Sets expectations for telemetry (runtime, suggestion counts, acceptance metrics) and validation tooling.

## Non-goals

- Modifying live trading, risk, execution, or broker pipelines.
- Introducing heavy ML dependencies or end-to-end training code in this iteration.
- Allowing RIM to bypass governance or execute trades.

## Safety

- Suggestions pass through a governance gate before any operational impact.
- Kill switch in configuration disables emissions while keeping tooling operational.
- Telemetry and validation tooling catch schema drift and anomalous suggestion volumes.
- Retention policy (30 days) for suggestion artifacts to support audits.

## Rollout Plan

1. **Shadow Mode:** Run RIM alongside Decision Diaries, publishing to artifacts/rim_suggestions/ without consuming downstream; validate schemas and telemetry.
2. **Governance-Gated:** Enable governance ingestion with manual approval, enforcing suggestion caps and confidence floors.
3. **Pilot:** Expand coverage to selected strategies post-governance sign-off, maintaining periodic audits and retraining checkpoints.
