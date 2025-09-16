# Architecture Overview

The EMP Professional Predator platform is built as a layered, event-driven
system that ingests market data, evaluates trading hypotheses, and executes FIX
orders against a simulator-first broker stack.  This document captures the
stabilized view after the recent roadmap push so contributors can orient
themselves without spelunking through legacy material.

## Layered domains

| Layer | Responsibilities | Representative modules |
| --- | --- | --- |
| **Core** | Fundamental abstractions that are dependency free: the event bus, configuration primitives, telemetry shims, and common exception types. | `src/core/event_bus.py`, `src/core/config_access.py`, `src/core/telemetry.py` |
| **Sensory** | Adapters that translate FIX market data and institutional feeds into normalized events placed on the bus. | `src/sensory/organs/dimensions/institutional_tracker.py`, `src/sensory/vendor/np_pd_shims.py` |
| **Thinking** | Signal processing and analytical pipelines that consume normalized events to detect opportunities. | `src/thinking/analysis/correlation_analyzer.py`, `src/thinking/patterns/trend_detector.py` |
| **Trading** | Portfolio state, order models, execution engines, and risk guards that convert decisions into FIX instructions. | `src/trading/execution/fix_executor.py`, `src/trading/models.py`, `src/risk/risk_manager_impl.py` |
| **Orchestration & Operational** | Runtime glue that composes the system, loads configuration, enforces policy, and exposes observability endpoints. | `main.py`, `src/orchestration/compose.py`, `src/operational/*` |

Dependencies flow downward only; each layer consumes services from layers below
it and communicates up-stack via domain events.  Import-linter contracts in
`contracts/importlinter.toml` enforce these edges.

## Event-driven runtime

1. **Bootstrap** – `main.py` loads the typed `SystemConfig`, validates the
   scientific stack, and activates guardrails (policy enforcement, dependency
   sanity checks).
2. **Wiring** – Core services such as the asynchronous event bus and telemetry
   facades are created and passed to sensory/trading subsystems.
3. **Ingestion** – Sensory organs translate broker feeds, Yahoo Finance
   snapshots, or replay fixtures into strongly typed events on the bus.
4. **Decision loop** – Thinking modules subscribe to topics, enrich the market
   state, and emit hypotheses for execution.
5. **Execution** – Trading pipelines evaluate risk, construct FIX orders, and
   send them to the simulator while updating telemetry counters.
6. **Observation** – Metrics exporters and structured logs surface system
   health.  The CI workflow publishes concise summaries for failure triage.

## Configuration story

* **SystemConfig** (in `src/governance/system_config.py`) is the canonical,
  strongly typed configuration object used by new code.  It guarantees enums,
  strict coercion, and environment overrides.
* **Legacy `core.configuration.Configuration`** remains for older integration
  glue.  Regression tests now cover its environment overrides, dot-path
  accessors, and YAML round-tripping so refactors can proceed safely.
* Runtime defaults live in `config.yaml`, which is aligned with the FIX-only
  policy and simulator-first posture.

## Observability baseline

* Structured logging is emitted via the standard `logging` package with module
  loggers.  Policy and dependency failures abort startup loudly.
* CI publishes a step summary and uploads pytest logs to surface failures
  without the retired Kilocode bridge.
* `scripts/audit_dead_code.py` and the CI baseline report in `docs/reports/`
  provide periodic hygiene snapshots.

Refer to `docs/architecture/refactor_roadmap.md` for the long-term decomposition
plan and to `docs/operations/observability_plan.md` for alerting strategy
details.
