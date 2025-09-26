# CI Recovery Plan: Mypy Backlog

## Current Situation
- `types` job in CI executes `mypy --config-file mypy.ini src` against the entire codebase and the latest run completes with **0 errors** (`mypy --config-file mypy.ini src`, 2024-06-03 snapshot).
- Recently remediated failure themes included unsafe numeric conversions on `object` payloads, incorrect standard-library `Event` construction, missing annotations, dict variance mismatches, stale ignores, and integration protocol gaps. Keep these patterns on the radar for regression prevention.
- `mypy.ini` is intentionally lenient globally with targeted "L3" strictness on select packages (e.g., `src.sensory.utils.*`, `src.core.performance.*`).

## Objectives
1. Restore a green CI pipeline without abandoning visibility into typing issues.
2. Pay down the mypy backlog iteratively by package, using the existing strictness tiers.
3. Institutionalize workflows so that new code does not reintroduce untyped regressions.

## Guiding Principles
- **Visibility over silence:** Keep mypy reporting actionable information even if we temporarily downgrade failure severity.
- **Iterative hardening:** Expand strict coverage only after a package clears all outstanding errors.
- **Ownership:** Assign maintainers per package to distribute the remediation load.
- **Automation support:** Utilize stubs, helper utilities, and lint automation to accelerate fixes.

## Phase 0 — Contain the CI Failures
- [x] Update `.github/workflows/ci.yml` to scope the failing mypy job to strict pilot packages (start with `src/sensory/utils`) _or_ allow failure via `--ignore-errors` while emitting output.
- [x] Announce interim policy: CI stays green while mypy errors are treated as P1 tech debt until backlog resolves. (See [Interim Communication Draft](#interim-communication-draft).)
- [x] Add a dashboard link or nightly artifact (e.g., upload mypy report) so the wider team can monitor progress.

## Phase 1 — Inventory and Categorize Errors
- [x] Export the current `mypy --config-file mypy.ini src` output to `mypy_snapshots/` as a baseline artifact (`mypy_snapshots/2024-05-16_full.txt`).
- [x] Bucket findings by error class (conversion safety, bad constructor args, missing types, variance, stale ignores, missing stubs). Summary captured in [`docs/mypy_backlog_inventory.md`](mypy_backlog_inventory.md).
- [x] For each bucket, draft remediation patterns (helper functions, type aliases, or refactors) to standardize fixes. (See the remediation guidance table in the inventory.)
- [x] Identify high-traffic modules (recent git history) and flag for early remediation to reduce merge conflicts.

## Phase 2 — Package-by-Package Remediation

### Pilot (Strict) Packages
- [x] **`src.sensory.utils.*`** — Verified L3 strictness clean via targeted mypy run (2024-05-17).
- [x] **`src.core.performance.*`** — Confirmed no outstanding errors under strict settings.
- [x] **`src.validation.accuracy.*`** — Ensured package passes strict configuration with current tests.
- [x] **`src.operational.state.*`** — Validated annotations and protocol usage satisfy strict tier.
- [x] **`src.ui.models.*`** — Checked Event handling and container typing remain compliant.

### Sensory Dimension Remediation (May 17, 2024)
- [x] Hardened `src.sensory.dimensions.why`, `src.sensory.why`, and enhanced telemetry adapters to eliminate dict invariance and untyped metadata issues flagged by mypy.
- [x] Normalised HOW/ANOMALY sensor metadata payloads so `SensorSignal` consumers receive `dict[str, object]` structures without `type: ignore` usage.
- [x] Extend the telemetry typing cleanup to the WHEN sensory package so SensorSignal metadata conforms to `dict[str, object]`.
- [x] Carry the same telemetry typing cleanup through the WHAT sensory package and remaining legacy organs.

### Batch Sweep 2 Targets
- [x] **`src.trading.models.*`** — Annotated dataclass hooks and validated helper APIs so package-level mypy runs pass without `no-untyped-def` violations.
- [x] **`src.trading.performance.analytics.*`** — Harden numeric pathways and generics; verified clean targeted run (2024-06-03).
- [x] **`src.ui.api.*`** — Confirm importable stubs and request/response typing; validated via dedicated mypy pass (2024-06-03).
- [x] **`src.trading.risk.*`** — Normalise telemetry payloads, intent enrichment, and helper utilities so mypy accepts the risk gateway pipeline.
- [x] **`src.compliance.*`** — Share numeric coercion helpers across KYC and workflow telemetry to eliminate unsafe `int()` conversions.
- [x] **`src.trading.monitoring.portfolio_monitor` / `src.trading.trading_manager`** — Adopt shared coercion helpers, expose Redis telemetry stubs, and align experiment event metadata so the trading manager stack type-checks without `type: ignore` usage.

### Data Foundation Sweep
- [x] **`src.data_foundation.cache.timescale_query_cache`** — Normalised timestamp handling, canonicalised cache signatures, and removed legacy `type: ignore` usage in the Redis-backed Timescale cache adapter.
- [x] **`src.data_foundation.cache.redis_cache`** — Added a typed optional import shim for the redis module so helper factories no longer rely on unused ignores when the dependency is absent.
- [x] **`src.data_foundation.services.macro_events`** — Hardened macro event coercion and metadata aggregation to use typed numeric conversions and UTC timestamp normalisation.
- [x] **`src.data_foundation.fabric.market_data_fabric`** — Wrapped callable connectors with typed adapters so asynchronous fetchers expose consistent signatures without runtime TypeErrors.
- [x] **`src.data_foundation.persist.*` / `fabric.timescale_connector`** — Adopt the shared timestamp coercion helpers and add return annotations for ingest record factories to eliminate the remaining mypy errors emitted alongside the cache/service modules.
- [x] **`src.data_foundation.ingest.*` (health, recovery, scheduler telemetry, timescale pipeline)** — Normalised metadata payloads, guarded optional event plans, and ensured helper utilities respect typed fetcher signatures to keep ingest readiness telemetry type-safe.
- [x] **`src.data_foundation.ingest.observability`** — Hardened metrics payload typing, coerced numeric fields, and normalised recovery summaries so observability snapshots avoid `object` mutations.
- [x] **`src.data_foundation.streaming.kafka_stream`** — Added confluent-kafka stubs, removed legacy ignores, and cast producer/consumer factories so ingest provisioning and consumer bridges satisfy the strict package settings.

### Observability Instrumentation Sweep
- [x] **`src.observability.tracing`** — Typed optional OpenTelemetry fallbacks, narrowed resource attribute mappings, and aligned exporter headers with `dict[str, str]` expectations so the package passes targeted mypy runs (2024-06-03).

### Operations Telemetry Sweep
- [x] **`src.operations.backup` / `incident_response` / `compliance_readiness`** — Guarded metadata payloads, reused numeric coercion helpers, and ensured Markdown rendering only iterates over typed mappings.
- [x] **`src.operations.system_validation` / `slo` / `roi` / `kafka_readiness`** — Adopted shared coercion helpers, hardened event bus publishing, and annotated snapshot metadata to remove legacy `type: ignore` usage.
- [x] **`src.operations.data_backbone` / `evolution_tuning`** — Normalised plan metadata typing, scheduler snapshot handling, and strategy status defaults to unblock targeted mypy runs.
- [x] **`src.operations.configuration_audit`** — Normalised configuration extras handling and change registration metadata so snapshot diffs operate on typed mappings without Callable keyword mismatches.
- [x] **`src.operations.fix_pilot` / `cross_region_failover` / `bootstrap_control_center`** — Normalised telemetry issue lists, switched event bus publishing to the synchronous helper, and hardened bootstrap reporting helpers to guard optional champion payloads and numeric coercion.

### Runtime Integration Sweep
- [x] **`src.runtime.fix_dropcopy` / `src.trading.integration.fix_broker_interface`** — Enforced coroutine-based task factories and dropcopy publication guards so asyncio task spawning and event bus fan-out no longer rely on loose `Awaitable` typing or unavailable `emit_nowait` helpers.
- [x] **`src.runtime.task_supervisor` / `src.runtime.predator_app`** — Propagated coroutine typing through task supervision, renamed duplicate payload variables, and aligned factory adapters with the stricter task signature expected by the core event bus.

### Intelligence Systems Sweep
- [x] **`src.intelligence.*`** — Annotated lazy facade helpers, async harnesses, and sentient adaptation constructors so the package runs clean under L3 strictness and is now promoted in `mypy.ini`.

### Backlog Packages (Non-strict yet)
- [x] Prioritize modules with the highest error counts once strict packages are clean.
      * 2024-06-15: Backlog retired; see the [weekly status log](mypy_status_log.md) for continuing zero-error verification.
- [x] Apply the remediation patterns established earlier.
      * Shared coercion helpers, stub coverage, and telemetry adapters applied across remaining backlog modules during the final sweep.
- [x] Promote each cleaned package to L3 in `mypy.ini` during or immediately after remediation.
  - [x] `src.intelligence.*` lazy facades promoted after the strict typing sweep.
  - [x] Repository-wide `check_untyped_defs = True` enabled on 2024-06-15 following a clean full-project run.
- [x] Cleared governance vision alignment and system configuration helpers alongside the depth-aware liquidity prober to remove
      unsafe numeric coercions and `Any` returns (2024-05-25).
- [x] Cleared runtime predator orchestration (`src/runtime/predator_app.py`), runtime builder/healthcheck helpers, bootstrap stack intent payloads, and risk manager limit handling to eliminate the final 31-error backlog (2024-06-02).

## Phase 3 — Structural Improvements
- [x] Introduce shared helper functions for safe numeric casting (e.g., `coerce_int`, `coerce_float`).
- [x] Create/expand protocol and stub definitions (`stubs/`) for external dependencies (added redis client surface, 2024-06-04).
- [x] Establish a conventions guide covering when to use `Mapping` vs `dict`, handling events, and annotation defaults (see [`docs/mypy_conventions.md`](mypy_conventions.md)).
- [x] Integrate linting automation (pre-commit hook invoking mypy on changed files with `--config-file mypy.ini`).
  - Added a `pre-push` mypy hook to `.pre-commit-config.yaml` so contributors automatically type-check staged Python files with the project configuration.

## Phase 4 — CI Tightening & Regression Safeguards
- [x] Expand the CI mypy target beyond pilot packages as they reach zero errors (2024-06-04).
- [x] Remove temporary `--ignore-errors` or scoped limitations once backlog addressed.
- [x] Enable `warn_unused_ignores` globally and prune obsolete ignores (2024-06-08).
- [x] Consider enabling stricter defaults (e.g., `disallow_untyped_defs = True`) repository-wide after remediation.
  - Completed 2024-06-15 by enabling global `check_untyped_defs = True` and validating zero-error snapshot `2024-06-15_full.txt`.
- [x] Add a nightly build that fails on any mypy regression for early warning. (See `.github/workflows/mypy-nightly.yml`, scheduled daily at 05:00 UTC.)

## Phase 5 — Documentation & Knowledge Transfer
- [x] Record playbooks for the top error categories and remediation techniques. (See [`docs/mypy_playbooks.md`](mypy_playbooks.md).)
- [x] Host an internal brownbag or async write-up summarizing lessons learned. (See [`docs/ci_recovery_lessons.md`](ci_recovery_lessons.md).)
- [x] Update onboarding materials to include typing expectations and CI guardrails. (See [`docs/mypy_onboarding.md`](mypy_onboarding.md).)

## Ongoing Maintenance
- [x] Track progress weekly (error count delta, packages cleared) and circulate status updates. (See [`docs/mypy_status_log.md`](mypy_status_log.md).)
- [x] Gate merges on passing mypy for touched files (strict-on-touch) even before full CI reinstatement.
  - Added a CI gate that runs mypy with --check-untyped-defs on changed Python files via `tools/run_mypy_strict_on_changed.py`.
- [x] Review new dependencies for available type stubs before adoption. (Follow the [`type stub intake checklist`](mypy_dependency_checklist.md).)

## Escalation Triggers
- If the error count rises week-over-week, convene a focused remediation swarm.
- If a package cannot be remediated within its sprint allocation, reassess architecture (possible refactor or tech debt story).
- For blockers tied to missing third-party stubs, evaluate contributing upstream or temporarily annotating as `Any` with TODOs.

## Interim Communication Draft

> **Subject:** CI mypy containment plan — temporary policy
>
> Team,
>
> With the backlog cleared, the CI `types` job now runs `mypy --config-file mypy.ini src` and blocks merges on regressions. Treat any mypy failure as **P0 technical debt**: resolve it before landing the originating change, and loop in the owning team immediately if you need support. Continue leaning on the shared coercion helpers and conventions guide to avoid reintroducing unsafe patterns. Weekly updates will track the (target) zero-error status and surface any new strictness expansions.

Circulate this announcement in #eng-infra and link the [mypy backlog inventory](mypy_backlog_inventory.md) for detailed remediation guidance.

