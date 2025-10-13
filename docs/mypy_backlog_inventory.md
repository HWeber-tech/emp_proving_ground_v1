# Mypy Backlog Inventory — 2024-05-16

> **2024-06-07 update:** Latest `mypy --config-file mypy.ini src` run completes with **0 errors**. Historical backlog details
> remain below for context and future regression triage.

## Snapshot Source
- Command: `mypy --config-file mypy.ini src`
- Output captured to [`mypy_snapshots/2024-05-16_full.txt`](../mypy_snapshots/2024-05-16_full.txt)
- Total errors reported: 259 across 56 modules

## Error Buckets & Remediation Patterns
| Bucket | Count | Representative Modules | Recommended Remediation |
| --- | --- | --- | --- |
| Collection type mismatch (dict/list variance, incompatible assignments) | 28 | `src/sensory/dimensions/why/yield_signal.py`, `src/trading/trading_manager.py` | Use `Mapping`/`Sequence` interfaces or widen value unions; ensure dict literals match declared value types before return. |
| Missing attribute access due to `object` typing | 28 | `src/operations/backup.py`, `src/governance/system_config.py` | Replace opaque `object` with concrete TypedDict/dataclass, or add runtime guards (`isinstance`) before attribute access; propagate precise typing from data sources. |
| Unused `# type: ignore[...]` directives | 23 | `src/sensory/why/why_sensor.py`, `src/trading/risk/risk_gateway.py` | Remove ignores where no longer necessary; otherwise attach narrow error codes confirmed by current failures. |
| Unsafe numeric conversion calls (`int`/`float`) | 17 | `src/trading/liquidity/depth_aware_prober.py`, `src/trading/risk/policy_telemetry.py` | Introduce coercion helpers that validate strings/Decimals before casting; treat `None` and `object` inputs defensively. |
| Functions returning `Any` | 16 | `src/governance/vision_alignment.py`, `src/trading/performance/report_builder.py` | Tighten return annotations or cast results to concrete `Mapping`/`Sequence` types before returning. |
| Missing type annotations | 14 | `src/data_foundation/persist/timescale.py` | Add explicit function returns and module-level variable annotations; prefer `Final` for constants. |
| Event constructor mismatches | 9 | `src/runtime/predator_app.py`, `src/trading/risk/risk_gateway.py` | Swap to project-specific `Event` factory or update constructor usage to match the typed signature; consider helper wrapper returning annotated events. |
| Generics & override violations | 9 | `src/sensory/enhanced/_shared.py`, `src/trading/risk/risk_gateway.py` | Supply type parameters for generic containers, and align method signatures (`__hash__`, private helpers) with their supertype definitions. |
| Overly generic mutation of `object` containers (`Unsupported target for indexed assignment`) | 8 | `src/operations/backup.py`, `src/data_foundation/cache/timescale_query_cache.py` | Refine metadata payload types to `dict[str, Any]` or structured models before in-place mutation. |
| Metadata/telemetry payload mismatches | 7 → 3 | `src/sensory/how/how_sensor.py`, `src/sensory/anomaly/anomaly_sensor.py` | Align sensor constructors with telemetry metadata schema or relax metadata typing via TypedDict updates. Recent fixes standardised HOW/WHY/ANOMALY payloads on `dict[str, object]` to remove invariance failures. |
| Other | 100 | Multiple modules | Remaining errors include protocol mismatches, forward reference issues, and missing stub coverage. These will be triaged in package-specific passes. |

## Top Modules by Error Count
1. `src/compliance/workflow.py` — 20 errors
2. `src/data_foundation/streaming/kafka_stream.py` — 19 errors
3. `src/trading/trading_manager.py` — 18 errors
4. `src/data_foundation/persist/timescale.py` — 16 errors
5. `src/data_foundation/ingest/observability.py` — 14 errors
6. `src/trading/risk/risk_gateway.py` — 12 errors
7. `src/runtime/predator_app.py` — 12 errors
8. `src/data_foundation/cache/timescale_query_cache.py` — 10 errors
9. `src/risk/telemetry.py` — 8 errors
10. `src/operations/system_validation.py` — 8 errors

## High-Traffic Modules (last 90 days)
Prioritize the following frequently modified modules to minimize merge conflicts during remediation:
- `src/thinking/adversarial/market_gan.py`
- `src/sensory/organs/dimensions/base_organ.py`
- `src/core/population_manager.py`
- `src/trading/trading_manager.py`
- `src/thinking/adversarial/red_team_ai.py`
- `src/integration/component_integrator_impl.py`
- `src/core/interfaces.py`
- `src/validation/phase2_validation_suite.py`
- `src/thinking/prediction/predictive_market_modeler.py`
- `src/thinking/competitive/competitive_understanding_system.py`

## Immediate Follow-Ups
- Share this inventory with package owners and align on tranche sequencing.
- For each strict package, spin up dedicated branches targeting the buckets above.
- Add progress notes back to the [CI recovery plan](ci_recovery_plan.md) as remediation lands.

## Progress Log
- **2024-05-17:** Sensory WHY/HOW/ANOMALY modules now conform to strict metadata typing and numeric coercion rules, eliminating 10 invariance/arg-type errors without suppressions. Telemetry helpers in `src/sensory/enhanced/_shared.py` were generalised for heterogeneous payloads, unlocking downstream adoption in the remaining sensory organs.
- **2024-05-18:** WHEN sensor metadata normalised to `dict[str, object]` with explicit numeric coercions, clearing the remaining arg-type violations in `src/sensory/when/when_sensor.py`.
- **2024-05-19:** WHAT sensory pipeline now emits structured telemetry with coercion helpers and legacy FIX organ task spawning aligns with asyncio typing, unlocking the next tranche of sensory remediation.
- **2024-05-20:** Trading risk gateway, policy telemetry, and risk telemetry modules now coerce numeric payloads, guard metadata hydration, and drop stale ignores so the `src.trading.risk.*` package passes dedicated mypy runs.
- **2024-05-21:** Shared numeric coercion helpers now back compliance KYC/workflow telemetry, clearing unsafe `int()` conversions and narrowing checklist metrics for stricter mypy runs.
- **2024-05-22:** Data foundation cache, macro events, and market data fabric modules now normalise pandas timestamps, remove broad `type: ignore` escapes, and wrap connector callables with typed adapters, reducing the Timescale-related backlog to the reader/connector factories.
- **2024-05-23:** Timescale persistence and connector modules now reuse shared coercion helpers, decode JSON payloads safely, and expose typed row adapters so ingest factories and Timescale readers satisfy mypy without `type: ignore` directives.
- **2024-05-24:** Operations telemetry (backup, system validation, compliance readiness, ROI, Kafka readiness, data backbone) now reuses shared coercion helpers, guards metadata payloads, and publishes events synchronously when available; ingest recovery/health utilities normalise metadata and scheduler telemetry to unblock targeted mypy runs.
- **2024-05-25:** Governance vision alignment/system configuration modules and the trading depth-aware liquidity prober now guard optional
  mappings and numeric payloads, removing the `Any` return and unsafe `float()` conversions highlighted in the backlog snapshot.
- **2024-05-26:** Trading manager, portfolio monitor, and Kafka streaming bridges now rely on shared coercion helpers, typed Redis/Kafka stubs, and cleaned telemetry metadata so strict package checks pass without legacy `type: ignore` directives.
- **2024-05-27:** Ingest observability snapshots now coerce metric payloads, guard recovery summaries, and normalise symbol lists so `src.data_foundation.ingest.observability` passes targeted mypy runs without `object` attribute errors.
- **2024-05-28:** Trading models dataclasses now declare explicit `__post_init__` return types, eliminating `no-untyped-def` errors and keeping the package's helper APIs aligned with strict mypy settings.
- **2024-05-29:** Runtime FIX integration and supervision components now enforce coroutine-based task factories, restore TopicBus-safe event publication, and tidy duplicate payload construction in the predator app so targeted runtime mypy runs complete cleanly.
- **2024-05-30:** Configuration audit telemetry now coerces extras mappings and registers change metadata without keyword mismatches, clearing the remaining mypy errors in `src.operations.configuration_audit`.
- **2024-05-31:** Fixed FIX pilot telemetry publishing and bootstrap control center typing, added a redis optional-import protocol shim, and reduced the backlog to 31 errors across five modules (per `mypy --config-file mypy.ini --follow-imports=skip src`).
- **2024-06-02:** Refactored the predator runtime stack (bootstrap stack intents, risk manager limits, runtime builder, healthcheck telemetry) to share typed coercion/helpers, clearing the remaining 31-error backlog so `mypy --config-file mypy.ini --follow-imports=skip src` now passes cleanly.
- **2024-06-03:** Typed the OpenTelemetry tracing fallbacks, constrained resource attribute mappings, and normalised exporter header handling so `src.observability.tracing` and the full `mypy --config-file mypy.ini src` run complete without errors.
- **2024-06-04:** Added a focused `redis` client stub to the local shim inventory so cache helpers and runtime trading orchestration can rely on precise client interfaces without suppressions.
- **2024-06-05:** Published a typing conventions guide (`docs/mypy_conventions.md`) covering container protocols, event construction, annotation defaults, and ignore hygiene to reinforce consistent fixes during ongoing remediation.
- **2024-06-06:** Wired mypy into the pre-push pre-commit stage so changed Python files run `mypy --config-file mypy.ini` locally before landing, reducing the chance of reintroducing backlog errors.
- **2024-06-07:** Re-enabled full-project mypy enforcement in CI, replacing the scoped pilot run with a blocking `mypy --config-file mypy.ini src` step and publishing the report artifact for regression visibility.
- **2024-06-08:** Enabled global `warn_unused_ignores`, removed redundant suppression comments across bootstrap control center, runtime builders, predator app, and FIX integrations, and introduced helper utilities to safely coerce snapshot payloads so cleanup leaves mypy at zero errors.
- **2024-06-09:** Published remediation playbooks (`docs/mypy_playbooks.md`) summarising fixes for the top error classes so follow-up contributors can resolve regressions consistently.
- **2024-06-10:** Added a scheduled nightly mypy workflow (`.github/workflows/mypy-nightly.yml`) that fails on any regression and publishes the full report artifact for early detection.
- **2024-06-11:** Introduced a strict-on-touch CI step that runs mypy with `--check-untyped-defs` on changed Python files so pull requests surface untyped definitions before merge.
- **2024-06-12:** Promoted the `src.intelligence.*` facades to L3 strictness by annotating lazy symbol wrappers and adaptation orchestrators, keeping targeted mypy runs clean while expanding strict coverage.
- **2024-06-13:** Published an async brownbag summary (`docs/ci_recovery_lessons.md`) to broadcast CI recovery takeaways and keep teams aligned on ongoing mypy guardrails.
- **2024-06-15:** Enabled repository-wide `check_untyped_defs`, captured snapshot [`2024-06-15_full.txt`](../mypy_snapshots/2024-06-15_full.txt), and published onboarding/maintenance guides to hold the zero-error baseline.
