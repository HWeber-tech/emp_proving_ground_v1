# Batch10 Fix14 Diagnostics Plan (Diagnostics-only, no code changes)

Source snapshot:
- Snapshot: [mypy_snapshots/mypy_snapshot_2025-08-27T07-05-31Z.txt](mypy_snapshots/mypy_snapshot_2025-08-27T07-05-31Z.txt:1)
- Summary: [mypy_snapshots/mypy_summary_2025-08-27T07-05-31Z.txt](mypy_snapshots/mypy_summary_2025-08-27T07-05-31Z.txt:1)
- Ranked offenders: [mypy_snapshots/mypy_ranked_offenders_2025-08-27T07-05-31Z.csv](mypy_snapshots/mypy_ranked_offenders_2025-08-27T07-05-31Z.csv:1)
- Candidates list (authoritative): [mypy_snapshots/candidates_fix14_2025-08-27T07-05-31Z.txt](mypy_snapshots/candidates_fix14_2025-08-27T07-05-31Z.txt:1)

Plan scope:
- Total files planned: 12
- Total mypy errors across these files (from snapshot): 18

Global acceptance criteria:
- Round A proposals per file are behavior-preserving and ≤5 edits per file
- After applying Round A proposals and running ruff/isort/black plus mypy base + strict-on-touch on that file in isolation, the file reports zero mypy errors
- No source code, config, or workflow changes are performed by this document; it is a plan only
- Only the 12 modules listed in candidates_fix14 are included


## 1) src/core/_event_bus_impl.py

Module: [src/core/_event_bus_impl.py](src/core/_event_bus_impl.py:1)

Current error count: 5

Exact errors from snapshot:
- [src/core/_event_bus_impl.py](src/core/_event_bus_impl.py:293): error: Incompatible types in assignment (expression has type "Callable[[Event], Awaitable[None] | None]", variable has type "Callable[[Event], None] | Callable[[Event], Awaitable[None]] | None")  [assignment]
- [src/core/_event_bus_impl.py](src/core/_event_bus_impl.py:294): error: Incompatible types in assignment (expression has type "None", target has type "Callable[[Event], None] | Callable[[Event], Awaitable[None]]")  [assignment]
- [src/core/_event_bus_impl.py](src/core/_event_bus_impl.py:295): error: Argument 2 to "subscribe" of "AsyncEventBus" has incompatible type "Callable[[Event], None] | Callable[[Event], Awaitable[None]] | None"; expected "Callable[[Event], Awaitable[None]] | Callable[[Event], None]"  [arg-type]
- [src/core/_event_bus_impl.py](src/core/_event_bus_impl.py:405): error: Argument 1 to "get" of "dict" has incompatible type "tuple[str, Callable[[Event], object]]"; expected "tuple[str, Callable[[Event], None] | Callable[[Event], Awaitable[None]]]"  [arg-type]
- [src/core/_event_bus_impl.py](src/core/_event_bus_impl.py:407): error: Argument "handler" to "SubscriptionHandle" has incompatible type "Callable[[Event], object]"; expected "Callable[[Event], Awaitable[None]] | Callable[[Event], None]"  [arg-type]

Categories:
- Optional/union narrowing and incompatible callable types
- Any/object callable leakage to stricter callable union
- Map key/value types needing callable type alignment

Round A minimal changes (≤5 edits):
1) Narrow optional adapter to a non-None union and early-return when adapter is None; explicitly type adapter_fn as Callable[[Event], None] | Callable[[Event], Awaitable[None]] to avoid None flowing into the map or subscribe.
2) Guard map assignment with an if to ensure only non-None adapter_fn is stored; avoid assigning None into structures typed for callables.
3) When calling subscribe, ensure adapter_fn is non-None; if necessary, cast via [typing.cast()](typing:1) to Callable[[Event], None] | Callable[[Event], Awaitable[None]].
4) Where getting _pair_to_id with a tuple containing callback, cast callback to Callable[[Event], None] | Callable[[Event], Awaitable[None]] via [typing.cast()](typing:1) to align the tuple type.
5) When constructing SubscriptionHandle(handler=...), cast callback to the expected union using [typing.cast()](typing:1).

Round B single smallest additional change (placeholder):
- If mypy still flags the adapter callable union, introduce a small helper def that accepts Callable[[Event], None] | Callable[[Event], Awaitable[None]] and returns the same, to centralize and enforce narrowing at a single call site.


## 2) src/sensory/organs/sentiment_organ.py

Module: [src/sensory/organs/sentiment_organ.py](src/sensory/organs/sentiment_organ.py:1)

Current error count: 2

Exact errors from snapshot:
- [src/sensory/organs/sentiment_organ.py](src/sensory/organs/sentiment_organ.py:18): error: Name "SensoryOrgan" already defined (possibly by an import)  [no-redef]
- [src/sensory/organs/sentiment_organ.py](src/sensory/organs/sentiment_organ.py:23): error: Name "SensoryReading" already defined (possibly by an import)  [no-redef]

Categories:
- Redefinition of imported types via local Protocol fallback declarations

Round A minimal changes (≤5 edits):
1) Remove the local Protocol fallback class SensoryOrgan that shadows an imported name.
2) Remove the local Protocol fallback class SensoryReading that shadows an imported name.
3) Ensure types are imported only for typing by moving related imports under [TYPE_CHECKING](typing:1) where appropriate.
4) If runtime placeholders are needed, rename local fallbacks to private names (e.g., _SensoryOrganProto, _SensoryReadingProto) without re-exporting the public names.

Round B single smallest additional change (placeholder):
- If usages require local fallback names, add a single type alias mapping under [TYPE_CHECKING](typing:1) only to the canonical imports to avoid public name duplication at runtime and typing-time.


## 3) src/ecosystem/optimization/ecosystem_optimizer.py

Module: [src/ecosystem/optimization/ecosystem_optimizer.py](src/ecosystem/optimization/ecosystem_optimizer.py:1)

Current error count: 2

Exact errors from snapshot:
- [src/ecosystem/optimization/ecosystem_optimizer.py](src/ecosystem/optimization/ecosystem_optimizer.py:78): error: Return type "Coroutine[Any, Any, Mapping[str, Sequence[src.genome.models.genome.DecisionGenome]]]" of "optimize_ecosystem" incompatible with return type "Coroutine[Any, Any, Mapping[str, Sequence[src.core.interfaces.DecisionGenome]]]" in supertype "src.core.interfaces.IEcosystemOptimizer"  [override]
- [src/ecosystem/optimization/ecosystem_optimizer.py](src/ecosystem/optimization/ecosystem_optimizer.py:80): error: Argument 1 of "optimize_ecosystem" is incompatible with supertype "src.core.interfaces.IEcosystemOptimizer"; supertype defines the argument type as "Mapping[str, Sequence[DecisionGenome]]"  [override]

Categories:
- Incompatible override against interface due to mismatched DecisionGenome type path (implementation uses genome.DecisionGenome; interface expects core.interfaces.DecisionGenome)

Round A minimal changes (≤5 edits):
1) Import or alias the interface DecisionGenome under [TYPE_CHECKING](typing:1) and use it in the method signature types.
2) Update the optimize_ecosystem parameter annotation to Mapping[str, Sequence[DecisionGenome]] aligning to the interface package path.
3) Update the optimize_ecosystem return annotation to Mapping[str, Sequence[DecisionGenome]] aligning to the interface.
4) If an internal CanonDecisionGenome alias exists, use [typing.cast()](typing:1) at conversion boundaries instead of changing internal logic.

Round B single smallest additional change (placeholder):
- If needed, add a local type alias DecisionGenome = InterfaceDecisionGenome under [TYPE_CHECKING](typing:1) to document intent and avoid future drift.


## 4) src/thinking/prediction/predictive_modeler.py

Module: [src/thinking/prediction/predictive_modeler.py](src/thinking/prediction/predictive_modeler.py:1)

Current error count: 1

Exact errors from snapshot:
- [src/thinking/prediction/predictive_modeler.py](src/thinking/prediction/predictive_modeler.py:74): error: Argument 1 to "predict_market_scenarios" of "PredictiveMarketModeler" has incompatible type "dict[str, float]"; expected "dict[str, object]"  [arg-type]

Categories:
- dict invariance: passing dict[str, float] where dict[str, object] is expected

Round A minimal changes (≤5 edits):
1) Cast the dict argument at the call site via [typing.cast()](typing:1) to dict[str, object].
2) If the value types are numpy/pandas scalars, normalize to Python scalars using [float()](python:1) where applicable before the cast (only if required by other checks).

Round B single smallest additional change (placeholder):
- Narrow the callee’s parameter type to Mapping[str, object] if upstream invariance continues to be problematic.


## 5) src/thinking/prediction/predictive_market_modeler.py

Module: [src/thinking/prediction/predictive_market_modeler.py](src/thinking/prediction/predictive_market_modeler.py:1)

Current error count: 1

Exact errors from snapshot:
- [src/thinking/prediction/predictive_market_modeler.py](src/thinking/prediction/predictive_market_modeler.py:475): error: Argument 1 to "append" of "list" has incompatible type "dict[str, float]"; expected "dict[str, object]"  [arg-type]

Categories:
- dict invariance: list expects dict[str, object] but provided dict[str, float]

Round A minimal changes (≤5 edits):
1) At the append call site, wrap with [typing.cast()](typing:1) to dict[str, object].
2) Alternatively, adjust the local payload_list type annotation to list[Mapping[str, object]] if that reduces casts while remaining behavior-preserving.

Round B single smallest additional change (placeholder):
- Update normalize_prediction’s return annotation to dict[str, object] if that is the established convention across the predictive flow.


## 6) src/thinking/patterns/cvd_divergence_detector.py

Module: [src/thinking/patterns/cvd_divergence_detector.py](src/thinking/patterns/cvd_divergence_detector.py:1)

Current error count: 1

Exact errors from snapshot:
- [src/thinking/patterns/cvd_divergence_detector.py](src/thinking/patterns/cvd_divergence_detector.py:15): error: Name "ContextPacket" already defined (possibly by an import)  [no-redef]

Categories:
- Redefinition via local TypeAlias colliding with an imported symbol

Round A minimal changes (≤5 edits):
1) Remove the local ContextPacket: TypeAlias = Any declaration that conflicts.
2) If a local alias is required, rename to _ContextPacket or ContextPacketLike to avoid shadowing.

Round B single smallest additional change (placeholder):
- Introduce a more precise contextual type alias using [typing.cast()](typing:1) where the alias is consumed, if needed by adjacent code.


## 7) src/thinking/analysis/correlation_analyzer.py

Module: [src/thinking/analysis/correlation_analyzer.py](src/thinking/analysis/correlation_analyzer.py:1)

Current error count: 1

Exact errors from snapshot:
- [src/thinking/analysis/correlation_analyzer.py](src/thinking/analysis/correlation_analyzer.py:17): error: Module "src.core.exceptions" has no attribute "ThinkingException"; maybe "TradingException"?  [attr-defined]

Categories:
- attr-defined on import: missing symbol in exceptions module

Round A minimal changes (≤5 edits):
1) Change the import to use TradingException and alias it locally as ThinkingException, e.g., from src.core.exceptions import TradingException as ThinkingException.
2) No other code changes needed if only the name is used for raising/catching.

Round B single smallest additional change (placeholder):
- If the exception is only used for typing, relocate the import under [TYPE_CHECKING](typing:1) to minimize runtime coupling.


## 8) src/thinking/adaptation/tactical_adaptation_engine.py

Module: [src/thinking/adaptation/tactical_adaptation_engine.py](src/thinking/adaptation/tactical_adaptation_engine.py:1)

Current error count: 1

Exact errors from snapshot:
- [src/thinking/adaptation/tactical_adaptation_engine.py](src/thinking/adaptation/tactical_adaptation_engine.py:56): error: "FAISSPatternMemory" has no attribute "find_similar_experiences"  [attr-defined]

Categories:
- attr-defined mismatch against a dependency lacking the method in its stub/type

Round A minimal changes (≤5 edits):
1) Narrow the call site via [typing.cast()](typing:1) to Any or to a local Protocol that includes find_similar_experiences, e.g., cast(Any, self.pattern_memory).find_similar_experiences(...).
2) If a local Protocol exists already, cast to that Protocol at the assignment site instead.

Round B single smallest additional change (placeholder):
- Replace the direct attribute access with a small helper function that takes a Protocol-typed parameter and calls the method, limiting casts to one location.


## 9) src/sensory/organs/volume_organ.py

Module: [src/sensory/organs/volume_organ.py](src/sensory/organs/volume_organ.py:1)

Current error count: 1

Exact errors from snapshot:
- [src/sensory/organs/volume_organ.py](src/sensory/organs/volume_organ.py:22): error: Name "SensoryOrgan" already defined (possibly by an import)  [no-redef]

Categories:
- Redefinition of imported type via local Protocol fallback

Round A minimal changes (≤5 edits):
1) Remove the local Protocol fallback class named SensoryOrgan that duplicates an import.
2) Ensure type imports are under [TYPE_CHECKING](typing:1) where possible to avoid runtime duplication.

Round B single smallest additional change (placeholder):
- If runtime usage insists on a local Protocol, rename it to _SensoryOrganProto and keep usages typed against the imported name under [TYPE_CHECKING](typing:1).


## 10) src/sensory/organs/price_organ.py

Module: [src/sensory/organs/price_organ.py](src/sensory/organs/price_organ.py:1)

Current error count: 1

Exact errors from snapshot:
- [src/sensory/organs/price_organ.py](src/sensory/organs/price_organ.py:21): error: Name "SensoryOrgan" already defined (possibly by an import)  [no-redef]

Categories:
- Redefinition of imported type via local Protocol fallback

Round A minimal changes (≤5 edits):
1) Remove the local Protocol fallback class named SensoryOrgan that duplicates an import.
2) Keep any necessary typing-only imports under [TYPE_CHECKING](typing:1) to prevent redefinition.

Round B single smallest additional change (placeholder):
- Introduce a private-named Protocol for runtime behavior only if absolutely necessary, avoiding conflicts with the imported public type.


## 11) src/evolution/ambusher/ambusher_orchestrator.py

Module: [src/evolution/ambusher/ambusher_orchestrator.py](src/evolution/ambusher/ambusher_orchestrator.py:1)

Current error count: 1

Exact errors from snapshot:
- [src/evolution/ambusher/ambusher_orchestrator.py](src/evolution/ambusher/ambusher_orchestrator.py:36): error: Too many arguments for "AmbusherFitnessFunction"  [call-arg]

Categories:
- Call signature mismatch: extra positional/keyword arguments supplied

Round A minimal changes (≤5 edits):
1) Adjust the constructor call to pass only the supported argument(s); for example, pass a single config mapping or the minimal required positional argument set consistent with the target signature.
2) If ambiguity remains, refactor the call to use explicit keyword arguments that match the signature, dropping unsupported ones.

Round B single smallest additional change (placeholder):
- If stricter typing still fails, introduce a lightweight factory wrapper that adapts local config to the accepted constructor signature.


## 12) src/data_foundation/config/sizing_config.py

Module: [src/data_foundation/config/sizing_config.py](src/data_foundation/config/sizing_config.py:1)

Current error count: 1

Exact errors from snapshot:
- [src/data_foundation/config/sizing_config.py](src/data_foundation/config/sizing_config.py:10): error: Incompatible types in assignment (expression has type "None", variable has type Module)  [assignment]

Categories:
- Optional typing for module handle not declared, assigning None to Module-typed variable

Round A minimal changes (≤5 edits):
1) Import ModuleType from types and annotate the variable as ModuleType | None so that assigning None is valid.
2) If used only for typing, move the import of ModuleType under [TYPE_CHECKING](typing:1) and keep the runtime assignment unchanged.

Round B single smallest additional change (placeholder):
- If consumers assume non-None, add a guard and raise a descriptive exception when the module is unavailable, preserving type narrowing.


# Plan-level acceptance criteria

- After Round A changes are applied per file:
  - ruff, isort, and black produce no diffs beyond formatting the modified lines
  - mypy base + strict-on-touch for the touched file yields zero errors for that file
- No functional behavior changes beyond explicit, type-only coercions and guardings:
  - Numeric coercions limited to [float()](python:1) and [int()](python:1) where necessary
  - Optional guards using isinstance, attribute checks, or [typing.cast()](typing:1)
  - Type-only imports under [TYPE_CHECKING](typing:1)
  - Dict/list/tuple annotations tightened locally; payloads normalized to dict[str, object] where required
  - No cross-module API redesigns; prefer local casts/aliases

# Tally

- Files planned: 12
- Errors covered: 18

# Notes

- This document is diagnostics-only; it does not modify code.
- All file paths and line numbers are taken directly from the latest snapshot [mypy_snapshots/mypy_snapshot_2025-08-27T07-05-31Z.txt](mypy_snapshots/mypy_snapshot_2025-08-27T07-05-31Z.txt:1).