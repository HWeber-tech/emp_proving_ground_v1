# Batch10 Fix7 Diagnostics Plan — 2025-08-26

Scope: Diagnostics-only plan for 12 files from [changed_files_batch10_fix7_candidates.txt](changed_files_batch10_fix7_candidates.txt:1). No code changes in this commit.

Validation inputs:
- Snapshot: [mypy_snapshots/mypy_snapshot_2025-08-26T10-42-16Z.txt](mypy_snapshots/mypy_snapshot_2025-08-26T10-42-16Z.txt:1)
- Summary: [mypy_snapshots/mypy_summary_2025-08-26T10-42-16Z.txt](mypy_snapshots/mypy_summary_2025-08-26T10-42-16Z.txt:1)
- Headline: Found 268 errors in 69 files (Post-Batch10 fix6 snapshot)
- Exclusions honored: all files listed for Fix3–Fix6 are excluded.

Candidate set confirmation (12/12):
- [src/orchestration/enhanced_intelligence_engine.py](src/orchestration/enhanced_intelligence_engine.py)
- [src/integration/component_integrator.py](src/integration/component_integrator.py)
- [src/trading/risk_management/__init__.py](src/trading/risk_management/__init__.py)
- [src/orchestration/compose.py](src/orchestration/compose.py)
- [src/genome/models/genome_adapter.py](src/genome/models/genome_adapter.py)
- [src/validation/validation_framework.py](src/validation/validation_framework.py)
- [src/ui/ui_manager.py](src/ui/ui_manager.py)
- [src/trading/execution/fix_executor.py](src/trading/execution/fix_executor.py)
- [src/thinking/thinking_manager.py](src/thinking/thinking_manager.py)
- [src/sentient/sentient_predator.py](src/sentient/sentient_predator.py)
- [src/ecosystem/evaluation/niche_detector.py](src/ecosystem/evaluation/niche_detector.py)
- [src/core/population_manager.py](src/core/population_manager.py)

Method:
1) Verified candidates against the file list above.
2) Extracted exact mypy errors from the raw snapshot with file:line references.
3) Proposed Round A minimal edits (≤5 per file) using established safe patterns:
   - Safe numeric coercions: [float()](mypy_snapshots/mypy_summary_2025-08-26T10-42-16Z.txt:1), [int()](mypy_snapshots/mypy_summary_2025-08-26T10-42-16Z.txt:1), [Decimal()](mypy_snapshots/mypy_summary_2025-08-26T10-42-16Z.txt:1) via Decimal(str(...))
   - Guard Optionals via isinstance and [typing.cast()](mypy_snapshots/mypy_summary_2025-08-26T10-42-16Z.txt:1)
   - Add [-> None](mypy_snapshots/mypy_summary_2025-08-26T10-42-16Z.txt:1) to procedures
   - Annotate locals: list[T], dict[str, V]; heterogeneous blobs as dict[str, object]
   - Asyncio gather narrowing: [cast[Tuple[…]]](mypy_snapshots/mypy_summary_2025-08-26T10-42-16Z.txt:1)
   - Timestamp normalization helpers where applicable
4) Defined acceptance criteria per file.

---

## 1) [src/orchestration/enhanced_intelligence_engine.py](src/orchestration/enhanced_intelligence_engine.py)

Current errors (6):
- [src/orchestration/enhanced_intelligence_engine.py](src/orchestration/enhanced_intelligence_engine.py:147): Returning Any from function declared to return "float"  [no-any-return]
- [src/orchestration/enhanced_intelligence_engine.py](src/orchestration/enhanced_intelligence_engine.py:174): "object" not callable  [operator]
- [src/orchestration/enhanced_intelligence_engine.py](src/orchestration/enhanced_intelligence_engine.py:175): "object" not callable  [operator]
- [src/orchestration/enhanced_intelligence_engine.py](src/orchestration/enhanced_intelligence_engine.py:176): "object" not callable  [operator]
- [src/orchestration/enhanced_intelligence_engine.py](src/orchestration/enhanced_intelligence_engine.py:177): "object" not callable  [operator]
- [src/orchestration/enhanced_intelligence_engine.py](src/orchestration/enhanced_intelligence_engine.py:178): "object" not callable  [operator]

Categorization:
- Any leaks
- Return type mismatch

Round A minimal changes (≤5):
1) Coerce correlation to float on return path: clamp over [float()](mypy_snapshots/mypy_snapshot_2025-08-26T10-42-16Z.txt:1) value.
2) Ensure engine attributes are typed instances/factories rather than `object`; if needed, guard with `callable(x)` before invocation.
3) Add explicit attribute type hints for `_why/_how/_what/_when/_anomaly` to concrete interfaces; use [typing.cast()](mypy_snapshots/mypy_snapshot_2025-08-26T10-42-16Z.txt:1) where necessary.
4) If placeholders exist (e.g., `Engine = object`), replace with Protocols or `TYPE_CHECKING` imports.
5) Add constructor/assignment guards to avoid calling non-callables.

Round B single additional change:
- Introduce typed factory helpers returning the specific sub-engine interfaces and assign attributes from them.

Acceptance criteria:
- Zero mypy errors for [src/orchestration/enhanced_intelligence_engine.py](src/orchestration/enhanced_intelligence_engine.py) after Round A under ruff/isort/black + mypy base + strict-on-touch.

---

## 2) [src/integration/component_integrator.py](src/integration/component_integrator.py)

Current errors (6):
- [src/integration/component_integrator.py](src/integration/component_integrator.py:23): Unused "type: ignore" comment  [unused-ignore]
- [src/integration/component_integrator.py](src/integration/component_integrator.py:25): Cannot assign to a type  [misc]
- [src/integration/component_integrator.py](src/integration/component_integrator.py:89): Unused "type: ignore" comment  [unused-ignore]
- [src/integration/component_integrator.py](src/integration/component_integrator.py:95): Unused "type: ignore" comment  [unused-ignore]
- [src/integration/component_integrator.py](src/integration/component_integrator.py:189): Unused "type: ignore" comment  [unused-ignore]
- [src/integration/component_integrator.py](src/integration/component_integrator.py:191): Unused "type: ignore" comment  [unused-ignore]

Categorization:
- Generics/typing misuse
- Any leaks (suppressed incorrectly)

Round A minimal changes (≤5):
1) Remove the five unused `# type: ignore[...]` comments.
2) Replace placeholder multi-assignments to types with properly typed Optional variables or Protocol references (no assignments to types).
3) If late-binding imports are intended, use `TYPE_CHECKING` imports and runtime imports separated from annotations.
4) Fix constructor calls to have correct arguments or wrap with [typing.cast()](mypy_snapshots/mypy_snapshot_2025-08-26T10-42-16Z.txt:1) if runtime-safe.
5) Annotate component registry as `dict[str, object]` or a small TypedDict.

Round B single additional change:
- Introduce a TypedDict/Protocol for component records to avoid repetitive casts.

Acceptance criteria:
- Zero mypy errors for [src/integration/component_integrator.py](src/integration/component_integrator.py) after Round A.

---

## 3) [src/trading/risk_management/__init__.py](src/trading/risk_management/__init__.py)

Current errors (5):
- [src/trading/risk_management/__init__.py](src/trading/risk_management/__init__.py:18): Module "src.core.risk.stress_testing" has no attribute "StressTester"  [attr-defined]
- [src/trading/risk_management/__init__.py](src/trading/risk_management/__init__.py:19): Module "src.core.risk.var_calculator" has no attribute "VarCalculator"  [attr-defined]
- [src/trading/risk_management/__init__.py](src/trading/risk_management/__init__.py:22): Cannot assign to a type  [misc]
- [src/trading/risk_management/__init__.py](src/trading/risk_management/__init__.py:22): Incompatible types in assignment (expression has type "type[object]", variable has type "type[RiskManager]")  [assignment]
- [src/trading/risk_management/__init__.py](src/trading/risk_management/__init__.py:24): All conditional function variants must have identical signatures  [misc]

Categorization:
- Any leaks (attr-defined)
- Typing misuse in module re-exports

Round A minimal changes (≤5):
1) Import correct public risk interfaces or re-export via stubs; otherwise `TYPE_CHECKING` + [typing.cast()](mypy_snapshots/mypy_snapshot_2025-08-26T10-42-16Z.txt:1).
2) Remove/replace `RiskManager = object`-style patterns with Protocols or interface aliases.
3) Unify `KellyCriterion` into a single typed function (no aliasing with differing signatures).
4) If aliasing a function, assign with `Callable[..., float]` and [typing.cast()](mypy_snapshots/mypy_snapshot_2025-08-26T10-42-16Z.txt:1) to avoid overload mismatch.
5) Provide minimal local Protocol placeholders if upstream types are intentionally not present.

Round B single additional change:
- Add a narrow stub in stubs/ for missing attributes to satisfy attr-defined.

Acceptance criteria:
- Zero mypy errors for [src/trading/risk_management/__init__.py](src/trading/risk_management/__init__.py) after Round A.

---

## 4) [src/orchestration/compose.py](src/orchestration/compose.py)

Current errors (5):
- [src/orchestration/compose.py](src/orchestration/compose.py:139): Unused "type: ignore" comment  [unused-ignore]
- [src/orchestration/compose.py](src/orchestration/compose.py:275): Unused "type: ignore" comment  [unused-ignore]
- [src/orchestration/compose.py](src/orchestration/compose.py:328): Unused "type: ignore" comment  [unused-ignore]
- [src/orchestration/compose.py](src/orchestration/compose.py:351): Unused "type: ignore" comment  [unused-ignore]
- [src/orchestration/compose.py](src/orchestration/compose.py:376): Unused "type: ignore" comment  [unused-ignore]

Categorization:
- Typing hygiene (unused ignores)

Round A minimal changes (≤5):
1) Remove the five unused ignores above.
2) Where ignores hid attribute/index typing, add local `dict[str, object]`/`Mapping[str, object]` annotations.
3) For `asyncio.to_thread`, add a result variable with explicit type and [typing.cast()](mypy_snapshots/mypy_snapshot_2025-08-26T10-42-16Z.txt:1) if needed.
4) For `getattr` dynamic access, use `cast(Callable[..., T], getattr(...))` before call.
5) Normalize dictionary key access with `TypedDict` for the specific shapes if stable.

Round B single additional change:
- Add a small TypedDict for the object shape being indexed.

Acceptance criteria:
- Zero mypy errors for [src/orchestration/compose.py](src/orchestration/compose.py) after Round A.

---

## 5) [src/genome/models/genome_adapter.py](src/genome/models/genome_adapter.py)

Current errors (5):
- [src/genome/models/genome_adapter.py](src/genome/models/genome_adapter.py:30): Cannot assign to a type  [misc]
- [src/genome/models/genome_adapter.py](src/genome/models/genome_adapter.py:31): Incompatible types in assignment (expression has type "None", variable has type "Callable[[DecisionGenome, str, dict[str, float]], DecisionGenome]")  [assignment]
- [src/genome/models/genome_adapter.py](src/genome/models/genome_adapter.py:32): Incompatible types in assignment (expression has type "None", variable has type "Callable[[str, dict[str, float], int, str | None], DecisionGenome]")  [assignment]
- [src/genome/models/genome_adapter.py](src/genome/models/genome_adapter.py:40): Incompatible types in assignment (expression has type "None", variable has type "Callable[[Any], DecisionGenome]")  [assignment]
- [src/genome/models/genome_adapter.py](src/genome/models/genome_adapter.py:41): Incompatible types in assignment (expression has type "None", variable has type "Callable[[DecisionGenome], dict[str, Any]]")  [assignment]

Categorization:
- Generics/typing misuse

Round A minimal changes (≤5):
1) Replace `None` placeholders for callables with def-stubs raising `NotImplementedError` and correct signatures.
2) If injection is deferred, annotate as `Optional[Callable[...]]` and guard at call sites.
3) Avoid assignments to type aliases; use instance variables with values.
4) Use Protocol to define the adapter behavior and implement concrete functions.
5) Ensure return annotation `dict[str, Any]` for `_to_legacy_view` and use [typing.cast()](mypy_snapshots/mypy_snapshot_2025-08-26T10-42-16Z.txt:1) on dynamic paths.

Round B single additional change:
- Add a minimal DecisionGenome factory to remove Any in `_from_legacy`.

Acceptance criteria:
- Zero mypy errors for [src/genome/models/genome_adapter.py](src/genome/models/genome_adapter.py) after Round A.

---

## 6) [src/validation/validation_framework.py](src/validation/validation_framework.py)

Current errors (4):
- [src/validation/validation_framework.py](src/validation/validation_framework.py:27): Function is missing a return type annotation  [no-untyped-def]
- [src/validation/validation_framework.py](src/validation/validation_framework.py:28): Missing type parameters for generic type "Callable"  [type-arg]
- [src/validation/validation_framework.py](src/validation/validation_framework.py:56): Missing positional arguments "strategy_id", "symbols", "params" in call to "MovingAverageStrategy"  [call-arg]
- [src/validation/validation_framework.py](src/validation/validation_framework.py:412): Function is missing a return type annotation  [no-untyped-def]

Categorization:
- Untyped defs
- Generics
- Call arg mismatch

Round A minimal changes (≤5):
1) Add [-> None](mypy_snapshots/mypy_snapshot_2025-08-26T10-42-16Z.txt:1) to `__init__` and `async def main`.
2) Specify `self.validators: dict[str, Callable[[Mapping[str, object]], bool]]` (or exact signature).
3) Construct `MovingAverageStrategy` with required args; if not available, wrap with [typing.cast()](mypy_snapshots/mypy_snapshot_2025-08-26T10-42-16Z.txt:1) and provide defaults for diagnostics mode.
4) Annotate any payloads passed to validators as `dict[str, object]`.
5) Ensure any timestamp or numeric fields normalized via [float()](mypy_snapshots/mypy_snapshot_2025-08-26T10-42-16Z.txt:1)/[int()](mypy_snapshots/mypy_snapshot_2025-08-26T10-42-16Z.txt:1).

Round B single additional change:
- Provide a defaulted factory for `MovingAverageStrategy`.

Acceptance criteria:
- Zero mypy errors for [src/validation/validation_framework.py](src/validation/validation_framework.py) after Round A.

---

## 7) [src/ui/ui_manager.py](src/ui/ui_manager.py)

Current errors (4):
- [src/ui/ui_manager.py](src/ui/ui_manager.py:16): Module "src.governance.strategy_registry" has no attribute "StrategyStatus"  [attr-defined]
- [src/ui/ui_manager.py](src/ui/ui_manager.py:19): Name "EventBus" already defined (possibly by an import)  [no-redef]
- [src/ui/ui_manager.py](src/ui/ui_manager.py:32): Name "StrategyStatus" already defined (possibly by an import)  [no-redef]
- [src/ui/ui_manager.py](src/ui/ui_manager.py:38): Name "StrategyRegistry" already defined (possibly by an import)  [no-redef]

Categorization:
- Any leaks (attr-defined)
- Redeclaration errors

Round A minimal changes (≤5):
1) Remove local placeholders that shadow imported names; or rename local stubs with `_Stub` suffix.
2) If imported modules do not provide symbols at type time, use `TYPE_CHECKING` imports plus local Protocols for runtime.
3) Replace local placeholder classes with Protocols capturing used attributes only.
4) Unify the source of `StrategyStatus` and `StrategyRegistry` across the file.
5) Add narrow stubs under stubs/ (future action) if upstream types are missing.

Round B single additional change:
- Add a local Enum-style Protocol for `StrategyStatus` under `TYPE_CHECKING`.

Acceptance criteria:
- Zero mypy errors for [src/ui/ui_manager.py](src/ui/ui_manager.py) after Round A.

---

## 8) [src/trading/execution/fix_executor.py](src/trading/execution/fix_executor.py)

Current errors (4):
- [src/trading/execution/fix_executor.py](src/trading/execution/fix_executor.py:16): Module "src.core.interfaces" has no attribute "IExecutionEngine"  [attr-defined]
- [src/trading/execution/fix_executor.py](src/trading/execution/fix_executor.py:166): Non-overlapping container check (element type: "OrderType", container item type: "str")  [comparison-overlap]
- [src/trading/execution/fix_executor.py](src/trading/execution/fix_executor.py:200): Unsupported operand types for * ("None" and "float")  [operator]
- [src/trading/execution/fix_executor.py](src/trading/execution/fix_executor.py:204): Unsupported operand types for - ("None" and "float")  [operator]

Categorization:
- Any leaks (attr-defined)
- Optional misuse

Round A minimal changes (≤5):
1) Import correct interface or define local `IExecutionEngine` Protocol under `TYPE_CHECKING`.
2) Compare `order.order_type` against an Enum or `set[OrderType]` rather than `list[str]`.
3) Guard arithmetic with Optional numbers using `if x is None: ...` or [float()](mypy_snapshots/mypy_snapshot_2025-08-26T10-42-16Z.txt:1) coercion where semantically correct.
4) Normalize numeric mixing via [Decimal()](mypy_snapshots/mypy_snapshot_2025-08-26T10-42-16Z.txt:1) as needed for consistency.
5) Add local variable annotations for intermediate computations as `float`.

Round B single additional change:
- Add `safe_price(x: float | None) -> float` helper and use at call sites.

Acceptance criteria:
- Zero mypy errors for [src/trading/execution/fix_executor.py](src/trading/execution/fix_executor.py) after Round A.

---

## 9) [src/thinking/thinking_manager.py](src/thinking/thinking_manager.py)

Current errors (4):
- [src/thinking/thinking_manager.py](src/thinking/thinking_manager.py:32): Need type annotation for "market_data_buffer"  [var-annotated]
- [src/thinking/thinking_manager.py](src/thinking/thinking_manager.py:65): Argument 1 to "float" has incompatible type "object"  [arg-type]
- [src/thinking/thinking_manager.py](src/thinking/thinking_manager.py:66): Argument 1 to "float" has incompatible type "object"  [arg-type]
- [src/thinking/thinking_manager.py](src/thinking/thinking_manager.py:82): Argument "current_state" to "predict_market_scenarios" has incompatible type "dict[str, float]"; expected "dict[str, object]"  [arg-type]

Categorization:
- Optional misuse (numeric coercions)
- dict[str, object] assembly
- Missing annotations

Round A minimal changes (≤5):
1) Annotate `self.market_data_buffer: deque[dict[str, object]] = deque(maxlen=100)`.
2) Coerce numerics with [float()](mypy_snapshots/mypy_snapshot_2025-08-26T10-42-16Z.txt:1) using `market_data.get("...") or 0.0`.
3) Build `current_state: dict[str, object]` ensuring values are floats/objects as required by downstream signature.
4) Add local annotations for intermediate dicts to satisfy invariance.
5) Normalize timestamps if present to float epoch seconds.

Round B single additional change:
- Factor `normalize_state(market_data) -> dict[str, object]` helper.

Acceptance criteria:
- Zero mypy errors for [src/thinking/thinking_manager.py](src/thinking/thinking_manager.py) after Round A.

---

## 10) [src/sentient/sentient_predator.py](src/sentient/sentient_predator.py)

Current errors (4):
- [src/sentient/sentient_predator.py](src/sentient/sentient_predator.py:40): Function is missing a return type annotation  [no-untyped-def]
- [src/sentient/sentient_predator.py](src/sentient/sentient_predator.py:45): Function is missing a return type annotation  [no-untyped-def]
- [src/sentient/sentient_predator.py](src/sentient/sentient_predator.py:129): Returning Any from function declared to return "ndarray[Any, dtype[Any]]"  [no-any-return]
- [src/sentient/sentient_predator.py](src/sentient/sentient_predator.py:178): Function is missing a return type annotation  [no-untyped-def]

Categorization:
- Untyped defs
- Any leaks

Round A minimal changes (≤5):
1) Add [-> None](mypy_snapshots/mypy_snapshot_2025-08-26T10-42-16Z.txt:1) to `async def start`, `async def stop`, and `async def reset`.
2) Ensure vector function returns `np.ndarray`: wrap with `np.asarray(..., dtype=float)`.
3) Add local annotation for `vector` as `np.ndarray` where built incrementally.
4) Coerce Optional numeric components via [float()](mypy_snapshots/mypy_snapshot_2025-08-26T10-42-16Z.txt:1) before array creation.
5) Ensure async procedures explicitly return `None`.

Round B single additional change:
- Add `to_ndarray(values: Sequence[float | int]) -> np.ndarray` helper.

Acceptance criteria:
- Zero mypy errors for [src/sentient/sentient_predator.py](src/sentient/sentient_predator.py) after Round A.

---

## 11) [src/ecosystem/evaluation/niche_detector.py](src/ecosystem/evaluation/niche_detector.py)

Current errors (4):
- [src/ecosystem/evaluation/niche_detector.py](src/ecosystem/evaluation/niche_detector.py:142): Need type annotation for "x"  [var-annotated]
- [src/ecosystem/evaluation/niche_detector.py](src/ecosystem/evaluation/niche_detector.py:187): Incompatible return value type (got "Series[float]", expected "Series[str]")  [return-value]
- [src/ecosystem/evaluation/niche_detector.py](src/ecosystem/evaluation/niche_detector.py:209): Incompatible types in assignment (expression has type "Series[float]", variable has type "Series[str]")  [assignment]
- [src/ecosystem/evaluation/niche_detector.py](src/ecosystem/evaluation/niche_detector.py:210): Argument 1 to "update" of "Series" has incompatible type "Series[float]"; expected "Series[str] | Sequence[str] | Mapping[int, str]"  [arg-type]

Categorization:
- Missing annotations
- Return type mismatch
- Generics

Round A minimal changes (≤5):
1) Annotate `x: np.ndarray = np.arange(len(arr), dtype=float)`.
2) Convert scoring results to regime label strings prior to returning `pd.Series[str]`.
3) When building `full_regimes: pd.Series[str]`, convert `regimes.astype(str)` before assignment and update.
4) Ensure index types for `features` are compatible (e.g., `Sequence[str]` or `pd.Index`).
5) Add specific type args on intermediate Pandas Series creations.

Round B single additional change:
- Helper `to_regime_labels(scores: pd.Series[float]) -> pd.Series[str]`.

Acceptance criteria:
- Zero mypy errors for [src/ecosystem/evaluation/niche_detector.py](src/ecosystem/evaluation/niche_detector.py) after Round A.

---

## 12) [src/core/population_manager.py](src/core/population_manager.py)

Current errors (4):
- [src/core/population_manager.py](src/core/population_manager.py:46): List comprehension has incompatible type List[object]; expected List[DecisionGenome]  [misc]
- [src/core/population_manager.py](src/core/population_manager.py:77): List comprehension has incompatible type List[object]; expected List[DecisionGenome]  [misc]
- [src/core/population_manager.py](src/core/population_manager.py:192): Argument 1 to "append" of "list" has incompatible type "object"; expected "DecisionGenome"  [arg-type]
- [src/core/population_manager.py](src/core/population_manager.py:227): Argument 1 to "append" of "list" has incompatible type "object"; expected "DecisionGenome"  [arg-type]

Categorization:
- Generics
- Any narrowing

Round A minimal changes (≤5):
1) Annotate `self.population: list[DecisionGenome]`.
2) Narrow provider outputs with `isinstance(..., DecisionGenome)` or [typing.cast()](mypy_snapshots/mypy_snapshot_2025-08-26T10-42-16Z.txt:1) where validated.
3) Ensure `new_population: list[DecisionGenome]` in comprehensions and map functions.
4) Validate before `append` and adapt/convert to `DecisionGenome` or raise.
5) Provide adapter `from_legacy(...) -> DecisionGenome` with correct signature usage.

Round B single additional change:
- Add `ensure_genome(x: object) -> DecisionGenome` helper for central narrowing.

Acceptance criteria:
- Zero mypy errors for [src/core/population_manager.py](src/core/population_manager.py) after Round A.

---

## Error tally for Fix7 set

Total files planned: 12  
Total errors across these 12 files: 55

Breakdown (from snapshot lines cited above):
- [src/orchestration/enhanced_intelligence_engine.py](src/orchestration/enhanced_intelligence_engine.py:1): 6
- [src/integration/component_integrator.py](src/integration/component_integrator.py:1): 6
- [src/trading/risk_management/__init__.py](src/trading/risk_management/__init__.py:1): 5
- [src/orchestration/compose.py](src/orchestration/compose.py:1): 5
- [src/genome/models/genome_adapter.py](src/genome/models/genome_adapter.py:1): 5
- [src/validation/validation_framework.py](src/validation/validation_framework.py:1): 4
- [src/ui/ui_manager.py](src/ui/ui_manager.py:1): 4
- [src/trading/execution/fix_executor.py](src/trading/execution/fix_executor.py:1): 4
- [src/thinking/thinking_manager.py](src/thinking/thinking_manager.py:1): 4
- [src/sentient/sentient_predator.py](src/sentient/sentient_predator.py:1): 4
- [src/ecosystem/evaluation/niche_detector.py](src/ecosystem/evaluation/niche_detector.py:1): 4
- [src/core/population_manager.py](src/core/population_manager.py:1): 4