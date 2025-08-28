# Batch10 Fix6 Diagnostics Plan — 2025-08-26

Context: Latest snapshot [mypy_snapshots/mypy_snapshot_2025-08-26T09-26-46Z.txt](mypy_snapshots/mypy_snapshot_2025-08-26T09-26-46Z.txt:1) and summary [mypy_snapshots/mypy_summary_2025-08-26T09-26-46Z.txt](mypy_snapshots/mypy_summary_2025-08-26T09-26-46Z.txt:1). Candidates confirmed from [changed_files_batch10_fix6_candidates.txt](changed_files_batch10_fix6_candidates.txt:1).

Snapshot totals/delta: Errors=341, Files=73; Delta vs previous: errors=-84, files=-3. This plan covers 12 files with 96 snapshot errors.

Scope and method:
- Diagnostics-only; no source edits. Deliverable is this plan file.
- Grepped errors per file from the raw snapshot cited above, using exact line numbers.
- Propose Round A minimal, behavior-preserving edits (≤5 per file) and a Round B placeholder.

Global acceptance criteria:
- This plan compiles sections for all 12 candidates, with clickable file references [path.py](path.py:line).
- For each file: after Round A changes, ruff/black/isort and mypy base + strict-on-touch on the edited file(s) yield zero mypy errors for that file in isolation.
- No source code changes have been made by this task; only this plan is produced.

Files covered:
- [src/thinking/patterns/cycle_detector.py](src/thinking/patterns/cycle_detector.py:1)
- [src/validation/phase2d_integration_validator.py](src/validation/phase2d_integration_validator.py:1)
- [src/validation/phase2c_validation_suite.py](src/validation/phase2c_validation_suite.py:1)
- [src/thinking/analysis/correlation_analyzer.py](src/thinking/analysis/correlation_analyzer.py:1)
- [src/evolution/ambusher/ambusher_orchestrator.py](src/evolution/ambusher/ambusher_orchestrator.py:1)
- [src/validation/real_market_validation.py](src/validation/real_market_validation.py:1)
- [src/validation/honest_validation_framework.py](src/validation/honest_validation_framework.py:1)
- [src/sensory/organs/dimensions/macro_intelligence.py](src/sensory/organs/dimensions/macro_intelligence.py:1)
- [src/genome/models/genome.py](src/genome/models/genome.py:1)
- [src/ecosystem/optimization/ecosystem_optimizer.py](src/ecosystem/optimization/ecosystem_optimizer.py:1)
- [src/trading/trading_manager.py](src/trading/trading_manager.py:1)
- [src/thinking/adversarial/market_gan.py](src/thinking/adversarial/market_gan.py:1)

-----

## 1) [src/thinking/patterns/cycle_detector.py](src/thinking/patterns/cycle_detector.py:1)

Current mypy errors: 10

Error entries:
- [src/thinking/patterns/cycle_detector.py](src/thinking/patterns/cycle_detector.py:18): Cannot assign to a type  [misc]
- [src/thinking/patterns/cycle_detector.py](src/thinking/patterns/cycle_detector.py:18): Incompatible types in assignment (expression has type type[object], variable has type type[ThinkingPattern])  [assignment]
- [src/thinking/patterns/cycle_detector.py](src/thinking/patterns/cycle_detector.py:18): Incompatible types in assignment (expression has type type[object], variable has type type[SensorySignal])  [assignment]
- [src/thinking/patterns/cycle_detector.py](src/thinking/patterns/cycle_detector.py:18): Incompatible types in assignment (expression has type type[object], variable has type type[AnalysisResult])  [assignment]
- [src/thinking/patterns/cycle_detector.py](src/thinking/patterns/cycle_detector.py:19): Module src.core.exceptions has no attribute ThinkingException; maybe TradingException?  [attr-defined]
- [src/thinking/patterns/cycle_detector.py](src/thinking/patterns/cycle_detector.py:64): Extra keys (timestamp, analysis_type, result, confidence, metadata) for TypedDict AnalysisResult  [typeddict-unknown-key]
- [src/thinking/patterns/cycle_detector.py](src/thinking/patterns/cycle_detector.py:84): Argument 1 of learn is incompatible with supertype src.core.interfaces.ThinkingPattern; supertype defines the argument type as Mapping[str, object]  [override]
- [src/thinking/patterns/cycle_detector.py](src/thinking/patterns/cycle_detector.py:116): SensorySignal has no attribute signal_type  [attr-defined]
- [src/thinking/patterns/cycle_detector.py](src/thinking/patterns/cycle_detector.py:133): SensorySignal has no attribute value  [attr-defined]
- [src/thinking/patterns/cycle_detector.py](src/thinking/patterns/cycle_detector.py:320): Extra keys (timestamp, analysis_type, result, confidence, metadata) for TypedDict AnalysisResult  [typeddict-unknown-key]

Error categories:
- Invalid sentinel type assignment
- Missing/incorrect exceptions import
- TypedDict key shape mismatch
- Incompatible override vs supertype (Mapping vs dict)
- Attribute access on loosely-typed SensorySignal

Round A minimal changes (≤5):
- Replace the dummy sentinel assignment at line 18 with proper typing: remove the object assignment, import types if available, and otherwise use from typing import Any and cast where accessed. This removes the misc/assignment errors around ThinkingPattern, SensorySignal, AnalysisResult.
- Update learn signature to use Mapping[str, object] for feedback (line 84) to align with the interface.
- For attribute access, cast signals to Any at use sites to satisfy the attribute reads without runtime change, e.g., float(cast(Any, s).value) and cast(Any, signal).signal_type in the guarded branches.
- For returns currently constructed as AnalysisResult(...), switch to returning a plain dict[str, object] or wrap the dict literal in a typing.cast to the expected alias in this file to eliminate TypedDict key errors.
- Import TradingException as ThinkingException or guard imports to avoid the attr-defined error on ThinkingException.

Round B single smallest additional change (if needed later):
- If mypy still flags SensorySignal attribute access, introduce a local Protocol for SensorySignal in a TYPE_CHECKING block and use typing.cast to that Protocol at access sites.

Acceptance criteria for this file:
- After Round A changes, mypy reports zero errors for [src/thinking/patterns/cycle_detector.py](src/thinking/patterns/cycle_detector.py:1) under base and strict-on-touch settings.

-----

## 2) [src/validation/phase2d_integration_validator.py](src/validation/phase2d_integration_validator.py:1)

Current mypy errors: 9

Error entries:
- [src/validation/phase2d_integration_validator.py](src/validation/phase2d_integration_validator.py:47): Need type annotation for results (hint: results: list[<type>] = ...)  [var-annotated]
- [src/validation/phase2d_integration_validator.py](src/validation/phase2d_integration_validator.py:67): Argument 1 to len has incompatible type object; expected Sized  [arg-type]
- [src/validation/phase2d_integration_validator.py](src/validation/phase2d_integration_validator.py:80): Argument 1 to detect_regime of RegimeClassifier has incompatible type object; expected Mapping[str, object]  [arg-type]
- [src/validation/phase2d_integration_validator.py](src/validation/phase2d_integration_validator.py:93): Cannot instantiate protocol class DecisionGenome  [misc]
- [src/validation/phase2d_integration_validator.py](src/validation/phase2d_integration_validator.py:96): Argument 2 to _evaluate_genome_with_real_data has incompatible type object; expected DataFrame  [arg-type]
- [src/validation/phase2d_integration_validator.py](src/validation/phase2d_integration_validator.py:173): Argument 1 to len has incompatible type object; expected Sized  [arg-type]
- [src/validation/phase2d_integration_validator.py](src/validation/phase2d_integration_validator.py:181): Unsupported target for indexed assignment object  [index]
- [src/validation/phase2d_integration_validator.py](src/validation/phase2d_integration_validator.py:181): Value of type object is not indexable  [index]
- [src/validation/phase2d_integration_validator.py](src/validation/phase2d_integration_validator.py:182): object has no attribute dropna  [attr-defined]

Error categories:
- Missing variable type annotation
- Pandas DataFrame typing and size checks on object
- Protocol instantiation
- Incompatible argument types (Mapping vs object)
- DataFrame indexing methods on object

Round A minimal changes (≤5):
- Annotate results as list[dict[str, object]] (or a dedicated TypedDict if available) at assignment.
- Where len(data) is used, ensure data: pd.DataFrame by adding an isinstance check or a typing.cast to DataFrame immediately prior to sizing/indexing; then operate on the typed variable.
- When calling detect_regime, pass a Mapping[str, object] (e.g., data.to_dict(orient='list')) or cast(test_data, Mapping[str, object]) if the structure is already mapping-like.
- Replace DecisionGenome() placeholder instantiation with a typing.cast(DecisionGenome, object()) or fetch a real genome from a provider/factory if available in scope.
- Before using DataFrame indexing and dropna, cast data to pd.DataFrame and assign to a typed local to satisfy the index/dropna operations.

Round B single smallest additional change (if needed later):
- If DataFrame operations still flagged, add a narrow helper: def _as_df(x: object) -> pd.DataFrame: return cast(pd.DataFrame, x) and reuse before DataFrame-only operations.

Acceptance criteria for this file:
- After Round A changes, mypy reports zero errors for [src/validation/phase2d_integration_validator.py](src/validation/phase2d_integration_validator.py:1).

-----

## 3) [src/validation/phase2c_validation_suite.py](src/validation/phase2c_validation_suite.py:1)

Current mypy errors: 9

Error entries:
- [src/validation/phase2c_validation_suite.py](src/validation/phase2c_validation_suite.py:27): Function is missing a return type annotation  [no-untyped-def]
- [src/validation/phase2c_validation_suite.py](src/validation/phase2c_validation_suite.py:84): Sequence[str] has no attribute append  [attr-defined]
- [src/validation/phase2c_validation_suite.py](src/validation/phase2c_validation_suite.py:88): Sequence[str] has no attribute append  [attr-defined]
- [src/validation/phase2c_validation_suite.py](src/validation/phase2c_validation_suite.py:92): Sequence[str] has no attribute append  [attr-defined]
- [src/validation/phase2c_validation_suite.py](src/validation/phase2c_validation_suite.py:95): Generator has incompatible item type int; expected bool  [misc]
- [src/validation/phase2c_validation_suite.py](src/validation/phase2c_validation_suite.py:95): Invalid index type str for str; expected type SupportsIndex | slice[Any, Any, Any]  [index]
- [src/validation/phase2c_validation_suite.py](src/validation/phase2c_validation_suite.py:98): Incompatible types in assignment (expression has type dict[str, float | int | str], target has type Sequence[str])  [assignment]
- [src/validation/phase2c_validation_suite.py](src/validation/phase2c_validation_suite.py:149): Function is missing a return type annotation  [no-untyped-def]
- [src/validation/phase2c_validation_suite.py](src/validation/phase2c_validation_suite.py:180): Function is missing a return type annotation  [no-untyped-def]

Error categories:
- Missing -> None on procedures
- Using Sequence where a mutable list is required
- Mis-typed dict vs Sequence fields
- Summation/generator typing

Round A minimal changes (≤5):
- Add -> None to __init__ and to the print/reporting and main coroutine functions.
- Change accuracy_results['tests'] to a mutable list type, e.g., list[dict[str, object]], and initialize accordingly so .append is valid.
- Ensure passed calculation sums booleans or ints without conflicting expected types; for example, passed: int = sum(1 for test in accuracy_results['tests'] if bool(test.get('passed'))) to keep generator items as int consistently.
- Correct accuracy_results['summary'] to be a dict[str, object] rather than a Sequence, matching how it is populated.
- If any intermediate variables are implicitly typed as Sequence[str], annotate concrete list types to avoid invariance issues.

Round B single smallest additional change (if needed later):
- If needed, replace generator expression with explicit loop accumulating an int to sidestep generator typing constraints.

Acceptance criteria for this file:
- After Round A changes, mypy reports zero errors for [src/validation/phase2c_validation_suite.py](src/validation/phase2c_validation_suite.py:1).

-----

## 4) [src/thinking/analysis/correlation_analyzer.py](src/thinking/analysis/correlation_analyzer.py:1)

Current mypy errors: 9

Error entries:
- [src/thinking/analysis/correlation_analyzer.py](src/thinking/analysis/correlation_analyzer.py:16): Module src.core.exceptions has no attribute ThinkingException; maybe TradingException?  [attr-defined]
- [src/thinking/analysis/correlation_analyzer.py](src/thinking/analysis/correlation_analyzer.py:21): Cannot assign to a type  [misc]
- [src/thinking/analysis/correlation_analyzer.py](src/thinking/analysis/correlation_analyzer.py:21): Incompatible types in assignment (expression has type type[object], variable has type type[ThinkingPattern])  [assignment]
- [src/thinking/analysis/correlation_analyzer.py](src/thinking/analysis/correlation_analyzer.py:21): Incompatible types in assignment (expression has type type[object], variable has type type[SensorySignal])  [assignment]
- [src/thinking/analysis/correlation_analyzer.py](src/thinking/analysis/correlation_analyzer.py:21): Incompatible types in assignment (expression has type type[object], variable has type type[AnalysisResult])  [assignment]
- [src/thinking/analysis/correlation_analyzer.py](src/thinking/analysis/correlation_analyzer.py:53): Extra keys (timestamp, analysis_type, result, confidence, metadata) for TypedDict AnalysisResult  [typeddict-unknown-key]
- [src/thinking/analysis/correlation_analyzer.py](src/thinking/analysis/correlation_analyzer.py:76): Argument 1 of learn is incompatible with supertype src.core.interfaces.ThinkingPattern; supertype defines the argument type as Mapping[str, object]  [override]
- [src/thinking/analysis/correlation_analyzer.py](src/thinking/analysis/correlation_analyzer.py:98): SensorySignal has no attribute signal_type  [attr-defined]
- [src/thinking/analysis/correlation_analyzer.py](src/thinking/analysis/correlation_analyzer.py:103): SensorySignal has no attribute value  [attr-defined]

Error categories:
- Invalid sentinel type assignment
- Missing/incorrect exceptions import
- TypedDict key shape mismatch
- Incompatible override vs supertype (Mapping vs dict)
- Attribute access on loosely-typed SensorySignal

Round A minimal changes (≤5):
- Remove the object sentinel assignment at line 21 and replace with proper imports or Any+cast usage at attribute access sites.
- Update learn signature to accept Mapping[str, object].
- Replace AnalysisResult(...) returns with a typed dict[str, object] and, if needed, typing.cast to local AnalysisResult alias.
- Coerce/cast SensorySignal attributes when recording into history, e.g., self._signal_history[signal_type].append(float(cast(Any, signal).value)).
- Import TradingException as ThinkingException or change the exception type reference accordingly.

Round B single smallest additional change (if needed later):
- If history container is untyped, annotate it as dict[str, list[float]] and initialize eagerly to avoid Any-inference.

Acceptance criteria for this file:
- After Round A changes, mypy reports zero errors for [src/thinking/analysis/correlation_analyzer.py](src/thinking/analysis/correlation_analyzer.py:1).

-----

## 5) [src/evolution/ambusher/ambusher_orchestrator.py](src/evolution/ambusher/ambusher_orchestrator.py:1)

Current mypy errors: 9

Error entries:
- [src/evolution/ambusher/ambusher_orchestrator.py](src/evolution/ambusher/ambusher_orchestrator.py:20): Function is missing a type annotation  [no-untyped-def]
- [src/evolution/ambusher/ambusher_orchestrator.py](src/evolution/ambusher/ambusher_orchestrator.py:33): Too many arguments for AmbusherFitnessFunction  [call-arg]
- [src/evolution/ambusher/ambusher_orchestrator.py](src/evolution/ambusher/ambusher_orchestrator.py:54): Function is missing a return type annotation (async start)  [no-untyped-def]
- [src/evolution/ambusher/ambusher_orchestrator.py](src/evolution/ambusher/ambusher_orchestrator.py:61): EvolutionEngine has no attribute load_genome  [attr-defined]
- [src/evolution/ambusher/ambusher_orchestrator.py](src/evolution/ambusher/ambusher_orchestrator.py:66): Function is missing a return type annotation (async stop)  [no-untyped-def]
- [src/evolution/ambusher/ambusher_orchestrator.py](src/evolution/ambusher/ambusher_orchestrator.py:80): evolve of EvolutionEngine does not return a value (it only ever returns None)  [func-returns-value]
- [src/evolution/ambusher/ambusher_orchestrator.py](src/evolution/ambusher/ambusher_orchestrator.py:119): Returning Any from function declared to return dict[str, Any] | None  [no-any-return]
- [src/evolution/ambusher/ambusher_orchestrator.py](src/evolution/ambusher/ambusher_orchestrator.py:131): Function is missing a return type annotation (async reset)  [no-untyped-def]
- [src/evolution/ambusher/ambusher_orchestrator.py](src/evolution/ambusher/ambusher_orchestrator.py:155): Function is missing a type annotation (add -> None)  [no-untyped-def]

Error categories:
- Missing -> None on coroutines/procedures
- Call signature mismatch
- Absent attribute on engine API
- Any leaking from to_dict or similar

Round A minimal changes (≤5):
- Add -> None to __init__, start, stop, reset, and update_trade_metrics.
- Adjust AmbusherFitnessFunction(...) invocation to match its supported signature (e.g., pass config.get('fitness', {})) or reduce positional arguments to the expected set.
- For engine.load_genome, avoid attribute error by using typing.cast(Any, self.genetic_engine).load_genome(...) unless a correct API name exists.
- Remove the unused assignment of the evolve() result; just call self.genetic_engine.evolve() without capturing a value.
- When returning self.current_genome.to_dict(), wrap with typing.cast(dict[str, Any], self.current_genome.to_dict()) to avoid Any.

Round B single smallest additional change (if needed later):
- If AmbusherFitnessFunction construction remains mismatched, inject a thin factory adapter locally to normalize the argument shape.

Acceptance criteria for this file:
- After Round A changes, mypy reports zero errors for [src/evolution/ambusher/ambusher_orchestrator.py](src/evolution/ambusher/ambusher_orchestrator.py:1).

-----

## 6) [src/validation/real_market_validation.py](src/validation/real_market_validation.py:1)

Current mypy errors: 8

Error entries:
- [src/validation/real_market_validation.py](src/validation/real_market_validation.py:141): Returning Any from function declared to return datetime | None  [no-any-return]
- [src/validation/real_market_validation.py](src/validation/real_market_validation.py:242): Argument 1 to detect_regime of RegimeClassifier has incompatible type DataFrame; expected Mapping[str, object]  [arg-type]
- [src/validation/real_market_validation.py](src/validation/real_market_validation.py:264): Unsupported operand types for <= (datetime and DatetimeIndex)  [operator]
- [src/validation/real_market_validation.py](src/validation/real_market_validation.py:264): Unsupported operand types for <= (DatetimeIndex and datetime)  [operator]
- [src/validation/real_market_validation.py](src/validation/real_market_validation.py:265): object has no attribute upper  [attr-defined]
- [src/validation/real_market_validation.py](src/validation/real_market_validation.py:342): Argument 1 to detect_regime of RegimeClassifier has incompatible type DataFrame; expected Mapping[str, object]  [arg-type]
- [src/validation/real_market_validation.py](src/validation/real_market_validation.py:659): Function is missing a return type annotation  [no-untyped-def]
- [src/validation/real_market_validation.py](src/validation/real_market_validation.py:691): Function is missing a return type annotation  [no-untyped-def]

Error categories:
- Any result from pandas conversion
- Mapping vs DataFrame typing
- Datetime vs DatetimeIndex comparison
- String handling on object
- Missing -> None on print/main

Round A minimal changes (≤5):
- In the timestamp normalization method at line 141, cast the pandas result to datetime: return cast(datetime, pd.to_datetime(ts).to_pydatetime()).
- Before detect_regime calls, pass a Mapping by using data.to_dict(orient='list') or cast(data, Mapping[str, object]) if appropriate.
- Normalize regime_date to a datetime before comparison: regime_date_dt = pd.to_datetime(regime_date).to_pydatetime(); compare crisis_start <= regime_date_dt <= crisis_end.
- When upper() is used, ensure the value is str: key = regime['regime']; if isinstance(key, str): use key.upper(); else coerce with str(key).upper().
- Add -> None to print_comprehensive_report and to async main().

Round B single smallest additional change (if needed later):
- If crisis_start/crisis_end are pandas Timestamps, convert both sides to comparable native datetime via .to_pydatetime() prior to comparison.

Acceptance criteria for this file:
- After Round A changes, mypy reports zero errors for [src/validation/real_market_validation.py](src/validation/real_market_validation.py:1).

-----

## 7) [src/validation/honest_validation_framework.py](src/validation/honest_validation_framework.py:1)

Current mypy errors: 8

Error entries:
- [src/validation/honest_validation_framework.py](src/validation/honest_validation_framework.py:176): Argument 1 to detect_regime of RegimeClassifier has incompatible type DataFrame; expected Mapping[str, object]  [arg-type]
- [src/validation/honest_validation_framework.py](src/validation/honest_validation_framework.py:216): Unused type: ignore comment  [unused-ignore]
- [src/validation/honest_validation_framework.py](src/validation/honest_validation_framework.py:216): Cannot instantiate protocol class DecisionGenome  [misc]
- [src/validation/honest_validation_framework.py](src/validation/honest_validation_framework.py:236): Unused type: ignore comment  [unused-ignore]
- [src/validation/honest_validation_framework.py](src/validation/honest_validation_framework.py:277): Unused type: ignore comment  [unused-ignore]
- [src/validation/honest_validation_framework.py](src/validation/honest_validation_framework.py:325): Unused type: ignore comment  [unused-ignore]
- [src/validation/honest_validation_framework.py](src/validation/honest_validation_framework.py:436): Function is missing a return type annotation  [no-untyped-def]
- [src/validation/honest_validation_framework.py](src/validation/honest_validation_framework.py:461): Function is missing a return type annotation  [no-untyped-def]

Error categories:
- Mapping vs DataFrame typing
- Unnecessary type: ignore usage
- Protocol instantiation
- Missing -> None on procedures

Round A minimal changes (≤5):
- Provide Mapping[str, object] to detect_regime (e.g., data.to_dict(orient='list') or typing.cast).
- Replace DecisionGenome() placeholder with typing.cast(DecisionGenome, object()); remove the associated # type: ignore comments.
- Remove the flagged unused type: ignore comments at lines 236, 277, 325.
- Add -> None to print_report and async main().
- If any other # type: ignore remain around these lines, replace with narrow casting instead of suppressions.

Round B single smallest additional change (if needed later):
- If DecisionGenome interactions require attributes, introduce a lightweight NamedTuple or dataclass locally and cast to DecisionGenome for tests-only scaffolding.

Acceptance criteria for this file:
- After Round A changes, mypy reports zero errors for [src/validation/honest_validation_framework.py](src/validation/honest_validation_framework.py:1).

-----

## 8) [src/sensory/organs/dimensions/macro_intelligence.py](src/sensory/organs/dimensions/macro_intelligence.py:1)

Current mypy errors: 8

Error entries:
- [src/sensory/organs/dimensions/macro_intelligence.py](src/sensory/organs/dimensions/macro_intelligence.py:303): Returning Any from function declared to return MarketRegime  [no-any-return]
- [src/sensory/organs/dimensions/macro_intelligence.py](src/sensory/organs/dimensions/macro_intelligence.py:303): type[MarketRegime] has no attribute HIGH_VOLATILITY  [attr-defined]
- [src/sensory/organs/dimensions/macro_intelligence.py](src/sensory/organs/dimensions/macro_intelligence.py:305): Returning Any from function declared to return MarketRegime  [no-any-return]
- [src/sensory/organs/dimensions/macro_intelligence.py](src/sensory/organs/dimensions/macro_intelligence.py:305): type[MarketRegime] has no attribute BULLISH  [attr-defined]
- [src/sensory/organs/dimensions/macro_intelligence.py](src/sensory/organs/dimensions/macro_intelligence.py:307): Returning Any from function declared to return MarketRegime  [no-any-return]
- [src/sensory/organs/dimensions/macro_intelligence.py](src/sensory/organs/dimensions/macro_intelligence.py:307): type[MarketRegime] has no attribute BEARISH  [attr-defined]
- [src/sensory/organs/dimensions/macro_intelligence.py](src/sensory/organs/dimensions/macro_intelligence.py:309): Returning Any from function declared to return MarketRegime  [no-any-return]
- [src/sensory/organs/dimensions/macro_intelligence.py](src/sensory/organs/dimensions/macro_intelligence.py:309): type[MarketRegime] has no attribute RANGING  [attr-defined]

Error categories:
- Returning Any instead of MarketRegime
- Enum attribute names not present on MarketRegime stub/type

Round A minimal changes (≤5):
- Add a local helper def _to_market_regime(name: str) -> MarketRegime: return cast(MarketRegime, getattr(MarketRegime, name, getattr(MarketRegime, 'UNKNOWN', name))).
- Replace return MarketRegime.HIGH_VOLATILITY with return _to_market_regime('HIGH_VOLATILITY').
- Replace return MarketRegime.BULLISH with return _to_market_regime('BULLISH').
- Replace return MarketRegime.BEARISH with return _to_market_regime('BEARISH').
- Replace return MarketRegime.RANGING with return _to_market_regime('RANGING').

Round B single smallest additional change (if needed later):
- If mypy still flags Any, explicitly cast each call site return as cast(MarketRegime, _to_market_regime(...)).

Acceptance criteria for this file:
- After Round A changes, mypy reports zero errors for [src/sensory/organs/dimensions/macro_intelligence.py](src/sensory/organs/dimensions/macro_intelligence.py:1).

-----

## 9) [src/genome/models/genome.py](src/genome/models/genome.py:1)

Current mypy errors: 7

Error entries:
- [src/genome/models/genome.py](src/genome/models/genome.py:237): Argument 1 to DecisionGenome has incompatible type **dict[str, list[str] | dict[str, float] | float | str | None]; expected str  [arg-type]
- [src/genome/models/genome.py](src/genome/models/genome.py:237): Argument 1 to DecisionGenome has incompatible type **dict[str, list[str] | dict[str, float] | float | str | None]; expected dict[str, float]  [arg-type]
- [src/genome/models/genome.py](src/genome/models/genome.py:237): Argument 1 to DecisionGenome has incompatible type **dict[str, list[str] | dict[str, float] | float | str | None]; expected float | None  [arg-type]
- [src/genome/models/genome.py](src/genome/models/genome.py:237): Argument 1 to DecisionGenome has incompatible type **dict[str, list[str] | dict[str, float] | float | str | None]; expected int  [arg-type]
- [src/genome/models/genome.py](src/genome/models/genome.py:237): Argument 1 to DecisionGenome has incompatible type **dict[str, list[str] | dict[str, float] | float | str | None]; expected str | None  [arg-type]
- [src/genome/models/genome.py](src/genome/models/genome.py:237): Argument 1 to DecisionGenome has incompatible type **dict[str, list[str] | dict[str, float] | float | str | None]; expected list[str]  [arg-type]
- [src/genome/models/genome.py](src/genome/models/genome.py:237): Argument 1 to DecisionGenome has incompatible type **dict[str, list[str] | dict[str, float] | float | str | None]; expected float  [arg-type]

Error categories:
- Passing heterogeneous dict via ** to a strongly-typed constructor

Round A minimal changes (≤5):
- At line 237, cast the dict before expansion to a permissive type to satisfy the constructor: return DecisionGenome(**cast(dict[str, Any], data)). This avoids variance issues across multiple fields with a single change.

Round B single smallest additional change (if needed later):
- If available, replace data with a well-typed dataclass/TypedDict and ensure field-by-field construction, otherwise keep the cast.

Acceptance criteria for this file:
- After Round A changes, mypy reports zero errors for [src/genome/models/genome.py](src/genome/models/genome.py:1).

-----

## 10) [src/ecosystem/optimization/ecosystem_optimizer.py](src/ecosystem/optimization/ecosystem_optimizer.py:1)

Current mypy errors: 7

Error entries:
- [src/ecosystem/optimization/ecosystem_optimizer.py](src/ecosystem/optimization/ecosystem_optimizer.py:76): Return type Coroutine[Any, Any, dict[str, list[src.genome.models.genome.DecisionGenome]]] incompatible with return type Coroutine[Any, Any, Mapping[str, Sequence[src.core.interfaces.DecisionGenome]]] in supertype  [override]
- [src/ecosystem/optimization/ecosystem_optimizer.py](src/ecosystem/optimization/ecosystem_optimizer.py:78): Argument 1 of optimize_ecosystem is incompatible with supertype (supertype defines Mapping[str, Sequence[DecisionGenome]])  [override]
- [src/ecosystem/optimization/ecosystem_optimizer.py](src/ecosystem/optimization/ecosystem_optimizer.py:201): Incompatible types in assignment (expression has type DecisionGenome | None, variable has type DecisionGenome)  [assignment]
- [src/ecosystem/optimization/ecosystem_optimizer.py](src/ecosystem/optimization/ecosystem_optimizer.py:440): Unused type: ignore comment  [unused-ignore]
- [src/ecosystem/optimization/ecosystem_optimizer.py](src/ecosystem/optimization/ecosystem_optimizer.py:520): Function is missing a type annotation for one or more arguments  [no-untyped-def]
- [src/ecosystem/optimization/ecosystem_optimizer.py](src/ecosystem/optimization/ecosystem_optimizer.py:531): Function is missing a type annotation for one or more arguments  [no-untyped-def]
- [src/ecosystem/optimization/ecosystem_optimizer.py](src/ecosystem/optimization/ecosystem_optimizer.py:546): Returning Any from function declared to return float  [no-any-return]

Error categories:
- Incompatible override signature (Mapping/Sequence vs concrete types)
- Optional assignment to non-Optional target
- Unnecessary type: ignore
- Missing annotations on helper methods
- Any returning from computation

Round A minimal changes (≤5):
- Align optimize_ecosystem signature with interface: species_populations: Mapping[str, Sequence[DecisionGenome]] and return Mapping[str, Sequence[DecisionGenome]].
- Where a variable may be None (line 201), annotate as Optional[DecisionGenome] and guard before use or assert it is not None after the selection step.
- Remove the unused # type: ignore at line 440; if needed, replace with a local cast instead of suppression.
- Annotate helper methods: _calculate_regime_bonus(self, market_regime: str) -> float and _calculate_adaptability_score(self, genome: DecisionGenome, market_data: Mapping[str, object]) -> float.
- Ensure numeric return is float by wrapping with float(...), e.g., return float(adaptability).

Round B single smallest additional change (if needed later):
- If Mapping vs dict variance persists at return sites, wrap the concrete dict in a typing.cast(Mapping[str, Sequence[DecisionGenome]], result).

Acceptance criteria for this file:
- After Round A changes, mypy reports zero errors for [src/ecosystem/optimization/ecosystem_optimizer.py](src/ecosystem/optimization/ecosystem_optimizer.py:1).

-----

## 11) [src/trading/trading_manager.py](src/trading/trading_manager.py:1)

Current mypy errors: 6

Error entries:
- [src/trading/trading_manager.py](src/trading/trading_manager.py:63): Missing positional arguments risk_per_trade, stop_loss_pct in call to position_size  [call-arg]
- [src/trading/trading_manager.py](src/trading/trading_manager.py:63): Argument 1 to position_size has incompatible type float; expected Decimal  [arg-type]
- [src/trading/trading_manager.py](src/trading/trading_manager.py:64): None not callable  [misc]
- [src/trading/trading_manager.py](src/trading/trading_manager.py:94): PortfolioMonitor has no attribute get_state  [attr-defined]
- [src/trading/trading_manager.py](src/trading/trading_manager.py:110): PortfolioMonitor has no attribute increment_positions  [attr-defined]
- [src/trading/trading_manager.py](src/trading/trading_manager.py:143): PortfolioMonitor has no attribute get_state  [attr-defined]

Error categories:
- Call signature mismatch and numeric coercion
- Shadowed symbol or mis-declared class reference used as callable
- Missing methods on dependent type (treat as Any)

Round A minimal changes (≤5):
- Provide the required constructor arguments when creating PositionSizer and coerce numerics to Decimal, e.g., PositionSizer(Decimal(str(risk_per_trade)), Decimal(str(stop_loss_pct))).
- Resolve None not callable by ensuring a class reference is used: either rename a shadowing variable or cast the callable symbol to Any at the call site.
- Treat PortfolioMonitor as Any at call sites that rely on untyped methods: cast(Any, self.portfolio_monitor).get_state().
- Similarly cast for increment_positions.
- Likewise cast for the later get_state usage around line 143.

Round B single smallest additional change (if needed later):
- Introduce a minimal Protocol for PortfolioMonitor exposing get_state and increment_positions and cast to that Protocol instead of Any.

Acceptance criteria for this file:
- After Round A changes, mypy reports zero errors for [src/trading/trading_manager.py](src/trading/trading_manager.py:1).

-----

## 12) [src/thinking/adversarial/market_gan.py](src/thinking/adversarial/market_gan.py:1)

Current mypy errors: 6

Error entries:
- [src/thinking/adversarial/market_gan.py](src/thinking/adversarial/market_gan.py:177): Argument 2 to train_generator of AdversarialTrainer has incompatible type list[dict[str, float]]; expected list[object]  [arg-type]
- [src/thinking/adversarial/market_gan.py](src/thinking/adversarial/market_gan.py:182): Argument 1 to train_discriminator has incompatible type list[str]; expected list[object]  [arg-type]
- [src/thinking/adversarial/market_gan.py](src/thinking/adversarial/market_gan.py:182): Argument 2 to train_discriminator has incompatible type list[MarketScenario]; expected list[object]  [arg-type]
- [src/thinking/adversarial/market_gan.py](src/thinking/adversarial/market_gan.py:182): Argument 3 to train_discriminator has incompatible type list[dict[str, float]]; expected list[object]  [arg-type]
- [src/thinking/adversarial/market_gan.py](src/thinking/adversarial/market_gan.py:194): Argument 4 to _store_training_results of MarketGAN has incompatible type list[object]; expected list[str]  [arg-type]
- [src/thinking/adversarial/market_gan.py](src/thinking/adversarial/market_gan.py:198): Incompatible return value type (got list[object], expected list[str])  [return-value]

Error categories:
- list invariance at call sites (expects list[object])
- Function parameter and return typing (list[str] vs list[object])

Round A minimal changes (≤5):
- At the points where norm_sr is defined, annotate it as Sequence[object] to satisfy calls consuming it as list[object] without per-call casts.
- Annotate strategy_population as Sequence[object] at its creation.
- Annotate synthetic_scenarios as Sequence[object] at its creation.
- Update _store_training_results signature to accept Sequence[object] and coerce internally to list[str] with [str(x) for x in improved] for storage/logging.
- When returning improved_strategies, convert to list[str] via [str(x) for x in improved_strategies] to satisfy the annotated return type.

Round B single smallest additional change (if needed later):
- If any call still complains, add a narrow cast at the call site: typing.cast(list[object], variable).

Acceptance criteria for this file:
- After Round A changes, mypy reports zero errors for [src/thinking/adversarial/market_gan.py](src/thinking/adversarial/market_gan.py:1).

-----

Summary

- Planned files: 12
- Total mypy errors tallied across these files: 96

End of diagnostics plan.