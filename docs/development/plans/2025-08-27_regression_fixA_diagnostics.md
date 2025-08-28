# Batch10 regression: Fix A diagnostics plan (2025-08-27)

Context links:
- Baseline summary: [mypy_summary_2025-08-27T09-43-50Z.txt](../../../mypy_snapshots/mypy_summary_2025-08-27T09-43-50Z.txt:1)
- Current summary: [mypy_summary_2025-08-27T15-21-18Z.txt](../../../mypy_snapshots/mypy_summary_2025-08-27T15-21-18Z.txt:1)
- Regression report: [2025-08-27_mypy_regression_report.md](../reports/2025-08-27_mypy_regression_report.md:1)
- Top10 report: [2025-08-27_mypy_regression_top10.md](../reports/2025-08-27_mypy_regression_top10.md:1)
- Detailed snapshot: [mypy_snapshot_detailed_2025-08-27T15-44-57Z.txt](../../../mypy_snapshots/mypy_snapshot_detailed_2025-08-27T15-44-57Z.txt:1)
- Per-module histogram (derived): [regression_per_module_2025-08-27T15-44-57Z.csv](../../../mypy_snapshots/regression_per_module_2025-08-27T15-44-57Z.csv:1)

Totals:
- Found 124 errors in 28 files (checked 343 source files) (from [mypy_summary_2025-08-27T15-21-18Z.txt](../../../mypy_snapshots/mypy_summary_2025-08-27T15-21-18Z.txt:1))

Candidate set (Top 10 paths):
1) src/thinking/patterns/trend_detector.py
2) src/thinking/patterns/cycle_detector.py
3) src/data_integration/data_fusion.py
4) src/intelligence/sentient_adaptation.py
5) src/sensory/organs/dimensions/institutional_tracker.py
6) src/trading/portfolio/real_portfolio_monitor.py
7) src/sensory/organs/dimensions/integration_orchestrator.py
8) src/evolution/mutation/gaussian_mutation.py
9) src/genome/models/genome_adapter.py
10) src/orchestration/enhanced_intelligence_engine.py

Notes on error-code landscape:
- Global histogram: see [mypy_error_codes_2025-08-27T15-44-57Z.csv](../../../mypy_snapshots/mypy_error_codes_2025-08-27T15-44-57Z.csv:1). Dominant codes: attr-defined, assignment, misc, arg-type, call-arg, operator, no-any-return, import-not-found.

Reference for no-behavior-change recipes:
- Typing recipes: [typing_recipes.md](../typing_recipes.md:1)

Per-candidate diagnostics

— src/thinking/patterns/trend_detector.py
- Total errors for path: 18
- Top codes:
  - attr-defined: 13
  - assignment: 3
  - misc: 1
  - typeddict-unknown-key: 1
- Representative lines:
  - src/thinking/patterns/trend_detector.py:141: error: "SensorySignal" has no attribute "signal_type"  [attr-defined]
  - src/thinking/patterns/trend_detector.py:170: error: "SensorySignal" has no attribute "confidence"  [attr-defined]
  - src/thinking/patterns/trend_detector.py:190: error: Extra keys ("timestamp", "analysis_type", "result", "confidence", "metadata") for TypedDict "AnalysisResult"  [typeddict-unknown-key]
- Round A no-behavior-change patterns (map to codes):
  - [attr-defined] → isinstance guards + cast, typed protocol/alias refinement; see [typing_recipes.md](../typing_recipes.md:1)
  - [assignment]/[misc] → stop assigning to type aliases; introduce typed locals; see [typing_recipes.md](../typing_recipes.md:1)
  - [typeddict-unknown-key] → reconcile TypedDict schema vs returned keys; consider total=False or auxiliary cast; see [typing_recipes.md](../typing_recipes.md:1)

— src/thinking/patterns/cycle_detector.py
- Total errors for path: 11
- Top codes:
  - misc: 3
  - assignment: 3
  - type-var: 3
  - attr-defined: 2
- Representative lines:
  - src/thinking/patterns/cycle_detector.py:23: error: Cannot assign to a type  [misc]
  - src/thinking/patterns/cycle_detector.py:222: error: Module "src.core.exceptions" has no attribute "ThinkingException"; maybe "TradingException"?  [attr-defined]
  - src/thinking/patterns/cycle_detector.py:330: error: Value of type variable "SupportsRichComparisonT" of "max" cannot be "floating[Any] | float"  [type-var]
- Round A patterns:
  - [misc]/[assignment] → remove runtime reassignments of type names; replace with forward-declared Protocols or typing aliases under TYPE_CHECKING; [typing_recipes.md](../typing_recipes.md:1)
  - [type-var] → normalize arithmetic to concrete float via float() wrappers; [typing_recipes.md](../typing_recipes.md:1)
  - [attr-defined] → import correction or TYPE_CHECKING alias; [typing_recipes.md](../typing_recipes.md:1)

— src/data_integration/data_fusion.py
- Total errors for path: 10
- Top codes:
  - attr-defined: 5
  - assignment: 3
  - return-value: 1
  - arg-type: 1
- Representative lines:
  - src/data_integration/data_fusion.py:270: error: "MarketData" has no attribute "volatility"  [attr-defined]
  - src/data_integration/data_fusion.py:274: error: Incompatible return value type (got "tuple[None, list[str]]", expected "tuple[MarketData, list[str]]")  [return-value]
  - src/data_integration/data_fusion.py:303: error: Argument "volume" to "FusedDataPoint" has incompatible type "float"; expected "int"  [arg-type]
- Round A patterns:
  - [attr-defined] → guard attribute existence or define typed interfaces with optional fields; [typing_recipes.md](../typing_recipes.md:1)
  - [assignment]/[arg-type] → numeric normalization via explicit float/int conversions; [typing_recipes.md](../typing_recipes.md:1)
  - [return-value] → ensure explicit return annotations and return shape; [typing_recipes.md](../typing_recipes.md:1)

— src/intelligence/sentient_adaptation.py
- Total errors for path: 9
- Top codes:
  - call-arg: 4
  - attr-defined: 3
  - arg-type: 2
- Representative lines:
  - src/intelligence/sentient_adaptation.py:142: error: Missing positional argument "config" in call to "RealTimeLearningEngine"  [call-arg]
  - src/intelligence/sentient_adaptation.py:324: error: "RealTimeLearningEngine" has no attribute "process_outcome"  [attr-defined]
  - src/intelligence/sentient_adaptation.py:343: error: Argument 1 to "apply_adaptations" ... expected "dict[str, Any]"  [arg-type]
- Round A patterns:
  - [call-arg] → overload narrowing or factory helpers that supply defaults; [typing_recipes.md](../typing_recipes.md:1)
  - [attr-defined] → interface tightening with Protocols + cast where needed; [typing_recipes.md](../typing_recipes.md:1)
  - [arg-type] → adapters and value-shaping helpers before API calls; [typing_recipes.md](../typing_recipes.md:1)

— src/sensory/organs/dimensions/institutional_tracker.py
- Total errors for path: 8
- Top codes:
  - attr-defined: 5
  - no-any-return: 3
- Representative lines:
  - src/sensory/.../institutional_tracker.py:171: error: "MarketData" has no attribute "spread"  [attr-defined]
  - src/sensory/.../institutional_tracker.py:640: error: Returning Any from function declared to return "MarketRegime"  [no-any-return]
  - src/sensory/.../institutional_tracker.py:642: error: "type[MarketRegime]" has no attribute "BEARISH"  [attr-defined]
- Round A patterns:
  - [attr-defined] → isinstance guards, typed DTOs, or conditional attribute access; [typing_recipes.md](../typing_recipes.md:1)
  - [no-any-return] → add precise return annotations or cast known enum values; [typing_recipes.md](../typing_recipes.md:1)

— src/trading/portfolio/real_portfolio_monitor.py
- Total errors for path: 7
- Top codes:
  - attr-defined: 6
  - index: 1
- Representative lines:
  - src/trading/portfolio/real_portfolio_monitor.py:14: error: ... no attribute "PortfolioSnapshot"  [attr-defined]
  - src/trading/portfolio/real_portfolio_monitor.py:441: error: Invalid index type "str | int | None" ... expected "str"  [index]
  - src/trading/portfolio/real_portfolio_monitor.py:446: error: "Position" has no attribute "close"  [attr-defined]
- Round A patterns:
  - [attr-defined] → import target stabilization via TYPE_CHECKING + re-exports; [typing_recipes.md](../typing_recipes.md:1)
  - [index] → key type guards and normalization; [typing_recipes.md](../typing_recipes.md:1)

— src/sensory/organs/dimensions/integration_orchestrator.py
- Total errors for path: 6
- Top codes:
  - import-not-found: 4
  - arg-type: 2
- Representative lines:
  - src/sensory/.../integration_orchestrator.py:26: error: Cannot find implementation or library stub ... [import-not-found]
  - src/sensory/.../integration_orchestrator.py:166: error: incompatible type "DataFrame"; expected "list[MarketData]"  [arg-type]
  - src/sensory/.../integration_orchestrator.py:168: error: incompatible type "OrchestratorMarketData | Mapping[str, object]"; expected "dict[str, Any]"  [arg-type]
- Round A patterns:
  - [import-not-found] → defer heavy imports under TYPE_CHECKING, add stubs, or localize imports; [typing_recipes.md](../typing_recipes.md:1)
  - [arg-type] → adapters and type-safe transformation functions; [typing_recipes.md](../typing_recipes.md:1)

— src/evolution/mutation/gaussian_mutation.py
- Total errors for path: 7
- Top codes:
  - attr-defined: 5
  - assignment: 1
  - misc: 1
- Representative lines:
  - src/evolution/mutation/gaussian_mutation.py:10: error: ... no attribute "IMutationStrategy"  [attr-defined]
  - src/evolution/mutation/gaussian_mutation.py:18: error: Cannot assign to a type  [misc]
  - src/evolution/mutation/gaussian_mutation.py:129: error: "DecisionGenome" has no attribute "sensory"  [attr-defined]
- Round A patterns:
  - [attr-defined] → protocolize and cast or add guards; [typing_recipes.md](../typing_recipes.md:1)
  - [assignment]/[misc] → avoid runtime alias reassignment; introduce typed placeholders under TYPE_CHECKING; [typing_recipes.md](../typing_recipes.md:1)

— src/genome/models/genome_adapter.py
- Total errors for path: 6
- Top codes:
  - assignment: 5
  - misc: 1
- Representative lines:
  - src/genome/models/genome_adapter.py:30: error: Cannot assign to a type  [misc]
  - src/genome/models/genome_adapter.py:31: error: Incompatible types in assignment ... variable has type "Callable[[DecisionGenome, str, dict[str, float]], DecisionGenome]"  [assignment]
  - src/genome/models/genome_adapter.py:38: error: Incompatible types in assignment ... "Callable[[Any], DecisionGenome]"  [assignment]
- Round A patterns:
  - [assignment]/[misc] → replace None placeholders with Optional[...] and default no-op callables behind typed factories; [typing_recipes.md](../typing_recipes.md:1)

— src/orchestration/enhanced_intelligence_engine.py
- Total errors for path: 5
- Top codes:
  - operator: 5
- Representative lines:
  - src/orchestration/enhanced_intelligence_engine.py:196: error: "object" not callable  [operator]
  - src/orchestration/enhanced_intelligence_engine.py:201: error: "object" not callable  [operator]
- Round A patterns:
  - [operator] → ensure callable types via precise type annotations and cast after isinstance checks; [typing_recipes.md](../typing_recipes.md:1)

Scope note:
- Diagnostics-only; no code edits in this plan. Implementation will occur in a subsequent fix PR.

Environment note:
- Local environment observed: Python 3.12.3; mypy 1.17.1. Baseline CI uses Python 3.11. Action: verify CI environment alignment (Python/mypy) before code changes to avoid environment-induced discrepancies.