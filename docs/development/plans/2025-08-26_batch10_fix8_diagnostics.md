# Batch10 Fix8 Diagnostics-Only Plan (2025-08-26)

Snapshot headline: Found 248 errors in 64 files (Post-Batch10 fix7 snapshot). Delta vs previous snapshot: Errors=0, Files=0

Scope and constraints
- Files included (exactly 12): 
  - [src/trading/execution/liquidity_prober.py](src/trading/execution/liquidity_prober.py)
  - [src/sensory/services/symbol_mapper.py](src/sensory/services/symbol_mapper.py)
  - [src/risk/risk_manager_impl.py](src/risk/risk_manager_impl.py)
  - [src/pnl.py](src/pnl.py)
  - [src/evolution/__init__.py](src/evolution/__init__.py)
  - [src/data_foundation/config/sizing_config.py](src/data_foundation/config/sizing_config.py)
  - [src/core/_event_bus_impl.py](src/core/_event_bus_impl.py)
  - [src/thinking/prediction/predictive_modeler.py](src/thinking/prediction/predictive_modeler.py)
  - [src/thinking/patterns/cvd_divergence_detector.py](src/thinking/patterns/cvd_divergence_detector.py)
  - [src/sensory/organs/volume_organ.py](src/sensory/organs/volume_organ.py)
  - [src/sensory/organs/sentiment_organ.py](src/sensory/organs/sentiment_organ.py)
  - [src/sensory/organs/price_organ.py](src/sensory/organs/price_organ.py)
- Source for errors: [mypy_snapshots/mypy_snapshot_2025-08-26T14-20-55Z.txt](mypy_snapshots/mypy_snapshot_2025-08-26T14-20-55Z.txt)
- Diagnostics-only: no source changes in this plan; proposals are suggested edits only.
- Formatting: clickable references use [path.py](path.py:line) to jump to locations.

Global acceptance criteria
- After applying Round A proposals for any of the files and running ruff/isort/black and mypy (base + strict-on-touch) on the edited file(s), there are zero mypy errors for that file in isolation.
- This plan compiles all 12 candidates with structured sections, clickable references, and no code changes performed here.

Method
1) Confirmed the 12 candidates from [changed_files_batch10_fix8_candidates.txt](changed_files_batch10_fix8_candidates.txt).
2) Extracted mypy error entries for each candidate from [mypy_snapshots/mypy_snapshot_2025-08-26T14-20-55Z.txt](mypy_snapshots/mypy_snapshot_2025-08-26T14-20-55Z.txt).
3) Categorized errors and proposed Round A minimal changes (≤5 edits per file) plus a Round B placeholder.
4) Compiled this diagnostics-only plan.

Totals for this batch
- Files planned: 12
- Total mypy errors across these files: 31

-------------------------------------------------------------------------------

File 1 of 12: [src/trading/execution/liquidity_prober.py](src/trading/execution/liquidity_prober.py)
- Current error count: 3
- Exact mypy entries
  - [src/trading/execution/liquidity_prober.py](src/trading/execution/liquidity_prober.py:47) — error: Argument 1 to "float" has incompatible type "object"; expected "str | Buffer | SupportsFloat | SupportsIndex"  [arg-type]
  - [src/trading/execution/liquidity_prober.py](src/trading/execution/liquidity_prober.py:48) — error: No overload variant of "int" matches argument type "object"  [call-overload]
  - [src/trading/execution/liquidity_prober.py](src/trading/execution/liquidity_prober.py:200) — error: Argument 1 to "float" has incompatible type "object"; expected "str | Buffer | SupportsFloat | SupportsIndex"  [arg-type]
- Categorization
  - Safe numeric coercion at initialization/return sites
  - Object/Any flowing into numeric constructors (arg-type, call-overload)
- Round A minimal changes (≤5, behavior-preserving)
  - Guard config values before coercion:
    - timeout_seconds: v = self.config.get("timeout_seconds"); self.timeout_seconds = float(v) if isinstance(v, (int, float, str)) else 1.0
  - Guard and coerce max_concurrent_probes similarly with isinstance(..., (int, str)) then int(v), else default 10
  - For filled_qty: v = status.get("filled_qty"); return float(v) if isinstance(v, (int, float, str)) else 0.0
  - Optional: extract small helper _to_float(x: object, default: float) -> float to standardize coercion
- Round B single smallest additional change
  - Add precise TypedDict/Mapping type for self.config to eliminate object returns at source
- Acceptance criteria
  - With Round A, mypy reports zero errors for [src/trading/execution/liquidity_prober.py](src/trading/execution/liquidity_prober.py) in isolation.

-------------------------------------------------------------------------------

File 2 of 12: [src/sensory/services/symbol_mapper.py](src/sensory/services/symbol_mapper.py)
- Current error count: 3
- Exact mypy entries
  - [src/sensory/services/symbol_mapper.py](src/sensory/services/symbol_mapper.py:96) — error: Argument 1 to "SymbolInfo" has incompatible type "**dict[str, object]"; expected "int"  [arg-type]
  - [src/sensory/services/symbol_mapper.py](src/sensory/services/symbol_mapper.py:96) — error: Argument 1 to "SymbolInfo" has incompatible type "**dict[str, object]"; expected "str"  [arg-type]
  - [src/sensory/services/symbol_mapper.py](src/sensory/services/symbol_mapper.py:96) — error: Argument 1 to "SymbolInfo" has incompatible type "**dict[str, object]"; expected "float"  [arg-type]
- Categorization
  - Any/object in heterogeneous dict unpack into strongly typed constructor
  - dict[str, object] to typed dataclass/constructor mismatch
- Round A minimal changes (≤5, behavior-preserving)
  - Replace SymbolInfo(**info) with explicit field assembly using safe coercions:
    - SymbolInfo(id=int(info["id"]), name=str(info["name"]), tick_size=float(info["tick_size"]), ...) using get/guarded access
  - If info comes from external JSON, define a TypedDict SymbolInfoPayload and cast incoming Mapping[str, object] to it before construction
  - Ensure self._symbols is typed as dict[str, SymbolInfo] to aid inference
- Round B single smallest additional change
  - Add a factory @classmethod SymbolInfo.from_payload(payload: Mapping[str, object]) -> SymbolInfo centralizing coercions
- Acceptance criteria
  - With Round A, mypy reports zero errors for [src/sensory/services/symbol_mapper.py](src/sensory/services/symbol_mapper.py).

-------------------------------------------------------------------------------

File 3 of 12: [src/risk/risk_manager_impl.py](src/risk/risk_manager_impl.py)
- Current error count: 3
- Exact mypy entries
  - [src/risk/risk_manager_impl.py](src/risk/risk_manager_impl.py:260) — error: Argument 2 of "evaluate_portfolio_risk" is incompatible with supertype "src.core.interfaces.RiskManager"; supertype defines the argument type as "Mapping[str, object] | None"  [override]
  - [src/risk/risk_manager_impl.py](src/risk/risk_manager_impl.py:272) — error: Argument 2 of "propose_rebalance" is incompatible with supertype "src.core.interfaces.RiskManager"; supertype defines the argument type as "Mapping[str, object] | None"  [override]
  - [src/risk/risk_manager_impl.py](src/risk/risk_manager_impl.py:277) — error: Argument 1 of "update_limits" is incompatible with supertype "src.core.interfaces.RiskManager"; supertype defines the argument type as "Mapping[str, object]"  [override]
- Categorization
  - Incompatible overrides vs interface (LSP)
  - Parameter type widening/narrowing mismatches
- Round A minimal changes (≤5, behavior-preserving)
  - Align method signatures to interface:
    - evaluate_portfolio_risk(..., context: Mapping[str, object] | None = None) -> ...
    - propose_rebalance(..., constraints: Mapping[str, object] | None = None) -> ...
    - update_limits(self, limits: Mapping[str, object]) -> ...
  - If internal code expects concrete dict, accept Mapping in signature and cast locally as dict[str, object] when needed
- Round B single smallest additional change
  - Import and use the interface’s aliases (e.g., JSONObject = Mapping[str, object]) to keep consistency
- Acceptance criteria
  - With Round A, mypy reports zero errors for [src/risk/risk_manager_impl.py](src/risk/risk_manager_impl.py).

-------------------------------------------------------------------------------

File 4 of 12: [src/pnl.py](src/pnl.py)
- Current error count: 3
- Exact mypy entries
  - [src/pnl.py](src/pnl.py:200) — error: "Instrument" has no attribute "swap_time"  [attr-defined]
  - [src/pnl.py](src/pnl.py:213) — error: "Instrument" has no attribute "long_swap_rate"  [attr-defined]
  - [src/pnl.py](src/pnl.py:215) — error: "Instrument" has no attribute "short_swap_rate"  [attr-defined]
- Categorization
  - Attribute presence on external type (attr-defined)
  - Optional/feature-flagged fields on instrument models
- Round A minimal changes (≤5, behavior-preserving)
  - Guard access with getattr defaults:
    - swap_time_s = str(getattr(instrument, "swap_time", "00:00"))
    - long_rate = float(getattr(instrument, "long_swap_rate", 0.0))
    - short_rate = float(getattr(instrument, "short_swap_rate", 0.0))
  - Validate and coerce before use (e.g., map(int, swap_time_s.split(":")))
- Round B single smallest additional change
  - Introduce a Protocol InstrumentSwapInfo with these optional attrs and cast instrument to it locally to narrow types
- Acceptance criteria
  - With Round A, mypy reports zero errors for [src/pnl.py](src/pnl.py).

-------------------------------------------------------------------------------

File 5 of 12: [src/evolution/__init__.py](src/evolution/__init__.py)
- Current error count: 3
- Exact mypy entries
  - [src/evolution/__init__.py](src/evolution/__init__.py:17) — error: Unused "type: ignore" comment  [unused-ignore]
  - [src/evolution/__init__.py](src/evolution/__init__.py:18) — error: Unused "type: ignore" comment  [unused-ignore]
  - [src/evolution/__init__.py](src/evolution/__init__.py:20) — error: Unused "type: ignore" comment  [unused-ignore]
- Categorization
  - Unused type ignores
- Round A minimal changes (≤5, behavior-preserving)
  - Remove the three unused "# type: ignore" comments at lines 17, 18, 20
- Round B single smallest additional change
  - If re-exports cause transient mypy issues, replace ignores with explicit from ... import ... aliases annotated with correct types
- Acceptance criteria
  - With Round A, mypy reports zero errors for [src/evolution/__init__.py](src/evolution/__init__.py).

-------------------------------------------------------------------------------

File 6 of 12: [src/data_foundation/config/sizing_config.py](src/data_foundation/config/sizing_config.py)
- Current error count: 3
- Exact mypy entries
  - [src/data_foundation/config/sizing_config.py](src/data_foundation/config/sizing_config.py:10) — error: Incompatible types in assignment (expression has type "None", variable has type Module)  [assignment]
  - [src/data_foundation/config/sizing_config.py](src/data_foundation/config/sizing_config.py:18) — error: Missing type parameters for generic type "dict"  [type-arg]
  - [src/data_foundation/config/sizing_config.py](src/data_foundation/config/sizing_config.py:18) — error: Incompatible types in assignment (expression has type "None", variable has type "dict[Any, Any]")  [assignment]
- Categorization
  - Module shadowing with sentinel (yaml = None)
  - Missing type args for dict
  - Optional assignment to non-Optional
- Round A minimal changes (≤5, behavior-preserving)
  - Avoid module shadowing: rename sentinel to _yaml_mod or import within try/except:
    - try: import yaml as _yaml_mod; except Exception: _yaml_mod = None
  - Annotate regime_multipliers as Optional with explicit types:
    - regime_multipliers: dict[str, float] | None = None
  - Where used, handle None with a default empty mapping or conditional path
- Round B single smallest additional change
  - If yaml usage is only for I/O, gate imports behind TYPE_CHECKING and pass instances into functions at call sites
- Acceptance criteria
  - With Round A, mypy reports zero errors for [src/data_foundation/config/sizing_config.py](src/data_foundation/config/sizing_config.py).

-------------------------------------------------------------------------------

File 7 of 12: [src/core/_event_bus_impl.py](src/core/_event_bus_impl.py)
- Current error count: 3
- Exact mypy entries
  - [src/core/_event_bus_impl.py](src/core/_event_bus_impl.py:295) — error: Argument 2 to "subscribe" of "AsyncEventBus" has incompatible type "Callable[[Event], object]"; expected "Callable[[Event], Awaitable[None]] | Callable[[Event], None]"  [arg-type]
  - [src/core/_event_bus_impl.py](src/core/_event_bus_impl.py:325) — error: Argument 2 to "subscribe_topic" of "TopicBus" has incompatible type "Callable[[str, object], object]"; expected "Callable[[str, Any], Awaitable[None] | None]"  [arg-type]
  - [src/core/_event_bus_impl.py](src/core/_event_bus_impl.py:407) — error: Argument "handler" to "SubscriptionHandle" has incompatible type "Callable[[Event], object]"; expected "Callable[[Event], Awaitable[None]] | Callable[[Event], None]"  [arg-type]
- Categorization
  - Handler callable return type/object leakage
  - Async vs sync handler signatures
- Round A minimal changes (≤5, behavior-preserving)
  - Wrap adapter callbacks to return None explicitly:
    - def adapter_fn(evt: Event) -> None: _ = user_cb(evt); return None
  - For async contexts, detect coroutine and await it; otherwise return None; type annotate adapter as Callable[[Event], None] or Callable[[Event], Awaitable[None]]
  - When constructing SubscriptionHandle, pass a handler with return type None by wrapping the callback as above
- Round B single smallest additional change
  - Introduce a small utility ensure_none_return(cb) -> handler that erases return type to None with proper overloads
- Acceptance criteria
  - With Round A, mypy reports zero errors for [src/core/_event_bus_impl.py](src/core/_event_bus_impl.py).

-------------------------------------------------------------------------------

File 8 of 12: [src/thinking/prediction/predictive_modeler.py](src/thinking/prediction/predictive_modeler.py)
- Current error count: 2
- Exact mypy entries
  - [src/thinking/prediction/predictive_modeler.py](src/thinking/prediction/predictive_modeler.py:43) — error: Argument 1 to "PredictiveMarketModeler" has incompatible type "str"; expected "StateStore"  [arg-type]
  - [src/thinking/prediction/predictive_modeler.py](src/thinking/prediction/predictive_modeler.py:57) — error: "PredictiveMarketModeler" has no attribute "forecast"  [attr-defined]
- Categorization
  - Constructor argument type mismatch vs expected interface (arg-type)
  - Referenced attribute not defined on the class (attr-defined)
- Round A minimal changes (≤5, behavior-preserving)
  - Ensure correct constructor usage in local demo/test code within this module:
    - Instantiate a proper StateStore (e.g., InMemoryStateStore()) and pass it to PredictiveMarketModeler instead of a str.
  - Provide a thin alias method on PredictiveMarketModeler if examples call forecast:
    - def forecast(self, data: Any) -> Awaitable[Any] | Any: return self.predict(data)
    - Keep signature aligned with existing predict semantics (async/sync as applicable).
- Round B single smallest additional change
  - If examples are non-executable documentation, wrap them under if TYPE_CHECKING: or convert to doctest with correct types to avoid mypy analysis of illustrative code.
- Acceptance criteria
  - With Round A, mypy reports zero errors for [src/thinking/prediction/predictive_modeler.py](src/thinking/prediction/predictive_modeler.py) in isolation.

-------------------------------------------------------------------------------

File 9 of 12: [src/thinking/patterns/cvd_divergence_detector.py](src/thinking/patterns/cvd_divergence_detector.py)
- Current error count: 2
- Exact mypy entries
  - [src/thinking/patterns/cvd_divergence_detector.py](src/thinking/patterns/cvd_divergence_detector.py:14) — error: Cannot assign to a type  [misc]
  - [src/thinking/patterns/cvd_divergence_detector.py](src/thinking/patterns/cvd_divergence_detector.py:14) — error: Incompatible types in assignment (expression has type "type[object]", variable has type "type[ContextPacket]")  [assignment]
- Categorization
  - Illegal reassignment of type alias/name to a runtime object
- Round A minimal changes (≤5, behavior-preserving)
  - Replace the assignment with a proper alias:
    - from typing import Any, TypeAlias; ContextPacket: TypeAlias = Any
  - If a real ContextPacket type exists elsewhere, import it instead of reassigning
- Round B single smallest additional change
  - Introduce a Protocol for ContextPacket if only selected attributes are required
- Acceptance criteria
  - With Round A, mypy reports zero errors for [src/thinking/patterns/cvd_divergence_detector.py](src/thinking/patterns/cvd_divergence_detector.py).

-------------------------------------------------------------------------------

File 10 of 12: [src/sensory/organs/volume_organ.py](src/sensory/organs/volume_organ.py)
- Current error count: 2
- Exact mypy entries
  - [src/sensory/organs/volume_organ.py](src/sensory/organs/volume_organ.py:15) — error: Module "src.core.base" has no attribute "SensoryOrgan"  [attr-defined]
  - [src/sensory/organs/volume_organ.py](src/sensory/organs/volume_organ.py:16) — error: Module "src.core.exceptions" has no attribute "SensoryException"; maybe "ResourceException"?  [attr-defined]
- Categorization
  - Importing missing types from modules (attr-defined)
- Round A minimal changes (≤5, behavior-preserving)
  - Replace missing imports with local type fallbacks:
    - from typing import Protocol, Any; class SensoryOrgan(Protocol): ...
  - Import the actual exception available and alias:
    - from src.core.exceptions import ResourceException as SensoryException
  - If runtime code uses SensoryException, ensure the alias preserves behavior
- Round B single smallest additional change
  - Gate heavy type-only imports under TYPE_CHECKING and use runtime-safe fallbacks
- Acceptance criteria
  - With Round A, mypy reports zero errors for [src/sensory/organs/volume_organ.py](src/sensory/organs/volume_organ.py).

-------------------------------------------------------------------------------

File 11 of 12: [src/sensory/organs/sentiment_organ.py](src/sensory/organs/sentiment_organ.py)
- Current error count: 2
- Exact mypy entries
  - [src/sensory/organs/sentiment_organ.py](src/sensory/organs/sentiment_organ.py:14) — error: Module "src.core.base" has no attribute "SensoryOrgan"  [attr-defined]
  - [src/sensory/organs/sentiment_organ.py](src/sensory/organs/sentiment_organ.py:14) — error: Module "src.core.base" has no attribute "SensoryReading"  [attr-defined]
- Categorization
  - Importing missing types from modules (attr-defined)
- Round A minimal changes (≤5, behavior-preserving)
  - Define local Protocols for missing types for typing only:
    - from typing import Protocol, Mapping, Any; class SensoryOrgan(Protocol): ...; class SensoryReading(Protocol): ...
  - Keep MarketData import if valid; otherwise use a TypeAlias MarketData = Mapping[str, object]
- Round B single smallest additional change
  - Identify the canonical source of these types (e.g., src.core.interfaces) and import from there to remove local Protocols
- Acceptance criteria
  - With Round A, mypy reports zero errors for [src/sensory/organs/sentiment_organ.py](src/sensory/organs/sentiment_organ.py).

-------------------------------------------------------------------------------

File 12 of 12: [src/sensory/organs/price_organ.py](src/sensory/organs/price_organ.py)
- Current error count: 2
- Exact mypy entries
  - [src/sensory/organs/price_organ.py](src/sensory/organs/price_organ.py:15) — error: Module "src.core.base" has no attribute "SensoryOrgan"  [attr-defined]
  - [src/sensory/organs/price_organ.py](src/sensory/organs/price_organ.py:16) — error: Module "src.core.exceptions" has no attribute "SensoryException"; maybe "ResourceException"?  [attr-defined]
- Categorization
  - Importing missing types from modules (attr-defined)
- Round A minimal changes (≤5, behavior-preserving)
  - Define local Protocol for SensoryOrgan (typing only) if not available from core
  - Import and alias ResourceException as SensoryException:
    - from src.core.exceptions import ResourceException as SensoryException
- Round B single smallest additional change
  - Replace local Protocol with the actual core interface when confirmed
- Acceptance criteria
  - With Round A, mypy reports zero errors for [src/sensory/organs/price_organ.py](src/sensory/organs/price_organ.py).

-------------------------------------------------------------------------------

Appendix: supporting candidate with additional references

- Cross-check: The listed entries reflect all snapshot-reported errors for these 12 files per [mypy_snapshots/mypy_snapshot_2025-08-26T14-20-55Z.txt](mypy_snapshots/mypy_snapshot_2025-08-26T14-20-55Z.txt).
- Formatting policy adhered to: clickable references [path.py](path.py:line), proposals are behavior-preserving and ≤5 edits per file, Round B placeholder included.