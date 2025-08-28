# Batch10 Fix9 Diagnostics Plan — 2025-08-26

Scope: Diagnostics-only. No source edits. The plan targets exactly the 12 Fix9 candidates.

Snapshot baseline: Found 240 errors in 60 files (Post-Batch10 fix8). Delta vs previous snapshot: errors=-8, files=-4.

Method:
1) Confirmed the 12 candidates match [changed_files_batch10_fix9_candidates.txt](changed_files_batch10_fix9_candidates.txt)
2) Grepped error entries for these files from [mypy_snapshots/mypy_snapshot_2025-08-26T16-24-11Z.txt](mypy_snapshots/mypy_snapshot_2025-08-26T16-24-11Z.txt)
3) Drafted Round A minimal-change proposals (≤5 edits per file), strictly behavior-preserving and aligned with established typing patterns
4) Added a single Round B placeholder per file for smallest follow-up if residuals appear

Global acceptance criteria:
- This plan includes all 12 files with structured sections and clickable references [path.py](path.py:line).
- After applying Round A proposals and running ruff/isort/black and mypy base + strict-on-touch on the edited file(s), zero mypy errors remain for that file in isolation.

Total errors across these 12 files: 19


Candidates (exact 12):
- [src/ecosystem/evolution/specialized_predator_evolution.py](src/ecosystem/evolution/specialized_predator_evolution.py)
- [src/evolution/engine/__init__.py](src/evolution/engine/__init__.py)
- [src/sensory/organs/dimensions/how_organ.py](src/sensory/organs/dimensions/how_organ.py)
- [src/sensory/organs/dimensions/when_organ.py](src/sensory/organs/dimensions/when_organ.py)
- [src/sensory/organs/economic_organ.py](src/sensory/organs/economic_organ.py)
- [src/sensory/organs/news_organ.py](src/sensory/organs/news_organ.py)
- [src/sensory/organs/orderbook_organ.py](src/sensory/organs/orderbook_organ.py)
- [src/core/configuration.py](src/core/configuration.py)
- [src/data_foundation/config/execution_config.py](src/data_foundation/config/execution_config.py)
- [src/data_foundation/config/risk_portfolio_config.py](src/data_foundation/config/risk_portfolio_config.py)
- [src/data_foundation/config/why_config.py](src/data_foundation/config/why_config.py)
- [src/data_foundation/persist/parquet_writer.py](src/data_foundation/persist/parquet_writer.py)


1) src/ecosystem/evolution/specialized_predator_evolution.py

Errors from snapshot:
- [src/ecosystem/evolution/specialized_predator_evolution.py](src/ecosystem/evolution/specialized_predator_evolution.py:102): error: Unused "type: ignore" comment  [unused-ignore]
- [src/ecosystem/evolution/specialized_predator_evolution.py](src/ecosystem/evolution/specialized_predator_evolution.py:135): error: Unused "type: ignore" comment  [unused-ignore]

Categories:
- Unused ignore pragmas

Round A minimal changes (≤5):
- Remove the two unused ignore comments at lines 102 and 135.

Round B single smallest placeholder:
- If new type issues surface after removal, apply a localized typing.cast[...] only at the await site to narrow the return type.


2) src/evolution/engine/__init__.py

Errors from snapshot:
- [src/evolution/engine/__init__.py](src/evolution/engine/__init__.py:8): error: Unused "type: ignore" comment  [unused-ignore]
- [src/evolution/engine/__init__.py](src/evolution/engine/__init__.py:11): error: Unused "type: ignore" comment  [unused-ignore]

Categories:
- Unused ignore pragmas

Round A minimal changes (≤5):
- Remove the two unused ignore comments on the import lines at 8 and 11.

Round B single smallest placeholder:
- If imports later require isolation, move them under TYPE_CHECKING and export runtime aliases only as needed.


3) src/sensory/organs/dimensions/how_organ.py

Errors from snapshot:
- [src/sensory/organs/dimensions/how_organ.py](src/sensory/organs/dimensions/how_organ.py:297): error: "MarketData" has no attribute "spread"  [attr-defined]
- [src/sensory/organs/dimensions/how_organ.py](src/sensory/organs/dimensions/how_organ.py:298): error: "MarketData" has no attribute "mid_price"  [attr-defined]

Categories:
- Attribute presence on typed model (microstructure fields)

Round A minimal changes (≤5):
- Use getattr(md, "spread", 0.0) and getattr(md, "mid_price", float("nan")) at the immediate access sites.
- If downstream needs numeric, wrap with float(...) to normalize.
- Alternatively, insert a local Protocol under TYPE_CHECKING with attributes spread: float and mid_price: float, and cast at the read sites.

Round B single smallest placeholder:
- Introduce a tiny shared Microstructure protocol in a types module and import it type-only here.


4) src/sensory/organs/dimensions/when_organ.py

Errors from snapshot:
- [src/sensory/organs/dimensions/when_organ.py](src/sensory/organs/dimensions/when_organ.py:364): error: "MarketData" has no attribute "spread"  [attr-defined]
- [src/sensory/organs/dimensions/when_organ.py](src/sensory/organs/dimensions/when_organ.py:365): error: "MarketData" has no attribute "mid_price"  [attr-defined]

Categories:
- Attribute presence on typed model (microstructure fields)

Round A minimal changes (≤5):
- Replace direct attribute access with getattr(md, "spread", 0.0) and getattr(md, "mid_price", float("nan")), with float(...) normalization where needed.
- Optionally cast to a local Protocol under TYPE_CHECKING at the access site if preferred.

Round B single smallest placeholder:
- Small MarketData adapter with optional microstructure fields, used type-only.


5) src/sensory/organs/economic_organ.py

Errors from snapshot:
- [src/sensory/organs/economic_organ.py](src/sensory/organs/economic_organ.py:14): error: Module "src.core.base" has no attribute "SensoryOrgan"  [attr-defined]
- [src/sensory/organs/economic_organ.py](src/sensory/organs/economic_organ.py:14): error: Module "src.core.base" has no attribute "SensoryReading"  [attr-defined]

Categories:
- Import typing-time vs runtime symbol availability

Round A minimal changes (≤5):
- Move SensoryOrgan and SensoryReading imports under TYPE_CHECKING.
- If referenced at runtime, provide minimal local runtime placeholders (e.g., Protocol-like stubs guarded for runtime without behavioral impact).

Round B single smallest placeholder:
- Centralize sensory protocols in a shared typing module and import type-only here.


6) src/sensory/organs/news_organ.py

Errors from snapshot:
- [src/sensory/organs/news_organ.py](src/sensory/organs/news_organ.py:14): error: Module "src.core.base" has no attribute "SensoryOrgan"  [attr-defined]
- [src/sensory/organs/news_organ.py](src/sensory/organs/news_organ.py:14): error: Module "src.core.base" has no attribute "SensoryReading"  [attr-defined]

Categories:
- Import typing-time vs runtime symbol availability

Round A minimal changes (≤5):
- Place SensoryOrgan and SensoryReading imports under TYPE_CHECKING.
- Provide minimal local runtime placeholders only if names are used at runtime.

Round B single smallest placeholder:
- Reuse the centralized sensory protocols module once introduced.


7) src/sensory/organs/orderbook_organ.py

Errors from snapshot:
- [src/sensory/organs/orderbook_organ.py](src/sensory/organs/orderbook_organ.py:15): error: Module "src.core.base" has no attribute "SensoryOrgan"  [attr-defined]
- [src/sensory/organs/orderbook_organ.py](src/sensory/organs/orderbook_organ.py:16): error: Module "src.core.exceptions" has no attribute "SensoryException"; maybe "ResourceException"?  [attr-defined]

Categories:
- Import typing-time vs runtime symbol availability
- Exception type import mismatch

Round A minimal changes (≤5):
- Move SensoryOrgan under TYPE_CHECKING with a local runtime placeholder if referenced.
- If SensoryException is only for typing, import it under TYPE_CHECKING; if used at runtime, alias to an available exception (e.g., ResourceException) without changing behavior.

Round B single smallest placeholder:
- Add a tiny exception alias block mapping SensoryException to the closest available base in core exceptions.


8) src/core/configuration.py

Errors from snapshot:
- [src/core/configuration.py](src/core/configuration.py:124): error: Unsupported target for indexed assignment ("Configuration")  [index]

Categories:
- Indexed assignment on a non-mapping typed object

Round A minimal changes (≤5):
- Redirect assignment to the underlying mapping attribute if present (e.g., self._data[key] = value).
- Or apply typing.cast[MutableMapping[str, object]](...) at the use-site to satisfy setitem where the runtime object is a mapping.

Round B single smallest placeholder:
- Introduce a minimal Protocol with __setitem__ and cast to it at the assignment site (type-only).


9) src/data_foundation/config/execution_config.py

Errors from snapshot:
- [src/data_foundation/config/execution_config.py](src/data_foundation/config/execution_config.py:10): error: Incompatible types in assignment (expression has type "None", variable has type Module)  [assignment]

Categories:
- Reassigning a module symbol to None

Round A minimal changes (≤5):
- Avoid reassigning the imported module name; use a separate optional handle variable (e.g., _yaml_mod: object | None = None) and guard uses.

Round B single smallest placeholder:
- Introduce a tiny optional-import helper that returns a module handle or None, reused across config files.


10) src/data_foundation/config/risk_portfolio_config.py

Errors from snapshot:
- [src/data_foundation/config/risk_portfolio_config.py](src/data_foundation/config/risk_portfolio_config.py:10): error: Incompatible types in assignment (expression has type "None", variable has type Module)  [assignment]

Categories:
- Reassigning a module symbol to None

Round A minimal changes (≤5):
- Stop assigning None to the module symbol; keep the module import intact and store optional availability in a distinct variable.

Round B single smallest placeholder:
- Reuse the optional-import helper from execution_config.


11) src/data_foundation/config/why_config.py

Errors from snapshot:
- [src/data_foundation/config/why_config.py](src/data_foundation/config/why_config.py:10): error: Incompatible types in assignment (expression has type "None", variable has type Module)  [assignment]

Categories:
- Reassigning a module symbol to None

Round A minimal changes (≤5):
- Replace reassignment with a separate optional handle variable; keep type-time imports under TYPE_CHECKING as needed.

Round B single smallest placeholder:
- Adopt the same optional-import helper for consistency.


12) src/data_foundation/persist/parquet_writer.py

Errors from snapshot:
- [src/data_foundation/persist/parquet_writer.py](src/data_foundation/persist/parquet_writer.py:10): error: Incompatible types in assignment (expression has type "None", variable has type Module)  [assignment]

Categories:
- Reassigning a module symbol to None

Round A minimal changes (≤5):
- Do not assign None to the pandas module symbol; create an optional handle variable for runtime availability checks and use that in guarded calls.

Round B single smallest placeholder:
- Minimal adapter function that encapsulates the optional pandas import and write path for parquet persistence.


Per-file acceptance criteria:
- After applying the Round A proposals for that file and running ruff/isort/black and mypy (base + strict-on-touch) on the edited file(s), that file reports zero mypy errors in isolation.

Global acceptance criteria:
- This diagnostics plan contains all 12 candidate files with structured sections, clickable [path.py](path.py:line) references, and no source code modifications performed here.