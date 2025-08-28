# Batch10 fix11 — Diagnostics-Only Plan

Scope
- Snapshot source: [mypy_snapshots/mypy_snapshot_2025-08-26T18-11-40Z.txt](mypy_snapshots/mypy_snapshot_2025-08-26T18-11-40Z.txt)
- Candidates list: [mypy_snapshots/candidates_fix11_2025-08-26T18-16-25Z.txt](mypy_snapshots/candidates_fix11_2025-08-26T18-16-25Z.txt)
- Exactly 5 files included:
  1) [src/sensory/organs/dimensions/anomaly_dimension.py](src/sensory/organs/dimensions/anomaly_dimension.py)
  2) [src/thinking/memory/pattern_memory.py](src/thinking/memory/pattern_memory.py)
  3) [src/sensory/what/what_sensor.py](src/sensory/what/what_sensor.py)
  4) [src/sensory/organs/dimensions/base_organ.py](src/sensory/organs/dimensions/base_organ.py)
  5) [src/integration/component_integrator_impl.py](src/integration/component_integrator_impl.py)

Global acceptance criteria
- After applying Round A proposals and running ruff/isort/black and mypy base + strict-on-touch on the edited file(s), each of the 5 files reports zero mypy errors in isolation.
- Plan contains structured sections for all 5 candidates, uses clickable [path.py](path.py:line) references for each listed error, and makes no source code changes.

Reference conventions for Round A
- Safe numeric coercions at return/aggregation sites: [float()](python:1), [int()](python:1), [Decimal()](decimal:1)
- Guard and narrow Optionals/objects: [isinstance()](python:1), [typing.cast()](typing:1)
- Add explicit return annotations for procedures (e.g., -> None for [__init__()](python:1))
- Typing-time imports: [TYPE_CHECKING](typing:1) to avoid runtime coupling


Summary of errors across the 5 files
- Total files planned: 5
- Total mypy errors tallied from snapshot: 8


File 1 of 5: src/sensory/organs/dimensions/anomaly_dimension.py
- Current error count: 4

Exact mypy entries
- [src/sensory/organs/dimensions/anomaly_dimension.py](src/sensory/organs/dimensions/anomaly_dimension.py:93): error: Argument 1 to "zscore" has incompatible type "ExtensionArray | ndarray[tuple[Any, ...], dtype[Any]]"; expected "CanArray[tuple[Any, ...], dtype[integer[Any] | numpy.bool[builtins.bool]]]"  [arg-type]
- [src/sensory/organs/dimensions/anomaly_dimension.py](src/sensory/organs/dimensions/anomaly_dimension.py:93): error: Argument 1 to "zscore" has incompatible type "ExtensionArray | ndarray[tuple[Any, ...], dtype[Any]]"; expected "CanArray[Any, dtype[integer[Any] | numpy.bool[builtins.bool]]]"  [arg-type]
- [src/sensory/organs/dimensions/anomaly_dimension.py](src/sensory/organs/dimensions/anomaly_dimension.py:121): error: Argument 1 to "mean" has incompatible type "ExtensionArray | ndarray[tuple[Any, ...], dtype[Any]]"; expected "_SupportsArray[dtype[numpy.bool[builtins.bool] | integer[Any] | floating[Any]]] | _NestedSequence[_SupportsArray[dtype[numpy.bool[builtins.bool] | integer[Any] | floating[Any]]]] | float | _NestedSequence[float]"  [arg-type]
- [src/sensory/organs/dimensions/anomaly_dimension.py](src/sensory/organs/dimensions/anomaly_dimension.py:122): error: Argument 1 to "std" has incompatible type "ExtensionArray | ndarray[tuple[Any, ...], dtype[Any]]"; expected "_SupportsArray[dtype[numpy.bool[builtins.bool] | number[Any, int | float | complex]]] | _NestedSequence[_SupportsArray[dtype[numpy.bool[builtins.bool] | number[Any, int | float | complex]]]] | complex | _NestedSequence[complex]"  [arg-type]

Categorization
- Numpy/scipy numeric array typing: pandas ExtensionArray passed into numeric APIs expecting numeric ndarrays.
- Scalar/array normalization.

Round A minimal changes (≤5 edits; behavior-preserving)
1) Pre-normalize price series to float ndarray before zscore:
   - Introduce local: prices_np = np.asarray(prices, dtype=float); call stats.zscore(prices_np).
2) Pre-normalize volumes to float ndarray before mean:
   - volumes_np = np.asarray(volumes, dtype=float); use np.mean(volumes_np).
3) Use the same volumes_np for std:
   - std_volume = np.std(volumes_np).
4) Optional local type hints if needed to quiet inference:
   - prices_np: "ndarray[Any, dtype[floating[Any]]]" or simpler approach: rely on np.asarray dtype=float without extra annotations.
5) Only if upstream Optionals/object surfaces exist, guard via [isinstance()](python:1) and narrow with [typing.cast()](typing:1); avoid unless required.

Round B single smallest additional change (placeholder)
- Factor a tiny helper normalize_to_float_ndarray(x) -> np.ndarray that returns np.asarray(x, dtype=float) and call it for prices/volumes.

Acceptance criteria (file-specific)
- After Round A, zero mypy errors for [src/sensory/organs/dimensions/anomaly_dimension.py](src/sensory/organs/dimensions/anomaly_dimension.py) when checked in isolation with base + strict-on-touch.


File 2 of 5: src/thinking/memory/pattern_memory.py
- Current error count: 1

Exact mypy entry
- [src/thinking/memory/pattern_memory.py](src/thinking/memory/pattern_memory.py:155): error: Argument 1 to "mean" has incompatible type "list[object]"; expected "_SupportsArray[dtype[numpy.bool[builtins.bool] | integer[Any] | floating[Any]]] | _NestedSequence[_SupportsArray[dtype[numpy.bool[builtins.bool] | integer[Any] | floating[Any]]]] | float | _NestedSequence[float]"  [arg-type]

Categorization
- Any/object to numeric narrowing at aggregation site.

Round A minimal changes (≤5 edits; behavior-preserving)
1) Coerce values to float at the use site:
   - avg_outcome = float(np.mean([float(x) for x in outcomes])) if outcomes else 0

Round B single smallest additional change (placeholder)
- Type outcomes earlier as list[float] at construction to avoid repeated coercions.

Acceptance criteria (file-specific)
- After Round A, zero mypy errors for [src/thinking/memory/pattern_memory.py](src/thinking/memory/pattern_memory.py) in isolation.


File 3 of 5: src/sensory/what/what_sensor.py
- Current error count: 1

Exact mypy entry
- [src/sensory/what/what_sensor.py](src/sensory/what/what_sensor.py:38): error: Missing type parameters for generic type "dict"  [type-arg]

Categorization
- Explicit type arguments needed for invariant generic dict.

Round A minimal changes (≤5 edits; behavior-preserving)
1) Specify dict type parameters at declaration:
   - patterns: dict[str, Any] = {}
2) Ensure import exists:
   - from typing import Any (add only if missing).

Round B single smallest additional change (placeholder)
- If schema is known, narrow value type from Any to a concrete TypedDict or Mapping[str, object].

Acceptance criteria (file-specific)
- After Round A, zero mypy errors for [src/sensory/what/what_sensor.py](src/sensory/what/what_sensor.py) in isolation.


File 4 of 5: src/sensory/organs/dimensions/base_organ.py
- Current error count: 1

Exact mypy entry
- [src/sensory/organs/dimensions/base_organ.py](src/sensory/organs/dimensions/base_organ.py:18): error: Module "pydantic" has no attribute "model_validator"; maybe "root_validator"?  [attr-defined]

Categorization
- Pydantic v1/v2 compatibility; attr-defined on import symbol.

Round A minimal changes (≤5 edits; behavior-preserving)
1) Provide a version-flexible import shim:
   - try:
     - from pydantic import BaseModel, Field, model_validator, validator
   - except ImportError:
     - from pydantic import BaseModel, Field, root_validator as model_validator, validator

Round B single smallest additional change (placeholder)
- If mypy still flags, gate the import via [TYPE_CHECKING](typing:1) and alias for static typing while keeping runtime shim.

Acceptance criteria (file-specific)
- After Round A, zero mypy errors for [src/sensory/organs/dimensions/base_organ.py](src/sensory/organs/dimensions/base_organ.py) in isolation.


File 5 of 5: src/integration/component_integrator_impl.py
- Current error count: 1

Exact mypy entry
- [src/integration/component_integrator_impl.py](src/integration/component_integrator_impl.py:97): error: Module "src.core.risk.manager" has no attribute "RiskConfig"  [attr-defined]

Categorization
- Attr-defined on cross-module import; type-only usage should avoid runtime dependency.

Round A minimal changes (≤5 edits; behavior-preserving)
1) Gate the import under [TYPE_CHECKING](typing:1) and use forward references:
   - from typing import TYPE_CHECKING
   - if TYPE_CHECKING:
     - from src.core.risk.manager import RiskConfig
   - Use 'RiskConfig' (string) in annotations where referenced.
2) Only if symbol is needed at runtime, wrap a lazy/try import and substitute Any; otherwise rely on type-only import.

Round B single smallest additional change (placeholder)
- Provide a minimal local Protocol for RiskConfig with just accessed attributes to satisfy typing without importing the module.

Acceptance criteria (file-specific)
- After Round A, zero mypy errors for [src/integration/component_integrator_impl.py](src/integration/component_integrator_impl.py) in isolation.


Method
1) Confirm candidates from the list file. Done.
2) Extract exact mypy entries from the snapshot for those 5 files. Done above with clickable refs.
3) Apply Round A minimal, behavior-preserving edits (≤5 per file).
4) Run ruff/isort/black; re-run mypy base + strict-on-touch on changed files; verify zero errors for each file.
5) If any residual issue remains, use the single Round B placeholder per file.

Notes
- This document is diagnostics-only; no source code changes are performed here.
- Prefer site-local normalizations and typing-only imports to minimize blast radius and preserve behavior.