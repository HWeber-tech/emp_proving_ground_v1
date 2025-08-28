# Typing recipes (Batch10)

Context:
- Latest clean snapshot: [mypy_summary_2025-08-27T09-43-50Z.txt](../../mypy_snapshots/mypy_summary_2025-08-27T09-43-50Z.txt:1)
- PR typing workflow: [typing.yml](../../.github/workflows/typing.yml:1)
- Nightly typing workflow: [typing-nightly.yml](../../.github/workflows/typing-nightly.yml:1)

Numeric normalization
- Normalize early at boundaries: prefer `float(x)`/`int(x)` and `Decimal(str(x))` when precision matters. This prevents mypy float|str unions from leaking into core logic.
- For NumPy inputs, coerce deterministically: `np.asarray(x, dtype=float)` to avoid object dtype and downstream `Any` propagation.

Optional/object guards and narrowing
- Guard Optionals and heterogeneous inputs explicitly: `if x is None: return ...` or `assert x is not None` only where logically guaranteed.
- Use `isinstance` to narrow unions; where inference falls short, apply `typing.cast(T, value)` after a runtime check to make intent explicit.

Function and local typing hygiene
- Always annotate returns, including `-> None` for procedures.
- Prefer typed locals to lock intent: `items: list[T] = []`, `index: dict[str, V] = {}`.
- For heterogeneous payloads crossing layers, normalize to `dict[str, object]` at the seam and re-validate on the receiving side with guards and casts.

Type-only imports and lightweight protocols
- Use `from typing import TYPE_CHECKING` and import heavy modules/types under `if TYPE_CHECKING:` to keep runtime deps slim.
- Introduce minimal `Protocol` or placeholder classes only when needed to decouple imports; avoid widening types to `Any`—encode the minimum behavioral contract.

Import hygiene and layering
- Keep imports pointing inward to stable contracts, not concrete leaf modules. This reduces cycles and stub drift.
- Where importing would cause cycles or heavy import at startup, use dynamic imports at usage points with narrow surfaces and immediate casts after validation.

Async results narrowing
- When async utilities return tuple unions, narrow explicitly:
  - After a guard, use `result = typing.cast[tuple[int, str], result]` to make structure clear to mypy and to readers.
  - Prefer small dataclasses or TypedDicts over ambiguous positionals when feasible.

Stubs posture
- Prefer native third-party typings when available; remove local stubs as upstreams add coverage.
- Trim or delete stale entries under [stubs/src](../../stubs/src:1) once validated by CI snapshots to prevent shadowing upstream improvements.

Formatting and linting alignment
- Align on ruff (autofix), black, and isort (profile=black). Keep config single-sourced and let ruff remove unused imports/ignores.
- Periodically drop obsolete `# type: ignore[...]` after ruff/mypy improvements; avoid blanket ignores.

CI posture and local runs
- PRs enforce changed-files strict-on-touch plus a full-repo mypy job via [typing.yml](../../.github/workflows/typing.yml:1).
- Nightly full-repo snapshot via [typing-nightly.yml](../../.github/workflows/typing-nightly.yml:1) ensures drift is caught early and recorded alongside the [latest snapshot](../../mypy_snapshots/mypy_summary_2025-08-27T09-43-50Z.txt:1).
- Local: run `pre-commit run -a` to apply ruff/black/isort and `mypy` to validate zero-error baseline before pushing.