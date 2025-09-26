# Typing Conventions Guide

This document captures the shared conventions we are following while working
through the CI mypy recovery effort. Treat it as a living reference – propose
updates in the same PR whenever you notice a new recurring pattern or we agree
on an additional rule of thumb.

## Prefer `Mapping`/`Sequence` Interfaces

- **Function inputs:** Accept the most permissive read-only interface that suits
  the behaviour of the function. For example, prefer `Mapping[str, float]` over
  `dict[str, float]` when you only need key/value access.
- **Function outputs:** Return the concrete collection you actually construct,
  but declare the return type as the appropriate protocol (e.g., `Mapping`,
  `Sequence`, `Iterable`) unless mutation semantics are part of the contract.
- **Downstream mutation:** If callers mutate the result, document that
  explicitly and return a `dict` or `list`. Otherwise, lean on protocols to
  avoid invariance issues when widening value types.

## Numeric Conversion Helpers

- Use the shared helpers in `src/core/coercion.py` (`coerce_float`,
  `coerce_int`, etc.) whenever ingesting user or telemetry-provided values.
- Avoid calling `int()` / `float()` on `object`-typed inputs. First narrow the
  type via `isinstance` checks or the shared helpers. Always handle `None`
  inputs gracefully and document the fallback behaviour.

## Event Construction

- Project code should import the project-specific `Event` factory instead of
  the stdlib `threading.Event`. If a module truly needs the stdlib type,
  annotate it explicitly as `threading.Event` to avoid signature confusion.
- When creating telemetry events, prefer dedicated helper factories (e.g.,
  `build_event(...)`) that centralise the accepted keyword arguments. This keeps
  constructor changes local and mypy-compliant.

## Annotation Defaults

- Always annotate public function returns, even when returning `None` – e.g.,
  `def refresh(...) -> None:`.
- Module-level dictionaries or other mutable defaults **must** carry type
  annotations (`payload_cache: dict[str, str] = {}`) to prevent implicit `Any`
  propagation.
- Prefer `Final` for module constants (`MAX_RETRIES: Final = 3`) and `TypedDict`
  or dataclasses when a mapping carries a stable schema.

## Using `Any`

- Treat `Any` as a last resort. If interoperability with third-party libraries
  requires it, add a `TODO` referencing the follow-up plan (stub contribution,
  wrapper refactor, etc.).
- When stubbing third-party packages under `stubs/`, match the minimal surface
  required by our code and link to the upstream docs in a comment.

## `# type: ignore[...]` Hygiene

- Every ignore must carry a reason code (`# type: ignore[arg-type]`) and an
  inline comment explaining the remediation plan if the ignore is expected to
  persist.
- Remove ignores as soon as the underlying issue is resolved; let `mypy` fail
  locally if an ignore becomes obsolete.

## Adding New Modules

- New modules should ship with type annotations from day one. Run
  `mypy --config-file mypy.ini <module>` locally before opening a PR.
- If the module imports optional dependencies, gate those imports behind
  `typing.TYPE_CHECKING` or helper factories that provide typed shims when the
  dependency is absent.

