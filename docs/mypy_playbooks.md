# Mypy Remediation Playbooks

These playbooks capture the concrete fixes we used while clearing the
historical mypy backlog. Each section pairs a recurring error pattern
with a lightweight diagnostic flow so engineers can resolve new findings
without rediscovering the solution from scratch.

Use this guide alongside [`docs/mypy_conventions.md`](mypy_conventions.md)
and the backlog inventory table when triaging a fresh mypy report.

## Numeric Conversion Failures (`int()` / `float()`)

**Symptoms**
- `Argument 1 to "int" has incompatible type "object"; expected "SupportsInt"`
- `Unsupported operand type(s) for +: "float" and "None"`

**Playbook**
1. Identify the upstream value source. Annotate it as a union of the
   actual runtime types (`str | int | float | Decimal | None`).
2. Replace direct casts with helpers from `src/core/coercion.py`
   (`coerce_int`, `coerce_float`, `coerce_decimal`).
3. Provide an explicit fallback for `None` or invalid inputs and log the
   failure when it indicates data quality issues.
4. Update call sites and tests to expect the normalised behaviour.

## Invalid Event Construction

**Symptoms**
- `Unexpected keyword argument "type" for "Event"`
- `Cannot determine type of "payload"` when instantiating events

**Playbook**
1. Confirm the intended event class. Import the project-specific event
   factory instead of the stdlib `threading.Event` when payload metadata
   is required.
2. Where a helper wraps event creation, annotate its return type and
   validate the accepted keyword arguments in one place.
3. For third-party types without keywords, introduce a thin adapter that
   translates our richer metadata into the supported constructor shape.

## Missing Annotations & `Any` Leakage

**Symptoms**
- `Function is missing a return type annotation`
- `Need type annotation for "meta"`

**Playbook**
1. Add return annotations to every function touched while remediating an
   error. Prefer `-> None` over leaving it implicit.
2. Annotate module-level variables—especially mutables—with their
   precise container types (`dict[str, object]`, `list[Plan]`, etc.).
3. Where dynamic keys exist, introduce `TypedDict` or dataclasses to
   define the schema explicitly.
4. Rerun mypy with `--warn-return-any` locally when editing a module that
   previously leaked `Any` to confirm new annotations are effective.

## Container Variance & Mapping Mismatches

**Symptoms**
- `Incompatible return value type (got "dict[str, float]", expected "dict[str, str]")`
- `Argument 1 to "update" has incompatible type "Mapping[str, object]"; expected "Mapping[str, str]"`

**Playbook**
1. Audit the contract—should callers mutate the return value? If not,
   widen to `Mapping`/`Sequence` in the signature.
2. Introduce value unions that capture the real payload (`dict[str, str | float | None]`).
3. When mutation is necessary, normalise the data before mutation via a
   helper function that owns the concrete container type.
4. Add regression tests or fixtures that cover mixed-type payloads to
   guard the widened typing.

## Optional Imports & Stub Gaps

**Symptoms**
- `Cannot find implementation or library stub for module named "redis"`
- `Value of type "KafkaProducer" has no attribute "flush"`

**Playbook**
1. Gate imports behind `typing.TYPE_CHECKING` and expose runtime shims
   returning `Any` when the dependency is absent.
2. Add a `.pyi` stub under `stubs/` covering the subset of the external
   API we exercise. Include doc links in comments for maintainability.
3. Where behaviour varies by backend (e.g., optional Redis clients), add
   protocol definitions that capture the shared surface.
4. Document optional dependency expectations in README or module docs so
   future contributors keep shims aligned.

## `# type: ignore` Hygiene

**Symptoms**
- `unused 'type: ignore' comment`
- `"type: ignore" comment should target specific error codes`

**Playbook**
1. Run mypy with `--warn-unused-ignores` locally to identify stale
   suppressions.
2. Remove unused ignores; if the error persists, annotate the root cause
   instead of masking it.
3. Where a temporary ignore is unavoidable (e.g., third-party bug), add
   an inline TODO with owner and exit criteria plus the specific error
   code (`# type: ignore[attr-defined]  # TODO(app-team): replace after stub update`).

## Protocol & Callable Mismatches

**Symptoms**
- `Argument 1 to "spawn_task" has incompatible type "Callable[..., Awaitable[Any]]"; expected "Coroutine[Any, Any, Any]"`
- `Type "TimescaleConnector" does not satisfy the protocol "RedisLike"`

**Playbook**
1. Update protocol definitions with the minimum required surface area and
   defaulted generics where appropriate.
2. Wrap callables or coroutines with adapters so the signature matches
   the protocol exactly (e.g., convert `Callable` to `Coroutine` via an
   async wrapper).
3. Leverage `typing.Protocol` with `@runtime_checkable` to validate
   compliance during tests when practical.
4. Add regression tests that execute the adapter path so mismatched
   signatures are caught before reaching mypy.

## Operational Checklist

When remediating a backlog module:
- [ ] Capture the failing command and error message in the PR description
      or commit body for traceability.
- [ ] Apply the relevant playbook above and update documentation if a
      new pattern emerges.
- [ ] Re-run `mypy --config-file mypy.ini <module>` and `pytest -q` to
      confirm both type safety and runtime health.
- [ ] Remove any obsolete suppressions and keep the conventions guide in
      sync with the fix.

