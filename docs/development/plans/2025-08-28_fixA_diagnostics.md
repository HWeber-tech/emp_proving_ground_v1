# Fix A — Diagnostics and minimal remediation plan (2025-08-28)

1) Overview
- Stable baseline under py311 + mypy==1.17.1: Found 125 errors in 31 files (checked 343 source files)
- Scope: Fix A focuses on environment regression remediation via minimal, behavior-preserving edits only. No functional changes; keep per-file edits ≤5 where possible.
- Posture references:
  - Python: 3.11
  - mypy: 1.17.1
  - mypy posture: see [mypy.ini](mypy.ini:1) — globally transition-friendly with package-level L3 strictness in many domains (e.g., thinking.*, trading.*, evolution.*, validation.*, etc.).

2) Working set selection (≤10 modules)
Derived from [mypy_ranked_offenders_py311_2025-08-28T02-42-45Z.csv](../../mypy_snapshots/mypy_ranked_offenders_py311_2025-08-28T02-42-45Z.csv:1) using heuristics:
- Highest error yield yet amenable to minimal corrections (localized patterns, repetitive issues).
- Prefer src/*; exclude third-party, generated, migrations, stubs.
- Avoid core orchestrator hubs likely to induce cycles; bias to leaf-ish modules.
- Balance across packages to reduce cross-coupling risk.

Chosen modules with current error counts:
- src/thinking/patterns/trend_detector.py — 18
- src/data_integration/data_fusion.py — 10
- src/intelligence/sentient_adaptation.py — 9
- src/thinking/patterns/cycle_detector.py — 8
- src/evolution/mutation/gaussian_mutation.py — 7
- src/trading/portfolio/real_portfolio_monitor.py — 7
- src/genome/models/genome_adapter.py — 6
- src/orchestration/enhanced_intelligence_engine.py — 5
- src/sensory/organs/dimensions/institutional_tracker.py — 5
- src/validation/phase2d_simple_integration.py — 5

Rationale summary:
- Captures top offenders by count while balancing across thinking, data_integration, intelligence, evolution, trading, genome, orchestration, sensory, and validation.
- Selected modules appear leaf-ish or mid-level with localized logic; orchestration modules included sparingly with emphasis on import hygiene to avoid cycles.
- Excluded near-peers where patterns duplicate to maintain breadth and keep total expected edits ≤50.

3) Per-module quick diagnosis and minimal remediation steps
Reference patterns from [typing_recipes.md](../typing_recipes.md:1). Apply only behavior-preserving, low-risk changes. Use targeted fixes that clear clusters of errors, e.g., Optional guards, typed locals, and precise return annotations. Use type-only imports via [typing.TYPE_CHECKING](python.TYPE_CHECKING:1) when needed.

A) src/thinking/patterns/trend_detector.py (18)
- Likely dominant mypy codes: operator, index, return-value, union-attr, attr-defined.
- Quick diagnosis: time series math and window ops mixing int/float; Optional inputs producing union-attr errors; inferred Any lists/dicts.
- Minimal remediation:
  - Normalize numeric inputs early via [float()](python.float():1)/[int()](python.int():1) per boundary recipe.
  - Add explicit return types for public functions.
  - Guard Optionals: explicit early returns; use isinstance() + [typing.cast()](python.cast():1) when narrowing unions becomes necessary.
  - Type critical locals, e.g., windows: list[float], payload: [dict[str, object]](python.dict():1).
- Est. edits: 5 (may clear majority); residuals, if any, deferred to next batch.
- Risk/deferral note: If more than 5 edits required to settle union/attr-defined chains, stop at 5 and defer.

B) src/data_integration/data_fusion.py (10)
- Likely codes: assignment, index, call-arg, return-value.
- Quick diagnosis: heterogeneous record payloads crossing boundaries; missing return annotations; dict access on loosely-typed inputs.
- Minimal remediation:
  - Add return annotations and annotate key locals: records: list[dict[str, object]].
  - Normalize input payloads to [dict[str, object]](python.dict():1) at boundaries; downstream guard + cast as needed.
  - Introduce small [TypedDict](python.TypedDict():1) only if a single, tiny schema clears multiple errors; otherwise prefer cast after guards.
- Est. edits: 4–5.

C) src/intelligence/sentient_adaptation.py (9)
- Likely codes: attr-defined, union-attr, return-value.
- Quick diagnosis: adaptation steps reading from optional state; helper calls lacking precise annotations.
- Minimal remediation:
  - Optional guards with explicit early returns.
  - Add explicit return types; annotate pivotal parameters.
  - Where runtime checks already exist, follow with [typing.cast()](python.cast():1) to lock inference.
- Est. edits: 4–5.

D) src/thinking/patterns/cycle_detector.py (8)
- Likely codes: operator, index, return-value.
- Quick diagnosis: cycle metrics blending ints/floats; untyped locals; indexing on unions.
- Minimal remediation:
  - Early normalization with [float()](python.float():1)/[int()](python.int():1).
  - Annotate locals for rolling windows and periods (list[float], int).
  - Add explicit return types; guard Optionals and narrow with isinstance().
- Est. edits: 4–5.

E) src/evolution/mutation/gaussian_mutation.py (7)
- Likely codes: operator, call-arg, assignment.
- Quick diagnosis: numeric ops and RNG parameters inferred as Any/Union; mixed numeric types.
- Minimal remediation:
  - Explicit types for sigma, mu, and mutation rate; normalize to float.
  - Add return annotations; type any RNG/np arrays where locally held.
  - If heavy imports cause cycles, move type-only imports under [typing.TYPE_CHECKING](python.TYPE_CHECKING:1).
- Est. edits: 3–4.

F) src/trading/portfolio/real_portfolio_monitor.py (7)
- Likely codes: attr-defined, union-attr, return-value.
- Quick diagnosis: portfolio snapshots may be Optional; attribute access without guards; monitoring outputs lack explicit types.
- Minimal remediation:
  - Optional guards with early returns; isinstance() checks on message payloads.
  - Add return annotations for monitor tick/update functions.
  - Type-only imports for models under [typing.TYPE_CHECKING](python.TYPE_CHECKING:1) if import pressure exists.
- Est. edits: 4–5.

G) src/genome/models/genome_adapter.py (6)
- Likely codes: attr-defined, type-var/Protocol gaps, return-value.
- Quick diagnosis: adapter interface implied but not typed; repeated attribute assumptions.
- Minimal remediation:
  - Introduce minimal [Protocol](python.Protocol():1) for the adapter surface only if it collapses several errors at once; otherwise rely on explicit return annotations and targeted [typing.cast()](python.cast():1) after guards.
  - Add precise returns and key param annotations.
- Est. edits: 4–5.
- Note: If Protocol requires >5 edits to thread, prefer local casts and defer Protocol formalization.

H) src/orchestration/enhanced_intelligence_engine.py (5)
- Likely codes: return-value, attr-defined, import cycle sensitivity.
- Quick diagnosis: orchestrator referencing concrete implementations; potential import cycles.
- Minimal remediation:
  - Add return annotations; annotate key locals for stages/pipelines.
  - Move heavy type dependencies to [typing.TYPE_CHECKING](python.TYPE_CHECKING:1); use runtime-local imports if necessary at use sites.
  - Narrow via isinstance() + [typing.cast()](python.cast():1) where interface variance confuses inference.
- Est. edits: 4–5.

I) src/sensory/organs/dimensions/institutional_tracker.py (5)
- Likely codes: index, attr-defined, assignment.
- Quick diagnosis: dimension payloads and feature maps using loose dicts; missing annotations.
- Minimal remediation:
  - Normalize intermediate payloads to [dict[str, object]](python.dict():1); add guards before attribute/index access; apply [typing.cast()](python.cast():1) post-guard.
  - Add explicit returns for public entry points.
- Est. edits: 3–4.

J) src/validation/phase2d_simple_integration.py (5)
- Likely codes: return-value, call-arg, union-attr.
- Quick diagnosis: integration harness returning tuples or heterogeneous results; optional resources.
- Minimal remediation:
  - Explicit return annotations; if tuples are used, annotate precisely or narrow via [typing.cast()](python.cast():1) after runtime checks.
  - Guard Optionals for external resources; annotate locals used across phases.
- Est. edits: 3–4.

Total expected changes: ~41–47 edits across the working set. Any file exceeding 5 edits during implementation should stop at 5 and defer residuals to a later batch.

4) Verification checklist for Code mode
- Formatting and hygiene:
  - Run ruff with autofix, black, isort (profile=black) on edited files.
- Static typing:
  - Run mypy base config and strict-on-touch for the edited files; ensure zero new errors, and net errors decrease.
- Snapshot and reporting:
  - Update mypy snapshot artifacts alongside the ranked offenders CSV.
  - Append “Fix A results” to [2025-08-27_mypy_env_comparison.md](../reports/2025-08-27_mypy_env_comparison.md:1) capturing deltas: total error count change, files improved, any deferrals.
- Import hygiene:
  - If import cycles arise, move type imports under [typing.TYPE_CHECKING](python.TYPE_CHECKING:1) and, where needed, perform runtime-local imports at use sites with immediate validation + [typing.cast()](python.cast():1).

5) Out-of-scope for Fix A
- No refactors or public API changes.
- No cross-module architectural moves beyond established import hygiene (type-only imports, local runtime imports as needed).
- No widening to Any; tighten types where obvious and avoid blanket [cast(Any, ...)](python.cast():1).

References
- Typing patterns and posture: [typing_recipes.md](../typing_recipes.md:1)
- Configuration: [mypy.ini](mypy.ini:1)
- Ranked offenders input: [mypy_ranked_offenders_py311_2025-08-28T02-42-45Z.csv](../../mypy_snapshots/mypy_ranked_offenders_py311_2025-08-28T02-42-45Z.csv:1)