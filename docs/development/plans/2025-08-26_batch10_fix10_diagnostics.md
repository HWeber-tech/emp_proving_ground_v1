# Batch10 Fix10 — Diagnostics-only Plan (2025-08-26)

Snapshot context: Found 221 errors in 48 files (Post-Batch10 fix9 snapshot) — see [mypy_snapshots/mypy_summary_2025-08-26T17-41-02Z.txt](mypy_snapshots/mypy_summary_2025-08-26T17-41-02Z.txt:1)

Scope: Exactly these 8 candidates from [changed_files_batch10_fix10_candidates.txt](changed_files_batch10_fix10_candidates.txt:1)
- [src/domain/__init__.py](src/domain/__init__.py)
- [src/sensory/enhanced/when_dimension.py](src/sensory/enhanced/when_dimension.py)
- [src/sensory/enhanced/what_dimension.py](src/sensory/enhanced/what_dimension.py)
- [src/operational/health_monitor.py](src/operational/health_monitor.py)
- [src/sentient/learning/real_time_learning_engine.py](src/sentient/learning/real_time_learning_engine.py)
- [src/ecosystem/coordination/coordination_engine.py](src/ecosystem/coordination/coordination_engine.py)
- [src/intelligence/portfolio_evolution.py](src/intelligence/portfolio_evolution.py)
- [src/thinking/ecosystem/specialized_predator_evolution.py](src/thinking/ecosystem/specialized_predator_evolution.py)

Method used
1) Enumerated 8 candidates via [changed_files_batch10_fix10_candidates.txt](changed_files_batch10_fix10_candidates.txt:1).
2) Extracted mypy entries for each from [mypy_snapshots/mypy_snapshot_2025-08-26T17-41-02Z.txt](mypy_snapshots/mypy_snapshot_2025-08-26T17-41-02Z.txt:1).
3) Drafted per-file analysis, minimal change proposals (Round A, ≤5 edits), and a Round B single smallest follow-up placeholder.
4) No source code changes are made in this plan.

Conventions aligned with prior batches
- Prefer behavior-preserving, narrow edits.
- Safe numeric coercions at return/compare sites via [float()](python:1), [int()](python:1).
- Guard Optionals/object via [isinstance()](python:1) and [typing.cast()](typing:1).
- Add explicit return annotations where missing (e.g., -> None).
- Annotate locals using list[T], dict[str, V]; heterogeneous payloads as dict[str, object].
- Use [TYPE_CHECKING](typing:1) to isolate type-time imports. Minimal runtime stubs only if strictly necessary.

---

File 1 of 8 — [src/domain/__init__.py](src/domain/__init__.py)

Errors (1)
- [src/domain/__init__.py](src/domain/__init__.py:12) — Module "src.core.risk.manager" has no attribute "RiskConfig" [attr-defined]. Extracted from [mypy_snapshots/mypy_snapshot_2025-08-26T17-41-02Z.txt](mypy_snapshots/mypy_snapshot_2025-08-26T17-41-02Z.txt:9)

Category
- Typing-time vs runtime import (attr-defined)

Round A minimal changes (≤5, behavior-preserving)
1) Gate the re-export under [TYPE_CHECKING](typing:1) and provide a runtime fallback:
   - Add: from typing import TYPE_CHECKING
   - if TYPE_CHECKING: from src.core.risk.manager import RiskConfig as RiskConfig
   - else: RiskConfig = object  # type: ignore[assignment]
2) If __all__ references RiskConfig, keep as-is; gating retains runtime behavior.

Round B single smallest additional change
- Replace runtime fallback with a tiny Protocol stub defining only attributes actually used, if mypy later requires structure.

Acceptance criteria (per-file)
- Running ruff/isort/black and mypy (base + strict-on-touch) after these edits yields zero mypy errors for [src/domain/__init__.py](src/domain/__init__.py).

---

File 2 of 8 — [src/sensory/enhanced/when_dimension.py](src/sensory/enhanced/when_dimension.py)

Errors (1)
- [src/sensory/enhanced/when_dimension.py](src/sensory/enhanced/when_dimension.py:43) — Argument "regime" to "DimensionalReading" has incompatible type "src.sensory.organs.dimensions.base_organ.MarketRegime"; expected "src.core.base.MarketRegime" [arg-type]. Extracted from [mypy_snapshots/mypy_snapshot_2025-08-26T17-41-02Z.txt](mypy_snapshots/mypy_snapshot_2025-08-26T17-41-02Z.txt:135)

Category
- Protocol/type mismatch (enum type divergence between domains)

Round A minimal changes (≤5)
1) Import the expected enum: from src.core.base import MarketRegime as MarketRegime
2) Ensure construction uses the correct enum: regime=MarketRegime.UNKNOWN
   - If other references to MarketRegime in this file exist, consistently use the core enum.

Round B single smallest additional change
- If both enums are needed, add a tiny adapter mapping sensory → core enum values before constructing DimensionalReading.

Acceptance criteria (per-file)
- After edits + formatters + mypy, zero errors remain for [src/sensory/enhanced/when_dimension.py](src/sensory/enhanced/when_dimension.py).

---

File 3 of 8 — [src/sensory/enhanced/what_dimension.py](src/sensory/enhanced/what_dimension.py)

Errors (1)
- [src/sensory/enhanced/what_dimension.py](src/sensory/enhanced/what_dimension.py:43) — Argument "regime" to "DimensionalReading" has incompatible type "src.sensory.organs.dimensions.base_organ.MarketRegime"; expected "src.core.base.MarketRegime" [arg-type]. Extracted from [mypy_snapshots/mypy_snapshot_2025-08-26T17-41-02Z.txt](mypy_snapshots/mypy_snapshot_2025-08-26T17-41-02Z.txt:141)

Category
- Protocol/type mismatch (enum type divergence between domains)

Round A minimal changes (≤5)
1) Import expected enum: from src.core.base import MarketRegime as MarketRegime
2) Use correct enum in construction: regime=MarketRegime.UNKNOWN

Round B single smallest additional change
- Add a minimal mapping utility if both enum spaces must coexist and values need translation.

Acceptance criteria (per-file)
- After edits + formatters + mypy, zero errors remain for [src/sensory/enhanced/what_dimension.py](src/sensory/enhanced/what_dimension.py).

---

File 4 of 8 — [src/operational/health_monitor.py](src/operational/health_monitor.py)

Errors (1)
- [src/operational/health_monitor.py](src/operational/health_monitor.py:28) — Need type annotation for "health_history" (hint: "health_history: list[<type>] = ...") [var-annotated]. Extracted from [mypy_snapshots/mypy_snapshot_2025-08-26T17-41-02Z.txt](mypy_snapshots/mypy_snapshot_2025-08-26T17-41-02Z.txt:270)

Category
- Untyped variable (var-annotated)

Round A minimal changes (≤5)
1) Add explicit variable annotation at assignment:
   - self.health_history: list[dict[str, object]] = []
   - Rationale: history entries are typically heterogeneous dict payloads; using dict[str, object] keeps behavior and unblocks typing.

Round B single smallest additional change
- Introduce a TypedDict for health entries if structure stabilizes, and annotate to list[HealthEntryTD].

Acceptance criteria (per-file)
- After edits + formatters + mypy, zero errors remain for [src/operational/health_monitor.py](src/operational/health_monitor.py).

---

File 5 of 8 — [src/sentient/learning/real_time_learning_engine.py](src/sentient/learning/real_time_learning_engine.py)

Errors (1)
- [src/sentient/learning/real_time_learning_engine.py](src/sentient/learning/real_time_learning_engine.py:128) — Returning Any from function declared to return "bool" [no-any-return]. Extracted from [mypy_snapshots/mypy_snapshot_2025-08-26T17-41-02Z.txt](mypy_snapshots/mypy_snapshot_2025-08-26T17-41-02Z.txt:515)

Category
- Any leaks (boolean expression operands are Any)

Round A minimal changes (≤5)
1) Coerce numeric operands to float and wrap entire expression in bool:
   - return bool(float(price_change) > 0.001 and float(volume) > 3 * float(avg_volume))
   - Behavior-preserving; normalizes scalars.
2) Optional if needed: add local typed temps, e.g., pc = float(price_change), etc., then return bool(pc > 0.001 and vol > 3 * avg), but keep within ≤5 edits.

Round B single smallest additional change
- If inputs are consistently numeric, annotate the method signature parameters as float to prevent future Any propagation.

Acceptance criteria (per-file)
- After edits + formatters + mypy, zero errors remain for [src/sentient/learning/real_time_learning_engine.py](src/sentient/learning/real_time_learning_engine.py).

---

File 6 of 8 — [src/ecosystem/coordination/coordination_engine.py](src/ecosystem/coordination/coordination_engine.py)

Errors (1)
- [src/ecosystem/coordination/coordination_engine.py](src/ecosystem/coordination/coordination_engine.py:299) — Return type "Coroutine[Any, Any, JSONObject]" of "get_portfolio_summary" incompatible with return type "Coroutine[Any, Any, dict[str, object]]" in supertype "src.core.interfaces.ICoordinationEngine" [override]. Extracted from [mypy_snapshots/mypy_snapshot_2025-08-26T17-41-02Z.txt](mypy_snapshots/mypy_snapshot_2025-08-26T17-41-02Z.txt:527)

Category
- Protocol mismatch (override return type)

Round A minimal changes (≤5)
1) Align signature with interface: async def get_portfolio_summary(self) -> dict[str, object]:
2) If the implementation composes a JSON-like alias, cast at return site to satisfy invariance: return [typing.cast()](typing:1)[dict[str, object]](payload)

Round B single smallest additional change
- If multiple call sites require Mapping, change return type to Mapping[str, object] across interface and implementation in a coordinated batch (out of scope for this single-file plan).

Acceptance criteria (per-file)
- After edits + formatters + mypy, zero errors remain for [src/ecosystem/coordination/coordination_engine.py](src/ecosystem/coordination/coordination_engine.py).

---

File 7 of 8 — [src/intelligence/portfolio_evolution.py](src/intelligence/portfolio_evolution.py)

Errors (1)
- [src/intelligence/portfolio_evolution.py](src/intelligence/portfolio_evolution.py:558) — Need type annotation for "weighted_volatility" [var-annotated]. Extracted from [mypy_snapshots/mypy_snapshot_2025-08-26T17-41-02Z.txt](mypy_snapshots/mypy_snapshot_2025-08-26T17-41-02Z.txt:614)

Category
- Untyped variable (var-annotated) + numeric normalization

Round A minimal changes (≤5)
1) Add explicit annotation and normalize numpy scalar to builtin:
   - weighted_volatility: float = float(np.sum(weights * volatilities))
   - Keeps runtime behavior while satisfying typing.

Round B single smallest additional change
- If weights/volatilities are numpy arrays, annotate their types to ndarray[...] to prevent future Any propagation.

Acceptance criteria (per-file)
- After edits + formatters + mypy, zero errors remain for [src/intelligence/portfolio_evolution.py](src/intelligence/portfolio_evolution.py).

---

File 8 of 8 — [src/thinking/ecosystem/specialized_predator_evolution.py](src/thinking/ecosystem/specialized_predator_evolution.py)

Errors (1)
- [src/thinking/ecosystem/specialized_predator_evolution.py](src/thinking/ecosystem/specialized_predator_evolution.py:33) — Module "src.ecosystem.species.species_manager" has no attribute "SpeciesManager" [attr-defined]. Extracted from [mypy_snapshots/mypy_snapshot_2025-08-26T17-41-02Z.txt](mypy_snapshots/mypy_snapshot_2025-08-26T17-41-02Z.txt:928)

Category
- Typing-time vs runtime import (attr-defined)

Round A minimal changes (≤5)
1) Gate import under [TYPE_CHECKING](typing:1) and provide a runtime fallback:
   - from typing import TYPE_CHECKING
   - if TYPE_CHECKING: from src.ecosystem.species.species_manager import SpeciesManager as SpeciesManager
   - else: SpeciesManager = object  # type: ignore[assignment]
2) Leave any runtime construction paths unchanged; the fallback only affects type checking.

Round B single smallest additional change
- Introduce a minimal Protocol stub for SpeciesManager if mypy requires attribute access in annotations.

Acceptance criteria (per-file)
- After edits + formatters + mypy, zero errors remain for [src/thinking/ecosystem/specialized_predator_evolution.py](src/thinking/ecosystem/specialized_predator_evolution.py).

---

Global acceptance criteria
- The plan covers exactly the 8 candidates from [changed_files_batch10_fix10_candidates.txt](changed_files_batch10_fix10_candidates.txt:1).
- Each file section includes current mypy errors with clickable [path.py](path.py:line) references, error type categorization, Round A proposals (≤5 edits/file, behavior-preserving), and a single Round B placeholder.
- After applying Round A proposals and running ruff/isort/black plus mypy (base + strict-on-touch) on the edited files, each targeted file is free of mypy errors in isolation.

Summary of tallied errors in candidates
- Files planned: 8
- Total errors across these files (from snapshot): 8

At-a-glance mapping: file → count
- [src/domain/__init__.py](src/domain/__init__.py): 1
- [src/sensory/enhanced/when_dimension.py](src/sensory/enhanced/when_dimension.py): 1
- [src/sensory/enhanced/what_dimension.py](src/sensory/enhanced/what_dimension.py): 1
- [src/operational/health_monitor.py](src/operational/health_monitor.py): 1
- [src/sentient/learning/real_time_learning_engine.py](src/sentient/learning/real_time_learning_engine.py): 1
- [src/ecosystem/coordination/coordination_engine.py](src/ecosystem/coordination/coordination_engine.py): 1
- [src/intelligence/portfolio_evolution.py](src/intelligence/portfolio_evolution.py): 1
- [src/thinking/ecosystem/specialized_predator_evolution.py](src/thinking/ecosystem/specialized_predator_evolution.py): 1