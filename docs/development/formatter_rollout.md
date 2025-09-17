# Ruff Formatter Rollout Plan

The repository still contains hundreds of legacy files that do not pass
`ruff format`. Rather than landing one giant, review-hostile reformat, we will
normalize the tree in staged slices. Each slice enrolls a directory (or focused
module) in formatter enforcement once it has been manually cleaned up.

## Slicing strategy

| Stage | Scope | Gating tasks | Owner |
| --- | --- | --- | --- |
| 0 | `tests/current/` | Land mechanical formatting PRs for all current regression tests and update fixtures as needed. | QA Guild |
| 1 | `src/system/` & `src/core/configuration.py` | Confirm tests stay green after formatting and document any manual adjustments to config helpers. | Platform |
| 2 | `src/trading/execution/` & `src/trading/models/` | Pair formatting with targeted regression coverage from Phase 7 to keep behavior visible. | Trading |
| 3 | `src/sensory/organs/dimensions/` (per organ) | Format one organ at a time, logging tricky lint suppressions directly in PR descriptions. | Sensory |
| 4 | Remaining packages | Continue package-by-package once the high-traffic areas above are stable. | All |

Stages can proceed in parallel as long as each owner keeps their PRs focused and
communicates any mechanical churn that will affect neighbors (for example shared
fixtures or import paths).

## Opting a directory into enforcement

1. Run `ruff format <path>` locally and ensure the diff only contains mechanical
   whitespace/import updates.
2. Re-run the relevant tests (`pytest tests/current -q`) to catch accidental
   behavior changes.
3. Add the directory (or specific files) to
   `config/formatter/ruff_format_allowlist.txt`, keeping the list alphabetized.
4. Commit the formatting diff and the allowlist change in the same PR so CI will
   enforce the new slice immediately.

The CI workflow now calls `scripts/check_formatter_allowlist.py`, which reads the
allowlist and executes `ruff format --check` for every entry. As the allowlist
grows, the guardrail automatically expands.

## Contributor workflow updates

* Run `ruff format` before opening a PR when you touch a file that already lives
  in the allowlist. CI will fail if the formatter drifted.
* When you work outside the allowlist, prefer `ruff check --select I` to nudge
  import ordering toward the formatter output ahead of time.
* Use review comments or PR descriptions to flag any surprising manual fixes so
  later stages can avoid repeat work.

Check the modernization [roadmap](../roadmap.md) for the current stage status
and ownership expectations.

## Rollout change log

- 2025-09-16 – Stage 0 (`tests/current/`) landed via `ruff format` with no manual
  edits beyond whitespace normalization. Verified `pytest tests/current -q`
  locally before expanding the allowlist.
- 2025-09-16 – Introduced pytest flake telemetry (`tests/.telemetry/flake_runs.json`)
  so future slices can correlate formatter churn with any transient test behavior.
- 2025-09-17 – Stage 1 (`src/system/`, `src/core/configuration.py`) normalized with
  `ruff format`. The formatter collapsed a few multi-line string appends into
  single lines without changing behavior. Confirmed `pytest tests/current -q`
  after updating the formatter allowlist.
- 2025-09-18 – Stage 2 (`src/trading/execution/`, `src/trading/models/`) normalized
  with `ruff format`, expanding the allowlist. Ruff merged an adjacent log string
  in `LiquidityProber`; no manual edits were required. Re-ran
  `pytest tests/current -q` to confirm coverage stayed green.
- 2025-09-19 – Stage 3 kicked off with `src/sensory/organs/dimensions/anomaly_detection.py`
  normalized via `ruff format`. The allowlist now enforces the module; Ruff simply
  reflowed long assignments and no manual edits were necessary.
- 2025-09-19 – Stage 3 continued with `src/sensory/organs/dimensions/base_organ.py`
  normalized. Ruff removed redundant blank lines; no behavioral adjustments were
  needed before expanding the allowlist.
- 2025-09-19 – Stage 3 progressed with `src/sensory/organs/dimensions/chaos_adaptation.py`
  normalized. Ruff only reflowed a nested conditional; pytest stayed green before
  updating the allowlist and roadmap.
- 2025-09-19 – Stage 3 advanced with `src/sensory/organs/dimensions/chaos_dimension.py`
  normalized. No edits were required beyond running `ruff format`; pytest stayed
  green ahead of extending the allowlist and refreshing the roadmap snapshot.
- 2025-09-19 – Stage 3 continued with `src/sensory/organs/dimensions/anomaly_dimension.py`
  normalized. Ruff left the file unchanged; pytest stayed green before extending
  the allowlist and recording the slice across roadmap and debt snapshots.
- 2025-09-19 – Stage 3 progressed with
  `src/sensory/organs/dimensions/integration_orchestrator.py` and
  `src/sensory/organs/dimensions/institutional_tracker.py` normalized. Ruff
  reflowed a long `analyze_timing` call and stripped redundant blank lines; pytest
  stayed green before expanding the allowlist, adding
  `src/sensory/organs/dimensions/data_integration.py`,
  `src/sensory/organs/dimensions/order_flow.py`,
  `src/sensory/organs/dimensions/pattern_engine.py`,
  `src/sensory/organs/dimensions/patterns.py`,
  `src/sensory/organs/dimensions/regime_detection.py`, and
  `src/sensory/organs/dimensions/sensory_signal.py` after verifying they already
  matched the formatter output, and updating the roadmap snapshots.
- 2025-09-19 – Stage 3 continued with
  `src/sensory/organs/dimensions/economic_analysis.py`,
  `src/sensory/organs/dimensions/how_organ.py`,
  `src/sensory/organs/dimensions/indicators.py`,
  `src/sensory/organs/dimensions/macro_intelligence.py`, and
  `src/sensory/organs/dimensions/temporal_system.py` normalized. Ruff reported no
  manual edits; pytest stayed green ahead of expanding the allowlist and queuing
  `src/sensory/organs/dimensions/utils.py` as the next target.
- 2025-09-20 – Stage 3 wrapped with
  `src/sensory/organs/dimensions/__init__.py`,
  `src/sensory/organs/dimensions/utils.py`,
  `src/sensory/organs/dimensions/what_organ.py`,
  `src/sensory/organs/dimensions/when_organ.py`, and
  `src/sensory/organs/dimensions/why_organ.py` confirmed clean under `ruff format`.
  No manual edits were required. The allowlist now enforces the entire
  `src/sensory/organs/dimensions/` package, pytest remained green, and Stage 4
  prep has begun with `src/sensory/organs/analyzers/` on deck.
- 2025-09-21 – Stage 4 kicked off with `src/sensory/organs/analyzers/` verified
  clean under `ruff format`. No manual edits were necessary; the allowlist now
  enforces the package and `pytest tests/current -q` stayed green ahead of
  queueing `src/sensory/organs/economic_organ.py` for the next slice.
- 2025-09-21 – Stage 4 continued with `src/sensory/organs/economic_organ.py`
  confirmed clean under `ruff format`. Ruff reported no changes, pytest stayed
  green, and the allowlist now enforces the module while
  `src/sensory/organs/news_organ.py` moves into the rotation next.
- 2025-09-21 – Stage 4 advanced with `src/sensory/organs/news_organ.py`
  confirmed clean under `ruff format`. No manual edits were required, pytest
  stayed green, and the allowlist now enforces the organ while
  `src/sensory/organs/orderbook_organ.py` lines up next in the rotation.
- 2025-09-21 – Stage 4 progressed with `src/sensory/organs/orderbook_organ.py`
  confirmed clean under `ruff format`. No manual edits were necessary, pytest
  stayed green, and the allowlist now enforces the organ while
  `src/sensory/organs/price_organ.py` lines up next in the rotation.
- 2025-09-21 – Stage 4 expanded to cover the entire `src/sensory/` tree (organs,
  services, vendor shims, tests, and supporting packages). Collapsed the
  allowlist to a single `src/sensory/` entry after verifying `ruff format`
  produced no behavioral edits; only quote normalization in
  `src/sensory/__init__.py`, trailing blank removal in
  `src/sensory/anomaly/__init__.py`, and assertion wrapping in
  `src/sensory/tests/test_integration.py` appeared, and pytest stayed green
  before recording the slice across roadmap, CI, and debt snapshots.
  `src/data_foundation/config/` now queues next for Stage 4.
- 2025-09-22 – Stage 4 continued with `src/data_foundation/config/` verified
  clean under `ruff format`. No edits were required, pytest stayed green, and the
  allowlist now enforces the package while `src/data_foundation/ingest/` lines up
  next alongside other high-churn data foundation modules.
- 2025-09-22 – Stage 4 verified `src/data_foundation/ingest/` and
  `src/data_foundation/persist/` under `ruff format`. The formatter only reflowed
  docstrings and import blocks; no manual edits were required. Expanded the
  allowlist to cover both packages and re-ran
  `pytest tests/current/test_data_integration_smoke.py -q` to confirm behavior
  remained stable ahead of sequencing the remaining data foundation modules.
- 2025-09-23 – Stage 4 normalized `src/data_foundation/replay/` and
  `src/data_foundation/schemas.py`. Ruff reflowed a few signatures and docstrings;
  targeted replay smoke checks and `pytest tests/current/test_risk_manager_impl.py`
  stayed green. Published `docs/status/formatter_stage4_briefing.md` so the
  operational/performance slices have owners, reviewer rotations, and freeze
  windows ahead of formatting.

## Remaining Stage 4 sequencing (updated 2025-09-23)

| Order | Target | Owner | Notes |
| --- | --- | --- | --- |
| 1 | `src/data_integration/dukascopy_ingestor.py`, `src/data_integration/persist/` | Market Data | Align with ingestion owners to avoid conflicts during feed updates. |
| 2 | `src/operational/metrics.py`, `src/operational/metrics_registry.py` | Platform | Follow the freeze window and reviewer rota captured in `docs/status/formatter_stage4_briefing.md`; rerun `tests/current/test_operational_metrics_*`. |
| 3 | `src/performance/vectorized_indicators.py`, `src/performance/__init__.py` | Performance | Use the Stage 4 briefing checklist for reviewer assignments and validation. |
| 4 | `src/operational/state_store.py`, `src/operational/event_bus.py` | Platform | Touches async primitives; schedule with orchestrator regression coverage to keep flake noise low. |
| 5 | `scripts/check_formatter_allowlist.py` and supporting `scripts/analysis/` helpers | Tooling | Format once Stage 4 directories stabilize so CI helpers match enforced style. |
