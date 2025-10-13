# Contributing to AlphaTrade

This guide adds the operational context that complements `docs/development/contributing.md`. Keep that typing and workflow reference handy for day-to-day work; this file focuses on how to exercise the runtime safely and how to reason about the current feature flags.

## Run Profiles

Three run profiles map to the roadmap acceptance gates. The environment variables below are read by `SystemConfig.from_env()`; you can also provide the same values via a YAML payload and load it with `SystemConfig.from_yaml`.

### Simulation (`sim`)
- Purpose: deterministic replay for fast iteration, unit and regression tests, or CI smoke runs. No external services required.
- Prerequisites: optional local DuckDB file (`data/tier0.duckdb`) seeded via recorded fixtures. No Redis, Timescale, or Kafka needed.
- Recommended command:
  ```sh
  make run-sim
  ```
  The target wires the bootstrap runtime with ingest disabled, trading disabled, and a capped tick budget so the deterministic loop completes quickly while exporting a decision diary and `artifacts/sim/summary.json` evidence bundle.
- Notes:
  - Override defaults with `SIM_TIMEOUT=<seconds>`, `SIM_MAX_TICKS=<n>`, `SIM_SYMBOLS=EURUSD,GBPUSD`, or `SIM_EXTRA_ARGS="--enable-trading"` when you need different postures.
  - The wrapper writes the diary to `artifacts/diaries/sim.jsonl`; point to another path with `SIM_DIARY=<path>`.
  - Add `FAST_WEIGHT_EXCITATORY_ONLY=true` or other `FAST_WEIGHT_*` extras when you need to exercise sparsity guardrails (see `docs/context/examples/understanding_router.md`).
  - The CLI summary (`python -m src.runtime.cli summary --json --no-trading`) is a fast pre-flight check for CI jobs that only assert config sanity.

### Paper Trading (`paper`)
- Purpose: long-running paper sessions with guardian telemetry to satisfy roadmap acceptance criteria and governance evidence.
- Prerequisites:
  - Start the dev data backbone: `docker compose up -d redis postgres kafka` (or run the provided `docker/dev` profile).
  - Load institutional extras via `config/system/dev_data_backbone.yaml` or environment variables (Timescale, Redis, Kafka endpoints).
- Recommended command:
  ```sh
  export RUN_MODE=paper
  export EMP_ENVIRONMENT=demo
  export EMP_TIER=tier_0
  export DATA_BACKBONE_MODE=institutional
  export DECISION_DIARY_PATH=artifacts/diaries/paper.jsonl
  export PERFORMANCE_METRICS_PATH=artifacts/perf/paper.json
  mkdir -p artifacts/diaries artifacts/perf
  make run-paper RUN_ARGS="--duration-hours 0.5 --progress-interval 60 --report-path artifacts/perf/paper_guardian.json"
  ```
- Notes:
  - `make run-paper` wraps `python -m src.runtime.cli paper-run`; pass additional `RUN_ARGS` to tighten guardian thresholds (for example `--latency-p99-max 0.25`).
  - The guardian exits with `0` (pass), `1` (degraded), or `2` (failed). Archive the JSON summary it writes if you need governance evidence.
  - Keep feature flags conservative: leave `fast_weights_live` disabled unless the sprint alignment brief explicitly calls for the experiment.

### Live Shadow (`live-shadow`)
- Purpose: mirror live operations without touching capital. Requires the institutional data backbone and ingest scheduler.
- Prerequisites:
  1. Run the operational backbone starter: `python tools/data_ingest/run_live_shadow.py --config config/system/dev_data_backbone.yaml --duration 600 --format markdown --output artifacts/live_shadow/ingest.md`.
  2. Ensure Timescale/Redis/Kafka containers are healthy (`docker compose ps`).
  3. Confirm the ingest script reports green probes before starting the runtime loop.
- Recommended command sequence:
  ```sh
  export RUN_MODE=paper
  export EMP_ENVIRONMENT=staging
  export EMP_TIER=tier_1
  export DATA_BACKBONE_MODE=institutional
  export DECISION_DIARY_PATH=artifacts/diaries/live_shadow.jsonl
  export PERFORMANCE_METRICS_PATH=artifacts/perf/live_shadow.json
  export FAST_WEIGHT_MAX_ACTIVE_FRACTION=0.2
  mkdir -p artifacts/diaries artifacts/perf artifacts/live_shadow
  python main.py --symbols EURUSD,GBPUSD --db data/tier0.duckdb
  ```
- Notes:
  - The live-shadow profile keeps `confirm_live` implicit (`False`). Only flip `CONFIRM_LIVE=true` after governance sign-off.
  - Keep the feature flags in their conservative posture (`fast_weights_live=false`, `EVOLUTION_ENABLE_ADAPTIVE_RUNS` unset) unless a controlled experiment is approved.
  - Capture the ingest summary, guardian report, and decision diary in the run directory (for example `artifacts/live_shadow/2025-10-12/`).

## Feature Flags and Runtime Extras

Use the table below to track the high-signal flags that gate roadmap functionality. Unless stated otherwise, flags are read from environment variables and mirrored into `SystemConfig.extras`.

| Flag | Default | How to enable | Scope | Notes |
| --- | --- | --- | --- | --- |
| `fast_weights_live` | Off for bootstrap tiers | Add `feature_flag: fast_weights_live` with `default_fast_weights_enabled: true` in `UnderstandingRouterConfig`, or supply `feature_flags={"fast_weights_live": true}` in belief snapshots | Understanding loop (`src/understanding/router.py`, `src/orchestration/alpha_trade_loop.py`) | Gate fast-weight adapters. Keep disabled until governance approves experimentation; see the config examples in `docs/context/examples/understanding_router.md`. |
| `EVOLUTION_ENABLE_ADAPTIVE_RUNS` | Unset (treated as false) | `export EVOLUTION_ENABLE_ADAPTIVE_RUNS=1` | Evolution orchestration (`src/evolution/feature_flags.py`, `src/thinking/adaptation/evolution_manager.py`) | Enables adaptive evolution cycles. Leave off in sim/paper runs unless you are exercising the evolution readiness playbooks. |
| `POLICY_LEDGER_REQUIRE_DIARY` | `true` | `export POLICY_LEDGER_REQUIRE_DIARY=false` when running migration scripts | Policy ledger (`src/governance/policy_ledger.py`) | Controls whether promotions require matching decision diary evidence. Only disable for backfills with a clear audit trail. |
| `PAPER_TRADE_GA_MA_CROSSOVER` | Unset (promotion skipped) | `export PAPER_TRADE_GA_MA_CROSSOVER=paper` or `=live` | GA promotion tooling (`src/evolution/experiments/promotion.py`) | Dictates the target status for GA champions. Use `paper` to stage paper-only tactics, `live` only after governance review. |
| `FAST_WEIGHT_*` extras | Not set | Export values such as `FAST_WEIGHT_MAX_ACTIVE_FRACTION=0.2`, `FAST_WEIGHT_ACTIVATION_THRESHOLD=1.1`, `FAST_WEIGHT_EXCITATORY_ONLY=true` | Fast-weight controller (`src/thinking/adaptation/fast_weights.py`) | Tune sparsity and stability constraints. Recommended during live-shadow rehearsals to keep activations sparse. |
| `FINAL_DRY_RUN_*` extras | Not set | Set by `tools/operations/final_dry_run_orchestrator.py` | Final dry-run harness (`src/runtime/final_dry_run_support.py`) | When present, these paths are mirrored onto `DECISION_DIARY_PATH` and `PERFORMANCE_METRICS_PATH` automatically. |

### Reserved Flags
- Linear attention router and exploration toggles are not wired yet. Do not invent ad-hoc flags; coordinate with the roadmap owner before introducing them.
- When new flags ship, extend this table and link the owning module plus guardrail tests.

## Evidence Checklist

Regardless of the run profile, capture the following artefacts for governance:
- Decision diary JSONL (`DECISION_DIARY_PATH`).
- Performance metrics snapshot (`PERFORMANCE_METRICS_PATH`).
- Guardian or ingest summaries (paper, live-shadow).
- Feature flag posture recorded alongside the run (print `SystemConfig.extras`).

Keeping these artefacts consistent prevents drift between the roadmap checklist and the implementation.
