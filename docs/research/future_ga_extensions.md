# Future Genetic Algorithm Extensions

## Purpose
This document enumerates the advanced genetic algorithm (GA) capabilities that
extend the current evolution engine toward the encyclopedia's Tier‑2/Tier‑3
vision. Each extension describes prerequisites, owner responsibilities, and exit
criteria so that the roadmap can promote items into engineering tickets without
additional discovery cycles.

## Extension backlog

### 1. Speciation & diversity preservation
- **Objective:** Prevent premature convergence and maintain heterogeneous
  strategy behaviour across market regimes.
- **Prerequisites:**
  - Catalogued fitness telemetry stored via
    `src/evolution/lineage_telemetry.py`.
  - Strategy metadata enriched with factor exposures and risk signatures.
  - Population replay harness in `tests/evolution` to compare species health.
- **Implementation sketch:**
  - Introduce a `SpeciesClassifier` that clusters genomes using behavioural
    similarity (Sharpe, turnover, exposure vectors).
  - Extend `PopulationManager` with niche counts and stagnation thresholds.
  - Persist species metrics to the evolution lineage journal for paper/live
    audit.
- **Validation:**
  - Unit tests covering clustering and reproduction quotas.
  - Integration backtest showing improved diversity vs. baseline GA.
  - Telemetry surface `telemetry.evolution.species` summarising population share
    and stagnation scores.

### 2. Multi-objective optimisation (Pareto fronts)
- **Objective:** Optimise strategies against risk-adjusted return, drawdown, and
  execution impact simultaneously.
- **Prerequisites:**
  - Extended fitness schema capturing risk ratios and execution penalties.
  - Pareto dominance utilities within `src/evolution/fitness.py`.
  - Dashboard tiles in `docs/reports/evolution/` for Pareto frontier snapshots.
- **Implementation sketch:**
  - Implement `ParetoFrontier` helper with incremental update capability.
  - Adapt selection operators to rank individuals by Pareto layer and crowding
    distance.
  - Surface Pareto set to the strategy promotion workflow so humans can choose
    along the risk/return frontier.
- **Validation:**
  - Deterministic test harness verifying Pareto sorting and crowding metrics.
  - Integration scenario demonstrating non-dominated set persists between
    generations.

### 3. Live evolution with streaming feedback
- **Objective:** Enable continuous adaptation using streaming telemetry and
  guard rails suitable for paper/live pilots.
- **Prerequisites:**
  - Stable ingestion telemetry (`telemetry.ingest.*`) already in place.
  - Risk guard rails encoded in `src/risk/risk_manager_impl.py`.
  - Back-pressure capable task orchestrator for asynchronous evaluation.
- **Implementation sketch:**
  - Introduce evolution scheduler consuming live PnL, drawdown, and latency
    metrics from Kafka topics.
  - Gate promotions behind compliance workflows before activating updated
    strategies in live experiments.
  - Persist live-evolution audit trail to
    `artifacts/evolution/live_runs/<timestamp>/` with manifest metadata.
- **Validation:**
  - Dry-run script exercising end-to-end scheduler with mocked telemetry.
  - Canary paper trading exercise showing safe rollback on guard-rail breach.

## Delivery governance
- Assign each extension to a dedicated epic with acceptance tests listed in the
  story backlog.
- Review extension status during the monthly research council meeting and update
  `docs/research/research_debt_register.md` with open questions.
- Promote extensions into implementation only when telemetry, risk, and
  compliance dependencies have landed in the professional runtime.
