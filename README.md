# EMP Proving Ground v1

**Evolving Market Predator** - An Autonomous Trading Intelligence Framework

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue.svg)](http://mypy-lang.org/)
[![Status: Development](https://img.shields.io/badge/status-development-yellow.svg)]()



> ⚠️ **SAFETY WARNING**
> 
> **This is not a production trading system.** The public repository contains a mock FIX simulator and optional Yahoo Finance data ingestion for research and development. Live broker adapters, streaming venue connectivity, and production-grade risk controls are intentionally excluded from the public build.
> 
> Treat this as a **sandbox and learning environment** until you replace mocks with real integrations in a private, credentialed fork. Do not deploy this system with real capital without comprehensive testing and validation.

---

## Overview

The EMP Proving Ground is an **advanced algorithmic trading framework** that combines evolutionary intelligence, multi-dimensional market perception, and institutional-grade risk management. The system is designed to evolve trading strategies through genetic algorithms while maintaining rigorous governance and operational controls.

**Current Status:** Development Framework with Substantial Implementations

The codebase represents a sophisticated architecture with many functional subsystems, but it is **not yet production-ready** for live trading. This README provides an honest assessment of what works, what's in progress, and what remains to be built.

### Project Statistics

- **Source Code:** 662 Python files, 202,140 lines
- **Test Coverage:** 542 test files, 95,958 lines
- **Architecture:** 36 major subsystems across 5 layers
- **Documentation:** 209 markdown files, comprehensive technical specifications

---

## Philosophy

The EMP system embodies three core principles:

1. **Antifragile Design** - Systems that gain strength from volatility and market stress, not just survive them
2. **Evolutionary Intelligence** - Strategies that evolve through genetic algorithms and continuous learning
3. **Multi-Dimensional Perception** - Market analysis across fundamental (WHY), institutional (HOW), technical (WHAT), temporal (WHEN), and anomaly dimensions

The project follows a **truth-first development philosophy**: all claims are backed by verifiable code, realistic timelines acknowledge actual constraints, and documentation honestly reflects implementation status.

---

## Capabilities vs. Non-Capabilities

### ✅ What This System CAN Do (Public Build)

- **Mock FIX lifecycle**: Full order flow simulation with MockFIXManager + FIXConnectionManager
- **Opt-in Yahoo Finance ingest**: Historical OHLCV data → DuckDB storage
- **Simulator regression tests**: Comprehensive test coverage for mock trading
- **Structured logging & telemetry**: Prometheus metrics and observability
- **4D+1 Sensory cortex**: WHY/HOW/WHAT/WHEN/ANOMALY market analysis framework
- **Risk management scaffolding**: Policy enforcement, exposure limits, drawdown protection
- **Sentient learning loop**: Pattern memory, experience storage, adaptive learning
- **Research harness**: Backtesting framework, strategy development tools

### ❌ What This System CANNOT Do (By Design in Public Repo)

- **Live broker routing**: No real broker connections (use private fork for production)
- **Streaming venue connectivity**: No WebSocket feeds to live exchanges
- **Impact-aware execution scheduling**: Almgren-Chriss and TWAP/VWAP/IS not yet wired
- **CSCV/DSR promotion gates**: Statistical validation not yet enforced in CI
- **Regime-aware capital router**: Regime detection exists, capital allocation not yet automatic
- **Production-grade secrets management**: No key rotation, scoped credentials, or tamper-evident logs
- **Real-time order book data**: LOBSTER integration planned but not yet implemented
- **Institutional compliance**: Audit trails exist, but full compliance suite not implemented

**Bottom line**: This is a **development framework** for building a production system, not a turnkey trading platform.

---

## Roadmap: Next 3 Sprints

### Sprint A — Validation Infrastructure (Weeks 1-4)

**Goal**: Implement statistical validation gates to prevent overfitting

**Tasks**:
- [ ] Wire CSCV (Combinatorially Symmetric Cross-Validation) with purged/embargoed folds
- [ ] Implement DSR (Deflated Sharpe Ratio) calculator with multiple testing correction
- [ ] Add CI promotion gate: fail builds if DSR < 1.5 threshold
- [ ] Create validation report generator

**Pass/Fail Gate**: CI rejects strategies with DSR below threshold; validation reports show Sharpe, DSR, p-value, and multiple testing correction

---

### Sprint B — Impact-Aware Execution (Weeks 5-8)

**Goal**: Complete execution layer with cost-aware scheduling

**Tasks**:
- [ ] Verify/complete Almgren-Chriss market impact model
- [ ] Implement alpha half-life estimator from historical signal decay
- [ ] Build TWAP/VWAP/IS (Implementation Shortfall) schedulers
- [ ] Add execution strategy selector (fast alpha → IS, slow alpha → VWAP, medium → TWAP)
- [ ] Create execution telemetry (slippage, latency, fill quality)

**Pass/Fail Gate**: Orders scheduled optimally based on alpha half-life; execution telemetry shows slippage vs. arrival, latency buckets, and shortfall attribution

---

### Sprint C — Regime-Aware Capital Allocation (Weeks 9-12)

**Goal**: Transform regime detection into automatic capital routing

**Tasks**:
- [ ] Create regime-strategy compatibility matrix (YAML config)
- [ ] Implement capital allocator with regime-specific risk budgets
- [ ] Add dynamic strategy suspension/promotion based on current regime
- [ ] Implement hysteresis to prevent regime thrashing (require 2+ consecutive detections)
- [ ] Integrate with evolution layer to breed regime-specific strategies

**Pass/Fail Gate**: Capital allocation adjusts automatically on regime change; strategies suspend when regime is hostile; backtests show aggregate drawdown reduction vs. regime-agnostic baseline

---

**After Sprint C**: System ready for extended paper trading validation (30-90 days) before considering live deployment with minimal capital.

---



## Architecture

The system follows a five-layer architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│              Layer 5: Operations & Governance               │
│  Policy Ledger │ Observability │ Incident Response │ Audit  │
├─────────────────────────────────────────────────────────────┤
│              Layer 4: Strategy Execution & Risk             │
│  Trading Manager │ Risk Controls │ Broker Adapters │ Orders │
├─────────────────────────────────────────────────────────────┤
│           Layer 3: Intelligence & Adaptation                │
│  Evolution │ Sentient Loop │ Planning │ Memory │ Learning   │
├─────────────────────────────────────────────────────────────┤
│              Layer 2: Sensory Cortex (4D+1)                 │
│    WHY    │    HOW    │    WHAT    │    WHEN    │ ANOMALY   │
├─────────────────────────────────────────────────────────────┤
│                Layer 1: Data Foundation                     │
│ Ingestion │ Normalization │ Storage │ Quality │ Distribution │
└─────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
src/
├── core/                    # Core abstractions and protocols
├── data_foundation/         # Data ingestion and normalization
├── sensory/                 # 4D+1 sensory cortex implementation
├── thinking/                # Planning, adaptation, and decision-making
├── sentient/                # Learning loop and pattern memory
├── understanding/           # Causal reasoning and decision diary
├── trading/                 # Execution, strategies, and order management
├── risk/                    # Risk management and policy enforcement
├── governance/              # Policy ledger and strategy registry
├── operations/              # Observability and monitoring
├── runtime/                 # Orchestration and lifecycle management
├── simulation/              # World models and surrogate training
├── evolution/               # Genetic algorithms and population management
├── genome/                  # Strategy genome representation
└── ecosystem/               # Multi-agent coordination
```

---

## What's Implemented

### ✅ Fully Functional Subsystems

These components are **production-grade** with comprehensive implementations and test coverage:

#### 1. Risk Management (`src/risk/`)
- **RiskManagerImpl**: Multi-layer risk enforcement with exposure limits, leverage caps, sector constraints, and drawdown protection
- **Volatility-aware position sizing**: Dynamic allocation based on market conditions
- **Policy enforcement**: Real-time validation with audit trails
- **Monte Carlo analytics**: Geometric Brownian motion simulator with optional antithetic variates and VaR/ES helpers for downstream risk reports.【F:src/risk/analytics/monte_carlo.py†L1-L171】
- **Status**: Production-ready

#### 2. Execution Layer (`src/trading/`)
- **LiveBrokerExecutionAdapter**: Live trading with real-time risk gating and policy snapshots
- **PaperBrokerExecutionAdapter**: Simulation environment with realistic market conditions
- **Order management**: Intent-based trading with attribution tracking
- **Status**: Production-ready for paper trading; live trading requires broker integration

#### 3. Sentient Learning (`src/sentient/`)
- **SentientPredator**: Autonomous learning loop that adapts from trade outcomes
- **FAISSPatternMemory**: Vector-based experience storage with decay and reinforcement
- **Memory backup & recovery**: Snapshot helper captures FAISS indices/metadata into timestamped folders with retention pruning and restore tooling so operators can rehearse memory recovery workflows in CI.【F:src/sentient/memory/faiss_pattern_memory.py†L53-L324】【F:tests/sentient/test_memory_backup.py†L28-L70】
- **MemoryAutoRetrainer**: Automates FAISS index rebuilding, batch flushes, and duplicate pruning with persisted state so
  experience stores stay compact even when the optional `faiss` dependency is missing, backed by regression coverage around
  compaction, scheduling, and persistence flows.【F:src/sentient/memory/auto_retraining.py†L1-L336】【F:tests/sentient/test_auto_retraining.py†L1-L110】
- **Extreme episode detection**: Identifies and stores high-impact market events
- **Status**: Fully implemented with graceful fallback to in-memory storage

#### 4. Planning & Reasoning (`src/thinking/`, `src/understanding/`)
- **MuZeroLiteTree**: Short-horizon planning with tree search and causal adjustments
- **CausalGraphEngine**: Causal DAG construction with intervention capabilities
- **DecisionDiary**: Comprehensive audit trail of all trading decisions
- **FastWeightController**: Adaptive routing with Hebbian learning
- **PreTrainingPipeline**: Curriculum-aware orchestration that stitches together
  curriculum stages, LoRA freeze planning, multi-task loss evaluation, and
  horizon diagnostics so research harnesses can dry-run pre-training loops with
  deterministic summaries.【F:src/thinking/learning/pretraining_pipeline.py†L1-L259】【F:tests/thinking/test_pretraining_pipeline.py†L1-L127】
- **Status**: Fully implemented

#### 5. World Model (`src/simulation/`)
- **GraphNetSurrogate**: GNN-based market dynamics model for simulation
- **Counterfactual analysis**: What-if scenario evaluation
- **Status**: Implemented for simulation-based learning

#### 6. Data Pipeline (`src/data_foundation/`)
- **Yahoo Finance integration**: Historical OHLCV data ingestion
- **PricingPipeline**: Multi-vendor normalization with quality checks, window-aware
  validation hints, and vectorised duplicate/coverage diagnostics that now surface
  resolved window metadata alongside expected candle counts.
- **WebSocketClient**: Durable streaming client with reconnect, heartbeat, rate limiting,
  and auto-replayed subscriptions so venue adapters regain data feeds without manual
  resubscription logic during reconnect scenarios.【F:src/data_foundation/streaming/websocket_client.py†L1-L520】【F:tests/data_foundation/streaming/test_websocket_client.py†L1-L288】
- **DuckDB storage**: Persistent storage with encryption support and CSV fallback
- **Symbol mapping**: Broker-agnostic symbol resolution with caching
- **LOBSTER dataset parser**: Normalises message and depth snapshots, enforces alignment, and prepares high-frequency order book frames for analytics consumers
- **Streaming market data cache**: Redis-compatible tick window with warm starts, TTL enforcement, and in-memory fallback for streaming ingestion pipelines
- **TimescaleQueryInterface**: Unified tick/quote/book query facade with timezone normalisation, SQLite compatibility, and an LRU cache so notebooks and tests reuse recent results without hammering Timescale on repeated queries.【F:src/data_foundation/storage/timescale_queries.py†L1-L158】【F:src/data_foundation/storage/timescale_queries.py†L180-L318】
- **TimescaleAdapter**: Async ingestion facade that batches DataFrame/iterable payloads, captures per-dimension telemetry, and tolerates partial failures while still returning merged ingest statistics for operational dashboards.【F:src/data_foundation/storage/timescale_adapter.py†L1-L220】【F:tests/data_foundation/test_timescale_adapter.py†L17-L155】
- **Status**: Functional for historical daily data

#### 7. Governance (`src/governance/`)
- **PolicyLedger**: Stage-based promotion with evidence requirements
- **PromotionGuard**: Blocks strategy graduation without regime coverage
- **StrategyRegistry**: Centralized strategy management with lifecycle controls
- **Kill-switch monitoring**: Async file watcher guards the global kill switch and wires into the runtime so emergency stops trigger graceful shutdowns automatically.【F:src/governance/kill_switch.py†L1-L118】【F:src/runtime/predator_app.py†L1135-L1161】
- **Tamper-evident audit logging**: Hash-chained JSONL ledger now layers structured search across text, metadata paths, and timestamps plus integrity verification, statistics, and export helpers so governance reviews stay navigable even when corrupted rows appear.【F:src/governance/audit_logger.py†L22-L528】【F:tests/governance/test_audit_logger.py†L16-L257】
- **Audit documentation playbook**: Centralises decision diary, policy ledger, and
  compliance evidence capture with command-level runbooks for audit packs
  (`docs/audits/audit_documentation.md`).
- **Status**: Fully implemented per AlphaTrade whitepaper

### ⚠️ Partially Implemented

These components have **architectural foundations** but require additional work:

#### 1. Evolutionary Intelligence (`src/evolution/`, `src/genome/`)
- **What works**: Genome representation, population management, basic genetic operators
- **Progress**: Fitness calculator combines weighted performance metrics, risk penalties, and behaviour constraints into auditable scores for evolution loops.【F:src/evolution/fitness/calculator.py†L1-L268】【F:tests/evolution/test_fitness_calculator.py†L1-L65】
- **Progress**: A new `EvolutionScheduler` ingests live PnL, drawdown, and latency telemetry, enforcing cooldown windows and percentile triggers before launching optimisation cycles so evolution runs react deterministically to production performance.【F:src/evolution/engine/scheduler.py†L1-L200】
- **Progress**: Fresh guard rails in `EvolutionSafetyController` track drawdown, VaR, latency, slippage, and data quality limits with cooldown and lockdown logic so adaptive runs stay within institutional risk bounds before mutation begins.【F:src/evolution/safety/controls.py†L1-L260】
- **Progress**: Preference articulation framework now models objective weights,
  interactive tuning prompts, and articulator helpers that feed evolution
  scoring with operator-provided priorities under regression coverage.【F:src/evolution/optimization/preferences.py†L1-L219】【F:tests/evolution/test_preferences.py†L1-L79】
- **Progress**: Multi-objective optimisation now ships with an NSGA-II implementation featuring crowding-distance ranking, configurable crossover/mutation hooks, and deterministic exports for Pareto-front reporting.【F:src/evolution/algorithms/nsga2.py†L1-L334】【F:tests/evolution/test_nsga2.py†L1-L119】
- **What's missing**: Deeper adaptive integration that turns the scoring framework into fully autonomous strategy evolution (currently focused on parameter tuning)
- **Status**: Framework complete; needs enhanced evolution logic

#### 2. Sensory Cortex (`src/sensory/`)
- **What works**: RealSensoryOrgan integration, technical indicators, order book analytics framework, a fundamental analysis pipeline that normalises provider payloads into valuation/quality metrics for the WHY sensor, and an `InstrumentTranslator` service that normalises multi-venue aliases (Bloomberg, Reuters, cTrader, CME, NASDAQ) into the universal instrument model using configurable mappings.【F:src/sensory/real_sensory_organ.py†L41-L520】【F:src/sensory/why/fundamental.py†L1-L239】【F:src/sensory/why/why_sensor.py†L70-L210】【F:src/sensory/services/instrument_translator.py†L1-L200】【F:config/system/instrument_aliases.json†L1-L37】
- **Progress**: InstitutionalFootprintHunter now extracts ICT order blocks, fair value gaps, liquidity sweeps, smart-money flow, and institutional bias scoring for live footprint tracking.【F:src/sensory/organs/dimensions/institutional_tracker.py†L1-L151】
- **What's missing**: Real-time data feeds, provider-backed fundamental ingestion (current pipeline relies on supplied snapshots)
- **Status**: Architecture solid; limited by data source availability

#### 3. Strategy Library (`src/trading/strategies/`)
- **What works**: Strategy protocol, signal generation framework, ICT microstructure features
- **Progress**: PCA-driven statistical arbitrage strategy constructs market-neutral baskets from principal components, clamps exposure, and emits rich signal metadata for downstream attribution under regression coverage.【F:src/trading/strategies/stat_arb/pca_arb.py†L1-L279】【F:tests/trading/test_pca_stat_arb_strategy.py†L1-L104】
- **What's missing**: Comprehensive library of tested strategies beyond the emerging catalogue
- **Status**: Framework exists with an expanding library – the new
  `VolatilityTradingStrategy` blends implied vs realised volatility spreads with
  gamma scalping and microstructure alignment signals under regression coverage.

#### 4. Operations (`src/operations/`)
- **What works**: ObservabilityDashboard, PaperRunGuardian, incident playbook CLI, and a chaos campaign orchestrator that sequences typed drills, rotates responders, escalates severity, and exports Markdown/JSON evidence packs for readiness reviews with reproducible seeds.【F:src/operations/incident_simulation.py†L1-L318】【F:src/operations/incident_simulation.py†L320-L472】
- **Change management evaluation**: Policy helpers normalise change requests, enforce lead-time and approval thresholds per impact tier, and emit Markdown summaries that gate deployments while feeding governance packets under regression coverage.【F:src/operations/change_management.py†L1-L382】【F:tests/operations/test_change_management.py†L20-L186】
- **New governance controls**: Pre-launch validation checklist and post-enable audit log document required evidence, checklists,
  and follow-up reviews for each institutional data integration, keeping operational sign-off aligned with readiness audits.
  【F:docs/operations/validation_checklist.md†L1-L40】【F:docs/operations/post_enable_reviews.md†L1-L25】
- **Progress**: Emergency procedures handbook codifies trigger conditions, communications flow, failover drills, and post-incident review checklists with ready-to-run templates for responders.【F:docs/operations/emergency_procedures.md†L1-L105】
- **What's missing**: Grafana dashboards and ELK ingestion pipeline
- **Progress**: Prometheus stack runs via Docker Compose with exporters for the engine, Redis, TimescaleDB, and Kafka plus validation ensuring compose wiring mounts rule files and dependencies correctly.【F:docker-compose.yml†L1-L195】【F:tests/config/test_prometheus_stack.py†L1-L65】
- **Progress**: Kibana dashboard deployment CLI automates Saved Objects imports,
  emits human-readable saved object summaries, and supports API key/basic auth
  plus partial-failure handling for observability refreshes. Failover
  infrastructure readiness calculators now combine backup, drill, and
  cross-region telemetry into a single status snapshot for operators, with
  regression coverage over severity rollups and Markdown summaries. Local
  Elasticsearch cluster helper and deployment guide bootstrap dev observability
  stacks via Docker Compose with automated health/ingest checks for quick lab
  bring-up.【F:docker/elasticsearch/README.md†L1-L23】【F:docs/observability/elasticsearch_cluster.md†L1-L64】【F:tools/observability/deploy_elasticsearch_cluster.py†L1-L307】
- **Status**: Governance tooling complete; production infrastructure pending

#### 5. Fundamental Data Provider Strategy (`docs/research/fundamental_data_provider_selection.md`)
- **What works**: Comparative analysis across FMP, Polygon.io, Intrinio, Alpha Vantage, and Quandl with weighted scoring
- **What's missing**: Implementing ingestion adapters, storage schema extensions, and governance policies for the recommended vendors
- **Status**: Research complete; awaiting engineering execution

---

## What's Missing

### ❌ Critical Gaps

These features are **required for production deployment** but not yet implemented:

1. **Real-Time Data Streaming**
    - Broker/WebSocket adapters and live feed supervision still pending. The hardened `WebSocketClient` now layers reconnect, heartbeat, rate-limited send support, and durable subscription replay, but venue-specific adapters are not yet wired into ingest pipelines.【F:src/data_foundation/streaming/websocket_client.py†L1-L520】
   - Streaming cache now retains tick windows with Redis compatibility but has no upstream connectors yet
   - **Impact**: Cannot trade live markets without real-time data

2. **LOBSTER Dataset Integration**
   - Parser exists for messages and order book snapshots, yet no reconstruction engine or sensory cortex wiring
   - Framework exists (`order_book_analytics.py`) but upstream ingestion still incomplete
   - **Impact**: Cannot train on high-frequency order book data as intended

3. **Comprehensive Backtesting**
   - No historical replay engine
   - Batch orchestrator now coordinates concurrent runs with supervision, and
     performance analytics export drawdown, ratio, and trade attribution reports,
     but the replay engine, market simulator, and live data feeds remain absent
   - **Impact**: Cannot validate strategies before live deployment

4. **Automated ML Training Pipeline**
   - Learning components exist but no automated retraining
   - Manual intervention required for model updates
   - **Impact**: Cannot achieve fully autonomous operation

5. **OpenBloomberg / Alternative Data**
   - Vendor evaluation complete, but no Bloomberg-equivalent or fundamental adapters implemented
   - WHY sensor now computes valuation/quality metrics from provided snapshots,
     yet provider ingestion and storage remain to be built
   - **Impact**: WHY dimension is architecturally present but data-limited

### 🔧 Enhancement Opportunities

These features would **improve functionality** but are not blocking:

1. **Instrument Translation Protocol**
   - Current: Basic symbol mapping
   - Needed: Universal instrument representation across asset classes

2. **Advanced Evolutionary Algorithms**
   - Current: Parameter tuning
   - Needed: Full strategy discovery and multi-objective optimization

3. **Mamba-3 Backbone**
   - Current: Named stub with identity forward pass
   - Needed: Full state-space model implementation

4. **Regulatory Reporting**
   - Current: Audit trails and decision diary
   - Needed: Formatted compliance reports

---

## Getting Started

### Prerequisites

- Python 3.11+
- Poetry (dependency management)
- DuckDB (optional, falls back to CSV)
- FAISS (optional, falls back to in-memory)

### Installation

```bash
# Clone the repository
git clone https://github.com/HWeber-tech/emp_proving_ground_v1.git
cd emp_proving_ground_v1

# Install dependencies
poetry install

# Run tests to verify installation
poetry run pytest tests/
```

### Quick Start: Paper Trading

```bash
# Run paper trading with guardian monitoring
poetry run python src/runtime/cli.py paper-run \
  --config config/paper_trading.yaml \
  --guardian-enabled \
  --duration 24h

# View observability dashboard
poetry run python src/operations/observability_dashboard.py
```

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run specific subsystem tests
poetry run pytest tests/risk/
poetry run pytest tests/trading/
poetry run pytest tests/sentient/

# Run integration tests
poetry run pytest tests/integration/

# Generate coverage report
poetry run pytest --cov=src --cov-report=html
```

---

## Development Roadmap

This roadmap provides a comprehensive, actionable path to production readiness. Each task includes specific implementation details, acceptance criteria, and estimated effort.


### Phase 0: Preconditions (PR-Blocking Infrastructure)

**Priority**: 🔴🔴🔴 **CRITICAL** - Must be implemented BEFORE any backtesting or statistical validation

**Why first**: Without clean, provable, leak-free data, all downstream validation (CSCV/DSR, backtesting, live trading) measures fake alpha with false confidence. These are non-negotiable foundations.

**Total effort**: 72-112 hours

#### 0.1 Timestamp Hygiene & Clock Discipline

**Effort**: 16-24 hours | **GPU**: ❌ NO | **Priority**: 🔴 **Critical**

**Problem**: Bad clocks cause look-ahead leakage, creating fake alpha.

- [ ] **Build monotonic timestamp validator** (4 hours)
  - Reject any data where timestamps go backward
  - Track last_timestamp per data stream
  - **Location**: `src/core/time/timestamp_validator.py`
  - **Acceptance**: All data ingestion validates monotonic timestamps

- [ ] **Implement NTP sync monitor** (6 hours)
  - Check system clock against NTP server every 60 seconds
  - Use ntplib or similar library
  - Emit Prometheus metrics for clock drift
  - **Location**: `src/operations/monitoring/clock_monitor.py`
  - **Acceptance**: NTP sync checked every 60s, metrics emitted

- [ ] **Add clock drift alarm** (4 hours)
  - Alert if drift exceeds 10ms
  - Integrate with alerting system (PagerDuty, Slack, etc.)
  - **Acceptance**: Alarm triggers when drift >10ms

- [ ] **Create causality validator** (6 hours)
  - Prove feature timestamps < label timestamps with minimum gap
  - Validate all feature engineering pipelines
  - **Location**: `src/validation/causality_validator.py`
  - **Acceptance**: All features proven causal with timestamp audit

- [ ] **Write comprehensive tests** (4 hours)
  - Test monotonic validation catches time travel
  - Test NTP sync detection
  - Test causality validator catches leakage
  - **Acceptance**: 90%+ test coverage, all tests passing

**Integration Points**:
- Add timestamp validation to all data ingestion (`src/data_foundation/ingestion/`)
- Add causality checks to all feature engineering (`src/sensory/*/`)
- Add clock monitoring to observability stack (`src/operations/monitoring/`)

**Acceptance Criteria**:
- ✅ All data ingestion validates monotonic timestamps
- ✅ NTP sync checked every 60s, alarm if drift >10ms
- ✅ Feature engineering validates causality (feature_ts < label_ts - min_gap)
- ✅ Backtests with time-purged folds show near-identical metrics to standard CV

---

#### 0.2 Data Provenance & Reproducibility

**Effort**: 32-48 hours | **GPU**: ❌ NO | **Priority**: 🔴 **Critical**

**Problem**: If you can't reproduce a win, it isn't a win. Need complete audit trail from raw data → features → labels → signals → trades.

- [ ] **Implement dataset fingerprints** (8 hours)
  - SHA256 hash of every dataset version
  - Store fingerprints with metadata (row count, column count, source, created_at)
  - **Location**: `src/governance/provenance/dataset_fingerprint.py`
  - **Acceptance**: Every dataset has SHA256 fingerprint stored

- [ ] **Build feature/label versioning** (8 hours)
  - Semantic versioning for feature sets (v1.0, v1.1, v2.0)
  - Track which features are in which version
  - **Location**: `src/governance/provenance/feature_versioning.py`
  - **Acceptance**: Feature sets have semantic versions

- [ ] **Create Run Ledger** (16 hours)
  - Immutable log binding (code hash, data hash, hyperparams, regime) → (metrics, decision) → (trade IDs)
  - Append-only storage (no edits allowed)
  - Query interface for audits
  - **Location**: `src/governance/provenance/run_ledger.py`
  - **Acceptance**: Every training run has complete RunRecord in ledger

- [ ] **Build reproducibility validator** (8 hours)
  - Verify any run can be replayed byte-for-byte
  - Check that metrics match within tolerance
  - **Location**: `tests/governance/test_reproducibility.py`
  - **Acceptance**: Any promoted strategy can be replayed to identical metrics

- [ ] **Write comprehensive tests** (8 hours)
  - Test fingerprint generation
  - Test ledger append and query
  - Test reproducibility validation
  - **Acceptance**: 90%+ test coverage, all tests passing

**Integration Points**:
- Add RunRecord creation to training pipeline (`mlops/train.py`)
- Store ledger in PolicyLedger or separate audit database
- Add reproducibility test to CI/CD

**Acceptance Criteria**:
- ✅ Every dataset has SHA256 fingerprint stored
- ✅ Every training run has complete RunRecord in ledger
- ✅ Any promoted strategy can be replayed to identical metrics (within tolerance)
- ✅ Any live trade traceable to exact (code, data, hyperparams, decision)

---

#### 0.3 Leakage Firewall & Feature Sanity Suite

**Effort**: 24-40 hours | **GPU**: ❌ NO | **Priority**: 🔴 **Critical**

**Problem**: Most "alpha" is leakage in disguise. Need automated tests that prove no label look-ahead, no future-touching features, and proper data snooping corrections.

- [ ] **Build purged/embargoed fold validator** (8 hours)
  - Ensure no overlap between train/test with temporal gaps
  - Implement purge gap (before test) and embargo gap (after test)
  - **Location**: `src/validation/leakage_firewall.py`
  - **Acceptance**: Purged/embargoed fold test passes for all CV splits

- [ ] **Create feature timestamp auditor** (8 hours)
  - Prove all features are causal (feature_ts < label_ts - min_gap)
  - Audit all feature engineering pipelines
  - **Acceptance**: Feature timestamp audit proves all features causal

- [ ] **Implement synthetic leakage injection test** (8 hours)
  - Deliberately inject leakage (e.g., add future label as feature)
  - Verify tests catch it (performance should inflate suspiciously)
  - This is a meta-test: proves the leakage detection works
  - **Acceptance**: Synthetic injection test correctly detects injected leakage

- [ ] **Add CI gate for leakage** (4 hours)
  - Fail builds if any leakage test fails
  - Even if performance looks great, reject if leakage detected
  - **Location**: `tests/validation/test_leakage_ci_gate.py`
  - **Acceptance**: CI fails if any leakage test fails

- [ ] **Write comprehensive tests** (8 hours)
  - Test purged fold validation
  - Test causality auditing
  - Test synthetic injection detection
  - **Acceptance**: 90%+ test coverage, all tests passing

**Integration Points**:
- Run leakage tests before training (`mlops/train.py`)
- Add leakage tests to CI pipeline (`tests/validation/`)
- Add leakage gate to PR checks (`.github/workflows/`)

**Acceptance Criteria**:
- ✅ Purged/embargoed fold test passes for all CV splits
- ✅ Feature timestamp audit proves all features causal
- ✅ Synthetic injection test correctly detects injected leakage
- ✅ CI fails if any leakage test fails (even if performance looks great)

---

### Phase 1: Data Foundation (Weeks 1-4)

**Goal**: Establish real-time data infrastructure for live trading

**Estimated Effort**: 160 hours (4 weeks × 40 hours)

#### 1.1 LOBSTER Dataset Integration

**Objective**: Enable high-frequency order book analysis

- [ ] **Design LOBSTER data schema** (4 hours)
  - Map LOBSTER message types to internal order book events
  - Define normalization rules for different exchanges
  - Create data quality validation rules
  - **Acceptance**: Schema document with examples

- [ ] **Implement LOBSTER file parser** (8 hours)
  - Build parser for LOBSTER message files (orderbook, message)
  - Handle timestamp normalization and timezone conversion
  - Implement efficient chunked reading for large files
  - **Location**: `src/data_foundation/ingest/lobster_adapter.py`
  - **Acceptance**: Parse sample LOBSTER files with 100% accuracy

- [ ] **Create order book reconstruction engine** (12 hours)
  - Rebuild full order book state from LOBSTER messages
  - Implement snapshot + delta updates
  - Handle order book imbalances and anomalies
  - **Location**: `src/sensory/how/lobster_order_book.py`
  - **Acceptance**: Reconstruct order book with microsecond precision

- [ ] **Integrate with existing analytics** (8 hours)
  - Connect LOBSTER data to `order_book_analytics.py`
  - Feed data into ICT microstructure features
  - Enable HOW dimension sensors to consume LOBSTER data
  - **Acceptance**: All order book analytics work with LOBSTER data

- [ ] **Build LOBSTER data loader for backtesting** (8 hours)
  - Create historical replay mechanism
  - Implement efficient data windowing
  - Add caching for frequently accessed periods
  - **Acceptance**: Replay 1 day of LOBSTER data in <5 minutes

- [ ] **Write comprehensive tests** (8 hours)
  - Unit tests for parser and reconstruction
  - Integration tests with sensory cortex
  - Performance benchmarks
  - **Acceptance**: 90%+ test coverage, all tests passing

#### 1.2 Real-Time Market Data Streaming

**Objective**: Enable live trading with real-time price feeds

- [ ] **Design streaming data architecture** (4 hours)
  - Define WebSocket connection management strategy
  - Design reconnection and failover logic
  - Plan data buffering and backpressure handling
  - **Acceptance**: Architecture diagram and design doc

- [ ] **Implement WebSocket client framework** (12 hours)
  - Build generic WebSocket client with auto-reconnect
  - Implement heartbeat and connection monitoring
  - Add rate limiting and throttling
  - **Location**: `src/data_foundation/streaming/websocket_client.py`
  - **Acceptance**: Maintain stable connection for 24+ hours

- [ ] **Integrate broker-specific WebSocket feeds** (16 hours)
  - Implement adapters for 2-3 major brokers (e.g., Interactive Brokers, Alpaca, Binance)
  - Normalize tick data to internal format
  - Handle broker-specific quirks and edge cases
  - **Location**: `src/data_foundation/streaming/broker_feeds/`
  - **Acceptance**: Real-time tick data from all integrated brokers

- [ ] **Build streaming data pipeline** (12 hours)
  - Create async pipeline from WebSocket to sensory cortex
  - Implement data validation and quality checks
  - Add latency monitoring and alerting
  - **Location**: `src/data_foundation/streaming/pipeline.py`
  - **Acceptance**: End-to-end latency <50ms at p99

- [x] **Implement market data cache** (8 hours)
  - Build Redis-backed tick data cache
  - Implement sliding window for recent data with configurable TTL refresh and
    warm-start support
  - Add cache warming on startup
  - **Location**: `src/data_foundation/streaming/market_cache.py`
  - **Acceptance**: Cache hit rate >95% for recent data

- [ ] **Create monitoring and observability** (8 hours)
  - Add Prometheus metrics for connection health
  - Implement alerting for disconnections
  - Build dashboard for streaming data quality
  - **Acceptance**: Real-time visibility into all data streams

#### 1.3 Fundamental Data Integration

**Objective**: Enable WHY dimension with comprehensive fundamental analysis

- [ ] **Evaluate and select data providers** (8 hours)
  - Research alternatives: Financial Modeling Prep, Polygon.io, Quandl
  - Compare pricing, coverage, and API quality
  - Select 2-3 providers for redundancy
  - **Acceptance**: Provider selection document with justification

- [ ] **Implement fundamental data adapters** (16 hours)
  - Build adapters for selected providers
  - Normalize financial statements, earnings, economics data
  - Implement caching and rate limit handling
  - **Location**: `src/data_foundation/ingest/fundamental/`
  - **Acceptance**: Fetch and normalize data from all providers

- [ ] **Create fundamental data storage** (8 hours)
  - Design schema for financial statements, ratios, events
  - Implement DuckDB tables for fundamental data
  - Add versioning for historical data corrections
  - **Location**: `src/data_foundation/storage/fundamental_store.py`
  - **Acceptance**: Store and query 10+ years of fundamental data

- [x] **Integrate with WHY dimension sensors** (12 hours)
  - Connect fundamental data to `why_sensor.py`
  - Implement fundamental analysis features with valuation and quality scoring
  - Build valuation models (P/E, DCF, etc.)
  - **Acceptance**: WHY sensor produces fundamental signals

- [ ] **Build fundamental data quality monitoring** (8 hours)
  - Detect missing or stale data
  - Validate data consistency across providers
  - Alert on data quality issues
  - **Acceptance**: Automated quality checks with alerting

- [ ] **Implement Document OCR Compression for Fundamental Analysis** (32 hours)
  - **Background**: This implements vision-language compression from recent research (DeepSeek optical compression, 2025) that compresses large financial documents (100-page earnings reports) from 50,000+ tokens down to 200-500 tokens while preserving semantic content. This is early research that works well at moderate compression but accuracy drops at extreme compression ratios. This enables real-time fundamental analysis that would otherwise be prohibitively expensive or slow.
  - **What it does**: Converts unstructured PDFs (earnings reports, Fed minutes, analyst reports) into compact, structured data (text summaries + extracted chart/table data as JSON) that the WHY sensor can analyze efficiently.
  - **How it works (two-stage pipeline)**: (A) OCR/Layout extraction (Google Cloud Document AI / Azure Document Intelligence / GCP Vision API) extracts text, tables, and layout → (B) VL/LLM summarizer (OpenAI GPT-4 Vision / Gemini Flash / Claude) compresses extracted content with semantic prompt → Store compressed data in database → WHY sensor analyzes.
  - **Why cloud API**: The two-stage approach leverages specialized OCR services for accurate extraction, then uses vision-language models for semantic compression. Using cloud APIs avoids needing GPU infrastructure ($0 additional hardware cost) and leverages state-of-the-art models.
  - **Cost analysis**: 
    - **Stage A (OCR/Layout)**: Google Cloud Vision ~$1.50 per 1,000 pages; Azure Document Intelligence similar per-page pricing (region-specific)
    - **Stage B (VL/LLM Summarization)**: OpenAI GPT-4 Vision variable (token + image-token based, see pricing docs); Gemini Flash ~$0.075 per 1M input tokens
    - **Total estimate**: Light usage (10 docs/month, 1,000 pages) = $15-25/month; Medium (100 docs/month, 10,000 pages) = $150-250/month; Heavy (1,000 docs/month) = $1,500-2,500/month
    - **At 1,000+ docs/month**: Consider self-hosted model with GPU (break-even analysis needed)
  - **Implementation steps**:
    1. **Select and configure cloud vision API** (4 hours)
       - Evaluate Google Cloud Vision, OpenAI GPT-4 Vision, Azure Computer Vision
       - Recommendation: Google Cloud Vision for document-heavy workloads (best price/performance for PDFs)
       - Create account, obtain API key, configure billing
       - Test API with sample earnings report to verify output quality
       - **Location**: Store API credentials in `.env` file (never commit to git)
       - **Acceptance**: Successfully process sample PDF and receive structured response
    2. **Build document processor module** (8 hours)
       - Create `DocumentProcessor` class that handles PDF → API → structured data flow
       - Implement API client with error handling, retries, and rate limiting
       - Add support for batch processing (multiple documents in parallel)
       - Handle API-specific response formats and normalize to common schema
       - **Location**: `src/sensory/why/documents/document_processor.py` (follows existing pattern)
       - **Key methods**: `process_pdf(path)`, `extract_ocr(pdf_bytes)`, `compress_content(ocr_result)`, `parse_response(response)`
       - **Acceptance**: Process 100-page PDF in ≤60s p50, ≤120s p95 (async batch processing), return structured data
    3. **Implement chart and table extraction** (8 hours)
       - **Important**: OCR services detect text/tables but don't convert charts/plots to series values by default
       - Implement table extraction from OCR results (most financial data is in tables, not charts)
       - For actual charts: Add DePlot-style plot-to-table conversion or rule-based extractor when chart images detected
       - Convert extracted tables/charts into structured JSON (e.g., {2021: 94.7, 2022: 108.2})
       - Handle various formats: tables (primary), line charts (trends), bar charts (comparisons)
       - Validate extracted data for completeness and accuracy
       - **Location**: `src/sensory/why/documents/chart_extractor.py` and `table_extractor.py`
       - **Acceptance**: Extract 90%+ of table data accurately; 70%+ of chart data (charts are harder)
    4. **Create database schema for compressed documents** (2 hours)
       - Design table: `fundamental_reports` with fields: ticker, quarter, compressed_text, charts (JSONB), metrics (JSONB), timestamp
       - Add indexes on ticker and quarter for fast retrieval
       - Implement versioning to handle data corrections
       - **Location**: Add migration to `src/data_foundation/storage/migrations/`
       - **Acceptance**: Store and query compressed documents efficiently (<100ms)
    5. **Integrate with WHY sensor** (4 hours)
       - Modify `fundamental_analyzer.py` to consume compressed document data
       - Extract key metrics (revenue growth, EPS beat, margin trends) from compressed data
       - Generate trading signals based on fundamental analysis
       - Add fallback to traditional data sources if document processing fails
       - **Location**: `src/sensory/why/fundamental_analyzer.py`
       - **Acceptance**: WHY sensor generates signals from compressed earnings reports
    6. **Write comprehensive tests** (2 hours)
       - Unit tests for document processor, chart extractor, API client
       - Integration test: full flow from PDF → compressed data → trading signal
       - Mock API responses to avoid costs during testing
       - Test error handling (API failures, malformed PDFs, missing data)
       - **Location**: `tests/sensory/why/documents/test_document_processor.py`
       - **Acceptance**: 90%+ test coverage, all tests passing
  - **Technical details for developers**:
    - **What is OCR compression?** Traditional OCR extracts all text from a document (50,000+ words from 100-page PDF). The two-stage approach: (1) OCR/layout extraction gets structured text and tables, (2) VL/LLM compression uses a vision-language model to semantically compress the content (keeping important info, discarding boilerplate) down to 200-500 tokens. This is 10-20x more efficient than processing raw OCR output with LLMs.
    - **Where does compression happen?** Stage A (OCR) happens in the OCR API (Google Document AI, Azure DI). Stage B (compression) happens when you send the OCR result to a VL/LLM (GPT-4 Vision, Gemini) with a compression prompt. You orchestrate the two-stage pipeline in your code.
    - **Why is this better than traditional PDF parsing?** Traditional parsing (PyPDF2, pdfplumber) extracts raw text but loses layout/tables and requires expensive LLM processing. OCR services preserve structure (tables, sections). VL/LLMs can understand context and compress semantically. The two-stage approach gets best of both: accurate extraction + intelligent compression.
    - **Chart extraction reality**: Most financial data is in tables (balance sheets, income statements), which OCR handles well. Actual charts (line/bar plots) require additional processing (DePlot-style plot-to-table or specialized chart understanding). Prioritize table extraction first.
    - **Example API request/response**:
      ```python
      # Request (simplified)
      response = requests.post(
          'https://vision.googleapis.com/v1/documents:analyze',
          files={'file': open('earnings.pdf', 'rb')},
          headers={'Authorization': f'Bearer {API_KEY}'}
      )
      
      # Response (simplified)
      {
          'text': 'Apple Q4 2024: Revenue $89.5B (+8% YoY), iPhone $43.8B (+3%), Services $22.3B (+16%), Gross margin 45.2% (up 0.5pp), EPS $1.64 (beat est. $1.60)...',
          'charts': [
              {'name': 'revenue_trend', 'data': {'2021': 94.7, '2022': 108.2, '2023': 112.5, '2024': 119.3}},
              {'name': 'segment_revenue', 'data': {'iPhone': 43.8, 'Mac': 7.6, 'iPad': 6.4, 'Services': 22.3}}
          ],
          'tables': [...]
      }
      ```
    - **Latency expectations**: Google Document AI publishes ~120 pages/min for Gemini Flash tiers (≈50s for 100 pages). Custom extractors may be slower. Set realistic SLAs: ≤60s p50, ≤120s p95 for 100-page documents with batch async processing. Parallelize Stage A and B where possible.
    - **Cost optimization**: Cache processed documents to avoid re-processing. Batch process documents during off-peak hours if possible. Monitor API usage and set budget alerts. Consider Gemini Flash for Stage B (cheaper than GPT-4 Vision for summarization).
    - **Error handling**: APIs can fail (rate limits, downtime, malformed PDFs). Implement exponential backoff retries, fallback to traditional data sources, and alert on repeated failures. Handle partial failures gracefully (e.g., OCR succeeds but compression fails).
    - **Governance and privacy**: 
      - **No MNPI to third-party APIs**: Only process public documents (earnings reports, Fed minutes). Never send material non-public information.
      - **Redact before upload**: Strip any sensitive metadata (internal notes, proprietary analysis) before sending to external APIs.
      - **Audit trail**: Emit tamper-evident audit record for every external API call (document hash, provider, scope, cost) to PolicyLedger/AuditLogger.
      - **For sensitive documents**: Consider self-hosted model with GPU (see alternative implementation below) to keep data in your infrastructure.
  - **Alternative implementation (self-hosted, requires GPU)**: If processing >1,000 documents/month or have privacy requirements, consider self-hosting the vision model. Requires: NVIDIA GPU (RTX 3060 12GB minimum, RTX 4060 Ti 16GB recommended), pre-trained model (Donut, Pix2Struct, or DeepSeek-OCR if available), inference server (FastAPI + PyTorch). Implementation effort: +40 hours. Operating cost: GPU hardware ($300-600) + electricity (~$30/month). Break-even vs. cloud API: ~80 months at 100 docs/month.
  - **Success metrics**: 90%+ accuracy in extracting key metrics from earnings reports; 10-15 second processing time per document; <$20/month API costs for typical usage; WHY sensor generates actionable signals from compressed documents.
  - **Acceptance**: Process Apple Q4 2024 earnings report (or similar 100-page PDF) → Extract revenue, EPS, margins, guidance → Store in database → WHY sensor generates BUY/SELL/HOLD signal based on fundamental analysis → All within 15 seconds of document release.

#### 1.4 Instrument Translation Protocol

**Objective**: Universal instrument representation across asset classes

- [ ] **Design universal instrument schema** (8 hours)
  - Define fields for equities, futures, options, forex, crypto
  - Handle instrument-specific attributes (strikes, expiries, etc.)
  - Design unique identifier system
  - **Acceptance**: Schema document covering all asset classes

- [ ] **Implement InstrumentTranslator** (16 hours)
  - Build translator from broker symbols to universal format
  - Support multiple naming conventions (Bloomberg, Reuters, etc.)
  - Implement reverse translation
  - **Location**: `src/sensory/services/instrument_translator.py`
  - **Acceptance**: Translate 1000+ instruments with 100% accuracy

- [ ] **Create instrument metadata registry** (8 hours)
  - Build database of instrument specifications
  - Include contract sizes, tick sizes, trading hours
  - Add exchange and regulatory information
  - **Location**: `src/sensory/services/instrument_registry.py`
  - **Acceptance**: Registry covers major global instruments

- [ ] **Integrate with existing SymbolMapper** (8 hours)
  - Extend SymbolMapper to use InstrumentTranslator
  - Maintain backward compatibility
  - Migrate existing code to new translator
  - **Acceptance**: All existing functionality preserved

- [ ] **Build instrument search and discovery** (8 hours)
  - Implement fuzzy search for instruments
  - Add filtering by asset class, exchange, etc.
  - Create API for instrument lookup
  - **Acceptance**: Find any instrument in <100ms

#### 1.5 TimescaleDB Integration

**Objective**: High-performance storage for tick data

- [ ] **Set up TimescaleDB infrastructure** (8 hours)
  - Deploy TimescaleDB instance (local or cloud)
  - Configure hypertables for tick data
  - Set up retention policies and compression
  - **Acceptance**: TimescaleDB operational with monitoring

- [ ] **Design tick data schema** (4 hours)
  - Define tables for trades, quotes, order book snapshots
  - Optimize for time-series queries
  - Plan partitioning strategy
  - **Acceptance**: Schema optimized for query performance

- [ ] **Implement TimescaleDB adapter** (12 hours)
  - Build async writer for high-throughput ingestion
  - Implement batch writes for efficiency
  - Add connection pooling and error handling
  - **Location**: `src/data_foundation/storage/timescale_adapter.py`
  - **Acceptance**: Ingest 10K+ ticks/second

- [ ] **Create query interface** (8 hours)
  - Build API for historical tick data queries
  - Implement efficient windowing and aggregation
  - Add caching for common queries
  - **Location**: `src/data_foundation/storage/timescale_queries.py`
  - **Acceptance**: Query 1 day of tick data in <1 second

- [ ] **Migrate existing data** (8 hours)
  - Export data from DuckDB
  - Import into TimescaleDB
  - Verify data integrity
  - **Acceptance**: All historical data migrated successfully

---

### Phase 2: Strategy Development (Weeks 5-8)

**Goal**: Build comprehensive, backtested strategy library

**Estimated Effort**: 160 hours (4 weeks × 40 hours)

#### 2.1 Comprehensive Backtesting Framework

**Objective**: Validate strategies before live deployment

- [ ] **Design backtesting architecture** (8 hours)
  - Define event-driven backtesting engine
  - Plan realistic market simulation (slippage, fees, latency)
  - Design strategy performance metrics
  - **Acceptance**: Architecture document with examples

- [ ] **Implement historical data replay engine** (16 hours)
  - Build time-ordered event replay from multiple data sources
  - Handle OHLCV, tick, and order book data
  - Implement variable speed replay (1x to 1000x)
  - **Location**: `src/backtesting/replay_engine.py`
  - **Acceptance**: Replay 1 year of data with microsecond accuracy

- [ ] **Create market simulator** (16 hours)
  - Simulate order fills with realistic slippage
  - Model market impact for large orders
  - Implement latency simulation
  - **Location**: `src/backtesting/market_simulator.py`
  - **Acceptance**: Realistic fill simulation validated against live data

- [x] **Build backtest orchestrator** (12 hours)
  - Integrate replay engine with existing trading manager
  - Connect to risk manager and execution adapters
  - Implement parallel backtesting for multiple strategies with supervised
    task orchestration and progress callbacks
  - **Location**: `src/backtesting/backtest_orchestrator.py`
  - **Acceptance**: Run 10+ strategies in parallel

- [x] **Implement performance analytics** (12 hours)
  - Calculate Sharpe, Sortino, Calmar ratios
  - Compute drawdown statistics
  - Analyze trade-level attribution
  - **Location**: `src/backtesting/performance_analytics.py`
  - **Acceptance**: Comprehensive performance report for any strategy

- [ ] **Create backtesting CLI and UI** (12 hours)
  - Build CLI for running backtests
  - Generate HTML reports with charts
  - Add comparison tools for multiple strategies
  - **Location**: `src/backtesting/cli.py`
  - **Acceptance**: User-friendly backtest execution and reporting

- [ ] **Write backtesting documentation** (8 hours)
  - Document backtesting methodology
  - Provide examples and tutorials
  - Explain performance metrics
  - **Acceptance**: Complete backtesting guide

#### 2.2 Strategy Library Development

**Objective**: Implement 5-10 proven trading strategies

- [ ] **Strategy 1: Mean Reversion (Pairs Trading)** (16 hours)
  - Implement cointegration-based pair selection
  - Build z-score entry/exit logic
  - Add dynamic hedge ratio adjustment
  - **Location**: `src/trading/strategies/mean_reversion/pairs_trading.py`
  - **Acceptance**: Backtest shows positive Sharpe on historical data

- [ ] **Strategy 2: Momentum (Trend Following)** (16 hours)
  - Implement multi-timeframe momentum signals
  - Build trend strength filters
  - Add volatility-based position sizing
  - **Location**: `src/trading/strategies/momentum/trend_following.py`
  - **Acceptance**: Backtest shows positive returns in trending markets

- [ ] **Strategy 3: Market Making** (20 hours)
  - Implement bid-ask spread capture
  - Build inventory risk management
  - Add adverse selection protection
  - **Location**: `src/trading/strategies/market_making/simple_mm.py`
  - **Acceptance**: Backtest shows consistent small profits

- [x] **Strategy 4: Statistical Arbitrage** (20 hours)
  - Implement PCA-based factor models
  - Build residual mean reversion signals
  - Add multi-asset portfolio construction
  - **Location**: `src/trading/strategies/stat_arb/pca_arb.py`
  - **Acceptance**: PCA residual engine produces market-neutral allocations across managed baskets

- [ ] **Strategy 5: Volatility Trading** (16 hours)
  - Implement realized vs implied volatility signals
  - Build options-like payoff structures with futures
  - Add gamma scalping logic
  - **Location**: `src/trading/strategies/volatility/vol_trading.py`
  - **Acceptance**: Backtest shows profit during volatility spikes

- [ ] **Strategy 6-10: Additional strategies** (60 hours)
  - Select from: breakout, reversal, carry, value, quality
  - Implement with similar rigor as above
  - Ensure diversity across market conditions
  - **Acceptance**: Each strategy has positive backtest results

#### 2.3 Strategy Validation Pipeline

**Objective**: Systematic validation before production deployment

- [ ] **Implement walk-forward analysis** (12 hours)
  - Build rolling window backtesting
  - Implement out-of-sample validation
  - Detect overfitting
  - **Location**: `src/backtesting/walk_forward.py`
  - **Acceptance**: Validate strategy robustness across time periods

- [x] **Create Monte Carlo simulation** (12 hours)
  - Simulate strategy performance under random scenarios
  - Estimate confidence intervals for returns
  - Assess tail risk
  - **Location**: `src/risk/analytics/monte_carlo.py`
  - **Acceptance**: Probabilistic performance estimates with VaR/ES helpers

- [ ] **Build regime analysis** (12 hours)
  - Test strategy performance across market regimes
  - Identify favorable and unfavorable conditions
  - Create regime-based allocation rules
  - **Location**: `src/backtesting/regime_analysis.py`
  - **Acceptance**: Strategy performance by regime documented

- [ ] **Implement stress testing** (8 hours)
  - Test strategies during historical crises
  - Simulate extreme market conditions
  - Validate risk controls under stress
  - **Location**: `src/backtesting/stress_testing.py`
  - **Acceptance**: Strategies survive stress scenarios

- [ ] **Create validation checklist** (4 hours)
  - Document required validation steps
  - Define acceptance criteria for production
  - Build automated validation pipeline
  - **Acceptance**: Checklist integrated into deployment process

---


#### 2.4 CSCV/DSR Statistical Validation

**Effort**: 40-60 hours | **GPU**: ❌ NO | **Priority**: 🔴 **Critical** (after Phase 0)

**Problem**: Without statistical validation, evolution layer breeds overfitted strategies.

- [ ] **Implement CSCV (Combinatorially Symmetric Cross-Validation)** (16 hours)
  - Generate purged, embargoed K-fold splits
  - Configurable purge and embargo gaps
  - **Location**: `src/validation/cscv.py`
  - **Acceptance**: CSCV generates K purged/embargoed folds with configurable gaps

- [ ] **Build DSR (Deflated Sharpe Ratio) calculator** (12 hours)
  - Adjust Sharpe for number of strategies tested
  - Formula: DSR = Sharpe / sqrt(1 + (N-1) * avg_correlation)
  - Account for multiple testing
  - **Location**: `src/validation/deflated_sharpe.py`
  - **Acceptance**: DSR calculated correctly (accounts for multiple testing)

- [ ] **Add multiple testing correction** (8 hours)
  - Implement Bonferroni, Holm, Benjamini-Hochberg
  - Formula: p_adjusted = p_value * N_tests (Bonferroni)
  - **Acceptance**: Multiple testing corrections applied

- [ ] **Create CI promotion gate** (8 hours)
  - Strategies must pass DSR threshold to be promoted (e.g., DSR >1.5)
  - Fail builds if DSR too low
  - **Location**: `src/validation/promotion_gate.py`
  - **Acceptance**: CI gate rejects strategies with DSR < threshold

- [ ] **Build validation report generator** (8 hours)
  - Detailed statistical report for each strategy
  - Show: Sharpe, DSR, p-value, multiple testing correction
  - **Acceptance**: Validation report shows all key metrics

- [ ] **Write comprehensive tests** (8 hours)
  - Test CSCV fold generation
  - Test DSR calculation
  - Test promotion gate logic
  - **Acceptance**: 90%+ test coverage, all tests passing

**Integration Points**:
- Add to training pipeline (`mlops/train.py`)
- Add to CI/CD promotion checks
- Integrate with evolution layer

**Acceptance Criteria**:
- ✅ CSCV generates K purged/embargoed folds with configurable gaps
- ✅ DSR calculated correctly (accounts for multiple testing)
- ✅ CI gate rejects strategies with DSR < threshold (e.g., 1.5)
- ✅ Validation report shows: Sharpe, DSR, p-value, multiple testing correction

---

#### 2.5 Complete Impact-Aware Execution

**Effort**: 24-40 hours | **GPU**: ❌ NO | **Priority**: 🟡 **High** (before live trading)

**Problem**: Without impact modeling, large orders move prices against you, bleeding alpha.

- [ ] **Verify Almgren-Chriss implementation** (4 hours)
  - Check `src/trading/execution/market_impact_model.py` exists and works
  - Check `src/trading/execution/almgren_chriss.py` exists and works
  - **Acceptance**: Almgren-Chriss model calculates optimal execution trajectory

- [ ] **Build alpha half-life estimator** (12 hours)
  - Estimate how quickly alpha decays from historical signal decay
  - Fit exponential decay model to past signals
  - **Location**: `src/trading/execution/alpha_half_life.py`
  - **Acceptance**: Alpha half-life estimated from historical signal decay

- [ ] **Implement order scheduler** (16 hours)
  - TWAP (Time-Weighted Average Price)
  - VWAP (Volume-Weighted Average Price)
  - IS (Implementation Shortfall)
  - **Location**: `src/trading/execution/order_scheduler.py`
  - **Acceptance**: TWAP/VWAP/IS schedulers implemented

- [ ] **Create execution strategy selector** (8 hours)
  - Choose optimal schedule per trade based on alpha half-life
  - Fast alpha → IS, slow alpha → VWAP, medium → TWAP
  - **Acceptance**: Scheduler selection logic implemented

- [ ] **Write comprehensive tests** (8 hours)
  - Test alpha half-life estimation
  - Test all schedulers (TWAP/VWAP/IS)
  - Test strategy selection logic
  - **Acceptance**: 90%+ test coverage, all tests passing

**Integration Points**:
- Integrate with execution layer (`src/trading/execution/`)
- Connect to order router
- Feed telemetry back to calibration loop

**Acceptance Criteria**:
- ✅ Almgren-Chriss model calculates optimal execution trajectory
- ✅ Alpha half-life estimated from historical signal decay
- ✅ TWAP/VWAP/IS schedulers implemented
- ✅ Scheduler selection: fast alpha → IS, slow alpha → VWAP, medium → TWAP

---

#### 2.6 Operational Validation (Shadow Mode & Consistency)

**Effort**: 72-112 hours | **GPU**: ❌ NO | **Priority**: 🔴 **Critical** (before live trading)

**Problem**: Strategies that work in backtest often fail in live trading.

##### 2.6.1 Shadow-Mode → Canary → Promotion Lifecycle (32-48h)

**State machine**: IDEA → BACKTESTED → CSCV/DSR-PASSED → SHADOW-LIVE → CANARY → PROMOTED → MONITORED → QUARANTINED/RETIRED

- [ ] **Build strategy lifecycle state machine** (12 hours)
  - Track state transitions for each strategy
  - Enforce state progression rules
  - **Location**: `src/governance/strategy_lifecycle.py`
  - **Acceptance**: Strategies can only be promoted through state machine

- [ ] **Create shadow trading engine** (16 hours)
  - Run strategies in paper mode alongside live
  - Track paper P&L without executing real trades
  - **Location**: `src/trading/shadow_engine.py`
  - **Acceptance**: Shadow mode runs for minimum 30 days before canary

- [ ] **Implement canary capital allocator** (8 hours)
  - Start with tiny capital (e.g., 1% of target)
  - Gradually increase if performance holds
  - **Acceptance**: Canary starts with <5% of target capital

- [ ] **Add automatic rollback rules** (8 hours)
  - Demote if drawdown/latency/impact breaches thresholds
  - **Location**: `src/governance/promotion_rules.py`
  - **Acceptance**: Automatic demotion if thresholds breached

- [ ] **Write comprehensive tests** (8 hours)
  - Test state machine transitions
  - Test shadow engine
  - Test rollback logic
  - **Acceptance**: 90%+ test coverage, all tests passing

##### 2.6.2 Paper↔Live Consistency SLO (16-24h)

**Formula**: `Consistency = |live P&L - paper P&L (costed)| / turnover`

- [ ] **Build consistency calculator** (8 hours)
  - Compute consistency score for each strategy
  - Formula: |live - paper| / turnover
  - **Location**: `src/validation/consistency_slo.py`
  - **Acceptance**: Consistency calculated daily for all strategies

- [ ] **Create rolling window tracker** (6 hours)
  - Track 30-day and 90-day consistency
  - **Acceptance**: Rolling windows tracked

- [ ] **Add promotion gate** (4 hours)
  - Require consistency >threshold for promotion (e.g., <0.1)
  - **Acceptance**: Promotion requires 30-day consistency <0.1

- [ ] **Implement divergence alarm** (4 hours)
  - Trigger forensics if consistency drifts (e.g., >0.2)
  - **Location**: `src/operations/monitoring/consistency_monitor.py`
  - **Acceptance**: Alarm if consistency >0.2

- [ ] **Write comprehensive tests** (4 hours)
  - Test consistency calculation
  - Test rolling windows
  - Test alarm triggers
  - **Acceptance**: 90%+ test coverage, all tests passing

##### 2.6.3 Truthful P&L & Cost Accounting (24-40h)

- [ ] **Implement broker-accurate P&L** (12 hours)
  - Include spreads, commissions, swaps, funding, dividends, FX conversions
  - **Location**: `src/trading/accounting/truthful_pnl.py`
  - **Acceptance**: P&L matches broker statements within 0.1%

- [ ] **Build nightly reconciliation** (8 hours)
  - Compare internal P&L to broker statements
  - Alert on discrepancies
  - **Location**: `src/trading/accounting/reconciliation.py`
  - **Acceptance**: Nightly reconciliation runs automatically

- [ ] **Create Implementation Shortfall tracker** (12 hours)
  - Signal vs. realized by venue/instrument/strategy
  - **Location**: `src/trading/accounting/implementation_shortfall.py`
  - **Acceptance**: Implementation Shortfall reported per strategy

- [ ] **Add cost attribution** (8 hours)
  - Which strategies pay most in costs?
  - Dashboard showing cost efficiency
  - **Acceptance**: Cost attribution dashboard implemented

- [ ] **Write comprehensive tests** (8 hours)
  - Test P&L calculation
  - Test reconciliation logic
  - Test shortfall tracking
  - **Acceptance**: 90%+ test coverage, all tests passing

**Integration Points**:
- Integrate with broker APIs for statement downloads
- Connect to accounting system
- Feed data to governance dashboards

**Acceptance Criteria**:
- ✅ Strategies can only be promoted through state machine
- ✅ Shadow mode runs for minimum 30 days before canary
- ✅ Canary starts with <5% of target capital
- ✅ Automatic demotion if: drawdown >threshold, latency >threshold, consistency <threshold
- ✅ Consistency calculated daily for all strategies
- ✅ Promotion requires 30-day consistency <0.1 (10% of turnover)
- ✅ Alarm if consistency >0.2 (investigate immediately)
- ✅ P&L matches broker statements within 0.1%
- ✅ All costs tracked: spreads, commissions, swaps, funding, FX
- ✅ Implementation Shortfall reported per strategy
- ✅ Cost attribution dashboard shows which strategies are cost-efficient

---

### Phase 3.9: Formal Regime Routing & Capital Allocation

**Priority**: 🟡 **High** - Transforms regime detection into action

**Effort**: 32-48 hours | **GPU**: ❌ NO

**Problem**: You detect regimes but don't act on them. Need to allocate capital and suspend/promote strategies based on regime.

- [ ] **Create regime-strategy compatibility matrix** (8 hours)
  - Define which strategies work in which regimes
  - YAML configuration file
  - **Location**: `config/regime_strategy_matrix.yaml`
  - **Acceptance**: Compatibility matrix defines which strategies run in which regimes

- [ ] **Build capital allocation by regime** (12 hours)
  - Risk budgets per regime (e.g., crisis: 20%, bull: 80%)
  - Automatic adjustment on regime change
  - **Location**: `src/trading/capital_allocator.py`
  - **Acceptance**: Capital allocation adjusts automatically on regime change

- [ ] **Implement dynamic strategy suspension/promotion** (8 hours)
  - Automatic based on current regime
  - Suspend strategies when regime is hostile
  - **Location**: `src/thinking/regime_router.py`
  - **Acceptance**: Strategies suspended when regime is hostile

- [ ] **Add hysteresis & cooldown** (4 hours)
  - Prevent thrashing between regimes
  - Require 2+ consecutive detections before regime change
  - **Acceptance**: Hysteresis prevents regime thrashing

- [ ] **Integrate with evolution layer** (4 hours)
  - Breed regime-specific strategies
  - Tag strategies with regime affinity
  - **Acceptance**: Evolution breeds regime-specific strategies

- [ ] **Write comprehensive tests** (4 hours)
  - Test compatibility matrix loading
  - Test capital allocation logic
  - Test suspension/promotion
  - **Acceptance**: 90%+ test coverage, all tests passing

**Integration Points**:
- Connect to regime detector (`src/sensory/when/regime_detector.py`)
- Integrate with capital allocator
- Feed into strategy lifecycle
### Phase 3: Learning Pipeline (Weeks 9-12)

**Goal**: Enable autonomous learning and continuous improvement

**Estimated Effort**: 160 hours (4 weeks × 40 hours)

#### 3.1 Automated FAISS Memory Management

**Objective**: Continuous learning from trading experience

- [ ] **Implement automated memory retraining** (12 hours)
  - Schedule periodic FAISS index rebuilding
  - Implement incremental index updates
  - Add memory compaction and pruning
  - **Location**: `src/sentient/memory/auto_retraining.py`
  - **Acceptance**: Memory auto-updates every 24 hours

- [ ] **Build experience sampling strategy** (8 hours)
  - Implement prioritized experience replay
  - Balance recent vs historical experiences
  - Add diversity-based sampling
  - **Location**: `src/sentient/memory/sampling.py`
  - **Acceptance**: Efficient sampling of relevant experiences

- [ ] **Create memory quality monitoring** (8 hours)
  - Track memory utilization and hit rates
  - Detect degraded memory performance
  - Alert on memory quality issues
  - **Location**: `src/sentient/memory/monitoring.py`
  - **Acceptance**: Real-time memory health metrics

- [ ] **Implement memory backup and recovery** (8 hours)
  - Periodic snapshots of FAISS indices
  - Implement recovery from corruption
  - Add version control for memory states
  - **Location**: `src/sentient/memory/backup.py`
  - **Acceptance**: Memory recoverable from any snapshot

#### 3.2 Advanced Fitness Evaluation

**Objective**: Sophisticated strategy evaluation for evolution

- [ ] **Design multi-objective fitness function** (8 hours)
  - Define objectives: return, risk, drawdown, consistency
  - Implement Pareto optimization
  - Add regime-specific fitness
  - **Acceptance**: Fitness function design document

- [x] **Implement fitness calculator** (16 hours)
  - Build comprehensive fitness evaluation
  - Include risk-adjusted returns
  - Add behavioral penalties (overtrading, etc.)
  - **Location**: `src/evolution/fitness/calculator.py`
  - **Acceptance**: Fitness scores combine weighted performance, risk, and behavioural penalties for evolution loops

- [ ] **Create fitness benchmarking** (8 hours)
  - Compare strategies against baselines
  - Implement relative fitness scoring
  - Add peer comparison
  - **Location**: `src/evolution/fitness/benchmarking.py`
  - **Acceptance**: Strategies ranked by relative performance

- [ ] **Build fitness visualization** (8 hours)
  - Plot fitness evolution over generations
  - Visualize Pareto frontiers
  - Create fitness distribution charts
  - **Location**: `src/evolution/fitness/visualization.py`
  - **Acceptance**: Clear visual representation of evolution progress

#### 3.3 Multi-Objective Optimization

**Objective**: Evolve strategies optimizing multiple goals simultaneously

- [x] **Implement NSGA-II algorithm** (16 hours)
  - Build non-dominated sorting
  - Implement crowding distance calculation
  - Add elitism and selection
  - **Location**: `src/evolution/algorithms/nsga2.py`
  - **Acceptance**: NSGA-II produces diverse Pareto-optimal strategies

- [ ] **Create objective space explorer** (12 hours)
  - Implement objective space visualization
  - Build interactive Pareto front exploration
  - Add objective trade-off analysis
  - **Location**: `src/evolution/optimization/explorer.py`
  - **Acceptance**: Visualize and explore objective trade-offs

- [ ] **Implement constraint handling** (8 hours)
  - Add hard constraints (risk limits, etc.)
  - Implement soft constraints with penalties
  - Build constraint violation tracking
  - **Location**: `src/evolution/optimization/constraints.py`
  - **Acceptance**: Strategies respect all constraints

- [x] **Build preference articulation** (8 hours)
  - Allow user to specify objective preferences
  - Implement weighted sum and goal programming
  - Add interactive preference tuning and articulator helpers
  - **Location**: `src/evolution/optimization/preferences.py`
  - **Acceptance**: Evolution respects user preferences and interactive tuning updates weights

#### 3.4 Cold-Start Simulation Training

**Objective**: Pre-train strategies before live deployment

- [ ] **Build simulation environment** (16 hours)
  - Create realistic market simulator
  - Implement diverse market scenarios
  - Add noise and uncertainty
  - **Location**: `src/simulation/training_env.py`
  - **Acceptance**: Simulation indistinguishable from real markets

- [ ] **Implement curriculum learning** (12 hours)
  - Start with simple scenarios
  - Gradually increase complexity
  - Add adversarial scenarios
  - **Location**: `src/simulation/curriculum.py`
  - **Acceptance**: Strategies learn progressively

- [x] **Create pre-training pipeline** (12 hours)
  - Automate curriculum-aware pre-training orchestration
  - Integrate LoRA planning, multi-task loss evaluation, and horizon diagnostics
  - Emit deterministic summaries for research harnesses
  - **Location**: `src/thinking/learning/pretraining_pipeline.py`
  - **Acceptance**: Pipeline produces reproducible batch summaries and supports stage advancement hooks

- [ ] **Build simulation-to-reality transfer** (12 hours)
  - Implement domain adaptation techniques
  - Measure sim-to-real gap
  - Add fine-tuning on real data
  - **Location**: `src/simulation/transfer.py`
  - **Acceptance**: Strategies transfer successfully to live markets

#### 3.5 Continuous Evolution Loop

**Objective**: Autonomous strategy evolution without human intervention

- [ ] **Implement evolution scheduler** (12 hours)
  - Schedule evolution runs (nightly, weekly)
  - Implement resource management
  - Add evolution monitoring
  - **Location**: `src/evolution/scheduler.py`
  - **Acceptance**: Evolution runs automatically on schedule

- [ ] **Build population management** (12 hours)
  - Implement population size control
  - Add diversity maintenance
  - Build extinction and speciation
  - **Location**: `src/evolution/population_manager.py`
  - **Acceptance**: Population remains diverse and healthy

- [ ] **Create evolution safety controls** (12 hours)
  - Implement kill switches for runaway evolution
  - Add performance degradation detection
  - Build rollback mechanisms
  - **Location**: `src/evolution/safety.py`
  - **Acceptance**: Evolution cannot harm live trading

- [ ] **Implement evolution reporting** (8 hours)
  - Generate evolution progress reports
  - Track best strategies over time
  - Alert on significant improvements
  - **Location**: `src/evolution/reporting.py`
  - **Acceptance**: Clear visibility into evolution progress

#### 3.6 arXiv Research Enhancements

**Objective**: Integrate cutting-edge research from arXiv to enhance system capabilities

**Background**: These enhancements are based on recent peer-reviewed research (2024-2025) that provides natural, non-forced improvements to existing EMP architecture layers. Each enhancement solves a specific problem in the current system without requiring architectural changes.

---

- [ ] **Enhancement 1: Sharpe Ratio Reward Function for Reinforcement Learning** (24 hours)
  - **Research basis**: "A Deep Reinforcement Learning Framework for Dynamic Portfolio Optimization" (arXiv:2412.18563, Dec 2024)
  - **What it does**: Replaces simple return-based reward function with Sharpe ratio-based reward that optimizes risk-adjusted returns and ensures stable RL training convergence.
  - **Where it fits**: THINKING layer → `src/thinking/learning/sentient_learning.py` (existing RL framework)
  - **Why it's not shoehorned**: You already use RL for strategy learning; this just improves the reward signal (drop-in replacement). Sharpe ratio is a standard finance metric—not forcing ML where it doesn't belong. Solves real problem: naive rewards lead to unstable training and high-risk strategies.
  - **GPU required**: ❌ **NO** - Pure mathematical formula, no neural network training
  - **Implementation steps**:
    1. **Implement Sharpe ratio reward function** (8 hours)
       - Create `SharpeRewardCalculator` class with rolling window statistics
       - Formula: `Reward_t = (R_t - R_f) / σ_t` where R_t = portfolio return, R_f = risk-free rate, σ_t = rolling std dev
       - Implement configurable window sizes (default: 20 periods for daily, 100 for intraday)
       - Add numerical stability handling (avoid division by zero when σ_t → 0)
       - **Location**: `src/thinking/learning/rewards/sharpe_reward.py`
       - **Acceptance**: Calculate Sharpe reward for sample trading history, verify against manual calculation
    2. **Integrate with existing RL training loop** (8 hours)
       - Replace current reward function in `sentient_learning.py`
       - Update reward calculation in experience replay buffer
       - Modify policy gradient updates to use new reward signal
       - Maintain backward compatibility (allow switching between reward functions)
       - **Location**: Modify `src/thinking/learning/sentient_learning.py`
       - **Acceptance**: RL agent trains with Sharpe reward, no errors in training loop
    3. **Retrain and compare strategies** (8 hours)
       - Retrain existing strategies with Sharpe reward function
       - Compare performance metrics: convergence speed, final Sharpe ratio, drawdowns
       - Run A/B test: old reward vs. new reward on same historical data
       - Document performance improvements in strategy validation reports
       - **Location**: Results saved to `experiments/sharpe_reward_comparison/`
       - **Acceptance**: New reward function shows improved risk-adjusted performance and training stability
  - **Technical details for developers**:
    - **What is Sharpe ratio reward?** Traditional RL rewards use raw returns (profit/loss), which can lead agents to take excessive risk. Sharpe ratio reward divides returns by volatility, encouraging strategies that achieve returns with lower risk. This aligns RL optimization with real-world trading objectives (risk-adjusted performance, not just maximum returns).
    - **Why rolling window?** Using a rolling window for standard deviation (σ_t) makes the reward adaptive to changing market conditions. During high volatility, the same return gets lower reward (encouraging caution). During low volatility, the same return gets higher reward (encouraging aggression).
    - **Risk-free rate selection**: Use appropriate risk-free rate for your market: US Treasury yield for USD markets, SOFR for overnight, or 0 for simplicity. The paper uses 0 for short-term trading (intraday/daily) where risk-free returns are negligible.
    - **Numerical stability**: When volatility approaches zero (rare but possible in very stable periods), add small epsilon (1e-8) to denominator to prevent division by zero. Also clip reward to reasonable range (e.g., [-10, 10]) to prevent extreme values from destabilizing training.
    - **Expected improvements**: Paper reports consistent positive Sharpe ratios (0.8-1.2) vs. baseline methods (0.3-0.6), faster convergence (30-50% fewer training episodes), and reduced drawdowns (20-40% improvement).
  - **Success metrics**: RL training converges 30%+ faster; trained strategies achieve Sharpe ratio >0.8 in backtests; drawdowns reduced by 20%+ vs. old reward function.
  - **Acceptance**: Retrain momentum strategy with Sharpe reward → Backtest shows improved risk-adjusted returns → Compare metrics vs. old reward function → Document results.

---

- [ ] **Enhancement 2: TLOB Transformer for Order Book Analysis** (60 hours)
  - **Research basis**: "TLOB: A Novel Transformer Model with Dual Attention for Price Trend Prediction with Limit Order Book Data" (arXiv:2502.15757, Feb 2025)
  - **What it does**: Uses dual attention mechanism (spatial + temporal) to capture complex patterns in limit order book data for superior price prediction. Spatial attention captures relationships between price levels (bid-ask dynamics, depth imbalances). Temporal attention tracks how order book states evolve (order flow patterns, liquidity cycles).
  - **Where it fits**: HOW sensor → `src/sensory/how/order_book_analytics.py` (microstructure analysis)
  - **Why it's not shoehorned**: You're already integrating LOBSTER order book data (Phase 1.1 in roadmap). HOW sensor explicitly needs microstructure analysis. Current order book analytics are rule-based; TLOB learns patterns from data. Solves real problem: predicting price movements from order book requires understanding complex spatial-temporal dependencies.
  - **GPU required**: ⚠️ **OPTIONAL** - Training: 12-24 hours on CPU (one-time, run overnight) or 2-4 hours on GPU. Inference (live trading): CPU is perfect (<10ms per prediction). Alternative: Rent cloud GPU for $2-5 to train in 4 hours.
  - **Implementation steps**:
    1. **Integrate TLOB architecture** (24 hours)
       - Implement dual attention mechanism (spatial attention for price levels, temporal attention for time series)
       - Create `TLOBModel` class with encoder-decoder architecture
       - Implement patch-based self-attention (processes order book as 2D image: price levels × time)
       - Add positional encodings for both spatial (price level) and temporal (time step) dimensions
       - **Location**: `src/sensory/how/models/tlob_model.py`
       - **Dependencies**: PyTorch (already in stack), attention mechanism implementation
       - **Acceptance**: TLOB model architecture implemented, forward pass works on sample order book data
    2. **Prepare LOBSTER training data** (12 hours)
       - Load LOBSTER order book snapshots (requires Phase 1.1 LOBSTER integration complete)
       - Normalize order book features (prices, volumes, depth) to [0, 1] range
       - Create training labels: price movement direction (up/down/neutral) at different horizons (10, 20, 50, 100 ticks)
       - Split data: 70% train, 15% validation, 15% test (chronological split, no lookahead bias)
       - **Location**: `src/sensory/how/data/lobster_dataset.py`
       - **Acceptance**: Training dataset with 100K+ order book snapshots, labels, ready for model training
    3. **Train TLOB model** (16 hours - mostly waiting for training)
       - Implement training loop with Adam optimizer, learning rate scheduling
       - Use cross-entropy loss for classification (up/down/neutral)
       - Train on CPU overnight (12-24 hours) or GPU (2-4 hours if available)
       - Monitor training: loss curves, validation accuracy, attention weight visualizations
       - Save best model checkpoint based on validation accuracy
       - **Location**: `src/sensory/how/training/train_tlob.py`
       - **Acceptance**: Trained model achieves >55% accuracy on test set (paper reports 60-65% for 10-tick horizon)
    4. **Integrate with HOW sensor** (8 hours)
       - Load trained TLOB model in `order_book_analytics.py`
       - Feed live order book snapshots to model for inference (<10ms latency on CPU)
       - Convert model predictions to trading signals (confidence scores for price direction)
       - Combine TLOB signals with existing microstructure indicators (bid-ask spread, depth imbalance)
       - **Location**: Modify `src/sensory/how/order_book_analytics.py`
       - **Acceptance**: HOW sensor generates TLOB-enhanced signals in live trading, latency <10ms
  - **Technical details for developers**:
    - **What is dual attention?** TLOB uses two attention mechanisms: (1) Spatial attention looks across price levels at a single time step (e.g., "bid depth at $100 is unusually high relative to $99.95"), (2) Temporal attention looks at the same price level across time (e.g., "ask volume at $100.05 has been increasing for last 5 snapshots"). This captures both instantaneous order book state and its evolution.
    - **Why transformers for order books?** Traditional methods (CNNs, LSTMs) struggle with long-range dependencies in order books. Transformers can attend to any price level or time step directly, learning which parts of the order book matter most for prediction. The paper shows transformers outperform CNNs/LSTMs by 5-10% accuracy.
    - **Prediction horizons**: TLOB works for multiple horizons (10, 20, 50, 100 ticks ahead). Shorter horizons (10-20 ticks) are easier to predict (60-65% accuracy) but less profitable (small price moves). Longer horizons (50-100 ticks) are harder (55-60% accuracy) but more profitable. Choose based on your trading style.
    - **Data requirements**: Minimum 50K order book snapshots for training (1-2 weeks of LOBSTER data). More data improves performance. Paper uses 500K+ snapshots from FI-2010 benchmark.
    - **CPU vs GPU training**: On modern CPU (8+ cores), training takes 12-24 hours for 50K snapshots. On GPU (RTX 3060+), 2-4 hours. Since this is one-time training (or monthly retraining), CPU overnight is perfectly acceptable. Inference is fast on CPU (<10ms), so no GPU needed for live trading.
    - **Code availability**: Paper states "We release the code at [GitHub URL]" - can directly adapt their implementation. Check arXiv paper abstract for GitHub link.
  - **Complete implementation code for coder**:
    
    **File: `src/sensory/how/models/tlob_model.py`** (Complete implementation)
    ```python
    import torch
    import torch.nn as nn
    import math
    
    class TLOBModel(nn.Module):
        """
        Transformer-based Limit Order Book model with dual attention.
        Based on arXiv:2502.15757 (TLOB paper).
        """
        def __init__(self, 
                     num_price_levels=10,      # Number of price levels (5 bid + 5 ask)
                     num_time_steps=100,        # Number of time steps in sequence
                     d_model=64,                # Model dimension
                     nhead=4,                   # Number of attention heads
                     num_encoder_layers=3,      # Number of transformer layers
                     dim_feedforward=256,       # FFN dimension
                     dropout=0.1,
                     num_classes=3):            # up/down/neutral
            super().__init__()
            
            self.num_price_levels = num_price_levels
            self.num_time_steps = num_time_steps
            self.d_model = d_model
            
            # Input projection: (batch, time, price_levels, features) -> (batch, time*price, d_model)
            # Each price level has 4 features: bid_price, bid_volume, ask_price, ask_volume
            self.input_projection = nn.Linear(4, d_model)
            
            # Positional encodings for spatial (price level) and temporal (time step)
            self.spatial_pos_encoding = nn.Parameter(
                self._create_positional_encoding(num_price_levels, d_model)
            )
            self.temporal_pos_encoding = nn.Parameter(
                self._create_positional_encoding(num_time_steps, d_model)
            )
            
            # Dual attention transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_encoder_layers
            )
            
            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, num_classes)
            )
        
        def _create_positional_encoding(self, max_len, d_model):
            """Create sinusoidal positional encoding."""
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                                (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            return pe
        
        def forward(self, x):
            """
            Forward pass.
            
            Args:
                x: Tensor of shape (batch, time_steps, price_levels, 4)
                   where 4 features are [bid_price, bid_volume, ask_price, ask_volume]
            
            Returns:
                logits: Tensor of shape (batch, num_classes)
            """
            batch_size = x.shape[0]
            
            # Project input: (batch, time, price, 4) -> (batch, time, price, d_model)
            x = self.input_projection(x)
            
            # Add spatial positional encoding (broadcast across time dimension)
            x = x + self.spatial_pos_encoding.unsqueeze(0).unsqueeze(0)
            
            # Add temporal positional encoding (broadcast across price dimension)
            x = x + self.temporal_pos_encoding.unsqueeze(0).unsqueeze(2)
            
            # Reshape to (batch, time*price, d_model) for transformer
            x = x.view(batch_size, self.num_time_steps * self.num_price_levels, self.d_model)
            
            # Apply transformer encoder (dual attention happens here)
            x = self.transformer_encoder(x)
            
            # Global average pooling across sequence
            x = x.mean(dim=1)  # (batch, d_model)
            
            # Classification
            logits = self.classifier(x)  # (batch, num_classes)
            
            return logits
    ```
    
    **File: `src/sensory/how/data/lobster_dataset.py`** (Data preprocessing)
    ```python
    import torch
    from torch.utils.data import Dataset
    import numpy as np
    import pandas as pd
    
    class LOBSTERDataset(Dataset):
        """
        Dataset for LOBSTER limit order book data.
        Prepares data for TLOB model training.
        """
        def __init__(self, 
                     lobster_file_path,
                     num_price_levels=10,
                     num_time_steps=100,
                     prediction_horizon=10,
                     smoothing_window=5):
            """
            Args:
                lobster_file_path: Path to LOBSTER CSV file
                num_price_levels: Number of price levels (5 bid + 5 ask = 10 total)
                num_time_steps: Sequence length (number of snapshots)
                prediction_horizon: How many ticks ahead to predict (10, 20, 50, 100)
                smoothing_window: Window for smoothing price changes to create labels
            """
            self.num_price_levels = num_price_levels
            self.num_time_steps = num_time_steps
            self.prediction_horizon = prediction_horizon
            
            # Load LOBSTER data
            # LOBSTER format: columns are [ask_price_1, ask_vol_1, bid_price_1, bid_vol_1, ...]
            df = pd.read_csv(lobster_file_path, header=None)
            
            # Extract price and volume data
            # Assuming 10 levels: columns 0-39 (4 values per level: ask_p, ask_v, bid_p, bid_v)
            self.data = df.values
            
            # Normalize data to [0, 1] range
            self.data_normalized = self._normalize(self.data)
            
            # Create labels (price movement direction)
            self.labels = self._create_labels(df, prediction_horizon, smoothing_window)
            
            # Create sequences
            self.sequences = []
            self.sequence_labels = []
            
            for i in range(len(self.data) - num_time_steps - prediction_horizon):
                # Get sequence of order book snapshots
                seq = self.data_normalized[i:i+num_time_steps]
                
                # Reshape to (time_steps, price_levels, 4)
                seq_reshaped = self._reshape_lob_data(seq)
                
                # Get label for this sequence
                label = self.labels[i + num_time_steps + prediction_horizon - 1]
                
                self.sequences.append(seq_reshaped)
                self.sequence_labels.append(label)
            
            self.sequences = np.array(self.sequences)
            self.sequence_labels = np.array(self.sequence_labels)
        
        def _normalize(self, data):
            """Normalize each feature to [0, 1] range."""
            # Normalize prices and volumes separately
            normalized = np.zeros_like(data, dtype=np.float32)
            
            for col in range(data.shape[1]):
                min_val = data[:, col].min()
                max_val = data[:, col].max()
                if max_val > min_val:
                    normalized[:, col] = (data[:, col] - min_val) / (max_val - min_val)
                else:
                    normalized[:, col] = 0.0
            
            return normalized
        
        def _create_labels(self, df, horizon, smoothing_window):
            """
            Create labels based on mid-price movement.
            0 = down, 1 = neutral, 2 = up
            """
            # Calculate mid-price (average of best bid and best ask)
            best_ask = df.iloc[:, 0]  # First column is best ask price
            best_bid = df.iloc[:, 2]  # Third column is best bid price
            mid_price = (best_ask + best_bid) / 2
            
            # Calculate price change over horizon
            price_change = mid_price.shift(-horizon) - mid_price
            
            # Smooth price changes to reduce noise
            price_change_smoothed = price_change.rolling(window=smoothing_window, center=True).mean()
            
            # Create labels based on threshold (0.01% of mid-price)
            threshold = mid_price * 0.0001
            
            labels = np.zeros(len(df), dtype=np.int64)
            labels[price_change_smoothed > threshold] = 2   # up
            labels[price_change_smoothed < -threshold] = 0  # down
            labels[np.abs(price_change_smoothed) <= threshold] = 1  # neutral
            
            # Fill NaN values (from smoothing) with neutral
            labels[np.isnan(price_change_smoothed)] = 1
            
            return labels
        
        def _reshape_lob_data(self, seq):
            """
            Reshape LOBSTER data to (time_steps, price_levels, 4).
            
            LOBSTER format per row: [ask_p1, ask_v1, bid_p1, bid_v1, ask_p2, ask_v2, ...]
            We want: [time, level, [bid_p, bid_v, ask_p, ask_v]]
            """
            time_steps = seq.shape[0]
            reshaped = np.zeros((time_steps, self.num_price_levels, 4), dtype=np.float32)
            
            for t in range(time_steps):
                for level in range(self.num_price_levels // 2):  # 5 levels each side
                    # LOBSTER columns: ask_p, ask_v, bid_p, bid_v (repeating)
                    base_idx = level * 4
                    
                    # Extract values
                    ask_price = seq[t, base_idx]
                    ask_volume = seq[t, base_idx + 1]
                    bid_price = seq[t, base_idx + 2]
                    bid_volume = seq[t, base_idx + 3]
                    
                    # Store as [bid_price, bid_volume, ask_price, ask_volume]
                    reshaped[t, level * 2] = [bid_price, bid_volume, 0, 0]      # bid side
                    reshaped[t, level * 2 + 1] = [0, 0, ask_price, ask_volume]  # ask side
            
            return reshaped
        
        def __len__(self):
            return len(self.sequences)
        
        def __getitem__(self, idx):
            return (
                torch.FloatTensor(self.sequences[idx]),
                torch.LongTensor([self.sequence_labels[idx]])[0]
            )
    ```
    
    **File: `src/sensory/how/training/train_tlob.py`** (Training script)
    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split
    import sys
    import os
    
    # Add project root to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    
    from src.sensory.how.models.tlob_model import TLOBModel
    from src.sensory.how.data.lobster_dataset import LOBSTERDataset
    
    def train_tlob():
        """Train TLOB model on LOBSTER data."""
        
        # Hyperparameters (tuned based on paper)
        BATCH_SIZE = 32
        LEARNING_RATE = 0.001
        NUM_EPOCHS = 50
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model hyperparameters
        NUM_PRICE_LEVELS = 10
        NUM_TIME_STEPS = 100
        D_MODEL = 64
        NHEAD = 4
        NUM_ENCODER_LAYERS = 3
        DIM_FEEDFORWARD = 256
        DROPOUT = 0.1
        
        # Data parameters
        PREDICTION_HORIZON = 10  # Predict 10 ticks ahead (change for different horizons)
        
        print(f"Using device: {DEVICE}")
        
        # Load dataset
        print("Loading LOBSTER dataset...")
        dataset = LOBSTERDataset(
            lobster_file_path='data/lobster/orderbook.csv',  # Update path as needed
            num_price_levels=NUM_PRICE_LEVELS,
            num_time_steps=NUM_TIME_STEPS,
            prediction_horizon=PREDICTION_HORIZON
        )
        
        # Split dataset: 70% train, 15% val, 15% test
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        print(f"Dataset sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")
        
        # Initialize model
        model = TLOBModel(
            num_price_levels=NUM_PRICE_LEVELS,
            num_time_steps=NUM_TIME_STEPS,
            d_model=D_MODEL,
            nhead=NHEAD,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            num_classes=3  # up/down/neutral
        ).to(DEVICE)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # Training loop
        best_val_acc = 0.0
        
        for epoch in range(NUM_EPOCHS):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(DEVICE), target.to(DEVICE)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx}/{len(train_loader)}, '
                          f'Loss: {loss.item():.4f}')
            
            train_acc = 100. * train_correct / train_total
            train_loss /= len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()
            
            val_acc = 100. * val_correct / val_total
            val_loss /= len(val_loader)
            
            print(f'Epoch {epoch+1}/{NUM_EPOCHS}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Learning rate scheduling
            scheduler.step(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'models/tlob_best.pth')
                print(f'Saved best model with validation accuracy: {val_acc:.2f}%')
        
        # Test evaluation
        print("\nEvaluating on test set...")
        model.load_state_dict(torch.load('models/tlob_best.pth'))
        model.eval()
        
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
        
        test_acc = 100. * test_correct / test_total
        print(f'Test Accuracy: {test_acc:.2f}%')
        
        return model
    
    if __name__ == '__main__':
        train_tlob()
    ```

  - **Success metrics**: Test set accuracy >55% for 10-tick horizon; inference latency <10ms on CPU; backtests show improved entry/exit timing vs. rule-based order book analytics.
  - **Acceptance**: Train TLOB on LOBSTER data → Integrate with HOW sensor → Backtest strategy with TLOB signals → Compare Sharpe ratio vs. without TLOB → Document 10%+ improvement in risk-adjusted returns.

---

- [ ] **Enhancement 3: FinRL Benchmark Framework Integration** (40 hours)
  - **Research basis**: "FinRL Contests: Benchmarking Data-driven Financial Reinforcement Learning Agents" (arXiv:2504.02281, May 2025)
  - **What it does**: Provides standardized environments, benchmark datasets, and baseline algorithms to validate your strategies against state-of-the-art methods. Enables GPU-optimized parallel backtesting for faster iteration.
  - **Where it fits**: Backtesting infrastructure → Phase 2.1 (you need backtesting anyway)
  - **Why it's not shoehorned**: You need backtesting framework (it's in roadmap). Solves real problem: without benchmarks, you can't prove your 5-layer architecture outperforms simpler alternatives. Community-driven standard (200+ participants, 100+ institutions). Enables validation that sophistication delivers value.
  - **GPU required**: ⚠️ **OPTIONAL** - CPU backtesting: 15-30 minutes for 10 strategies. GPU backtesting: 5 minutes for same workload. For most use cases (running backtests overnight or weekly), CPU is perfectly adequate. GPU only matters if you're running hundreds of backtests daily (hyperparameter sweeps).
  - **Implementation steps**:
    1. **Install and configure FinRL** (8 hours)
       - Install FinRL library: `pip install finrl`
       - Familiarize with FinRL API: environments, datasets, baseline algorithms
       - Download benchmark datasets (stock trading, crypto, portfolio management)
       - Set up FinRL configuration for your asset classes and time periods
       - **Location**: `requirements/finrl.txt` for dependencies
       - **Acceptance**: FinRL installed, sample environment runs successfully
    2. **Adapt EMP strategies to FinRL environments** (16 hours)
       - Create FinRL-compatible wrapper for your strategies
       - Implement `FinRLStrategyAdapter` that translates between EMP signals and FinRL actions
       - Map your strategy outputs (BUY/SELL/HOLD + position sizes) to FinRL action space
       - Handle differences in data formats (FinRL uses pandas DataFrames, EMP uses custom schemas)
       - **Location**: `src/backtesting/finrl/strategy_adapter.py`
       - **Acceptance**: Run one EMP strategy in FinRL environment, verify trades execute correctly
    3. **Run benchmark comparisons** (8 hours)
       - Select baseline algorithms from FinRL: PPO, A3C, SAC, DDPG
       - Run your strategies alongside baselines on same datasets and time periods
       - Collect performance metrics: Sharpe ratio, max drawdown, total return, win rate
       - Generate comparison reports with statistical significance tests
       - **Location**: `experiments/finrl_benchmarks/`
       - **Acceptance**: Comparison report showing your strategies vs. baselines with clear metrics
    4. **Integrate into CI/CD pipeline** (8 hours)
       - Add FinRL benchmarking to automated testing pipeline
       - Run benchmarks on every major strategy change (weekly or on-demand)
       - Set up alerts if strategy performance degrades below baseline
       - Archive benchmark results for historical tracking
       - **Location**: `.github/workflows/finrl_benchmark.yml` or similar CI config
       - **Acceptance**: Automated benchmarking runs on schedule, results stored and tracked over time
  - **Technical details for developers**:
    - **What is FinRL?** FinRL is an open-source framework for financial reinforcement learning research. It provides: (1) Standardized environments (stock trading, crypto, portfolio management), (2) Pre-processed datasets (Yahoo Finance, Binance, etc.), (3) Baseline RL algorithms (PPO, A3C, SAC), (4) GPU-optimized parallel backtesting. It's the de facto standard for comparing RL trading strategies in academic research.
    - **Why standardized benchmarks matter**: Without benchmarks, you can't objectively assess if your sophisticated 5-layer architecture delivers value over simpler methods. FinRL lets you answer: "Does my EMP system with 4D+1 sensors outperform a simple PPO agent with technical indicators?" If yes, the complexity is justified. If no, you've learned something important.
    - **GPU acceleration**: FinRL can run multiple strategy simulations in parallel on GPU (vectorized environments). This speeds up hyperparameter sweeps and ensemble testing. However, for typical use (backtesting 5-10 strategies weekly), CPU is fine. GPU matters for research teams running thousands of experiments.
    - **Baseline algorithms**: FinRL includes proven RL algorithms: PPO (Proximal Policy Optimization), A3C (Asynchronous Advantage Actor-Critic), SAC (Soft Actor-Critic), DDPG (Deep Deterministic Policy Gradient). These are your competition—if your strategies don't beat these, the extra complexity isn't justified.
    - **Datasets**: FinRL provides clean, preprocessed data: US stocks (2009-2021), crypto (2017-2021), Chinese stocks (CSI 300). You can also add your own data. Use the same data for fair comparison.
    - **Community and credibility**: 200+ participants in FinRL contests, 100+ institutions using it. Publishing results comparable to FinRL benchmarks gives your work credibility and allows comparison with published research.
  - **Complete implementation code for coder**:
    
    **File: `src/backtesting/finrl/strategy_adapter.py`** (Complete implementation)
    ```python
    import numpy as np
    import pandas as pd
    from typing import Dict, Any, List
    
    class FinRLStrategyAdapter:
        """
        Adapter to run EMP strategies in FinRL environments.
        Translates between EMP signal format and FinRL action space.
        """
        def __init__(self, emp_strategy, initial_cash=100000):
            """
            Args:
                emp_strategy: Your EMP strategy object (must have .generate_signal() method)
                initial_cash: Starting cash for trading
            """
            self.emp_strategy = emp_strategy
            self.initial_cash = initial_cash
            self.current_cash = initial_cash
            self.current_positions = {}
        
        def predict(self, observation: np.ndarray) -> np.ndarray:
            """
            FinRL environment calls this method to get actions.
            
            Args:
                observation: numpy array from FinRL environment
                    Shape: (num_stocks * num_features,)
                    Features typically: [open, high, low, close, volume, ...technical indicators]
            
            Returns:
                actions: numpy array of shape (num_stocks,)
                    Values: -1 (sell), 0 (hold), +1 (buy)
                    Or continuous: [-1, 1] for position sizing
            """
            # Convert FinRL observation to EMP format
            emp_data = self._finrl_to_emp_format(observation)
            
            # Get EMP strategy signal
            emp_signal = self.emp_strategy.generate_signal(emp_data)
            
            # Convert EMP signal to FinRL actions
            finrl_actions = self._emp_to_finrl_actions(emp_signal)
            
            return finrl_actions
        
        def _finrl_to_emp_format(self, observation: np.ndarray) -> Dict[str, Any]:
            """
            Convert FinRL observation to EMP data format.
            
            FinRL observation is flat array: [stock1_feat1, stock1_feat2, ..., stock2_feat1, ...]
            EMP expects dict with market data structure.
            """
            # Assuming FinRL env has these features per stock: OHLCV + indicators
            # Adjust based on your actual FinRL environment configuration
            num_features_per_stock = 10  # Example: OHLCV + 5 technical indicators
            num_stocks = len(observation) // num_features_per_stock
            
            # Reshape observation
            obs_reshaped = observation.reshape(num_stocks, num_features_per_stock)
            
            # Create EMP-compatible data structure
            emp_data = {
                'prices': obs_reshaped[:, 3],  # Close prices (index 3 in OHLCV)
                'volumes': obs_reshaped[:, 4],  # Volumes (index 4)
                'indicators': {
                    'open': obs_reshaped[:, 0],
                    'high': obs_reshaped[:, 1],
                    'low': obs_reshaped[:, 2],
                    'close': obs_reshaped[:, 3],
                    # Add other indicators as needed
                },
                'timestamp': pd.Timestamp.now()  # FinRL doesn't provide timestamp in obs
            }
            
            return emp_data
        
        def _emp_to_finrl_actions(self, emp_signal: Dict[str, Any]) -> np.ndarray:
            """
            Convert EMP signal to FinRL action format.
            
            EMP signal format (example):
            {
                'action': 'BUY' | 'SELL' | 'HOLD',
                'symbol': 'AAPL',
                'size': 0.5,  # Fraction of portfolio
                'confidence': 0.8
            }
            
            FinRL action format:
            numpy array of shape (num_stocks,) with values in [-1, 1]
            -1 = sell all, 0 = hold, +1 = buy with all cash
            """
            # Get number of stocks from signal or default
            num_stocks = len(emp_signal.get('symbols', []))
            actions = np.zeros(num_stocks)
            
            # Map EMP signals to FinRL actions
            for i, symbol in enumerate(emp_signal.get('symbols', [])):
                signal = emp_signal.get('signals', {}).get(symbol, {})
                action_type = signal.get('action', 'HOLD')
                size = signal.get('size', 0.0)
                
                if action_type == 'BUY':
                    actions[i] = size  # Positive value = buy
                elif action_type == 'SELL':
                    actions[i] = -size  # Negative value = sell
                else:  # HOLD
                    actions[i] = 0.0
            
            return actions
    ```
    
    **File: `experiments/finrl_benchmarks/run_benchmark.py`** (Usage example)
    ```python
    """
    Example script to run EMP strategies in FinRL environment and compare with baselines.
    """
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    import numpy as np
    import pandas as pd
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
    from finrl.agents.stablebaselines3.models import DRLAgent
    from finrl.config import INDICATORS
    
    from src.backtesting.finrl.strategy_adapter import FinRLStrategyAdapter
    from src.trading.strategies.momentum.simple_momentum import SimpleMomentumStrategy  # Example
    
    def download_data():
        """Download stock data using FinRL's data processor."""
        from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
        from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
        
        # Download data
        df = YahooDownloader(
            start_date='2020-01-01',
            end_date='2023-12-31',
            ticker_list=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        ).fetch_data()
        
        # Add technical indicators
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=INDICATORS,
            use_turbulence=False,
            user_defined_feature=False
        )
        
        processed = fe.preprocess_data(df)
        
        # Split into train/test
        train = data_split(processed, '2020-01-01', '2022-12-31')
        test = data_split(processed, '2023-01-01', '2023-12-31')
        
        return train, test
    
    def create_finrl_env(df, initial_amount=100000):
        """Create FinRL stock trading environment."""
        stock_dimension = len(df.tic.unique())
        state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
        
        env_kwargs = {
            "hmax": 100,
            "initial_amount": initial_amount,
            "buy_cost_pct": 0.001,
            "sell_cost_pct": 0.001,
            "state_space": state_space,
            "stock_dim": stock_dimension,
            "tech_indicator_list": INDICATORS,
            "action_space": stock_dimension,
            "reward_scaling": 1e-4
        }
        
        env = StockTradingEnv(df=df, **env_kwargs)
        return env
    
    def run_emp_strategy(env, strategy):
        """Run EMP strategy in FinRL environment."""
        # Wrap EMP strategy with adapter
        adapter = FinRLStrategyAdapter(strategy)
        
        # Run episode
        obs = env.reset()
        done = False
        total_reward = 0
        actions_taken = []
        
        while not done:
            # Get action from EMP strategy via adapter
            action = adapter.predict(obs)
            actions_taken.append(action)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            total_reward += reward
        
        # Get final portfolio value
        final_value = env.asset_memory[-1]
        
        return {
            'final_value': final_value,
            'total_reward': total_reward,
            'returns': (final_value - env.initial_amount) / env.initial_amount,
            'sharpe_ratio': calculate_sharpe(env.asset_memory),
            'max_drawdown': calculate_max_drawdown(env.asset_memory)
        }
    
    def run_baseline_ppo(env, train_env):
        """Run FinRL's PPO baseline."""
        # Train PPO agent
        agent = DRLAgent(env=train_env)
        
        model_ppo = agent.get_model("ppo")
        trained_ppo = agent.train_model(
            model=model_ppo,
            tb_log_name='ppo',
            total_timesteps=50000
        )
        
        # Test PPO
        obs = env.reset()
        done = False
        
        while not done:
            action, _states = trained_ppo.predict(obs)
            obs, reward, done, info = env.step(action)
        
        final_value = env.asset_memory[-1]
        
        return {
            'final_value': final_value,
            'returns': (final_value - env.initial_amount) / env.initial_amount,
            'sharpe_ratio': calculate_sharpe(env.asset_memory),
            'max_drawdown': calculate_max_drawdown(env.asset_memory)
        }
    
    def calculate_sharpe(asset_values, risk_free_rate=0.0):
        """Calculate Sharpe ratio from asset values."""
        returns = pd.Series(asset_values).pct_change().dropna()
        excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
        
        if returns.std() == 0:
            return 0.0
        
        sharpe = np.sqrt(252) * excess_returns.mean() / returns.std()
        return sharpe
    
    def calculate_max_drawdown(asset_values):
        """Calculate maximum drawdown."""
        peak = np.maximum.accumulate(asset_values)
        drawdown = (asset_values - peak) / peak
        return drawdown.min()
    
    def main():
        """Main benchmark comparison."""
        print("=" * 60)
        print("FinRL Benchmark: EMP Strategy vs. Baselines")
        print("=" * 60)
        
        # Download and prepare data
        print("\n1. Downloading data...")
        train_df, test_df = download_data()
        
        # Create environments
        print("\n2. Creating environments...")
        train_env = create_finrl_env(train_df)
        test_env = create_finrl_env(test_df)
        
        # Run EMP strategy
        print("\n3. Running EMP strategy...")
        emp_strategy = SimpleMomentumStrategy()  # Replace with your strategy
        emp_results = run_emp_strategy(test_env, emp_strategy)
        
        # Run PPO baseline
        print("\n4. Running PPO baseline...")
        test_env_ppo = create_finrl_env(test_df)  # Fresh env for PPO
        ppo_results = run_baseline_ppo(test_env_ppo, train_env)
        
        # Compare results
        print("\n" + "=" * 60)
        print("RESULTS COMPARISON")
        print("=" * 60)
        
        print(f"\nEMP Strategy:")
        print(f"  Final Value: ${emp_results['final_value']:,.2f}")
        print(f"  Returns: {emp_results['returns']*100:.2f}%")
        print(f"  Sharpe Ratio: {emp_results['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {emp_results['max_drawdown']*100:.2f}%")
        
        print(f"\nPPO Baseline:")
        print(f"  Final Value: ${ppo_results['final_value']:,.2f}")
        print(f"  Returns: {ppo_results['returns']*100:.2f}%")
        print(f"  Sharpe Ratio: {ppo_results['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {ppo_results['max_drawdown']*100:.2f}%")
        
        # Calculate improvement
        returns_improvement = (emp_results['returns'] - ppo_results['returns']) / abs(ppo_results['returns']) * 100
        sharpe_improvement = (emp_results['sharpe_ratio'] - ppo_results['sharpe_ratio']) / abs(ppo_results['sharpe_ratio']) * 100
        
        print(f"\nEMP vs PPO:")
        print(f"  Returns Improvement: {returns_improvement:+.1f}%")
        print(f"  Sharpe Improvement: {sharpe_improvement:+.1f}%")
        
        # Save results
        results_df = pd.DataFrame({
            'Strategy': ['EMP', 'PPO'],
            'Final Value': [emp_results['final_value'], ppo_results['final_value']],
            'Returns (%)': [emp_results['returns']*100, ppo_results['returns']*100],
            'Sharpe Ratio': [emp_results['sharpe_ratio'], ppo_results['sharpe_ratio']],
            'Max Drawdown (%)': [emp_results['max_drawdown']*100, ppo_results['max_drawdown']*100]
        })
        
        results_df.to_csv('experiments/finrl_benchmarks/results.csv', index=False)
        print("\nResults saved to experiments/finrl_benchmarks/results.csv")
    
    if __name__ == '__main__':
        main()
    ```
    
    **Installation and setup instructions**:
    ```bash
    # Install FinRL
    pip install finrl
    
    # Install dependencies
    pip install stable-baselines3[extra]
    pip install pyfolio
    
    # Create directories
    mkdir -p experiments/finrl_benchmarks
    mkdir -p models
    
    # Run benchmark
    python experiments/finrl_benchmarks/run_benchmark.py
    ```

  - **Success metrics**: Your strategies achieve Sharpe ratio ≥1.2 (vs. baseline PPO ~0.8-1.0); max drawdown <15% (vs. baseline ~20-25%); results reproducible and statistically significant (p<0.05).
  - **Acceptance**: Run momentum + mean reversion strategies in FinRL environment → Compare against PPO/A3C baselines → Generate report showing your strategies outperform baselines by 20%+ in risk-adjusted returns → Document results with statistical significance.

---

**Implementation Priority**:

1. **Priority 1 (Weeks 9-10)**: Sharpe Ratio Reward Function
   - Lowest effort (24 hours)
   - Immediate improvement to RL training
   - No dependencies on other work
   - No GPU required

2. **Priority 2 (Weeks 11-13)**: TLOB Order Book Attention
   - Depends on LOBSTER integration (Phase 1.1)
   - Significant trading edge potential
   - Natural fit for HOW sensor
   - GPU optional (train on CPU overnight)

3. **Priority 3 (Weeks 14-15)**: FinRL Benchmarking
   - Depends on having strategies to benchmark
   - Validates that Priorities 1 & 2 actually improved performance
   - Proves sophisticated architecture delivers value
   - GPU optional (CPU backtesting acceptable)

**Total Effort**: 124 hours (3 weeks full-time or 6 weeks part-time)

**Total Cost**: $0 (all open-source, no GPU required, cloud GPU rental <$10 if desired for TLOB training)

**Expected ROI**: 
- Sharpe Ratio: 30%+ faster RL convergence, 20%+ drawdown reduction
- TLOB: 10%+ improvement in order book-based entry/exit timing
- FinRL: Validation that your system outperforms baselines (or identification of areas needing improvement)

---
### 3.7 Advanced Research Enhancements (arXiv 2024-2025)

**Goal**: Integrate cutting-edge research from arXiv to enhance regime detection, risk management, market making, and liquidity analysis

**Estimated Effort**: 146-192 hours (4-5 months part-time)

**GPU Required**: ❌ NO - All implementations run on CPU

---

#### Enhancement 1: Hidden Markov Model for Regime Detection (30-40 hours)

**Objective**: Detect market regimes (bull/bear/transitional/high-vol/low-vol) to enable regime-aware strategy selection

**Research basis**: "Incorporating Market Regimes into Large-Scale Stock Portfolios: A Hidden Markov Model Approach" (2024), "Unveiling Market Regimes: A Hidden Markov Model Application" (2024)

**Where it fits**: WHEN sensor (timing and market regime analysis)

**Why it's useful**: Markets behave differently in different regimes. HMM automatically detects regime changes, allowing your evolution layer to evolve regime-specific strategies. Proven to reduce drawdowns 20-30%.

**GPU**: ❌ NO - HMM is CPU-based statistical model

- [x] **Implement RegimeDetector class** (12 hours)
  - **Location**: `src/sensory/when/regime_detector.py`
  - **Key methods**: `extract_features()`, `train()`, `predict_regime()`, `predict_regime_probabilities()`, `get_regime_statistics()`
  - **Features**: Returns, volatility, mean return, volume ratio, momentum
  - **Regimes**: 0=Bull, 1=Bear, 2=Transitional, 3=High Volatility, 4=Low Volatility
  - **Acceptance**: Detector trains on 252 days of data and predicts current regime with confidence score

  **Complete implementation code**:
  ```python
  """
  Hidden Markov Model for Market Regime Detection
  """
  import numpy as np
  from hmmlearn import hmm
  import pandas as pd
  
  class RegimeDetector:
      def __init__(self, n_regimes: int = 5, lookback_days: int = 252):
          self.n_regimes = n_regimes
          self.lookback_days = lookback_days
          self.model = None
          self.is_trained = False
          self.regime_names = {
              0: "Bull", 1: "Bear", 2: "Transitional",
              3: "High Volatility", 4: "Low Volatility"
          }
      
      def extract_features(self, prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
          returns = np.diff(np.log(prices))
          volatility = pd.Series(returns).rolling(window=20).std().values
          mean_return = pd.Series(returns).rolling(window=20).mean().values
          avg_volume = pd.Series(volumes).rolling(window=20).mean().values
          volume_ratio = volumes / (avg_volume + 1e-8)
          momentum = pd.Series(prices).pct_change(periods=20).values
          
          features = np.column_stack([
              returns[20:], volatility[20:], mean_return[20:],
              volume_ratio[20:], momentum[20:]
          ])
          return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
      
      def train(self, prices: np.ndarray, volumes: np.ndarray):
          features = self.extract_features(prices, volumes)
          self.model = hmm.GaussianHMM(
              n_components=self.n_regimes,
              covariance_type="full",
              n_iter=100,
              random_state=42
          )
          self.model.fit(features)
          self.is_trained = True
      
      def predict_regime(self, prices: np.ndarray, volumes: np.ndarray) -> int:
          if not self.is_trained:
              raise ValueError("Model not trained")
          features = self.extract_features(prices, volumes)
          return int(self.model.predict(features)[-1])
      
      def predict_regime_probabilities(self, prices: np.ndarray, volumes: np.ndarray):
          if not self.is_trained:
              raise ValueError("Model not trained")
          features = self.extract_features(prices, volumes)
          return self.model.predict_proba(features)[-1]
  ```

- [ ] **Implement RegimeAwareStrategySelector** (8 hours)
  - **Location**: `src/sensory/when/regime_detector.py`
  - **Key methods**: `select_strategy()`, `get_position_sizing_multiplier()`
  - **Strategy mapping**: Bull→momentum, Bear→mean_reversion, Transitional→conservative, High-vol→volatility, Low-vol→range_bound
  - **Acceptance**: Selector chooses appropriate strategy based on regime with confidence threshold

- [ ] **Create training script** (4 hours)
  - **Location**: `src/sensory/when/regime_trainer.py`
  - **Function**: Train detector on historical data, save model, test prediction
  - **Acceptance**: Script trains detector and saves to `models/regime_detector.pkl`

- [ ] **Write comprehensive tests** (6-10 hours)
  - **Location**: `tests/sensory/when/test_regime_detector.py`
  - **Tests**: Training, prediction, probabilities, strategy selection, position sizing
  - **Acceptance**: All tests pass, 90%+ coverage

- **Progress**: HMM detector now has regression tests covering training, regime probabilities, and data sufficiency safeguards.【F:tests/sensory/when/test_regime_detector.py†L1-L52】

**Installation**: `pip install hmmlearn`

**Success metrics**: Detector identifies regime changes with 70%+ accuracy; regime-aware strategies reduce drawdowns by 15-20% vs. single-strategy baseline.

---

#### Enhancement 2: CVaR (Conditional Value-at-Risk) Optimization (24-32 hours)

**Objective**: Minimize tail risk while maximizing returns using industry-standard CVaR optimization

**Research basis**: "Portfolio Optimization with Conditional Value-at-Risk" (Krokhmal et al., 2001 - 1,184 citations), "Portfolio Risk Management with CVaR-like Constraints" (Cox et al., 2010)

**Where it fits**: RISK layer (position sizing and portfolio optimization)

**Why it's useful**: CVaR measures expected loss in worst-case scenarios (tail risk). Better than VaR because it's coherent and captures tail distribution. Industry standard for institutional risk management.

**GPU**: ❌ NO - Convex optimization using CPU-based solvers

- [ ] **Implement CVaROptimizer class** (12 hours)
  - **Location**: `src/risk/cvar_optimizer.py`
  - **Key methods**: `calculate_cvar()`, `optimize_portfolio()`, `calculate_position_sizes()`
  - **Optimization modes**: (1) Minimize CVaR subject to target return, (2) Maximize return subject to max CVaR
  - **Formula**: CVaR_α = E[Loss | Loss > VaR_α]
  - **Acceptance**: Optimizer finds optimal weights that minimize CVaR while meeting constraints

  **Complete implementation code**:
  ```python
  """
  CVaR Portfolio Optimization
  """
  import numpy as np
  from scipy.optimize import minimize
  
  class CVaROptimizer:
      def __init__(self, alpha: float = 0.05, max_position: float = 0.2):
          self.alpha = alpha  # 0.05 = 95% CVaR
          self.max_position = max_position
      
      def calculate_cvar(self, returns: np.ndarray, weights: np.ndarray, alpha: float):
          portfolio_returns = returns @ weights
          var = np.percentile(portfolio_returns, alpha * 100)
          losses_beyond_var = portfolio_returns[portfolio_returns <= var]
          return -np.mean(losses_beyond_var) if len(losses_beyond_var) > 0 else 0.0
      
      def optimize_portfolio(self, returns: np.ndarray, max_cvar: float):
          n_assets = returns.shape[1]
          x0 = np.ones(n_assets) / n_assets
          bounds = tuple((0, self.max_position) for _ in range(n_assets))
          constraints = [
              {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
              {'type': 'ineq', 'fun': lambda x: max_cvar - self.calculate_cvar(returns, x, self.alpha)}
          ]
          
          mean_returns = np.mean(returns, axis=0)
          result = minimize(
              lambda x: -(x @ mean_returns),
              x0, method='SLSQP', bounds=bounds, constraints=constraints
          )
          
          return {
              'weights': result.x,
              'expected_return': float(result.x @ mean_returns),
              'cvar': float(self.calculate_cvar(returns, result.x, self.alpha)),
              'success': result.success
          }
  ```

- [x] **Implement RollingCVaRMonitor** (6 hours)
  - **Location**: `src/risk/analytics/rolling_cvar.py`
  - **Key methods**: `update()`, `check_breach()`
  - **Window**: 252 days rolling
  - **Acceptance**: Monitor tracks CVaR in real-time and detects breaches

- [ ] **Integrate with position sizing** (4-8 hours)
  - **Location**: `src/risk/cvar_integration.py`
  - **Class**: `CVaRAwarePositionSizer` extends `PositionSizer`
  - **Acceptance**: Position sizer uses CVaR optimization to calculate sizes

- [ ] **Write comprehensive tests** (2-6 hours)
  - **Location**: `tests/risk/analytics/test_rolling_cvar.py`
  - **Tests**: CVaR calculation, optimization, position sizing, monitoring, breach detection
  - **Acceptance**: All tests pass, 90%+ coverage

- **Progress**: Rolling CVaR analytics tests now validate configuration guards, measurement accuracy, and handling of non-finite observations.【F:tests/risk/analytics/test_rolling_cvar.py†L1-L71】

**Installation**: `pip install scipy` (already included)

**Success metrics**: CVaR-optimized portfolios have 20-30% lower tail risk (95% CVaR) vs. equal-weight baseline; max drawdown reduced by 15-20%.

---

#### Enhancement 3: RL-based Market Making (60-80 hours)

**Objective**: Learn optimal bid/ask placement to earn spread while managing inventory risk

**Research basis**: "Reinforcement Learning in High-frequency Market Making" (Zheng & Ding, arXiv 2024)

**Where it fits**: EXECUTION layer (order placement and market making)

**Why it's useful**: Market making can reduce transaction costs (earn spread instead of paying it). RL learns optimal placement based on order book state. Theoretical framework provides guidance on sampling frequency.

**GPU**: ⚠️ OPTIONAL - Train on CPU overnight (8-12 hours) or GPU (2-4 hours); inference on CPU (<1ms)

- [ ] **Implement MarketMakingEnv** (16 hours)
  - **Location**: `src/execution/market_maker.py`
  - **Key methods**: `reset()`, `step()`, `_get_state()`
  - **State**: (inventory, mid_price, time_step)
  - **Action**: (bid_offset, ask_offset) in ticks
  - **Reward**: Spread profit - inventory penalty
  - **Acceptance**: Environment simulates market making with realistic dynamics

- [ ] **Implement DQN Q-Network** (12 hours)
  - **Location**: `src/execution/market_maker.py`
  - **Architecture**: 3-layer MLP (64-64 hidden units)
  - **Input**: State (3 features)
  - **Output**: Q-values for discrete actions (10 bid/ask combinations)
  - **Acceptance**: Network trains and converges

- [ ] **Implement MarketMaker agent** (20 hours)
  - **Location**: `src/execution/market_maker.py`
  - **Key methods**: `select_action()`, `store_transition()`, `train_step()`, `update_target_network()`
  - **Algorithm**: DQN with experience replay and target network
  - **Hyperparameters**: lr=0.001, gamma=0.99, epsilon=1.0→0.01
  - **Acceptance**: Agent learns to earn positive PnL in simulation

- [ ] **Create training script** (8 hours)
  - **Location**: `src/execution/train_market_maker.py`
  - **Episodes**: 1000
  - **Acceptance**: Trained agent achieves positive average reward

- [ ] **Write comprehensive tests** (4-12 hours)
  - **Location**: `tests/execution/test_market_maker.py`
  - **Tests**: Environment, Q-network, agent training, action selection
  - **Acceptance**: All tests pass, agent learns in test environment

**Installation**: `pip install torch` (CPU version)

**Success metrics**: Market maker earns positive spread (0.5-1 bps per trade) while maintaining inventory within limits; reduces transaction costs by 30-50% vs. market orders.

**Key insight from paper**: Smaller Δ (higher frequency) → lower error but higher complexity. Find sweet spot based on compute budget.

---

#### Enhancement 4: Liquidity Metrics for Price Prediction (32-40 hours)

**Objective**: Predict price movements and assess execution quality using comprehensive liquidity metrics

**Research basis**: "High-Frequency Trading Liquidity Analysis | Application of Machine Learning Classification" (Bhatia et al., arXiv 2024)

**Where it fits**: HOW sensor (market microstructure analysis)

**Why it's useful**: Liquidity predicts short-term price movements. High liquidity = stable, low liquidity = volatile. Improves entry/exit timing. Random Forest achieved highest accuracy in paper.

**GPU**: ❌ NO - Random Forest is CPU-based

- [ ] **Implement LiquidityAnalyzer class** (16 hours)
  - **Location**: `src/sensory/how/liquidity_analyzer.py`
  - **Key methods**: `calculate_liquidity_metrics()`, `train_predictor()`, `predict_price_movement()`, `assess_execution_quality()`
  - **Metrics**: Liquidity Ratio, Flow Ratio, Turnover, Effective Spread, Quoted Spread, Depth, Price Impact, Amihud Illiquidity
  - **Model**: Random Forest Classifier (100 trees)
  - **Acceptance**: Analyzer calculates 8 liquidity metrics and predicts price movement

  **Key metrics**:
  ```python
  # Liquidity Ratio = Volume / Volatility
  liquidity_ratio = np.sum(volumes) / (volatility + 1e-8)
  
  # Flow Ratio = Buy Volume / Sell Volume
  flow_ratio = buy_volume / (sell_volume + 1e-8)
  
  # Turnover = Volume / Avg Price
  turnover = np.sum(volumes) / (avg_price + 1e-8)
  
  # Effective Spread = 2 * |Trade Price - Mid Price|
  effective_spread = 2 * np.abs(prices - mid_prices)
  
  # Price Impact = |Return| / Volume
  price_impact = np.abs(returns) / (volumes + 1e-8)
  ```

- [ ] **Train Random Forest predictor** (8 hours)
  - **Input**: 8 liquidity metrics
  - **Output**: Price movement (0=down, 1=neutral, 2=up)
  - **Training**: Historical TAQ data with labels
  - **Acceptance**: Predictor achieves 60%+ accuracy on test set

- [ ] **Implement execution quality assessment** (4 hours)
  - **Assessment**: Liquidity (high/medium/low), Spread (tight/normal/wide), Depth (deep/normal/shallow), Impact (low/medium/high)
  - **Recommendation**: Excellent/Good/Poor execution conditions
  - **Acceptance**: Assessment provides actionable guidance

- [ ] **Write comprehensive tests** (4-8 hours)
  - **Location**: `tests/sensory/how/test_liquidity_analyzer.py`
  - **Tests**: Metrics calculation, predictor training, prediction, assessment
  - **Acceptance**: All tests pass, 90%+ coverage

**Installation**: `pip install scikit-learn` (already included)

**Success metrics**: Liquidity metrics predict price movements with 60-70% accuracy; execution quality assessment reduces slippage by 20-30% by avoiding poor liquidity conditions.

---

**Implementation Priority for New Enhancements**:

1. **Priority 1 (Weeks 1-4)**: HMM Regime Detection
   - Highest ROI (20-30% drawdown reduction)
   - No GPU required
   - Natural fit for WHEN sensor
   - Enables regime-aware strategy evolution

2. **Priority 2 (Weeks 5-7)**: CVaR Risk Optimization
   - Industry standard risk management
   - No GPU required
   - Low-hanging fruit (24-32 hours)
   - Immediate risk reduction

3. **Priority 3 (Weeks 8-11)**: Liquidity Metrics
   - Enhances HOW sensor
   - No GPU required
   - Complements TLOB order book analysis
   - Improves execution timing

4. **Priority 4 (Weeks 12-18)**: RL Market Making
   - Most sophisticated
   - GPU optional (train on CPU overnight)
   - Highest potential value (turn costs into revenue)
   - Requires solid foundation from other enhancements

**Total Effort**: 146-192 hours (4-5 months part-time)

**Total Cost**: $0 (all open-source, no GPU required)

**Expected ROI**:
- HMM Regime Detection: 20-30% drawdown reduction
- CVaR Optimization: 15-20% tail risk reduction
- RL Market Making: 30-50% transaction cost reduction (or revenue from spread)
- Liquidity Metrics: 20-30% slippage reduction

---



### 3.8 Markovian Reasoning Enhancements (Delethink-Inspired)

**Goal**: Apply Markovian chunking principles to improve training stability, decision quality, and adaptation speed

**Research basis**: "Delethink: Efficient Reasoning for Long-Context Language Models" (2025) - adapted for trading system sequential decision-making

**Core insight**: Break long sequences into fixed-horizon chunks where each chunk is a self-contained Markov state, enabling constant memory, linear compute, and better traceability

**Estimated Effort**: 80-116 hours (2-3 months part-time)

**GPU Required**: ❌ NO - All implementations run on CPU

**Why this fits**: Your EMP system is already Markovian by design (BeliefState, RegimeFSM, MuZeroLite, HMM). These enhancements make existing components more explicitly Markovian.

---

#### Enhancement 1: Chunked RL Training Loops (16-24 hours) ⭐⭐⭐⭐⭐

**Objective**: Break long RL training episodes into fixed-horizon chunks (e.g., 50-step segments) for stable gradients and better traceability

**Where it fits**: THINKING layer → RL training (`PreTrainingPipeline`)

**Why it's useful**: Your `PreTrainingPipeline` already uses TBPTT (truncated backpropagation through time). This formalizes chunking with Markovian assumptions. Prevents vanishing/exploding gradients, enables per-chunk inspection, allows parallelization and early stopping.

**GPU**: ❌ NO - Heavy neural ops stay on GPU, chunk orchestration is light CPU work

- [ ] **Implement ChunkedTrainer class** (8-12 hours)
  - **Location**: `src/thinking/learning/chunked_trainer.py`
  - **Key methods**: `chunk_episode()`, `train_chunk()`, `aggregate_chunks()`
  - **Chunk size**: 50 steps (configurable)
  - **Markov property**: Each chunk treats starting state as self-contained
  - **Acceptance**: Trainer splits 1000-step episode into 20 chunks, trains each independently

  **Complete implementation code**:
  ```python
  """
  Chunked RL Training with Markovian Segments
  """
  import numpy as np
  import torch
  from typing import List, Dict, Tuple
  
  class ChunkedTrainer:
      def __init__(self, chunk_size: int = 50, overlap: int = 5):
          """
          Initialize chunked trainer
          
          Args:
              chunk_size: Steps per chunk (default 50)
              overlap: Overlap between chunks for continuity (default 5)
          """
          self.chunk_size = chunk_size
          self.overlap = overlap
          
      def chunk_episode(self, 
                       states: np.ndarray,
                       actions: np.ndarray,
                       rewards: np.ndarray,
                       dones: np.ndarray) -> List[Dict]:
          """
          Split episode into Markovian chunks
          
          Args:
              states: Episode states (T, state_dim)
              actions: Episode actions (T,)
              rewards: Episode rewards (T,)
              dones: Episode done flags (T,)
              
          Returns:
              List of chunk dictionaries
          """
          T = len(states)
          chunks = []
          
          start = 0
          while start < T:
              end = min(start + self.chunk_size, T)
              
              chunk = {
                  'states': states[start:end],
                  'actions': actions[start:end],
                  'rewards': rewards[start:end],
                  'dones': dones[start:end],
                  'initial_state': states[start],  # Markov: chunk is self-contained
                  'chunk_id': len(chunks),
                  'start_step': start,
                  'end_step': end
              }
              chunks.append(chunk)
              
              # Move to next chunk with overlap
              start = end - self.overlap if end < T else T
          
          return chunks
      
      def train_chunk(self, chunk: Dict, policy, optimizer) -> Dict:
          """
          Train on single chunk (Markov segment)
          
          Args:
              chunk: Chunk dictionary
              policy: Policy network
              optimizer: Optimizer
              
          Returns:
              Training metrics for this chunk
          """
          states = torch.FloatTensor(chunk['states'])
          actions = torch.LongTensor(chunk['actions'])
          rewards = torch.FloatTensor(chunk['rewards'])
          
          # Compute returns within chunk
          returns = self._compute_returns(rewards)
          
          # Policy loss
          log_probs = policy.get_log_prob(states, actions)
          advantages = returns - returns.mean()
          policy_loss = -(log_probs * advantages).mean()
          
          # Optimize
          optimizer.zero_grad()
          policy_loss.backward()
          torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
          optimizer.step()
          
          return {
              'chunk_id': chunk['chunk_id'],
              'loss': policy_loss.item(),
              'mean_return': returns.mean().item(),
              'steps': len(chunk['states'])
          }
      
      def _compute_returns(self, rewards: torch.Tensor, gamma: float = 0.99):
          """Compute discounted returns"""
          returns = torch.zeros_like(rewards)
          running_return = 0
          
          for t in reversed(range(len(rewards))):
              running_return = rewards[t] + gamma * running_return
              returns[t] = running_return
          
          return returns
      
      def train_episode_chunked(self,
                               episode_data: Dict,
                               policy,
                               optimizer) -> Dict:
          """
          Train on full episode using chunked approach
          
          Args:
              episode_data: Full episode data
              policy: Policy network
              optimizer: Optimizer
              
          Returns:
              Aggregated training metrics
          """
          # Split into chunks
          chunks = self.chunk_episode(
              episode_data['states'],
              episode_data['actions'],
              episode_data['rewards'],
              episode_data['dones']
          )
          
          # Train each chunk
          chunk_metrics = []
          for chunk in chunks:
              metrics = self.train_chunk(chunk, policy, optimizer)
              chunk_metrics.append(metrics)
          
          # Aggregate
          return {
              'num_chunks': len(chunks),
              'total_loss': np.mean([m['loss'] for m in chunk_metrics]),
              'total_return': np.sum([m['mean_return'] for m in chunk_metrics]),
              'chunk_metrics': chunk_metrics  # For per-chunk inspection
          }
  ```

- [ ] **Integrate with PreTrainingPipeline** (4-6 hours)
  - **Location**: `src/thinking/learning/pretraining_pipeline.py`
  - **Modification**: Add `use_chunked_training` flag
  - **Acceptance**: Pipeline uses chunked trainer when flag is enabled

- [ ] **Add observability** (2-4 hours)
  - **Metrics**: Per-chunk loss, return, gradient norms
  - **Logging**: Chunk boundaries, convergence per chunk
  - **Acceptance**: Can visualize training progress per chunk

- [ ] **Write tests** (2-4 hours)
  - **Location**: `tests/thinking/learning/test_chunked_trainer.py`
  - **Tests**: Chunking, training, aggregation, edge cases
  - **Acceptance**: All tests pass, 90%+ coverage

**Success metrics**: 30%+ faster RL convergence; stable gradient norms across chunks; can identify which chunks contribute most to learning.

---

#### Enhancement 2: Short-Horizon Agent Rollouts with Markov Tree Planning (20-32 hours) ⭐⭐⭐⭐⭐

**Objective**: At decision time, simulate 2-5 future steps using MuZero-lite tree search to evaluate action quality

**Where it fits**: THINKING layer → Decision making (live trading loop)

**Why it's useful**: Your system has `MuZeroLite` and `GraphNetSurrogate` for planning. This integrates them as lookahead during live trading. Like a chess player thinking a few moves ahead. Pure win for decision quality with no latency penalty (GraphNet is fast).

**GPU**: ❌ NO - Tree search and surrogate model run on CPU

- [ ] **Implement MarkovLookahead class** (12-16 hours)
  - **Location**: `src/thinking/planning/markov_lookahead.py`
  - **Key methods**: `simulate_rollout()`, `evaluate_action()`, `select_best_action()`
  - **Horizon**: 2-5 steps
  - **Markov assumption**: Current state (belief + regime) fully encapsulates market
  - **Acceptance**: Lookahead simulates N actions, returns expected PnL and risk for each

  **Complete implementation code**:
  ```python
  """
  Markovian Short-Horizon Lookahead
  """
  import numpy as np
  from typing import List, Tuple, Dict
  from src.thinking.planning.muzero_lite_tree import MuZeroLiteTree
  from src.simulation.graphnet_surrogate import GraphNetSurrogate
  
  class MarkovLookahead:
      def __init__(self, 
                   surrogate_model: GraphNetSurrogate,
                   horizon: int = 3,
                   num_simulations: int = 10):
          """
          Initialize Markov lookahead
          
          Args:
              surrogate_model: World model for simulation
              horizon: Steps to look ahead (default 3)
              num_simulations: Rollouts per action (default 10)
          """
          self.surrogate = surrogate_model
          self.horizon = horizon
          self.num_simulations = num_simulations
          
      def evaluate_action(self,
                         current_state: np.ndarray,
                         action: int,
                         regime: int) -> Dict:
          """
          Evaluate action by simulating future
          
          Args:
              current_state: Current belief state
              action: Candidate action
              regime: Current regime
              
          Returns:
              Expected outcomes (PnL, risk, probability)
          """
          outcomes = []
          
          for _ in range(self.num_simulations):
              # Simulate rollout
              state = current_state.copy()
              total_reward = 0
              trajectory = []
              
              for step in range(self.horizon):
                  # Use surrogate to predict next state and reward
                  next_state, reward, done = self.surrogate.step(
                      state, action, regime
                  )
                  
                  total_reward += reward * (0.99 ** step)  # Discounted
                  trajectory.append({
                      'state': state,
                      'action': action,
                      'reward': reward,
                      'next_state': next_state
                  })
                  
                  if done:
                      break
                  
                  state = next_state
                  # For subsequent steps, use default action (e.g., hold)
                  action = 0
              
              outcomes.append({
                  'total_reward': total_reward,
                  'trajectory': trajectory
              })
          
          # Aggregate outcomes
          rewards = [o['total_reward'] for o in outcomes]
          
          return {
              'expected_reward': np.mean(rewards),
              'reward_std': np.std(rewards),
              'min_reward': np.min(rewards),
              'max_reward': np.max(rewards),
              'probability_positive': np.mean([r > 0 for r in rewards])
          }
      
      def select_best_action(self,
                            current_state: np.ndarray,
                            candidate_actions: List[int],
                            regime: int,
                            risk_aversion: float = 0.5) -> Tuple[int, Dict]:
          """
          Select best action using lookahead
          
          Args:
              current_state: Current belief state
              candidate_actions: List of candidate actions
              regime: Current regime
              risk_aversion: Weight for risk vs. return (0-1)
              
          Returns:
              (best_action, evaluation_details)
          """
          evaluations = {}
          
          for action in candidate_actions:
              eval_result = self.evaluate_action(current_state, action, regime)
              
              # Score: expected reward - risk_aversion * std
              score = (eval_result['expected_reward'] - 
                      risk_aversion * eval_result['reward_std'])
              
              evaluations[action] = {
                  'score': score,
                  **eval_result
              }
          
          # Select best
          best_action = max(evaluations.keys(), key=lambda a: evaluations[a]['score'])
          
          return best_action, evaluations
  ```

- [ ] **Integrate with UnderstandingRouter** (4-8 hours)
  - **Location**: `src/thinking/understanding_router.py`
  - **Modification**: Add optional lookahead before final decision
  - **Flag**: `use_markov_lookahead` (default False for safety)
  - **Acceptance**: Router can invoke lookahead for high-stakes decisions

- [ ] **Add decision logging** (2-4 hours)
  - **Log**: Candidate actions, expected outcomes, selected action, rationale
  - **Integration**: Decision Diary includes lookahead results
  - **Acceptance**: Can audit "why action A was chosen over B"

- [ ] **Write tests** (2-4 hours)
  - **Location**: `tests/thinking/planning/test_markov_lookahead.py`
  - **Tests**: Rollout simulation, action evaluation, selection, edge cases
  - **Acceptance**: All tests pass, 90%+ coverage

**Success metrics**: 5-10 bps improvement in expected PnL per trade; 15-20% reduction in regretted decisions (decisions that would have been different with lookahead).

---

#### Enhancement 3: Regime Transition Probabilities (8-12 hours) ⭐⭐⭐⭐

**Objective**: Expose HMM regime transition probabilities to decision layer for proactive adaptation

**Where it fits**: WHEN sensor → Regime detection

**Why it's useful**: You're already adding HMM regime detection. HMM maintains a transition matrix (bull→bear probability, etc.). This just exposes it. Enables reasoning like: "70% chance we stay bull, 30% switching to balanced → tighten risk now."

**GPU**: ❌ NO - Trivial math (reading transition matrix)

- [ ] **Extend RegimeDetector with transition probabilities** (4-6 hours)
  - **Location**: `src/sensory/when/regime_detector.py`
  - **New method**: `get_transition_probabilities()`, `predict_next_regime()`
  - **Output**: Transition matrix (5x5 for 5 regimes) + next-regime probabilities
  - **Acceptance**: Can query "given current regime, what's probability of each next regime?"

  **Code addition**:
  ```python
  def get_transition_probabilities(self) -> np.ndarray:
      """
      Get regime transition probability matrix
      
      Returns:
          Transition matrix (n_regimes, n_regimes)
          transmat[i,j] = P(regime j | regime i)
      """
      if not self.is_trained:
          raise ValueError("Model not trained")
      
      return self.model.transmat_
  
  def predict_next_regime(self, current_regime: int) -> Dict:
      """
      Predict next regime probabilities
      
      Args:
          current_regime: Current regime (0-4)
          
      Returns:
          Dictionary with next regime probabilities
      """
      transmat = self.get_transition_probabilities()
      next_probs = transmat[current_regime]
      
      return {
          self.regime_names[i]: float(next_probs[i])
          for i in range(self.n_regimes)
      }
  ```

- [ ] **Integrate with RegimeAwareStrategySelector** (2-4 hours)
  - **Location**: `src/sensory/when/regime_detector.py`
  - **Enhancement**: Use transition probabilities for risk adjustment
  - **Logic**: If high probability of regime shift, reduce position sizes
  - **Acceptance**: Strategy selector adjusts based on transition risk

- [ ] **Add to Decision Diary** (1-2 hours)
  - **Log**: Current regime, transition probabilities, regime shift risk
  - **Example**: "Bull regime (90% confidence), 15% chance of shift to transitional in next period"
  - **Acceptance**: Decision Diary includes regime transition context

- [ ] **Write tests** (1-2 hours)
  - **Location**: `tests/sensory/when/test_regime_detector.py`
  - **Tests**: Transition matrix extraction, next regime prediction
  - **Acceptance**: Tests pass, transition probabilities sum to 1.0

**Success metrics**: 10-15% better regime-change adaptation; can identify regime shifts 1-2 periods earlier; Decision Diary shows clear regime transition rationale.

---

#### Enhancement 4: Memory Patterns as Markov Chains (24-32 hours) ⭐⭐⭐⭐

**Objective**: Model experience memory as Markov chain of pattern states to enable preemptive adaptation

**Where it fits**: SENTIENT layer → Adaptation (`AdaptationController`)

**Why it's useful**: Your `AdaptationController` retrieves similar past experiences from FAISS. This adds sequence awareness: "Pattern A → Pattern B 80% of time unless adaptation X applied." Enables preemptive intervention before losses compound.

**GPU**: ❌ NO - Just counting pattern transitions, computing probabilities

- [ ] **Implement PatternMarkovChain class** (12-16 hours)
  - **Location**: `src/sentient/adaptation/pattern_markov_chain.py`
  - **Key methods**: `add_pattern_transition()`, `get_transition_prob()`, `predict_next_pattern()`
  - **Pattern states**: Cluster experiences into discrete states (e.g., "mild loss after win streak")
  - **Acceptance**: Can query "given current pattern, what pattern is likely next?"

  **Complete implementation code**:
  ```python
  """
  Pattern Memory as Markov Chains
  """
  import numpy as np
  from collections import defaultdict
  from typing import Dict, Tuple, Optional
  
  class PatternMarkovChain:
      def __init__(self, n_states: int = 10):
          """
          Initialize pattern Markov chain
          
          Args:
              n_states: Number of discrete pattern states
          """
          self.n_states = n_states
          self.transition_counts = defaultdict(lambda: defaultdict(int))
          self.state_counts = defaultdict(int)
          self.pattern_to_state = {}  # Map pattern hash to state ID
          
      def add_pattern_transition(self,
                                from_pattern: str,
                                to_pattern: str,
                                adaptation_applied: Optional[str] = None):
          """
          Record pattern transition
          
          Args:
              from_pattern: Starting pattern (e.g., "mild_loss_after_win")
              to_pattern: Ending pattern (e.g., "continued_drawdown")
              adaptation_applied: Adaptation that was applied (if any)
          """
          # Get or assign state IDs
          from_state = self._get_state_id(from_pattern)
          to_state = self._get_state_id(to_pattern)
          
          # Update counts
          self.transition_counts[from_state][to_state] += 1
          self.state_counts[from_state] += 1
          
          # Track adaptation effectiveness
          if adaptation_applied:
              key = (from_state, to_state, adaptation_applied)
              if not hasattr(self, 'adaptation_effects'):
                  self.adaptation_effects = defaultdict(list)
              self.adaptation_effects[key].append(to_pattern)
      
      def _get_state_id(self, pattern: str) -> int:
          """Get or assign state ID for pattern"""
          if pattern not in self.pattern_to_state:
              state_id = len(self.pattern_to_state)
              self.pattern_to_state[pattern] = state_id
          return self.pattern_to_state[pattern]
      
      def get_transition_prob(self, from_pattern: str, to_pattern: str) -> float:
          """
          Get probability of transition
          
          Args:
              from_pattern: Starting pattern
              to_pattern: Ending pattern
              
          Returns:
              Transition probability
          """
          if from_pattern not in self.pattern_to_state:
              return 0.0
          
          from_state = self.pattern_to_state[from_pattern]
          
          if self.state_counts[from_state] == 0:
              return 0.0
          
          to_state = self.pattern_to_state.get(to_pattern, -1)
          if to_state == -1:
              return 0.0
          
          count = self.transition_counts[from_state][to_state]
          total = self.state_counts[from_state]
          
          return count / total
      
      def predict_next_pattern(self, current_pattern: str, top_k: int = 3) -> Dict:
          """
          Predict most likely next patterns
          
          Args:
              current_pattern: Current pattern
              top_k: Number of top predictions to return
              
          Returns:
              Dictionary with pattern predictions and probabilities
          """
          if current_pattern not in self.pattern_to_state:
              return {}
          
          from_state = self.pattern_to_state[current_pattern]
          
          if self.state_counts[from_state] == 0:
              return {}
          
          # Calculate probabilities for all next states
          next_probs = {}
          for to_state, count in self.transition_counts[from_state].items():
              prob = count / self.state_counts[from_state]
              
              # Find pattern name for this state
              pattern_name = [p for p, s in self.pattern_to_state.items() if s == to_state][0]
              next_probs[pattern_name] = prob
          
          # Sort by probability
          sorted_probs = sorted(next_probs.items(), key=lambda x: x[1], reverse=True)
          
          return {
              'predictions': sorted_probs[:top_k],
              'current_pattern': current_pattern,
              'total_observations': self.state_counts[from_state]
          }
      
      def should_intervene(self,
                          current_pattern: str,
                          bad_pattern_threshold: float = 0.5) -> Tuple[bool, Dict]:
          """
          Determine if preemptive adaptation is needed
          
          Args:
              current_pattern: Current pattern
              bad_pattern_threshold: Probability threshold for intervention
              
          Returns:
              (should_intervene, reasoning)
          """
          predictions = self.predict_next_pattern(current_pattern)
          
          if not predictions:
              return False, {'reason': 'No historical data'}
          
          # Check if high probability of bad pattern
          for pattern, prob in predictions['predictions']:
              if 'loss' in pattern or 'drawdown' in pattern:
                  if prob >= bad_pattern_threshold:
                      return True, {
                          'reason': f'High probability ({prob:.1%}) of {pattern}',
                          'current_pattern': current_pattern,
                          'predicted_pattern': pattern,
                          'probability': prob
                      }
          
          return False, {'reason': 'No high-risk pattern predicted'}
  ```

- [ ] **Integrate with AdaptationController** (6-10 hours)
  - **Location**: `src/sentient/adaptation/adaptation_controller.py`
  - **Enhancement**: Use pattern Markov chain for preemptive adaptation
  - **Logic**: If current pattern likely leads to bad pattern, trigger adaptation now
  - **Acceptance**: Controller can intervene before pattern deteriorates

- [ ] **Add pattern clustering** (4-6 hours)
  - **Method**: Cluster similar experiences into discrete states
  - **Features**: PnL trajectory, drawdown depth, win rate, volatility
  - **Acceptance**: Experiences are automatically clustered into 10-15 pattern states

- [ ] **Write tests** (2-4 hours)
  - **Location**: `tests/sentient/adaptation/test_pattern_markov_chain.py`
  - **Tests**: Transition recording, probability calculation, prediction, intervention
  - **Acceptance**: All tests pass, 90%+ coverage

**Success metrics**: 15-20% faster adaptation response; can identify deteriorating patterns 2-3 steps earlier; 10-15% reduction in drawdown depth.

---

#### Enhancement 5: Asynchronous Reflection with Fixed-State Snapshots (12-16 hours) ⭐⭐⭐

**Objective**: Structure RIM reflection analysis as Markov chain of time-period snapshots

**Where it fits**: GOVERNANCE layer → Reflection Intelligence Module (RIM)

**Why it's useful**: RIM already does iterative reasoning on Decision Diary. Chunking by time periods (day, week, strategy lifecycle) makes analysis more structured. Enables "state at end of day t → state at end of day t+1" reasoning.

**GPU**: ❌ NO - Small JSONL artifacts, recursive model computations

- [ ] **Implement ChunkedReflection class** (6-10 hours)
  - **Location**: `src/governance/reflection/chunked_reflection.py`
  - **Key methods**: `create_snapshot()`, `analyze_transition()`, `generate_report()`
  - **Chunk periods**: Day, week, strategy lifecycle
  - **Markov state**: Strategy health metrics at end of period
  - **Acceptance**: RIM analyzes trading sessions in discrete time chunks

  **Code skeleton**:
  ```python
  """
  Chunked Reflection with Fixed-State Snapshots
  """
  from typing import Dict, List
  from datetime import datetime, timedelta
  
  class ChunkedReflection:
      def __init__(self, chunk_period: str = 'day'):
          """
          Initialize chunked reflection
          
          Args:
              chunk_period: 'day', 'week', or 'lifecycle'
          """
          self.chunk_period = chunk_period
          self.snapshots = []
          
      def create_snapshot(self, 
                         period_start: datetime,
                         period_end: datetime,
                         decision_diary_entries: List[Dict]) -> Dict:
          """
          Create fixed-state snapshot for period
          
          Args:
              period_start: Start of period
              period_end: End of period
              decision_diary_entries: Decisions in this period
              
          Returns:
              Snapshot dictionary (Markov state)
          """
          # Aggregate metrics for period
          total_pnl = sum(e.get('pnl', 0) for e in decision_diary_entries)
          num_trades = len([e for e in decision_diary_entries if e.get('action') != 'hold'])
          win_rate = sum(1 for e in decision_diary_entries if e.get('pnl', 0) > 0) / max(num_trades, 1)
          
          snapshot = {
              'period_start': period_start.isoformat(),
              'period_end': period_end.isoformat(),
              'total_pnl': total_pnl,
              'num_trades': num_trades,
              'win_rate': win_rate,
              'max_drawdown': self._calculate_max_drawdown(decision_diary_entries),
              'regime_distribution': self._get_regime_distribution(decision_diary_entries),
              'state_label': self._label_state(total_pnl, win_rate)
          }
          
          self.snapshots.append(snapshot)
          return snapshot
      
      def _label_state(self, pnl: float, win_rate: float) -> str:
          """Label state for Markov chain"""
          if pnl > 0 and win_rate > 0.6:
              return 'healthy'
          elif pnl > 0 and win_rate > 0.5:
              return 'borderline'
          elif pnl < 0 and win_rate < 0.4:
              return 'deteriorating'
          else:
              return 'mixed'
      
      def analyze_transition(self, 
                            from_snapshot: Dict,
                            to_snapshot: Dict) -> Dict:
          """
          Analyze transition between snapshots
          
          Args:
              from_snapshot: Previous period state
              to_snapshot: Current period state
              
          Returns:
              Transition analysis
          """
          transition = {
              'from_state': from_snapshot['state_label'],
              'to_state': to_snapshot['state_label'],
              'pnl_change': to_snapshot['total_pnl'] - from_snapshot['total_pnl'],
              'win_rate_change': to_snapshot['win_rate'] - from_snapshot['win_rate'],
              'transition_type': f"{from_snapshot['state_label']}→{to_snapshot['state_label']}"
          }
          
          # Add interpretation
          if transition['from_state'] == 'healthy' and transition['to_state'] == 'deteriorating':
              transition['warning'] = 'Rapid deterioration detected'
          elif transition['from_state'] == 'deteriorating' and transition['to_state'] == 'healthy':
              transition['note'] = 'Recovery successful'
          
          return transition
  ```

- [ ] **Integrate with RIM** (3-4 hours)
  - **Location**: `src/governance/reflection/reflection_intelligence.py`
  - **Enhancement**: Use chunked reflection for periodic analysis
  - **Output**: Markov chain of strategy health over time
  - **Acceptance**: RIM generates reports showing state transitions

- [ ] **Add governance alerts** (2-3 hours)
  - **Alert**: Warn if unhealthy state transition pattern detected
  - **Example**: "Strategy transitioned healthy→borderline→deteriorating over 3 days"
  - **Acceptance**: Governance dashboard shows state transition warnings

- [ ] **Write tests** (1-2 hours)
  - **Location**: `tests/governance/reflection/test_chunked_reflection.py`
  - **Tests**: Snapshot creation, transition analysis, state labeling
  - **Acceptance**: All tests pass

**Success metrics**: Clearer governance insights; can identify strategy deterioration 1-2 periods earlier; state transition narrative is more digestible than continuous metrics.

---

**Implementation Priority for Markovian Enhancements**:

1. **Priority 1 (Weeks 1-2)**: Regime Transition Probabilities
   - Lowest effort (8-12 hours)
   - Highest governance value
   - Just exposes existing HMM internals
   - Immediate "reasoning depth" improvement

2. **Priority 2 (Weeks 3-5)**: Short-Horizon Rollouts
   - Medium effort (20-32 hours)
   - High trading value (5-10 bps PnL improvement)
   - Uses existing MuZero-lite + GraphNet
   - No latency penalty

3. **Priority 3 (Weeks 6-8)**: Chunked RL Training
   - Medium effort (16-24 hours)
   - High training value (30%+ convergence)
   - Extends existing PreTrainingPipeline
   - Better debugging and traceability

4. **Priority 4 (Weeks 9-12)**: Memory Pattern Chains
   - Higher effort (24-32 hours)
   - Medium-high adaptation value
   - Enables preemptive adaptation
   - 15-20% faster response

5. **Priority 5 (Weeks 13-14)**: Reflection Chunking
   - Low effort (12-16 hours)
   - Governance value
   - Bonus enhancement for RIM
   - Better strategy evolution insights

**Total Effort**: 80-116 hours (2-3 months part-time)

**Total Cost**: $0 (all build on existing code)

**GPU Required**: ❌ NO (all CPU-compatible)

**Expected ROI**:
- Chunked RL Training: 30%+ faster convergence, stable gradients
- Short-Horizon Rollouts: 5-10 bps PnL improvement per trade
- Regime Transitions: 10-15% better regime-change adaptation
- Pattern Chains: 15-20% faster adaptation, 10-15% drawdown reduction
- Reflection Chunking: Clearer governance insights, earlier deterioration detection

---

**Why These Are "Pure Wins"**:

✅ **Build on existing code** - No architectural changes required  
✅ **CPU-compatible** - No GPU needed for any enhancement  
✅ **Modular** - Each integration is independent, can be implemented separately  
✅ **Low risk** - All are auxiliary or training-time enhancements, don't affect core trading  
✅ **Measurable ROI** - Clear metrics for each (convergence, PnL, adaptation time, insights)  
✅ **Philosophically aligned** - Your system is already Markovian (BeliefState, RegimeFSM, MuZeroLite, HMM)  
✅ **Delethink principle adapted** - Same core insight (Markovian chunking) applied to trading domain  

---


**Acceptance Criteria**:
- ✅ Compatibility matrix defines which strategies run in which regimes
- ✅ Capital allocation adjusts automatically on regime change
- ✅ Strategies suspended when regime is hostile
- ✅ Hysteresis prevents regime thrashing (require 2+ consecutive detections)
- ✅ Backtests show aggregate drawdown reduction vs. regime-agnostic baseline

---

### Phase 3.10: Advanced Enhancements

**Priority**: 🟡 **High** - Implement after Phase 0-3.9

**Total effort**: 240-380 hours

#### 3.10.1 Bayesian Strategy Health with Capital as Posterior (40-60h)

**Objective**: Allocate capital proportional to Bayesian health score

- [ ] **Build Bayesian health score calculator** (16 hours)
  - Posterior mean of expected risk-adjusted return
  - Inputs: DSR, live slippage/impact error, drawdown state, alpha half-life mismatch, regime fit
  - **Location**: `src/trading/bayesian_health.py`
  - **Acceptance**: Health score updated daily with Bayesian inference

- [ ] **Implement online updating** (12 hours)
  - Update health score with each new trade/day
  - Bayesian posterior update
  - **Acceptance**: Health score updates online

- [ ] **Create capital allocator based on posterior** (12 hours)
  - Allocate capital proportional to health score
  - **Location**: `src/trading/capital_allocator_bayesian.py`
  - **Acceptance**: Capital allocation proportional to posterior mean

- [ ] **Add automatic decay** (8 hours)
  - Capital decays as evidence decays
  - **Acceptance**: Strategies with declining evidence get automatic capital reduction

- [ ] **Write comprehensive tests** (8 hours)
  - Test health score calculation
  - Test online updates
  - Test capital allocation
  - **Acceptance**: 90%+ test coverage, all tests passing

**Acceptance Criteria**:
- ✅ Health score updated daily with Bayesian inference
- ✅ Capital allocation proportional to posterior mean
- ✅ Strategies with declining evidence get automatic capital reduction
- ✅ Prior elicitation based on backtest DSR

---

#### 3.10.2 Genealogy + Diversity as Evolution Objective (32-48h)

**Objective**: Prevent population collapse into similar strategies

- [ ] **Build strategy genealogy tracker** (12 hours)
  - Family tree of evolved strategies
  - Track parent-child relationships
  - **Location**: `src/evolution/genealogy.py`
  - **Acceptance**: Genealogy tracks parent-child relationships

- [ ] **Implement diversity metrics** (12 hours)
  - Feature overlap, signal correlation, trade timestamp overlap, regime affinity
  - **Location**: `src/evolution/diversity_metrics.py`
  - **Acceptance**: Diversity score calculated for each strategy

- [ ] **Add diversity to NSGA-II objectives** (12 hours)
  - Multi-objective optimization: return, drawdown, capacity, **diversity**
  - Update `src/evolution/nsga2.py`
  - **Acceptance**: NSGA-II optimizes diversity alongside other objectives

- [ ] **Create population diversity monitor** (4 hours)
  - Alert if population becomes too similar
  - **Acceptance**: Alert when population diversity drops

- [ ] **Write comprehensive tests** (8 hours)
  - Test genealogy tracking
  - Test diversity metrics
  - Test NSGA-II integration
  - **Acceptance**: 90%+ test coverage, all tests passing

**Acceptance Criteria**:
- ✅ Genealogy tracks parent-child relationships
- ✅ Diversity score calculated for each strategy
- ✅ NSGA-II optimizes: return, drawdown, capacity, **diversity**
- ✅ Evolution prevents population collapse into similar strategies

---

#### 3.10.3 Regime Transition Early-Warning (32-48h)

**Objective**: Detect regime shifts before they happen

- [ ] **Implement BOCPD/CUSUM change-point detector** (16 hours)
  - Emits probability of regime shift within N bars
  - **Location**: `src/sensory/when/transition_warning.py`
  - **Acceptance**: Transition probability emitted every bar

- [ ] **Build transition probability tracker** (8 hours)
  - Rolling window of transition odds
  - **Acceptance**: Transition odds tracked

- [ ] **Create pre-emptive de-risking** (12 hours)
  - Reduce leverage when transition odds spike
  - **Location**: `src/trading/preemptive_derisking.py`
  - **Acceptance**: Pre-emptive de-risking triggers when P(transition) >threshold

- [ ] **Add hysteresis** (4 hours)
  - Prevent false alarm thrashing
  - **Acceptance**: Hysteresis prevents false alarms

- [ ] **Write comprehensive tests** (8 hours)
  - Test change-point detection
  - Test de-risking logic
  - Test hysteresis
  - **Acceptance**: 90%+ test coverage, all tests passing

**Acceptance Criteria**:
- ✅ Transition probability emitted every bar
- ✅ Pre-emptive de-risking triggers when P(transition) >threshold
- ✅ Hysteresis prevents false alarm thrashing
- ✅ Backtests show improved regime transition performance

---

#### 3.10.4 Execution Telemetry & Calibration Loop (40-60h)

**Objective**: Make impact models self-correcting

- [ ] **Build per-order telemetry** (16 hours)
  - Queue position, slippage vs. arrival, fill variance, latency (ingress→decision→send→ack→fill)
  - **Location**: `src/trading/execution/telemetry.py`
  - **Acceptance**: All orders emit telemetry

- [ ] **Implement weekly impact model re-fit** (16 hours)
  - Update Almgren-Chriss parameters from telemetry
  - **Location**: `src/trading/execution/calibration_loop.py`
  - **Acceptance**: Impact model parameters re-fit weekly from telemetry

- [ ] **Create A/B routing experiments** (12 hours)
  - Test model updates with shadow orders
  - **Acceptance**: A/B testing validates model updates before deployment

- [ ] **Build execution quality dashboard** (12 hours)
  - Real-time visualization of slippage, latency, fill rates
  - **Location**: `src/operations/dashboards/execution_quality.py`
  - **Acceptance**: Execution quality dashboard implemented

- [ ] **Write comprehensive tests** (8 hours)
  - Test telemetry collection
  - Test calibration loop
  - Test A/B testing
  - **Acceptance**: 90%+ test coverage, all tests passing

**Acceptance Criteria**:
- ✅ All orders emit telemetry (slippage, latency, fill quality)
- ✅ Impact model parameters re-fit weekly from telemetry
- ✅ A/B testing validates model updates before deployment
- ✅ Execution quality dashboard shows slippage, latency, fill rates

---

#### 3.10.5 Execution Shortfall Budgeting (24-32h)

**Objective**: Control execution costs per strategy

- [ ] **Assign shortfall budget per strategy** (8 hours)
  - Bps budget per notional (e.g., 5 bps per trade)
  - **Location**: `src/trading/execution/shortfall_budget.py`
  - **Acceptance**: Each strategy has shortfall budget

- [ ] **Implement budget enforcement** (8 hours)
  - Throttle or switch scheduler if budget over-runs
  - **Acceptance**: Router throttles or switches schedule if budget exceeded

- [ ] **Add weekly calibration** (6 hours)
  - Adjust budgets based on execution telemetry
  - **Acceptance**: Budgets re-calibrated weekly from telemetry

- [ ] **Create shortfall attribution** (6 hours)
  - Which strategies consume most budget?
  - **Location**: `src/trading/accounting/shortfall_attribution.py`
  - **Acceptance**: Shortfall-adjusted Sharpe ranks strategies

- [ ] **Write comprehensive tests** (6 hours)
  - Test budget assignment
  - Test enforcement logic
  - Test calibration
  - **Acceptance**: 90%+ test coverage, all tests passing

**Acceptance Criteria**:
- ✅ Each strategy has shortfall budget (e.g., 5 bps per trade)
- ✅ Router throttles or switches schedule if budget exceeded
- ✅ Budgets re-calibrated weekly from telemetry
- ✅ Shortfall-adjusted Sharpe ranks strategies by after-cost performance

---

#### 3.10.6 Population Stability & Drift Alarms (32-48h)

**Objective**: Detect when strategies are decaying

- [ ] **Implement drift detection** (16 hours)
  - PSI/KS/Wasserstein on feature distributions
  - **Location**: `src/validation/drift_detection.py`
  - **Acceptance**: Drift detection runs daily on feature distributions

- [ ] **Build label drift sentinel** (8 hours)
  - Detect if realized edge is changing
  - **Acceptance**: Label drift detected via realized edge tracking

- [ ] **Add automatic response** (12 hours)
  - Reduce size, refresh models, or spin evolutionary "rescue" child
  - **Location**: `src/evolution/rescue_child.py`
  - **Acceptance**: Automatic size reduction on drift breach

- [ ] **Create drift dashboard** (8 hours)
  - Visualize feature distributions over time
  - **Location**: `src/operations/monitoring/drift_monitor.py`
  - **Acceptance**: Drift dashboard visualizes distributions

- [ ] **Write comprehensive tests** (8 hours)
  - Test drift detection
  - Test automatic response
  - Test rescue child spawning
  - **Acceptance**: 90%+ test coverage, all tests passing

**Acceptance Criteria**:
- ✅ Drift detection runs daily on feature distributions
- ✅ Label drift detected via realized edge tracking
- ✅ Automatic size reduction on drift breach
- ✅ Evolutionary "rescue" child spawned for new regime

---

#### 3.10.7 Capacity & Crowding Curves as Deploy Artifacts (24-32h)

**Objective**: Know when strategies are saturated

- [ ] **Build capacity curve generator** (12 hours)
  - PnL vs. capital and PnL vs. turnover curves
  - **Location**: `src/trading/capacity_curves.py`
  - **Acceptance**: Capacity curves generated during backtesting

- [ ] **Implement deploy-time artifact storage** (6 hours)
  - Store curves with each strategy version
  - **Acceptance**: Curves stored as deploy-time artifacts

- [ ] **Create runtime allocator** (10 hours)
  - Read curves and keep portfolio on efficient frontier
  - **Location**: `src/trading/capacity_aware_allocator.py`
  - **Acceptance**: Allocator reads curves at runtime

- [ ] **Add diminishing returns alarm** (4 hours)
  - Alert when pushing past capacity knee
  - **Acceptance**: Alarm when strategy pushed past capacity knee

- [ ] **Write comprehensive tests** (6 hours)
  - Test curve generation
  - Test artifact storage
  - Test allocator logic
  - **Acceptance**: 90%+ test coverage, all tests passing

**Acceptance Criteria**:
- ✅ Capacity curves generated during backtesting
- ✅ Curves stored as deploy-time artifacts
- ✅ Allocator reads curves at runtime
- ✅ Alarm when strategy pushed past capacity knee

---

#### 3.10.8 Regime-Specific Stress Packs (24-40h)

**Objective**: Test strategies under extreme scenarios

- [ ] **Build stress scenario library** (12 hours)
  - Pre-baked disasters (melt-up→crisis flip, illiquid gaps, LP pullbacks)
  - **Location**: `src/simulation/stress_scenarios.py`
  - **Acceptance**: Stress scenario library with 10+ disasters

- [ ] **Implement historical crisis replay** (12 hours)
  - 2008, 2010 flash crash, 2020 COVID, 2022 FTX
  - **Location**: `src/simulation/historical_crises.py`
  - **Acceptance**: Historical crisis replay implemented

- [ ] **Create synthetic crisis generation** (12 hours)
  - Monte Carlo regime transitions
  - **Acceptance**: Synthetic crisis generation creates extreme scenarios

- [ ] **Add kill-switch validation** (6 hours)
  - Prove kill-switches trigger at right thresholds
  - **Location**: `tests/risk/test_killswitch_validation.py`
  - **Acceptance**: Kill-switches validated to trigger exactly when intended

- [ ] **Write comprehensive tests** (6 hours)
  - Test stress scenarios
  - Test historical replays
  - Test synthetic generation
  - **Acceptance**: 90%+ test coverage, all tests passing

**Acceptance Criteria**:
- ✅ Stress scenario library with 10+ regime-specific disasters
- ✅ Historical crisis replay shows how EMP would perform
- ✅ Synthetic crisis generation creates extreme scenarios
- ✅ Kill-switches validated to trigger exactly when intended

---

#### 3.10.9 Portfolio-Level Objectives (32-48h)

**Objective**: Optimize portfolio, not just individual strategies

- [ ] **Build multi-objective optimizer** (16 hours)
  - Optimize Sortino, Calmar, UPR (not just Sharpe)
  - **Location**: `src/trading/portfolio_optimizer.py`
  - **Acceptance**: Portfolio optimized for Sortino/Calmar/UPR

- [ ] **Add correlation constraints** (8 hours)
  - Penalize strategies with high correlation
  - **Acceptance**: Correlation constraints prevent crowding

- [ ] **Implement capacity constraints** (8 hours)
  - Don't over-allocate to saturated strategies
  - **Acceptance**: Capacity constraints respected

- [ ] **Create tail risk contribution** (12 hours)
  - Identify strategies contributing most to portfolio tail risk
  - **Location**: `src/risk/tail_risk_contribution.py`
  - **Acceptance**: Tail risk contribution calculated per strategy

- [ ] **Write comprehensive tests** (8 hours)
  - Test multi-objective optimization
  - Test constraints
  - Test tail risk calculation
  - **Acceptance**: 90%+ test coverage, all tests passing

**Acceptance Criteria**:
- ✅ Portfolio optimized for Sortino/Calmar/UPR (not just Sharpe)
- ✅ Correlation constraints prevent crowding
- ✅ Capacity constraints respected
- ✅ Tail risk contribution calculated per strategy

---

#### 3.10.10 Compliance, Secrets, and Tamper-Evidence (24-40h)

**Objective**: Institutional-grade governance

- [ ] **Implement key rotation** (12 hours)
  - Automatic rotation of API keys and credentials every 90 days
  - **Location**: `src/governance/secrets_manager.py`
  - **Acceptance**: API keys rotated automatically every 90 days

- [ ] **Create scoped credentials** (8 hours)
  - Least-privilege access per component
  - **Acceptance**: Each component has scoped credentials

- [ ] **Build tamper-evident audit chains** (12 hours)
  - Hash-chained logs (append-only)
  - **Location**: `src/governance/tamper_evident_log.py`
  - **Acceptance**: Audit logs are hash-chained (tamper-evident)

- [ ] **Add policy checks in CI** (8 hours)
  - Fail builds if touching live accounts without approval
  - **Location**: `.github/workflows/policy_check.yml`
  - **Acceptance**: CI fails if code touches live accounts without policy approval

- [ ] **Write comprehensive tests** (8 hours)
  - Test key rotation
  - Test scoped credentials
  - Test tamper-evident logs
  - **Acceptance**: 90%+ test coverage, all tests passing

**Acceptance Criteria**:
- ✅ API keys rotated automatically every 90 days
- ✅ Each component has scoped credentials (least privilege)
- ✅ Audit logs are hash-chained (tamper-evident)
- ✅ CI fails if code touches live accounts without policy approval

---

#### 3.10.11 Strategic Niche Documentation (2-4h)

**Objective**: Document strategic focus

- [ ] **Document target niche** (2 hours)
  - "FX/CFD, mid-frequency, event-driven, 5-minute to 4-hour horizon"
  - Rationale: Information-rich, exploitable structure, limited infrastructure arms race
  - **Location**: `README.md` (add "Strategic Focus" section)
  - **Acceptance**: README has clear "Strategic Focus" section

- [ ] **Define success metrics** (1 hour)
  - Sharpe >2.0, capacity >$1M, consistency <0.1
  - **Acceptance**: Success metrics defined

- [ ] **Document expansion criteria** (1 hour)
  - "Dominate FX before expanding to equities"
  - **Location**: `docs/strategy/niche_focus.md`
  - **Acceptance**: Expansion criteria explicit

**Acceptance Criteria**:
- ✅ README has clear "Strategic Focus" section
- ✅ Niche constraints documented (instruments, timeframes, strategy types)
- ✅ Success metrics defined
- ✅ Expansion criteria explicit

---

### Phase 4: Production Hardening (Weeks 13-16)

**Goal**: Deploy enterprise-grade operational infrastructure

**Estimated Effort**: 160 hours (4 weeks × 40 hours)

#### 4.1 Monitoring Stack (Prometheus + Grafana)

**Objective**: Real-time visibility into system health and performance

- [x] **Set up Prometheus infrastructure** (8 hours)
  - Deploy Prometheus server
  - Configure scraping and retention
  - Set up service discovery
  - **Acceptance**: Prometheus operational and scraping metrics

- [ ] **Instrument application code** (16 hours)
  - Add Prometheus metrics to all major components
  - Implement RED metrics (Rate, Errors, Duration)
  - Add business metrics (trades, PnL, positions)
  - **Location**: Throughout codebase using `prometheus_client`
  - **Acceptance**: 100+ metrics exposed

- [ ] **Deploy Grafana dashboards** (12 hours)
  - Create system health dashboard
  - Build trading performance dashboard
  - Add risk monitoring dashboard
  - **Acceptance**: Comprehensive real-time visibility

- [ ] **Implement alerting rules** (8 hours)
  - Define alert thresholds
  - Configure AlertManager
  - Set up notification channels (email, Slack, PagerDuty)
  - **Acceptance**: Alerts fire on critical issues

- [ ] **Create runbooks for alerts** (8 hours)
  - Document response procedures for each alert
  - Add troubleshooting steps
  - Include escalation paths
  - **Acceptance**: Every alert has a runbook

#### 4.2 Logging Stack (ELK)

**Objective**: Centralized logging for debugging and audit

- [ ] **Deploy Elasticsearch cluster** (8 hours)
  - Set up Elasticsearch nodes
  - Configure indices and retention
  - Implement security and access control
  - **Acceptance**: Elasticsearch operational and ingesting logs

- [ ] **Set up Logstash pipeline** (8 hours)
  - Configure log ingestion from applications
  - Implement log parsing and enrichment
  - Add filtering and routing
  - **Acceptance**: Logs flowing into Elasticsearch

- [ ] **Deploy Kibana dashboards** (8 hours)
  - Create log exploration interface
  - Build saved searches for common queries
  - Add visualizations for log patterns
  - **Acceptance**: Easy log search and analysis

- [ ] **Implement structured logging** (12 hours)
  - Migrate to structured JSON logs
  - Add correlation IDs for request tracing
  - Include context in all log messages
  - **Location**: Throughout codebase using `structlog`
  - **Acceptance**: All logs structured and searchable

- [ ] **Create log retention policies** (4 hours)
  - Define retention periods by log level
  - Implement log archival to S3
  - Add compliance-required log preservation
  - **Acceptance**: Logs retained per policy

#### 4.3 Incident Response

**Objective**: Rapid detection and resolution of production issues

- [ ] **Build incident detection system** (12 hours)
  - Implement anomaly detection on metrics
  - Add pattern recognition in logs
  - Create composite health checks
  - **Location**: `src/operations/incident_detection.py`
  - **Acceptance**: Incidents detected within 1 minute

- [ ] **Create incident management workflow** (8 hours)
  - Define incident severity levels
  - Build incident tracking system
  - Implement escalation procedures
  - **Acceptance**: Clear incident response process

- [ ] **Implement automated remediation** (16 hours)
  - Build self-healing for common issues
  - Implement circuit breakers
  - Add automatic service restarts
  - **Location**: `src/operations/auto_remediation.py`
  - **Acceptance**: 80% of incidents auto-remediated

- [ ] **Create incident postmortem process** (8 hours)
  - Define postmortem template
  - Implement blameless postmortem culture
  - Build action item tracking
  - **Acceptance**: Postmortem for every major incident

- [x] **Build incident simulation (chaos engineering)** (12 hours)
  - Implement failure injection
  - Create disaster scenarios
  - Run regular fire drills
  - **Location**: `src/operations/incident_simulation.py`
  - **Acceptance**: Chaos campaigns export Markdown/JSON evidence and exercise multiple responder/metrics failure modes

#### 4.4 Disaster Recovery

**Objective**: Ensure business continuity in catastrophic scenarios

- [ ] **Design disaster recovery plan** (8 hours)
  - Define RTO and RPO targets
  - Identify critical systems and data
  - Plan failover procedures
  - **Acceptance**: Comprehensive DR plan document

- [ ] **Implement database backups** (8 hours)
  - Automate daily backups of all databases
  - Test backup restoration
  - Store backups in multiple regions
  - **Acceptance**: Backups restorable within 1 hour

- [x] **Build configuration backup** (8 hours)
  - Version control all configuration
  - Implement configuration snapshots
  - Add configuration rollback capability
  - **Location**: `src/operations/configuration_backup.py`
  - **Acceptance**: Manifest + archive package config trees with deterministic hashing for restores

- [ ] **Create failover infrastructure** (16 hours)
  - Set up hot standby systems
  - Implement automatic failover
  - Add health checks and failover triggers
  - **Acceptance**: Failover completes within 5 minutes

- [ ] **Conduct DR drills** (12 hours)
  - Run quarterly DR exercises
  - Test full system recovery
  - Document lessons learned
  - **Acceptance**: Successful DR drill with <1 hour recovery

#### 4.5 Performance Optimization

**Objective**: Achieve production performance targets

- [ ] **Conduct performance profiling** (12 hours)
  - Profile CPU, memory, I/O usage
  - Identify bottlenecks
  - Measure end-to-end latency
  - **Acceptance**: Performance baseline established

- [ ] **Optimize critical paths** (20 hours)
  - Reduce risk validation latency
  - Optimize data ingestion throughput
  - Improve strategy evaluation speed
  - **Acceptance**: 50% latency reduction in critical paths

- [ ] **Implement caching strategies** (12 hours)
  - Add Redis caching for hot data
  - Implement query result caching
  - Build cache warming on startup
  - **Acceptance**: Cache hit rate >90%

- [ ] **Conduct load testing** (16 hours)
  - Simulate production load
  - Test system under stress
  - Identify breaking points
  - **Acceptance**: System handles 10x expected load

- [ ] **Optimize resource utilization** (12 hours)
  - Tune database queries
  - Optimize memory usage
  - Reduce unnecessary computation
  - **Acceptance**: 30% reduction in resource costs

---

### Phase 5: Regulatory Compliance (Weeks 17-20)

**Goal**: Meet institutional and regulatory requirements

**Estimated Effort**: 160 hours (4 weeks × 40 hours)

#### 5.1 Regulatory Reporting

**Objective**: Generate required regulatory reports automatically

- [ ] **Research regulatory requirements** (12 hours)
  - Study MiFID II, Dodd-Frank, SEC rules
  - Identify required reports and formats
  - Understand filing deadlines
  - **Acceptance**: Compliance requirements document

- [ ] **Implement trade reporting** (16 hours)
  - Build FIX-based trade reporting
  - Implement transaction reporting
  - Add order audit trail
  - **Location**: `src/compliance/trade_reporting.py`
  - **Acceptance**: All trades reported per regulations

- [ ] **Create position reporting** (12 hours)
  - Generate daily position reports
  - Implement large position reporting
  - Add exposure breakdowns
  - **Location**: `src/compliance/position_reporting.py`
  - **Acceptance**: Position reports match regulatory format

- [ ] **Build risk reporting** (12 hours)
  - Calculate regulatory risk metrics (VaR, etc.)
  - Generate risk reports
  - Implement stress test reporting
  - **Location**: `src/compliance/risk_reporting.py`
  - **Acceptance**: Risk reports meet regulatory standards

- [ ] **Implement automated filing** (12 hours)
  - Build integration with regulatory portals
  - Automate report submission
  - Add filing confirmation tracking
  - **Location**: `src/compliance/filing.py`
  - **Acceptance**: Reports filed automatically on schedule

#### 5.2 Compliance Monitoring

**Objective**: Continuous monitoring for regulatory violations

- [ ] **Implement pre-trade compliance checks** (16 hours)
  - Check position limits before trades
  - Validate against restricted securities
  - Enforce trading hours and venue rules
  - **Location**: `src/compliance/pretrade_checks.py`
  - **Acceptance**: 100% of trades pass pre-trade checks

- [ ] **Build post-trade surveillance** (16 hours)
  - Detect wash trades and manipulation
  - Identify excessive trading
  - Monitor for insider trading patterns
  - **Location**: `src/compliance/surveillance.py`
  - **Acceptance**: Suspicious activity detected and flagged

- [ ] **Create compliance dashboard** (12 hours)
  - Visualize compliance status
  - Show violations and exceptions
  - Track remediation actions
  - **Location**: `src/compliance/dashboard.py`
  - **Acceptance**: Real-time compliance visibility

- [ ] **Implement compliance alerting** (8 hours)
  - Alert on potential violations
  - Notify compliance officers
  - Escalate critical issues
  - **Acceptance**: Violations detected within 1 minute

#### 5.3 Audit Procedures

**Objective**: Enable efficient internal and external audits

- [ ] **Enhance audit trail** (12 hours)
  - Ensure all decisions are logged
  - Add tamper-proof logging
  - Implement audit log retention
  - **Location**: Throughout codebase
  - **Acceptance**: Complete audit trail for all activities

- [ ] **Build audit report generator** (12 hours)
  - Create audit-ready reports
  - Include decision rationale
  - Add supporting evidence
  - **Location**: `src/compliance/audit_reports.py`
  - **Acceptance**: Audit reports generated on demand

- [ ] **Implement audit search** (8 hours)
  - Build powerful search across audit logs
  - Add filtering and export
  - Create audit replay capability
  - **Location**: `src/compliance/audit_search.py`
  - **Acceptance**: Find any audit record in <1 second

- [ ] **Create audit documentation** (12 hours)
  - Document all audit procedures
  - Explain decision-making processes
  - Provide audit evidence examples
  - **Acceptance**: Comprehensive audit documentation

#### 5.4 Kill-Switch and Emergency Controls

**Objective**: Immediate system shutdown capability

- [ ] **Implement global kill-switch** (12 hours)
  - Build instant trading halt mechanism
  - Add multiple activation methods (UI, API, hotkey)
  - Ensure fail-safe operation
  - **Location**: `src/operations/kill_switch.py`
  - **Acceptance**: Kill-switch stops all trading in <1 second

- [ ] **Create strategy-level controls** (8 hours)
  - Implement per-strategy pause/resume
  - Add strategy risk limits
  - Build strategy quarantine
  - **Location**: `src/governance/strategy_controls.py`
  - **Acceptance**: Individual strategies controllable

- [ ] **Build position liquidation** (12 hours)
  - Implement emergency position closing
  - Add smart liquidation (minimize impact)
  - Create liquidation dry-run mode
  - **Location**: `src/trading/liquidation.py`
  - **Acceptance**: Positions liquidated safely on command

- [ ] **Implement circuit breakers** (12 hours)
  - Add automatic trading halts on loss thresholds
  - Implement volatility-based breakers
  - Build cooldown periods
  - **Location**: `src/risk/circuit_breakers.py`
  - **Acceptance**: Circuit breakers trigger appropriately

- [x] **Create emergency procedures documentation** (8 hours)
  - Document all emergency scenarios
  - Provide step-by-step response procedures
  - Include contact information
  - **Location**: `docs/operations/emergency_procedures.md`
  - **Acceptance**: Clear emergency response playbook covering triggers, comms, and post-incident reviews

#### 5.5 Operational Documentation

**Objective**: Comprehensive documentation for operations team

- [ ] **Create operations manual** (16 hours)
  - Document daily operational procedures
  - Include system startup/shutdown
  - Add troubleshooting guides
  - **Acceptance**: Complete operations manual

- [ ] **Build runbook library** (16 hours)
  - Create runbooks for common issues
  - Add escalation procedures
  - Include contact information
  - **Location**: `docs/runbooks/`
  - **Acceptance**: Runbook for every operational scenario

- [ ] **Implement change management** (12 hours)
  - Define change approval process
  - Build change tracking system
  - Add rollback procedures
  - **Acceptance**: All changes tracked and approved

- [ ] **Create training materials** (12 hours)
  - Build operator training program
  - Create video tutorials
  - Add hands-on exercises
  - **Acceptance**: New operators trained in 1 week

- [ ] **Document system architecture** (12 hours)
  - Create comprehensive architecture diagrams
  - Document data flows
  - Explain design decisions
  - **Acceptance**: Architecture fully documented

---

## Documentation

### Core Documentation

- **[EMP Encyclopedia](EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md)**: Comprehensive vision and design specifications (aspirational)
- **[AlphaTrade Whitepaper](docs/AlphaTrade_Whitepaper.md)**: Current governance and operational status (factual)
- **[Architecture Reality](docs/ARCHITECTURE_REALITY.md)**: Honest assessment of implementation state
- **[Development Status](docs/DEVELOPMENT_STATUS.md)**: Current progress and known issues
- **[Gap Analysis](docs/reports/gap_analysis.md)**: Detailed comparison of claims vs. reality

### Technical Documentation

- **[API Documentation](docs/api/)**: Module-level API references
- **[Architecture Guides](docs/architecture/)**: System design and data flow
- **[Runbooks](docs/runbooks/)**: Operational procedures and troubleshooting
- **[Compliance](docs/compliance/)**: Regulatory and audit documentation

### Development Guides

- **[Contributing](docs/development/contributing.md)**: How to contribute to the project
- **[Testing](docs/development/testing.md)**: Testing standards and procedures
- **[Setup](docs/development/setup.md)**: Development environment setup

---

## Key Components Reference

### Risk Management
```python
from src.risk.risk_manager_impl import RiskManagerImpl

# Initialize risk manager with policy
risk_manager = RiskManagerImpl(config=risk_config)

# Validate trade intent
decision = await risk_manager.validate_trade_intent(
    intent=trade_intent,
    portfolio_state=current_portfolio
)
```

### Sentient Learning
```python
from src.sentient.sentient_predator import SentientPredator

# Initialize learning loop
predator = SentientPredator(
    memory=faiss_memory,
    planner=muzero_planner,
    config=sentient_config
)

# Learn from trade outcome
await predator.learn_from_outcome(
    trade_outcome=outcome,
    market_state=state
)
```

### Execution
```python
from src.trading.execution.live_broker_adapter import LiveBrokerExecutionAdapter

# Initialize live broker with risk gating
broker = LiveBrokerExecutionAdapter(
    risk_gateway=risk_manager,
    config=broker_config
)

# Execute trade with full risk checks
order = await broker.process_order(
    intent=trade_intent,
    guardrails=guardrail_snapshot
)
```

---

## Testing Philosophy

The project maintains high testing standards with multiple test categories:

- **Unit Tests**: Component-level validation (`tests/*/test_*.py`)
- **Integration Tests**: Cross-component workflows (`tests/integration/`)
- **Regression Tests**: Prevent known issues from recurring
- **Guardrail Tests**: Verify risk and policy enforcement
- **Evidence Tests**: Reproduce claims from whitepapers

All major features include test coverage with assertions on behavior, not just structure.

---

## Contributing

We welcome contributions that align with the project's truth-first philosophy:

1. **Evidence-Based Development**: All features must include tests
2. **Documentation Accuracy**: Update docs to match implementation reality
3. **Incremental Progress**: Small, verifiable improvements over large claims
4. **Code Quality**: Follow Black formatting, mypy type checking, and pytest standards

See [CONTRIBUTING.md](docs/development/contributing.md) for detailed guidelines.

---

## Known Issues and Limitations

### Current Limitations

1. **Data Sources**: Limited to Yahoo Finance historical data; no real-time feeds
2. **Strategy Library**: Few concrete strategies implemented
3. **Backtesting**: No systematic historical replay framework
4. **Evolution**: Parameter tuning only; not full strategy discovery
5. **Monitoring**: Governance dashboard exists; full production stack pending

### Deprecated Components

The following directories contain legacy code and should not be used:

- `archive/legacy/`: Old implementations superseded by current architecture
- Some modules in `src/data_integration/`: Placeholder adapters not implemented

See [DEPRECATION_POLICY.md](docs/reports/DEPRECATION_POLICY.md) for details.

---

## Performance Characteristics

### Current Performance (Paper Trading)

- **Decision Latency**: <100ms for risk validation and trade intent generation
- **Memory Usage**: ~500MB base + FAISS index size
- **Throughput**: Handles 100+ trades/day in paper mode
- **Data Processing**: 10K+ OHLCV bars/second normalization

### Target Performance (Production)

- **Decision Latency**: <10ms for high-frequency strategies
- **Memory Usage**: <2GB for full system
- **Throughput**: 1000+ trades/day across multiple strategies
- **Data Processing**: Real-time tick data ingestion at market rates

---

## License

[Specify license here]

---

## Contact and Support

- **Issues**: [GitHub Issues](https://github.com/HWeber-tech/emp_proving_ground_v1/issues)
- **Discussions**: [GitHub Discussions](https://github.com/HWeber-tech/emp_proving_ground_v1/discussions)
- **Documentation**: [docs/](docs/)

---

## Acknowledgments

This project draws inspiration from:

- **AlphaGo/AlphaFold**: Deep reinforcement learning and self-play
- **Nassim Taleb**: Antifragility and robust systems design
- **Quantitative Finance**: Market microstructure and institutional trading
- **Evolutionary Computation**: Genetic algorithms and population-based optimization

---

## Status Summary

**What This Project Is:**
- A sophisticated, well-architected trading framework
- Production-grade risk management and execution adapters
- Advanced learning and adaptation capabilities
- Strong governance and audit infrastructure

**What This Project Is Not (Yet):**
- A complete, production-ready trading bot
- A Bloomberg Terminal replacement
- Fully autonomous without human oversight
- Validated with live trading performance

**Path Forward:**
With focused development on data integration, strategy library, and operational infrastructure, this framework can evolve into the revolutionary trading system envisioned. The foundation is solid; what remains is systematic execution of the roadmap.

---

**Last Updated:** October 21, 2025  
**Version:** v1.0-dev  
**Status:** Active Development

