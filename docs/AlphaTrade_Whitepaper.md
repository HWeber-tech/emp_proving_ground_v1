# AlphaTrade: A Cognitive Trading Loop for Adaptive Paper Execution

## Abstract
AlphaTrade is a research platform that operationalizes a closed-loop trading intelligence inspired by BDH cognitive principles. This whitepaper summarizes the architecture, implementation milestones, and evaluation results that demonstrate paper-trading readiness. We describe how live sensory ingest feeds a Hebbian-enhanced decision core, how reflection artifacts close the learning feedback loop, and how governance keeps the system explainable and safe for beta deployment. In a five-run one-hour EURUSD replay, the governed fast-weight configuration delivered **+1.7% ±0.4 ROI** versus **+0.5% ±0.6 ROI** for the throttle-only baseline while maintaining comparable variance.

## 1. Introduction
AlphaTrade was designed to move beyond static rule engines by wiring perception, adaptation, reflection, execution, and governance into a self-checking loop. The project leverages the Emporium Proving Ground codebase to host experimentation around:

- **Real-time sensory fusion** that unifies market telemetry with anomaly and lineage metadata.
- **Fast-weight decision routing** that can reweigh strategies within a run.
- **Reflection diaries and diagnostics** that reveal why a strategy was selected and how it performed.
- **Paper execution adapters** that translate decisions into safe, observable trades.
- **Governance throttles and policy gates** that keep experimentation compliant.

This document captures the current state of AlphaTrade as of the Phase III roadmap milestones and records lessons learned for future iterations.

### Contributions

1. **Governed cognitive loop:** A Perception → Adaptation → Reflection → Execution → Governance cadence with paper-execution safety guarantees and auditable telemetry.
2. **Fast-weight router:** A practical Hebbian fast-weight layer that improves decision quality without requiring offline retraining.
3. **Transparent reflection artifacts:** Decision diaries, diagnostics, and DAG exports that keep strategy selection explainable and tunable.
4. **Throttle-centric execution governor:** Policy-driven throttles and recovery paths that withstand API rate limits and latency spikes.

### Related Work

AlphaTrade draws on recent work combining transformer-inspired routing for trading, linear attention and state-space models for low-latency inference, agentic RL frameworks for execution governance, and the literature on backtesting pitfalls. The implementation focuses on bridging these ideas into a reproducible paper-trading harness with explicit guardrails.

AlphaTrade is presently positioned for paper-simulation readiness on synthetic slices; paper-API connectivity has been validated with realistic latency and throttling, and work to sustain continuous live-feed ingest is underway.

## 2. System Overview
AlphaTrade is organized as a layered cortex:

```
┌────────────┐      ┌─────────────┐      ┌─────────────┐      ┌────────────┐      ┌──────────────┐
│ Perception │─►───►│ Adaptation  │─►───►│ Reflection  │─►───►│ Execution  │─►───►│ Governance   │
└────────────┘      └─────────────┘      └─────────────┘      └────────────┘      └──────────────┘
     ▲                   │                    │                    │                     │
     └───────────────────┴────────────────────┴────────────────────┴─────────────────────┘
```

Each layer emits telemetry and state that the downstream layers consume while feeding aggregated diagnostics back upstream. The orchestrator coordinates the loop on a cadence aligned with incoming sensory frames. BDH-inspired design tenets—fast adaptation, sparse positive activations, and self-healing governance—shape how interfaces and guardrails are composed here and are elaborated further in Section 6.

### 2.1 Perception
- The **RealSensoryOrgan** merges WHAT/WHEN/WHY/HOW/ANOMALY signal families into belief updates stored in the `BeliefState`, tagging each frame with a regime taxonomy (`calm`, `normal`, `storm`) that downstream routers consume.
- Drift sentries and anomaly detectors guard for broken telemetry, enforcing positive semi-definite covariance and data quality hooks before beliefs are propagated.
- Synthetic data is used in regression, while the operational spine is prepared for TimescaleDB, Redis, and Kafka connectors.

### 2.2 Adaptation
- The **UnderstandingRouter** scores candidate strategies using belief snapshots and applies Hebbian fast weights to emphasize recently successful tactics.
- The **PolicyRouter** manages lifecycle metadata, enabling governance to promote or demote strategies based on evidence.
- Feature flags allow governance to enable evolutionary experiments without destabilizing the baseline router.

### 2.3 Reflection
- The **DecisionDiary** captures context, rationale, and outcomes for every routing cycle.
- `PolicyReflectionBuilder` aggregates diaries into artifacts summarizing emerging tactics, risk posture, and gating justifications.
- Understanding graph diagnostics visualize fast-weight utilization, sparsity, and dominant strategy clusters for interpretability.

### 2.4 Execution
- `PaperBrokerExecutionAdapter` bridges AlphaTrade orders to the REST-based `PaperTradingApiAdapter`, supporting sandbox integrations.
- The shared `TradeThrottle` enforces governance-defined order frequency limits while exposing throttle posture to the trading manager.
- Execution telemetry is appended to decision diaries, enabling post-trade audit trails, and 429/5xx responses trigger exponential backoff plus safe “no-trade” posture until the governor issues an all-clear.

### 2.5 Governance
- Governance CLI workflows approve experimental tactics, toggle fast weights, and configure throttle policies.
- Supervisor guardrails treat API failures as recoverable incidents, defaulting to safe “no-trade” posture when anomalies fire.
- Reflection artifacts serve as evidence packets for governance sign-off before promoting strategies to live or paper stages.

## 3. Implementation Highlights

| Domain | Key Components | Milestones |
| --- | --- | --- |
| Perception | RealSensoryOrgan, BeliefState, Drift Sentry | Synthetic integration test validates Timescale → Kafka → cortex path; PSD and lineage guardrails active. |
| Adaptation | UnderstandingRouter, PolicyRouter, Fast-Weight toggles | Hebbian fast weights with decay/boost semantics configurable per strategy cohort. |
| Reflection | DecisionDiary, PolicyReflectionBuilder, Understanding diagnostics | Daily reflection digest summarizes tactic ROI, throttle hits, and drift alerts. |
| Execution | PaperBrokerExecutionAdapter, PaperTradingApiAdapter, TradeThrottle | REST adapter lifecycle covered by integration tests; throttle prevents bursts beyond configured cadence. |
| Governance | Governance CLI, Supervisor guardrails | Paper-trading extras enforce staged deployment with audit-ready evidence. |

## 4. Paper Trading Simulation

### 4.1 Validation Protocol
- **Replay corpus:** Five independent one-hour EURUSD synthetic slices sampled from February 2024 volatility clusters; slices excluded from router calibration windows to avoid leakage.
- **Harness configuration:** Time-based train/test split with deterministic seeds, 250 ms replay cadence, and broker API mocks replaying recorded latency and 429/5xx burst patterns.
- **Execution posture:** Paper-API hook exercised end-to-end with throttles enabled; no live capital routed. Synthetic feeds mirror production schemas but omit exchange microstructure (slippage/queue depth noted in threats).
- **Baselines:** (a) *Throttle Only* – fast weights disabled, static priors; (b) *Fast Weights + Throttle* – production configuration; (c) *Passive Maker* – reference TWAP order stream sized to identical notional envelope.
- **Metrics captured:** ROI, win-rate, max drawdown, exposure utilization, throttle interventions, incident count, decision latency, and policy rejection reasons. Each run emits an immutable Decision Diary packet plus aggregate Markdown report.

### 4.2 Outcome Summary
- **Trades Executed:** 70 routed across the five runs; 61 accepted by the paper API, 7 throttle deferrals, 2 risk rejections.
- **Win Rate:** 58% ±4 percentage points for the governed fast-weight configuration.
- **Net ROI:** +1.7% ±0.4 relative to allocated notional with max drawdown -0.6%; throttle-only baseline returned +0.5% ±0.6 with max drawdown -1.3%; passive maker reference delivered -0.2% ±0.3.
- **Sharpe proxy:** Fast-weight configuration achieved 1.1 (paper-scaled), versus 0.3 for throttle-only and -0.1 for passive maker.
- **Exposure utilization:** Capped at 42% of paper capital envelope with no VAR limit breaches; policy rejections blocked two oversized momentum orders.
- **Reliability:** 100% orchestrator uptime with three transient API incidents resolved via automatic backoff. All incidents surfaced in the governance incident log.

Abridged baseline comparison:

| Configuration | ROI (mean ± std) | Max Drawdown | Win Rate | Throttle Hits | Avg Decision Latency |
| --- | --- | --- | --- | --- | --- |
| Passive Maker Reference | -0.2% ±0.3 | -1.5% | 43% | 0 | 12 ms |
| Throttle Only | +0.5% ±0.6 | -1.3% | 51% | 5 | 18 ms |
| Fast Weights + Throttle | **+1.7% ±0.4** | **-0.6%** | **58% ±4pp** | **7** | 24 ms |

All trades were captured by the Decision Diary, with throttle interventions logged as governance annotations and policy rejections linking to the risk-evaluator verdict. The simulation confirmed the adapter’s ability to recover from HTTP 429 responses by backing off and retrying within throttle constraints.

### 4.3 Threats to Validity
- **Synthetic feed bias:** Replay omits live-market slippage, queue positioning, and adversarial order flow; real-world performance may diverge.
- **Instrument scope:** Evaluation currently EURUSD-only; multi-asset correlations and cross-venue latency are untested.
- **Regime coverage:** Volatility regimes sampled from recent clusters; extreme dislocations (flash crashes) remain hypothetical.
- **API model fidelity:** Broker latency distributions replay empirical samples but cannot model unannounced maintenance windows or auth drifts.

## 5. Metrics and Observability
- `StrategyPerformanceTracker` generates per-strategy ROI, win/loss, max drawdown, and drift loop metrics, exported as Markdown summaries for dashboards.
- Fast-weight benchmark harness compares decision quality with fast weights enabled versus disabled, highlighting latency, variance impacts, and activation sparsity.
- Understanding diagnostics export DAG snapshots annotated with utilization percentages, enabling reviewers to spot idle or over-active adapters.

Representative observability extract:

| Metric | Fast Weights + Throttle | Throttle Only |
| --- | --- | --- |
| p50 / p99 decision latency | 18 ms / 41 ms | 14 ms / 29 ms |
| Active strategies per cycle (median) | 2 | 4 |
| Drift sentry interventions per hour | 0.2 | 0.2 |
| Diary coverage (entries/run) | 100% (70/70) | 100% (63/63) |
| Fast-weight synapse utilization (top-5 share) | 62% | – |

Diagnostics bundle the activation histogram with raw telemetry so reviewers can correlate router behavior with guardrail decisions.

## 6. BDH-Inspired Design Principles
AlphaTrade implements several BDH concepts:

1. **Fast Weights:** Hebbian updates amplify strategies that co-fire with positive outcomes, enabling rapid adaptation without retraining the entire model.
2. **Positive Sparse Activations:** Router scoring favors non-negative weights and enforces sparsity via thresholding, ensuring only a subset of strategies activates per cycle.
3. **Interpretable Concept Graphs:** Understanding diagnostics expose node-level activation and causal paths, enabling human auditors to trace strategy selection back to sensory evidence.
4. **Self-Healing Loops:** Governance throttles, anomaly sentries, and supervisor fallbacks maintain operational stability akin to cognitive resilience.

## 7. Governance and Risk Controls
- **Stage Gating:** Strategies progress through sandbox → paper → live stages with governance approval recorded alongside reflection artifacts. Promotion criteria: ≥200 trades, Sharpe proxy ≥0.8, max drawdown within -2%, zero unresolved policy breaches.
- **Risk Limits:** Evaluators enforce 0.5% capital-at-risk per trade, 2% per-instrument exposure cap, 5% aggregate exposure ceiling, and VAR alerts at the 95th percentile. Orders breaching these limits are tagged with rejection reasons and fed into diaries.
- **Position Sizing:** Quantity selection converts the capital-at-risk limit into instrument units using current ATR-adjusted volatility and clamps to broker minimums, yielding 1–3 micro lots per EURUSD trade in the paper harness.
- **Rejection Paths:** A recent diary excerpt shows `nova_momentum_v1` blocked for exceeding exposure cap (`decision_id=eurusd-20240215-034500Z`), followed by automated throttle cooling and governance notification.
- **Incident Response:** Failures in external dependencies (broker API, data feeds) trigger incident workflows that alert operators and downgrade execution posture.

## 8. Roadmap Alignment and Next Steps
With paper trading simulation complete, remaining roadmap efforts focus on:

1. Performance profiling under high-frequency data replay to document CPU and memory envelopes (acceptance: sustain 4× replay speed for 3 hours with <70% CPU, no missed cadences).
2. Expanding live data ingest connectors beyond synthetic slices to continuous market feeds (acceptance: 24-hour uninterrupted capture with ≤0.1% packet loss and validated PSD covariance checks).
3. Authoring acceptance criteria for multi-day dry runs and preparing governance review packets for final sign-off (acceptance: 72-hour paper run with Sharpe proxy ≥0.9, zero unmitigated incidents, governance sign-off archived).

## 9. Conclusion
AlphaTrade now operates as a cohesive, explainable trading intelligence capable of sustained paper execution. The platform integrates perception, adaptation, reflection, execution, and governance in a single loop, backed by guardrail tests and supervisory controls. Future work will extend live connectivity, deepen evolutionary strategy generation, and formalize publication artifacts.

## Appendix A. Glossary
- **BeliefState:** Data structure capturing posterior beliefs, regimes, and covariance for downstream routing.
- **DecisionDiary:** Persistent log of each decision’s context, action, and outcome.
- **Fast Weights:** Short-term Hebbian adjustments applied to strategy preference vectors.
- **TradeThrottle:** Governance component that limits order cadence to protect broker and capital.
- **Understanding Diagnostics:** Visualization utilities that export DAG representations of the decision pipeline.

## Appendix B. Simulation Configuration
- **Data Source:** Synthetic EURUSD tick slices replayed through Timescale → Kafka harness.
- **Strategies Enabled:** `atlas_meanrev_v2`, `nova_momentum_v1`, `sentinel_volbreak`.
- **Risk Settings:** 1 contract max position, 0.5% capital-at-risk per trade, throttle = 1 trade/minute.
- **Environment:** Dockerized runtime with Redis and Kafka harness, paper API mock responding with real latency distribution sampled from broker telemetry.

## Appendix C. Reproducibility Checklist
- **Repository State:** Tag or commit at publication time; record via `git rev-parse HEAD` when executing harness.
- **Harness Invocation:** `poetry run python tools/run_paper_harness.py --symbol EURUSD --start 2024-02-12T00:00Z --duration 3600 --runs 5 --fast-weights {on|off}`.
- **Container Image:** `alphatrade/replay:2024.02` (Dockerfile hash `sha256:8f32...`).
- **Random Seeds:** `alpha_seed=4242`, `market_seed=1138`, `latency_seed=77`.
- **Resource Limits:** BLAS threads pinned to 2; Python 3.11.7; Redis 7.2; Kafka 3.6; environment hash recorded in Decision Diary headers.
- **Artifacts:** Decision Diaries, metrics Markdown (`artifacts/eurusd_paper/metrics_run_*.md`), and throttle incident logs stored under `artifacts/eurusd_paper/` for each replay.
