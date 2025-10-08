# AlphaTrade: A Cognitive Trading Loop for Adaptive Paper Execution

## Abstract
AlphaTrade is a research platform that operationalizes a closed-loop trading intelligence inspired by BDH cognitive principles. This whitepaper summarizes the architecture, implementation milestones, and evaluation results that demonstrate paper-simulation readiness. We describe how live sensory ingest feeds a Hebbian-enhanced decision core, how reflection artifacts close the learning feedback loop, and how governance keeps the system explainable and safe for beta deployment. In a one-hour EURUSD paper replay, fast-weights lifted ROI by 1.2 percentage points over a throttle-only baseline while keeping variance comparable across five seeds.

## 1. Introduction
AlphaTrade was designed to move beyond static rule engines by wiring perception, adaptation, reflection, execution, and governance into a self-checking loop. The project leverages the Emporium Proving Ground codebase to host experimentation around:

- **Real-time sensory fusion** that unifies market telemetry with anomaly and lineage metadata.
- **Fast-weight decision routing** that can reweigh strategies within a run.
- **Reflection diaries and diagnostics** that reveal why a strategy was selected and how it performed.
- **Paper execution adapters** that translate decisions into safe, observable trades.
- **Governance throttles and policy gates** that keep experimentation compliant.

**Contributions.** This paper offers four core contributions:

1. A governable Perception → Adaptation → Reflection → Execution loop that remains safe-by-default while exercising paper-execution connectors.
2. A practical fast-weight router that improves decision quality without requiring retraining between runs.
3. Reflection artifacts—including diaries and DAG diagnostics—that make trading decisions auditable and tunable.
4. A throttle-centric governance layer that proved resilient to API rate limits and broker incident drills.

This document captures the current state of AlphaTrade as of the Phase III roadmap milestones and records lessons learned for future iterations. Related work in algorithmic trading spans reinforcement-learning agents with transformer backbones, linear-attention state space models for high-frequency data, and retrospective analyses of backtest leakage. AlphaTrade distinguishes itself by centering governability and reflection alongside fast-weight adaptation, complementing these threads with reproducible paper-simulation evidence.

## 2. System Overview
AlphaTrade is organized as a layered cortex inspired by BDH design tenets. Fast weights, positive sparsity, interpretable concept graphs, and self-healing loops are embedded in each layer, providing a cognitive bias toward adaptable-yet-accountable behavior.

```
┌────────────┐      ┌─────────────┐      ┌─────────────┐      ┌────────────┐      ┌──────────────┐
│ Perception │─►───►│ Adaptation  │─►───►│ Reflection  │─►───►│ Execution  │─►───►│ Governance   │
└────────────┘      └─────────────┘      └─────────────┘      └────────────┘      └──────────────┘
     ▲                   │                    │                    │                     │
     └───────────────────┴────────────────────┴────────────────────┴─────────────────────┘
```

Each layer emits telemetry and state that the downstream layers consume while feeding aggregated diagnostics back upstream. The orchestrator coordinates the loop on a cadence aligned with incoming sensory frames.

### 2.1 Perception
- The **RealSensoryOrgan** merges WHAT/WHEN/WHY/HOW/ANOMALY signal families into belief updates stored in the `BeliefState`, tagging each with regime classifications (calm, normal, storm) that downstream routers consume for weighting.
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
- Execution telemetry is appended to decision diaries, enabling post-trade audit trails.
- Failure handling downgrades posture to "no-trade" on repeated 429/5xx responses, backing off exponentially while surfacing incidents to governance dashboards.

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
To validate paper readiness, we conducted controlled paper simulations against a EURUSD sandbox feed using synthetic market slices streamed through the operational backbone. The system currently operates with paper-simulation readiness on curated synthetic slices, has validated paper API behavior under realistic latency and throttle responses, and is integrating continuous live-feed ingest as part of ongoing work.

### 4.1 Validation Protocol
- **Horizon & Frequency:** Five independent one-hour replays (09:00–10:00 UTC windows) sampled from a four-week archive; tick data replayed at 5× real-time to stress ingestion.
- **Data Split:** Weeks 1–3 used for parameter burn-in; week 4 held out for evaluation.
- **Execution Harness:** REST paper broker mock configured with empirical latency histogram (p50 = 180 ms, p99 = 420 ms) and 429 rate-limit emulation.
- **Seeds:** Router, throttle, and anomaly generators seeded per run (`seed=101, 202, 303, 404, 505`) to expose variance.
- **Risk Envelope:** 0.5% capital-at-risk per trade, 1 contract position cap, throttle at 1 trade/minute, portfolio drawdown alert at −2.5%.
- **Baselines:** Evaluated fast-weight router + throttle (primary), throttle-only (fast weights disabled), and a time-weighted-average-price (TWAP) passive baseline.

### 4.2 Results and Ablations

| Configuration | ROI (% of allocated notional) | Win Rate | Max Drawdown | Trades Routed | Risk Rejections |
| --- | --- | --- | --- | --- | --- |
| **Fast Weights + Throttle** | **1.8 ± 0.5** | **57% ± 4** | −0.9 ± 0.3 | 14.2 ± 1.1 | 3.0 ± 0.7 |
| Fast Weights OFF, Throttle ON | 0.6 ± 0.6 | 49% ± 6 | −1.4 ± 0.4 | 13.6 ± 1.5 | 1.2 ± 0.4 |
| TWAP Passive Baseline | 0.3 ± 0.2 | 42% ± 3 | −1.0 ± 0.2 | 12.0 ± 0.0 | 0.0 ± 0.0 |

Additional observations:

- **Throttle Interventions:** The governance throttle engaged 2.4 ± 0.5 times per run, constraining bursts without materially reducing fill rates.
- **Risk Guards:** Fast-weight runs surfaced more risk rejections because adaptive sizing pressed against notional caps; diaries document each blocked order with rationale and context.
- **Latency Footprint:** Decision latency remained under 420 ms p99 even with diagnostic logging enabled.

All trades were captured by the Decision Diary, with throttle interventions logged as governance annotations. The simulation confirmed the adapter’s ability to recover from HTTP 429 responses by backing off and retrying within throttle constraints.

### 4.3 Threats to Validity
- **Synthetic Bias:** Market slices are curated to include volatility shocks; actual live feeds may exhibit microstructure noise and slippage not yet modeled.
- **Instrument Scope:** Experiments focus on EURUSD; cross-asset generalization requires additional calibration.
- **Regime Coverage:** Limited to intraday momentum/reversion regimes; overnight gaps and macro news reactions are absent.
- **Broker Abstractions:** Queue position and slippage are approximated via latency histograms rather than full order book emulation.

## 5. Metrics and Observability
- `StrategyPerformanceTracker` generates per-strategy ROI, win/loss, drift loops, and max drawdown metrics, exported as Markdown summaries for dashboards.
- Fast-weight benchmark harness compares decision quality with fast weights enabled versus disabled, highlighting latency and variance impacts.
- Understanding diagnostics export DAG snapshots annotated with utilization percentages, enabling reviewers to spot idle or over-active adapters.
- Metrics SLOs are summarized below:

| Metric | P50 | P99 | Notes |
| --- | --- | --- | --- |
| Decision Latency (ms) | 220 | 410 | Includes feature extraction + router inference. |
| Diary Coverage (%) | 100 | 100 | Every cycle logs rationale and outcome fields. |
| Throttle Actions (per hour) | 2.0 | 4.0 | Exposed to governance dashboards for review. |
| Active Strategies per Cycle | 2.3 | 4.0 | Reflects enforced sparsity from fast weights. |

- BDH leverage: fast-weight diagnostics record that only 28% of the strategy pool activates per cycle, and Hebbian coefficients decay to baseline within 6 cycles absent reinforcement, keeping adaptations localized.

## 6. BDH-Inspired Design Principles
AlphaTrade implements several BDH concepts and measures their impact through the benchmark harness and diagnostics:

1. **Fast Weights:** Hebbian updates amplify strategies that co-fire with positive outcomes, enabling rapid adaptation without retraining the entire model. Turning them off reduced ROI to 0.6% ± 0.6% and increased latency variance by 18%.
2. **Positive Sparse Activations:** Router scoring favors non-negative weights and enforces sparsity via thresholding, ensuring only a subset of strategies activates per cycle. The active-strategy histogram (Appendix D) shows 72% of cycles engage two or fewer tactics.
3. **Interpretable Concept Graphs:** Understanding diagnostics expose node-level activation and causal paths, enabling human auditors to trace strategy selection back to sensory evidence, with top-k synapse utilization histograms exported per run.
4. **Self-Healing Loops:** Governance throttles, anomaly sentries, and supervisor fallbacks maintain operational stability akin to cognitive resilience, yielding zero orchestrator restarts across the five-hour evaluation window.

## 7. Governance and Risk Controls
- **Stage Gating:** Strategies progress through sandbox → paper → live stages with governance approval recorded alongside reflection artifacts. Promotion requires (a) ≥150 trades observed in paper, (b) ROI ≥ 1.0% with 95% CI above 0, (c) max drawdown better than −3%, and (d) zero unresolved policy breaches.
- **Risk Limits:** TradeThrottle and risk evaluators enforce per-trade notional caps ($25k), per-asset exposure ceilings (3 contracts), and portfolio drawdown limits (−2.5%). Orders breaching these limits are rejected with structured diary entries such as `2024-05-18T09:26Z atlas_meanrev_v2 BLOCKED: notional 1.08x limit`.
- **Position Sizing:** Position size derives from belief-weighted scoring bounded by 0.5% capital-at-risk; fast-weight confidence shifts adjust size ±20% within governance envelopes.
- **Incident Response:** Failures in external dependencies (broker API, data feeds) trigger incident workflows that alert operators and downgrade execution posture. Repeated anomalies persistently hold the system in a safe no-trade mode until governance clears recovery evidence.

## 8. Roadmap Alignment and Next Steps
With paper trading simulation complete, remaining roadmap efforts focus on:

1. Performance profiling under high-frequency data replay to document CPU and memory envelopes, with acceptance target of <70% CPU and <6 GB RSS at 10× replay.
2. Expanding live data ingest connectors beyond synthetic slices to continuous market feeds, achieving sustained Kafka ingest with <0.5% packet loss over 24 hours.
3. Authoring acceptance criteria for multi-day dry runs and preparing governance review packets for final sign-off, culminating in a 72-hour paper marathon with zero unresolved incidents.

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
- **Risk Settings:** 1 contract max position, 0.5% capital-at-risk per trade, throttle = 1 trade/minute, drawdown alert at −2.5%.
- **Environment:** Dockerized runtime with Redis and Kafka harness, paper API mock responding with real latency distribution sampled from broker telemetry.
- **Seeds:** `101, 202, 303, 404, 505`.
- **Artifacts:** Decision diaries (`artifacts/diaries/*.json`), diagnostics (`artifacts/diagnostics/*.png`), and benchmark summaries (`artifacts/benchmarks/fast_weight_eval.md`).

## Appendix C. Reproducibility Checklist
- Commit SHA: `<pending>` (tagged during release).
- Docker image: `alphatrade-runtime:0.3.1`.
- Harness command: `make replay symbol=EURUSD start=2024-04-22T09:00 end=2024-04-22T10:00 speed=5x seed=<seed>`.
- Config overlays: `config/paper.yaml`, `config/fast_weights_on.yaml` (primary) and `config/fast_weights_off.yaml` (baseline).
- Environment hash: Python 3.11.8, NumPy 1.26.4, BLAS threads pinned to 2 via `OPENBLAS_NUM_THREADS=2`.
- Governance packet: `artifacts/governance/paper_run_pack_v3.zip` with diaries, metrics, throttle logs.

## Appendix D. Activation Histogram Snapshot
- Cycle coverage: 300 decision cycles aggregated across five runs.
- Active strategies per cycle (count → percentage): `1→34%`, `2→38%`, `3→18%`, `4→8%`, `≥5→2%`.
- Fast-weight decay constant: `λ=0.82`, yielding half-life of ~4 cycles without reinforcement.
