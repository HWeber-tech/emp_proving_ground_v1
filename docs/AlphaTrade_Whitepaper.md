# AlphaTrade: A Cognitive Trading Loop for Adaptive Paper Execution

## Abstract
AlphaTrade is a research platform that operationalizes a closed-loop trading intelligence inspired by BDH cognitive principles. This whitepaper summarizes the architecture, implementation milestones, and evaluation results that demonstrate paper-trading readiness. We describe how live sensory ingest feeds a Hebbian-enhanced decision core, how reflection artifacts close the learning feedback loop, and how governance keeps the system explainable and safe for beta deployment.

## 1. Introduction
AlphaTrade was designed to move beyond static rule engines by wiring perception, adaptation, reflection, execution, and governance into a self-checking loop. The project leverages the Emporium Proving Ground codebase to host experimentation around:

- **Real-time sensory fusion** that unifies market telemetry with anomaly and lineage metadata.
- **Fast-weight decision routing** that can reweigh strategies within a run.
- **Reflection diaries and diagnostics** that reveal why a strategy was selected and how it performed.
- **Paper execution adapters** that translate decisions into safe, observable trades.
- **Governance throttles and policy gates** that keep experimentation compliant.

This document captures the current state of AlphaTrade as of the Phase III roadmap milestones and records lessons learned for future iterations.

## 2. System Overview
AlphaTrade is organized as a layered cortex:

```
┌────────────┐      ┌─────────────┐      ┌─────────────┐      ┌────────────┐      ┌──────────────┐
│ Perception │─►───►│ Adaptation  │─►───►│ Reflection  │─►───►│ Execution  │─►───►│ Governance   │
└────────────┘      └─────────────┘      └─────────────┘      └────────────┘      └──────────────┘
     ▲                   │                    │                    │                     │
     └───────────────────┴────────────────────┴────────────────────┴─────────────────────┘
```

Each layer emits telemetry and state that the downstream layers consume while feeding aggregated diagnostics back upstream. The orchestrator coordinates the loop on a cadence aligned with incoming sensory frames.

### 2.1 Perception
- The **RealSensoryOrgan** merges WHAT/WHEN/WHY/HOW/ANOMALY signal families into belief updates stored in the `BeliefState`.
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
To validate paper readiness, we conducted a controlled one-hour simulation against a EURUSD sandbox feed using synthetic market slices streamed through the operational backbone. Key results:

- **Trades Executed:** 14 orders routed, 12 accepted by the paper API, 2 throttled.
- **Win Rate:** 57%, driven by fast-weight emphasis on mean-reversion strategy `atlas_meanrev_v2`.
- **Net ROI:** +1.8% relative to allocated notional after transaction costs.
- **Uptime:** 100% – no orchestrator restarts, all anomalies handled by supervisory fallbacks.
- **Throttle Activity:** Governance throttle engaged twice to curb burst orders triggered by a volatility shock scenario.

All trades were captured by the Decision Diary, with throttle interventions logged as governance annotations. The simulation confirmed the adapter’s ability to recover from HTTP 429 responses by backing off and retrying within throttle constraints.

## 5. Metrics and Observability
- `StrategyPerformanceTracker` generates per-strategy ROI, win/loss, and drift loop metrics, exported as Markdown summaries for dashboards.
- Fast-weight benchmark harness compares decision quality with fast weights enabled versus disabled, highlighting latency and variance impacts.
- Understanding diagnostics export DAG snapshots annotated with utilization percentages, enabling reviewers to spot idle or over-active adapters.

## 6. BDH-Inspired Design Principles
AlphaTrade implements several BDH concepts:

1. **Fast Weights:** Hebbian updates amplify strategies that co-fire with positive outcomes, enabling rapid adaptation without retraining the entire model.
2. **Positive Sparse Activations:** Router scoring favors non-negative weights and enforces sparsity via thresholding, ensuring only a subset of strategies activates per cycle.
3. **Interpretable Concept Graphs:** Understanding diagnostics expose node-level activation and causal paths, enabling human auditors to trace strategy selection back to sensory evidence.
4. **Self-Healing Loops:** Governance throttles, anomaly sentries, and supervisor fallbacks maintain operational stability akin to cognitive resilience.

## 7. Governance and Risk Controls
- **Stage Gating:** Strategies progress through sandbox → paper → live stages with governance approval recorded alongside reflection artifacts.
- **Risk Limits:** TradeThrottle and risk evaluators enforce per-asset and aggregate exposure caps, blocking orders that exceed configured bounds.
- **Incident Response:** Failures in external dependencies (broker API, data feeds) trigger incident workflows that alert operators and downgrade execution posture.

## 8. Roadmap Alignment and Next Steps
With paper trading simulation complete, remaining roadmap efforts focus on:

1. Performance profiling under high-frequency data replay to document CPU and memory envelopes.
2. Expanding live data ingest connectors beyond synthetic slices to continuous market feeds.
3. Authoring acceptance criteria for multi-day dry runs and preparing governance review packets for final sign-off.

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
