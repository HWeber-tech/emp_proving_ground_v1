# AlphaTrade Whitepaper (Draft)

## Executive Summary
AlphaTrade is an intelligent trading loop that fuses sensory perception, adaptive decision making, reflective learning, risk-aware execution, and governance controls into a cohesive platform. The system ingests heterogeneous market telemetry, synthesises beliefs about prevailing regimes, routes strategies with fast-weight adaptation, records transparent decision diaries, and enforces safety constraints before placing trades through broker adapters. This draft captures the current implementation, the operational readiness of the paper-trading stack, and the remaining gaps required for a controlled launch.

## System Goals & Design Principles
- **Closed-loop intelligence.** Each iteration of the AlphaTrade loop should observe markets, understand context, act, and learn, using perception → adaptation → reflection → execution → governance as the organising backbone.
- **Explainable autonomy.** Decisions are logged via decision diaries, policy reflection artefacts, and understanding graphs to surface rationale and fast-weight effects.
- **Safety and resilience first.** Trade throttles, governance promotion gates, and event-bus failover routines ensure the system degrades gracefully under stress.
- **Composable infrastructure.** Components are separated by domain boundaries (sensory, understanding, trading, governance) with dataclass or Pydantic contracts to promote reuse and rigorous testing.

## Architecture Overview
AlphaTrade follows the layered architecture encoded in the repository:

1. **Perception Layer** – Canonical sensors live in `src/sensory`, and the `RealSensoryOrgan` fuses WHAT/WHEN/WHY/HOW/ANOMALY signals into integrated sensory snapshots. Drift detection, lineage publication, and audit trails are embedded so downstream components can trust signal provenance.
2. **Belief & Understanding Layer** – Belief states incorporate regime inferences and fast-weight toggles. The `UnderstandingRouter` coordinates policy routing, applying feature-gated fast-weight adapters before delegating to tactic selection.
3. **Adaptation Layer** – Policy routers, tactical adaptation engines, and evolution scaffolding (under `src/thinking/adaptation`) manage strategy lifecycles, metadata, and experiment gating. Fast-weight adapters apply Hebbian updates to emphasise recent evidence.
4. **Reflection Layer** – Decision diaries (`src/understanding/decision_diary.py`) persist each decision’s context, outcomes, probes, and belief snapshots. Reflection builders aggregate diaries into governance-ready summaries, while diagnostics export DAG views of the understanding loop.
5. **Execution & Risk Layer** – Execution is orchestrated under `src/trading`, where the `TradeThrottle` enforces per-window trade limits, and the `trading_manager` binds throttle posture into live routing. Integrations provide adapters for paper trading APIs.
6. **Governance & Safety Layer** – Governance packages oversee policy promotion, audit trails, and safety managers. They regulate which strategies can graduate to production and ensure configuration drift is tracked.

### Data Flow Summary
1. **Ingest & Fusion:** Market frames, order books, macro narratives, and anomaly payloads enter the `RealSensoryOrgan`, generating integrated sensory snapshots with lineage metadata.
2. **Belief Updates:** Belief states consume sensory output, maintain PSD covariance checks, and toggle fast-weight eligibility.
3. **Routing & Action:** The `UnderstandingRouter` adjusts tactic weights using Hebbian adapters and returns a decision enriched with fast-weight summaries.
4. **Risk & Execution:** Decisions traverse governance filters and throttles before reaching the execution adapters. When in paper mode, the paper trading API adapter submits REST orders to the configured sandbox endpoint.
5. **Reflection & Learning:** Outcomes loop back via decision diaries, strategy performance tracking, and diagnostics to inform future policy tuning and governance reviews.

## Perception & Belief Formation
- **Sensory organs** (`what`, `when`, `why`, `how`, `anomaly`) expose `process` methods returning canonical signal structures. Each emits lineage and telemetry metadata, ensuring observability by design.
- **Fusion & Drift Monitoring:** `RealSensoryOrgan.observe` captures sensor outputs, builds integrated signals, publishes lineage through the event bus, and keeps rolling audit trails for drift evaluation. The optional `SensoryDriftConfig` enforces minimum sample windows and z-score thresholds.
- **Eventing & Failover:** Sensory payloads can publish to the global topic bus; failure paths are mitigated via `publish_event_with_failover`, allowing the perception layer to continue operating even when downstream queues are degraded.
- **Belief snapshots:** The belief module captures symbol context, feature vectors, and fast-weight eligibility flags, enabling routers to calculate multipliers conditioned on real-time regime factors.

## Adaptation & Fast-Weight Routing
- **Policy Router:** Strategies register with the policy router alongside metadata such as guardrails and experiment tags. Routing computes tactic selection based on belief-regime compatibility and optional fast-weight overrides.
- **Fast-weight adapters:** Each `FastWeightAdapter` declares an adapter ID, target tactic, multipliers, and optional Hebbian configuration. When a belief snapshot marks fast weights enabled, adapters update multipliers based on feature activations, providing sparse, positive adjustments aligned with BDH heuristics.
- **Experiment governance:** Adaptation hooks integrate with governance toggles so that high-risk adapters or evolution pipelines only activate when authorised. This ensures experimental tactics cannot bypass safety checks.
- **Evolution scaffolding:** While a full evolutionary engine remains future work, the current scaffolding hosts placeholders for mutation pipelines and catalogue integration, providing extension points for future research.

## Reflection & Knowledge Capture
- **Decision diaries:** Every loop iteration creates a `DecisionDiaryEntry` capturing decision details, guardrail states, outcomes, and optional belief metadata. Markdown exports support human audits.
- **Policy reflection artefacts:** Aggregations summarise tactic utilisation, promotions, demotions, and gating reasons, feeding governance review sessions.
- **Understanding diagnostics:** Graph exports illustrate the end-to-end reasoning chain, highlighting active adapters, sparsity metrics, and dominant strategies. These diagnostics are used in guardrail tests to ensure interpretability.
- **KPIs & loop metrics:** The `StrategyPerformanceTracker` compiles ROI, win rates, drawdowns, and fast-weight uplifts per strategy. Loop metrics report regime accuracy and drift alert statistics, providing quantitative insight into cognition quality.

## Execution, Risk, and Paper Trading
- **Trade throttle:** `TradeThrottle` enforces configurable rate limits with optional cooldowns and size multipliers. Snapshots expose state, reasons for throttling, and retry suggestions for journaling and monitoring.
- **Trading manager integration:** The trading manager consumes throttle decisions, risk checks, and governance verdicts before emitting execution intents. It exposes throttle posture via runtime APIs so operators can inspect rate limiting in effect.
- **Paper trading adapter:** `PaperTradingApiAdapter` converts AlphaTrade orders into REST payloads, handling authentication headers, timeout management, idempotent client order IDs, and session lifecycle. Integration tests validate order submission, error propagation, and cleanup.
- **Fail-safe execution:** Supervisor routines catch adapter exceptions and favour “no-trade” failover rather than crashing the loop, ensuring paper sessions continue despite transient broker issues.

## Data Backbone & Infrastructure Readiness
- **Synthetic-to-live migration:** Current integration tests stream synthetic EURUSD slices through the Timescale → Kafka → RealSensoryOrgan pipeline, demonstrating end-to-end wiring. Production readiness requires swapping synthetic publishers for live market feeds and securing credentials for broker sandboxes.
- **Configuration management:** Environment templates capture required secrets (`PAPER_TRADING_API_URL`, keys, etc.). Pydantic models validate configuration at boot, preventing silent misconfiguration.
- **Observability hooks:** Event bus failover metrics, throttle snapshots, and diary exports provide structured data for dashboards. Logging defaults to JSON-friendly formats to support ingestion into central observability stacks.

## Paper Trading Simulation Plan
1. **Environment setup:** Provision sandbox credentials, configure `PaperTradingApiSettings`, and enable the paper broker execution path during bootstrap.
2. **Dry-run rehearsal:** Execute short test sessions using replayed historical data to validate order placement, throttle responses, and diary logging.
3. **Live paper session:** Run a full trading day using live market feeds. Monitor throughput, ensure at least one trade attempt, and verify diary entries capture both attempted and throttled trades.
4. **Post-run analysis:** Generate KPI reports, extract diary excerpts, and review throttle snapshots alongside governance logs to certify behaviour.

## Performance & Reliability Considerations
- **Throughput safeguards:** Profiling should target sensory fusion, belief updates, and routing loops—ensuring real-time data does not backlog. Vectorised operations and asyncio boundaries are in place, but sustained load testing will confirm headroom.
- **Resource monitoring:** Capture CPU, memory, and event-loop lag metrics during simulations. Compare runs with fast weights enabled vs disabled to quantify overhead.
- **Resilience drills:** Simulate broker outages, event bus failures, and sensor anomalies to validate graceful degradation pathways.

## Observability & Reporting
- **Daily KPI reports:** `StrategyPerformanceReport.to_markdown` produces operator-friendly summaries enumerating per-strategy ROI, win rates, and drawdowns. These should accompany diary digests after each simulation.
- **Loop diagnostics:** Understanding graph dumps should be archived for each session, highlighting adapter utilisation and sparsity trends.
- **Governance ledger:** Policy change logs and promotion decisions must remain synchronised with diary evidence to satisfy audit requirements.

## Roadmap & Future Work
- **Live data integration:** Replace synthetic streams with production-grade connectors for TimescaleDB, Redis, or direct broker feeds.
- **Evolution engine:** Implement strategy mutation pipelines leveraging performance telemetry to generate and test new tactics automatically.
- **Automated reflection feedback:** Close the loop by feeding KPI deltas and diary insights back into router weight adjustments or governance triggers.
- **Extended acceptance testing:** Conduct multi-day paper runs (target: ≥3 consecutive days) with automated anomaly detection for diaries, throttle, and performance metrics.
- **Publication packaging:** Convert this draft into a polished whitepaper with diagrams, benchmarking charts, and appendices detailing experiment methodologies.

## Appendices
- **Glossary:** Maintain a shared vocabulary for sensory organs, belief metrics, fast weights, and governance states to ensure cross-team clarity.
- **Testing Artefacts:** Link guardrail tests, integration suites, and regression fixtures relevant to the paper trading stack for reproducibility.
- **Operational Runbooks:** Provide scripts and command sequences for bootstrapping paper sessions, rotating credentials, and performing incident response during simulations.

