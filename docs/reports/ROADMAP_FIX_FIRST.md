## EMP FIX-First Implementation Roadmap

### Current Status (snapshot)
- FIX is the exclusive broker pathway; OpenAPI disabled at entry points and config.
- Demo FIX flows green end-to-end:
  - Logon/auth for price and trade sessions.
  - Market Data Request (35=V) confirmed; snapshot arrives; incremental handler present.
  - Order placement (limit/market) with correct numeric 55 and min volume.
  - Cancel pipeline hardened: 11/41 minimal, conditional 37 retry, state-gated with optional 35=H probe.
- Session-tracked cancel-all utility available; venue-level 35=q rejected (expected on demo).
- Config loading and Pydantic compat fixed; demo scripts run with minimal delays and UTC timestamps.

### Architectural Principle
- All analysis belongs to a Sense (4D+1 Sensory Cortex). Integration layers only transport and normalize data.

---

### Phase 0 — FIX Hardening and Completeness (Production-Grade)
- FIX Core
  - Reconnect and session recovery (seq numbers, 2=ResendRequest, 4=GapFill).
  - Heartbeat/TestRequest coverage; clock-drift monitoring; latency tracing.
  - Persist orders/session state (disk/Redis) for crash-safe recovery and idempotent replays.
  - Full MD parsing: proper 268 repeating groups (279/269/270/271) and depth maintenance.
  - Order lifecycle: Cancel/Replace (35=G), TIF variants, standardized error taxonomy.
  - Position/Account sync: venue-supported queries or deterministic tracking + reconciliation hooks.
- Reliability & Limits
  - Rate limiting, pacing, backoff; defensive timeouts and retries.
  - Structured logs with correlation IDs; Prometheus metrics (FIX I/O, rejects, ER latency).
- Deliverables
  - Green tests for: MD subscribe/unsubscribe; NOS, CANCEL, REPLACE, REJECT; recovery after disconnect.
  - Internal status endpoints and dashboards with FIX/session metrics.

### Phase 1 — Sensory Cortex Integration (4D+1) and Data Contracts
- Contracts
  - Define normalized events from `FIXSensoryOrgan`: quotes, trades, book updates, order events.
- Senses
  - WHAT: price/volume structure; L1/L2 features (spreads, imbalance, liquidity pockets).
  - WHEN: session/volatility regimes; market state timing.
  - HOW: microstructure flow, execution cost surfaces, slippage risk.
  - WHY: macro/calendar scaffolding via economic organ.
  - ANOMALY: outlier detectors on spreads/prints/book churn.
- Deliverables
  - Feature manifests, unit tests per sense, latency budgets; benchmark report in this folder.

### Phase 2 — Strategy and Execution Cohesion
- Strategy Engine
  - Wire senses to decision loop via typed inputs; guardrails (kill-switches, drawdown caps).
- OMS/Risk
  - Pre-trade validation (min vol/tick, exposure, per-instrument rules).
  - Post-trade audits; PnL attribution; failure escalation.
- Execution
  - Passive/aggressive routing, retry and slip tolerances; replace/cancel policies.
- Deliverables
  - Integration tests: decide → route → broker ER → portfolio update.
  - Paper portfolio tracking parity with broker fills.

### Phase 3 — Backtesting, Simulation, Honest Validation
- Data Foundation
  - Capture FIX feed to parquet; deterministic replayer (L1/L2).
- Simulators
  - Microstructure sim for fill modeling; stress injectors (latency/burst/packet loss).
- Validation
  - Honest validation with leakage checks; reproducible reports in this folder.
- Deliverables
  - Backtest harness producing confidence bands; CI gate on validation KPIs.

### Phase 4 — Paper and Live Ops
- Paper Trading
  - Continuous demo with SLOs (uptime, ER latency, cancel success rate).
  - Alerting: abnormal rejects, MD gaps, seq gaps, risk breaches.
- Go-Live (staged)
  - Playbook: credential rotation, canary symbols, roll-forward/rollback scripts.
  - Observability: Grafana dashboards fed by Prometheus metrics.
- Deliverables
  - Runbooks; on-call alerts; daily risk/pnl reports in this folder.

### Phase 5 — Scale and Evolution
- Performance
  - Multi-symbol concurrency; batching; zero-copy MD hot path.
  - Horizontal scaling (k8s): separate quote/trade pods, stateless workers.
- Evolutionary Intelligence
  - Evolve strategy parameters/structures; antifragile adaptation loops.
  - MLOps: dataset lineage, registries, live A/B with caps.
- Deliverables
  - Throughput/latency SLAs; evolutionary cycles with measurable uplift.

---

### Immediate Next Actions (1–2 sprints)
- FIX
  - Seq recovery + resend/gapfill + reconnect; persist order/session state.
  - Complete MD incremental parser; validate order book integrity.
- Sensory
  - Finalize sensory event contracts; move L1/L2 derivations into `src/sensory/`.
- Tests/Monitoring
  - Deterministic tests for replace/cancel/timeout/reject; ER latency SLOs; Prometheus metrics + Grafana.

### Exit Criteria per Phase (abridged)
- Phase 0: FIX flows green under disconnect/reconnect; persisted state; full MD parser; metrics online.
- Phase 1: Senses emit validated features with latency budgets; zero analysis outside `src/sensory/`.
- Phase 2: Strategy round-trip to broker with risk gates; portfolio parity proven.
- Phase 3: Honest validation reports reproducible; CI gate enforced.
- Phase 4: 2-week stable paper ops; staged live playbook executed.
- Phase 5: Horizontal scale proven; evolutionary cycle uplift documented.


