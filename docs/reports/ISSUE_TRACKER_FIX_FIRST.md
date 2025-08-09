## FIX-First Issue Tracker (Comprehensive To-Do)

Legend: [P0]=critical, [P1]=high, [P2]=normal, [P3]=nice-to-have

### Section A — FIX Core and Session Management
- [ ] [P0] FIX-001: Reconnect & session lifecycle
  - [x] Reconnect on disconnect; exponential backoff
  - [x] Clean shutdown without socket errors
  - [x] Burn-in script `scripts/fix_burn_in.py`; target: 100 cycles w/o leaked threads or 10038 errors

- [ ] [P0] FIX-002: Sequence number recovery
  - [ ] Implement ResendRequest (35=2) handling and GapFill (35=4)
  - [ ] Out-of-order and duplicate detection
  - Acceptance: Inject seq gaps; fully heals; no state loss

- [ ] [P1] FIX-003: Heartbeat/TestRequest hardening
  - [ ] Missed heartbeat detection with TestRequest (35=1)
  - [ ] Clock drift guard; latency histogram
  - Acceptance: Alerts/logs on heartbeat delay; metrics exposed

- [ ] [P1] FIX-004: Persistent session + order state
  - [x] Write-through cache to disk for `self.orders` and seq state (JSON store)
  - [x] Redis adapter parity (optional)
  - [x] Crash-safe restart; idempotent replays (status probe on start)
  - Acceptance: Kill/restart during active order; state reconciled

### Section B — Market Data (MD) Parsing and Integrity
- [ ] [P0] MD-001: Full incremental parser (W/X)
  - [x] Properly parse 268 repeating groups with 279/269/270/271
  - [x] Maintain depth correctly; side-aware updates
  - [x] Entry-ID aware updates/deletes; price fallback
  - Acceptance: Deterministic book after scripted deltas

- [ ] [P1] MD-002: Subscription lifecycle & recovery
  - [ ] Unsubscribe, resubscribe, recover after quote session drop
  - [ ] Per-symbol health; auto re-subscribe
  - Acceptance: Fuzz disconnect → book consistent within SLO

- [ ] [P2] MD-003: Feed capture & replay hooks
  - [ ] Persist snapshots and deltas to parquet
  - [ ] Minimal replayer to drive backtest/sim

### Section C — Order Lifecycle & Risk
- [ ] [P0] ORD-001: NOS/Cancel/Replace completeness
  - [x] Replace (35=G) support with validations
  - [x] TIF normalization; price/qty alignment to venue constraints
  - Acceptance: Integration tests for New/Cancel/Replace/Reject

- [ ] [P0] ORD-002: Cancel gate + status probe (done; extend tests)
  - [ ] Gate on OrdStatus ∈ {New, PendingNew}
  - [ ] Optional 35=H probe; skip when Filled/Closed
  - Acceptance: Deterministic tests; zero spurious ORDER_NOT_FOUND

- [ ] [P1] ORD-003: Pre-trade risk checks
  - [ ] Min vol, tick, exposure, per-instrument caps
  - [ ] Kill-switch hooks
  - Acceptance: Violations blocked; logged; metrics emitted

### Section D — Observability & Tooling
- [ ] [P0] OBS-001: Structured logging & correlation IDs
  - [ ] Message-level trace IDs; per-clOrdID correlation
  - [ ] Redact credentials in logs

- [ ] [P1] OBS-002: Prometheus metrics
  - [x] Counters: messages by type, rejects, reconnects
  - [x] Histograms: ER latency, cancel latency
  - [x] Gauges: MD staleness, session connectivity
  - [x] HTTP `/metrics` endpoint
  - [x] Grafana dashboard JSON committed under docs/reports

- [ ] [P2] OBS-003: Health/status endpoints
  - [ ] Internal HTTP with session/queue stats

### Section E — Sensory Cortex Integration (All analysis in senses)
- [ ] [P1] SNS-001: Event contracts from `FIXSensoryOrgan`
  - [ ] Typed schemas for quotes, trades, book updates, order events
  - Acceptance: Versioned schemas; consumer tests

- [ ] [P1] SNS-002: WHAT sense features
  - [ ] L1/L2: spread, imbalance, liquidity pockets, volatility seeds

- [ ] [P2] SNS-003: WHEN/HOW/WHY/ANOMALY scaffolding
  - [ ] Regime timing, microstructure flow, macro hooks, anomaly detectors

### Section F — Strategy, OMS, and Portfolio
- [ ] [P1] STR-001: Decision loop wiring
  - [ ] Define inputs/outputs; guardrails (drawdown caps)

- [ ] [P1] OMS-001: Post-trade audits & PnL attribution
  - [x] ER fill event hook for portfolio updates
  - [x] Portfolio tracker consuming order callbacks; basic PnL
  - [x] Broker parity checks (framework + metrics)

### Section G — Backtesting, Simulation, Validation
- [ ] [P1] SIM-001: Backtest harness w/ replay
  - [ ] Deterministic runs from captured MD; CI jobs

- [ ] [P1] VAL-001: Honest validation framework
  - [ ] Leakage checks; reproducible reports in docs/reports

### Section H — Paper Ops and Go-Live
- [ ] [P1] OPS-001: Paper trading SLOs
  - [ ] Uptime, ER latency, cancel success; alerting pathways

- [ ] [P1] OPS-002: Staged go-live playbook
  - [ ] Canary symbols, roll-forward/rollback, creds rotation

### Section I — Scale & Evolution
- [ ] [P2] SCL-001: Concurrency & throughput
  - [ ] Multi-symbol routing; batching; zero-copy hot path

- [ ] [P2] EVO-001: Evolutionary loops (guardrailed)
  - [ ] Parameter evolution; antifragile adaptation; caps

---

### Cross-Cutting Acceptance & References
- Tests: expand `tests/integration` and `tests/unit` for FIX flows, MD integrity, risk gates
- Metrics: publish Prometheus exporter; dashboard JSON committed under `docs/reports/`
- Docs: keep `docs/reports/ROADMAP_FIX_FIRST.md` as the narrative; this file as the actionable checklist



