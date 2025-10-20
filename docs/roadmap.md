# ROADMAP.md - EMP / Alphatrade

> **North star:** human-tweakable parameters are bandaids.  
> We build an autonomous, instrument-agnostic trading system that reasons in **dimensionless units** (ticks, spreads, Ïƒ) and promotes models only through **hard, machine-readable gates**.

---

## 0) Quick start (single-box bootstrap)

- **Target deployment:** one Hetzner dedicated server (> 8 cores / 32-64 GB RAM / NVMe) for runtime + data services; your **RTX  3090 home PC** handles training.  
- **One instrument** to start (EURUSD or XAUUSD).  
- **Paper mode** first; upgrade to live only after gates are consistently green.

---

## Phase A - Fix the foundation (wiring, config, runability)

**Goal:** engine boots deterministically; mock ? paper runs end-to-end; tests & metrics exist.

### A.1 Event bus & runtime correctness
- [ ] **A.1.1** Add `emit()` alias to EventBus that forwards to `publish_async()`; deprecate later.  
- [ ] **A.1.2** Make per-instrument queues **bounded**; publish `DEGRADED_MODE` when back-pressure triggers; auto-throttle decisions to size = 0.  
- [ ] **A.1.3** Guarantee **ordering per instrument** (monotone {ts,seqno}); drop or quarantine out-of-order with a reason code.

### A.2 Config single source of truth (SoT)
- [x] **A.2.1** `SystemConfig` reads YAML first; env only overrides explicitly set fields.  
- [x] **A.2.2** In `mock` mode **skip live-cred validation**; provide `examples/run_mock.sh`.  
- [ ] **A.2.3** Normalize `connection_protocol` once (`.lower().strip()`); remove `CONNECTION_PROTOCOL` drift.

### A.3 Market access & sensory organs
- [x] **A.3.1** Implement FIX **MarketDataRequest** subscribe/unsubscribe on start/stop.  
- [ ] **A.3.2** Parse W/X messages: build L1-L5 snapshots; emit "market_data_update" with `{bid,ask,bid_sz,ask_sz,depth[L],ts,seq}`.  
- [ ] **A.3.3** cTrader adapter: replace missing domain events with typed payloads or plain bus events; ensure async usage is correct.

### A.4 Minimal compose & health
- [x] **A.4.1** `docker-compose.yml` services: **TimescaleDB**, **Redis**, **Kafka**, **engine**.  
- [x] **A.4.2** `/health` + `/metrics` (Prometheus): `event_lag_ms`, `queue_depth`, `p50/90/99_infer_ms`, `drops`, `risk_halted`.

### A.5 Tests & validation
- [x] **A.5.1** Unit tests for: MDEntry parsing; bus ordering; config precedence.  
- [x] **A.5.2** Turn `system_validation_report` into **CI**; build fails if any check red.  
- [x] **A.5.3** Archive deprecated logs/docs under `archive/` and pin a **truthful README** (mock-paper status).

**Definition of Done (A):** `docker compose up` runs mock end-to-end; FIX replay test passes; `/metrics` exposes latency & queue stats; CI green.

---

## Phase B - Perception: translation adapter (dimensionless invariants)

**Goal:** instrument-agnostic **MarketState** emitted on every event, no leakage.

### B.1 Depth & TOB features
- [x] **B.1.1** **Tick-space depth**: flip ask axis (best at index  0), **share** conv weights; 1-D conv → GLU → adaptive pool to **D=8-16** dims.  
- [x] **B.1.2** Emit `has_depth` and **zero-mask** depth features when absent (spot FX).  
- [x] **B.1.3** Derived features (leak-free):  
  - `microprice`, `spread_ticks > 1`, `ofi_norm` (pre-event states + trade sign), `refresh_hz`, `stale_ms`, `slope/curve` (2-term poly each side).

### B.2 Meta/context tokens (data-driven)
- [ ] **B.2.1** `asset_class`: {equity, fx_fut, fx_spot}; `venue`: {nasdaq, globex, spot_agg}.  
- [ ] **B.2.2** `liquidity_bucket`: daily quantiles by median L1 size/spread → {low, mid, high}.  
- [x] **B.2.3** `session`: {Asia, London, NY, auction_open, auction_close, halt/resume}.

### B.3 Targets & guards
- [x] **B.3.1** Robust σ: EWMA(|returns|) or rolling MAD×1.4826 over past‑only window  
- [x] **B.3.2** Dimensionless delta_hat = (mid[t+H]-mid[t])/(tick*max(spread, kσ).  
- [ ] **B.3.3** Dual horizons: event-time {1,5,20} **and** wall-time {100ms, 500ms, 2s}.  
- [x] **B.3.4** Daily **class prior** estimation for `pos_weight`; no future peeking.  
- [ ] **B.3.5** Unit test: masking future data must not change features/labels.

**DoD (B):** Adapter emits **MarketState** with invariants; leakage tests pass; features align across instruments (smoke PSI < 0.25).

---

## Phase C - Backbone & heads (Mamba-3, streaming, TBPTT)

**Goal:** long memory, sub-ms inference, stable training.

### C.1 Backbone swap & toggles
- [x] **C.1.1** `BackboneSSM(impl="mamba2"|"mamba3")` with identical `forward(x,state)`.  
- [x] **C.1.2** Keep local-attention sandwich; add **RMSNorm + layer-scale 0.1**.  
- [x] **C.1.3** YAML toggles: `model.ssm_impl: mamba3`, `fallback_impl: mamba2`; auto-fallback on latency fail.

### C.2 True streaming state
- [x] **C.2.1** **Per-instrument state table** (pinned); TTL & reset on session boundary/gaps/halts.  
- [x] **C.2.2** Determinism: no dropout live; state versioned by model hash; hot-reload invalidates state.  
- [x] **C.2.3** Clone-state API for planner (no mutation).

### C.3 Chunked BPTT
- [x] **C.3.1** Trainer chunker: **burn-in B=512**, **train_len T=2048**; carry state, **detach** at chunk edges.  
- [ ] **C.3.2** Curriculum seq_len: 4k → 8k → 16k.  
- [x] **C.3.3** Optimizer: AdamW lr=2e-4 cosine; grad_clip=1.0.

### C.4 Heads & calibration
- [x] **C.4.1** Tiny **per-domain heads**; optional shared head + per-domain affine/temperature.  
- [ ] **C.4.2** Add **quantile head** (τ=0.25/0.5/0.75).  
- [x] **C.4.3** Calibrate: temperature scaling or isotonic on held-out day; report **ECE** & **Brier**.

**DoD (C):** p99 model latency â‰¤ **0.35 ms**, total â‰¤ **0.85 ms**; training converges at 16k; ECE/Brier non-worse vs baseline.

---

## Phase D — Risk, execution, and slow context

**Goal:** trade only when after?fee edge beats costs; size by uncertainty; be sane in macro.

### D.1 Decision rule & costs
- [x] **D.1.1** edge_ticks = delta_hat * max(spread, kσ); cost = ½·spread + slip + fees + AS_penalty.
- [x] **D.1.2** Actions: {cross, post‑and‑chase(±1 tick, TTL X ms), hold}.

### D.2 Queue & adverse selection
- [x] **D.2.1** L1 queue fill prob ~ our_size / (queue_size+?) × trade_flow_factor.  
- [x] **D.2.2** **Adverse selection**: microprice drift over last k events conditional on our action.

### D.3 Sizing & inventory
- [x] **D.3.1** Size ∝ edge / σ̂ (from quantile head).  
- [x] **D.3.2** Inventory as a state with mean?reversion pressure; turnover caps per minute/hour.

### D.4 Slow context (OpenBB)
- [ ] **D.4.1** Macro/vol/earnings → **size multiplier** ∈ {0, 0.3, 1}.  
- [ ] **D.4.2** Macro event ±120?s → 0; VIX>35 → 0.3; else 1.  
- [ ] **D.4.3** Emit reason codes for gate decisions.

**DoD (D):** execution sim shows after?fee alpha ? thresholds; macro halts respected; inventory bounded under stress.

---

## Phase E — Training strategy: LOBSTER pretrain ? FX finetune

**Goal:** bilingual skill without forgetting; promotion by hard gates.

### E.1 LOBSTER pretrain
- [x] **E.1.1** Multi?task losses: Huber/quantile + direction + big?move; aux: next?event, masked?depth, queue?depletion.  
- [x] **E.1.2** Eval by horizon (events+time); report ECE, Brier, alpha?after?fees.

### E.2 FX adaptation
- [x] **E.2.1** Freeze bottom 60–80%; enable **LoRA rank 8–16** on top 30–40%.  
- [x] **E.2.2** **EWC or L2?SP** + **20–30% equity rehearsal**.  
- [ ] **E.2.3** Retention gates per horizon; reject if any exceed cap.

**DoD (E):** Equity retention drop ≤ {1ev 3%, 5ev 4%, 20ev 5%}; FX gains ≥ 3% F1 with matched turnover.

---

## Phase F — Planner & league (foresight + robustness)

**Goal:** look ahead a few steps under a tight budget; harden against exploitation.

### F.1 MuZero?lite planner
- [x] **F.1.1** Learn compact state?transition: next `MarketState` essentials + reward proxy.  
- [ ] **F.1.2** Depth-2/3 MCTS over actions {cross, post, hold}; budget **0.3–0.5?ms**; auto?disable if SLA breached.  
- [ ] **F.1.3** Gate: correlation between imagined edge and realized edge ≥ 0.2 on hold?out day.

### F.2 Mini?league self?play
- [x] **F.2.1** League {**Current**, **Best**, **Exploit**, **Chaos**}.  
- [x] **F.2.2** Replay buffers: main + **rare-regime** (NFP, halts); 80/20 sampling with temp schedule.  
- [ ] **F.2.3** **Lagrangian constraints** for turnover/inventory variance; no manual tuning.  
- [x] **F.2.4** Exploitability metric (ΔSharpe vs Best/Exploit at matched turnover); promote only if gap shrinks WoW.

**DoD (F):** planner respects latency gates; exploitability gap narrows without turnover spike.

---

## Phase G — Surrogate simulator & capacity

**Goal:** fast stress testing without lying to ourselves.

- [ ] **G.1** GraphNet surrogate trained on your event sim; **5–10× faster** rollouts.  
- [x] **G.2** A/B validator: surrogate vs ground‑truth sim **α within 5%**, turnover within 10%; otherwise retrain.  
- [x] **G.3** Capacity sweep: ensure strategy size never exceeds, say, 2% of L1 depth percentile.

**DoD (G):** surrogate gated; capacity curves published per instrument.

---

## Phase H — Observability, drift, chaos

**Goal:** you can see why every action happened; system survives faults.

- [x] **H.1** Action logs: `{reason_code, edge_ticks, cost_to_take, context_mult, inventory, latency_ms}`.  
- [x] **H.2** Drift monitors: **PSI** for 8–12 core features; alert if PSI > 0.25.  
- [x] **H.3** Dumb baseline comparator (e.g., 1×spread mean?revert); alert on sustained underperformance.  
- [ ] **H.4** Chaos suite: 5% event drop/dup, 300?ms stall, order rejects; engine must **flatten within 200?ms**; idempotent dedupe of `(clOrdID, execID)`.

**DoD (H):** chaos drills pass; drift alerts behave; reasons are queryable.

---

## Phase I - Acceptance gates & CI

**Goal:** promotion by YAML, not vibes.

```yaml
gates:
  retention_f1_drop_pct: {ev1: 3.0, ev5: 4.0, ev20: 5.0}
  fx_gain_f1_min_pct: {ev1: 3.0, ev5: 3.0, ev20: 3.0}
  ece_max: 0.08
  brier_max: 0.19
  alpha_after_fees_bps_min: {ev1: 0.2, ev5: 0.5, ev20: 0.7}
  session_var_spread_pct_max: 35.0
  latency_p99_ms: {adapter: 0.50, model: 0.35, total: 0.85}
  psi_max: 0.25
  turnover_match_pct: 10
  planner_edge_realization_corr_min: 0.20
  exploitability_gap_wow_delta_max: -0.0   # must shrink or be equal
actions_on_fail: ["reject", "fallback_previous_model"]
```

- [x] **I.1** CI job runs ablations ({no-depth, no-OFI, LoRA vs per-domain head, k âˆˆ {0.3,0.5,0.7}}) and enforces **all gates**.  
- [ ] **I.2** Model tagging: `APPROVED_DEFAULT` on pass; **auto-revert** otherwise.

**DoD (I):** one-button promotion with audited gates; rollback automatic.

---

## Phase J - Deployment: Hetzner single-box

**Goal:** full stack on one server post-training.

- [x] **J.1** Provision Hetzner: **8+ cores / 32-64 GB / NVMe / Ubuntu  22.04**.  
- [x] **J.2** Install Docker & Compose; deploy **TimescaleDB**, **Redis**, **Kafka**, **engine**.  
- [x] **J.3** Secure `.env` for data/broker creds; firewall; optional WireGuard.  
- [x] **J.4** Paper trading on 1 instrument; monitor CPU, RAM, I/O, p99 latency.  
- [ ] **J.5** Live promotion only after **2+ weeks** green gates (paper).

**DoD (J):** single server runs the full stack; paper loop stable with headroom.

---

## Milestone timeline (suggested)

- **Week 1-2:** Phase A + B (foundation, adapter)  
- **Week 3-4:** Phase C (Mamba-3, streaming, TBPTT)  
- **Week 5:** Phase D (execution/risk)  
- **Week 6:** Phase E (LOBSTER ? FX)  
- **Week 7:** Phase F (planner & league v1)  
- **Week 8:** Phase G/H (surrogate, chaos, observability)  
- **Week 9:** Phase I/J (gates, CI, Hetzner rollout)

---

## API & data contracts (reference)

### MarketState (dimensionless)
```
spread_ticks:int; microprice_off:float; ofi_norm:float; qimbalance:float;
refresh_hz:float; stale_ms:float; slope_bid:float; slope_ask:float;
curve_bid:float; curve_ask:float; depth_feat[8..16]:float; has_depth:bool;
session_id:int; liquidity_id:int; venue_id:int; ts:int64; seq:int64
```

### Events (bus)
- "market_data_update" ? `MarketState`  
- "order_update" ? `{clOrdID, execID, status, filled, price, ts}`  
- "risk_breach" ? `{reason, metric, value, limit, ts}`  
- "degraded_mode" ? `{reason, queue_depth, lag_ms}`

---

## Operational runbook (condensed)

- **Warm-up:** burn-in B events per symbol; size=0 until warm.  
- **Resets:** session flip, halt/resume, stale feed, long gap ? reset state; cool-down N events.  
- **Planner:** budgeted; disabled automatically on SLA breach.  
- **Kill-switch:** max_daily_loss or drawdown ? cancel all, flatten, publish `risk_breach`.  
- **Post-mortem:** every action carries reason codes & inputs for audit.

---

## Issue template (copy into `.github/ISSUE_TEMPLATE/task.yml`)

```yaml
name: Task
body:
- type: input
  id: id
  attributes: { label: Task ID (e.g., C.2.1) }
- type: textarea
  id: desc
  attributes: { label: Description }
- type: checkboxes
  id: ac
  attributes:
    label: Acceptance criteria
    options:
      - label: Unit tests added
      - label: E2E test passes
      - label: Metrics validated
      - label: Docs updated
```

## Automation updates — 2025-10-20T07:33:29Z

### Last 4 commits
- a1fc6a32 feat(artifacts): add 21 files (2025-10-20)
- a504f467 refactor(trading): tune 3 files (2025-10-20)
- 8e2ac3d6 refactor(data_foundation): tune 5 files (2025-10-20)
- 94ba9da6 feat(thinking): add 3 files (2025-10-20)
