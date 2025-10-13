# AlphaTrade â€” ROADMAP.md (vNext, Q4 2025)

> **North Star**: Build **AlphaTrade** â€” a governed, continuouslyâ€‘learning trading organism that runs a **Perception â†’ Adaptation â†’ Reflection** loop with deterministic rails and risk-as-law.

**How to use this roadmap**
- Treat this as a living checklist. Check off items youâ€™ve completed (`[x]`) and keep open items as `[ ]`.
- Each milestone has **deliverables** and **acceptance criteria** (DoD). Promotion to the next milestone requires all DoD to pass.
- Labels: `[core]` Perception/Belief, `[adapt]` Policy/Router/Evolution, `[reflect]` RIM/TRM/Diary, `[risk]` Risk/Governance, `[obs]` Observability/Telemetry, `[sim]` Replay/Sim/Paper, `[ops]` Deploy/Runtime, `[docs]` Documentation.

---

## M0 â€” Baseline & Safety Gates (0â€“3 days)

**Deliverables**
- [ ] `[risk]` Preâ€‘trade invariants verified (exposure, price bands, inventory, drawdown); killâ€‘switch wired.
- [ ] `[adapt]` Feature flags for **fastâ€‘weights**, **linear attention**, and **exploration** (on/off per env).
- [ ] `[obs]` Minimal **heartbeat** & latency counters (ingestâ†’signalâ†’orderâ†’ack p50/p99/p99.9).
- [ ] `[docs]` Update **CONTRIBUTING** with run profiles (sim/paper/liveâ€‘shadow) and featureâ€‘flag table.

**Acceptance (DoD)**
- [ ] `[risk]` Synthetic invariant breach â†’ order rejected; killâ€‘switch test passes.
- [ ] `[obs]` Heartbeat visible during a 30â€‘minute run; latency counters populated.
- [ ] `[ops]` Deterministic run reproducible via one command (`make run-sim` or equivalent).

---

## M1 â€” Understanding Loop v1 (Weeks 1â€“2)

**Deliverables**
- [ ] `[core]` **BeliefEmitter** with **ReLU + topâ€‘k sparsity**; persisted `belief_snapshots` (DuckDB/Parquet).
- [ ] `[core]` **RegimeFSM v1** (ruleâ€‘based thresholds + confidence + transitions logged).
- [ ] `[adapt]` **LinearAttentionRouter** (flagâ€‘guarded) for policy arbitration.
- [ ] `[adapt]` **Fastâ€‘weights (Ïƒ) kernel** (Hebbian decay + potentiation); **lowâ€‘rank** implementation; clamps/decay.
- [ ] `[reflect]` **Decision Diary** table + writer (belief, policy_hash, exec_topo, risk_template, features, decision, ex_post).
- [ ] `[reflect]` 3â€“5 **synapse probes** (e.g., opening auction, sweep risk, imbalance surge).
- [ ] `[reflect]` **Drift Sentry** (Pageâ€“Hinkley/CUSUM) + actions (freeze exploration, halve size) + â€œtheory packetâ€.
- [ ] `[obs]` Graph diagnostics nightly job: degree hist, modularity, coreâ€“periphery (thresholds set).

**Acceptance (DoD)**
- [ ] `[sim]` **Determinism**: Replay same tape + seeds â‡’ identical diary & PnL.
- [ ] `[adapt]` **Regimeâ€‘aware routing**: regime flip â‡’ topology switch within N ms; proven in diary.
- [ ] `[reflect]` **Drift throttle**: injected alpha decay â‡’ sentry fires within 1 decision step; theory packet written.
- [ ] `[obs]` Attribution coverage â‰¥ **90%** of orders have belief + probes; no Ïƒ explosions (bounded norms).
- [ ] `[risk]` **0** invariant violations in a 4â€‘hour sim run.

---

## M2 â€” Evolution Engine v1 (Weeks 3â€“6)

**Genotype/Phenotype & Operators**
- [ ] `[adapt]` **StrategyGenotype/Phenotype** contracts (fields: features, exec topology, risk template, tunables).
- [ ] `[adapt]` Operators: `op_add_feature`, `op_drop_feature`, `op_swap_execution_topology`, `op_tighten_risk`.
- [ ] `[adapt]` Operator constraints (allowed domain, regimeâ€‘aware rules).

**Search & Selection**
- [ ] `[adapt]` **Tournament selection** over **regime grid** (multiâ€‘regime fitness table).
- [ ] `[adapt]` **Novelty archive** (genotype signature + probe vector; novelty score).
- [ ] `[sim]` **Compute scheduler** for candidate replays (budgeted batches, fairâ€‘share across instruments).

**Budgeted, Safe Exploration**
- [ ] `[adapt]` **Global exploration budget** (â‰¤ X% flow, mutate every K decisions) enforced in router.
- [ ] `[adapt]` **Counterfactual guardrails** (passive vs aggro delta bounds) for live candidates.
- [ ] `[reflect]` **Autoâ€‘freeze** exploration on drift or risk warnings (hooks wired).

**Provenance & Governance**
- [ ] `[reflect]` **Policy Ledger**: promotion checklist (OOS regimeâ€‘grid, leakage checks, risk audit) enforced.
- [ ] `[reflect]` `rebuild_strategy(policy_hash)` produces byteâ€‘identical runtime config.
- [ ] `[docs]` Promotion gate documented (thresholds, required artifacts).

**Acceptance (DoD)**
- [ ] `[sim]` Spawn â†’ score â†’ **promote** a **new topology** (not just parameter tweak) via ledger gates.
- [ ] `[risk]` **0** invariant violations during exploration; freeze triggers on violations/drift immediately.
- [ ] `[obs]` Evolution KPIs live: timeâ€‘toâ€‘candidate, promotion rate, budget usage, rollback latency.
- [ ] `[adapt]` p50/p99 decision latency **not worse** than M1 baseline.

---

## M3 â€” Governed Reflection Feedback (Weeks 5â€“8)

**Deliverables**
- [ ] `[reflect]` **RIM/TRM proposal schema** (confidence, rationale, affected regimes, evidence pointers).
- [ ] `[reflect]` Governance rule: **autoâ€‘apply** proposals IF (a) OOS uplift â‰¥ threshold, (b) 0 risk hits in replay, (c) budget available.
- [ ] `[reflect]` Shadow job: nightly RIM run â‡’ proposals â‡’ governance gate â‡’ staged application (flagâ€‘guarded).
- [ ] `[reflect]` Ledger entries for accepted/rejected proposals + human signâ€‘offs.

**Acceptance (DoD)**
- [ ] `[sim]` At least **one** RIMâ€‘driven change applied via governance rule in sim/paper.
- [ ] `[reflect]` Every proposal traceable (input diary slice, code hash, config hash).
- [ ] `[risk]` No autoâ€‘applied proposal can bypass invariants or budget constraints.

---

## M4 â€” Paper 24/7 & Observability (Weeks 6â€“10)

**Deliverables**
- [ ] `[ops]` Containerized runtime (Docker) + deployment profile (dev/paper); health checks.
- [ ] `[core]` Live market data ingest configured (API keys, symbols, session calendars).
- [ ] `[sim]` **Paper broker** connector smokeâ€‘tested; failover & reconnect logic.
- [ ] `[obs]` Monitoring: dashboards (latency, throughput, P&L swings, memory); alerts on tail spikes and drift.
- [ ] `[ops]` Replay harness scheduled nightly; artifacts persisted (diary, ledger, drift reports).

**Acceptance (DoD)**
- [ ] `[sim]` **24/7 paper run** for â‰¥ 7 days with **zero** invariant violations, stable p99 latency, and no memory leaks.
- [ ] `[obs]` Alerts fired & acknowledged in drill; dashboards show stable metrics.
- [ ] `[docs]` Incident playbook validated (killâ€‘switch, replay, rollback).

---

## M5 â€” Tinyâ€‘Capital Live Pilot (Weeks 10â€“12, gated)

**Deliverables**
- [ ] `[ops]` Live broker integration (sandbox/prod) behind same interfaces; credential rotation & secrets mgmt.
- [ ] `[risk]` â€œLimitedâ€‘liveâ€ governance gate (explicit ledger entry required to enable any real trades).
- [ ] `[ops]` Endâ€‘toâ€‘end audit log export; compliance artifact pack.

**Acceptance (DoD)**
- [ ] `[ops]` Liveâ€‘pilot drill: turn on tiny capital; trigger killâ€‘switch; rollback; reconcile â€” all green.
- [ ] `[risk]` **0** invariant violations; exploration locked to **0%** in live (candidates only in paper).

---

## Continuous Quality Bars (always on)

- [ ] `[risk]` Weekly invariants audit & redâ€‘team scenarios (extreme volatility, symbol halts, bad prints).
- [ ] `[reflect]` Diary coverage â‰¥ **95%**; missingâ€‘data alerts.
- [ ] `[obs]` Graph health in band: modularity, heavyâ€‘tail degree (alerts on collapse/overâ€‘smoothing).
- [ ] `[docs]` Honest README & system diagram reflect current reality (mock vs real clearly labeled).

---

## Backlog / Niceâ€‘toâ€‘Haves

- [ ] `[adapt]` (Âµ+Î») evolution mode in addition to tournament selection.
- [ ] `[adapt]` `op_mix_strategies` (ensembles) with stability penalties (switching frictions).
- [ ] `[reflect]` Counterfactual explainers per trade (why not alternative topology).
- [ ] `[core]` HMMâ€‘based RegimeFSM v2; learned transition priors.
- [ ] `[obs]` Prometheus/Grafana (or cloud) monitoring; SLO alerting as code.
- [ ] `[ops]` K8s deployment (dev/paper); sealed secrets; autoscaling for replay jobs.
- [ ] `[docs]` Whitepaper v1 (architecture + governance + empirical results).

---

## Success Metrics (Northâ€‘Star KPIs)

- [ ] **Timeâ€‘toâ€‘candidate** â‰¤ 24h (idea â†’ scored in replay).
- [ ] **Promotion integrity**: 100% promoted strategies have ledger artifacts & pass regimeâ€‘grid gates.
- [ ] **Guardrail integrity**: risk violations in paper/live = **0**; nearâ€‘misses logged & actioned.
- [ ] **Attribution coverage** â‰¥ 90% (orders with belief + probes + brief explanation).
- [ ] **Operator leverage**: experiments/week/person â†‘ without quality loss.

---

## Commands & Artifacts (to standardize)

- [ ] `make run-sim` â€” deterministic sim/replay run (acceptance tests).
- [ ] `make run-paper` â€” paper 24/7 profile with dashboards.
- [ ] `make rebuild-policy HASH=...` â€” reproduce phenotype from ledger.
- [ ] `make rim-shadow` â€” nightly RIM/TRM proposals + governance report.
- [ ] `artifacts/` â€” diaries, drift reports, ledger exports, evolution KPIs (dated folders).

---

### Notes
- If you have already implemented any item above, **check it now** to keep the roadmap honest.
- Keep feature flags conservative by default (`fast-weights=off`, `exploration=off`, `auto-governed-feedback=off`) and enable progressively per environment.
