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
- [x] `[docs]` Update **CONTRIBUTING** with run profiles (sim/paper/liveâ€‘shadow) and featureâ€‘flag table. _Progress: CONTRIBUTING.md now documents simulation, paper, and liveâ€‘shadow launch recipes alongside a featureâ€‘flag matrix that complements the workflow guide in `docs/development/contributing.md`, giving operators the runtime posture and evidence checklist they need for roadmap gates.【F:CONTRIBUTING.md†L1-L96】_

**Acceptance (DoD)**
- [ ] `[risk]` Synthetic invariant breach â†’ order rejected; killâ€‘switch test passes.
- [ ] `[obs]` Heartbeat visible during a 30â€‘minute run; latency counters populated.
- [x] `[ops]` Deterministic run reproducible via one command (`make run-sim` or equivalent). _Progress: The `run-sim` wrapper now boots the bootstrap runtime with deterministic defaults, writes summaries/diaries, exposes CLI overrides, and ships a Makefile target plus regression coverage so a single command reproduces the acceptance drill with evidence artifacts.【F:tools/runtime/run_simulation.py†L1-L210】【F:tests/tools/test_run_simulation.py†L1-L138】【F:Makefile†L103-L121】_

---

## M1 â€” Understanding Loop v1 (Weeks 1â€“2)

**Deliverables**
- [ ] `[core]` **BeliefEmitter** with **ReLU + topâ€‘k sparsity**; persisted `belief_snapshots` (DuckDB/Parquet).
- [x] `[core]` **RegimeFSM v1** (ruleâ€‘based thresholds + confidence + transitions logged). _Progress: RegimeFSM now emits structured transition events with latency/volatility context, keeps a bounded transition history, and exposes health metrics under regression coverage so regime flips remain auditable for policy routing.【F:src/understanding/belief.py†L695-L878】【F:tests/understanding/test_belief_updates.py†L327-L394】_
- [ ] `[adapt]` **LinearAttentionRouter** (flagâ€‘guarded) for policy arbitration.
- [x] `[adapt]` **Fastâ€‘weights (Ïƒ) kernel** (Hebbian decay + potentiation); **lowâ€‘rank** implementation; clamps/decay. _Progress: UnderstandingRouter now applies feature-gated Hebbian adapters with deterministic decay, persists multiplier history, and exposes fast-weight metrics so governance can audit adaptive runs under regression coverage.【F:src/understanding/router.py†L70-L240】【F:tests/understanding/test_understanding_router.py†L1-L185】_
- [x] `[reflect]` **Decision Diary** table + writer (belief, policy_hash, exec_topo, risk_template, features, decision, ex_post). _Progress: DecisionDiaryStore normalises policy decisions, attaches probe ownership, records reflection summaries, and the CLI exports Markdown/JSON transcripts with guardrail tests for governance evidence.【F:src/understanding/decision_diary.py†L1-L240】【F:tests/tools/test_decision_diary_cli.py†L1-L188】_
- [ ] `[reflect]` 3â€“5 **synapse probes** (e.g., opening auction, sweep risk, imbalance surge). _Progress: WHAT sensor now emits trend-strength telemetry, metadata, and lineage parity so probes discriminate bullish versus bearish sequences under regression coverage.【F:src/sensory/what/what_sensor.py†L83-L200】【F:tests/sensory/test_primary_dimension_sensors.py†L35-L83】_
- [ ] `[reflect]` **Drift Sentry** (Pageâ€“Hinkley/CUSUM) + actions (freeze exploration, halve size) + â€œtheory packetâ€.
- [ ] `[obs]` Graph diagnostics nightly job: degree hist, modularity, coreâ€“periphery (thresholds set).

**Acceptance (DoD)**
- [ ] `[sim]` **Determinism**: Replay same tape + seeds â‡’ identical diary & PnL.
- [x] `[adapt]` **Regimeâ€‘aware routing**: regime flip â‡’ topology switch within N ms; proven in diary. _Progress: PolicyRouter enforces topology switches on regime changes, tracks switch latency, and reflection summaries capture the transition under dedicated pytest coverage.【F:src/thinking/adaptation/policy_router.py†L201-L520】【F:tests/thinking/test_policy_router.py†L60-L152】_
- [ ] `[reflect]` **Drift throttle**: injected alpha decay â‡’ sentry fires within 1 decision step; theory packet written.
- [ ] `[obs]` Attribution coverage â‰¥ **90%** of orders have belief + probes; no Ïƒ explosions (bounded norms). _Progress: AlphaTradeLoopRunner now attaches belief/probe attribution payloads to trade metadata and decision diaries, while TradingManager records `orders_with_attribution` and `attribution_coverage` stats and warns when coverage slips below the 90% target under guardrail tests for executed intents.【F:src/orchestration/alpha_trade_runner.py†L168-L234】【F:src/trading/trading_manager.py†L3587-L3637】【F:tests/trading/test_trading_manager_execution.py†L760-L821】_
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
- [x] `[reflect]` Shadow job: nightly RIM run â‡’ proposals â‡’ governance gate â‡’ staged application (flagâ€‘guarded). _Progress: The shadow runner now enforces the governance gate, writes skip digests, and stamps auto-apply decisions into the queue/markdown artifacts, with regression coverage ensuring nightly RIM runs stage approved changes without bypassing safeguards.【F:tools/rim_shadow_run.py†L160-L222】【F:src/reflection/trm/governance.py†L19-L247】【F:tests/tools/test_rim_shadow_run.py†L44-L139】【F:tests/reflection/test_trm_governance.py†L16-L126】_
- [ ] `[reflect]` Ledger entries for accepted/rejected proposals + human signâ€‘offs.

**Acceptance (DoD)**
- [ ] `[sim]` At least **one** RIMâ€‘driven change applied via governance rule in sim/paper. _Progress: Auto-apply governance now promotes qualifying TRM suggestions directly in the queue/digest artifacts under test coverage, paving the way for paper-run promotion drills.【F:src/reflection/trm/governance.py†L94-L247】【F:tests/reflection/test_trm_governance.py†L16-L126】_
- [ ] `[reflect]` Every proposal traceable (input diary slice, code hash, config hash).
- [ ] `[risk]` No autoâ€‘applied proposal can bypass invariants or budget constraints.

---

## M4 â€” Paper 24/7 & Observability (Weeks 6â€“10)

**Deliverables**
- [ ] `[ops]` Containerized runtime (Docker) + deployment profile (dev/paper); health checks.
- [ ] `[core]` Live market data ingest configured (API keys, symbols, session calendars).
- [ ] `[sim]` **Paper broker** connector smokeâ€‘tested; failover & reconnect logic. _Progress: Paper trading simulation reports persist aggregated order summaries with side/symbol splits and broker failover snapshots so operators inherit audit-ready context, locked by regression coverage.【F:src/runtime/paper_simulation.py†L339-L367】【F:tests/runtime/test_paper_trading_simulation_runner.py†L132-L188】【F:tests/integration/test_paper_trading_simulation.py†L394-L429】_
- [ ] `[obs]` Monitoring: dashboards (latency, throughput, P&L swings, memory); alerts on tail spikes and drift.
- [ ] `[ops]` Replay harness scheduled nightly; artifacts persisted (diary, ledger, drift reports).

**Acceptance (DoD)**
- [ ] `[sim]` **24/7 paper run** for â‰¥ 7 days with **zero** invariant violations, stable p99 latency, and no memory leaks. _Progress: Paper run guardian now monitors long-horizon paper sessions, enforces latency/invariant thresholds, tracks memory growth, and persists summaries for governance review via the runtime CLI with pytest coverage for breach detection and exports.【F:src/runtime/paper_run_guardian.py†L1-L360】【F:tests/runtime/test_paper_run_guardian.py†L1-L184】【F:src/runtime/cli.py†L1-L220】_
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
- [x] `[adapt]` `op_mix_strategies` (ensembles) with stability penalties (switching frictions). _Progress: The new strategy mixer operator blends scored tactics with friction, decay, and bounds enforcement, exports typed dataclasses, and ships regression coverage for score prioritisation, friction decay, and max-share enforcement so ensemble evolution can progress beyond the backlog stub.【F:src/evolution/mutation/strategy_mixer.py†L1-L200】【F:tests/evolution/test_strategy_mix_operator.py†L1-L118】_
- [ ] `[reflect]` Counterfactual explainers per trade (why not alternative topology).
- [ ] `[core]` HMMâ€‘based RegimeFSM v2; learned transition priors.
- [ ] `[obs]` Prometheus/Grafana (or cloud) monitoring; SLO alerting as code.
- [ ] `[ops]` K8s deployment (dev/paper); sealed secrets; autoscaling for replay jobs.
- [ ] `[docs]` Whitepaper v1 (architecture + governance + empirical results).

---

## Success Metrics (Northâ€‘Star KPIs)

- [ ] **Timeâ€‘toâ€‘candidate** â‰¤ 24h (idea â†’ scored in replay). _Progress: Findings memory now exposes SLA analytics and a CLI reports average/median/p90 turnaround and breach details so the experimentation loop can track adherence in real time under regression coverage.【F:emp/core/findings_memory.py†L1-L460】【F:emp/cli/emp_cycle_metrics.py†L1-L120】【F:tests/emp_cycle/test_time_to_candidate.py†L1-L86】_
- [ ] **Promotion integrity**: 100% promoted strategies have ledger artifacts & pass regimeâ€‘grid gates.
- [ ] **Guardrail integrity**: risk violations in paper/live = **0**; nearâ€‘misses logged & actioned.
- [ ] **Attribution coverage** â‰¥ 90% (orders with belief + probes + brief explanation). _Progress: Execution stats now surface attribution coverage metrics and end-to-end tests assert executed trades retain the enriched payload, enabling governance dashboards to track the 90% target quantitatively.【F:src/trading/trading_manager.py†L3587-L3637】【F:tests/trading/test_trading_manager_execution.py†L760-L821】_
- [ ] **Operator leverage**: experiments/week/person â†‘ without quality loss.

---

## Commands & Artifacts (to standardize)

- [x] `make run-sim` â€” deterministic sim/replay run (acceptance tests). _Progress: New tooling wraps the bootstrap runtime into `make run-sim`, wiring environment defaults, summary/diary exports, and pytest coverage so the acceptance drill is a single reproducible command.【F:Makefile†L103-L121】【F:tools/runtime/run_simulation.py†L1-L210】【F:tests/tools/test_run_simulation.py†L1-L138】_
- [x] `make run-paper` â€” paper 24/7 profile with dashboards. _Progress: Makefile routes to the runtime CLI `paper-run` subcommand which boots the guardian, streams progress, and persists structured summaries for dashboards under pytest coverage.【F:Makefile†L90-L98】【F:src/runtime/cli.py†L1-L220】【F:src/runtime/paper_run_guardian.py†L1-L360】【F:tests/runtime/test_paper_run_guardian.py†L1-L184】_
- [ ] `make rebuild-policy HASH=...` â€” reproduce phenotype from ledger.
- [x] `make rim-shadow` â€” nightly RIM/TRM proposals + governance report. _Progress: The `rim-shadow` target now drives the governance-gated shadow runner, emitting suggestions plus queue/digest markdown artifacts with auto-apply annotations under pytest coverage so nightly cron jobs stay audit-ready.【F:Makefile†L67-L85】【F:tools/rim_shadow_run.py†L160-L222】【F:tests/tools/test_rim_shadow_run.py†L44-L139】_
- [ ] `artifacts/` â€” diaries, drift reports, ledger exports, evolution KPIs (dated folders).

---

### Notes
- If you have already implemented any item above, **check it now** to keep the roadmap honest.
- Keep feature flags conservative by default (`fast-weights=off`, `exploration=off`, `auto-governed-feedback=off`) and enable progressively per environment.
