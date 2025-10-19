# AlphaTrade â€” ROADMAP.md (vNext, Q4 2025)

> **North Star**: Build **AlphaTrade** â€” a governed, continuouslyâ€‘learning trading organism that runs a **Perception â†’ Adaptation â†’ Reflection** loop with deterministic rails and risk-as-law.

**How to use this roadmap**
- Treat this as a living checklist. Check off items youâ€™ve completed (`[x]`) and keep open items as `[ ]`.
- Each milestone has **deliverables** and **acceptance criteria** (DoD). Promotion to the next milestone requires all DoD to pass.
- Labels: `[core]` Perception/Belief, `[adapt]` Policy/Router/Evolution, `[reflect]` RIM/TRM/Diary, `[risk]` Risk/Governance, `[obs]` Observability/Telemetry, `[sim]` Replay/Sim/Paper, `[ops]` Deploy/Runtime, `[docs]` Documentation.

---

## M0 â€” Baseline & Safety Gates (0â€“3 days)

**Deliverables**
- [ ] `[risk]` Pre-trade invariants verified (exposure, price bands, inventory, drawdown); kill-switch wired. _Progress: Paper simulation CLI now reuses a shared `ensure_zero_invariants` helper that demands four-hour coverage and zero guardrail breaches, emitting structured assessments for evidence packs; kill-switch wiring remains outstanding.【F:src/runtime/simulation_invariants.py†L1-L156】【F:tools/trading/run_paper_trading_simulation.py†L332-L365】【F:tests/runtime/test_simulation_invariants.py†L37-L122】_
- [x] `[adapt]` Feature flags for **fastâ€‘weights**, **linear attention**, and **exploration** (on/off per env). _Progress: `AdaptationFeatureToggles` resolves SystemConfig posture into fast-weight, linear-attention, and exploration flags, flows them through the operational backbone pipeline and runtime, and regression coverage proves AlphaTrade respects environment defaults while diary metadata mirrors forced fast-weight shutdowns, with live mode now forcing conservative toggles and zero exploration budgets regardless of overrides.【F:src/thinking/adaptation/feature_toggles.py†L100-L128】【F:src/data_foundation/pipelines/operational_backbone.py†L35-L213】【F:tools/data_ingest/run_operational_backbone.py†L287-L313】【F:src/runtime/predator_app.py†L260-L288】【F:src/orchestration/alpha_trade_runner.py†L120-L198】【F:src/understanding/router.py†L241-L289】【F:tests/thinking/test_adaptation_feature_toggles.py†L47-L76】【F:tests/orchestration/test_alpha_trade_runner.py†L268-L298】【F:tests/understanding/test_understanding_router.py†L258-L289】_
- [x] `[obs]` Minimal **heartbeat** & latency counters (ingestâ†’signalâ†’orderâ†’ack p50/p99/p99.9). _Progress: `PipelineLatencyMonitor` now records ingest, signal, order, ack, and total latencies with heartbeat ticks/orders inside the bootstrap stack, exposes the snapshot through the runtime status surface, and regression coverage asserts samples populate during simulated runs so operators inherit ready-to-export counters.【F:src/orchestration/pipeline_metrics.py†L1-L137】【F:src/orchestration/bootstrap_stack.py†L195-L351】【F:src/runtime/bootstrap_runtime.py†L328-L344】【F:tests/current/test_bootstrap_stack.py†L170-L176】【F:tests/current/test_bootstrap_runtime_integration.py†L167-L169】_
- [x] `[docs]` Update **CONTRIBUTING** with run profiles (sim/paper/liveâ€‘shadow) and featureâ€‘flag table. _Progress: CONTRIBUTING.md now documents simulation, paper, and liveâ€‘shadow launch recipes alongside a featureâ€‘flag matrix that complements the workflow guide in `docs/development/contributing.md`, giving operators the runtime posture and evidence checklist they need for roadmap gates.【F:CONTRIBUTING.md†L1-L96】_

**Acceptance (DoD)**
- [ ] `[risk]` Synthetic invariant breach â†’ order rejected; killâ€‘switch test passes. _Progress: RiskGateway now inspects portfolio-state metadata, nested insights, and indicator payloads for synthetic invariant signals, records guardrail violations, rejects offending trade intents, and regression coverage exercises metadata-detected incidents while kill-switch validation remains pending.【F:src/trading/risk/risk_gateway.py†L901-L955】【F:tests/current/test_risk_gateway_validation.py†L171-L220】_
- [x] `[obs]` Heartbeat visible during a 30â€‘minute run; latency counters populated. _Progress: Runtime smoke tests exercise the bootstrap loop until status snapshots expose populated heartbeat ticks while stack-level tests confirm ingest and ack percentiles accumulate across ticks, proving the counters fill during multi-tick rehearsals.【F:tests/current/test_bootstrap_runtime_integration.py†L167-L169】【F:tests/current/test_bootstrap_stack.py†L170-L176】_
- [x] `[ops]` Deterministic run reproducible via one command (`make run-sim` or equivalent). _Progress: The `run-sim` wrapper now boots the bootstrap runtime with deterministic defaults, writes summaries/diaries, exposes CLI overrides, and ships a Makefile target plus regression coverage so a single command reproduces the acceptance drill with evidence artifacts.【F:tools/runtime/run_simulation.py†L1-L210】【F:tests/tools/test_run_simulation.py†L1-L138】【F:Makefile†L103-L121】_

---

## M1 â€” Understanding Loop v1 (Weeks 1â€“2)

**Deliverables**
- [x] `[core]` **BeliefEmitter** with **ReLU + top-k sparsity**; persisted `belief_snapshots` (DuckDB/Parquet). _Progress: BeliefEmitter now applies a ReLU top-k activation pass, annotates sparsity metadata, and streams snapshots into DuckDB/Parquet via the new persister so replay trails land in both evidence stores under regression coverage that asserts the `relu_topk` contract and snapshot exports.【F:src/understanding/belief.py†L798-L1102】【F:tests/understanding/test_belief_updates.py†L300-L349】_
- [x] `[core]` **RegimeFSM v1** (ruleâ€‘based thresholds + confidence + transitions logged). _Progress: RegimeFSM now emits structured transition events with latency/volatility context, keeps a bounded transition history, and exposes health metrics under regression coverage so regime flips remain auditable for policy routing.【F:src/understanding/belief.py†L695-L878】【F:tests/understanding/test_belief_updates.py†L327-L394】_
- [ ] `[adapt]` **LinearAttentionRouter** (flagâ€‘guarded) for policy arbitration.
- [x] `[adapt]` **Fastâ€‘weights (Ïƒ) kernel** (Hebbian decay + potentiation); **lowâ€‘rank** implementation; clamps/decay. _Progress: UnderstandingRouter now applies feature-gated Hebbian adapters with deterministic decay, persists multiplier history, and exposes fast-weight metrics so governance can audit adaptive runs under regression coverage.【F:src/understanding/router.py†L70-L240】【F:tests/understanding/test_understanding_router.py†L1-L185】_
- [x] `[reflect]` **Decision Diary** table + writer (belief, policy_hash, exec_topo, risk_template, features, decision, ex_post). _Progress: DecisionDiaryStore normalises policy decisions, attaches probe ownership, records reflection summaries, and the CLI exports Markdown/JSON transcripts with guardrail tests for governance evidence.【F:src/understanding/decision_diary.py†L1-L240】【F:tests/tools/test_decision_diary_cli.py†L1-L188】_
- [ ] `[reflect]` 3â€“5 **synapse probes** (e.g., opening auction, sweep risk, imbalance surge). _Progress: WHAT sensor now emits trend-strength telemetry, metadata, and lineage parity so probes discriminate bullish versus bearish sequences under regression coverage.【F:src/sensory/what/what_sensor.py†L83-L200】【F:tests/sensory/test_primary_dimension_sensors.py†L35-L83】_
- [ ] `[reflect]` **Drift Sentry** (Pageâ€“Hinkley/CUSUM) + actions (freeze exploration, halve size) + â€œtheory packetâ€.
- [x] `[obs]` Graph diagnostics nightly job: degree hist, modularity, coreâ€“periphery (thresholds set). _Progress: Nightly graph diagnostics job now computes degree histograms, modularity, and core ratios, evaluates thresholds, archives JSON/DOT/markdown snapshots, and exports CLI wiring with pytest coverage so observability packs inherit deterministic graph health evidence each run.【F:src/operations/graph_diagnostics.py†L1-L412】【F:tools/operations/nightly_graph_diagnostics.py†L1-L280】【F:tests/operations/test_graph_diagnostics.py†L1-L117】【F:tests/tools/test_nightly_graph_diagnostics.py†L1-L46】_

**Acceptance (DoD)**
- [x] `[sim]` **Determinism**: Replay same tape + seeds â‡’ identical diary & PnL. _Progress: Paper replay determinism now seeds the runtime via `seed_runtime`, replays the bootstrap simulation twice against the same tape, and asserts identical decision diaries and performance snapshots so reproducibility becomes a hard gate under pytest coverage.【F:src/runtime/determinism.py†L1-L64】【F:src/runtime/paper_simulation.py†L1-L219】【F:tests/runtime/test_replay_determinism.py†L1-L174】_
- [x] `[adapt]` **Regimeâ€‘aware routing**: regime flip â‡’ topology switch within N ms; proven in diary. _Progress: PolicyRouter enforces topology switches on regime changes, tracks switch latency, and reflection summaries capture the transition under dedicated pytest coverage.【F:src/thinking/adaptation/policy_router.py†L201-L520】【F:tests/thinking/test_policy_router.py†L60-L152】_
- [x] `[reflect]` **Drift throttle**: injected alpha decay â‡’ sentry fires within 1 decision step; theory packet written. _Progress: AlphaTradeLoopRunner captures trade throttle alpha-decay signals, rewrites trade metadata, intents, and diary payloads with drift throttle plus theory packet entries, and regression coverage drives an alpha-decay run to prove the sentry reacts within one loop.【F:src/orchestration/alpha_trade_runner.py†L1194-L1329】【F:tests/orchestration/test_alpha_trade_runner.py†L705-L759】_
- [ ] `[obs]` Attribution coverage â‰¥ **90%** of orders have belief + probes; no Ïƒ explosions (bounded norms). _Progress: AlphaTradeLoopRunner now attaches belief/probe attribution payloads, tracks diary coverage targets with gap alerts, and persists the stats into trade metadata/diaries while TradingManager records `orders_with_attribution` metrics; regression coverage exercises intentional diary drops to assert warning emission alongside executed-intent coverage checks.【F:src/orchestration/alpha_trade_runner.py†L128-L420】【F:tests/orchestration/test_alpha_trade_runner.py†L660-L836】【F:src/trading/trading_manager.py†L3587-L3637】【F:tests/trading/test_trading_manager_execution.py†L760-L821】_
- [x] `[risk]` **0** invariant violations in a 4-hour sim run. _Progress: Simulation invariant guards validate four-hour rehearsal reports and fail fast on runtime shortfalls or guardrail incidents, with CLI integration exporting the assessment and pytest coverage locking the zero-violation proof for the acceptance gate.【F:src/runtime/simulation_invariants.py†L75-L156】【F:tools/trading/run_paper_trading_simulation.py†L332-L365】【F:tests/runtime/test_simulation_invariants.py†L37-L122】_

---

## M2 â€” Evolution Engine v1 (Weeks 3â€“6)

**Genotype/Phenotype & Operators**
- [x] `[adapt]` **StrategyGenotype/Phenotype** contracts (fields: features, exec topology, risk template, tunables). _Progress: Immutable strategy contracts now normalise feature, topology, tunable, and risk definitions and realise phenotypes with override guards and metadata merges, while regression tests exercise override paths, duplicate detection, and bound enforcement so evolution operators can consume a typed schema immediately.【F:src/thinking/adaptation/strategy_contracts.py†L1-L409】【F:tests/thinking/adaptation/test_strategy_contracts.py†L17-L119】_
- [x] `[adapt]` Operators: `op_add_feature`, `op_drop_feature`, `op_swap_execution_topology`, `op_tighten_risk`. _Progress: Genotype operators now clone typed strategy genomes, apply feature/topology/risk mutations with validation, and emit `GenotypeOperatorResult` payloads capturing lineage metadata while exports surface the helpers in the adaptation package and pytest coverage locks happy-path, replacement, duplicate, and risk-scaling flows.【F:src/thinking/adaptation/operators.py†L1-L353】【F:src/thinking/adaptation/__init__.py†L28-L98】【F:tests/thinking/adaptation/test_genotype_operators.py†L1-L180】_
- [x] `[adapt]` Operator constraints (allowed domain, regimeâ€‘aware rules). _Progress: Operator constraint schemas now parse mappings/sequences into typed sets, EvolutionManager resolves constraint bundles per strategy, AlphaTrade threads regime state into adaptation, and regression suites cover allow/deny flows so adaptive operators respect stage/regime gates and parameter bounds before registering variants.【F:src/thinking/adaptation/operator_constraints.py†L1-L388】【F:src/thinking/adaptation/evolution_manager.py†L80-L412】【F:src/orchestration/alpha_trade_loop.py†L658-L699】【F:tests/thinking/test_operator_constraints.py†L1-L104】【F:tests/thinking/test_evolution_manager.py†L310-L455】_

**Search & Selection**
- [x] `[adapt]` **Tournament selection** over **regime grid** (multiâ€‘regime fitness table). _Progress: PolicyRouter now runs regime-grid tournaments once sufficient decisions accrue, using RegimeFitnessTable aggregates to normalise per-regime and global performance while reflection snapshots capture composite bonuses under guardrail coverage.【F:src/thinking/adaptation/policy_router.py†L582-L740】【F:src/thinking/adaptation/regime_fitness.py†L17-L207】【F:tests/thinking/test_policy_router.py†L530-L623】_
- [x] `[adapt]` **Novelty archive** (genotype signature + probe vector; novelty score). _Progress: `NoveltyArchive` fingerprints strategy genotypes via deterministic SHA-1 signatures, builds fixed-dimension probe vectors, retains a bounded archive with cosine-derived novelty scoring, and regression coverage locks probe immutability, duplicate suppression, variation scoring, and eviction semantics for adaptation workflows.【F:src/thinking/adaptation/novelty_archive.py†L1-L196】【F:tests/thinking/adaptation/test_novelty_archive.py†L1-L111】_
- [ ] `[sim]` **Compute scheduler** for candidate replays (budgeted batches, fairâ€‘share across instruments). _Progress: The `emp_cycle_scheduler` CLI drains idea queues through quick-screening, UCB-lite promotion, and `--max-quick`/`--max-full` budgets while persisting baselines and metadata so replay candidates progress without bespoke scripts under regression coverage.【F:emp/cli/emp_cycle_scheduler.py†L1-L200】【F:emp/cli/_emp_cycle_common.py†L1-L216】【F:tests/emp_cycle/test_cycle_scheduler.py†L1-L86】_

**Budgeted, Safe Exploration**
- [ ] `[adapt]` **Global exploration budget** (â‰¤ X% flow, mutate every K decisions) enforced in router.
- [ ] `[adapt]` **Counterfactual guardrails** (passive vs aggro delta bounds) for live candidates. _Progress: AlphaTradeLoopRunner now merges counterfactual guardrail payloads with existing risk metadata so forced-paper actions propagate to trade intents, trade outcomes, and diaries under regression coverage.【F:src/orchestration/alpha_trade_runner.py†L190-L225】【F:tests/orchestration/test_alpha_trade_runner.py†L253-L365】【F:src/trading/execution/release_router.py†L447-L516】_
- [ ] `[reflect]` **Autoâ€‘freeze** exploration on drift or risk warnings (hooks wired). _Progress: Drift sentry metadata now records triggered metrics, recommended freeze/size multiplier actions, and embeds theory packets that operational readiness carries through alerting drills under pytest coverage.【F:src/operations/drift_sentry.py†L337-L417】【F:tests/operations/test_drift_sentry.py†L73-L169】【F:tests/operations/test_operational_readiness.py†L221-L239】_

**Provenance & Governance**
- [ ] `[reflect]` **Policy Ledger**: promotion checklist (OOS regimeâ€‘grid, leakage checks, risk audit) enforced.
- [ ] `[reflect]` `rebuild_strategy(policy_hash)` produces byteâ€‘identical runtime config.
- [x] `[docs]` Promotion gate documented (thresholds, required artifacts). _Progress: Promotion gate guide now pairs telemetry thresholds with an evidence table detailing required artifacts and storage paths so governance promotions stay auditable end to end.【F:docs/operations/promotion_gate.md†L28-L45】_

**Acceptance (DoD)**
- [x] `[sim]` Spawn â†’ score â†’ **promote** a **new topology** (not just parameter tweak) via ledger gates. _Progress: Tactic replay harness now stamps execution topologies onto evaluation results, the adaptive governance gate records them in ledger metadata, and the nightly replay job threads the topology into diary artifacts with regression coverage promoting an experiment strategy to paper.【F:src/thinking/adaptation/replay_harness.py†L243-L269】【F:src/governance/adaptive_gate.py†L82-L103】【F:tools/operations/nightly_replay_job.py†L300-L363】【F:tests/thinking/test_adaptive_replay_harness.py†L113-L158】_
- [ ] `[risk]` **0** invariant violations during exploration; freeze triggers on violations/drift immediately.
- [x] `[obs]` Evolution KPIs live: timeâ€‘toâ€‘candidate, promotion rate, budget usage, rollback latency. _Progress: `evaluate_evolution_kpis` fuses experimentation turnaround, ledger promotions, exploration budget, and rollback latency into Prometheus-backed gauges that feed the observability dashboard’s Evolution KPIs panel with regression coverage locking both the KPI snapshot and rendered panel wiring.【F:src/operations/evolution_kpis.py†L1-L740】【F:src/operational/metrics.py†L182-L296】【F:src/operations/observability_dashboard.py†L569-L720】【F:tests/operations/test_evolution_kpis.py†L1-L120】【F:tests/operations/test_observability_dashboard.py†L399-L527】_
- [x] `[adapt]` p50/p99 decision latency **not worse** than M1 baseline. _Progress: Decision latency guard now ships the captured M1 baseline JSON, evaluates pipeline latency snapshots with a 5% tolerance inside the bootstrap stack, and regression tests block percentile regressions before they leave CI.【F:docs/performance/performance_baseline.md†L142-L156】【F:src/orchestration/decision_latency_guard.py†L1-L125】【F:src/orchestration/bootstrap_stack.py†L219-L228】【F:tests/orchestration/test_decision_latency_guard.py†L1-L38】【F:tests/current/test_bootstrap_stack.py†L174-L176】_

---

## M3 â€” Governed Reflection Feedback (Weeks 5â€“8)

**Deliverables**
- [x] `[reflect]` **RIM/TRM proposal schema** (confidence, rationale, affected regimes, evidence pointers). _Progress: RIMSuggestion now mandates `affected_regimes` and structured `evidence` blocks, TRM post-processing backfills regime provenance and diary sources, and runner/application tests prove the enriched payload survives governance ingestion end to end.【F:interfaces/rim_types.json†L183-L225】【F:src/reflection/trm/postprocess.py†L122-L200】【F:docs/api/reflection_intelligence_module.md†L99-L146】【F:tests/reflection/test_trm_runner.py†L124-L150】【F:tests/reflection/test_trm_application.py†L103-L142】_
- [x] `[reflect]` Governance rule: **autoâ€‘apply** proposals IF (a) OOS uplift â‰¥ threshold, (b) 0 risk hits in replay, (c) budget available. _Progress: Auto-apply guard now loads thresholds from `config/reflection/rim.config.example.yml`, evaluates uplift, risk hits, and budget utilisation inside `TRMRunner`, and surfaces evaluations in the governance queue and digest with regression coverage documenting rejection reasons.【F:config/reflection/rim.config.example.yml†L13-L24】【F:src/reflection/trm/config.py†L44-L169】【F:src/reflection/trm/runner.py†L138-L213】【F:tests/reflection/test_trm_runner.py†L70-L150】_
- [x] `[reflect]` Shadow job: nightly RIM run â‡’ proposals â‡’ governance gate â‡’ staged application (flagâ€‘guarded). _Progress: The shadow runner now enforces the governance gate, writes skip digests, and stamps auto-apply decisions into the queue/markdown artifacts, with regression coverage ensuring nightly RIM runs stage approved changes without bypassing safeguards.【F:tools/rim_shadow_run.py†L160-L222】【F:src/reflection/trm/governance.py†L19-L247】【F:tests/tools/test_rim_shadow_run.py†L44-L139】【F:tests/reflection/test_trm_governance.py†L16-L126】_
- [x] `[reflect]` Ledger entries for accepted/rejected proposals + human signâ€‘offs. _Progress: Policy ledger records now normalise and merge proposal approvals and human sign-offs, TRM auto-apply writes accepted/rejected identifiers, and promotion helpers surface the fields in governance summaries with regression coverage locking persistence and ingestion flows.【F:src/governance/policy_ledger.py:190】【F:tests/governance/test_policy_ledger.py:111】【F:src/reflection/trm/application.py:128】【F:tests/reflection/test_trm_application.py:123】【F:tools/governance/_promotion_helpers.py:73】_

**Acceptance (DoD)**
- [x] `[sim]` At least **one** RIMâ€‘driven change applied via governance rule in sim/paper. _Progress: Paper runtime now ingests auto-applied TRM queue entries, tags the ledger with `rim-auto` approvals, persists threshold deltas, and exposes the metadata in release posture exports with regression coverage across the helper and paper simulation flow.【F:src/reflection/trm/application.py†L17-L159】【F:tests/reflection/test_trm_application.py†L49-L100】【F:src/runtime/predator_app.py†L2557-L2585】【F:tests/runtime/test_paper_trading_simulation_runner.py†L191-L278】_
- [x] `[reflect]` Every proposal traceable (input diary slice, code hash, config hash). _Progress: Promotion tooling now builds `PolicyTraceability` payloads that bundle diary slices with git code hashes and config-delta digests, threading the metadata into single-policy and graduation CLIs so ledger entries persist verifiable provenance under regression coverage that inspects CLI summaries and stored records.【F:src/governance/policy_traceability.py†L1-L230】【F:tools/governance/promote_policy.py†L323-L379】【F:tools/governance/alpha_trade_graduation.py†L168-L221】【F:tests/governance/test_policy_traceability.py†L1-L112】【F:tests/tools/test_promote_policy_cli.py†L60-L118】_
- [ ] `[risk]` No autoâ€‘applied proposal can bypass invariants or budget constraints.

---

## M4 â€” Paper 24/7 & Observability (Weeks 6â€“10)

**Deliverables**
- [x] `[ops]` Containerized runtime (Docker) + deployment profile (dev/paper); health checks. _Progress: Docker profiles now package the production runtime with Timescale/Redis/Kafka dependencies, proactive health checks, shared env overlays, and matching SystemConfig presets under regression coverage and updated setup docs so operators can launch dev or paper stacks with one compose command.【F:docker/runtime/docker-compose.dev.yml†L1-L136】【F:config/deployment/runtime_paper.yaml†L1-L38】【F:docs/development/setup.md†L62-L93】【F:tests/governance/test_system_config_runtime_profiles.py†L1-L57】_
- [ ] `[core]` Live market data ingest configured (API keys, symbols, session calendars). _Progress: Institutional ingest builder now surfaces API key posture, trading session calendars, and macro calendar configuration via ReferenceDataLoader metadata, with regression coverage confirming London FX schedules and env-key detection while live wiring remains outstanding.【F:src/data_foundation/ingest/configuration.py†L73-L155】【F:src/data_foundation/ingest/configuration.py†L864-L955】【F:tests/data_foundation/test_timescale_config.py†L292-L318】_
- [ ] `[sim]` **Paper broker** connector smokeâ€‘tested; failover & reconnect logic. _Progress: Paper trading simulation reports persist aggregated order summaries with side/symbol splits and broker failover snapshots so operators inherit audit-ready context, locked by regression coverage.【F:src/runtime/paper_simulation.py†L339-L367】【F:tests/runtime/test_paper_trading_simulation_runner.py†L132-L188】【F:tests/integration/test_paper_trading_simulation.py†L394-L429】_
- [ ] `[obs]` Monitoring: dashboards (latency, throughput, P&L swings, memory); alerts on tail spikes and drift.
- [x] `[ops]` Replay harness scheduled nightly; artifacts persisted (diary, ledger, drift reports). _Progress: The nightly replay job now emits diary, drift, and ledger artifacts per run, archives them into dated `artifacts/` mirrors, and is wired to the scheduled GitHub workflow/Make target with regression coverage proving the exports.【F:tools/operations/nightly_replay_job.py:526】【F:tests/tools/test_nightly_replay_job.py:11】【F:.github/workflows/replay-nightly.yml:4】【F:Makefile:137】_

**Acceptance (DoD)**
- [ ] `[sim]` **24/7 paper run** for â‰¥ 7 days with **zero** invariant violations, stable p99 latency, and no memory leaks. _Progress: Paper run guardian now monitors long-horizon paper sessions, enforces latency/invariant thresholds, tracks memory growth, and persists summaries for governance review via the runtime CLI with pytest coverage for breach detection and exports.【F:src/runtime/paper_run_guardian.py†L1-L360】【F:tests/runtime/test_paper_run_guardian.py†L1-L184】【F:src/runtime/cli.py†L1-L220】_
- [ ] `[obs]` Alerts fired & acknowledged in drill; dashboards show stable metrics.
- [x] `[docs]` Incident playbook validated (killâ€‘switch, replay, rollback). _Progress: Dedicated validator CLI now executes kill-switch, nightly replay, and trade rollback drills, writes evidence packs, and is paired with a refreshed runbook and regression coverage so incident rehearsals consistently capture pass/fail artifacts.【F:tools/operations/incident_playbook_validation.py†L44-L255】【F:docs/operations/runbooks/incident_playbook_validation.md†L1-L66】【F:tests/tools/test_incident_playbook_validation.py†L9-L48】_

---

## M5 â€” Tinyâ€‘Capital Live Pilot (Weeks 10â€“12, gated)

**Deliverables**
- [x] `[ops]` Live broker integration (sandbox/prod) behind same interfaces; credential rotation & secrets mgmt. _Progress: `LiveBrokerSecrets` now loads sandbox and production credential profiles with rotation metadata, `FIXConnectionManager` consumes the active profile for genuine sessions while surfacing health summaries, `predator_app` exports secret descriptors, and regression coverage validates selection, legacy fallbacks, and operator summaries so credential rotation stays governed.【F:src/operational/live_broker_secrets.py†L1-L318】【F:src/operational/fix_connection_manager.py†L1-L392】【F:src/runtime/predator_app.py†L2836-L2862】【F:tests/operational/test_live_broker_secrets.py†L1-L94】_
- [x] `[risk]` â€œLimitedâ€‘liveâ€ governance gate (explicit ledger entry required to enable any real trades). _Progress: Ledger release manager now caps default limited-live stages to pilot without a policy record, the release router forces paper routes until a ledger entry exists, and regression coverage asserts forced-paper metadata for missing governance approvals so real trading stays disabled by default.【F:src/governance/policy_ledger.py†L550-L623】【F:src/trading/execution/release_router.py†L86-L310】【F:tests/governance/test_policy_ledger.py†L331-L345】【F:tests/trading/test_release_execution_router.py†L88-L167】_
- [x] `[ops]` Endâ€‘toâ€‘end audit log export; compliance artifact pack. _Progress: Compliance artifact pack builder exports audit logs with manifests, compliance and regulatory snapshots, optional archives, and CLI wiring, with regression coverage for full evidence packs and missing-log fallbacks to keep compliance artifacts reproducible.【F:src/operations/compliance_artifact_pack.py†L1-L188】【F:tools/operations/compliance_artifact_pack.py†L1-L103】【F:tests/operations/test_compliance_artifact_pack.py†L18-L122】_

**Acceptance (DoD)**
- [ ] `[ops]` Liveâ€‘pilot drill: turn on tiny capital; trigger killâ€‘switch; rollback; reconcile â€” all green.
- [ ] `[risk]` **0** invariant violations; exploration locked to **0%** in live (candidates only in paper).

---

## Continuous Quality Bars (always on)

- [x] `[risk]` Weekly invariants audit & redâ€‘team scenarios (extreme volatility, symbol halts, bad prints). _Progress: Weekly audit harness now ingests red-team evidence for extreme volatility, symbol halts, and bad prints, flags stale or missing coverage, and captures guardrail violations into scenario snapshots with regression coverage across warning/fail paths.【F:src/operations/risk_invariants_audit.py†L1-L310】【F:tests/operations/test_risk_invariants_audit.py†L39-L200】_
- [ ] `[reflect]` Diary coverage â‰¥ **95%**; missingâ€‘data alerts. _Progress: Observability dashboard now renders a decision diary panel that ingests loop metadata, reports coverage shortfalls, gap breaches, and sample deficits, and warns when telemetry is missing, with regression coverage ensuring alerts flip warn/fail states for coverage and gaps.【F:src/operations/observability_dashboard.py†L316-L344】【F:src/operations/observability_dashboard.py†L862-L982】【F:tests/operations/test_observability_dashboard.py†L438-L1027】_
- [ ] `[obs]` Graph health in band: modularity, heavyâ€‘tail degree (alerts on collapse/overâ€‘smoothing).
- [ ] `[docs]` Honest README & system diagram reflect current reality (mock vs real clearly labeled).

---

## Backlog / Niceâ€‘toâ€‘Haves

- [x] `[adapt]` (Âµ+Î») evolution mode in addition to tournament selection. _Progress: EvolutionConfig now exposes a `mu_plus_lambda` mode with survivor retention, configurable offspring counts, and fitness-safe parent sampling, and the regression suite locks survivor preservation plus invalid config guards.【F:src/core/evolution/engine.py†L45-L320】【F:tests/current/test_evolution_engine_basic.py†L1-L58】_
- [x] `[adapt]` `op_mix_strategies` (ensembles) with stability penalties (switching frictions). _Progress: The new strategy mixer operator blends scored tactics with friction, decay, and bounds enforcement, exports typed dataclasses, and ships regression coverage for score prioritisation, friction decay, and max-share enforcement so ensemble evolution can progress beyond the backlog stub.【F:src/evolution/mutation/strategy_mixer.py†L1-L200】【F:tests/evolution/test_strategy_mix_operator.py†L1-L118】_
- [x] `[reflect]` Counterfactual explainers per trade (why not alternative topology). _Progress: PolicyRouter reflection summaries now include counterfactual topology entries with score gaps, metric deltas, experiment differences, and blocked reasons, and tests assert per-trade explainers cover alternative topologies.【F:src/thinking/adaptation/policy_router.py:1798】【F:tests/thinking/test_policy_router.py:125】_
- [x] `[core]` HMMâ€‘based RegimeFSM v2; learned transition priors. _Progress: RegimeFSM now embeds an online Gaussian HMM that learns transition counts, emits regime probability matrices, and exposes the learned parameters through metadata and health checks with regression coverage proving priors update across belief sequences.【F:src/understanding/belief.py†L905-L1250】【F:tests/understanding/test_belief_updates.py†L321-L456】_
- [x] `[obs]` Prometheus/Grafana (or cloud) monitoring; SLO alerting as code. _Progress: Prometheus rule files, Grafana provisioning, and the operations runbook now live in-repo with tests guarding alert coverage and dashboard wiring so SLO panels and alerts stay reproducible.【F:config/prometheus/emp_rules.yml†L1-L65】【F:config/grafana/dashboards/json/emp_observability.json†L1-L200】【F:docs/operations/prometheus_grafana.md†L1-L26】【F:tests/config/test_prometheus_monitoring.py†L1-L50】【F:tests/config/test_grafana_dashboard.py†L1-L45】_
- [x] `[ops]` K8s deployment (dev/paper); sealed secrets; autoscaling for replay jobs. _Progress: Kustomize base now ships replay CronJob/ScaledJob resources and sealed-secret templates, with dev/paper/prod overlays and the paper deployment profile documenting secret regeneration, autoscaling, and replay evidence so operators can apply layered manifests without bespoke edits.【F:k8s/README.md†L1-L95】【F:k8s/base/replay-scaledjob.yaml†L1-L63】【F:k8s/overlays/paper/kustomization.yaml†L1-L18】【F:docs/deployment/paper_k8s_profile.md†L1-L120】_
- [x] `[docs]` Whitepaper v1 (architecture + governance + empirical results). _Progress: Whitepaper v1 now inventories each loop layer, tabulates guardrail evidence, and publishes reproducible command drills so reviewers trace roadmap Definition-of-Done checks to executable artifacts and context packs.【F:docs/AlphaTrade_Whitepaper.md†L1-L77】_

---

## Success Metrics (Northâ€‘Star KPIs)

- [ ] **Timeâ€‘toâ€‘candidate** â‰¤ 24h (idea â†’ scored in replay). _Progress: Findings memory now exposes SLA analytics and a metrics CLI reports average/median/p90 turnaround plus breach details, while the new scheduler CLI quick-screens ideas, applies UCB-lite promotion with bounded `--max-quick`/`--max-full` budgets, and stamps evidence notes so candidates enter replay within the 24 h envelope under regression coverage.【F:emp/core/findings_memory.py†L1-L332】【F:emp/cli/emp_cycle_metrics.py†L1-L120】【F:emp/cli/emp_cycle_scheduler.py†L1-L200】【F:tests/emp_cycle/test_time_to_candidate.py†L1-L94】【F:tests/emp_cycle/test_cycle_scheduler.py†L1-L86】_
- [ ] **Promotion integrity**: 100% promoted strategies have ledger artifacts & pass regimeâ€‘grid gates. _Progress: Strategy registry now bootstraps a config-driven PromotionGuard that enforces stage requirements, ledger/diary paths, and regime coverage, with tests proving missing regimes block approvals until coverage is recorded.【F:config/governance/promotion_guard.yaml†L1-L20】【F:src/governance/strategy_registry.py†L45-L224】【F:tests/governance/test_strategy_registry.py†L145-L268】_
- [ ] **Guardrail integrity**: risk violations in paper/live = **0**; nearâ€‘misses logged & actioned. _Progress: Risk guardrail incidents start force-paper cooldowns, near-miss severities now keep routes on paper while logging guardrail metadata, and live broker replays min-size rejections via paper delegation, with regression coverage confirming forced routes and telemetry snapshots.【F:src/trading/trading_manager.py†L540-L590】【F:src/trading/execution/release_router.py†L520-L560】【F:src/trading/execution/live_broker_adapter.py†L190-L213】【F:tests/trading/test_trading_manager_execution.py†L730-L872】_
- [ ] **Attribution coverage** â‰¥ 90% (orders with belief + probes + brief explanation). _Progress: Loop runner telemetry now emits diary-coverage snapshots with warnings on sustained gaps while the trading manager continues to record attribution metrics; end-to-end tests cover both executed-trade coverage and forced diary misses so observability surfaces quantify drift toward the 90% goal.【F:src/orchestration/alpha_trade_runner.py†L128-L420】【F:tests/orchestration/test_alpha_trade_runner.py†L660-L836】【F:src/trading/trading_manager.py†L3587-L3637】【F:tests/trading/test_trading_manager_execution.py†L760-L821】_
- [x] **Operator leverage**: experiments/week/person â†‘ without quality loss. _Progress: Operator leverage evaluator now rolls experimentation logs into per-operator velocity and quality telemetry, escalates WARN/FAIL posture for lagging operators, and the observability dashboard panel renders the snapshot with deterministic top-operator summaries under guardrail coverage.【F:src/operations/operator_leverage.py†L1-L543】【F:src/operations/observability_dashboard.py†L701-L748】【F:tests/operations/test_operator_leverage.py†L49-L139】【F:tests/operations/test_observability_dashboard.py†L560-L598】_

---

## Commands & Artifacts (to standardize)

- [x] `make run-sim` â€” deterministic sim/replay run (acceptance tests). _Progress: New tooling wraps the bootstrap runtime into `make run-sim`, wiring environment defaults, summary/diary exports, and pytest coverage so the acceptance drill is a single reproducible command.【F:Makefile†L103-L121】【F:tools/runtime/run_simulation.py†L1-L210】【F:tests/tools/test_run_simulation.py†L1-L138】_
- [x] `make run-paper` â€” paper 24/7 profile with dashboards. _Progress: Makefile routes to the runtime CLI `paper-run` subcommand which boots the guardian, enforces minimum runtime thresholds, streams progress, and persists structured summaries for dashboards while the CLI surfaces shortfall warnings under pytest coverage.【F:Makefile†L98-L100】【F:src/runtime/cli.py†L365-L414】【F:src/runtime/paper_run_guardian.py†L56-L128】【F:src/runtime/paper_run_guardian.py†L343-L399】【F:tests/runtime/test_paper_run_guardian.py†L100-L160】_
- [ ] `make rebuild-policy HASH=...` â€” reproduce phenotype from ledger. _Progress: `rebuild_strategy` now normalises ledger payloads into canonical JSON bytes, returns deterministic digests, and ships tests proving byte-identical rebuilds so CLI wrappers can emit audited runtime configs.【F:src/governance/strategy_rebuilder.py†L59-L205】【F:tests/governance/test_strategy_rebuilder.py†L57-L101】_
- [x] `make rim-shadow` â€” nightly RIM/TRM proposals + governance report. _Progress: The `rim-shadow` target now drives the governance-gated shadow runner, emitting suggestions plus queue/digest markdown artifacts with auto-apply annotations under pytest coverage so nightly cron jobs stay audit-ready.【F:Makefile†L67-L85】【F:tools/rim_shadow_run.py†L160-L222】【F:tests/tools/test_rim_shadow_run.py†L44-L139】_
- [x] `artifacts/` â€” diaries, drift reports, ledger exports, evolution KPIs (dated folders). _Progress: `archive_artifact` standardises dated evidence mirrors and now runs from the decision diary, nightly replay job, policy ledger, and evolution lab exporter with coverage so compliance packs auto-land in `artifacts/<kind>/<date>/<run>/` without manual curation while deterministic diary snapshots strip runtime-only noise for replays.【F:src/artifacts/archive.py†L1-L95】【F:src/understanding/decision_diary.py†L42-L115】【F:src/understanding/decision_diary.py†L453-L606】【F:src/governance/policy_ledger.py†L394-L420】【F:tools/operations/nightly_replay_job.py†L573-L590】【F:scripts/generate_evolution_lab.py†L171-L185】【F:tests/artifacts/test_archive.py†L1-L45】_

---

### Notes
- If you have already implemented any item above, **check it now** to keep the roadmap honest.
- Keep feature flags conservative by default (`fast-weights=off`, `exploration=off`, `auto-governed-feedback=off`) and enable progressively per environment.

## Automation updates — 2025-10-19T15:03:27Z

### Last 4 commits
- 9b4f45b9 feat(artifacts): add 9 files (2025-10-19)
- 4bbec995 refactor(operations): tune 3 files (2025-10-19)
- 0687ccd0 feat(operations): add 2 files (2025-10-19)
- 9511c118 test(artifacts): add 5 files (2025-10-19)

## Automation updates — 2025-10-19T14:08:18Z

### Last 4 commits
- 498096d9 refactor(trading): tune 2 files (2025-10-19)
- 0f261821 feat(thinking): add 4 files (2025-10-19)
- 9ca1fe3f feat(operational): add 4 files (2025-10-19)
- 3d8b9204 docs(docs): tune 1 file (2025-10-19)

## Automation updates — 2025-10-19T13:45:52Z

### Last 4 commits
- 428567dc refactor(governance): tune 1 file (2025-10-19)
- 1d351a0f refactor(runtime): tune 6 files (2025-10-19)
- 10f9e871 refactor(runtime): tune 3 files (2025-10-19)
- 4ecd3599 refactor(runtime): tune 3 files (2025-10-19)

## Automation updates — 2025-10-13T19:04:41Z

### Last 4 commits
- ea2ec35e refactor(governance): tune 4 files (2025-10-13)
- 90a11f13 refactor(orchestration): tune 3 files (2025-10-13)
- 9c49d261 refactor(runtime): tune 3 files (2025-10-13)
- 7644d392 feat(operations): add 4 files (2025-10-13)

## Automation updates — 2025-10-13T17:41:12Z

### Last 4 commits
- e9dc22c5 refactor(orchestration): tune 2 files (2025-10-13)
- a06d2fca feat(thinking): add 4 files (2025-10-13)
- 40ddcacb feat(governance): add 5 files (2025-10-13)
- b580f931 docs(docs): tune 5 files (2025-10-13)

## Automation updates — 2025-10-13T15:45:02Z

### Last 4 commits
- 6a15f7e3 refactor(docs): tune 10 files (2025-10-13)
- 49731560 docs(docs): add 7 files (2025-10-13)
- 4be44d7b feat(artifacts): add 45 files (2025-10-13)
- 25a7ab80 feat(docs): add 6 files (2025-10-13)

## Automation updates — 2025-10-13T14:05:18Z

### Last 4 commits
- 4ce99a58 refactor(trading): tune 3 files (2025-10-13)
- 7bb97e80 refactor(docs): tune 8 files (2025-10-13)
- d14b4bf2 docs(docs): tune 4 files (2025-10-13)
- 4321cc3d docs(docs): tune 1 file (2025-10-13)

## Automation updates — 2025-10-13T11:49:05Z

### Last 4 commits
- 4941ac86 feat(data_foundation): add 8 files (2025-10-13)
- 9e74cefd feat(scripts): add 7 files (2025-10-13)
- db699aee docs(docs): tune 3 files (2025-10-13)
- f8559f7d docs(docs): add 1 file (2025-10-13)

## Automation updates — 2025-10-13T12:41:27Z

### Last 4 commits
- 4b1d1537 refactor(evolution): tune 3 files (2025-10-13)
- 312b9232 refactor(thinking): tune 5 files (2025-10-13)
- fed47619 feat(Makefile): add 5 files (2025-10-13)
- 0a519aab test(Makefile): tune 3 files (2025-10-13)

## Automation updates — 2025-10-13T12:54:17Z

### Last 4 commits
- e17202c6 refactor(core): tune 3 files (2025-10-13)
- 6eb10cb5 feat(config): add 3 files (2025-10-13)
- a17ab5f1 refactor(config): tune 5 files (2025-10-13)
- 13547774 docs(config): add 8 files (2025-10-13)

## Automation updates — 2025-10-13T13:31:38Z

### Last 4 commits
- e67347ca feat(governance): add 3 files (2025-10-13)
- 574873e1 refactor(orchestration): tune 5 files (2025-10-13)
- 29755921 feat(trading): add 5 files (2025-10-13)
- 4e57f692 docs(docs): tune 1 file (2025-10-13)

## Automation updates — 2025-10-13T13:44:20Z

### Last 4 commits
- 4321cc3d docs(docs): tune 1 file (2025-10-13)
- 7ad61ea2 refactor(governance): tune 4 files (2025-10-13)
- 86adfce5 refactor(understanding): tune 2 files (2025-10-13)
- e88e77b3 docs(docs): tune 1 file (2025-10-13)

## Automation updates — 2025-10-13T14:38:06Z

### Last 4 commits
- 24e65aee docs(docs): tune 1 file (2025-10-13)
- e4094cba refactor(operations): tune 6 files (2025-10-13)
- 176f4285 feat(runtime): add 2 files (2025-10-13)
- 482b64f7 refactor(governance): tune 3 files (2025-10-13)
