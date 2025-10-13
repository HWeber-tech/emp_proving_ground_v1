# AlphaTrade Whitepaper v1

## Abstract
AlphaTrade operationalises the Emporium Proving Ground (EMP) blueprint for a governed trading intelligence stack. The roadmap pairs a fast-weight decision loop with institution-grade observability, enforcing guardrails before any live order flow is enabled. This v1 whitepaper captures the current architecture, governance posture, and empirical evidence, grounding every claim in executable modules, guardrail tests, and the context packs that define the programme’s intent.【docs/context/alignment_briefs/institutional_data_backbone.md】【docs/context/alignment_briefs/operational_readiness.md】【docs/context/alignment_briefs/quality_observability.md】

## 1. System Overview
AlphaTrade follows the EMP encyclopedia’s layered domains, isolating sensory ingest, adaptation, reflection, execution, and governance so that each layer can evolve without breaking institutional safeguards.【docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md】 Runtime orchestration is delivered through supervised services that honour the roadmap principle of simulator-first rehearsals with auditable artefacts.【docs/architecture/overview.md】

### 1.1 Control Loop
```
┌────────────┐    ┌─────────────┐    ┌──────────────┐    ┌────────────┐    ┌───────────────┐
│ Perception │──► │ Adaptation  │──► │ Reflection   │──► │ Execution  │──► │ Governance    │
└─────┬──────┘    └─────┬───────┘    └──────┬──────┘    └──────┬─────┘    └──────┬────────┘
      │                feedback ▲             │                 │                │
      │                       │               ▼                 │                │
      │             telemetry + diaries  ─── decision ledger ───┘        policy + ops loops
      ▼
Timescale / Cache / Kafka ingest
```

### 1.2 Stage Inventory
| Stage | Role | Implementation anchors | Guardrails |
| --- | --- | --- | --- |
| Perception | Fuse multi-dimensional sensors into lineage-rich beliefs. | `src/sensory/real_sensory_organ.py:124`, `src/sensory/what/what_sensor.py:109`, `src/data_foundation/ingest/institutional_vertical.py:402` | `tests/integration/test_real_data_slice_ingest.py:12`, `tests/data_foundation/test_ingest_scheduler.py:177` |
| Adaptation | Apply feature-gated fast weights and routing heuristics. | `src/understanding/router.py:32`, `src/thinking/adaptation/fast_weights.py:1`, `src/orchestration/alpha_trade_runner.py:169` | `tests/thinking/test_fast_weights.py:13`, `tests/understanding/test_understanding_router.py:86` |
| Reflection | Persist decision diaries, probes, and attribution payloads. | `src/understanding/decision_diary.py:58`, `src/orchestration/alpha_trade_runner.py:197` | `tests/understanding/test_decision_diary.py:34`, `tests/trading/test_trading_manager_execution.py:761` |
| Execution | Enforce throttles, invariant checks, and broker integration. | `src/trading/trading_manager.py:3587`, `src/runtime/paper_run_guardian.py:1`, `src/runtime/cli.py:187` | `tests/trading/test_trading_manager_execution.py:780`, `tests/runtime/test_paper_run_guardian.py:18` |
| Governance | Ledger promotions, drift throttles, observability dashboards. | `src/governance/policy_ledger.py:572`, `src/governance/strategy_registry.py:158`, `src/operations/observability_dashboard.py:822`, `tools/operations/incident_playbook_validation.py:208` | `tests/governance/test_strategy_registry.py:145`, `tests/operations/test_observability_dashboard.py:582`, `tests/tools/test_incident_playbook_validation.py:9` |

## 2. Architecture Deep Dive

### 2.1 Perception
`RealSensoryOrgan` integrates WHAT/WHEN/WHY/HOW/ANOMALY organs, emitting calibrated payloads with confidence, quality metadata, and lineage so downstream analysis inherits auditable inputs.【src/sensory/real_sensory_organ.py:124】 The WHAT sensor now exposes trend-strength probes and telemetry, allowing synapse probes to discriminate bullish versus bearish regimes during regression coverage.【src/sensory/what/what_sensor.py:109】 Timescale ingest is supervised by institutional schedulers that expose manifest snapshots and Kafka bridges, aligning the data backbone story with the alignment brief.【src/data_foundation/ingest/institutional_vertical.py:402】 Integration tests hydrate real slices into the organ, verifying that volatility features and timezone hygiene survive the ingest path.【tests/integration/test_real_data_slice_ingest.py:12】 Scheduler guardrails assert deterministic shutdown and progress reporting so operators know when backfill loops halt.【tests/data_foundation/test_ingest_scheduler.py:177】

### 2.2 Adaptation
The UnderstandingRouter wraps the policy router with feature-gated fast-weight adapters, Hebbian learning decays, and top-k sparsity summaries so operators can reason about tactic boosts at every decision step.【src/understanding/router.py:32】 FastWeightController enforces non-negative multipliers, clamps inhibitory weights unless explicitly allowed, and surfaces sparsity metrics demanded by the roadmap.【src/thinking/adaptation/fast_weights.py:1】 Guardrail tests confirm inhibitory multipliers are suppressed in excitatory mode and that overrides from configuration are parsed faithfully.【tests/thinking/test_fast_weights.py:13】 During runtime, the AlphaTrade runner injects fast-weight, guardrail, and attribution payloads both into trade metadata and diary notes, preserving an end-to-end chain of explanation for each intent.【src/orchestration/alpha_trade_runner.py:169】【src/orchestration/alpha_trade_runner.py:197】

### 2.3 Reflection
DecisionDiary serialises policy decisions, regime states, belief snapshots, and probe activations with timezone-safe timestamps, ensuring reproducibility for governance replay.【src/understanding/decision_diary.py:58】 Probe activations inherit owner/contact metadata from the probe registry so audit surfaces can escalate to accountable humans without spelunking logs.【src/understanding/decision_diary.py:147】 TradingManager records guardrail near-misses, experiment events, and attribution coverage metrics, warning when coverage slips below the 90% policy target.【src/trading/trading_manager.py:3587】【tests/trading/test_trading_manager_execution.py:761】 The reflection pipeline therefore produces diary artefacts that satisfy the context pack’s accountability demands before live routing is considered.【docs/context/alignment_briefs/quality_observability.md】

### 2.4 Execution
The trading layer threads guardrails, throttle snapshots, and attribution bundles through every on_trade_intent call, ensuring router decisions remain explainable after execution.【src/orchestration/alpha_trade_runner.py:226】【src/trading/trading_manager.py:3599】 PaperRunGuardian extends the simulator runtime with 24/7 monitoring of latency percentiles, invariant breaches, and memory growth, converting long-horizon rehearsals into structured summaries for governance sign-off.【src/runtime/paper_run_guardian.py:1】 CLI support allows operators to launch guardian-backed paper runs with configurable thresholds, aligning runtime operations with preparedness drills.【src/runtime/cli.py:187】 Regression coverage asserts latency threshold breaches escalate the guardian status and invariant errors trigger stop conditions, so the roadmap’s 24/7 rehearsal acceptance criteria remain testable.【tests/runtime/test_paper_run_guardian.py:18】

### 2.5 Governance
PolicyLedger resolves stage thresholds, audit gaps, and evidence pointers so promotion decisions always cite ledger evidence and recorded diary slices.【src/governance/policy_ledger.py:572】 StrategyRegistry bootstraps PromotionGuard from configuration, demanding regime coverage and ledger artefacts before any tactic moves beyond paper.【src/governance/strategy_registry.py:158】【config/governance/promotion_guard.yaml:1】 Tests prove the guard blocks approval when bullish regimes lack diary coverage and only allows graduation once coverage is demonstrated.【tests/governance/test_strategy_registry.py:145】 DriftSentry gates evaluate sensory snapshots, apply stage-aware thresholds, and can force paper routes when drift reaches alert levels, integrating risk signals directly into execution decisions.【src/trading/gating/drift_sentry_gate.py:1】 Observability dashboards present policy graduation, drift status, and compliance panels so governance posture is visible in a single surface, fulfilling the operational readiness brief.【src/operations/observability_dashboard.py:822】【tests/operations/test_observability_dashboard.py:582】 Incident playbook validation packages kill-switch, nightly replay, and rollback drills into a CLI that persists evidence packs for auditors.【tools/operations/incident_playbook_validation.py:208】【tests/tools/test_incident_playbook_validation.py:9】

## 3. Governance and Operational Assurance
TaskSupervisor now wraps ingestion, trading, and auxiliary workloads, emitting per-workload manifests with restart policies and hang timeouts so operators inherit deterministic lifecycle control.【src/runtime/runtime_builder.py:1219】 Runtime builder snapshots bucket supervisor entries by workload, giving dashboards and CLI tooling a reliable service registry without scraping raw asyncio state.【src/runtime/runtime_builder.py:1247】 Drift analytics propagate to the governance layer, where PolicyLedger workflows enumerate approvals, threshold overrides, and policy deltas for each strategy so reviewers can audit stage posture quickly.【src/governance/policy_ledger.py:572】 The operational readiness brief called for supervised operations, observability, and documentation hygiene; the guardian, dashboard, and incident CLI satisfy those acceptance bullets with pytest coverage to guard against regressions.【docs/context/alignment_briefs/operational_readiness.md】 Counterfactual guardrails remain simulator-only, and live trading stays gated behind the “limited_live” ledger stage; no code path bypasses invariant enforcement or exploration budgets in paper mode.【docs/roadmap.md】【src/governance/strategy_registry.py:228】

## 4. Empirical Evidence
| Evidence | Result | Source |
| --- | --- | --- |
| Real data slice ingestion hydrates sensory organ, produces finite belief posterior, and writes Timescale rows. | Pass; volatility features and timezone integrity asserted. | `tests/integration/test_real_data_slice_ingest.py:12` |
| Fast-weight controller enforces sparse, non-negative multipliers, respecting excitatory-only mode. | Pass; inhibitory weights suppressed, metrics exposed. | `tests/thinking/test_fast_weights.py:13` |
| TradingManager records guardrail near-misses and maintains attribution coverage ≥90% during simulator runs. | Pass; guardrail counts tracked, outcomes recorded. | `tests/trading/test_trading_manager_execution.py:761` |
| Paper run guardian detects latency breaches, invariant violations, and exports summaries. | Pass; degraded and failed states asserted plus summary persistence. | `tests/runtime/test_paper_run_guardian.py:18` |
| Drift sentry replay triggers alert severity and populates metadata for governance review. | Pass; Page-Hinkley detectors escalate to alert. | `tests/operations/test_sensory_drift.py:186` |
| Promotion guard blocks approvals without ledger evidence or regime coverage. | Pass; missing regimes raise StrategyRegistryError until diary coverage exists. | `tests/governance/test_strategy_registry.py:145` |
| Observability dashboard surfaces policy graduation posture and regression warnings. | Pass; panel statuses drive fail/ok indicators. | `tests/operations/test_observability_dashboard.py:582` |
| Incident playbook CLI runs kill-switch, replay, and rollback drills, persisting JSON evidence. | Pass; CLI orchestration verified under pytest. | `tests/tools/test_incident_playbook_validation.py:9` |

These outcomes align with the roadmap acceptance criteria for architecture readiness, governance guardrails, and paper-mode resilience. Additional drills (e.g. live broker sandboxes) remain future work and are tracked in the roadmap backlog.【docs/roadmap.md】

## 5. Alignment with Context Packs
- **Institutional data backbone.** Timescale ingest supervision and manifest reporting meet the ingest quality targets described in the alignment brief.【docs/context/alignment_briefs/institutional_data_backbone.md】
- **Evolution engine.** Fast-weight routers, exploration throttles, and promotion guards preserve the adaptive experimentation story while containing risk.【docs/context/alignment_briefs/evolution_engine.md】
- **Operational readiness.** Guardian monitoring, observability dashboards, and incident drills close the supervision and evidence gaps called out by operations stakeholders.【docs/context/alignment_briefs/operational_readiness.md】
- **Quality & observability.** Diary coverage, attribution metrics, and dashboards keep documentation honest and auditable as demanded by the observability charter.【docs/context/alignment_briefs/quality_observability.md】

## Appendix A – Evidence Reproduction Commands
1. `poetry run pytest tests/integration/test_real_data_slice_ingest.py`
2. `poetry run pytest tests/thinking/test_fast_weights.py`
3. `poetry run pytest tests/trading/test_trading_manager_execution.py -k attribution`
4. `poetry run pytest tests/runtime/test_paper_run_guardian.py`
5. `poetry run python tools/operations/incident_playbook_validation.py --run-root artifacts/reports`

The commands above regenerate the artefacts cited in this whitepaper and reproduce the guardrail evidence pack for reviewers.
