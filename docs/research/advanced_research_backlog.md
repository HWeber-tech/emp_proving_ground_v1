# Advanced Research Backlog

## Purpose

This backlog captures the forward-looking research initiatives referenced in the
Phase 3C roadmap.  Each section records the objective, required prerequisites,
acceptance criteria, and sequencing guidance so the research stream can be
planned without introducing technical debt.

---

## Genetic Algorithm Expansion

**Objective:** Extend the current moving-average crossover genome so the
evolution laboratory can explore richer configurations while preserving the
roadmap's reproducibility standards.

**Prerequisites**
- Finalise telemetry export schema for experiment manifests (extends
  `artifacts/evolution/ma_crossover/manifest.json`).
- Harden the existing GA harness to accept plug-in fitness metrics.
- Confirm storage budgets for additional artefacts in the evolution catalogue.

**Scope**
- Introduce speciation and diversity preservation operators (fitness sharing,
  crowding distance) to avoid premature convergence.
- Implement Pareto-front selection for multi-objective scoring so Sharpe,
  Sortino, drawdown, and turnover can be optimised jointly.
- Add mutation operators for regime-specific feature toggles (volatility gate,
  liquidity filter) with validation hooks in
  `trading/strategies/registry.py`.
- Document configuration presets for nightly vs ad-hoc experiments.

**Acceptance Criteria**
- Leaderboard export lists diversity metrics alongside existing fitness
  statistics.
- Experiment manifests enumerate the new operators and the random seeds used.
- CI smoke tests replay a reduced experiment to verify determinism.

---

## NLP & News Sentiment Ingestion Roadmap

**Objective:** Define the ingestion and governance approach for sentiment data
covering encyclopedia Tier-2 signals.

**Prerequisites**
- Approve data vendor short-list (free: RSS/NewsAPI; premium: RavenPack,
  Bloomberg alternative feeds).
- Extend `data_foundation.ingest.configuration` to support categorised
  pipelines (market data vs news).
- Validate legal/compliance review checklist with the compliance team.

**Roadmap Phases**
1. **Prototype ingestion**: Build RSS/NewsAPI adapters, store raw payloads in
   tiered storage, and emit ingestion quality metrics via
   `operations.ingest_trends`.
2. **Sentiment modelling**: Evaluate baseline transformer models (FinBERT,
   Llama derivatives) with causal guardrails before production.
3. **Integration**: Feed structured sentiment scores into WHY sensors and
   document the configuration toggles for strategies and risk managers.
4. **Governance**: Implement retention, redaction, and access logging aligned
   with KYC/AML requirements and privacy guidance.

**Acceptance Criteria**
- Architecture decision record approved covering data lineage and governance.
- Synthetic dry-run demonstrates data quality metrics and telemetry coverage.
- Strategy registry exposes feature flags to consume sentiment signals.

---

## Success Metrics for Causal Inference & ML Classifiers

**Objective:** Establish measurable success criteria before expanding into
higher-complexity models.

| Domain | Primary Metrics | Guardrails | Review Cadence |
| --- | --- | --- | --- |
| Causal inference (ATE uplift) | Precision of treatment effect estimates, overlap diagnostics | Sensitivity analysis across confounder assumptions, placebo tests | Quarterly |
| Regime classification | Macro/volatility regime accuracy, confusion matrix stability | Maximum drawdown impact vs baseline, drift monitoring via `sensor_drift.py` | Monthly |
| Trade outcome prediction | ROC-AUC, precision/recall on out-of-sample sessions | Calibration error < 5%, slippage impact tracked in execution journal | Monthly |
| Sentiment scoring | Correlation with realised volatility and PnL attribution | Bias audits across asset classes, compliance sign-off for data usage | Monthly |

**Definition of Done**
- Metrics are embedded into experiment notebooks and CI checkpoints.
- Each model publishes a Markdown scorecard with historical performance and
  guardrail status.
- Alerts trigger when guardrails breach for two consecutive review cycles.

---

## Encyclopedia Tier-2/Tier-3 Epics Mapping

| Vision Item | Epic | Sequencing Notes |
| --- | --- | --- |
| Tier-2: Evolutionary orchestration | `EVO-201` – Multi-strategy genome exchange | Blocked by GA expansion milestone; target after diversity operators land. |
| Tier-2: Sentiment-aware execution | `EXEC-185` – Integrate sentiment features into execution model | Requires NLP ingestion MVP and risk sign-off. |
| Tier-3: Autonomous risk reallocations | `RISK-240` – Adaptive capital allocator | Depends on causal inference metrics and expanded telemetry. |
| Tier-3: Knowledge graph of macro factors | `DATA-178` – Macro knowledge graph ingestion | Start after sentiment governance is production-ready. |
| Tier-3: Antifragile reinforcement loop | `AI-310` – Reinforcement experimentation harness | Requires successful deployment of `EVO-201` and `RISK-240`. |

**Portfolio Guidance**
- Maintain the epic list in the engineering planning system, referencing this
  table for sequencing.
- Review dependencies during quarterly roadmap refreshes to prevent scheduling
  conflicts.

---

## Research Debt Register

| Category | Description | Owner | Status | Next Review |
| --- | --- | --- | --- | --- |
| Data gaps | Lack of high-frequency depth data for liquidity studies | Data Engineering | Open | 2025-05-15 |
| Model validation | Need for cross-venue robustness checks on GA outputs | Quant Research | Open | 2025-05-01 |
| Compliance | Clarify archival requirements for sentiment payloads | Compliance Lead | In progress | 2025-04-30 |
| Tooling | Automate experiment reproducibility audits | Platform | Open | 2025-05-10 |
| Documentation | Expand encyclopedia cross-references for new sensors | Research Ops | In progress | 2025-04-28 |

**Process Notes**
- The research steering committee reviews the register on the **first Monday of
  each month** and records outcomes in the meeting minutes.
- Outstanding items that persist beyond two review cycles escalate to the
  portfolio risk log for prioritisation.

