# Success Metrics for Causal Inference & ML Classifiers

## Purpose
This reference defines the quantitative and qualitative gates required before
productionising causal inference models or machine-learning classifiers within
the EMP stack.

## Global principles
- Metrics must demonstrate uplift relative to existing deterministic heuristics
  on out-of-sample windows.
- All results require reproducible experiment manifests stored in
  `artifacts/research/<project>/` with configuration hashes.
- Evaluation scripts run through CI via `tests/research/` to guard against
  regression.

## Causal inference models
- **Target metrics:**
  - Average Treatment Effect (ATE) error bound < 5% against synthetic ground
    truth benchmarks.
  - Uplift score improvement ≥ 10% vs. control strategies on rolling 3-month
    simulations.
  - Covariate balance measured via standardized mean difference < 0.1.
- **Validation workflow:**
  - Bootstrap resampling to produce 95% confidence intervals.
  - Sensitivity analysis for unobserved confounders documented in
    `docs/research/frontier_research_brief_Q1_2025.md` and follow-up reports.
  - Compliance check ensuring treatment assignments honour risk and regulatory
    constraints before experimentation.

## ML classifiers (supervised & semi-supervised)
- **Target metrics:**
  - ROC-AUC ≥ 0.75 on validation folds with ≤ 5% drift across instrument buckets.
  - Precision@k ≥ 0.6 for actionable trading signals where k aligns with average
    trade frequency.
  - Brier score ≤ 0.2 to ensure probabilistic calibration for risk sizing.
- **Calibration requirements:**
  - Reliability diagrams generated per asset class and reviewed during the
    fortnightly model governance session.
  - Platt scaling or isotonic regression applied when calibration drifts beyond
    acceptable thresholds.

## Operational readiness checks
- Automated monitoring hooks publishing `telemetry.ml.performance` and
  `telemetry.ml.drift` events with latest metrics.
- Feature attribution reports generated via SHAP or permutation importance for
  audit storage under `artifacts/research/<project>/attribution/`.
- Runbook updates referencing the relevant model guard rails prior to enabling
  live experiments.

## Promotion gate
A model advances from research to paper trading only when:
1. All metrics above meet thresholds across at least two market regimes.
2. Risk management signs off on drawdown simulations incorporating the model.
3. Compliance confirms data sourcing and decision logic adhere to documented
   policies.
4. `docs/research/research_debt_register.md` has no unresolved blockers for the
   candidate model.
