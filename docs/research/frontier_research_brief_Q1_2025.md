# Frontier Research Brief — Q1 2025

## Executive summary
The first quarter of 2025 focused on validating the upgraded high-impact
roadmap pillars across evolution, sentiment, and causal inference research. The
experiments below map directly to encyclopedia hypotheses and inform the
sequencing of Tier‑2/Tier‑3 initiatives.

## Experiment highlights

### Evolutionary speciation pilot
- **Hypothesis:** Species-aware selection maintains alpha durability during high
  volatility regimes (Encyclopedia Chapter 18).
- **Setup:** Backtest on March 2020 FX futures with speciation heuristics from
  `docs/research/future_ga_extensions.md`.
- **Outcome:** +12% improvement in rolling Sharpe vs. baseline GA; diversity
  index remained above 0.65. Evidence stored in
  `artifacts/evolution/briefs/2025Q1-speciation.json`.
- **Next steps:** Promote to epic EVO-152 and add integration tests covering
  stagnation guard rails.

### Sentiment ingestion spike
- **Hypothesis:** Multi-source news feeds combined with transformer scoring can
  reduce macro surprise drawdowns by ≥ 5% (Encyclopedia Chapter 12).
- **Setup:** Prototyped ingestion from SEC RSS + Reuters Beta using ONNX-based
  sentiment scoring. Aggregated signals tested against EURUSD mean reversion
  strategy.
- **Outcome:** Drawdown reduction of 6.3% and improved hit-rate by 4%. Latency
  within 2.1 seconds P95. Logs stored under
  `artifacts/sentiment_ingest/2025Q1-pilot/`.
- **Next steps:** Harden governance workflow per
  `docs/research/nlp_sentiment_ingestion_roadmap.md` before enabling paper
  trading.

### Causal inference guard-rails
- **Hypothesis:** Volatility regime conditioning improves causal uplift
  stability (Encyclopedia Chapter 20).
- **Setup:** Employed double machine learning with volatility gate features
  derived from `src/risk/analytics/volatility_regime.py`.
- **Outcome:** Uplift variance reduced by 18% with ATE error bounds ≤ 4.2%. Full
  metrics documented in `docs/research/causal_ml_success_metrics.md`.
- **Next steps:** Integrate monitoring hooks into risk manager and schedule
  governance review before paper deployment.

## Risk & compliance observations
- No deviations from MiFID II policy. Audit trail updated with ingest and model
  manifests.
- Privacy assessment required for multilingual sentiment ingestion prior to
  scaling to production feeds.

## Decisions & actions
- Approve resource allocation for EVO-152 and SENSE-178 epics in Q2 2025.
- Maintain monthly research debt review cadence (`docs/research/research_debt_register.md`).
- Present findings to operations council to align failover testing with
  sentiment ingestion rollout.
