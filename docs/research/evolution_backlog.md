# Evolution Lab Follow-On Backlog

This backlog captures the research-oriented extensions that succeed the
moving-average crossover GA proof of concept.  It translates the
unchecked roadmap items into actionable research themes so the
Evolution Lab stays aligned with the encyclopedia vision.

## Speciation & Diversity Controls
- Prototype Pareto-front selection that preserves genomes with unique
  risk/return trade-offs.
- Design speciation heuristics (e.g., niche distance on MA windows and
  risk toggles) and integrate with the population manager.
- Document evaluation metrics for diversity (Herfindahl index, entropy)
  and surface them beside the leaderboard snapshots.

## Multi-Objective Reporting
- Extend the fitness report manifest with vector scores (Sharpe,
  Sortino, max drawdown, Calmar) and compute Pareto dominance per
  generation.
- Generate Markdown/CSV artefacts that visualise the Pareto front and
  highlight promoted champions.

## Reproducibility & Experiment Manifests
- Persist experiment manifests (config, dataset hash, seed, commit
  SHA) to a version-controlled `artifacts/evolution/` tree.
- Introduce a lightweight schema validation step that ensures future
  experiments record the same manifest fields.

## Catalogue & Promotion Workflow
- Define thresholds for promoting genomes into the strategy catalogue.
- Add supervisory approval hooks to the strategy registry before
  activating Evolution Lab champions in paper trading.
- Capture promotion decisions (who approved, why) for auditability and
  align with governance controls.

## Data Expansion
- Backtest GA runs on diversified datasets (FX majors, equity indices,
  crypto) and record dataset manifests for comparison.
- Explore higher-frequency data slices and document performance
  degradations/improvements relative to the bootstrap daily feed.

## Operational Instrumentation
- Stream Evolution Lab telemetry (fitness trends, registration status)
  to the runtime event bus and expose in the professional dashboards.
- Add failure runbooks covering experiment crashes, registry locking,
  and manifest mismatches.

## Research Debt Register
- Track open questions (e.g., adaptive mutation rates, ensemble
  promotion strategies) in a quarterly report.
- Review backlog monthly with the governance committee and annotate
  resolved items with lessons learned.
