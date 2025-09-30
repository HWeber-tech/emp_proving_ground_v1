# Market Microstructure Observatory

The high-impact roadmap requires a "Market Microstructure Observatory" that
highlights liquidity, depth, and latency characteristics for priority venues.
This document pairs the automatically generated Markdown report with guidance
for analysts adding new studies.

## Artefact generation pipeline

1. Capture raw order-book telemetry into `docs/microstructure_raw_data.json`. The
   fixture shipped in this repository contains a representative EURUSD sample.
2. Run `scripts/generate_market_microstructure_observatory.py` to transform the
   JSON payload into a Markdown report stored at
   `artifacts/reports/market_microstructure_observatory.md`.
3. Optionally open the generated report in Jupyter or any Markdown notebook to
   annotate findings, attach charts, or embed additional venue comparisons.
4. Attach the artefact to the deployment readiness checklist so operations can
   track liquidity shifts over time.

```bash
python scripts/generate_market_microstructure_observatory.py \
  docs/microstructure_raw_data.json \
  --output artifacts/reports/market_microstructure_observatory.md
```

## Extending the observatory

- Add venue-specific raw captures under `docs/microstructure/` with descriptive
  filenames (e.g., `eurusd_ic_markets_20250202.json`).
- Update the script's input to point at the desired dataset or wrap the script in
  a simple loop to batch-generate reports.
- Incorporate additional metrics such as imbalance, sweep frequency, or spread
  persistence by enriching the JSON payload and extending
  `build_markdown()` in the script.
- When notebooks are preferred, import the generated Markdown into a notebook
  cell via `IPython.display.Markdown` to maintain a single source of truth for the
  calculations.

## Operational handshake

- Upload the latest observatory artefact alongside the nightly reconciliation
  bundle so traders and engineers can review market conditions during the daily
  stand-up.
- Raise alerts through `scripts/status_metrics.py` when latency or depth metrics
  diverge materially from the rolling baseline captured in these reports.
- Feed summary metrics into the observability dashboard described in the roadmap
  to visualise depth stability alongside FIX health and risk posture.
