# Market Data Lineage & Quality SLA

_Auto-generated via `scripts/generate_data_lineage.py`; do not edit manually._

This document captures the Tier-0/Tier-1 market data flow, owners, service levels, and retention expectations described in the EMP Encyclopedia. It is generated programmatically to ensure the documentation stays aligned with the implementation under src/data_foundation.

## Assumptions
- Freshness SLAs assume scheduled runs every 15 minutes within trading hours.
- Completeness targets exclude provider-wide outages acknowledged by vendor status pages.
- Retention policies inherit from config/data_foundation/retention.yaml when deployed in production.

## Service Levels by Dataset

| Layer | Dataset | Freshness SLA | Completeness Target | Retention | Owners |
| --- | --- | --- | --- | --- | --- |
| Analytic | Sensor Feature Store | 30m | 98.0% | Intraday features retained for 14 days; end-of-day aggregates kept for 1 year to support encyclopedia Tier-1 audits. | Sensory Team |
| Ingestion | Normalised OHLCV Bars | 20m | 99.5% | Rolling 90 days accessible in parquet format under data_foundation/cache/hot/. Quarterly snapshots compressed into artifacts/data/cache/ for encyclopaedia-aligned cold retention. | Data Foundation, Trading Ops |
| Source | Macro Calendar Snapshots | 1h | 97.0% | 12 months retained to support seasonality studies; older entries archived into docs/reports/macro/ per encyclopedia data governance guidance. | Data Foundation |
| Source | Raw Vendor Snapshots | 15m | 99.0% | Hot storage retains 30 days of tick/interval captures; cold storage archives monthly bundles for 2 years in artifacts/data/vendor/. | Data Foundation |
| Validation | Quality Diagnostics | 25m | 100.0% | Last 180 days stored in structured JSON under artifacts/data/quality/. Findings older than 180 days summarised into monthly markdown reports. | Data Reliability, Trading Ops |

### Sensor Feature Store

Feature parquet files produced by HOW/WHEN/WHY sensory organs after ingesting normalised bars and macro calendars. Serves both strategies and risk sizing.

- **Layer:** Analytic
- **Upstream:** Normalised OHLCV Bars, Quality Diagnostics, Macro Calendar Snapshots
- **Downstream:** Strategy Backtests, Evolution Lab, Risk Scenario Runner
- **Freshness SLA:** 30m
- **Completeness Target:** 98.0%
- **Retention:** Intraday features retained for 14 days; end-of-day aggregates kept for 1 year to support encyclopedia Tier-1 audits.
- **Quality Controls:**
  - Cross-sensor dependency checks
  - Schema fingerprinting
  - Sensor drift monitoring
- **Notes:**
  - Feature definitions documented via tools.sensory.registry exporter.

### Normalised OHLCV Bars

Canonical bar set returned by MultiSourceAggregator after column normalisation, timezone harmonisation, and symbol reconciliation.

- **Layer:** Ingestion
- **Upstream:** Raw Vendor Snapshots
- **Downstream:** CoverageValidator, CrossSourceDriftValidator, StalenessValidator, PricingPipeline
- **Freshness SLA:** 20m
- **Completeness Target:** 99.5%
- **Retention:** Rolling 90 days accessible in parquet format under data_foundation/cache/hot/. Quarterly snapshots compressed into artifacts/data/cache/ for encyclopaedia-aligned cold retention.
- **Quality Controls:**
  - Column schema enforcement
  - Symbol canonicalisation
  - Timezone coercion to UTC
- **Notes:**
  - Serves as the baseline dataset for pricing pipelines and sensors.

### Macro Calendar Snapshots

Economic calendar events curated from open data sources and normalised into WHY-dimension signals consumed by strategies and risk modules.

- **Layer:** Source
- **Upstream:** (root)
- **Downstream:** Sensor Feature Store, Risk Scenario Runner
- **Freshness SLA:** 1h
- **Completeness Target:** 97.0%
- **Retention:** 12 months retained to support seasonality studies; older entries archived into docs/reports/macro/ per encyclopedia data governance guidance.
- **Quality Controls:**
  - Duplicate event suppression
  - Timezone reconciliation
- **Notes:**
  - SLA follows encyclopedia Appendix B macro data timetable.

### Raw Vendor Snapshots

Source files fetched directly from Yahoo Finance, Alpha Vantage, and FRED using provider adapters registered with the multi-source aggregator. Files are stored under data/vendor/ for replayability.

- **Layer:** Source
- **Upstream:** (root)
- **Downstream:** MultiSourceAggregator
- **Freshness SLA:** 15m
- **Completeness Target:** 99.0%
- **Retention:** Hot storage retains 30 days of tick/interval captures; cold storage archives monthly bundles for 2 years in artifacts/data/vendor/.
- **Quality Controls:**
  - HTTP fetch status and schema checks
  - Provider-specific throttling alerts
- **Notes:**
  - Sourcing windows align with Encyclopedia Tier-0 free vendor cadence.

### Quality Diagnostics

Structured findings emitted by Coverage-, Drift-, and Staleness-validators in src.data_foundation.ingest.multi_source. Results drive alerting pipelines and reconciliation dashboards.

- **Layer:** Validation
- **Upstream:** Normalised OHLCV Bars
- **Downstream:** Risk Analytics, PnL Dashboard, Data Foundation Runbooks
- **Freshness SLA:** 25m
- **Completeness Target:** 100.0%
- **Retention:** Last 180 days stored in structured JSON under artifacts/data/quality/. Findings older than 180 days summarised into monthly markdown reports.
- **Quality Controls:**
  - Latency watermark comparisons
  - Missing-bar tolerance checks
  - Severity escalation thresholds
- **Notes:**
  - Validators cover Tier-0/Tier-1 encyclopedia requirements.
