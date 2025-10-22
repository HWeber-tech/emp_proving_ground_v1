# Fundamental Data Provider Selection

## Summary
This document evaluates multiple fundamental data providers for the EMP Proving Ground and recommends a primary/secondary combination that satisfies coverage, latency, and governance requirements. After comparing Financial Modeling Prep (FMP), Polygon.io, Intrinio, Alpha Vantage, and Quandl/Nasdaq Data Link, the recommended mix is:

- **Primary**: Financial Modeling Prep (FMP) for broad equity fundamentals, earnings calendars, and economic indicators at an attractive price point.
- **Secondary**: Polygon.io for U.S. equity fundamentals paired with high-quality real-time/streaming extensions and corporate action coverage.
- **Tertiary fallback**: Intrinio for specialty datasets (banking, options greeks) and as a compliance-friendly vendor with robust SLA options.

This trio balances data breadth, reliability, and integration complexity while leaving room to escalate to institutional vendors (FactSet, Bloomberg) once budgets allow.

## Evaluation Criteria
The following criteria are weighted according to Phase 1 roadmap objectives:

| Criterion | Weight | Rationale |
| --- | --- | --- |
| **Coverage breadth** | 25% | Need global equities plus ETFs/indices for WHY sensor inputs. |
| **Data freshness & latency** | 20% | Earnings updates and corporate actions must arrive within hours. |
| **Historical depth** | 15% | Backfilling valuation models requires 10+ years of statements. |
| **API quality & documentation** | 15% | Stable, well-documented APIs reduce integration effort. |
| **Cost & licensing** | 15% | Phase 1 budget assumes <$1k/month combined spend. |
| **Compliance & SLA options** | 10% | Vendor must allow redistribution to internal strategy components and support audit trails. |

Each provider was scored 1–5 per criterion (5 = best). Scores incorporate published specs, community feedback, and prior trading-stack experience.

## Provider Comparison

| Provider | Coverage | Freshness | Historical Depth | API Quality | Cost | Compliance | Weighted Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Financial Modeling Prep** | 4 | 4 | 4 | 4 | 5 | 3 | **4.1** |
| **Polygon.io** | 3 | 5 | 3 | 5 | 3 | 4 | **3.9** |
| **Intrinio** | 3 | 4 | 4 | 3 | 3 | 5 | **3.6** |
| Alpha Vantage | 2 | 2 | 2 | 3 | 5 | 2 | 2.8 |
| Quandl / Nasdaq Data Link | 3 | 3 | 5 | 3 | 2 | 3 | 3.1 |

### Financial Modeling Prep (FMP)
- **Strengths**: Comprehensive U.S. and growing international coverage, including financial statements, ratios, SEC filings, earnings, macroeconomic series. Straightforward REST API with generous rate limits on paid tiers. CSV/JSON outputs support deterministic ingestion.
- **Gaps**: SLA and uptime commitments are weaker than institutional vendors; best-effort support.
- **Integration Notes**: Map directly into `fundamental_store.py` schema; supports historical revision tracking via the "historical" endpoints.

### Polygon.io
- **Strengths**: Excellent API ergonomics, real-time WebSocket channels for future Phase 1.2 streaming, detailed corporate actions, dividends, and splits. Earnings surprises and analyst estimates supplement valuation models.
- **Gaps**: Fundamentals currently limited to U.S. equities; pricing higher once tick data is added.
- **Integration Notes**: Reuse existing `data_foundation/streaming` infrastructure for real-time expansions; REST fundamentals fit the same normalization pipeline.

### Intrinio
- **Strengths**: Strong compliance posture with signed agreements, tiered SLAs, and niche datasets (bank regulatory, options). Supports file-based bulk downloads for audit trails.
- **Gaps**: Higher costs per dataset; API rate limits stricter without enterprise contract.
- **Integration Notes**: Use for specialty coverage and as audit-grade fallback if primary feeds fail compliance checks.

### Alpha Vantage
- **Strengths**: Low cost, simple API, community familiarity.
- **Gaps**: Limited fundamentals, aggressive throttling, and inconsistent data quality; unsuitable for institutional-grade requirements.

### Quandl / Nasdaq Data Link
- **Strengths**: Excellent historical depth, especially for macroeconomic time series.
- **Gaps**: Patchwork licensing across datasets, slower updates for earnings, variable schema per dataset; higher maintenance burden.

## Recommended Strategy
1. **Contract FMP (Tier: Pro or Enterprise)** as the primary source for financial statements, ratios, earnings calendars, and macroeconomic indicators.
2. **Add Polygon.io (Starter or Advanced)** to capture corporate actions, dividends, and earnings surprises with lower latency plus prepare groundwork for streaming tick integration.
3. **Negotiate Intrinio Essentials** as an optional add-on for regulatory-grade datasets and redundancy for critical filings.
4. Implement ingestion adapters in priority order: `fmp_adapter.py`, `polygon_adapter.py`, then `intrinio_adapter.py`, each conforming to the normalized schema defined in `fundamental_store.py`.
5. Establish nightly quality checks comparing overlapping metrics (EPS, revenue) between FMP and Polygon, escalating to Intrinio when discrepancies exceed defined tolerances.

## Next Steps
- Draft procurement request with budget breakdown (~$600–$800/month combined).
- Schedule trial API keys and implement smoke tests in CI using sanitized fixtures.
- Update data governance policy with vendor-specific usage terms and retention limits.
- Coordinate with Observability team to capture API latency/error metrics once adapters are live.
