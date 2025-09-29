# Sofaworx consumer intelligence runway

## Concept hook from the encyclopedia
> "The WHY organ fuses macroeconomic telemetry with hyperlocal demand signals, letting the EMP sense when consumer narratives shift from confidence to caution before basket-level sales prints catch up."
>
> "Alternative data ingestion is not a novelty feature. It is a first-class sensory feed that should explain 20-30% of variance in discretionary consumer behavior ahead of earnings calls."

These promises anchor the Sofaworx expansion: a living case study that proves the 4D+1 sensory cortex can surface household-goods demand inflection points, wire them into strategy orchestration, and validate them with production-grade telemetry.

## Current implementation snapshot
- **Market narrative coverage:** WHY organ aggregates macro series (retail sales, CPI subcomponents) but lacks branded micro-signals for furniture retailers.
- **Data acquisition:** No structured ingest for Sofaworx marketing cadence, review velocity, or supply-chain telemetry. Existing ingestion stack supports REST/GraphQL connectors and S3 drops but templates are not populated.
- **Signal processing:** Sensory cortex pipelines expect normalized JSON payloads with quality bands; furniture vertical lacks schema definitions and anomaly thresholds.
- **Decision loop integration:** Strategy catalogue has discretionary retail archetypes but no canonical hook to consumer-durables demand pulses.
- **Validation artifacts:** Backtest lab includes discretionary-retail scenarios but no sofa/furniture-focused notebooks or telemetry dashboards.

## Gap closure plan
### 1. Data onboarding epics
- **EPIC:** `WHY.SOFAVORX.ALTERNATIVE_FEEDS`
  - Build connectors for:
    - Paid/organic campaign cadence (Sofaworx marketing RSS/API).
    - Social and review velocity (Trustpilot, Reddit mentions, scraping harness via existing BeautifulSoup utilities).
    - Supply-chain telemetry (containerized shipping indices tagged to HS codes for upholstered furniture).
  - Define schema in `docs/sensory_registry.md` with tiered quality gates.
  - Provision staging buckets and IaC snippets in `k8s/` to schedule nightly ingest.

### 2. Signal engineering epics
- **EPIC:** `SENSORY.SOFAVORX.FEATURE_FACTORIES`
  - Extend `src/sensory/organs/dimensions/why_organ.py` with feature builders:
    - Campaign pressure vs. conversion lag (normalized to baseline).
    - Sentiment-weighted review velocity.
    - Inventory lead-time stress index blending shipping data and store-level stockouts.
  - Calibrate anomaly bands with historical Sofaworx sales comps from SEC filings and alt-data proxies.
  - Register features in metrics facade for cross-organ aggregation.

### 3. Decision genome integration
- **EPIC:** `GENOME.SOFAVORX.DECISION_HOOKS`
  - Introduce Sofaworx demand pulse gene with typed adapters in `src/genome/models/`.
  - Wire gene into discretionary retail strategies through configuration patches in `strategies/consumer_discretionary/`.
  - Update evolutionary lab catalogue with Sofaworx personas for momentum vs. mean-reversion hypotheses.

### 4. Validation and telemetry
- **EPIC:** `TELEMETRY.SOFAVORX.REPORTING`
  - Spin up dashboards in `docs/reports/` illustrating:
    - Signal freshness (time-to-ingest, quality band score).
    - Strategy attribution vs. Sofaworx earnings surprises.
    - Guardrails: false-positive rate, missed-move diagnostics.
  - Expand `tests/current` scenarios with fixture data that simulates Sofaworx demand surges and supply shocks.

## Acceptance criteria
- Sofaworx connectors deliver validated payloads with <5% missing data across a 90-day rolling window.
- WHY organ publishes three canonical features with documented anomaly thresholds and downstream consumers.
- Decision genome expresses Sofaworx demand pulse gene and registers fitness telemetry in evolution lab runs.
- Backtests demonstrate strategy sensitivity to Sofaworx signals with statistically significant uplift vs. baseline.
- Production dashboards expose runbooks for analysts to audit Sofaworx-driven trades within 5 minutes of signal emission.

## Validation hooks
- Continuous integration jobs simulate Sofaworx payload ingestion using golden JSON fixtures.
- Mypy/pyright enforce typed adapters for new feature builders and genome genes.
- Telemetry smoke tests assert metrics registry publishes Sofaworx gauges and counters.
- Manual review checklist in `docs/operations/` updated to include Sofaworx-specific go/no-go gates before strategy promotion.

## Strategic payoff
Delivering the Sofaworx couch storyline turns a single consumer brand into a proving ground for EMP's alternative data cortex. It demonstrates that the platform can:
- Translate marketing breadcrumbs into trading signals before revenue guides shift.
- Coordinate multi-organ sensing (WHY, ANOMALY, WHEN) to protect against promotional head-fakes.
- Provide governance-grade traceability from raw JSON ingest to executed trades.

With Sofaworx live, the team can replicate the pattern across adjacent furniture and home-goods names, accelerating the roadmap toward a fully instrumented consumer discretionary pod.
