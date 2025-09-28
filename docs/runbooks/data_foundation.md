# Data Foundation Runbook

This runbook captures the operational workflows that back the roadmap's
"Data Foundation Hardening" track.  The goal is to ensure every developer and
operator can hydrate, inspect, and validate datasets without bespoke scripts.

## Canonical Pricing Pipeline

* Entrypoint: `scripts/data_bootstrap.py`
* Core logic: `src/data_foundation/pipelines/pricing_pipeline.py`

The bootstrap CLI wraps the `PricingPipeline` so Yahoo, Alpha Vantage, and
deterministic file vendors share the same normalisation and validation flow.
It persists artefacts via `PricingCache`, which writes Parquet (with CSV
fallback) plus JSON metadata into `data_foundation/cache/pricing/`.  Retention
and entry limits can be configured from the CLI (`--cache-retention-days` and
`--cache-max-entries`).

Common usage:

```bash
poetry run python scripts/data_bootstrap.py \
  --symbols "EURUSD=X,GBPUSD=X" \
  --vendor yahoo \
  --lookback-days 120 \
  --cache-retention-days 7
```

The command prints a JSON summary with dataset location, row counts, and any
quality issues raised by the pipeline validators.

## Reference Data Loader

* Module: `src/data_foundation/reference/reference_data_loader.py`

The `ReferenceDataLoader` consolidates instrument metadata, trading sessions,
and holiday calendars.  It defaults to the repository configs:

* Instruments — `config/system/instruments.json`
* Sessions — `config/reference/sessions.json`
* Holidays — `config/reference/holidays.json`

The loader memoizes results and exposes a `refresh` flag when changes need to
be detected at runtime.  Use it to seed services such as the strategy registry
or risk engine with a single, validated source of truth:

```python
from src.data_foundation.reference import ReferenceDataLoader

loader = ReferenceDataLoader()
dataset = loader.load_all()
instrument = dataset.instruments["EURUSD"]
```

Custom deployments can override the config paths when integrating with managed
data stores.

## Artefact Expectations

* Parquet/CSV datasets stored under `data_foundation/cache/pricing/`
* Metadata JSON files suffixed with `_metadata.json`
* Issues JSON files suffixed with `_issues.json`
* Reference data lives under `config/reference/`

All artefacts are Git-ignored and regenerated on demand; the runbook ensures a
repeatable workflow for onboarding new venues or data vendors without drifting
from the encyclopedia baseline.
