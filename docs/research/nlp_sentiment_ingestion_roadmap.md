# NLP & News Sentiment Ingestion Roadmap

## Vision
Deliver institutional-grade textual intelligence that augments the 4D+1 sensory
stack with real-time macro, corporate, and sentiment signals while respecting
data governance and compliance obligations.

## Phased implementation

### Phase 0 – Foundations (Weeks 1–2)
- Catalogue open-source and commercial feeds (Reuters, Bloomberg Beta, RavenPack
  Lite, SEC RSS) with licensing notes.
- Extend `src/data_foundation/ingest/multisource.py` to register textual
  connectors with schema contracts (headline, source, publication timestamp,
  sentiment score placeholder).
- Define Kafka topic schema (`telemetry.sentiment.raw`) mirroring ingest metadata
  for downstream consumers.
- Document consent, retention, and transformation policy in
  `docs/policies/data_governance.md`.

### Phase 1 – Sentiment scoring (Weeks 3–6)
- Implement lightweight transformer-based sentiment scoring service leveraging
  Hugging Face pipelines with ONNX runtime for deterministic deployment.
- Emit per-article sentiment, topic tags, and confidence metrics to the ingest
  Kafka topic.
- Surface aggregated sentiment factors via `SentimentSensor` within
  `src/sensory/why/` and document integration tests in
  `tests/sensory/test_sentiment_sensor.py`.
- Introduce drift detection comparing sentiment baselines against historical
  medians to guard against news feed outages or classifier shifts.

### Phase 2 – Contextual enrichment (Weeks 7–10)
- Fuse macro calendar context (`src/data_foundation/services/macro_events.py`)
  with sentiment events to tag articles by macro regime, release surprise, and
  affected instruments.
- Add entity-resolution against the instrument catalogue so portfolio risk can
  link textual signals to exposure.
- Provide dashboards in `docs/reports/sentiment/` summarising coverage, latency,
  and factor impacts.

### Phase 3 – Compliance & governance hardening (Weeks 11–12)
- Implement retention controls aligned with jurisdictional requirements,
  including data minimisation and right-to-erasure workflows.
- Extend compliance telemetry (`src/operations/regulatory_telemetry.py`) with
  sentiment ingestion audits (feed coverage, consent status, classifier version).
- Conduct privacy impact assessment; record outcomes in
  `docs/policies/privacy_impact_assessment.md`.

## Data governance checklist
- ✅ Licensing and terms-of-use documented before ingest begins.
- ✅ PII scrubbing pipeline validated with synthetic regression tests.
- ✅ Consent provenance stored alongside ingest metadata.
- ✅ Retention policies codified in Terraform/Ansible infrastructure scripts.
- ✅ Regular compliance review scheduled via
  `docs/research/research_debt_register.md`.

## Success criteria
- Sentiment signals consumed by at least two production strategies with recorded
  uplift in Sharpe ratio or drawdown resilience.
- Drift detection alerts integrated into the operations dashboard.
- All news ingestion operations logged for audit with reproducible manifests
  stored under `artifacts/sentiment_ingest/`.
