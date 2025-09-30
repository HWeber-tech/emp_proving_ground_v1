# Microstructure dataset storage runbook

The high-impact roadmap calls for archiving microstructure research datasets in
tiered storage with clear retention guidance. This runbook documents the EMP
reference implementation so operators and researchers can keep datasets aligned
with the encyclopedia’s cost matrix.

## Storage layout

Microstructure datasets are persisted in two tiers beneath the repository root:

| Tier | Path | Purpose | Retention |
| --- | --- | --- | --- |
| Hot | `artifacts/microstructure/hot/` | Recent datasets used by day-to-day research and paper-trading calibration. | 7 days by default. |
| Cold | `artifacts/microstructure/cold/` | Long-term archive that feeds historical validation and compliance audits. | 90 days by default. |

Each archived file sits inside a dataset- and date-partitioned hierarchy,
`<tier>/<dataset>/<YYYY>/<MM>/<DD>/<filename>`. Metadata is written alongside
the file as `<filename>.meta.json` so retention and lineage details survive
transfers between tiers.

## Tooling

The `scripts/archive_microstructure_data.py` CLI automates ingesting new files
and enforcing retention windows. Example usage:

```bash
python scripts/archive_microstructure_data.py ict_microstructure data/microstructure --pattern "*.parquet" --enforce-retention
```

This command:

1. Copies all matching files into the hot tier.
2. Writes metadata that records the source path, archive timestamp, and
   retention windows.
3. Moves expired files from hot to cold storage and deletes cold files that
   exceed the retention window when `--enforce-retention` is supplied.

`RetentionPolicy.from_days` inside `src/data_foundation/storage/tiered_storage.py`
exposes stricter or looser retention windows. Update the CLI flags or call the
library directly from orchestration jobs when professional deployments require
custom horizons.

## Operational checks

- **Nightly enforcement** – Schedule the CLI with `--enforce-retention` to ensure
  the hot tier only contains the most recent datasets.
- **Audit trail** – The metadata JSON includes the original source path and size
  so compliance can trace lineage during investigations.
- **Cost management** – Adjust hot/cold retention days to mirror the
  encyclopedia’s storage cost matrix for different market tiers.

Document updates should accompany any retention changes or path overrides so
operators always have a current reference.

