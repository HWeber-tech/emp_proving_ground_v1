# Market Microstructure Data Archival Playbook

The High-Impact Development Roadmap calls for tiered storage so that
microstructure datasets remain actionable for daily operations while also
supporting encyclopedia-aligned research. This playbook documents the
operational process now implemented in the repository.

## Overview

- **Hot tier** retains 5–14 days of recent market structure events for
  interactive diagnostics, dashboards, and reconciliations.
- **Cold tier** retains 6–12 months of aggregated datasets that feed the
  Market Microstructure Observatory notebooks and long-horizon studies.
- **Metadata** describing each archival run is written to
  `artifacts/microstructure/metadata/<dataset>.json` for traceability.

## Running the Archiver

```bash
python scripts/archive_microstructure_datasets.py \
  config/operational/microstructure_archival.yaml \
  --report artifacts/microstructure/archive_report.json
```

- Set `--dry-run` to preview actions without copying or deleting files.
- Reports are JSON encoded so they can be published as CI artifacts or
  ingested by downstream monitoring.

## Configuration

Policies live in
`config/operational/microstructure_archival.yaml`. Each dataset entry
specifies:

| Field | Description |
| --- | --- |
| `name` | Logical dataset name used for reporting and metadata |
| `source` | Directory containing raw captures |
| `hot` | Hot-tier destination |
| `cold` | Cold-tier destination (optional) |
| `hot_retention_days` | Days to keep in the hot tier |
| `cold_retention_days` | Days to keep in the cold tier |
| `description` | Context for operators and auditors |

## CI Integration

The archiver is designed for automation:

- CI workflows can run the script after paper-trading or backtests to
  persist new events.
- Metadata files are idempotent; rerunning the script overwrites the
  last summary so dashboards always reflect the latest state.
- Downstream analytics may tail the metadata directory to trigger
  notebook refreshes.

## Incident Response

If archival fails, operators should:

1. Inspect the JSON report for datasets marked `missing`.
2. Verify source directories are hydrated (e.g., sensors produced data).
3. Re-run the script with `--dry-run` to confirm corrective actions.
4. Escalate persistent failures to the data foundation squad and log the
   incident in the Ops Command checklist.

This procedure completes the roadmap requirement for tiered storage and
aligns the implementation with Encyclopedia Appendix D retention guidance.
