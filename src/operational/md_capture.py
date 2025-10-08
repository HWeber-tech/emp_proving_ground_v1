"""Legacy market-data capture shim removed.

Managed ingest and replay now run through the data foundation pipelines and
Timescale snapshots under ``src.data_foundation.ingest``.  Retire usage of this
module in favour of the canonical ingestion services.
"""

from __future__ import annotations

raise ModuleNotFoundError(
    "src.operational.md_capture was removed. Use the managed ingest pipelines under "
    "src.data_foundation.ingest instead."
)
