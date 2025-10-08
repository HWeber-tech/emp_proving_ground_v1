"""Legacy JSONL writer removed.

Event persistence flows now route through the canonical Timescale pipelines and
telemetry recorders under ``src.data_foundation.persist.timescale``.
This stub blocks the legacy helper from resurfacing.
"""

from __future__ import annotations

raise ModuleNotFoundError(
    "src.data_foundation.persist.jsonl_writer was removed. "
    "Use the Timescale persistence helpers in src.data_foundation.persist.timescale "
    "for managed ingest storage."
)
