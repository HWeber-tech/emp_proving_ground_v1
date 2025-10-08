"""Legacy Parquet writer removed.

Historical data exports now flow through governed persistence surfaces.  Use
``src.data_foundation.persist.timescale`` or domain-specific exporters instead of
this shim.
"""

from __future__ import annotations

raise ModuleNotFoundError(
    "src.data_foundation.persist.parquet_writer was removed. "
    "Adopt the canonical persistence modules (e.g. src.data_foundation.persist.timescale)."
)
