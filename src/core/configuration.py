"""src.core.configuration has been removed in favour of `SystemConfig`.

Importers must migrate to `src.governance.system_config` and use the
`SystemConfig` dataclass as the configuration source of truth.

The legacy module exposed a mutable compatibility layer that duplicated
configuration logic, masking governance overrides and breaking type safety.
It now raises a clear error to prevent silent usage.
"""

from __future__ import annotations

raise ModuleNotFoundError(
    "src.core.configuration was removed. Import SystemConfig from "
    "src.governance.system_config instead."
)
