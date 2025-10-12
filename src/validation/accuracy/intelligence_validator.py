"""Legacy understanding validator shim removed.

The accuracy helpers now live exclusively under
``src.validation.accuracy.understanding_validator``. Importing this module
previously re-exported the canonical implementation, which allowed
namespace drift to persist. We now raise a targeted error so remaining
references fail fast and adopt the understanding terminology.
"""

from __future__ import annotations

raise ModuleNotFoundError(
    "src.validation.accuracy.intelligence_validator was removed. Import "
    "UnderstandingValidator from "
    "src.validation.accuracy.understanding_validator instead."
)
