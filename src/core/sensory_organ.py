from __future__ import annotations

from typing import Any, Dict

# Re-export canonical SensoryOrgan base to avoid duplicate class definitions
from src.sensory.organs.dimensions.base_organ import SensoryOrgan as SensoryOrgan


class CoreSensoryOrgan:
    """Minimal sensory organ abstraction (legacy shim).

    This shim preserves runtime behavior for legacy factory usage while the name
    `SensoryOrgan` is re-exported from the canonical sensory base module.
    """

    def __init__(self, organ_type: str) -> None:
        self.organ_type = organ_type

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"processed": True, "organ": self.organ_type, "data": data}


def create_sensory_organ(organ_type: str) -> CoreSensoryOrgan:
    return CoreSensoryOrgan(organ_type)


# Predefined organ instances for compatibility
WHAT_ORGAN = create_sensory_organ("what")
WHEN_ORGAN = create_sensory_organ("when")
ANOMALY_ORGAN = create_sensory_organ("anomaly")
CHAOS_ORGAN = create_sensory_organ("chaos")


