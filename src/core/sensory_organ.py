from __future__ import annotations

from typing import Any, Dict


class SensoryOrgan:
    """Minimal sensory organ abstraction.

    Consolidated baseline to satisfy imports after cleanup; real implementations
    can extend this class and override `process`.
    """

    def __init__(self, organ_type: str) -> None:
        self.organ_type = organ_type

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"processed": True, "organ": self.organ_type, "data": data}


def create_sensory_organ(organ_type: str) -> SensoryOrgan:
    return SensoryOrgan(organ_type)


# Predefined organ instances for compatibility
WHAT_ORGAN = create_sensory_organ("what")
WHEN_ORGAN = create_sensory_organ("when")
ANOMALY_ORGAN = create_sensory_organ("anomaly")
CHAOS_ORGAN = create_sensory_organ("chaos")


