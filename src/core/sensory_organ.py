from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class SensoryOrganProto(Protocol):
    """Structural type for sensory organs (process-only surface we rely on)."""

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        ...


class CoreSensoryOrgan:
    """Minimal sensory organ abstraction (legacy shim).

    Notes:
    - This module intentionally avoids importing from src.sensory.* to satisfy
      layered architecture contracts. Use orchestration to wire concrete sensory organs.
    """

    def __init__(self, organ_type: str) -> None:
        self.organ_type = organ_type

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"processed": True, "organ": self.organ_type, "data": data}


# Backward-compat alias for legacy imports that expect `SensoryOrgan` to exist
SensoryOrgan = CoreSensoryOrgan


def create_sensory_organ(organ_type: str) -> CoreSensoryOrgan:
    return CoreSensoryOrgan(organ_type)


# Predefined organ instances for compatibility
WHAT_ORGAN = create_sensory_organ("what")
WHEN_ORGAN = create_sensory_organ("when")
ANOMALY_ORGAN = create_sensory_organ("anomaly")
CHAOS_ORGAN = create_sensory_organ("chaos")
