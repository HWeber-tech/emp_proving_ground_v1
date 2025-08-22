from __future__ import annotations

from typing import Mapping


class AnomalySensor:
    """Minimal anomaly sensor placeholder.

    Processes inputs and returns an empty list of signals for compatibility.
    """

    def process(self, data: Mapping[str, object]) -> list[dict[str, object]]:
        return []
