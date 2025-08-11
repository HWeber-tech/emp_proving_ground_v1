from __future__ import annotations

from typing import Any, Dict, List


class AnomalySensor:
    """Minimal anomaly sensor placeholder.

    Processes inputs and returns an empty list of signals for compatibility.
    """

    def process(self, data: Any) -> List[Dict[str, Any]]:
        return []


