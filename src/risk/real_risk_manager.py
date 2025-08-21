from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class RealRiskConfig:
    max_position_risk: float = 0.0
    max_drawdown: float = 0.0


class RealRiskManager:
    def __init__(self, config: RealRiskConfig) -> None:
        self.config = config

    def assess_risk(self, positions: Dict[str, float]) -> float:
        return 0.0


__all__ = ["RealRiskConfig", "RealRiskManager"]