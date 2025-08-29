from __future__ import annotations
from typing import Any, Dict, Optional

class RiskConfig:
    max_drawdown: float
    leverage: int
    def __init__(self) -> None: ...

def load_config(path: str) -> RiskConfig: ...