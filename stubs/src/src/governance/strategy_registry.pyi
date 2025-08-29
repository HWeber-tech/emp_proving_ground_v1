from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
from enum import Enum

class StrategyStatus(Enum):
    EVOLVED: str
    APPROVED: str
    ACTIVE: str
    INACTIVE: str

class StrategyRegistry:
    def register_strategy(self, strategy_id: str, config: Dict[str, Any]) -> bool: ...
    def get_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]: ...
    def list_strategies(self) -> List[Dict[str, Any]]: ...
    def update_strategy_status(self, strategy_id: str, status: Union["StrategyStatus", str]) -> bool: ...

__all__ = ["StrategyRegistry", "StrategyStatus"]