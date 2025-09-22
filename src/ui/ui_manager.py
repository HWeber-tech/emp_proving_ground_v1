"""
UIManager - Central interface for human system interaction

Provides unified interface for CLI and Web API to interact with:
- EventBus (NATS)
- StrategyRegistry
- System monitoring
- Strategy management
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, cast

if TYPE_CHECKING:
    from src.core.event_bus import EventBus
    from src.governance.strategy_registry import StrategyRegistry, StrategyStatus
else:
    try:
        from src.core.event_bus import EventBus
        from src.governance.strategy_registry import StrategyRegistry, StrategyStatus
    except ImportError:
        # Mock implementations for standalone testing (suffixed to avoid redefinition at type time)
        class EventBusStub:
            def __init__(self) -> None:
                self.connected = False

            async def connect(self) -> None:
                self.connected = True

            async def disconnect(self) -> None:
                self.connected = False

            async def publish(self, subject: str, data: Dict[str, Any]) -> None:
                print(f"📡 Event: {subject} -> {data}")

        class StrategyStatusStub:
            EVOLVED = "evolved"
            APPROVED = "approved"
            ACTIVE = "active"
            INACTIVE = "inactive"

        class StrategyRegistryStub:
            def __init__(self) -> None:
                self.strategies: Dict[str, Dict[str, Any]] = {}

            def register_strategy(self, strategy_id: str, config: Dict[str, Any]) -> bool:
                self.strategies[strategy_id] = {
                    "id": strategy_id,
                    "status": StrategyStatusStub.EVOLVED,
                    "config": config,
                    "created_at": datetime.now().isoformat(),
                }
                return True

            def get_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
                return self.strategies.get(strategy_id)

            def list_strategies(self) -> List[Dict[str, Any]]:
                return list(self.strategies.values())

            def update_strategy_status(self, strategy_id: str, status: str) -> bool:
                if strategy_id in self.strategies:
                    self.strategies[strategy_id]["status"] = status
                    return True
                return False

        # Runtime aliases
        EventBus = EventBusStub  # type: ignore[assignment]
        StrategyStatus = StrategyStatusStub  # type: ignore[assignment]
        StrategyRegistry = StrategyRegistryStub  # type: ignore[assignment]


class UIManager:
    """
    Central manager for human interface operations

    Provides unified interface for:
    - System status monitoring
    - Strategy management
    - Event broadcasting
    - Configuration access
    """

    def __init__(self) -> None:
        self.event_bus: Any = EventBus()
        self.strategy_registry = StrategyRegistry()
        self._connected = False

    async def initialize(self) -> bool:
        """Initialize all UI components"""
        try:
            await self.event_bus.connect()
            self._connected = True
            return True
        except Exception as e:
            print(f"❌ Failed to initialize UIManager: {e}")
            return False

    async def shutdown(self) -> None:
        """Clean shutdown of UI components"""
        if self._connected:
            await self.event_bus.disconnect()
            self._connected = False

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        casted = cast(
            Callable[[], List[Dict[str, Any]]],
            getattr(self.strategy_registry, "list_strategies", lambda: []),
        )
        strategies = casted()
        return {
            "timestamp": datetime.now().isoformat(),
            "event_bus_connected": self._connected,
            "total_strategies": len(strategies),
            "active_strategies": len(
                [s for s in strategies if s.get("status") == StrategyStatus.ACTIVE]
            ),
        }

    def list_strategies(self) -> List[Dict[str, Any]]:
        """Get all strategies with their status"""
        casted = cast(
            Callable[[], List[Dict[str, Any]]],
            getattr(self.strategy_registry, "list_strategies", lambda: []),
        )
        return casted()

    def get_strategy_details(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific strategy"""
        return cast(Optional[Dict[str, Any]], self.strategy_registry.get_strategy(strategy_id))

    def approve_strategy(self, strategy_id: str) -> bool:
        """Approve an evolved strategy for live trading"""
        return cast(
            bool,
            self.strategy_registry.update_strategy_status(
                strategy_id, StrategyStatus.APPROVED.value
            ),
        )

    def activate_strategy(self, strategy_id: str) -> bool:
        """Activate an approved strategy for live trading"""
        return cast(
            bool,
            self.strategy_registry.update_strategy_status(strategy_id, StrategyStatus.ACTIVE.value),
        )

    def deactivate_strategy(self, strategy_id: str) -> bool:
        """Deactivate an active strategy"""
        return cast(
            bool,
            self.strategy_registry.update_strategy_status(
                strategy_id, StrategyStatus.INACTIVE.value
            ),
        )

    async def broadcast_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Broadcast an event to all connected clients"""
        if self._connected:
            await self.event_bus.publish(f"ui.{event_type}", data)

    def format_strategy_table(self, strategies: List[Dict[str, Any]]) -> str:
        """Format strategies as a readable table"""
        if not strategies:
            return "No strategies found"

        headers = ["ID", "Status", "Created", "Config"]
        rows = []

        for strategy in strategies:
            rows.append(
                [
                    strategy.get("id", "N/A")[:20],
                    strategy.get("status", "N/A"),
                    strategy.get("created_at", "N/A")[:19],
                    str(strategy.get("config", {}))[:30],
                ]
            )

        # Simple table formatting
        parts: List[str] = []
        parts.append(f"{'ID':<20} {'Status':<10} {'Created':<19} {'Config':<30}\n")
        parts.append("-" * 80 + "\n")
        for row in rows:
            parts.append(f"{row[0]:<20} {row[1]:<10} {row[2]:<19} {row[3]:<30}\n")
        table = "".join(parts)
        return table


# Global UIManager instance
ui_manager = UIManager()
