"""
Chaos Engine for Adversarial Stress Testing

Provides configurable chaos injection for testing system resilience
under extreme market conditions and operational stress.
"""

import asyncio
import logging
import random
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

from src.core.events import EventBus
from src.operational.state_store import StateStore

logger = logging.getLogger(__name__)


@dataclass
class ChaosConfig:
    """Configuration for chaos testing."""
    enabled: bool = False
    slippage_factor: float = 0.001  # 0.1% slippage
    rejection_rate: float = 0.05    # 5% order rejection
    latency_spike_rate: float = 0.1  # 10% latency spikes
    latency_spike_duration: float = 5.0  # seconds
    black_swan_probability: float = 0.001  # 0.1% per minute
    black_swan_magnitude: float = 0.05  # 5% price shock
    memory_pressure_rate: float = 0.02  # 2% memory pressure
    connection_drop_rate: float = 0.01  # 1% connection drops


class ChaosEngine:
    """
    Engine for injecting controlled chaos into the system.
    
    Features:
    - Configurable chaos parameters
    - Real-time chaos injection
    - Event logging and metrics
    - Safe mode with automatic recovery
    """
    
    def __init__(self, event_bus: EventBus, state_store: StateStore):
        self.event_bus = event_bus
        self.state_store = state_store
        self.config = ChaosConfig()
        self.active = False
        self.chaos_events: List[Dict[str, Any]] = []
        self.start_time: Optional[datetime] = None
        self._chaos_task: Optional[asyncio.Task] = None
        
    async def initialize(self) -> None:
        """Initialize chaos engine."""
        await self.load_config()
        logger.info("Chaos engine initialized")
        
    async def start(self) -> None:
        """Start chaos injection."""
        if self.active:
            logger.warning("Chaos engine already active")
            return
            
        self.active = True
        self.start_time = datetime.utcnow()
        self._chaos_task = asyncio.create_task(self._chaos_loop())
        
        await self.event_bus.publish({
            'type': 'ChaosStarted',
            'data': {
                'timestamp': datetime.utcnow().isoformat(),
                'config': self.config.__dict__
            }
        })
        
        logger.info("Chaos engine started")
        
    async def stop(self) -> None:
        """Stop chaos injection."""
        if not self.active:
            logger.warning("Chaos engine not active")
            return
            
        self.active = False
        if self._chaos_task:
            self._chaos_task.cancel()
            try:
                await self._chaos_task
            except asyncio.CancelledError:
                pass
                
        await self.event_bus.publish({
            'type': 'ChaosStopped',
            'data': {
                'timestamp': datetime.utcnow().isoformat(),
                'total_events': len(self.chaos_events),
                'duration': str(datetime.utcnow() - self.start_time) if self.start_time else None
            }
        })
        
        logger.info("Chaos engine stopped")
        
    async def _chaos_loop(self) -> None:
        """Main chaos injection loop."""
        while self.active:
            try:
                await self._inject_chaos()
                await asyncio.sleep(1)  # Check every second
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in chaos loop: {e}")
                await asyncio.sleep(5)
                
    async def _inject_chaos(self) -> None:
        """Inject various types of chaos."""
        if not self.active:
            return
            
        # Price manipulation (black swan events)
        if random.random() < self.config.black_swan_probability / 60:  # Per second
            await self._inject_black_swan()
            
        # Order execution chaos
        if random.random() < self.config.rejection_rate / 60:
            await self._inject_order_rejection()
            
        # Network latency chaos
        if random.random() < self.config.latency_spike_rate / 60:
            await self._inject_latency_spike()
            
        # Memory pressure
        if random.random() < self.config.memory_pressure_rate / 60:
            await self._inject_memory_pressure()
            
        # Connection drops
        if random.random() < self.config.connection_drop_rate / 60:
            await self._inject_connection_drop()
            
    async def _inject_black_swan(self) -> None:
        """Inject a black swan price event."""
        direction = random.choice(['up', 'down'])
        magnitude = random.uniform(
            self.config.black_swan_magnitude * 0.5,
            self.config.black_swan_magnitude * 2.0
        )
        
        event = {
            'type': 'BlackSwanEvent',
            'timestamp': datetime.utcnow().isoformat(),
            'direction': direction,
            'magnitude': magnitude,
            'description': f"Black swan event: {direction} {magnitude*100:.1f}%"
        }
        
        self.chaos_events.append(event)
        
        await self.event_bus.publish({
            'type': 'ChaosEvent',
            'data': event
        })
        
        logger.warning(f"Black swan injected: {direction} {magnitude*100:.1f}%")
        
    async def _inject_order_rejection(self) -> None:
        """Inject order rejection."""
        reason = random.choice([
            'insufficient_margin',
            'market_closed',
            'price_too_far',
            'volume_exceeded',
            'system_error'
        ])
        
        event = {
            'type': 'OrderRejection',
            'timestamp': datetime.utcnow().isoformat(),
            'reason': reason,
            'description': f"Order rejected: {reason}"
        }
        
        self.chaos_events.append(event)
        
        await self.event_bus.publish({
            'type': 'ChaosEvent',
            'data': event
        })
        
        logger.warning(f"Order rejection injected: {reason}")
        
    async def _inject_latency_spike(self) -> None:
        """Inject network latency spike."""
        duration = random.uniform(
            self.config.latency_spike_duration * 0.5,
            self.config.latency_spike_duration * 2.0
        )
        
        event = {
            'type': 'LatencySpike',
            'timestamp': datetime.utcnow().isoformat(),
            'duration': duration,
            'description': f"Latency spike: {duration:.1f}s"
        }
        
        self.chaos_events.append(event)
        
        await self.event_bus.publish({
            'type': 'ChaosEvent',
            'data': event
        })
        
        logger.warning(f"Latency spike injected: {duration:.1f}s")
        
        # Simulate the latency
        await asyncio.sleep(duration)
        
    async def _inject_memory_pressure(self) -> None:
        """Inject memory pressure."""
        pressure_level = random.uniform(0.1, 0.5)
        
        event = {
            'type': 'MemoryPressure',
            'timestamp': datetime.utcnow().isoformat(),
            'pressure_level': pressure_level,
            'description': f"Memory pressure: {pressure_level*100:.1f}%"
        }
        
        self.chaos_events.append(event)
        
        await self.event_bus.publish({
            'type': 'ChaosEvent',
            'data': event
        })
        
        logger.warning(f"Memory pressure injected: {pressure_level*100:.1f}%")
        
    async def _inject_connection_drop(self) -> None:
        """Inject connection drop."""
        service = random.choice(['broker', 'redis', 'database', 'market_data'])
        
        event = {
            'type': 'ConnectionDrop',
            'timestamp': datetime.utcnow().isoformat(),
            'service': service,
            'description': f"Connection drop: {service}"
        }
        
        self.chaos_events.append(event)
        
        await self.event_bus.publish({
            'type': 'ChaosEvent',
            'data': event
        })
        
        logger.warning(f"Connection drop injected: {service}")
        
    def apply_slippage(self, price: float, is_buy: bool) -> float:
        """Apply slippage to price."""
        if not self.active:
            return price
            
        slippage = random.uniform(
            -self.config.slippage_factor,
            self.config.slippage_factor
        )
        
        # Bias slippage against the trade direction
        if is_buy:
            slippage = abs(slippage)  # Worse fill for buys
        else:
            slippage = -abs(slippage)  # Worse fill for sells
            
        return price * (1 + slippage)
        
    def should_reject_order(self) -> bool:
        """Determine if order should be rejected."""
        return self.active and random.random() < self.config.rejection_rate
        
    async def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update chaos configuration."""
        for key, value in new_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                
        await self.save_config()
        
        await self.event_bus.publish({
            'type': 'ChaosConfigUpdated',
            'data': {
                'timestamp': datetime.utcnow().isoformat(),
                'config': self.config.__dict__
            }
        })
        
        logger.info("Chaos configuration updated")
        
    async def load_config(self) -> None:
        """Load configuration from storage."""
        try:
            data = await self.state_store.get("emp:chaos_config")
            if data:
                config_data = json.loads(data)
                for key, value in config_data.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
        except Exception as e:
            logger.error(f"Failed to load chaos config: {e}")
            
    async def save_config(self) -> None:
        """Save configuration to storage."""
        try:
            await self.state_store.set(
                "emp:chaos_config",
                json.dumps(self.config.__dict__),
                expire=3600
            )
        except Exception as e:
            logger.error(f"Failed to save chaos config: {e}")
            
    def get_stats(self) -> Dict[str, Any]:
        """Get chaos injection statistics."""
        if not self.start_time:
            return {
                'active': False,
                'total_events': 0,
                'duration': None,
                'events_by_type': {}
            }
            
        duration = datetime.utcnow() - self.start_time
        
        events_by_type = {}
        for event in self.chaos_events:
            event_type = event['type']
            if event_type not in events_by_type:
                events_by_type[event_type] = 0
            events_by_type[event_type] += 1
            
        return {
            'active': self.active,
            'total_events': len(self.chaos_events),
            'duration': str(duration),
            'events_by_type': events_by_type,
            'config': self.config.__dict__
        }
        
    async def reset(self) -> None:
        """Reset chaos engine state."""
        self.chaos_events.clear()
        self.start_time = None
        
        await self.event_bus.publish({
            'type': 'ChaosReset',
            'data': {
                'timestamp': datetime.utcnow().isoformat()
            }
        })
        
        logger.info("Chaos engine reset")


# Global instance
_chaos_engine: Optional[ChaosEngine] = None


async def get_chaos_engine(event_bus: EventBus, state_store: StateStore) -> ChaosEngine:
    """Get or create global chaos engine instance."""
    global _chaos_engine
    if _chaos_engine is None:
        _chaos_engine = ChaosEngine(event_bus, state_store)
        await _chaos_engine.initialize()
    return _chaos_engine
