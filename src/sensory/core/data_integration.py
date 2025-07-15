from typing import List
from datetime import datetime, timedelta
import numpy as np
from .base import EconomicData
from dataclasses import dataclass, field

@dataclass
class OrderBookSnapshot:
    timestamp: datetime
    bids: List[float] = field(default_factory=list)  # Sorted descending
    asks: List[float] = field(default_factory=list)  # Sorted ascending
    bid_volumes: List[float] = field(default_factory=list)
    ask_volumes: List[float] = field(default_factory=list)

    def best_bid(self) -> float:
        return self.bids[0] if self.bids else 0.0

    def best_ask(self) -> float:
        return self.asks[0] if self.asks else 0.0

    def spread(self) -> float:
        if self.bids and self.asks:
            return self.asks[0] - self.bids[0]
        return 0.0

    def depth(self, side: str = 'both') -> float:
        if side == 'bid':
            return sum(self.bid_volumes)
        elif side == 'ask':
            return sum(self.ask_volumes)
        else:
            return sum(self.bid_volumes) + sum(self.ask_volumes)

class OrderFlowDataProvider:
    """
    Simulates or provides real order book data. For demonstration, generates synthetic data.
    """
    def __init__(self, levels: int = 10):
        self.levels = levels
        self.last_snapshot: OrderBookSnapshot = self._generate_snapshot()

    def _generate_snapshot(self) -> OrderBookSnapshot:
        timestamp = datetime.now()
        mid = 1.1000 + np.random.normal(0, 0.001)
        spread = 0.0002 + abs(np.random.normal(0, 0.00005))
        bids = [mid - spread/2 - i*0.0001 for i in range(self.levels)]
        asks = [mid + spread/2 + i*0.0001 for i in range(self.levels)]
        bid_volumes = np.abs(np.random.normal(5, 2, self.levels)).tolist()
        ask_volumes = np.abs(np.random.normal(5, 2, self.levels)).tolist()
        return OrderBookSnapshot(
            timestamp=timestamp,
            bids=bids,
            asks=asks,
            bid_volumes=bid_volumes,
            ask_volumes=ask_volumes
        )

    def get_latest_snapshot(self) -> OrderBookSnapshot:
        # In production, fetch from live feed; here, simulate
        self.last_snapshot = self._generate_snapshot()
        return self.last_snapshot

async def _get_simulated_ecb_rates(self, days_back: int) -> List[EconomicData]:
    """Generate simulated ECB rates for demonstration purposes"""
    
    data_points = []
    base_rate = 4.50  # Current ECB deposit facility rate
    
    for i in range(min(days_back // 30, 12)):  # Monthly data points
        date = datetime.now() - timedelta(days=i * 30)
        
        # Simulate rate changes with some randomness
        rate_change = np.random.normal(0, 0.1)  # Small random changes
        simulated_rate = max(0, base_rate + rate_change)
        
        data_points.append(EconomicData(
            indicator='ECB_DEPOSIT_RATE',
            value=simulated_rate,
            timestamp=date,
            frequency='monthly',
            surprise_factor=0.0,
            importance=0.9  # High importance for central bank rates
        ))
    
    return sorted(data_points, key=lambda x: x.timestamp)

