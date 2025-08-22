"""
Liquidity Prober Engine
Implements PROBE-40.1: The LiquidityProber Engine for active liquidity probing

This module provides the ability to actively probe the order book for hidden
liquidity and icebergs by sending rapid-fire "ping" orders to test liquidity.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Literal, Optional

from src.trading.integration.fix_broker_interface import FIXBrokerInterface

logger = logging.getLogger(__name__)


class LiquidityProber:
    """
    Core component that can send rapid-fire "ping" orders to test liquidity.
    
    This class implements the predator's sonar capability by actively probing
    the order book using IOC (Immediate Or Cancel) orders to detect hidden
    liquidity and iceberg orders.
    """
    
    def __init__(self, broker_interface: FIXBrokerInterface, config: Optional[dict[str, object]] = None):
        """
        Initialize the LiquidityProber with broker interface and configuration.
        
        Args:
            broker_interface: Interface to the trading broker (cTrader)
            config: Configuration dictionary for probe settings
        """
        self.broker = broker_interface
        self.config = config or {}
        
        # Probe configuration
        self.probe_size = Decimal(str(self.config.get('probe_size', 0.001)))  # 0.001 lots default
        self.timeout_seconds = self.config.get('timeout_seconds', 2.0)
        self.max_concurrent_probes = self.config.get('max_concurrent_probes', 5)
        
        # Tracking
        self.active_probes: Dict[str, asyncio.Task] = {}
        self.probe_results: Dict[str, Dict] = {}
        
        logger.info(
            f"LiquidityProber initialized: probe_size={self.probe_size}, "
            f"timeout={self.timeout_seconds}s, max_concurrent={self.max_concurrent_probes}"
        )
    
    async def probe_liquidity(
        self,
        symbol: str,
        price_levels: List[float],
        side: Literal["buy", "sell"]
    ) -> Dict[float, float]:
        """
        Probe liquidity at specified price levels using IOC orders.
        
        This method sends small IOC orders at each price level to test actual
        available liquidity, helping detect hidden liquidity and iceberg orders.
        
        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            price_levels: List of price levels to probe
            side: "buy" or "sell" indicating probe direction
            
        Returns:
            Dictionary mapping each probed price level to the volume of
            liquidity that was found there (filled quantity)
        """
        logger.info(
            f"Starting liquidity probe for {symbol} - {side} side "
            f"at {len(price_levels)} price levels"
        )
        
        results = {}
        semaphore = asyncio.Semaphore(self.max_concurrent_probes)
        
        # Create probe tasks
        probe_tasks = []
        for price in price_levels:
            task = self._single_probe(semaphore, symbol, price, side)
            probe_tasks.append((price, task))
        
        # Execute probes with timeout
        try:
            completed_probes = await asyncio.wait_for(
                self._execute_probes(probe_tasks),
                timeout=self.timeout_seconds * len(price_levels)
            )
            
            # Collect results
            for price, filled_volume in completed_probes:
                results[price] = filled_volume
                
        except asyncio.TimeoutError:
            logger.warning("Liquidity probe timed out, returning partial results")
            # Return whatever results we have
            for price, task in probe_tasks:
                if task.done() and not task.exception():
                    results[price] = task.result()
        
        logger.info(
            f"Completed liquidity probe for {symbol} - "
            f"found liquidity at {len(results)} levels"
        )
        
        return results
    
    async def _execute_probes(self, probe_tasks: List[tuple]) -> List[tuple]:
        """Execute all probe tasks and collect results."""
        results = []
        
        # Start all tasks
        for price, task in probe_tasks:
            try:
                filled_volume = await task
                results.append((price, filled_volume))
            except Exception as e:
                logger.error(f"Probe failed at price {price}: {e}")
                results.append((price, 0.0))
        
        return results
    
    async def _single_probe(
        self,
        semaphore: asyncio.Semaphore,
        symbol: str,
        price: float,
        side: Literal["buy", "sell"]
    ) -> float:
        """
        Execute a single liquidity probe at a specific price level.
        
        Args:
            semaphore: Concurrency control semaphore
            symbol: Trading symbol
            price: Price level to probe
            side: "buy" or "sell"
            
        Returns:
            Filled volume at this price level
        """
        async with semaphore:
            try:
                # Create unique probe ID
                probe_id = f"probe_{uuid.uuid4().hex[:8]}"
                
                # Determine order side based on probe direction
                # For FIXBrokerInterface, issue a small market order as probe
                order_id = await self.broker.place_market_order(
                    symbol=symbol,
                    side="BUY" if side == "buy" else "SELL",
                    quantity=float(self.probe_size)
                )
                
                if not order_id:
                    logger.warning(f"Failed to place probe order for {symbol} at {price}")
                    return 0.0
                
                # Wait for execution report with timeout
                filled_volume = await self._wait_for_execution(order_id, timeout=self.timeout_seconds)
                
                # Log probe result
                logger.debug(
                    f"Probe {probe_id}: {symbol} {side} @ {price} - "
                    f"filled {filled_volume}/{self.probe_size}"
                )
                
                return filled_volume
                
            except Exception as e:
                logger.error(f"Error in single probe for {symbol} at {price}: {e}")
                return 0.0
    
    async def _wait_for_execution(self, order_id: str, timeout: float) -> float:
        """
        Wait for order execution and return filled volume.
        
        Args:
            order_id: Order ID to monitor
            timeout: Maximum wait time in seconds
            
        Returns:
            Filled volume for the order
        """
        start_time = datetime.now()
        
        while (datetime.now() - start_time).total_seconds() < timeout:
            # Check order status
            # Read in-memory order state from FIXBrokerInterface
            status = self.broker.get_order_status(order_id)
            if status and status.get("status") in ("FILLED", "PARTIALLY_FILLED"):
                return float(status.get("filled_qty") or 0.0)
            
            # Small delay to prevent busy waiting
            await asyncio.sleep(0.1)
        
        # Timeout reached
        logger.debug(f"Order {order_id} execution timeout after {timeout}s")
        return 0.0
    
    def calculate_liquidity_confidence_score(
        self,
        probe_results: Dict[float, float],
        intended_volume: float
    ) -> float:
        """
        Calculate a liquidity confidence score based on probe results.
        
        Args:
            probe_results: Results from liquidity probing
            intended_volume: Volume of the intended trade
            
        Returns:
            Confidence score between 0.0 (no confidence) and 1.0 (high confidence)
        """
        if not probe_results or intended_volume <= 0:
            return 0.0
        
        # Calculate total available liquidity
        total_liquidity = sum(probe_results.values())
        
        # Calculate liquidity ratio
        liquidity_ratio = min(1.0, total_liquidity / float(intended_volume))
        
        # Calculate distribution quality (how evenly liquidity is distributed)
        if len(probe_results) > 1:
            avg_liquidity = total_liquidity / len(probe_results)
            variance = sum((vol - avg_liquidity) ** 2 for vol in probe_results.values()) / len(probe_results)
            distribution_quality = 1.0 / (1.0 + variance)  # Higher is better
        else:
            distribution_quality = 0.5
        
        # Calculate confidence score
        confidence_score = liquidity_ratio * (0.7 + 0.3 * distribution_quality)
        
        # Ensure score is between 0.0 and 1.0
        confidence_score = max(0.0, min(1.0, confidence_score))
        
        logger.debug(
            f"Liquidity confidence score: {confidence_score:.3f} "
            f"(ratio={liquidity_ratio:.3f}, dist={distribution_quality:.3f})"
        )
        
        return confidence_score
    
    def get_probe_summary(self, probe_results: Dict[float, float]) -> dict[str, object]:
        """
        Generate a summary of probe results for logging/monitoring.
        
        Args:
            probe_results: Results from liquidity probing
            
        Returns:
            Summary dictionary with key metrics
        """
        if not probe_results:
            return {"total_levels": 0, "total_liquidity": 0.0, "avg_liquidity": 0.0}
        
        total_liquidity = sum(probe_results.values())
        avg_liquidity = total_liquidity / len(probe_results)
        
        # Find best liquidity levels
        sorted_levels = sorted(probe_results.items(), key=lambda x: x[1], reverse=True)
        best_levels = sorted_levels[:3]
        
        return {
            "total_levels": len(probe_results),
            "total_liquidity": total_liquidity,
            "avg_liquidity": avg_liquidity,
            "best_levels": best_levels,
            "empty_levels": len([v for v in probe_results.values() if v == 0])
        }
    
    async def health_check(self) -> dict[str, object]:
        """
        Perform a health check on the LiquidityProber.
        
        Returns:
            Health status dictionary
        """
        return {
            "status": "healthy",
            "active_probes": len(self.active_probes),
            "config": {
                "probe_size": float(self.probe_size),
                "timeout_seconds": self.timeout_seconds,
                "max_concurrent_probes": self.max_concurrent_probes
            },
            "timestamp": datetime.now().isoformat()
        }
