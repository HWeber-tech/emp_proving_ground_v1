"""
AdversarialEngine: Intelligent market manipulation and adversarial events.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import random

logger = logging.getLogger(__name__)

class AdversarialEventType(Enum):
    SPOOFING = "spoofing"
    STOP_HUNT = "stop_hunt"
    NEWS_SHOCK = "news_shock"
    FLASH_CRASH = "flash_crash"
    LIQUIDITY_CRUNCH = "liquidity_crunch"

@dataclass
class AdversarialEvent:
    """Represents an adversarial market event"""
    event_type: AdversarialEventType
    timestamp: datetime
    duration: timedelta
    intensity: float  # 0.0 to 1.0
    parameters: Dict[str, Any] = field(default_factory=dict)
    active: bool = False

@dataclass
class LiquidityZone:
    """Represents a liquidity zone for stop hunting"""
    price_level: float
    zone_type: str  # "support", "resistance", "swing_high", "swing_low"
    confluence_score: float
    timestamp: datetime
    volume_profile: Dict[str, float] = field(default_factory=dict)

class AdversarialEngine:
    """
    Intelligent adversarial engine that creates realistic market manipulation.
    """
    
    def __init__(self, difficulty_level: float = 0.5, seed: Optional[int] = None):
        self.difficulty_level = max(0.0, min(1.0, difficulty_level))
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Configuration based on difficulty
        self.config = self._create_config(self.difficulty_level)
        
        # Active events and history
        self.active_events: List[AdversarialEvent] = []
        self.event_history: List[AdversarialEvent] = []
        
        # Liquidity zones for stop hunting
        self.liquidity_zones: List[LiquidityZone] = []
        self.consolidation_periods: List[Dict] = []
        
        # Market state tracking
        self.last_market_state: Optional[Any] = None
        self.price_history: List[float] = []
        self.volume_history: List[float] = []
        
        # Adversarial callbacks
        self.adversarial_callbacks: List[Callable] = []
        
        logger.info(f"Initialized AdversarialEngine with difficulty {self.difficulty_level}")
    
    def _create_config(self, difficulty: float) -> Dict:
        """Create configuration based on difficulty level."""
        base_config = {
            'stop_hunt_probability': 0.1,
            'spoofing_probability': 0.05,
            'news_shock_probability': 0.02,
            'flash_crash_probability': 0.01,
            'liquidity_crunch_probability': 0.03,
            
            'max_concurrent_events': 2,
            'min_event_duration_minutes': 5,
            'max_event_duration_minutes': 60,
            
            'liquidity_zone_memory_hours': 24,
            'consolidation_detection_window': 100,
            'breakout_threshold': 0.002,  # 2% price movement
        }
        
        # Scale probabilities with difficulty
        for key in base_config:
            if 'probability' in key:
                base_config[key] *= difficulty
        
        return base_config
    
    def _update_liquidity_zones(self, market_state, simulator):
        """Update liquidity zones based on recent price action."""
        if not hasattr(simulator, 'current_data') or simulator.current_data is None:
            return
        
        # Get recent OHLCV data
        recent_data = simulator.current_data.tail(200)  # Last 200 ticks
        
        if len(recent_data) < 50:
            return
        
        # Convert to M15 bars for swing point detection
        ohlcv = self._ticks_to_ohlcv(recent_data, '15T')
        
        if len(ohlcv) < 20:
            return
        
        # Find swing highs and lows
        swing_highs = self._find_swing_points(ohlcv, 'high', window=5)
        swing_lows = self._find_swing_points(ohlcv, 'low', window=5)
        
        # Create liquidity zones from swing points
        new_zones = []
        
        for idx in swing_highs:
            price_level = ohlcv.iloc[idx]['high']
            confluence = self._calculate_liquidity_confluence(ohlcv, idx, price_level, 'resistance')
            
            zone = LiquidityZone(
                price_level=price_level,
                zone_type='resistance',
                confluence_score=confluence,
                timestamp=ohlcv.index[idx]
            )
            new_zones.append(zone)
        
        for idx in swing_lows:
            price_level = ohlcv.iloc[idx]['low']
            confluence = self._calculate_liquidity_confluence(ohlcv, idx, price_level, 'support')
            
            zone = LiquidityZone(
                price_level=price_level,
                zone_type='support',
                confluence_score=confluence,
                timestamp=ohlcv.index[idx]
            )
            new_zones.append(zone)
        
        # Merge with existing zones
        self.liquidity_zones.extend(new_zones)
        
        # Remove old zones
        cutoff_time = market_state.timestamp - timedelta(hours=self.config['liquidity_zone_memory_hours'])
        self.liquidity_zones = [
            zone for zone in self.liquidity_zones 
            if zone.timestamp > cutoff_time
        ]
        
        # Merge nearby zones
        self._merge_nearby_zones()
    
    def _find_swing_points(self, df: pd.DataFrame, column: str, window: int = 5) -> List[int]:
        """Find swing highs or lows in the data."""
        peaks = []
        
        for i in range(window, len(df) - window):
            if column == 'high':
                if all(df.iloc[i][column] >= df.iloc[j][column] for j in range(i-window, i+window+1)):
                    peaks.append(i)
            else:  # low
                if all(df.iloc[i][column] <= df.iloc[j][column] for j in range(i-window, i+window+1)):
                    peaks.append(i)
        
        return peaks
    
    def _calculate_liquidity_confluence(self, ohlcv: pd.DataFrame, idx: int, 
                                      price_level: float, zone_type: str) -> float:
        """Calculate confluence score for a liquidity zone."""
        confluence = 0.0
        
        # Volume confluence
        if idx < len(ohlcv):
            volume = ohlcv.iloc[idx]['volume']
            avg_volume = ohlcv['volume'].mean()
            volume_factor = min(volume / avg_volume, 3.0) / 3.0
            confluence += volume_factor * 0.3
        
        # Price level confluence (how many times price touched this level)
        tolerance = price_level * 0.0001  # 1 pip tolerance
        
        touches = 0
        for i in range(max(0, idx-50), min(len(ohlcv), idx+50)):
            if zone_type == 'resistance':
                if abs(ohlcv.iloc[i]['high'] - price_level) <= tolerance:
                    touches += 1
            else:  # support
                if abs(ohlcv.iloc[i]['low'] - price_level) <= tolerance:
                    touches += 1
        
        touch_factor = min(touches / 10.0, 1.0)
        confluence += touch_factor * 0.4
        
        # Recency factor (more recent = higher confluence)
        recency_factor = 1.0 - (len(ohlcv) - idx) / len(ohlcv)
        confluence += recency_factor * 0.3
        
        return min(confluence, 1.0)
    
    def _merge_nearby_zones(self):
        """Merge liquidity zones that are close to each other."""
        if len(self.liquidity_zones) < 2:
            return
        
        merged_zones = []
        used_indices = set()
        
        for i, zone1 in enumerate(self.liquidity_zones):
            if i in used_indices:
                continue
            
            merged_zone = zone1
            used_indices.add(i)
            
            for j, zone2 in enumerate(self.liquidity_zones[i+1:], i+1):
                if j in used_indices:
                    continue
                
                # Check if zones are close
                price_diff = abs(zone1.price_level - zone2.price_level)
                if price_diff <= zone1.price_level * 0.0002:  # 2 pips tolerance
                    # Merge zones
                    merged_zone.price_level = (zone1.price_level + zone2.price_level) / 2
                    merged_zone.confluence_score = max(zone1.confluence_score, zone2.confluence_score)
                    used_indices.add(j)
            
            merged_zones.append(merged_zone)
        
        self.liquidity_zones = merged_zones
    
    def _update_consolidation_detection(self, market_state, simulator):
        """Detect price consolidation periods for breakout traps."""
        if not hasattr(simulator, 'current_data') or simulator.current_data is None:
            return
        
        # Get recent price data
        recent_data = simulator.current_data.tail(self.config['consolidation_detection_window'])
        
        if len(recent_data) < 50:
            return
        
        # Calculate price range and volatility
        prices = recent_data['mid_price'].values
        price_range = (np.max(prices) - np.min(prices)) / np.mean(prices)
        
        # Calculate ATR
        high_low = recent_data['ask'] - recent_data['bid']
        atr = np.mean(high_low)
        atr_ratio = atr / np.mean(prices)
        
        # Detect consolidation
        is_consolidating = (price_range < 0.005 and atr_ratio < 0.0002)  # 0.5% range, low ATR
        
        if is_consolidating:
            # Check if we're near a boundary
            current_price = market_state.mid_price
            price_min = np.min(prices)
            price_max = np.max(prices)
            
            # Calculate distance to boundaries
            distance_to_low = (current_price - price_min) / (price_max - price_min)
            distance_to_high = (price_max - current_price) / (price_max - price_min)
            
            # Add to consolidation periods
            consolidation = {
                'start_time': recent_data.iloc[0]['timestamp'],
                'end_time': recent_data.iloc[-1]['timestamp'],
                'price_min': price_min,
                'price_max': price_max,
                'current_price': current_price,
                'distance_to_low': distance_to_low,
                'distance_to_high': distance_to_high,
                'atr': atr,
                'price_range': price_range
            }
            
            self.consolidation_periods.append(consolidation)
            
            # Keep only recent consolidations
            cutoff_time = market_state.timestamp - timedelta(hours=4)
            self.consolidation_periods = [
                c for c in self.consolidation_periods 
                if c['end_time'] > cutoff_time
            ]
    
    def apply_adversarial_effects(self, market_state, simulator):
        """Apply adversarial effects to the market state."""
        # Update tracking
        self.last_market_state = market_state
        self.price_history.append(market_state.mid_price)
        self.volume_history.append(market_state.bid_volume + market_state.ask_volume)
        
        # Keep history manageable
        if len(self.price_history) > 1000:
            self.price_history = self.price_history[-500:]
            self.volume_history = self.volume_history[-500:]
        
        # Update liquidity zones and consolidation detection
        self._update_liquidity_zones(market_state, simulator)
        self._update_consolidation_detection(market_state, simulator)
        
        # Update active events
        self._update_active_events(market_state.timestamp)
        
        # Check for new manipulation triggers
        self._check_manipulation_triggers(market_state, simulator)
        
        # Apply active effects
        self._apply_active_effects(market_state, simulator)
    
    def _update_active_events(self, current_time: datetime):
        """Update active events and remove expired ones."""
        # Remove expired events
        self.active_events = [
            event for event in self.active_events
            if current_time < event.timestamp + event.duration
        ]
        
        # Deactivate events that have ended
        for event in self.active_events:
            if current_time >= event.timestamp + event.duration:
                event.active = False
                self.event_history.append(event)
    
    def _check_manipulation_triggers(self, market_state, simulator):
        """Check if conditions are right for new adversarial events."""
        if len(self.active_events) >= self.config['max_concurrent_events']:
            return
        
        # Check for breakout trap
        if self._should_trigger_breakout_trap(market_state, simulator):
            self._trigger_breakout_trap(market_state, simulator)
        
        # Check for intelligent stop hunt
        elif self._should_trigger_intelligent_stop_hunt(market_state, simulator):
            self._trigger_intelligent_stop_hunt(market_state, simulator)
        
        # Check for spoofing
        elif random.random() < self.config['spoofing_probability']:
            self._trigger_spoofing(market_state)
        
        # Check for stop hunt
        elif self._should_trigger_stop_hunt(market_state, simulator):
            self._trigger_stop_hunt(market_state, simulator)
        
        # Check for news shock
        elif random.random() < self.config['news_shock_probability']:
            self._trigger_news_shock(market_state)
        
        # Check for flash crash
        elif random.random() < self.config['flash_crash_probability']:
            self._trigger_flash_crash(market_state)
    
    def _should_trigger_breakout_trap(self, market_state, simulator) -> bool:
        """Check if conditions are right for a breakout trap."""
        if not self.consolidation_periods:
            return False
        
        # Get most recent consolidation
        consolidation = self.consolidation_periods[-1]
        
        # Check if we're near a boundary
        current_price = market_state.mid_price
        price_min = consolidation['price_min']
        price_max = consolidation['price_max']
        
        # Calculate distance to boundaries
        distance_to_low = (current_price - price_min) / (price_max - price_min)
        distance_to_high = (price_max - current_price) / (price_max - price_min)
        
        # Trigger if very close to boundary
        if distance_to_low < 0.1 or distance_to_high < 0.1:
            # Check for increased volume (fake breakout)
            if len(self.volume_history) >= 10:
                recent_volume = np.mean(self.volume_history[-5:])
                avg_volume = np.mean(self.volume_history[-20:])
                
                if recent_volume > avg_volume * 1.5:  # 50% volume increase
                    return True
        
        return False
    
    def _trigger_breakout_trap(self, market_state, simulator):
        """Trigger a breakout trap event."""
        consolidation = self.consolidation_periods[-1]
        
        # Determine trap direction
        current_price = market_state.mid_price
        price_min = consolidation['price_min']
        price_max = consolidation['price_max']
        
        distance_to_low = (current_price - price_min) / (price_max - price_min)
        distance_to_high = (price_max - current_price) / (price_max - price_min)
        
        if distance_to_low < distance_to_high:
            # Trap to the downside
            trap_direction = "down"
            target_price = price_min - (price_max - price_min) * 0.5
        else:
            # Trap to the upside
            trap_direction = "up"
            target_price = price_max + (price_max - price_min) * 0.5
        
        # Create event
        event = AdversarialEvent(
            event_type=AdversarialEventType.SPOOFING,
            timestamp=market_state.timestamp,
            duration=timedelta(minutes=random.randint(10, 30)),
            intensity=random.uniform(0.6, 0.9),
            parameters={
                'trap_direction': trap_direction,
                'target_price': target_price,
                'consolidation_range': (price_min, price_max),
                'fake_breakout': True
            },
            active=True
        )
        
        self.active_events.append(event)
        logger.info(f"Triggered breakout trap: {trap_direction} to {target_price:.5f}")
    
    def _should_trigger_intelligent_stop_hunt(self, market_state, simulator) -> bool:
        """Check if conditions are right for intelligent stop hunt."""
        if not self.liquidity_zones:
            return False
        
        # Find the highest confluence zone
        best_zone = max(self.liquidity_zones, key=lambda z: z.confluence_score)
        
        # Check if price is approaching the zone
        current_price = market_state.mid_price
        distance_to_zone = abs(current_price - best_zone.price_level) / current_price
        
        if distance_to_zone < 0.001:  # Within 1 pip
            # Check if confluence is high enough
            if best_zone.confluence_score > 0.7:
                # Check for trend direction
                if len(self.price_history) >= 20:
                    recent_trend = np.polyfit(range(20), self.price_history[-20:], 1)[0]
                    
                    # Hunt against the trend has higher probability
                    if best_zone.zone_type == 'resistance' and recent_trend > 0:
                        return random.random() < 0.8  # 80% chance
                    elif best_zone.zone_type == 'support' and recent_trend < 0:
                        return random.random() < 0.8  # 80% chance
                    else:
                        return random.random() < 0.4  # 40% chance
        
        return False
    
    def _trigger_intelligent_stop_hunt(self, market_state, simulator):
        """Trigger an intelligent stop hunt."""
        # Find the best target zone
        best_zone = max(self.liquidity_zones, key=lambda z: z.confluence_score)
        
        # Calculate hunt parameters
        current_price = market_state.mid_price
        hunt_distance = abs(current_price - best_zone.price_level)
        
        # Determine hunt direction and reversal probability
        if best_zone.zone_type == 'resistance':
            hunt_direction = 'up'
            reversal_prob = 0.8 if current_price < best_zone.price_level else 0.4
        else:  # support
            hunt_direction = 'down'
            reversal_prob = 0.8 if current_price > best_zone.price_level else 0.4
        
        # Create event
        event = AdversarialEvent(
            event_type=AdversarialEventType.STOP_HUNT,
            timestamp=market_state.timestamp,
            duration=timedelta(minutes=random.randint(5, 15)),
            intensity=random.uniform(0.7, 1.0),
            parameters={
                'target_zone': best_zone.price_level,
                'hunt_direction': hunt_direction,
                'reversal_probability': reversal_prob,
                'confluence_score': best_zone.confluence_score,
                'zone_type': best_zone.zone_type
            },
            active=True
        )
        
        self.active_events.append(event)
        logger.info(f"Triggered intelligent stop hunt: {hunt_direction} to {best_zone.price_level:.5f}")
    
    def _trigger_spoofing(self, market_state):
        """Trigger a spoofing event."""
        # Random spoofing direction
        direction = random.choice(['buy', 'sell'])
        
        # Create event
        event = AdversarialEvent(
            event_type=AdversarialEventType.SPOOFING,
            timestamp=market_state.timestamp,
            duration=timedelta(minutes=random.randint(5, 20)),
            intensity=random.uniform(0.5, 0.8),
            parameters={
                'direction': direction,
                'fake_orders': True,
                'order_size_multiplier': random.uniform(2.0, 5.0)
            },
            active=True
        )
        
        self.active_events.append(event)
        logger.info(f"Triggered spoofing: {direction} orders")
    
    def _should_trigger_stop_hunt(self, market_state, simulator) -> bool:
        """Check if conditions are right for a basic stop hunt."""
        if not self.liquidity_zones:
            return False
        
        # Simple probability-based trigger
        return random.random() < self.config['stop_hunt_probability']
    
    def _trigger_stop_hunt(self, market_state, simulator):
        """Trigger a basic stop hunt."""
        # Find a random liquidity zone
        if self.liquidity_zones:
            target_zone = random.choice(self.liquidity_zones)
            
            event = AdversarialEvent(
                event_type=AdversarialEventType.STOP_HUNT,
                timestamp=market_state.timestamp,
                duration=timedelta(minutes=random.randint(3, 10)),
                intensity=random.uniform(0.5, 0.8),
                parameters={
                    'target_zone': target_zone.price_level,
                    'zone_type': target_zone.zone_type
                },
                active=True
            )
            
            self.active_events.append(event)
            logger.info(f"Triggered stop hunt: {target_zone.zone_type} at {target_zone.price_level:.5f}")
    
    def _trigger_news_shock(self, market_state):
        """Trigger a news shock event."""
        # Random news shock parameters
        shock_direction = random.choice(['positive', 'negative'])
        shock_magnitude = random.uniform(0.001, 0.005)  # 1-5 pips
        
        event = AdversarialEvent(
            event_type=AdversarialEventType.NEWS_SHOCK,
            timestamp=market_state.timestamp,
            duration=timedelta(minutes=random.randint(15, 45)),
            intensity=random.uniform(0.6, 0.9),
            parameters={
                'direction': shock_direction,
                'magnitude': shock_magnitude,
                'volatility_increase': random.uniform(2.0, 5.0)
            },
            active=True
        )
        
        self.active_events.append(event)
        logger.info(f"Triggered news shock: {shock_direction} {shock_magnitude:.5f}")
    
    def _trigger_flash_crash(self, market_state):
        """Trigger a flash crash event."""
        # Flash crash parameters
        crash_direction = random.choice(['down', 'up'])
        crash_magnitude = random.uniform(0.005, 0.02)  # 5-20 pips
        
        event = AdversarialEvent(
            event_type=AdversarialEventType.FLASH_CRASH,
            timestamp=market_state.timestamp,
            duration=timedelta(minutes=random.randint(1, 5)),
            intensity=random.uniform(0.8, 1.0),
            parameters={
                'direction': crash_direction,
                'magnitude': crash_magnitude,
                'recovery_probability': random.uniform(0.3, 0.7)
            },
            active=True
        )
        
        self.active_events.append(event)
        logger.info(f"Triggered flash crash: {crash_direction} {crash_magnitude:.5f}")
    
    def _apply_active_effects(self, market_state, simulator):
        """Apply effects from active adversarial events."""
        for event in self.active_events:
            if not event.active:
                continue
            
            if event.event_type == AdversarialEventType.SPOOFING:
                self._apply_spoofing_effects(market_state, event)
            elif event.event_type == AdversarialEventType.STOP_HUNT:
                self._apply_stop_hunt_effects(market_state, event)
            elif event.event_type == AdversarialEventType.NEWS_SHOCK:
                self._apply_news_shock_effects(market_state, event)
            elif event.event_type == AdversarialEventType.FLASH_CRASH:
                self._apply_flash_crash_effects(market_state, event)
    
    def _apply_spoofing_effects(self, market_state, event: AdversarialEvent):
        """Apply spoofing effects to market state."""
        direction = event.parameters.get('direction', 'buy')
        order_size_multiplier = event.parameters.get('order_size_multiplier', 2.0)
        
        if direction == 'buy':
            # Fake buy orders - increase ask volume
            market_state.ask_volume *= order_size_multiplier
            # Slightly increase ask price
            market_state.ask += market_state.ask * 0.0001 * event.intensity
        else:
            # Fake sell orders - increase bid volume
            market_state.bid_volume *= order_size_multiplier
            # Slightly decrease bid price
            market_state.bid -= market_state.bid * 0.0001 * event.intensity
        
        # Update mid price and spread
        market_state.mid_price = (market_state.bid + market_state.ask) / 2
        market_state.spread = market_state.ask - market_state.bid
    
    def _apply_stop_hunt_effects(self, market_state, event: AdversarialEvent):
        """Apply stop hunt effects to market state."""
        target_zone = event.parameters.get('target_zone', market_state.mid_price)
        hunt_direction = event.parameters.get('hunt_direction', 'up')
        
        # Calculate distance to target
        distance = abs(market_state.mid_price - target_zone)
        
        if distance > 0.0001:  # More than 1 pip away
            # Move price towards target
            if hunt_direction == 'up':
                price_adjustment = min(distance * 0.1 * event.intensity, 0.0002)
                market_state.bid += price_adjustment
                market_state.ask += price_adjustment
            else:
                price_adjustment = min(distance * 0.1 * event.intensity, 0.0002)
                market_state.bid -= price_adjustment
                market_state.ask -= price_adjustment
            
            # Update mid price
            market_state.mid_price = (market_state.bid + market_state.ask) / 2
    
    def _apply_news_shock_effects(self, market_state, event: AdversarialEvent):
        """Apply news shock effects to market state."""
        direction = event.parameters.get('direction', 'positive')
        magnitude = event.parameters.get('magnitude', 0.002)
        volatility_increase = event.parameters.get('volatility_increase', 3.0)
        
        # Apply price movement
        if direction == 'positive':
            price_adjustment = magnitude * event.intensity
            market_state.bid += price_adjustment
            market_state.ask += price_adjustment
        else:
            price_adjustment = magnitude * event.intensity
            market_state.bid -= price_adjustment
            market_state.ask -= price_adjustment
        
        # Increase spread (volatility)
        spread_increase = market_state.spread * (volatility_increase - 1) * event.intensity
        market_state.ask += spread_increase / 2
        market_state.bid -= spread_increase / 2
        
        # Update mid price and spread
        market_state.mid_price = (market_state.bid + market_state.ask) / 2
        market_state.spread = market_state.ask - market_state.bid
    
    def _apply_flash_crash_effects(self, market_state, event: AdversarialEvent):
        """Apply flash crash effects to market state."""
        direction = event.parameters.get('direction', 'down')
        magnitude = event.parameters.get('magnitude', 0.01)
        
        # Apply sharp price movement
        if direction == 'down':
            price_adjustment = magnitude * event.intensity
            market_state.bid -= price_adjustment
            market_state.ask -= price_adjustment
        else:
            price_adjustment = magnitude * event.intensity
            market_state.bid += price_adjustment
            market_state.ask += price_adjustment
        
        # Dramatically increase spread
        spread_increase = market_state.spread * 5.0 * event.intensity
        market_state.ask += spread_increase / 2
        market_state.bid -= spread_increase / 2
        
        # Update mid price and spread
        market_state.mid_price = (market_state.bid + market_state.ask) / 2
        market_state.spread = market_state.ask - market_state.bid
    
    def _ticks_to_ohlcv(self, tick_data: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Convert tick data to OHLCV format."""
        if tick_data.empty:
            return pd.DataFrame()
        
        # Ensure timestamp is index
        tick_data = tick_data.set_index('timestamp')
        
        # Resample to OHLCV
        ohlcv = tick_data.resample(freq).agg({
            'bid': 'ohlc',
            'ask': 'ohlc',
            'bid_volume': 'sum',
            'ask_volume': 'sum'
        })
        
        if ohlcv.empty:
            return pd.DataFrame()
        
        # Flatten multi-index columns
        ohlcv.columns = [f"{col[0]}_{col[1]}" for col in ohlcv.columns]
        
        # Calculate OHLC from bid/ask
        ohlcv['open'] = (ohlcv['bid_open'] + ohlcv['ask_open']) / 2
        ohlcv['high'] = ohlcv['ask_high']
        ohlcv['low'] = ohlcv['bid_low']
        ohlcv['close'] = (ohlcv['bid_close'] + ohlcv['ask_close']) / 2
        ohlcv['volume'] = ohlcv['bid_volume'] + ohlcv['ask_volume']
        
        return ohlcv[['open', 'high', 'low', 'close', 'volume']]
    
    def get_active_events(self) -> List[AdversarialEvent]:
        """Get list of currently active events."""
        return [event for event in self.active_events if event.active]
    
    def get_event_history(self) -> List[AdversarialEvent]:
        """Get history of all events."""
        return self.event_history.copy()
    
    def add_adversarial_callback(self, callback: Callable):
        """Add a callback function for adversarial events."""
        self.adversarial_callbacks.append(callback)
    
    def get_difficulty_level(self) -> float:
        """Get current difficulty level."""
        return self.difficulty_level
    
    def set_difficulty_level(self, difficulty: float):
        """Set difficulty level and update configuration."""
        self.difficulty_level = max(0.0, min(1.0, difficulty))
        self.config = self._create_config(self.difficulty_level)
        logger.info(f"Updated difficulty level to {self.difficulty_level}") 