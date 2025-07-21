"""
WHAT Dimension - Pattern Synthesis Engine
========================================

Advanced technical pattern recognition and synthesis engine for the 5D+1 sensory cortex.
Implements sophisticated pattern detection beyond traditional indicators including:
- Fractal pattern recognition
- Harmonic pattern analysis
- Volume profile analysis
- Price action DNA synthesis
- Statistical pattern validation

Author: EMP Development Team
Phase: 2 - Truth-First Completion
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy.signal import find_peaks
from scipy.stats import linregress

logger = logging.getLogger(__name__)


@dataclass
class PatternSynthesis:
    """Unified pattern synthesis result"""
    fractal_patterns: List[Dict[str, Any]]
    harmonic_patterns: List[Dict[str, Any]]
    volume_profile: Dict[str, Any]
    price_action_dna: Dict[str, Any]
    pattern_strength: float
    confidence_score: float


@dataclass
class FractalPattern:
    """Represents a detected fractal pattern"""
    pattern_type: str
    start_time: datetime
    end_time: datetime
    start_price: float
    end_price: float
    retracement_levels: List[float]
    extension_levels: List[float]
    confidence: float
    strength: float


@dataclass
class HarmonicPattern:
    """Represents a detected harmonic pattern"""
    pattern_name: str
    points: List[Dict[str, float]]
    ratios: Dict[str, float]
    target_price: float
    stop_loss: float
    confidence: float
    validity: bool


@dataclass
class VolumeProfile:
    """Volume profile analysis results"""
    value_area_high: float
    value_area_low: float
    point_of_control: float
    high_volume_nodes: List[float]
    low_volume_nodes: List[float]
    volume_distribution: Dict[str, float]


@dataclass
class PriceActionDNA:
    """Price action characteristics"""
    dna_sequence: str
    volatility_signature: float
    momentum_signature: float
    volume_signature: float
    pattern_complexity: int
    uniqueness_score: float


class PatternSynthesisEngine:
    """
    Advanced pattern synthesis engine for technical analysis.
    
    Provides sophisticated pattern recognition beyond traditional indicators,
    including fractal patterns, harmonic patterns, volume profile analysis,
    and price action DNA synthesis.
    """
    
    def __init__(self):
        self.fractal_detector = FractalDetector()
        self.harmonic_analyzer = HarmonicAnalyzer()
        self.volume_profiler = VolumeProfiler()
        self.dna_synthesizer = PriceActionDNASynthesizer()
        self.validator = PatternValidator()
        
    async def synthesize_patterns(self, market_data: pd.DataFrame) -> PatternSynthesis:
        """
        Comprehensive pattern synthesis of market data.
        
        Args:
            market_data: OHLCV DataFrame with datetime index
            
        Returns:
            PatternSynthesis object containing all pattern analysis results
        """
        try:
            if market_data.empty or len(market_data) < 50:
                logger.warning("Insufficient data for pattern analysis")
                return self._get_fallback_synthesis()
            
            # Detect fractal patterns
            fractal_patterns = await self.fractal_detector.detect_fractals(market_data)
            
            # Analyze harmonic patterns
            harmonic_patterns = await self.harmonic_analyzer.detect_harmonics(market_data)
            
            # Generate volume profile
            volume_profile = await self.volume_profiler.analyze_volume_profile(market_data)
            
            # Synthesize price action DNA
            price_dna = await self.dna_synthesizer.synthesize_dna(market_data)
            
            # Validate pattern strength
            pattern_strength = await self.validator.validate_patterns(
                fractal_patterns, harmonic_patterns, market_data
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(
                fractal_patterns, harmonic_patterns, volume_profile
            )
            
            return PatternSynthesis(
                fractal_patterns=[self._fractal_to_dict(f) for f in fractal_patterns],
                harmonic_patterns=[self._harmonic_to_dict(h) for h in harmonic_patterns],
                volume_profile=self._volume_to_dict(volume_profile),
                price_action_dna=self._dna_to_dict(price_dna),
                pattern_strength=pattern_strength,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"Pattern synthesis failed: {e}")
            return self._get_fallback_synthesis()
    
    def _fractal_to_dict(self, fractal: FractalPattern) -> Dict[str, Any]:
        """Convert FractalPattern to dictionary"""
        return {
            'pattern_type': fractal.pattern_type,
            'start_time': fractal.start_time,
            'end_time': fractal.end_time,
            'start_price': fractal.start_price,
            'end_price': fractal.end_price,
            'retracement_levels': fractal.retracement_levels,
            'extension_levels': fractal.extension_levels,
            'confidence': fractal.confidence,
            'strength': fractal.strength
        }
    
    def _harmonic_to_dict(self, harmonic: HarmonicPattern) -> Dict[str, Any]:
        """Convert HarmonicPattern to dictionary"""
        return {
            'pattern_name': harmonic.pattern_name,
            'points': harmonic.points,
            'ratios': harmonic.ratios,
            'target_price': harmonic.target_price,
            'stop_loss': harmonic.stop_loss,
            'confidence': harmonic.confidence,
            'validity': harmonic.validity
        }
    
    def _volume_to_dict(self, volume: VolumeProfile) -> Dict[str, Any]:
        """Convert VolumeProfile to dictionary"""
        return {
            'value_area_high': volume.value_area_high,
            'value_area_low': volume.value_area_low,
            'point_of_control': volume.point_of_control,
            'high_volume_nodes': volume.high_volume_nodes,
            'low_volume_nodes': volume.low_volume_nodes,
            'volume_distribution': volume.volume_distribution
        }
    
    def _dna_to_dict(self, dna: PriceActionDNA) -> Dict[str, Any]:
        """Convert PriceActionDNA to dictionary"""
        return {
            'dna_sequence': dna.dna_sequence,
            'volatility_signature': dna.volatility_signature,
            'momentum_signature': dna.momentum_signature,
            'volume_signature': dna.volume_signature,
            'pattern_complexity': dna.pattern_complexity,
            'uniqueness_score': dna.uniqueness_score
        }
    
    def _calculate_confidence(self, fractals: List, harmonics: List, volume: VolumeProfile) -> float:
        """Calculate overall confidence in pattern analysis"""
        factors = [
            len(fractals) / 10.0,  # Normalize to 0-1
            len(harmonics) / 5.0,  # Normalize to 0-1
            0.8 if volume.point_of_control else 0.0,
            0.9 if volume.value_area_high > volume.value_area_low else 0.0
        ]
        return min(1.0, np.mean(factors))
    
    def _get_fallback_synthesis(self) -> PatternSynthesis:
        """Return fallback synthesis when pattern detection fails"""
        return PatternSynthesis(
            fractal_patterns=[],
            harmonic_patterns=[],
            volume_profile={},
            price_action_dna={},
            pattern_strength=0.0,
            confidence_score=0.1
        )


class FractalDetector:
    """Advanced fractal pattern detection"""
    
    async def detect_fractals(self, data: pd.DataFrame) -> List[FractalPattern]:
        """Detect fractal patterns in price data"""
        fractals = []
        
        # Detect Elliott Wave patterns
        elliott_waves = await self._detect_elliott_waves(data)
        fractals.extend(elliott_waves)
        
        # Detect Fibonacci retracements
        fib_retracements = await self._detect_fibonacci_retracements(data)
        fractals.extend(fib_retracements)
        
        # Detect extension patterns
        extensions = await self._detect_extension_patterns(data)
        fractals.extend(extensions)
        
        return fractals
    
    async def _detect_elliott_waves(self, data: pd.DataFrame) -> List[FractalPattern]:
        """Detect Elliott Wave patterns"""
        waves = []
        
        # Find significant highs and lows
        highs = find_peaks(data['high'].values, distance=5)[0]
        lows = find_peaks(-data['low'].values, distance=5)[0]
        
        # Look for 5-wave patterns
        if len(highs) >= 5 and len(lows) >= 5:
            # Basic Elliott wave detection logic
            for i in range(len(highs) - 4):
                wave_sequence = self._validate_elliott_sequence(
                    data, highs[i:i+5], lows[i:i+5]
                )
                if wave_sequence:
                    waves.append(wave_sequence)
        
        return waves
    
    async def _detect_fibonacci_retracements(self, data: pd.DataFrame) -> List[FractalPattern]:
        """Detect Fibonacci retracement patterns"""
        retracements = []
        
        # Find significant swings
        swings = self._identify_significant_swings(data)
        
        for swing in swings:
            retracement_levels = self._calculate_fibonacci_levels(swing)
            if retracement_levels:
                pattern = FractalPattern(
                    pattern_type="fibonacci_retracement",
                    start_time=swing['start_time'],
                    end_time=swing['end_time'],
                    start_price=swing['start_price'],
                    end_price=swing['end_price'],
                    retracement_levels=retracement_levels,
                    extension_levels=[],
                    confidence=swing['confidence'],
                    strength=swing['strength']
                )
                retracements.append(pattern)
        
        return retracements
    
    async def _detect_extension_patterns(self, data: pd.DataFrame) -> List[FractalPattern]:
        """Detect extension patterns"""
        extensions = []
        
        # Look for 1.618, 2.618, 4.236 extensions
        significant_moves = self._identify_significant_moves(data)
        
        for move in significant_moves:
            extension_levels = self._calculate_extension_levels(move)
            if extension_levels:
                pattern = FractalPattern(
                    pattern_type="fibonacci_extension",
                    start_time=move['start_time'],
                    end_time=move['end_time'],
                    start_price=move['start_price'],
                    end_price=move['end_price'],
                    retracement_levels=[],
                    extension_levels=extension_levels,
                    confidence=move['confidence'],
                    strength=move['strength']
                )
                extensions.append(pattern)
        
        return extensions
    
    def _identify_significant_swings(self, data: pd.DataFrame) -> List[Dict]:
        """Identify significant price swings"""
        swings = []
        
        # Find local maxima and minima
        highs = find_peaks(data['high'].values, distance=10, prominence=0.02)[0]
        lows = find_peaks(-data['low'].values, distance=10, prominence=0.02)[0]
        
        # Combine and sort points
        points = []
        for idx in highs:
            points.append({'time': data.index[idx], 'price': data['high'].iloc[idx], 'type': 'high'})
        for idx in lows:
            points.append({'time': data.index[idx], 'price': data['low'].iloc[idx], 'type': 'low'})
        
        points.sort(key=lambda x: x['time'])
        
        # Identify significant swings
        for i in range(1, len(points)):
            prev = points[i-1]
            curr = points[i]
            
            if prev['type'] != curr['type']:
                price_change = abs(curr['price'] - prev['price']) / prev['price']
                if price_change > 0.05:  # 5% minimum swing
                    swings.append({
                        'start_time': prev['time'],
                        'end_time': curr['time'],
                        'start_price': prev['price'],
                        'end_price': curr['price'],
                        'confidence': min(1.0, price_change * 10),
                        'strength': price_change
                    })
        
        return swings
    
    def _calculate_fibonacci_levels(self, swing: Dict) -> List[float]:
        """Calculate Fibonacci retracement levels"""
        start_price = swing['start_price']
        end_price = swing['end_price']
        price_range = abs(end_price - start_price)
        
        levels = []
        fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
        
        for ratio in fib_ratios:
            level = start_price + (end_price - start_price) * ratio
            levels.append(level)
        
        return levels
    
    def _identify_significant_moves(self, data: pd.DataFrame) -> List[Dict]:
        """Identify significant price moves for extension patterns"""
        return self._identify_significant_swings(data)
    
    def _calculate_extension_levels(self, move: Dict) -> List[float]:
        """Calculate Fibonacci extension levels"""
        start_price = move['start_price']
        end_price = move['end_price']
        move_distance = abs(end_price - start_price)
        
        extensions = []
        extension_ratios = [1.618, 2.618, 4.236, 6.854]
        
        direction = 1 if end_price > start_price else -1
        
        for ratio in extension_ratios:
            extension = end_price + (direction * move_distance * ratio)
            extensions.append(extension)
        
        return extensions
    
    def _validate_elliott_sequence(self, data: pd.DataFrame, highs: List, lows: List) -> Optional[FractalPattern]:
        """Validate Elliott wave sequence"""
        if len(highs) < 5 or len(lows) < 5:
            return None
        
        # Basic Elliott wave validation
        try:
            wave_points = []
            for i, (h, l) in enumerate(zip(highs[:5], lows[:5])):
                if i % 2 == 0:
                    wave_points.append({
                        'time': data.index[h],
                        'price': data['high'].iloc[h],
                        'type': 'high' if i in [0, 2, 4] else 'low'
                    })
                else:
                    wave_points.append({
                        'time': data.index[l],
                        'price': data['low'].iloc[l],
                        'type': 'low' if i in [1, 3] else 'high'
                    })
            
            # Calculate wave ratios
            wave1 = abs(wave_points[1]['price'] - wave_points[0]['price'])
            wave3 = abs(wave_points[3]['price'] - wave_points[2]['price'])
            wave5 = abs(wave_points[4]['price'] - wave_points[3]['price'])
            
            # Basic wave ratio validation
            if 0.8 <= wave3/wave1 <= 1.6 and 0.8 <= wave5/wave3 <= 1.6:
                return FractalPattern(
                    pattern_type="elliott_wave_5",
                    start_time=wave_points[0]['time'],
                    end_time=wave_points[4]['time'],
                    start_price=wave_points[0]['price'],
                    end_price=wave_points[4]['price'],
                    retracement_levels=[0.382, 0.5, 0.618],
                    extension_levels=[1.618, 2.618],
                    confidence=0.7,
                    strength=1.0
                )
                
        except (IndexError, ZeroDivisionError):
            pass
        
        return None


class HarmonicAnalyzer:
    """Harmonic pattern detection (Gartley, Butterfly, Crab, Bat)"""
    
    async def detect_harmonics(self, data: pd.DataFrame) -> List[HarmonicPattern]:
        """Detect harmonic patterns in price data"""
        harmonics = []
        
        # Detect Gartley patterns
        gartleys = await self._detect_gartley(data)
        harmonics.extend(gartleys)
        
        # Detect Butterfly patterns
        butterflies = await self._detect_butterfly(data)
        harmonics.extend(butterflies)
        
        # Detect Crab patterns
        crabs = await self._detect_crab(data)
        harmonics.extend(crabs)
        
        # Detect Bat patterns
        bats = await self._detect_bat(data)
        harmonics.extend(bats)
        
        return harmonics
    
    async def _detect_gartley(self, data: pd.DataFrame) -> List[HarmonicPattern]:
        """Detect Gartley harmonic patterns"""
        gartleys = []
        
        # Find potential XABCD patterns
        patterns = self._find_xabcd_patterns(data)
        
        for pattern in patterns:
            if self._validate_gartley_ratios(pattern):
                gartley = HarmonicPattern(
                    pattern_name="gartley",
                    points=pattern['points'],
                    ratios=pattern['ratios'],
                    target_price=pattern['target'],
                    stop_loss=pattern['stop'],
                    confidence=pattern['confidence'],
                    validity=True
                )
                gartleys.append(gartley)
        
        return gartleys
    
    async def _detect_butterfly(self, data: pd.DataFrame) -> List[HarmonicPattern]:
        """Detect Butterfly harmonic patterns"""
        butterflies = []
        
        patterns = self._find_xabcd_patterns(data)
        
        for pattern in patterns:
            if self._validate_butterfly_ratios(pattern):
                butterfly = HarmonicPattern(
                    pattern_name="butterfly",
                    points=pattern['points'],
                    ratios=pattern['ratios'],
                    target_price=pattern['target'],
                    stop_loss=pattern['stop'],
                    confidence=pattern['confidence'],
                    validity=True
                )
                butterflies.append(butterfly)
        
        return butterflies
    
    async def _detect_crab(self, data: pd.DataFrame) -> List[HarmonicPattern]:
        """Detect Crab harmonic patterns"""
        crabs = []
        
        patterns = self._find_xabcd_patterns(data)
        
        for pattern in patterns:
            if self._validate_crab_ratios(pattern):
                crab = HarmonicPattern(
                    pattern_name="crab",
                    points=pattern['points'],
                    ratios=pattern['ratios'],
                    target_price=pattern['target'],
                    stop_loss=pattern['stop'],
                    confidence=pattern['confidence'],
                    validity=True
                )
                crabs.append(crab)
        
        return crabs
    
    async def _detect_bat(self, data: pd.DataFrame) -> List[HarmonicPattern]:
        """Detect Bat harmonic patterns"""
        bats = []
        
        patterns = self._find_xabcd_patterns(data)
        
        for pattern in patterns:
            if self._validate_bat_ratios(pattern):
                bat = HarmonicPattern(
                    pattern_name="bat",
                    points=pattern['points'],
                    ratios=pattern['ratios'],
                    target_price=pattern['target'],
                    stop_loss=pattern['stop'],
                    confidence=pattern['confidence'],
                    validity=True
                )
                bats.append(bat)
        
        return bats
    
    def _find_xabcd_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """Find potential XABCD harmonic patterns"""
        patterns = []
        
        # Find significant turning points
        highs = find_peaks(data['high'].values, distance=10, prominence=0.02)[0]
        lows = find_peaks(-data['low'].values, distance=10, prominence=0.02)[0]
        
        # Combine and sort points
        points = []
        for idx in highs:
            points.append({'time': data.index[idx], 'price': data['high'].iloc[idx], 'type': 'high'})
        for idx in lows:
            points.append({'time': data.index[idx], 'price': data['low'].iloc[idx], 'type': 'low'})
        
        points.sort(key=lambda x: x['time'])
        
        # Look for 5-point patterns
        for i in range(len(points) - 4):
            x_point = points[i]
            a_point = points[i+1]
            b_point = points[i+2]
            c_point = points[i+3]
            d_point = points[i+4]
            
            pattern = {
                'points': [x_point, a_point, b_point, c_point, d_point],
                'ratios': self._calculate_harmonic_ratios(x_point, a_point, b_point, c_point, d_point),
                'target': d_point['price'] * 1.272,  # Basic target calculation
                'stop': d_point['price'] * 0.9,     # Basic stop loss
                'confidence': 0.7
            }
            patterns.append(pattern)
        
        return patterns
    
    def _calculate_harmonic_ratios(self, x, a, b, c, d) -> Dict[str, float]:
        """Calculate harmonic ratios for pattern validation"""
        xa = abs(a['price'] - x['price'])
        ab = abs(b['price'] - a['price'])
        bc = abs(c['price'] - b['price'])
        cd = abs(d['price'] - c['price'])
        
        ratios = {
            'xa': xa,
            'ab': ab,
            'bc': bc,
            'cd': cd,
            'ab_xa': ab / xa if xa > 0 else 0,
            'bc_ab': bc / ab if ab > 0 else 0,
            'cd_bc': cd / bc if bc > 0 else 0,
            'ad_xa': abs(d['price'] - x['price']) / xa if xa > 0 else 0
        }
        
        return ratios
    
    def _validate_gartley_ratios(self, pattern: Dict) -> bool:
        """Validate Gartley pattern ratios"""
        ratios = pattern['ratios']
        
        # Gartley ratios: AB=0.618XA, BC=0.382-0.886AB, CD=1.27-1.618BC, AD=0.786XA
        return (
            0.5 <= ratios['ab_xa'] <= 0.7 and
            0.3 <= ratios['bc_ab'] <= 0.9 and
            1.2 <= ratios['cd_bc'] <= 1.7 and
            0.7 <= ratios['ad_xa'] <= 0.85
        )
    
    def _validate_butterfly_ratios(self, pattern: Dict) -> bool:
        """Validate Butterfly pattern ratios"""
        ratios = pattern['ratios']
        
        # Butterfly ratios: AB=0.786XA, BC=0.382-0.886AB, CD=1.618-2.618BC, AD=1.27XA
        return (
            0.7 <= ratios['ab_xa'] <= 0.9 and
            0.3 <= ratios['bc_ab'] <= 0.9 and
            1.5 <= ratios['cd_bc'] <= 2.7 and
            1.2 <= ratios['ad_xa'] <= 1.4
        )
    
    def _validate_crab_ratios(self, pattern: Dict) -> bool:
        """Validate Crab pattern ratios"""
        ratios = pattern['ratios']
        
        # Crab ratios: AB=0.382-0.618XA, BC=0.382-0.886AB, CD=2.24-3.618BC, AD=1.618XA
        return (
            0.3 <= ratios['ab_xa'] <= 0.7 and
            0.3 <= ratios['bc_ab'] <= 0.9 and
            2.0 <= ratios['cd_bc'] <= 3.7 and
            1.5 <= ratios['ad_xa'] <= 1.7
        )
    
    def _validate_bat_ratios(self, pattern: Dict) -> bool:
        """Validate Bat pattern ratios"""
        ratios = pattern['ratios']
        
        # Bat ratios: AB=0.382-0.5XA, BC=0.382-0.886AB, CD=1.618-2.618BC, AD=0.886XA
        return (
            0.3 <= ratios['ab_xa'] <= 0.6 and
            0.3 <= ratios['bc_ab'] <= 0.9 and
            1.5 <= ratios['cd_bc'] <= 2.7 and
            0.8 <= ratios['ad_xa'] <= 0.95
        )


class VolumeProfiler:
    """Volume profile analysis"""
    
    async def analyze_volume_profile(self, data: pd.DataFrame) -> VolumeProfile:
        """Analyze volume profile for price levels"""
        try:
            if data.empty:
                return self._get_fallback_volume_profile()
            
            # Calculate price levels
            price_min = data['low'].min()
            price_max = data['high'].max()
            price_range = price_max - price_min
            
            # Create price bins
            num_bins = 20
            bin_width = price_range / num_bins
            bins = np.linspace(price_min, price_max, num_bins + 1)
            
            # Calculate volume at each price level
            volume_by_price = {}
            for i in range(len(bins) - 1):
                bin_low = bins[i]
                bin_high = bins[i + 1]
                
                # Find all candles that touch this price range
                mask = (data['low'] <= bin_high) & (data['high'] >= bin_low)
                bin_volume = data[mask]['volume'].sum()
                bin_price = (bin_low + bin_high) / 2
                
                volume_by_price[bin_price] = bin_volume
            
            if not volume_by_price:
                return self._get_fallback_volume_profile()
            
            # Find point of control (highest volume)
            poc_price = max(volume_by_price, key=volume_by_price.get)
            total_volume = sum(volume_by_price.values())
            
            # Calculate value area (70% of volume)
            sorted_volumes = sorted(volume_by_price.items(), key=lambda x: x[1], reverse=True)
            cumulative_volume = 0
            value_area_prices = []
            
            for price, volume in sorted_volumes:
                cumulative_volume += volume
                value_area_prices.append(price)
                if cumulative_volume >= total_volume * 0.7:
                    break
            
            value_area_high = max(value_area_prices)
            value_area_low = min(value_area_prices)
            
            # Identify high and low volume nodes
            avg_volume = total_volume / len(volume_by_price)
            high_volume_nodes = [p for p, v in volume_by_price.items() if v > avg_volume * 1.5]
            low_volume_nodes = [p for p, v in volume_by_price.items() if v < avg_volume * 0.5]
            
            return VolumeProfile(
                value_area_high=value_area_high,
                value_area_low=value_area_low,
                point_of_control=poc_price,
                high_volume_nodes=high_volume_nodes,
                low_volume_nodes=low_volume_nodes,
                volume_distribution=volume_by_price
            )
            
        except Exception as e:
            logger.error(f"Volume profile analysis failed: {e}")
            return self._get_fallback_volume_profile()
    
    def _get_fallback_volume_profile(self) -> VolumeProfile:
        """Return fallback volume profile"""
        return VolumeProfile(
            value_area_high=0.0,
            value_area_low=0.0,
            point_of_control=0.0,
            high_volume_nodes=[],
            low_volume_nodes=[],
            volume_distribution={}
        )


class PriceActionDNASynthesizer:
    """Price action DNA synthesis"""
    
    async def synthesize_dna(self, data: pd.DataFrame) -> PriceActionDNA:
        """Synthesize price action characteristics into DNA sequence"""
        try:
            if data.empty:
                return self._get_fallback_dna()
            
            # Calculate volatility signature
            volatility = data['high'].rolling(20).std().iloc[-1] / data['close'].iloc[-1]
            
            # Calculate momentum signature
            returns = data['close'].pct_change()
            momentum = returns.rolling(10).mean().iloc[-1]
            
            # Calculate volume signature
            volume_ratio = data['volume'].iloc[-1] / data['volume'].rolling(20).mean().iloc[-1]
            
            # Generate DNA sequence based on characteristics
            dna_sequence = self._generate_dna_sequence(volatility, momentum, volume_ratio)
            
            # Calculate pattern complexity
            complexity = self._calculate_pattern_complexity(data)
            
            # Calculate uniqueness score
            uniqueness = self._calculate_uniqueness_score(data)
            
            return PriceActionDNA(
                dna_sequence=dna_sequence,
                volatility_signature=volatility,
                momentum_signature=momentum,
                volume_signature=volume_ratio,
                pattern_complexity=complexity,
                uniqueness_score=uniqueness
            )
            
        except Exception as e:
            logger.error(f"DNA synthesis failed: {e}")
            return self._get_fallback_dna()
    
    def _generate_dna_sequence(self, volatility: float, momentum: float, volume: float) -> str:
        """Generate DNA sequence based on market characteristics"""
        # Create sequence based on characteristics
        vol_code = "V" if volatility > 0.02 else "v"
        mom_code = "M" if momentum > 0 else "m"
        vol_ratio_code = "R" if volume > 1.2 else "r"
        
        # Add pattern complexity indicators
        complexity_indicators = "ABC"  # Basic complexity
        
        return f"{vol_code}{mom_code}{vol_ratio_code}{complexity_indicators}"
    
    def _calculate_pattern_complexity(self, data: pd.DataFrame) -> int:
        """Calculate pattern complexity score"""
        # Simple complexity based on price movements
        price_changes = data['close'].pct_change().abs()
        complexity = int(np.sum(price_changes > 0.01))
        return min(complexity, 10)
    
    def _calculate_uniqueness_score(self, data: pd.DataFrame) -> float:
        """Calculate uniqueness score based on price action"""
        # Calculate price action uniqueness
        price_range = (data['high'].max() - data['low'].min()) / data['close'].mean()
        volume_range = data['volume'].max() / max(data['volume'].min(), 1)
        
        uniqueness = (price_range * 0.7 + np.log(volume_range) * 0.3) / 2
        return min(1.0, uniqueness)
    
    def _get_fallback_dna(self) -> PriceActionDNA:
        """Return fallback DNA"""
        return PriceActionDNA(
            dna_sequence="DEFAULT",
            volatility_signature=0.0,
            momentum_signature=0.0,
            volume_signature=0.0,
            pattern_complexity=0,
            uniqueness_score=0.0
        )


class PatternValidator:
    """Pattern validation and strength assessment"""
    
    async def validate_patterns(self, fractals: List[FractalPattern], 
                              harmonics: List[HarmonicPattern], 
                              data: pd.DataFrame) -> float:
        """Validate detected patterns and calculate overall strength"""
        try:
            if not fractals and not harmonics:
                return 0.0
            
            # Calculate fractal strength
            fractal_strength = np.mean([f.strength for f in fractals]) if fractals else 0.0
            
            # Calculate harmonic strength
            harmonic_strength = np.mean([h.confidence for h in harmonics]) if harmonics else 0.0
            
            # Calculate data quality score
            data_quality = self._calculate_data_quality(data)
            
            # Combine strengths
            total_strength = (fractal_strength * 0.4 + harmonic_strength * 0.4 + data_quality * 0.2)
            
            return min(1.0, total_strength)
            
        except Exception as e:
            logger.error(f"Pattern validation failed: {e}")
            return 0.0
    
    def _calculate_data_quality(self, data: pd.DataFrame) -> float:
        """Calculate data quality score"""
        if data.empty:
            return 0.0
        
        # Check for missing data
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        
        # Check for sufficient data points
        data_points_ratio = min(1.0, len(data) / 200)
        
        quality = (1 - missing_ratio) * 0.7 + data_points_ratio * 0.3
        return quality


# Integration class for WHAT dimension
class WhatDimension:
    """WHAT dimension implementation for 5D+1 sensory cortex"""
    
    def __init__(self):
        self.engine = PatternSynthesisEngine()
    
    async def analyze(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market data using WHAT dimension"""
        result = await self.engine.synthesize_patterns(market_data)
        return {
            'fractal_patterns': result.fractal_patterns,
            'harmonic_patterns': result.harmonic_patterns,
            'volume_profile': result.volume_profile,
            'price_action_dna': result.price_action_dna,
            'pattern_strength': result.pattern_strength,
            'confidence_score': result.confidence_score
        }
