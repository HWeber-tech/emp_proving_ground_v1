"""
ANOMALY Dimension - Manipulation Detection System
==============================================

Advanced market manipulation and anomaly detection engine for the 5D+1 sensory cortex.
Implements sophisticated detection algorithms for:
- Spoofing detection
- Wash trading identification
- Pump and dump patterns
- Regulatory arbitrage detection
- Market microstructure anomalies

Author: EMP Development Team
Phase: 2 - Truth-First Completion
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class SpoofingDetection:
    """Spoofing detection results"""
    detected: bool
    confidence: float
    spoofing_type: str  # 'layering', 'spoofing', 'quote_stuffing'
    affected_levels: List[float]
    spoofing_intensity: float
    duration_seconds: float


@dataclass
class WashTradingDetection:
    """Wash trading detection results"""
    detected: bool
    confidence: float
    wash_volume: float
    wash_ratio: float
    suspicious_patterns: List[Dict[str, Any]]
    regulatory_risk: str  # 'low', 'medium', 'high'


@dataclass
class PumpDumpDetection:
    """Pump and dump detection results"""
    detected: bool
    confidence: float
    pump_start: datetime
    dump_start: datetime
    pump_magnitude: float
    dump_magnitude: float
    volume_anomaly: float


@dataclass
class RegulatoryArbitrage:
    """Regulatory arbitrage detection"""
    arbitrage_type: str
    confidence: float
    affected_markets: List[str]
    profit_potential: float
    regulatory_risk: str
    detection_time: datetime


@dataclass
class MicrostructureAnomaly:
    """Market microstructure anomalies"""
    anomaly_type: str
    severity: float
    affected_metrics: List[str]
    confidence: float
    recommended_action: str


@dataclass
class AnomalyDetection:
    """Complete anomaly detection results"""
    spoofing: SpoofingDetection
    wash_trading: WashTradingDetection
    pump_dump: PumpDumpDetection
    regulatory_arbitrage: List[RegulatoryArbitrage]
    microstructure_anomalies: List[MicrostructureAnomaly]
    overall_risk_score: float
    confidence: float


class ManipulationDetectionSystem:
    """Advanced market manipulation detection system."""
    
    def __init__(self):
        self.spoof_detector = SpoofingDetector()
        self.wash_detector = WashTradingDetector()
        self.pump_detector = PumpDumpDetector()
        self.arbitrage_detector = RegulatoryArbitrageDetector()
        self.microstructure_analyzer = MicrostructureAnomalyDetector()
        
    async def detect_manipulation(self, market_data: pd.DataFrame) -> AnomalyDetection:
        """Comprehensive manipulation detection."""
        try:
            if market_data.empty:
                return self._get_fallback_detection()
            
            # Detect spoofing patterns
            spoofing = await self.spoof_detector.detect_spoofing(market_data)
            
            # Detect wash trading
            wash_trading = await self.wash_detector.detect_wash_trading(market_data)
            
            # Detect pump and dump patterns
            pump_dump = await self.pump_detector.detect_pump_dump(market_data)
            
            # Detect regulatory arbitrage
            regulatory_arbitrage = await self.arbitrage_detector.detect_arbitrage(market_data)
            
            # Detect microstructure anomalies
            microstructure_anomalies = await self.microstructure_analyzer.detect_anomalies(market_data)
            
            # Calculate overall risk score
            overall_risk = self._calculate_overall_risk(
                spoofing, wash_trading, pump_dump, regulatory_arbitrage, microstructure_anomalies
            )
            
            # Calculate overall confidence
            confidence = self._calculate_confidence(
                spoofing, wash_trading, pump_dump, regulatory_arbitrage, microstructure_anomalies
            )
            
            return AnomalyDetection(
                spoofing=spoofing,
                wash_trading=wash_trading,
                pump_dump=pump_dump,
                regulatory_arbitrage=regulatory_arbitrage,
                microstructure_anomalies=microstructure_anomalies,
                overall_risk_score=overall_risk,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Manipulation detection failed: {e}")
            return self._get_fallback_detection()
    
    def _calculate_overall_risk(self, spoofing: SpoofingDetection,
                              wash_trading: WashTradingDetection,
                              pump_dump: PumpDumpDetection,
                              regulatory_arbitrage: List[RegulatoryArbitrage],
                              microstructure_anomalies: List[MicrostructureAnomaly]) -> float:
        """Calculate overall risk score"""
        risk_factors = [
            1.0 if spoofing.detected else 0.0,
            1.0 if wash_trading.detected else 0.0,
            1.0 if pump_dump.detected else 0.0,
            min(len(regulatory_arbitrage) / 3.0, 1.0),
            min(len(microstructure_anomalies) / 5.0, 1.0)
        ]
        
        weights = [0.25, 0.25, 0.25, 0.15, 0.10]
        return np.average(risk_factors, weights=weights)
    
    def _calculate_confidence(self, spoofing: SpoofingDetection,
                            wash_trading: WashTradingDetection,
                            pump_dump: PumpDumpDetection,
                            regulatory_arbitrage: List[RegulatoryArbitrage],
                            microstructure_anomalies: List[MicrostructureAnomaly]) -> float:
        """Calculate overall confidence"""
        confidences = [
            spoofing.confidence,
            wash_trading.confidence,
            pump_dump.confidence,
            np.mean([ra.confidence for ra in regulatory_arbitrage]) if regulatory_arbitrage else 0.0,
            np.mean([ma.confidence for ma in microstructure_anomalies]) if microstructure_anomalies else 0.0
        ]
        return np.mean([c for c in confidences if c > 0])
    
    def _get_fallback_detection(self) -> AnomalyDetection:
        """Return fallback anomaly detection"""
        return AnomalyDetection(
            spoofing=SpoofingDetection(
                detected=False,
                confidence=0.0,
                spoofing_type='none',
                affected_levels=[],
                spoofing_intensity=0.0,
                duration_seconds=0.0
            ),
            wash_trading=WashTradingDetection(
                detected=False,
                confidence=0.0,
                wash_volume=0.0,
                wash_ratio=0.0,
                suspicious_patterns=[],
                regulatory_risk='low'
            ),
            pump_dump=PumpDumpDetection(
                detected=False,
                confidence=0.0,
                pump_start=datetime.now(),
                dump_start=datetime.now(),
                pump_magnitude=0.0,
                dump_magnitude=0.0,
                volume_anomaly=0.0
            ),
            regulatory_arbitrage=[],
            microstructure_anomalies=[],
            overall_risk_score=0.0,
            confidence=0.1
        )


class SpoofingDetector:
    """Spoofing and layering detection"""
    
    async def detect_spoofing(self, market_data: pd.DataFrame) -> SpoofingDetection:
        """Detect spoofing patterns in market data"""
        try:
            if market_data.empty:
                return SpoofingDetection(
                    detected=False,
                    confidence=0.0,
                    spoofing_type='none',
                    affected_levels=[],
                    spoofing_intensity=0.0,
                    duration_seconds=0.0
                )
            
            # Detect large orders that disappear
            volume_anomaly = self._detect_volume_anomaly(market_data)
            price_reversal = self._detect_price_reversal(market_data)
            
            # Calculate spoofing indicators
            spoofing_score = (volume_anomaly + price_reversal) / 2
            
            detected = spoofing_score > 0.7
            confidence = min(spoofing_score, 1.0)
            
            return SpoofingDetection(
                detected=detected,
                confidence=confidence,
                spoofing_type='layering' if detected else 'none',
                affected_levels=[market_data['high'].max(), market_data['low'].min()],
                spoofing_intensity=spoofing_score,
                duration_seconds=300.0
            )
            
        except Exception as e:
            logger.error(f"Spoofing detection failed: {e}")
            return SpoofingDetection(
                detected=False,
                confidence=0.0,
                spoofing_type='none',
                affected_levels=[],
                spoofing_intensity=0.0,
                duration_seconds=0.0
            )
    
    def _detect_volume_anomaly(self, data: pd.DataFrame) -> float:
        """Detect volume-based spoofing anomalies"""
        if len(data) < 10:
            return 0.0
        
        volume = data['volume']
        mean_vol = volume.mean()
        std_vol = volume.std()
        
        if std_vol == 0:
            return 0.0
        
        z_score = abs(volume.iloc[-1] - mean_vol) / std_vol
        return min(z_score / 3.0, 1.0)
    
    def _detect_price_reversal(self, data: pd.DataFrame) -> float:
        """Detect price reversal patterns"""
        if len(data) < 5:
            return 0.0
        
        returns = data['close'].pct_change().dropna()
        if len(returns) < 3:
            return 0.0
        
        # Look for sharp reversals
        reversal_strength = abs(returns.iloc[-1] - returns.iloc[-2])
        return min(reversal_strength * 10, 1.0)


class WashTradingDetector:
    """Wash trading detection"""
    
    async def detect_wash_trading(self, market_data: pd.DataFrame) -> WashTradingDetection:
        """Detect wash trading patterns"""
        try:
            if market_data.empty:
                return WashTradingDetection(
                    detected=False,
                    confidence=0.0,
                    wash_volume=0.0,
                    wash_ratio=0.0,
                    suspicious_patterns=[],
                    regulatory_risk='low'
                )
            
            # Analyze volume patterns
            volume_consistency = self._analyze_volume_consistency(market_data)
            price_stability = self._analyze_price_stability(market_data)
            
            # Calculate wash trading indicators
            wash_score = (volume_consistency + (1 - price_stability)) / 2
            
            detected = wash_score > 0.8
            confidence = min(wash_score, 1.0)
            
            # Calculate wash volume estimate
            total_volume = market_data['volume'].sum()
            wash_volume = total_volume * wash_score * 0.1 if detected else 0.0
            wash_ratio = wash_volume / total_volume if total_volume > 0 else 0.0
            
            return WashTradingDetection(
                detected=detected,
                confidence=confidence,
                wash_volume=wash_volume,
                wash_ratio=wash_ratio,
                suspicious_patterns=[{"type": "volume_anomaly", "severity": wash_score}],
                regulatory_risk='high' if detected else 'low'
            )
            
        except Exception as e:
            logger.error(f"Wash trading detection failed: {e}")
            return WashTradingDetection(
                detected=False,
                confidence=0.0,
                wash_volume=0.0,
                wash_ratio=0.0,
                suspicious_patterns=[],
                regulatory_risk='low'
            )
    
    def _analyze_volume_consistency(self, data: pd.DataFrame) -> float:
        """Analyze volume consistency patterns"""
        if len(data) < 5:
            return 0.0
        
        volume = data['volume']
        cv = volume.std() / volume.mean() if volume.mean() > 0 else 0.0
        return min(cv, 1.0)
    
    def _analyze_price_stability(self, data: pd.DataFrame) -> float:
        """Analyze price stability"""
        if len(data) < 5:
            return 0.0
        
        returns = data['close'].pct_change().dropna()
        if len(returns) == 0:
            return 0.0
        
        volatility = returns.std()
        return min(volatility * 10, 1.0)


class PumpDumpDetector:
    """Pump and dump detection"""
    
    async def detect_pump_dump(self, market_data: pd.DataFrame) -> PumpDumpDetection:
        """Detect pump and dump patterns"""
        try:
            if market_data.empty:
                return PumpDumpDetection(
                    detected=False,
                    confidence=0.0,
                    pump_start=datetime.now(),
                    dump_start=datetime.now(),
                    pump_magnitude=0.0,
                    dump_magnitude=0.0,
                    volume_anomaly=0.0
                )
            
            # Detect pump phase
            pump_detected, pump_start, pump_magnitude = self._detect_pump_phase(market_data)
            
            # Detect dump phase
            dump_detected, dump_start, dump_magnitude = self._detect_dump_phase(market_data)
            
            # Volume anomaly
            volume_anomaly = self._detect_volume_anomaly(market_data)
            
            # Combined detection
            detected = pump_detected and dump_detected
            confidence = min((pump_magnitude + dump_magnitude + volume_anomaly) / 3, 1.0)
            
            return PumpDumpDetection(
                detected=detected,
                confidence=confidence,
                pump_start=pump_start,
                dump_start=dump_start,
                pump_magnitude=pump_magnitude,
                dump_magnitude=dump_magnitude,
                volume_anomaly=volume_anomaly
            )
            
        except Exception as e:
            logger.error(f"Pump and dump detection failed: {e}")
            return PumpDumpDetection(
                detected=False,
                confidence=0.0,
                pump_start=datetime.now(),
                dump_start=datetime.now(),
                pump_magnitude=0.0,
                dump_magnitude=0.0,
                volume_anomaly=0.0
            )
    
    def _detect_pump_phase(self, data: pd.DataFrame) -> tuple:
        """Detect pump phase"""
        if len(data) < 10:
            return False, datetime.now(), 0.0
        
        # Look for rapid price increase
        price_change = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]
        volume_increase = data['volume'].iloc[-5:].mean() / data['volume'].iloc[:-5].mean() if len(data) > 5 else 1.0
        
        pump_detected = price_change > 0.05 and volume_increase > 2.0
        pump_magnitude = min(abs(price_change), 1.0)
        
        return pump_detected, data.index[0], pump_magnitude
    
    def _detect_dump_phase(self, data: pd.DataFrame) -> tuple:
        """Detect dump phase"""
        if len(data) < 10:
            return False, datetime.now(), 0.0
        
        # Look for rapid price decrease
        price_change = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]
        volume_increase = data['volume'].iloc[-5:].mean() / data['volume'].iloc[:-5].mean() if len(data) > 5 else 1.0
        
        dump_detected = price_change < -0.05 and volume_increase > 2.0
        dump_magnitude = min(abs(price_change), 1.0)
        
        return dump_detected, data.index[-1], dump_magnitude
    
    def _detect_volume_anomaly(self, data: pd.DataFrame) -> float:
        """Detect volume anomalies"""
        if len(data) < 5:
            return 0.0
        
        volume = data['volume']
        mean_vol = volume.mean()
        if mean_vol == 0:
            return 0.0
        
        max_vol = volume.max()
        return min((max_vol - mean_vol) / mean_vol, 1.0)


class RegulatoryArbitrageDetector:
    """Regulatory arbitrage detection"""
    
    async def detect_arbitrage(self, market_data: pd.DataFrame) -> List[RegulatoryArbitrage]:
        """Detect regulatory arbitrage opportunities"""
        try:
            if market_data.empty:
                return []
            
            # Simplified arbitrage detection
            arbitrage_opportunities = []
            
            # Check for price anomalies
            price_anomaly = self._detect_price_anomaly(market_data)
            if price_anomaly > 0.5:
                arbitrage = RegulatoryArbitrage(
                    arbitrage_type='price_discrepancy',
                    confidence=price_anomaly,
                    affected_markets=['primary', 'secondary'],
                    profit_potential=price_anomaly * 0.01,
                    regulatory_risk='medium',
                    detection_time=datetime.now()
                )
                arbitrage_opportunities.append(arbitrage)
            
            return arbitrage_opportunities
            
        except Exception as e:
            logger.error(f"Regulatory arbitrage detection failed: {e}")
            return []
    
    def _detect_price_anomaly(self, data: pd.DataFrame) -> float:
        """Detect price anomalies"""
        if len(data) < 5:
            return 0.0
        
        returns = data['close'].pct_change().dropna()
        if len(returns) == 0:
            return 0.0
        
        # Look for extreme price movements
        extreme_returns = abs(returns) > 0.05
        return min(extreme_returns.sum() / len(returns), 1.0)


class MicrostructureAnomalyDetector:
    """Market microstructure anomaly detection"""
    
    async def detect_anomalies(self, market_data: pd.DataFrame) -> List[MicrostructureAnomaly]:
        """Detect microstructure anomalies"""
        try:
            if market_data.empty:
                return []
            
            anomalies = []
            
            # Bid-ask spread anomaly
            spread_anomaly = self._detect_spread_anomaly(market_data)
            if spread_anomaly > 0.7:
                anomalies.append(MicrostructureAnomaly(
                    anomaly_type='spread_anomaly',
                    severity=spread_anomaly,
                    affected_metrics=['bid_ask_spread', 'liquidity'],
                    confidence=spread_anomaly,
                    recommended_action='monitor_closely'
                ))
            
            # Volume imbalance
            volume_imbalance = self._detect_volume_imbalance(market_data)
            if volume_imbalance > 0.8:
                anomalies.append(MicrostructureAnomaly(
                    anomaly_type='volume_imbalance',
                    severity=volume_imbalance,
                    affected_metrics=['volume', 'order_flow'],
                    confidence=volume_imbalance,
                    recommended_action='reduce_exposure'
                ))
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Microstructure anomaly detection failed: {e}")
            return []
    
    def _detect_spread_anomaly(self, data: pd.DataFrame) -> float:
        """Detect bid-ask spread anomalies"""
        if len(data) < 5:
            return 0.0
        
        # Use high-low as proxy for spread
        spreads = (data['high'] - data['low']) / data['close']
        mean_spread = spreads.mean()
        std_spread = spreads.std()
        
        if std_spread == 0:
            return 0.0
        
        z_score = abs(spreads.iloc[-1] - mean_spread) / std_spread
        return min(z_score / 3.0, 1.0)
    
    def _detect_volume_imbalance(self, data: pd.DataFrame) -> float:
        """Detect volume imbalance"""
        if len(data) < 5:
            return 0.0
        
        volume = data['volume']
        mean_vol = volume.mean()
        if mean_vol == 0:
            return 0.0
        
        # Check for extreme volume spikes
        max_vol = volume.max()
        return min((max_vol - mean_vol) / mean_vol, 1.0)
