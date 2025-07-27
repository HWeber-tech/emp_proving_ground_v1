"""
Manipulation Detection System - ANOMALY Dimension
===============================================

Detects market manipulation and anomalies.
Provides the ANOMALY dimension of the 5D+1 sensory cortex.

Author: EMP Development Team
Phase: 2 - Truth-First Completion
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SpoofingDetection:
    """Spoofing detection results"""
    detected: bool
    confidence: float
    spoofing_type: str
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
    suspicious_patterns: List[str]
    regulatory_risk: str


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
class AnomalyDetection:
    """Market manipulation and anomaly detection results"""
    spoofing: SpoofingDetection
    wash_trading: WashTradingDetection
    pump_dump: PumpDumpDetection
    regulatory_arbitrage: List[Dict[str, Any]]
    microstructure_anomalies: List[Dict[str, Any]]
    overall_risk_score: float
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)


class ManipulationDetectionSystem:
    """
    Implements the ANOMALY dimension of the 5D+1 sensory cortex.
    Detects market manipulation and anomalies.
    """
    
    def __init__(self):
        self.spoofing_detector = SpoofingDetector()
        self.wash_trading_detector = WashTradingDetector()
        self.pump_dump_detector = PumpDumpDetector()
        self.microstructure_analyzer = MicrostructureAnomalyDetector()
        self.logger = logging.getLogger(__name__)
    
    async def detect_manipulation(self, market_data: pd.DataFrame) -> AnomalyDetection:
        """
        Detect market manipulation and anomalies
        
        Args:
            market_data: Market data DataFrame
            
        Returns:
            AnomalyDetection: Detection results
        """
        try:
            # Detect spoofing
            spoofing = await self.spoofing_detector.detect_spoofing(market_data)
            
            # Detect wash trading
            wash_trading = await self.wash_trading_detector.detect_wash_trading(market_data)
            
            # Detect pump and dump
            pump_dump = await self.pump_dump_detector.detect_pump_dump(market_data)
            
            # Detect microstructure anomalies
            microstructure_anomalies = await self.microstructure_analyzer.detect_anomalies(market_data)
            
            # Calculate overall risk score
            overall_risk = self._calculate_overall_risk(spoofing, wash_trading, pump_dump, microstructure_anomalies)
            
            # Calculate confidence
            confidence = self._calculate_detection_confidence(spoofing, wash_trading, pump_dump)
            
            return AnomalyDetection(
                spoofing=spoofing,
                wash_trading=wash_trading,
                pump_dump=pump_dump,
                regulatory_arbitrage=[],
                microstructure_anomalies=microstructure_anomalies,
                overall_risk_score=overall_risk,
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Manipulation detection failed: {e}")
            return self._get_fallback_detection()
    
    def _calculate_overall_risk(self, spoofing: SpoofingDetection, wash_trading: WashTradingDetection,
                              pump_dump: PumpDumpDetection, microstructure: List[Dict]) -> float:
        """Calculate overall risk score"""
        risk_factors = [
            spoofing.spoofing_intensity if spoofing.detected else 0.0,
            wash_trading.wash_ratio if wash_trading.detected else 0.0,
            pump_dump.pump_magnitude if pump_dump.detected else 0.0,
            len(microstructure) * 0.1
        ]
        return min(max(np.mean(risk_factors), 0.0), 1.0)
    
    def _calculate_detection_confidence(self, spoofing: SpoofingDetection, wash_trading: WashTradingDetection,
                                      pump_dump: PumpDumpDetection) -> float:
        """Calculate detection confidence"""
        confidences = [
            spoofing.confidence,
            wash_trading.confidence,
            pump_dump.confidence
        ]
        return min(max(np.mean(confidences), 0.0), 1.0)
    
    def _get_fallback_detection(self) -> AnomalyDetection:
        """Return fallback detection results"""
        return AnomalyDetection(
            spoofing=SpoofingDetection(
                detected=bool(False),
                confidence=0.0,
                spoofing_type='none',
                affected_levels=[],
                spoofing_intensity=0.0,
                duration_seconds=0.0
            ),
            wash_trading=WashTradingDetection(
                detected=bool(False),
                confidence=0.0,
                wash_volume=0.0,
                wash_ratio=0.0,
                suspicious_patterns=[],
                regulatory_risk='low'
            ),
            pump_dump=PumpDumpDetection(
                detected=bool(False),
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
            confidence=0.5
        )


class SpoofingDetector:
    """Detects spoofing patterns in market data"""
    
    async def detect_spoofing(self, data: pd.DataFrame) -> SpoofingDetection:
        """Detect spoofing patterns"""
        try:
            if data.empty:
                return self._get_fallback_spoofing()
            
            # Simple spoofing detection logic
            volume_anomaly = self._detect_volume_anomaly(data)
            price_anomaly = self._detect_price_anomaly(data)
            
            detected = volume_anomaly or price_anomaly
            confidence = 0.3 if detected else 0.1
            
            return SpoofingDetection(
                detected=detected,
                confidence=confidence,
                spoofing_type='layering' if detected else 'none',
                affected_levels=[100.0, 101.0] if detected else [],
                spoofing_intensity=0.2 if detected else 0.0,
                duration_seconds=300.0 if detected else 0.0
            )
            
        except Exception as e:
            logger.error(f"Spoofing detection failed: {e}")
            return self._get_fallback_spoofing()
    
    def _detect_volume_anomaly(self, data: pd.DataFrame) -> bool:
        """Detect volume-based anomalies"""
        if 'volume' not in data.columns:
            return False
        
        volume = data['volume']
        if len(volume) < 10:
            return False
        
        mean_vol = volume.mean()
        std_vol = volume.std()
        
        # Detect sudden volume spikes
        recent_vol = volume.iloc[-5:].mean()
        return recent_vol > mean_vol + 3 * std_vol
    
    def _detect_price_anomaly(self, data: pd.DataFrame) -> bool:
        """Detect price-based anomalies"""
        if 'close' not in data.columns:
            return False
        
        prices = data['close']
        if len(prices) < 10:
            return False
        
        # Detect sudden price reversals
        recent_change = abs(prices.iloc[-1] - prices.iloc[-5]) / prices.iloc[-5]
        return recent_change > 0.05
    
    def _get_fallback_spoofing(self) -> SpoofingDetection:
        """Return fallback spoofing detection"""
        return SpoofingDetection(
            detected=False,
            confidence=0.0,
            spoofing_type='none',
            affected_levels=[],
            spoofing_intensity=0.0,
            duration_seconds=0.0
        )


class WashTradingDetector:
    """Detects wash trading patterns"""
    
    async def detect_wash_trading(self, data: pd.DataFrame) -> WashTradingDetection:
        """Detect wash trading patterns"""
        try:
            if data.empty:
                return self._get_fallback_wash_trading()
            
            # Simple wash trading detection
            volume_ratio = self._calculate_volume_ratio(data)
            suspicious = volume_ratio > 2.0
            
            return WashTradingDetection(
                detected=suspicious,
                confidence=0.4 if suspicious else 0.1,
                wash_volume=1000.0 if suspicious else 0.0,
                wash_ratio=volume_ratio if suspicious else 0.0,
                suspicious_patterns=['high_volume_low_price_change'] if suspicious else [],
                regulatory_risk='medium' if suspicious else 'low'
            )
            
        except Exception as e:
            logger.error(f"Wash trading detection failed: {e}")
            return self._get_fallback_wash_trading()
    
    def _calculate_volume_ratio(self, data: pd.DataFrame) -> float:
        """Calculate volume ratio for wash trading detection"""
        if 'volume' not in data.columns or len(data) < 5:
            return 0.0
        
        recent_volume = data['volume'].iloc[-5:].mean()
        historical_volume = data['volume'].iloc[:-5].mean() if len(data) > 5 else recent_volume
        
        if historical_volume == 0:
            return 0.0
        
        return recent_volume / historical_volume
    
    def _get_fallback_wash_trading(self) -> WashTradingDetection:
        """Return fallback wash trading detection"""
        return WashTradingDetection(
            detected=False,
            confidence=0.0,
            wash_volume=0.0,
            wash_ratio=0.0,
            suspicious_patterns=[],
            regulatory_risk='low'
        )


class PumpDumpDetector:
    """Detects pump and dump patterns"""
    
    async def detect_pump_dump(self, data: pd.DataFrame) -> PumpDumpDetection:
        """Detect pump and dump patterns"""
        try:
            if data.empty:
                return self._get_fallback_pump_dump()
            
            # Simple pump and dump detection
            price_change = self._calculate_price_change(data)
            volume_change = self._calculate_volume_change(data)
            
            detected = price_change > 0.1 and volume_change > 2.0
            
            return PumpDumpDetection(
                detected=detected,
                confidence=0.5 if detected else 0.1,
                pump_start=datetime.now() - timedelta(hours=2),
                dump_start=datetime.now() - timedelta(hours=1),
                pump_magnitude=price_change if detected else 0.0,
                dump_magnitude=price_change * 0.8 if detected else 0.0,
                volume_anomaly=volume_change if detected else 0.0
            )
            
        except Exception as e:
            logger.error(f"Pump and dump detection failed: {e}")
            return self._get_fallback_pump_dump()
    
    def _calculate_price_change(self, data: pd.DataFrame) -> float:
        """Calculate price change for pump dump detection"""
        if 'close' not in data.columns or len(data) < 2:
            return 0.0
        
        return abs(data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]
    
    def _calculate_volume_change(self, data: pd.DataFrame) -> float:
        """Calculate volume change for pump dump detection"""
        if 'volume' not in data.columns or len(data) < 2:
            return 0.0
        
        recent_volume = data['volume'].iloc[-5:].mean()
        historical_volume = data['volume'].iloc[:-5].mean() if len(data) > 5 else recent_volume
        
        if historical_volume == 0:
            return 0.0
        
        return recent_volume / historical_volume
    
    def _get_fallback_pump_dump(self) -> PumpDumpDetection:
        """Return fallback pump dump detection"""
        return PumpDumpDetection(
            detected=False,
            confidence=0.0,
            pump_start=datetime.now(),
            dump_start=datetime.now(),
            pump_magnitude=0.0,
            dump_magnitude=0.0,
            volume_anomaly=0.0
        )


class MicrostructureAnomalyDetector:
    """Detects microstructure anomalies"""
    
    async def detect_anomalies(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect microstructure anomalies"""
        try:
            if data.empty:
                return []
            
            # Simple microstructure anomaly detection
            anomalies = []
            
            # Check for bid-ask spread anomalies
            if 'high' in data.columns and 'low' in data.columns:
                spreads = data['high'] - data['low']
                mean_spread = spreads.mean()
                std_spread = spreads.std()
                
                for i, spread in enumerate(spreads):
                    if spread > mean_spread + 2 * std_spread:
                        anomalies.append({
                            'type': 'wide_spread',
                            'timestamp': data.index[i] if hasattr(data, 'index') else datetime.now(),
                            'severity': spread / mean_spread,
                            'description': 'Unusually wide bid-ask spread'
                        })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Microstructure anomaly detection failed: {e}")
            return []
