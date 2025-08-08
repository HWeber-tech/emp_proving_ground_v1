"""
Real Sensory Organ Implementation
Replaces the stub with functional multi-timeframe technical analysis
"""

import asyncio
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import sqlite3
import json

from ..models import SensoryReading, TechnicalSignal
from ..indicators import (
    RSIIndicator, MACDIndicator, BollingerBandsIndicator,
    VolumeIndicator, MomentumIndicator, SupportResistanceIndicator
)
from ...config.sensory_config import SensoryConfig

logger = logging.getLogger(__name__)

@dataclass
class MarketContext:
    """Market context for sensory processing"""
    symbol: str
    timeframe: str
    price_data: pd.DataFrame
    volume_data: pd.DataFrame
    timestamp: datetime

from src.sensory.real_sensory_organ import RealSensoryOrgan  # re-export canonical implementation
    
    def _init_database(self) -> None:
        """Initialize sensory database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensory_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timeframe TEXT,
                timestamp DATETIME,
                overall_sentiment TEXT,
                confidence_score REAL,
                technical_signals TEXT,
                market_context TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS calibration_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timeframe TEXT,
                calibration_date DATETIME,
                data_points INTEGER,
                parameters TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def process(self, market_data: Dict) -> SensoryReading:
        """
        Process market data and return sensory reading
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Sensory reading with technical analysis
        """
        try:
            # Update data buffers
            await self._update_data_buffers(market_data)
            
            # Ensure we have enough data
            if not self._has_sufficient_data():
                return self._create_empty_reading()
            
            # Calculate technical indicators
            technical_signals = await self._calculate_technical_signals()
            
            # Determine overall sentiment
            overall_sentiment = self._determine_sentiment(technical_signals)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(technical_signals)
            
            # Create reading
            reading = SensoryReading(
                symbol=self.symbol,
                timeframe=self.config.primary_timeframe,
                timestamp=datetime.now(),
                overall_sentiment=overall_sentiment,
                confidence_score=confidence_score,
                technical_signals=technical_signals,
                market_context=self._get_market_context()
            )
            
            # Store reading
            self._store_reading(reading)
            
            logger.info(f"Processed sensory reading for {self.symbol}: {overall_sentiment} ({confidence_score:.2f})")
            return reading
            
        except Exception as e:
            logger.error(f"Error processing sensory data: {e}")
            return self._create_empty_reading()
    
    async def _update_data_buffers(self, market_data: Dict) -> None:
        """Update price and volume data buffers"""
        try:
            timeframe = market_data.get('timeframe', 'M15')
            
            # Create DataFrame from market data
            df = pd.DataFrame([{
                'timestamp': market_data.get('timestamp', datetime.now()),
                'open': market_data.get('open', 0),
                'high': market_data.get('high', 0),
                'low': market_data.get('low', 0),
                'close': market_data.get('close', 0),
                'volume': market_data.get('volume', 0)
            }])
            
            # Update buffers
            if timeframe not in self.price_buffers:
                self.price_buffers[timeframe] = df
            else:
                self.price_buffers[timeframe] = pd.concat([
                    self.price_buffers[timeframe], df
                ]).tail(self.max_buffer_size)
                
            if timeframe not in self.volume_buffers:
                self.volume_buffers[timeframe] = df[['timestamp', 'volume']]
            else:
                self.volume_buffers[timeframe] = pd.concat([
                    self.volume_buffers[timeframe], df[['timestamp', 'volume']]
                ]).tail(self.max_buffer_size)
                
        except Exception as e:
            logger.error(f"Error updating data buffers: {e}")
    
    def _has_sufficient_data(self) -> bool:
        """Check if we have sufficient data for analysis"""
        for timeframe in self.timeframes:
            if timeframe not in self.price_buffers:
                return False
            if len(self.price_buffers[timeframe]) < 50:
                return False
        return True
    
    async def _calculate_technical_signals(self) -> List[TechnicalSignal]:
        """Calculate technical indicators and signals"""
        signals = []
        
        for timeframe in self.timeframes:
            if timeframe not in self.price_buffers:
                continue
                
            df = self.price_buffers[timeframe]
            
            # Calculate indicators
            for name, indicator in self.indicators.items():
                try:
                    result = indicator.calculate(df)
                    if result:
                        signal = TechnicalSignal(
                            indicator=name,
                            timeframe=timeframe,
                            value=result.get('value', 0),
                            signal_type=result.get('signal', 'neutral'),
                            strength=result.get('strength', 0.5),
                            timestamp=datetime.now()
                        )
                        signals.append(signal)
                except Exception as e:
                    logger.warning(f"Error calculating {name} for {timeframe}: {e}")
        
        return signals
    
    def _determine_sentiment(self, signals: List[TechnicalSignal]) -> str:
        """Determine overall market sentiment from signals"""
        if not signals:
            return 'neutral'
        
        bullish_count = sum(1 for s in signals if s.signal_type == 'bullish')
        bearish_count = sum(1 for s in signals if s.signal_type == 'bearish')
        
        if bullish_count > bearish_count:
            return 'bullish'
        elif bearish_count > bullish_count:
            return 'bearish'
        else:
            return 'neutral'
    
    def _calculate_confidence(self, signals: List[TechnicalSignal]) -> float:
        """Calculate confidence score based on signal strength and consistency"""
        if not signals:
            return 0.0
        
        avg_strength = np.mean([s.strength for s in signals])
        return min(1.0, avg_strength)
    
    def _get_market_context(self) -> Dict:
        """Get current market context"""
        return {
            'symbol': self.symbol,
            'timeframes': list(self.price_buffers.keys()),
            'data_points': sum(len(df) for df in self.price_buffers.values()),
            'last_update': datetime.now().isoformat()
        }
    
    def _create_empty_reading(self) -> SensoryReading:
        """Create empty sensory reading for fallback"""
        return SensoryReading(
            symbol=self.symbol,
            timeframe=self.config.primary_timeframe,
            timestamp=datetime.now(),
            overall_sentiment='neutral',
            confidence_score=0.0,
            technical_signals=[],
            market_context={}
        )
    
    def _store_reading(self, reading: SensoryReading) -> None:
        """Store sensory reading in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO sensory_readings 
                (symbol, timeframe, timestamp, overall_sentiment, confidence_score, technical_signals, market_context)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                reading.symbol,
                reading.timeframe,
                reading.timestamp,
                reading.overall_sentiment,
                reading.confidence_score,
                json.dumps([s.__dict__ for s in reading.technical_signals]),
                json.dumps(reading.market_context)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing reading: {e}")
    
    def calibrate(self, historical_data: List[Dict]) -> None:
        """
        Calibrate sensor based on historical data
        
        Args:
            historical_data: List of historical market data
        """
        try:
            logger.info(f"Calibrating sensor with {len(historical_data)} data points")
            
            # Process historical data
            for data in historical_data:
                asyncio.run(self._update_data_buffers(data))
            
            # Store calibration data
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for timeframe in self.timeframes:
                if timeframe in self.price_buffers:
                    cursor.execute('''
                        INSERT INTO calibration_data 
                        (symbol, timeframe, calibration_date, data_points, parameters)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        self.symbol,
                        timeframe,
                        datetime.now(),
                        len(self.price_buffers[timeframe]),
                        json.dumps(self.config.__dict__)
                    ))
            
            conn.commit()
            conn.close()
            
            self.calibration_data = {
                'calibration_date': datetime.now(),
                'data_points': len(historical_data),
                'timeframes': list(self.price_buffers.keys())
            }
            
            logger.info("Sensor calibration completed")
            
        except Exception as e:
            logger.error(f"Error calibrating sensor: {e}")
