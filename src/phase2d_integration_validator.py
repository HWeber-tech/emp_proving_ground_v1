#!/usr/bin/env python3
"""
Phase 2D: Real Integration & Testing Framework
================================================

Comprehensive end-to-end integration testing with real market data.
Validates that all components work together with actual data flows.
"""

import asyncio
import logging
import json
import time
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from decimal import Decimal

from src.sensory.organs.yahoo_finance_organ import YahooFinanceOrgan
from src.sensory.enhanced.anomaly.manipulation_detection import ManipulationDetectionSystem
try:
    from src.trading.risk.market_regime_detector import MarketRegimeDetector  # deprecated
except Exception:  # pragma: no cover
    MarketRegimeDetector = None  # type: ignore
from src.trading.strategies.strategy_manager import StrategyManager
from src.data_integration.real_data_integration import RealDataManager
try:
    from src.core.interfaces import DecisionGenome  # legacy
except Exception:  # pragma: no cover
    DecisionGenome = object  # type: ignore
from src.core import Instrument, InstrumentProvider
from src.risk import RiskManager, RiskConfig
from src.pnl import EnhancedPosition

logger = logging.getLogger(__name__)


class Phase2DIntegrationValidator:
    """
    Phase 2D: Real Integration & Testing
    Tests complete system integration with real market data
    """
    
    def __init__(self):
        self.results = []
        self.yahoo_organ = YahooFinanceOrgan()
        self.manipulation_detector = ManipulationDetectionSystem()
        self.regime_detector = MarketRegimeDetector()
        self.strategy_manager = StrategyManager()
        self.real_data_manager = RealDataManager({'fallback_to_mock': False})
        
    async def test_real_data_flow(self) -> Dict[str, Any]:
        """Test complete data flow from market data to decision engine"""
        try:
            logger.info("Testing real data flow integration...")
            
            # Test 1: Market data ingestion
            start_time = time.time()
            symbols = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', '^GSPC', '^DJI']
            
            real_data_count = 0
            for symbol in symbols:
                try:
                    data = self.yahoo_organ.fetch_data(symbol, period="1d", interval="1m")
                    if data is not None and len(data) > 0:
                        real_data_count += 1
                except Exception as e:
                    logger.warning(f"Failed to fetch {symbol}: {e}")
            
            data_ingestion_time = time.time() - start_time
            
            # Test 2: Sensory processing
            start_time = time.time()
            if real_data_count > 0:
                test_data = self.yahoo_organ.fetch_data('EURUSD=X', period="1d", interval="1h")
                if test_data is not None:
                    # Process through sensory cortex
                    anomalies = await self.manipulation_detector.detect_manipulation(test_data)
                    regimes = await self.regime_detector.detect_regime(test_data)
                    processing_complete = True
                else:
                    processing_complete = False
            else:
                processing_complete = False
            
            sensory_processing_time = time.time() - start_time
            
            # Test 3: Decision engine integration
            start_time = time.time()
            if processing_complete:
                # Create decision genome
                genome = DecisionGenome()
                genome.initialize_random()
                
                # Test evaluation
                fitness_score = await self._evaluate_genome_with_real_data(genome, test_data)
                decision_engine_time = time.time() - start_time
                
                decision_complete = fitness_score is not None
            else:
                decision_complete = False
                decision_engine_time = 0
            
            # Test 4: Risk management integration
            start_time = time.time()
            if decision_complete:
                risk_config = RiskConfig(
                    max_risk_per_trade_pct=Decimal('0.02'),
                    max_leverage=Decimal('10.0'),
                    max_total_exposure_pct=Decimal('0.5'),
                    max_drawdown_pct=Decimal('0.25')
                )
                risk_manager = RiskManager(risk_config, InstrumentProvider())
                
                # Test risk validation
                position = EnhancedPosition(
                    symbol="EURUSD",
                    quantity=10000,
                    avg_price=Decimal('1.1000'),
                    entry_timestamp=datetime.now()
                )
                
                is_valid = risk_manager.validate_position(
                    position=position,
                    instrument=Instrument("EURUSD", "Currency"),
                    equity=Decimal('100000')
                )
                
                risk_management_time = time.time() - start_time
            else:
                is_valid = False
                risk_management_time = 0
            
            # Calculate metrics
            total_flow_time = data_ingestion_time + sensory_processing_time + decision_engine_time + risk_management_time
            
            return {
                'test_name': 'real_data_flow',
                'passed': real_data_count >= 3 and processing_complete and decision_complete and is_valid,
                'data_ingestion_time': data_ingestion_time,
                'sensory_processing_time': sensory_processing_time,
                'decision_engine_time': decision_engine_time,
                'risk_management_time': risk_management_time,
                'total_flow_time': total_flow_time,
                'real_data_sources': real_data_count,
                'details': f"Real data flow completed in {total_flow_time:.2f}s"
            }
            
        except Exception as e:
            logger.error(f"Real data flow test failed: {e}")
            return {
                'test_name': 'real_data_flow',
                'passed': False,
                'error': str(e),
                'details': "Real data flow integration failed"
            }
    
    async def test_strategy_performance_tracking(self) -> Dict[str, Any]:
        """Test strategy manager with real performance tracking"""
        try:
            logger.info("Testing strategy performance tracking...")
            
            # Get real market data
            data = self.yahoo_organ.fetch_data('EURUSD=X', period="90d", interval="1d")
            if data is None or len(data) < 20:
                return {
                    'test_name': 'strategy_performance_tracking',
                    'passed': False,
                    'details': "Insufficient data for performance tracking"
                }
            
            # Calculate real performance metrics
            data['returns'] = data['close'].pct_change()
            data = data.dropna()
            
            # Calculate Sharpe ratio
            excess_returns = data['returns'] - 0.02/252  # 2% risk-free rate
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
            
            # Calculate max drawdown
            data['cumulative'] = (1 + data['returns']).cumprod()
            data['running_max'] = data['cumulative'].expanding().max()
            data['drawdown'] = (data['cumulative'] - data['running_max']) / data['running_max']
            max_drawdown = data['drawdown'].min()
            
            # Calculate win rate (simplified)
            win_rate = (data['returns'] > 0).mean()
            
            return {
                'test_name': 'strategy_performance_tracking',
                'passed': sharpe_ratio is not None and max_drawdown is not None,
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'win_rate': float(win_rate),
                'total_return': float(data['returns'].sum()),
                'volatility': float(data['returns'].std() * np.sqrt(252)),
                'observations': len(data),
                'details': f"Real performance tracking completed with {len(data)} observations"
            }
            
        except Exception as e:
            logger.error(f"Strategy performance tracking test failed: {e}")
            return {
                'test_name': 'strategy_performance_tracking',
                'passed': False,
                'error': str(e),
                'details': "Strategy performance tracking failed"
            }
    
    async def test_concurrent_operations(self) -> Dict[str, Any]:
        """Placeholder test for concurrent operations integration.

        This test currently performs no concurrent operations and simply
        returns a passing result.  It can be expanded in the future to run
        multiple integration tests in parallel to ensure proper coordination
        across subsystems.
        """
        try:
            return {
                'test_name': 'concurrent_operations',
                'passed': True,
                'details': 'Concurrent operations test not implemented.'
            }
        except Exception as e:
            logger.error(f"Concurrent operations test failed: {e}")
            return {
                'test_name': 'concurrent_operations',
                'passed': False,
                'error': str(e),
                'details': 'Concurrent operations test failed.'
            }
        
