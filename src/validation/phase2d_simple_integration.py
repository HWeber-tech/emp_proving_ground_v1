#!/usr/bin/env python3
# ruff: noqa: I001
"""
Phase 2D: Simple Real Integration Test
======================================

Simplified but complete integration testing with real market data.
Validates end-to-end functionality without complex dependencies.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from decimal import Decimal
from typing import List, Optional, TYPE_CHECKING, Protocol, runtime_checkable, Any, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # type-only imports to keep hints without creating runtime edges
    from src.core.interfaces import DecisionGenome

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@runtime_checkable
class MarketDataProvider(Protocol):
    """Adapter for market data fetching used by this validator."""
    def fetch_data(self, symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        ...


@runtime_checkable
class RegimeDetector(Protocol):
    """Adapter for market regime detection used by this validator."""
    async def detect_regime(self, data: pd.DataFrame) -> Optional[Any]:
        ...


@runtime_checkable
class InstrumentRegistry(Protocol):
    """Adapter for instrument resolution passed to risk manager (no methods used here)."""
    ...


@runtime_checkable
class ManipulationDetector(Protocol):
    """Adapter for manipulation/anomaly detection used by this validator."""
    async def detect_manipulation(self, data: pd.DataFrame) -> Optional[Sequence[Any]]:
        ...


def _make_market_data_provider() -> MarketDataProvider:
    """Factory: lazily import and construct the current market data provider."""
    from src.sensory.organs.yahoo_finance_organ import YahooFinanceOrgan
    return YahooFinanceOrgan()


def _make_regime_detector() -> RegimeDetector:
    """Factory: lazily import and construct the current regime detector."""
    try:
        from src.trading.risk.market_regime_detector import MarketRegimeDetector  # deprecated
    except Exception:  # pragma: no cover
        MarketRegimeDetector = None  # type: ignore
    return MarketRegimeDetector()  # type: ignore[operator]


def _make_instrument_registry() -> InstrumentRegistry:
    """Factory: lazily import and construct the current instrument registry."""
    from src.core import InstrumentProvider  # type: ignore[attr-defined]
    return InstrumentProvider()


def _make_manipulation_detector() -> ManipulationDetector:
    """Factory: lazily import and construct the current manipulation detector."""
    from src.sensory.enhanced.anomaly.manipulation_detection import ManipulationDetectionSystem
    return ManipulationDetectionSystem()


class SimplePhase2DValidator:
    """Simplified Phase 2D validator for real integration testing"""
    
    def __init__(self):
        # Localized factories to reduce cross-domain edges (preserve lazy imports)
        self._data_provider: MarketDataProvider = _make_market_data_provider()
        self._manipulation_detector: ManipulationDetector = _make_manipulation_detector()
        self._regime_detector: RegimeDetector = _make_regime_detector()
        self._instrument_registry: InstrumentRegistry = _make_instrument_registry()
        
    async def test_real_data_integration(self) -> dict:
        """Test real market data integration"""
        try:
            logger.info("Testing real market data integration...")
            
            # Test 1: Real data fetching
            start_time = time.time()
            symbols = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', '^GSPC']
            
            real_data_results = []
            for symbol in symbols:
                try:
                    data = self._data_provider.fetch_data(symbol, period="7d", interval="1h")
                    if data is not None and len(data) > 10:
                        real_data_results.append({
                            'symbol': symbol,
                            'data_points': len(data),
                            'success': True
                        })
                    else:
                        real_data_results.append({
                            'symbol': symbol,
                            'data_points': 0,
                            'success': False
                        })
                except Exception as e:
                    real_data_results.append({
                        'symbol': symbol,
                        'data_points': 0,
                        'success': False,
                        'error': str(e)
                    })
            
            data_fetch_time = time.time() - start_time
            
            # Test 2: Sensory processing
            start_time = time.time()
            if any(r['success'] for r in real_data_results):
                test_data = self._data_provider.fetch_data('EURUSD=X', period="30d", interval="1d")
                if test_data is not None:
                    anomalies = await self._manipulation_detector.detect_manipulation(test_data)
                    regime = await self._regime_detector.detect_regime(test_data)
                    
                    sensory_processing_time = time.time() - start_time
                    
                    return {
                        'test_name': 'real_data_integration',
                        'passed': True,
                        'data_fetch_time': data_fetch_time,
                        'sensory_processing_time': sensory_processing_time,
                        'real_data_sources': sum(1 for r in real_data_results if r['success']),
                        'anomalies_detected': len(anomalies) if anomalies else 0,
                        'regime_detected': regime.regime.value if regime else 'UNKNOWN',
                        'details': f"Successfully processed real data from {sum(1 for r in real_data_results if r['success'])} sources"
                    }
            
            return {
                'test_name': 'real_data_integration',
                'passed': False,
                'data_fetch_time': data_fetch_time,
                'real_data_sources': sum(1 for r in real_data_results if r['success']),
                'details': "Failed to process real market data"
            }
            
        except Exception as e:
            return {
                'test_name': 'real_data_integration',
                'passed': False,
                'error': str(e),
                'details': "Real data integration test failed"
            }
    
    async def test_performance_metrics(self) -> dict:
        """Test real performance metrics calculation"""
        try:
            logger.info("Testing real performance metrics...")
            
            # Get real market data for performance testing
            data = self._data_provider.fetch_data('^GSPC', period="365d", interval="1d")
            if data is None or len(data) < 20:
                return {
                    'test_name': 'performance_metrics',
                    'passed': False,
                    'details': "Insufficient data for performance testing"
                }
            
            # Calculate real performance metrics
            data['returns'] = data['close'].pct_change()
            data = data.dropna()
            
            # Sharpe ratio calculation
            excess_returns = data['returns'] - 0.02/252  # 2% risk-free rate
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
            
            # Max drawdown calculation
            data['cumulative'] = (1 + data['returns']).cumprod()
            data['running_max'] = data['cumulative'].expanding().max()
            data['drawdown'] = (data['cumulative'] - data['running_max']) / data['running_max']
            max_drawdown = data['drawdown'].min()
            
            # Win rate
            win_rate = (data['returns'] > 0).mean()
            
            # Volatility
            volatility = data['returns'].std() * np.sqrt(252)
            
            return {
                'test_name': 'performance_metrics',
                'passed': True,
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(abs(max_drawdown)),
                'win_rate': float(win_rate),
                'volatility': float(volatility),
                'total_return': float(data['returns'].sum()),
                'observations': len(data),
                'details': f"Calculated real performance metrics from {len(data)} observations"
            }
            
        except Exception as e:
            return {
                'test_name': 'performance_metrics',
                'passed': False,
                'error': str(e),
                'details': "Performance metrics test failed"
            }
    
    async def test_risk_management_integration(self) -> dict:
        """Test risk management with real data"""
        try:
            logger.info("Testing risk management integration...")
            
            from src.risk import RiskConfig, RiskManager  # type: ignore[attr-defined]
            from src.core import Instrument
            
            # Test risk configuration
            risk_config = RiskConfig(
                max_risk_per_trade_pct=Decimal('0.02'),
                max_leverage=Decimal('10.0'),
                max_total_exposure_pct=Decimal('0.5'),
                max_drawdown_pct=Decimal('0.03'),  # 3% max drawdown target
                min_position_size=1000,
                max_position_size=100000
            )
            
            risk_manager = RiskManager(risk_config, self._instrument_registry)
            
            # Test position validation
            position = {
                'symbol': 'EURUSD',
                'quantity': 10000,
                'avg_price': Decimal('1.1000'),
                'entry_timestamp': datetime.now()
            }
            
            instrument = Instrument("EURUSD", "Currency")
            equity = Decimal('100000')
            
            is_valid = risk_manager.validate_position(
                position=position,
                instrument=instrument,
                equity=equity
            )
            
            return {
                'test_name': 'risk_management_integration',
                'passed': is_valid,
                'max_drawdown_limit': 0.03,
                'risk_per_trade': 0.02,
                'details': "Risk management integration test completed"
            }
            
        except Exception as e:
            return {
                'test_name': 'risk_management_integration',
                'passed': False,
                'error': str(e),
                'details': "Risk management integration test failed"
            }
    
    async def test_concurrent_operations(self) -> dict:
        """Test concurrent operations capability"""
        try:
            logger.info("Testing concurrent operations...")
            
            symbols = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'NZDUSD=X']
            
            start_time = time.time()
            
            # Concurrent data fetching
            tasks = []
            for symbol in symbols:
                task = asyncio.create_task(
                    self._fetch_symbol_async(symbol)
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            concurrent_time = time.time() - start_time
            
            successful_ops = sum(1 for r in results if isinstance(r, dict) and r.get('success', False))
            throughput = successful_ops / concurrent_time if concurrent_time > 0 else 0
            
            return {
                'test_name': 'concurrent_operations',
                'passed': successful_ops >= 3 and throughput >= 1.0,
                'successful_operations': successful_ops,
                'total_operations': len(symbols),
                'concurrent_time': concurrent_time,
                'throughput': throughput,
                'details': f"Completed {successful_ops}/{len(symbols)} concurrent operations in {concurrent_time:.2f}s"
            }
            
        except Exception as e:
            return {
                'test_name': 'concurrent_operations',
                'passed': False,
                'error': str(e),
                'details': "Concurrent operations test failed"
            }
    
    async def _fetch_symbol_async(self, symbol: str) -> dict:
        """Helper for async symbol fetching"""
        try:
            data = self._data_provider.fetch_data(symbol, period="1d", interval="1h")
            if data is not None and len(data) > 0:
                return {'success': True, 'symbol': symbol, 'data_points': len(data)}
            return {'success': False, 'symbol': symbol}
        except Exception:
            return {'success': False, 'symbol': symbol}
    
    async def _evaluate_genome_with_real_data(self, genome: DecisionGenome, data: pd.DataFrame) -> Optional[float]:
        """Simple fitness evaluation with real data"""
        try:
            if len(data) < 10:
                return None
            
            returns = data['close'].pct_change().dropna()
            if len(returns) == 0:
                return None
            
            # Simple Sharpe-like metric
            mean_return = returns.mean()
            volatility = returns.std()
            
            if volatility == 0:
                return None
            
            fitness = mean_return / volatility
            return max(0, fitness)
            
        except Exception:
            return None
    
    async def run_phase2d_validation(self) -> dict:
        """Run complete Phase 2D validation"""
        logger.info("Starting Phase 2D: Real Integration & Testing...")
        
        # Run all tests
        tests = [
            self.test_real_data_integration(),
            self.test_performance_metrics(),
            self.test_risk_management_integration(),
            self.test_concurrent_operations()
        ]
        
        results = await asyncio.gather(*tests)
        
        # Calculate summary
        passed = sum(1 for r in results if r.get('passed', False))
        total = len(results)
        
        # Validate against real success criteria
        real_criteria = self._validate_real_success_criteria(results)
        
        # Create comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'phase': '2D',
            'title': 'Real Integration & Testing',
            'total_tests': total,
            'passed_tests': passed,
            'failed_tests': total - passed,
            'success_rate': passed / total if total > 0 else 0,
            'test_results': results,
            'real_success_criteria': real_criteria,
            'status': 'PASSED' if passed >= 3 and real_criteria['all_passed'] else 'FAILED',
            'summary': {
                'message': f"{passed}/{total} integration tests passed ({passed/total:.1%} success rate)",
                'real_criteria_status': 'ALL MET' if real_criteria['all_passed'] else 'SOME FAILED'
            }
        }
        
        # Save results
        with open('phase2d_simple_integration_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _validate_real_success_criteria(self, results: List[dict]) -> dict:
        """Validate against real success criteria"""
        criteria: dict[str, Any] = {
            'response_time': {'target': 1.0, 'actual': None, 'passed': False},
            'sharpe_ratio': {'target': 1.5, 'actual': None, 'passed': False},
            'max_drawdown': {'target': 0.03, 'actual': None, 'passed': False},
            'concurrent_ops': {'target': 5.0, 'actual': None, 'passed': False},
            'uptime': {'target': 99.9, 'actual': None, 'passed': False}
        }
        
        # Extract actual values
        for result in results:
            test_name = result.get('test_name')
            
            if test_name == 'real_data_integration':
                criteria['response_time']['actual'] = result.get('data_fetch_time', 999)
                
            elif test_name == 'performance_metrics':
                criteria['sharpe_ratio']['actual'] = result.get('sharpe_ratio', 0)
                criteria['max_drawdown']['actual'] = abs(result.get('max_drawdown', 999))
                
            elif test_name == 'concurrent_operations':
                criteria['concurrent_ops']['actual'] = result.get('throughput', 0)
        
        # Validate each criterion
        all_passed = True
        for key, criterion in criteria.items():
            if criterion['actual'] is not None:
                if key == 'max_drawdown':
                    criterion['passed'] = criterion['actual'] <= criterion['target']
                else:
                    criterion['passed'] = criterion['actual'] >= criterion['target']
                
                if not criterion['passed']:
                    all_passed = False
        
        criteria['all_passed'] = all_passed
        return criteria
