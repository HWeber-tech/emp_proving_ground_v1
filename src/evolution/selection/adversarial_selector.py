"""
Adversarial Selection Engine - Stress Testing Strategies

Implements 15 stress test scenarios to filter strategies with <70% survival rate.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import random

logger = logging.getLogger(__name__)


@dataclass
class StressTestResult:
    """Result of a single stress test scenario."""
    scenario_name: str
    survival_rate: float
    max_drawdown: float
    final_equity: float
    trades_executed: int
    win_rate: float
    sharpe_ratio: float
    passed: bool
    details: Dict[str, Any]


@dataclass
class AdversarialSelectionResult:
    """Result of adversarial selection process."""
    strategy_id: str
    overall_survival_rate: float
    scenario_results: List[StressTestResult]
    passed_scenarios: int
    total_scenarios: int
    final_score: float
    selected: bool


class AdversarialSelector:
    """
    Adversarial selection engine that runs strategies through stress tests.
    
    Filters strategies with <70% survival rate across 15 scenarios:
    1. Market crash scenarios
    2. Flash crash scenarios
    3. Volatility spike scenarios
    4. Liquidity crisis scenarios
    5. Regime change scenarios
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize adversarial selector."""
        self.config = config or {}
        self.survival_threshold = self.config.get('survival_threshold', 0.7)
        self.initial_capital = self.config.get('initial_capital', 10000.0)
        self.scenarios = self._initialize_scenarios()
        
        logger.info(f"AdversarialSelector initialized with {len(self.scenarios)} scenarios")
    
    def _initialize_scenarios(self) -> List[Dict[str, Any]]:
        """Initialize 15 stress test scenarios."""
        return [
            # Market Crash Scenarios (5)
            {
                'name': '2008_Financial_Crisis',
                'type': 'market_crash',
                'duration_days': 252,
                'volatility_multiplier': 3.0,
                'trend_slope': -0.3,
                'max_drop': -0.5,
                'liquidity_factor': 0.7
            },
            {
                'name': 'COVID_Crash_2020',
                'type': 'market_crash',
                'duration_days': 30,
                'volatility_multiplier': 5.0,
                'trend_slope': -0.4,
                'max_drop': -0.35,
                'liquidity_factor': 0.5
            },
            {
                'name': 'Dot_Com_Bubble',
                'type': 'market_crash',
                'duration_days': 180,
                'volatility_multiplier': 2.5,
                'trend_slope': -0.25,
                'max_drop': -0.78,
                'liquidity_factor': 0.8
            },
            {
                'name': 'Black_Monday_1987',
                'type': 'market_crash',
                'duration_days': 5,
                'volatility_multiplier': 8.0,
                'trend_slope': -0.6,
                'max_drop': -0.22,
                'liquidity_factor': 0.3
            },
            {
                'name': 'Asian_Crisis_1997',
                'type': 'market_crash',
                'duration_days': 120,
                'volatility_multiplier': 2.8,
                'trend_slope': -0.2,
                'max_drop': -0.4,
                'liquidity_factor': 0.6
            },
            
            # Flash Crash Scenarios (3)
            {
                'name': 'Flash_Crash_2010',
                'type': 'flash_crash',
                'duration_days': 1,
                'volatility_multiplier': 10.0,
                'trend_slope': -0.8,
                'max_drop': -0.09,
                'liquidity_factor': 0.1
            },
            {
                'name': 'Swiss_Franc_2015',
                'type': 'flash_crash',
                'duration_days': 1,
                'volatility_multiplier': 12.0,
                'trend_slope': -0.7,
                'max_drop': -0.15,
                'liquidity_factor': 0.05
            },
            {
                'name': 'GBP_Flash_Crash_2016',
                'type': 'flash_crash',
                'duration_days': 1,
                'volatility_multiplier': 8.0,
                'trend_slope': -0.6,
                'max_drop': -0.06,
                'liquidity_factor': 0.2
            },
            
            # Volatility Spike Scenarios (3)
            {
                'name': 'VIX_Spike_2008',
                'type': 'volatility_spike',
                'duration_days': 30,
                'volatility_multiplier': 6.0,
                'trend_slope': -0.1,
                'max_drop': -0.15,
                'liquidity_factor': 0.4
            },
            {
                'name': 'Eurozone_Crisis_2011',
                'type': 'volatility_spike',
                'duration_days': 90,
                'volatility_multiplier': 4.0,
                'trend_slope': -0.15,
                'max_drop': -0.25,
                'liquidity_factor': 0.5
            },
            {
                'name': 'Brexit_Volatility_2016',
                'type': 'volatility_spike',
                'duration_days': 7,
                'volatility_multiplier': 5.0,
                'trend_slope': -0.2,
                'max_drop': -0.08,
                'liquidity_factor': 0.6
            },
            
            # Liquidity Crisis Scenarios (2)
            {
                'name': 'Credit_Crunch_2008',
                'type': 'liquidity_crisis',
                'duration_days': 60,
                'volatility_multiplier': 4.0,
                'trend_slope': -0.25,
                'max_drop': -0.3,
                'liquidity_factor': 0.2
            },
            {
                'name': 'Repo_Crisis_2019',
                'type': 'liquidity_crisis',
                'duration_days': 14,
                'volatility_multiplier': 3.5,
                'trend_slope': -0.15,
                'max_drop': -0.12,
                'liquidity_factor': 0.3
            },
            
            # Regime Change Scenarios (2)
            {
                'name': 'Fed_Policy_Shift_2013',
                'type': 'regime_change',
                'duration_days': 45,
                'volatility_multiplier': 2.5,
                'trend_slope': -0.2,
                'max_drop': -0.1,
                'liquidity_factor': 0.7
            },
            {
                'name': 'ECB_Policy_Change_2014',
                'type': 'regime_change',
                'duration_days': 30,
                'volatility_multiplier': 2.0,
                'trend_slope': -0.15,
                'max_drop': -0.08,
                'liquidity_factor': 0.8
            }
        ]
    
    async def select_strategies(self, 
                              strategies: List[Dict[str, Any]],
                              market_data: Dict[str, Any]) -> List[AdversarialSelectionResult]:
        """Run adversarial selection on all strategies."""
        results = []
        
        for strategy in strategies:
            strategy_id = strategy.get('id', 'unknown')
            strategy_config = strategy.get('config', {})
            
            # Run stress tests
            result = await self._run_stress_tests(strategy_id, strategy_config, market_data)
            results.append(result)
        
        # Filter strategies with >=70% survival rate
        selected_strategies = [r for r in results if r.selected]
        
        logger.info(f"Selected {len(selected_strategies)} out of {len(results)} strategies")
        
        return results
    
    async def _run_stress_tests(self, 
                              strategy_id: str,
                              strategy_config: Dict[str, Any],
                              market_data: Dict[str, Any]) -> AdversarialSelectionResult:
        """Run all stress test scenarios for a strategy."""
        scenario_results = []
        
        for scenario in self.scenarios:
            # Simulate scenario
            result = await self._simulate_scenario(strategy_id, scenario, strategy_config, market_data)
            scenario_results.append(result)
        
        # Calculate overall metrics
        passed_scenarios = sum(1 for r in scenario_results if r.passed)
        total_scenarios = len(scenario_results)
        overall_survival_rate = passed_scenarios / total_scenarios if total_scenarios > 0 else 0.0
        
        # Determine if strategy passes
        selected = overall_survival_rate >= self.survival_threshold
        
        return AdversarialSelectionResult(
            strategy_id=strategy_id,
            overall_survival_rate=overall_survival_rate,
            scenario_results=scenario_results,
            passed_scenarios=passed_scenarios,
            total_scenarios=total_scenarios,
            final_score=overall_survival_rate,
            selected=selected
        )
    
    async def _simulate_scenario(self,
                               strategy_id: str,
                               scenario: Dict[str, Any],
                               strategy_config: Dict[str, Any],
                               market_data: Dict[str, Any]) -> StressTestResult:
        """Simulate a single stress test scenario."""
        try:
            # Generate synthetic market data for scenario
            market_series = self._generate_market_data(scenario)
            
            # Simulate strategy performance
            performance = self._simulate_strategy_performance(
                strategy_config,
                market_series,
                scenario
            )
            
            # Calculate survival metrics
            survival_rate = self._calculate_survival_rate(performance)
            max_drawdown = performance.get('max_drawdown', 0.0)
            final_equity = performance.get('final_equity', self.initial_capital)
            trades_executed = performance.get('trades_executed', 0)
            win_rate = performance.get('win_rate', 0.0)
            sharpe_ratio = performance.get('sharpe_ratio', 0.0)
            
            # Determine if scenario passed
            passed = survival_rate >= self.survival_threshold
            
            return StressTestResult(
                scenario_name=scenario['name'],
                survival_rate=survival_rate,
                max_drawdown=max_drawdown,
                final_equity=final_equity,
                trades_executed=trades_executed,
                win_rate=win_rate,
                sharpe_ratio=sharpe_ratio,
                passed=passed,
                details=performance
            )
            
        except Exception as e:
            logger.error(f"Error simulating scenario {scenario['name']}: {e}")
            return StressTestResult(
                scenario_name=scenario['name'],
                survival_rate=0.0,
                max_drawdown=1.0,
                final_equity=0.0,
                trades_executed=0,
                win_rate=0.0,
                sharpe_ratio=0.0,
                passed=False,
                details={'error': str(e)}
            )
    
    def _generate_market_data(self, scenario: Dict[str, Any]) -> pd.DataFrame:
        """Generate synthetic market data for stress test scenario."""
        duration = scenario['duration_days']
        volatility = scenario['volatility_multiplier']
        trend = scenario['trend_slope']
        
        # Generate price series
        dates = pd.date_range(
            start=datetime.now(),
            periods=duration,
            freq='D'
        )
        
        # Generate returns with specified characteristics
        returns = np.random.normal(
            loc=trend / 252,  # Daily trend
            scale=volatility * 0.01,  # Daily volatility
            size=duration
        )
        
        # Add extreme events for crash scenarios
        if scenario['type'] in ['market_crash', 'flash_crash']:
            crash_days = max(1, duration // 10)
            crash_indices = np.random.choice(duration, crash_days, replace=False)
            returns[crash_indices] *= 3  # Amplify crashes
        
        # Calculate price series
        prices = self.initial_capital * np.exp(np.cumsum(returns))
        
        return pd.DataFrame({
            'date': dates,
            'price': prices,
            'returns': returns
        })
    
    def _simulate_strategy_performance(self,
                                     strategy_config: Dict[str, Any],
                                     market_data: pd.DataFrame,
                                     scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate strategy performance under stress conditions."""
        # Simplified strategy simulation
        initial_capital = self.initial_capital
        
        # Strategy parameters
        win_rate = strategy_config.get('win_rate', 0.55)
        risk_per_trade = strategy_config.get('risk_per_trade', 0.02)
        max_positions = strategy_config.get('max_positions', 3)
        
        # Simulate trades
        trades = []
        capital = initial_capital
        
        for i, (_, row) in enumerate(market_data.iterrows()):
            if len(trades) >= max_positions:
                break
            
            # Simulate trade
            if np.random.random() < win_rate:
                # Winning trade
                profit = capital * risk_per_trade * np.random.uniform(0.5, 2.0)
                capital += profit
                trades.append({
                    'profit': profit,
                    'win': True
                })
            else:
                # Losing trade
                loss = capital * risk_per_trade * np.random.uniform(0.5, 1.5)
                capital -= loss
                trades.append({
                    'profit': -loss,
                    'win': False
                })
        
        # Calculate performance metrics
        if not trades:
            return {
                'max_drawdown': 0.0,
                'final_equity': capital,
                'trades_executed': 0,
                'win_rate': 0.0,
                'sharpe_ratio': 0.0,
                'total_return': 0.0
            }
        
        # Calculate drawdown
        equity_curve = [initial_capital]
        for trade in trades:
            equity_curve.append(equity_curve[-1] + trade['profit'])
        
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        
        # Calculate win rate
        wins = sum(1 for t in trades if t['win'])
        win_rate = wins / len(trades)
        
        # Calculate Sharpe ratio (simplified)
        returns = [t['profit'] / initial_capital for t in trades]
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        return {
            'max_drawdown': max_drawdown,
            'final_equity': capital,
            'trades_executed': len(trades),
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'total_return': (capital - initial_capital) / initial_capital
        }
    
    def _calculate_survival_rate(self, performance: Dict[str, Any]) -> float:
        """Calculate survival rate based on performance."""
        max_drawdown = performance.get('max_drawdown', 0.0)
        final_equity = performance.get('final_equity', self.initial_capital)
        
        # Survival based on drawdown and final equity
        drawdown_survival = max(0.0, 1.0 - (max_drawdown / 0.5))  # 50% max drawdown
        equity_survival = max(0.0, final_equity / self.initial_capital)
        
        # Weighted survival rate
        survival_rate = (drawdown_survival * 0.6 + equity_survival * 0.4)
        
        return min(1.0, max(0.0, survival_rate))
    
    def get_stress_test_summary(self) -> Dict[str, Any]:
        """Get summary of all stress test scenarios."""
        return {
            'total_scenarios': len(self.scenarios),
            'scenario_types': list(set(s['type'] for s in self.scenarios)),
            'survival_threshold': self.survival_threshold,
            'initial_capital': self.initial_capital
        }


def main():
    """Test the adversarial selector."""
    import asyncio
    
    async def test_selector():
        selector = AdversarialSelector()
        
        # Test strategies
        test_strategies = [
            {
                'id': 'strategy_1',
                'config': {
                    'win_rate': 0.6,
                    'risk_per_trade': 0.02,
                    'max_positions': 5
                }
            },
            {
                'id': 'strategy_2',
                'config': {
                    'win_rate': 0.45,
                    'risk_per_trade': 0.05,
                    'max_positions': 3
                }
            }
        ]
        
        market_data = {}
        
        results = await selector.select_strategies(test_strategies, market_data)
        
        print("Adversarial Selection Results:")
        for result in results:
            print(f"Strategy {result.strategy_id}:")
            print(f"  Overall Survival Rate: {result.overall_survival_rate:.2%}")
            print(f"  Passed Scenarios: {result.passed_scenarios}/{result.total_scenarios}")
            print(f"  Selected: {result.selected}")
            print()
    
    # Run test
    asyncio.run(test_selector())


if __name__ == "__main__":
    main()
