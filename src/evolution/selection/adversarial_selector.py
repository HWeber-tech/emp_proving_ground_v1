#!/usr/bin/env python3
"""
Adversarial Selector
===================

This module implements the adversarial selection system with 15 stress test scenarios:
1. Market crash (-20% drop)
2. Flash crash (-10% in 5 minutes)
3. Volatility spike (VIX > 40)
4. Liquidity crisis (bid-ask spread > 5%)
5. Regime change (sudden market shift)
6. Black swan event (-30% overnight)
7. Currency crisis (10% devaluation)
8. Interest rate shock (200bp change)
9. Geopolitical event impact
10. Economic recession scenario
11. Inflation spike (5% monthly)
12. Deflation spiral
13. Banking sector crisis
14. Commodity price shock (50% change)
15. Cyber attack on exchanges

The AdversarialSelector runs strategies against all scenarios and filters those with <70% survival rate.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Callable
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class StressTestResult:
    """Result of a single stress test"""
    scenario_name: str
    final_return: float
    max_drawdown: float
    volatility: float
    survived: bool
    details: Dict[str, Any]


class StressTestScenario:
    """Individual stress test scenario"""
    
    def __init__(self, name: str, description: str, impact_function: Callable):
        self.name = name
        self.description = description
        self.impact_function = impact_function
    
    def run_test(self, strategy, market_data: pd.DataFrame) -> StressTestResult:
        """Run the stress test and return results"""
        return self.impact_function(strategy, market_data)


class AdversarialSelector:
    """Advanced adversarial selection system with stress testing"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.survival_threshold = self.config.get('survival_threshold', 0.7)
        self.max_loss_threshold = self.config.get('max_loss_threshold', 0.05)
        
        # Initialize all 15 stress test scenarios
        self.stress_scenarios = self._initialize_stress_scenarios()
        
        logger.info(f"AdversarialSelector initialized with {len(self.stress_scenarios)} scenarios")
    
    def _initialize_stress_scenarios(self) -> List[StressTestScenario]:
        """Initialize all 15 stress test scenarios"""
        scenarios = [
            StressTestScenario(
                "market_crash",
                "Major market crash with 20% drop",
                self._simulate_market_crash
            ),
            StressTestScenario(
                "flash_crash",
                "Flash crash with 10% drop in 5 minutes",
                self._simulate_flash_crash
            ),
            StressTestScenario(
                "volatility_spike",
                "Extreme volatility spike (VIX > 40)",
                self._simulate_volatility_spike
            ),
            StressTestScenario(
                "liquidity_crisis",
                "Liquidity crisis with bid-ask spread > 5%",
                self._simulate_liquidity_crisis
            ),
            StressTestScenario(
                "regime_change",
                "Sudden market regime change",
                self._simulate_regime_change
            ),
            StressTestScenario(
                "black_swan",
                "Black swan event with 30% overnight drop",
                self._simulate_black_swan
            ),
            StressTestScenario(
                "currency_crisis",
                "Currency crisis with 10% devaluation",
                self._simulate_currency_crisis
            ),
            StressTestScenario(
                "interest_rate_shock",
                "Interest rate shock with 200bp change",
                self._simulate_interest_rate_shock
            ),
            StressTestScenario(
                "geopolitical_event",
                "Major geopolitical event impact",
                self._simulate_geopolitical_event
            ),
            StressTestScenario(
                "economic_recession",
                "Economic recession scenario",
                self._simulate_economic_recession
            ),
            StressTestScenario(
                "inflation_spike",
                "Inflation spike with 5% monthly increase",
                self._simulate_inflation_spike
            ),
            StressTestScenario(
                "deflation_spiral",
                "Deflation spiral scenario",
                self._simulate_deflation_spiral
            ),
            StressTestScenario(
                "banking_crisis",
                "Banking sector crisis",
                self._simulate_banking_crisis
            ),
            StressTestScenario(
                "commodity_shock",
                "Commodity price shock with 50% change",
                self._simulate_commodity_shock
            ),
            StressTestScenario(
                "cyber_attack",
                "Cyber attack on exchanges",
                self._simulate_cyber_attack
            )
        ]
        
        return scenarios
    
    def select_strategies(self, strategies: List[Any], market_data: pd.DataFrame) -> List[Any]:
        """Select strategies based on stress test survival rate"""
        selected_strategies = []
        
        for strategy in strategies:
            try:
                stress_results = self._run_stress_tests(strategy, market_data)
                
                # Calculate survival rate
                total_tests = len(stress_results)
                survived_tests = sum(1 for result in stress_results if result.survived)
                survival_rate = survived_tests / total_tests if total_tests > 0 else 0
                
                # Only select strategies with survival rate >= threshold
                if survival_rate >= self.survival_threshold:
                    strategy.survival_rate = survival_rate
                    strategy.stress_test_results = stress_results
                    selected_strategies.append(strategy)
                    logger.info(f"Strategy {strategy.strategy_id} selected - survival rate: {survival_rate:.2f}")
                else:
                    logger.info(f"Strategy {strategy.strategy_id} rejected - survival rate: {survival_rate:.2f}")
                    
            except Exception as e:
                logger.error(f"Error testing strategy {getattr(strategy, 'strategy_id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Selected {len(selected_strategies)} out of {len(strategies)} strategies")
        return selected_strategies
    
    def _run_stress_tests(self, strategy, market_data: pd.DataFrame) -> List[StressTestResult]:
        """Run all stress tests for a strategy"""
        results = []
        
        for scenario in self.stress_scenarios:
            try:
                result = scenario.run_test(strategy, market_data)
                results.append(result)
            except Exception as e:
                logger.error(f"Error running {scenario.name}: {e}")
                # Create failed result
                results.append(StressTestResult(
                    scenario_name=scenario.name,
                    final_return=-0.1,
                    max_drawdown=0.1,
                    volatility=0.5,
                    survived=False,
                    details={'error': str(e)}
                ))
        
        return results
    
    # Stress test simulation methods
    def _simulate_market_crash(self, strategy, market_data: pd.DataFrame) -> StressTestResult:
        """Simulate major market crash"""
        # Simulate 20% market drop
        crash_return = -0.20
        
        # Strategy performance during crash (varies by strategy type)
        if hasattr(strategy, 'strategy_type'):
            if strategy.strategy_type == 'momentum':
                strategy_return = crash_return * 1.2  # Momentum strategies hurt more
            elif strategy.strategy_type == 'mean_reversion':
                strategy_return = crash_return * 0.8  # Mean reversion may recover
            else:
                strategy_return = crash_return
        else:
            strategy_return = crash_return
        
        survived = abs(strategy_return) <= self.max_loss_threshold
        
        return StressTestResult(
            scenario_name="market_crash",
            final_return=strategy_return,
            max_drawdown=abs(strategy_return),
            volatility=0.35,
            survived=survived,
            details={'market_drop': -0.20, 'strategy_return': strategy_return}
        )
    
    def _simulate_flash_crash(self, strategy, market_data: pd.DataFrame) -> StressTestResult:
        """Simulate flash crash"""
        flash_return = -0.10
        
        # Flash crashes are harder to react to
        strategy_return = flash_return * 1.1
        
        survived = abs(strategy_return) <= self.max_loss_threshold
        
        return StressTestResult(
            scenario_name="flash_crash",
            final_return=strategy_return,
            max_drawdown=abs(strategy_return),
            volatility=0.45,
            survived=survived,
            details={'flash_drop': -0.10, 'strategy_return': strategy_return}
        )
    
    def _simulate_volatility_spike(self, strategy, market_data: pd.DataFrame) -> StressTestResult:
        """Simulate volatility spike"""
        # High volatility typically reduces returns
        volatility_impact = -0.15
        
        # Volatility-sensitive strategies may be affected more
        if hasattr(strategy, 'volatility_sensitivity'):
            strategy_return = volatility_impact * strategy.volatility_sensitivity
        else:
            strategy_return = volatility_impact
        
        survived = abs(strategy_return) <= self.max_loss_threshold
        
        return StressTestResult(
            scenario_name="volatility_spike",
            final_return=strategy_return,
            max_drawdown=abs(strategy_return),
            volatility=0.50,
            survived=survived,
            details={'volatility_level': 0.50, 'strategy_return': strategy_return}
        )
    
    def _simulate_liquidity_crisis(self, strategy, market_data: pd.DataFrame) -> StressTestResult:
        """Simulate liquidity crisis"""
        liquidity_impact = -0.12
        
        # Liquidity-sensitive strategies affected more
        if hasattr(strategy, 'liquidity_sensitivity'):
            strategy_return = liquidity_impact * strategy.liquidity_sensitivity
        else:
            strategy_return = liquidity_impact
        
        survived = abs(strategy_return) <= self.max_loss_threshold
        
        return StressTestResult(
            scenario_name="liquidity_crisis",
            final_return=strategy_return,
            max_drawdown=abs(strategy_return),
            volatility=0.40,
            survived=survived,
            details={'spread_increase': 0.05, 'strategy_return': strategy_return}
        )
    
    def _simulate_regime_change(self, strategy, market_data: pd.DataFrame) -> StressTestResult:
        """Simulate regime change"""
        regime_impact = -0.08
        
        # Adaptive strategies may handle regime changes better
        if hasattr(strategy, 'adaptability_score'):
            strategy_return = regime_impact * (2.0 - strategy.adaptability_score)
        else:
            strategy_return = regime_impact
        
        survived = abs(strategy_return) <= self.max_loss_threshold
        
        return StressTestResult(
            scenario_name="regime_change",
            final_return=strategy_return,
            max_drawdown=abs(strategy_return),
            volatility=0.30,
            survived=survived,
            details={'regime_shift': 'trending_to_ranging', 'strategy_return': strategy_return}
        )
    
    def _simulate_black_swan(self, strategy, market_data: pd.DataFrame) -> StressTestResult:
        """Simulate black swan event"""
        black_swan_return = -0.30
        
        # Most strategies will be severely impacted
        strategy_return = black_swan_return * 1.2
        
        survived = abs(strategy_return) <= self.max_loss_threshold
        
        return StressTestResult(
            scenario_name="black_swan",
            final_return=strategy_return,
            max_drawdown=abs(strategy_return),
            volatility=0.60,
            survived=survived,
            details={'overnight_drop': -0.30, 'strategy_return': strategy_return}
        )
    
    def _simulate_currency_crisis(self, strategy, market_data: pd.DataFrame) -> StressTestResult:
        """Simulate currency crisis"""
        currency_impact = -0.10
        
        # Currency-sensitive strategies affected more
        if hasattr(strategy, 'currency_exposure'):
            strategy_return = currency_impact * strategy.currency_exposure
        else:
            strategy_return = currency_impact
        
        survived = abs(strategy_return) <= self.max_loss_threshold
        
        return StressTestResult(
            scenario_name="currency_crisis",
            final_return=strategy_return,
            max_drawdown=abs(strategy_return),
            volatility=0.35,
            survived=survived,
            details={'devaluation': 0.10, 'strategy_return': strategy_return}
        )
    
    def _simulate_interest_rate_shock(self, strategy, market_data: pd.DataFrame) -> StressTestResult:
        """Simulate interest rate shock"""
        rate_impact = -0.08
        
        # Rate-sensitive strategies affected more
        if hasattr(strategy, 'duration_sensitivity'):
            strategy_return = rate_impact * strategy.duration_sensitivity
        else:
            strategy_return = rate_impact
        
        survived = abs(strategy_return) <= self.max_loss_threshold
        
        return StressTestResult(
            scenario_name="interest_rate_shock",
            final_return=strategy_return,
            max_drawdown=abs(strategy_return),
            volatility=0.25,
            survived=survived,
            details={'rate_change': 0.02, 'strategy_return': strategy_return}
        )
    
    def _simulate_geopolitical_event(self, strategy, market_data: pd.DataFrame) -> StressTestResult:
        """Simulate geopolitical event"""
        geo_impact = -0.12
        
        # Geopolitical risk affects all strategies
        strategy_return = geo_impact
        
        survived = abs(strategy_return) <= self.max_loss_threshold
        
        return StressTestResult(
            scenario_name="geopolitical_event",
            final_return=strategy_return,
            max_drawdown=abs(strategy_return),
            volatility=0.40,
            survived=survived,
            details={'event_type': 'major_conflict', 'strategy_return': strategy_return}
        )
    
    def _simulate_economic_recession(self, strategy, market_data: pd.DataFrame) -> StressTestResult:
        """Simulate economic recession"""
        recession_impact = -0.18
        
        # Recession affects all strategies
        strategy_return = recession_impact
        
        survived = abs(strategy_return) <= self.max_loss_threshold
        
        return StressTestResult(
            scenario_name="economic_recession",
            final_return=strategy_return,
            max_drawdown=abs(strategy_return),
            volatility=0.45,
            survived=survived,
            details={'recession_depth': 0.18, 'strategy_return': strategy_return}
        )
    
    def _simulate_inflation_spike(self, strategy, market_data: pd.DataFrame) -> StressTestResult:
        """Simulate inflation spike"""
        inflation_impact = -0.15
        
        # Inflation affects purchasing power
        strategy_return = inflation_impact
        
        survived = abs(strategy_return) <= self.max_loss_threshold
        
        return StressTestResult(
            scenario_name="inflation_spike",
            final_return=strategy_return,
            max_drawdown=abs(strategy_return),
            volatility=0.35,
            survived=survived,
            details={'inflation_rate': 0.05, 'strategy_return': strategy_return}
        )
    
    def _simulate_deflation_spiral(self, strategy, market_data: pd.DataFrame) -> StressTestResult:
        """Simulate deflation spiral"""
        deflation_impact = -0.20
        
        # Deflation spiral is severe
        strategy_return = deflation_impact
        
        survived = abs(strategy_return) <= self.max_loss_threshold
        
        return StressTestResult(
            scenario_name="deflation_spiral",
            final_return=strategy_return,
            max_drawdown=abs(strategy_return),
            volatility=0.50,
            survived=survived,
            details={'deflation_rate': 0.03, 'strategy_return': strategy_return}
        )
    
    def _simulate_banking_crisis(self, strategy, market_data: pd.DataFrame) -> StressTestResult:
        """Simulate banking crisis"""
        banking_impact = -0.25
        
        # Banking crisis is severe
        strategy_return = banking_impact
        
        survived = abs(strategy_return) <= self.max_loss_threshold
        
        return StressTestResult(
            scenario_name="banking_crisis",
            final_return=strategy_return,
            max_drawdown=abs(strategy_return),
            volatility=0.55,
            survived=survived,
            details={'banking_stress': 0.25, 'strategy_return': strategy_return}
        )
    
    def _simulate_commodity_shock(self, strategy, market_data: pd.DataFrame) -> StressTestResult:
        """Simulate commodity price shock"""
        commodity_impact = -0.15
        
        # Commodity shock affects related strategies
        strategy_return = commodity_impact
        
        survived = abs(strategy_return) <= self.max_loss_threshold
        
        return StressTestResult(
            scenario_name="commodity_shock",
            final_return=strategy_return,
            max_drawdown=abs(strategy_return),
            volatility=0.40,
            survived=survived,
            details={'commodity_change': 0.50, 'strategy_return': strategy_return}
        )
    
    def _simulate_cyber_attack(self, strategy, market_data: pd.DataFrame) -> StressTestResult:
        """Simulate cyber attack"""
        cyber_impact = -0.10
        
        # Cyber attack affects market confidence
        strategy_return = cyber_impact
        
        survived = abs(strategy_return) <= self.max_loss_threshold
        
        return StressTestResult(
            scenario_name="cyber_attack",
            final_return=strategy_return,
            max_drawdown=abs(strategy_return),
            volatility=0.30,
            survived=survived,
            details={'attack_severity': 'major', 'strategy_return': strategy_return}
        )


if __name__ == "__main__":
    """Test the adversarial selector"""
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    # Create mock strategies
    class MockStrategy:
        def __init__(self, strategy_id, strategy_type="momentum"):
            self.strategy_id = strategy_id
            self.strategy_type = strategy_type
    
    # Create test strategies
    strategies = [
        MockStrategy("strategy_1", "momentum"),
        MockStrategy("strategy_2", "mean_reversion"),
        MockStrategy("strategy_3", "arbitrage")
    ]
    
    # Create mock market data
    market_data = pd.DataFrame({
        'close': [1.0, 1.1, 1.2, 1.15, 1.18, 1.2, 1.19, 1.21],
        'volume': [1000, 1200, 1100, 1300, 1250, 1400, 1350, 1500]
    })
    
    # Test adversarial selection
    selector = AdversarialSelector()
    selected_strategies = selector.select_strategies(strategies, market_data)
    
    print(f"Selected {len(selected_strategies)} out of {len(strategies)} strategies")
    for strategy in selected_strategies:
        print(f"  {strategy.strategy_id}: survival rate = {getattr(strategy, 'survival_rate', 0):.2f}")
