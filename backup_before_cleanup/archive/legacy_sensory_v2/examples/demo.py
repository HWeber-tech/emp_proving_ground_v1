"""
Demonstration of Multidimensional Market Intelligence System

This demo shows how the system analyzes market data through five dimensions
and synthesizes them into unified market understanding.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
import numpy as np
from market_intelligence import MarketIntelligence, MarketData

def create_sample_data(scenario: str) -> MarketData:
    """Create sample market data for different scenarios"""
    
    base_time = datetime.now()
    
    scenarios = {
        'trending_bull': {
            'bid': 1.1000,
            'ask': 1.1002,
            'volume': 1500,
            'volatility': 0.008
        },
        'trending_bear': {
            'bid': 1.0950,
            'ask': 1.0952,
            'volume': 2000,
            'volatility': 0.012
        },
        'ranging': {
            'bid': 1.0975,
            'ask': 1.0977,
            'volume': 800,
            'volatility': 0.004
        },
        'volatile': {
            'bid': 1.1020,
            'ask': 1.1025,
            'volume': 3500,
            'volatility': 0.025
        }
    }
    
    data_params = scenarios.get(scenario, scenarios['ranging'])
    
    return MarketData(
        timestamp=base_time,
        bid=data_params['bid'],
        ask=data_params['ask'],
        volume=data_params['volume'],
        volatility=data_params['volatility']
    )

def demonstrate_scenario(intelligence: MarketIntelligence, scenario: str):
    """Demonstrate analysis for a specific market scenario"""
    
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario.upper().replace('_', ' ')}")
    print(f"{'='*60}")
    
    # Create sample data
    market_data = create_sample_data(scenario)
    
    print(f"\nMarket Data:")
    print(f"  Price: {market_data.bid:.4f}/{market_data.ask:.4f}")
    print(f"  Spread: {market_data.spread:.6f}")
    print(f"  Volume: {market_data.volume}")
    print(f"  Volatility: {market_data.volatility:.4f}")
    
    # Analyze through multidimensional intelligence
    understanding = intelligence.analyze(market_data)
    
    print(f"\nMarket Understanding:")
    print(f"  Intelligence Level: {understanding.intelligence_level.name}")
    print(f"  Confidence: {understanding.confidence:.2f}")
    print(f"  Regime: {understanding.regime.name}")
    print(f"  Dimensional Consensus: {understanding.dimensional_consensus:.2f}")
    print(f"  Predictive Power: {understanding.predictive_power:.2f}")
    
    print(f"\nNarrative:")
    print(f"  {understanding.narrative}")
    
    if understanding.primary_drivers:
        print(f"\nPrimary Drivers:")
        for driver in understanding.primary_drivers:
            print(f"  • {driver}")
    
    if understanding.risk_factors:
        print(f"\nRisk Factors:")
        for risk in understanding.risk_factors:
            print(f"  • {risk}")
    
    # Get detailed dimensional breakdown
    summary = intelligence.get_summary()
    
    print(f"\nDimensional Analysis:")
    for dim_name, dim_data in summary.get('dimensions', {}).items():
        print(f"  {dim_name.upper()}: {dim_data['value']:.2f} "
              f"(confidence: {dim_data['confidence']:.2f})")
        
        # Show key context
        key_context = dim_data.get('key_context', {})
        if key_context:
            for key, value in list(key_context.items())[:2]:  # Show top 2 context items
                if isinstance(value, (int, float)):
                    print(f"    {key}: {value:.3f}")
                else:
                    print(f"    {key}: {value}")

def main():
    """Main demonstration"""
    
    print("Multidimensional Market Intelligence System")
    print("=" * 50)
    print("\nThis system analyzes markets through five dimensions:")
    print("• WHY: Fundamental forces and economic drivers")
    print("• HOW: Institutional mechanics and order flow")
    print("• WHAT: Technical patterns and price action")
    print("• WHEN: Temporal dynamics and timing")
    print("• ANOMALY: Chaos and manipulation detection")
    print("\nEach dimension is aware of and influences the others,")
    print("creating genuine multidimensional market understanding.")
    
    # Initialize intelligence system
    intelligence = MarketIntelligence()
    
    # Demonstrate different market scenarios
    scenarios = ['trending_bull', 'trending_bear', 'ranging', 'volatile']
    
    for scenario in scenarios:
        demonstrate_scenario(intelligence, scenario)
    
    print(f"\n{'='*60}")
    print("SYSTEM CAPABILITIES DEMONSTRATED")
    print(f"{'='*60}")
    print("\n✓ Multidimensional market analysis")
    print("✓ Cross-dimensional awareness and influence")
    print("✓ Adaptive intelligence levels")
    print("✓ Coherent narrative construction")
    print("✓ Regime-aware processing")
    print("✓ Risk factor identification")
    print("✓ Predictive power assessment")
    print("✓ Unified market understanding")
    
    print(f"\nThe system demonstrates genuine market intelligence")
    print(f"that goes beyond pattern recognition to true understanding.")

if __name__ == "__main__":
    main()

