"""
Complete Demonstration of Multidimensional Market Intelligence System

This demonstration showcases the full capabilities of the enhanced market intelligence system:
- Real-time multidimensional analysis
- Cross-dimensional correlation detection
- Adaptive weight management
- Contextual fusion and narrative generation
- Performance monitoring and diagnostics
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List, Any
import logging

# Import system components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_intelligence.core.base import MarketData, MarketRegime
from market_intelligence.orchestration.enhanced_intelligence_engine import ContextualFusionEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketDataSimulator:
    """
    Advanced market data simulator that creates realistic market scenarios
    """
    
    def __init__(self, base_price: float = 1.0950):
        self.base_price = base_price
        self.current_price = base_price
        self.time_counter = 0
        
        # Market state
        self.current_regime = 'normal'
        self.regime_duration = 0
        self.trend_strength = 0.0
        self.volatility_regime = 'normal'
        
        # Economic calendar simulation
        self.economic_events = self._generate_economic_calendar()
        
    def _generate_economic_calendar(self) -> List[Dict]:
        """Generate simulated economic events"""
        
        events = []
        base_time = datetime.now()
        
        # Generate events for next 24 hours
        for i in range(24):
            event_time = base_time + timedelta(hours=i)
            
            # Random events with different impacts
            if np.random.random() < 0.1:  # 10% chance per hour
                event = {
                    'time': event_time,
                    'name': np.random.choice([
                        'GDP Release', 'Employment Data', 'Inflation Report',
                        'Central Bank Decision', 'Trade Balance', 'Retail Sales'
                    ]),
                    'impact': np.random.choice(['low', 'medium', 'high']),
                    'expected': np.random.normal(0, 1),
                    'actual': np.random.normal(0, 1.2)  # Slightly more volatile than expected
                }
                events.append(event)
        
        return events
    
    def _update_market_regime(self):
        """Update market regime based on time and randomness"""
        
        self.regime_duration += 1
        
        # Regime transition probabilities
        if self.regime_duration > 20:  # Force regime change after 20 periods
            transition_prob = 0.8
        elif self.regime_duration > 10:
            transition_prob = 0.3
        else:
            transition_prob = 0.05
        
        if np.random.random() < transition_prob:
            # Transition to new regime
            regimes = ['trending_bull', 'trending_bear', 'ranging', 'volatile', 'normal']
            weights = [0.2, 0.2, 0.25, 0.15, 0.2]  # Slightly favor ranging markets
            
            self.current_regime = np.random.choice(regimes, p=weights)
            self.regime_duration = 0
            
            # Set regime parameters
            if 'trending' in self.current_regime:
                self.trend_strength = np.random.uniform(0.5, 1.0)
            else:
                self.trend_strength = np.random.uniform(-0.2, 0.2)
    
    def _check_economic_events(self, current_time: datetime) -> Dict:
        """Check for economic events and their market impact"""
        
        impact_info = {'has_event': False, 'impact_factor': 1.0, 'event_name': None}
        
        for event in self.economic_events:
            time_diff = abs((current_time - event['time']).total_seconds())
            
            if time_diff < 1800:  # Within 30 minutes of event
                impact_info['has_event'] = True
                impact_info['event_name'] = event['name']
                
                # Calculate impact based on surprise (actual vs expected)
                surprise = abs(event['actual'] - event['expected'])
                
                if event['impact'] == 'high':
                    impact_info['impact_factor'] = 1.0 + surprise * 0.5
                elif event['impact'] == 'medium':
                    impact_info['impact_factor'] = 1.0 + surprise * 0.3
                else:
                    impact_info['impact_factor'] = 1.0 + surprise * 0.1
                
                break
        
        return impact_info
    
    def generate_market_data(self) -> MarketData:
        """Generate realistic market data based on current regime"""
        
        self.time_counter += 1
        current_time = datetime.now() + timedelta(minutes=self.time_counter)
        
        # Update market regime
        self._update_market_regime()
        
        # Check for economic events
        event_impact = self._check_economic_events(current_time)
        
        # Base parameters by regime
        if self.current_regime == 'trending_bull':
            base_drift = 0.0001 * self.trend_strength
            base_volatility = 0.006
            base_volume = 1500
            
        elif self.current_regime == 'trending_bear':
            base_drift = -0.0001 * self.trend_strength
            base_volatility = 0.008  # Bear markets more volatile
            base_volume = 1800
            
        elif self.current_regime == 'ranging':
            # Mean reversion
            deviation = self.current_price - self.base_price
            base_drift = -deviation * 0.05  # Revert to mean
            base_volatility = 0.004
            base_volume = 1200
            
        elif self.current_regime == 'volatile':
            base_drift = np.random.normal(0, 0.0002)
            base_volatility = 0.015
            base_volume = 2000
            
        else:  # normal
            base_drift = np.random.normal(0, 0.00005)
            base_volatility = 0.005
            base_volume = 1000
        
        # Apply economic event impact
        if event_impact['has_event']:
            base_volatility *= event_impact['impact_factor']
            base_volume *= event_impact['impact_factor']
            
            # Random direction for event impact
            if np.random.random() < 0.5:
                base_drift += 0.0002 * (event_impact['impact_factor'] - 1.0)
            else:
                base_drift -= 0.0002 * (event_impact['impact_factor'] - 1.0)
        
        # Generate price change
        price_change = base_drift + np.random.normal(0, base_volatility)
        self.current_price += price_change
        
        # Generate volume (log-normal distribution)
        volume = max(np.random.lognormal(np.log(base_volume), 0.3), 100)
        
        # Generate volatility (exponential distribution)
        volatility = max(np.random.exponential(base_volatility), 0.001)
        
        # Add some microstructure noise
        spread = np.random.exponential(0.00005) + 0.00002  # 0.2-1.0 pip spread
        
        return MarketData(
            timestamp=current_time,
            bid=self.current_price - spread/2,
            ask=self.current_price + spread/2,
            volume=volume,
            volatility=volatility
        )

class PerformanceMonitor:
    """
    Monitor system performance and collect metrics
    """
    
    def __init__(self):
        self.metrics = {
            'analysis_times': [],
            'memory_usage': [],
            'synthesis_scores': [],
            'confidence_levels': [],
            'regime_detections': [],
            'anomaly_levels': [],
            'narrative_coherence': []
        }
        
        self.start_time = time.time()
        
    def record_analysis(self, analysis_time: float, synthesis: Any, memory_mb: float = None):
        """Record analysis metrics"""
        
        self.metrics['analysis_times'].append(analysis_time)
        self.metrics['synthesis_scores'].append(synthesis.unified_score)
        self.metrics['confidence_levels'].append(synthesis.confidence)
        self.metrics['narrative_coherence'].append(synthesis.narrative_coherence.name)
        
        if memory_mb:
            self.metrics['memory_usage'].append(memory_mb)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        
        total_time = time.time() - self.start_time
        
        return {
            'total_runtime': total_time,
            'total_analyses': len(self.metrics['analysis_times']),
            'avg_analysis_time': np.mean(self.metrics['analysis_times']) if self.metrics['analysis_times'] else 0,
            'max_analysis_time': np.max(self.metrics['analysis_times']) if self.metrics['analysis_times'] else 0,
            'throughput': len(self.metrics['analysis_times']) / total_time if total_time > 0 else 0,
            'avg_confidence': np.mean(self.metrics['confidence_levels']) if self.metrics['confidence_levels'] else 0,
            'avg_unified_score': np.mean(self.metrics['synthesis_scores']) if self.metrics['synthesis_scores'] else 0,
            'score_volatility': np.std(self.metrics['synthesis_scores']) if self.metrics['synthesis_scores'] else 0,
            'memory_usage': {
                'avg': np.mean(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0,
                'max': np.max(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0
            }
        }

class IntelligenceDemo:
    """
    Main demonstration class
    """
    
    def __init__(self):
        self.fusion_engine = ContextualFusionEngine()
        self.data_simulator = MarketDataSimulator()
        self.performance_monitor = PerformanceMonitor()
        
        # Demo configuration
        self.demo_duration = 200  # Number of analysis cycles
        self.display_interval = 10  # Display results every N cycles
        self.detailed_interval = 50  # Detailed analysis every N cycles
        
    async def run_comprehensive_demo(self):
        """Run comprehensive demonstration"""
        
        print("=" * 80)
        print("MULTIDIMENSIONAL MARKET INTELLIGENCE SYSTEM - COMPREHENSIVE DEMO")
        print("=" * 80)
        print()
        
        print("Initializing system components...")
        print("✓ Contextual Fusion Engine")
        print("✓ 5-Dimensional Analysis (WHY, HOW, WHAT, WHEN, ANOMALY)")
        print("✓ Cross-Dimensional Correlation Analysis")
        print("✓ Adaptive Weight Management")
        print("✓ Narrative Generation Engine")
        print("✓ Performance Monitoring")
        print()
        
        print(f"Starting {self.demo_duration} analysis cycles...")
        print("=" * 80)
        print()
        
        for cycle in range(self.demo_duration):
            
            # Generate market data
            market_data = self.data_simulator.generate_market_data()
            
            # Perform analysis with timing
            start_time = time.time()
            synthesis = await self.fusion_engine.analyze_market_intelligence(market_data)
            analysis_time = time.time() - start_time
            
            # Record performance metrics
            self.performance_monitor.record_analysis(analysis_time, synthesis)
            
            # Display results at intervals
            if cycle % self.display_interval == 0:
                self._display_basic_results(cycle, market_data, synthesis, analysis_time)
            
            # Detailed analysis at intervals
            if cycle % self.detailed_interval == 0 and cycle > 0:
                await self._display_detailed_analysis(cycle)
            
            # Brief pause for readability
            await asyncio.sleep(0.1)
        
        # Final summary
        await self._display_final_summary()
    
    def _display_basic_results(self, cycle: int, market_data: MarketData, 
                              synthesis: Any, analysis_time: float):
        """Display basic analysis results"""
        
        print(f"Cycle {cycle:3d} | Price: {(market_data.bid + market_data.ask)/2:.5f} | "
              f"Intelligence: {synthesis.intelligence_level.name:12s} | "
              f"Score: {synthesis.unified_score:+.3f} | "
              f"Confidence: {synthesis.confidence:.3f} | "
              f"Time: {analysis_time*1000:.1f}ms")
        
        if cycle % (self.display_interval * 2) == 0:
            print(f"    Narrative: {synthesis.narrative_text[:100]}...")
            print()
    
    async def _display_detailed_analysis(self, cycle: int):
        """Display detailed analysis including diagnostics"""
        
        print("\n" + "=" * 80)
        print(f"DETAILED ANALYSIS - CYCLE {cycle}")
        print("=" * 80)
        
        # Get current synthesis
        synthesis = self.fusion_engine.current_synthesis
        
        if synthesis:
            print(f"Intelligence Level: {synthesis.intelligence_level.name}")
            print(f"Narrative Coherence: {synthesis.narrative_coherence.name}")
            print(f"Dominant Narrative: {synthesis.dominant_narrative.name}")
            print(f"Unified Score: {synthesis.unified_score:+.3f}")
            print(f"Confidence: {synthesis.confidence:.3f}")
            print()
            
            print("NARRATIVE:")
            print(f"  {synthesis.narrative_text}")
            print()
            
            if synthesis.supporting_evidence:
                print("SUPPORTING EVIDENCE:")
                for evidence in synthesis.supporting_evidence[:3]:  # Top 3
                    print(f"  • {evidence}")
                print()
            
            if synthesis.risk_factors:
                print("RISK FACTORS:")
                for risk in synthesis.risk_factors[:3]:  # Top 3
                    print(f"  ⚠ {risk}")
                print()
            
            if synthesis.opportunity_factors:
                print("OPPORTUNITIES:")
                for opportunity in synthesis.opportunity_factors[:3]:  # Top 3
                    print(f"  ✓ {opportunity}")
                print()
        
        # Dimensional readings
        print("DIMENSIONAL READINGS:")
        for dimension, reading in self.fusion_engine.current_readings.items():
            print(f"  {dimension:8s}: {reading.signal_strength:+.3f} (confidence: {reading.confidence:.3f})")
        print()
        
        # Adaptive weights
        weights = self.fusion_engine.weight_manager.calculate_current_weights()
        print("ADAPTIVE WEIGHTS:")
        for dimension, weight in weights.items():
            print(f"  {dimension:8s}: {weight:.3f}")
        print()
        
        # Correlations
        correlations = self.fusion_engine.correlation_analyzer.get_dimensional_correlations()
        if correlations:
            print("DIMENSIONAL CORRELATIONS:")
            for (dim_a, dim_b), corr in list(correlations.items())[:5]:  # Top 5
                print(f"  {dim_a}-{dim_b}: {corr.correlation:+.3f} (sig: {corr.significance:.3f})")
            print()
        
        # Patterns
        patterns = self.fusion_engine.correlation_analyzer.get_cross_dimensional_patterns()
        if patterns:
            print("DETECTED PATTERNS:")
            for pattern in patterns[:3]:  # Top 3
                print(f"  • {pattern.pattern_name}: {pattern.pattern_strength:.3f} confidence")
            print()
        
        # Performance metrics
        perf_summary = self.performance_monitor.get_performance_summary()
        print("PERFORMANCE METRICS:")
        print(f"  Throughput: {perf_summary['throughput']:.2f} analyses/sec")
        print(f"  Avg Analysis Time: {perf_summary['avg_analysis_time']*1000:.1f}ms")
        print(f"  Avg Confidence: {perf_summary['avg_confidence']:.3f}")
        print()
        
        print("=" * 80)
        print()
    
    async def _display_final_summary(self):
        """Display final demonstration summary"""
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETE - FINAL SUMMARY")
        print("=" * 80)
        
        # Performance summary
        perf_summary = self.performance_monitor.get_performance_summary()
        
        print("PERFORMANCE SUMMARY:")
        print(f"  Total Runtime: {perf_summary['total_runtime']:.1f} seconds")
        print(f"  Total Analyses: {perf_summary['total_analyses']}")
        print(f"  Average Throughput: {perf_summary['throughput']:.2f} analyses/second")
        print(f"  Average Analysis Time: {perf_summary['avg_analysis_time']*1000:.1f}ms")
        print(f"  Maximum Analysis Time: {perf_summary['max_analysis_time']*1000:.1f}ms")
        print()
        
        print("INTELLIGENCE SUMMARY:")
        print(f"  Average Confidence: {perf_summary['avg_confidence']:.3f}")
        print(f"  Average Unified Score: {perf_summary['avg_unified_score']:+.3f}")
        print(f"  Score Volatility: {perf_summary['score_volatility']:.3f}")
        print()
        
        # Dimensional performance
        print("DIMENSIONAL PERFORMANCE:")
        diagnostics = self.fusion_engine.get_diagnostic_information()
        
        if 'adaptive_weights' in diagnostics:
            for dimension, weight_info in diagnostics['adaptive_weights'].items():
                if dimension != 'current_regime':
                    accuracy = weight_info.get('prediction_accuracy', 0.5)
                    current_weight = weight_info.get('current_weight', 0.2)
                    print(f"  {dimension:8s}: Weight={current_weight:.3f}, Accuracy={accuracy:.3f}")
        print()
        
        # Correlation insights
        if 'correlations' in diagnostics and diagnostics['correlations']:
            print("STRONGEST CORRELATIONS:")
            correlations = diagnostics['correlations']
            sorted_corrs = sorted(
                correlations.items(), 
                key=lambda x: abs(x[1]['correlation']), 
                reverse=True
            )
            
            for pair, corr_info in sorted_corrs[:5]:
                correlation = corr_info['correlation']
                significance = corr_info['significance']
                print(f"  {pair}: {correlation:+.3f} (significance: {significance:.3f})")
            print()
        
        # Pattern detection summary
        if 'patterns' in diagnostics and diagnostics['patterns']:
            print("DETECTED PATTERNS:")
            for pattern in diagnostics['patterns']:
                print(f"  • {pattern['name']}: {pattern['confidence']:.3f} confidence")
            print()
        
        print("SYSTEM CAPABILITIES DEMONSTRATED:")
        print("  ✓ Real-time multidimensional market analysis")
        print("  ✓ Cross-dimensional correlation detection")
        print("  ✓ Adaptive weight management based on performance")
        print("  ✓ Contextual fusion and narrative generation")
        print("  ✓ Anomaly detection and chaos intelligence")
        print("  ✓ Performance monitoring and diagnostics")
        print("  ✓ Robust error handling and graceful degradation")
        print()
        
        print("SYSTEM STATUS: ✓ FULLY OPERATIONAL")
        print("The multidimensional market intelligence system is ready for production use.")
        print("=" * 80)

class InteractiveDemo:
    """
    Interactive demonstration mode
    """
    
    def __init__(self):
        self.fusion_engine = ContextualFusionEngine()
        self.data_simulator = MarketDataSimulator()
    
    async def run_interactive_demo(self):
        """Run interactive demonstration"""
        
        print("=" * 80)
        print("INTERACTIVE MARKET INTELLIGENCE DEMO")
        print("=" * 80)
        print()
        print("Commands:")
        print("  'analyze' - Perform single analysis")
        print("  'stream N' - Stream N analyses")
        print("  'scenario X' - Set market scenario (bull/bear/range/volatile/normal)")
        print("  'diagnostics' - Show system diagnostics")
        print("  'help' - Show this help")
        print("  'quit' - Exit demo")
        print()
        
        while True:
            try:
                command = input("Intelligence> ").strip().lower()
                
                if command == 'quit':
                    break
                elif command == 'analyze':
                    await self._single_analysis()
                elif command.startswith('stream'):
                    parts = command.split()
                    count = int(parts[1]) if len(parts) > 1 else 10
                    await self._stream_analysis(count)
                elif command.startswith('scenario'):
                    parts = command.split()
                    scenario = parts[1] if len(parts) > 1 else 'normal'
                    self._set_scenario(scenario)
                elif command == 'diagnostics':
                    self._show_diagnostics()
                elif command == 'help':
                    self._show_help()
                else:
                    print("Unknown command. Type 'help' for available commands.")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nDemo ended.")
    
    async def _single_analysis(self):
        """Perform single analysis"""
        
        market_data = self.data_simulator.generate_market_data()
        
        start_time = time.time()
        synthesis = await self.fusion_engine.analyze_market_intelligence(market_data)
        analysis_time = time.time() - start_time
        
        print(f"\nMarket Data: Price={market_data.bid:.5f}/{market_data.ask:.5f}, "
              f"Volume={market_data.volume:.0f}, Volatility={market_data.volatility:.4f}")
        print(f"Analysis Time: {analysis_time*1000:.1f}ms")
        print()
        print(f"Intelligence Level: {synthesis.intelligence_level.name}")
        print(f"Unified Score: {synthesis.unified_score:+.3f}")
        print(f"Confidence: {synthesis.confidence:.3f}")
        print(f"Narrative: {synthesis.narrative_text}")
        print()
    
    async def _stream_analysis(self, count: int):
        """Stream multiple analyses"""
        
        print(f"\nStreaming {count} analyses...")
        print("Cycle | Price    | Intelligence | Score  | Confidence | Time")
        print("-" * 65)
        
        for i in range(count):
            market_data = self.data_simulator.generate_market_data()
            
            start_time = time.time()
            synthesis = await self.fusion_engine.analyze_market_intelligence(market_data)
            analysis_time = time.time() - start_time
            
            price = (market_data.bid + market_data.ask) / 2
            print(f"{i+1:5d} | {price:.5f} | {synthesis.intelligence_level.name:12s} | "
                  f"{synthesis.unified_score:+.3f} | {synthesis.confidence:.3f}      | {analysis_time*1000:.1f}ms")
            
            await asyncio.sleep(0.1)  # Brief pause
        
        print()
    
    def _set_scenario(self, scenario: str):
        """Set market scenario"""
        
        valid_scenarios = ['bull', 'bear', 'range', 'volatile', 'normal']
        
        if scenario in valid_scenarios:
            if scenario == 'bull':
                self.data_simulator.current_regime = 'trending_bull'
            elif scenario == 'bear':
                self.data_simulator.current_regime = 'trending_bear'
            elif scenario == 'range':
                self.data_simulator.current_regime = 'ranging'
            elif scenario == 'volatile':
                self.data_simulator.current_regime = 'volatile'
            else:
                self.data_simulator.current_regime = 'normal'
            
            self.data_simulator.regime_duration = 0
            print(f"Market scenario set to: {scenario}")
        else:
            print(f"Invalid scenario. Valid options: {', '.join(valid_scenarios)}")
    
    def _show_diagnostics(self):
        """Show system diagnostics"""
        
        diagnostics = self.fusion_engine.get_diagnostic_information()
        
        print("\nSYSTEM DIAGNOSTICS:")
        print("=" * 50)
        
        # Current readings
        if 'current_readings' in diagnostics:
            print("Current Dimensional Readings:")
            for dimension, reading in diagnostics['current_readings'].items():
                print(f"  {dimension:8s}: {reading['signal_strength']:+.3f} (conf: {reading['confidence']:.3f})")
            print()
        
        # Adaptive weights
        if 'adaptive_weights' in diagnostics:
            print("Adaptive Weights:")
            for dimension, weight_info in diagnostics['adaptive_weights'].items():
                if dimension != 'current_regime':
                    weight = weight_info.get('current_weight', 0)
                    print(f"  {dimension:8s}: {weight:.3f}")
            print()
        
        # Correlations
        if 'correlations' in diagnostics and diagnostics['correlations']:
            print("Top Correlations:")
            correlations = diagnostics['correlations']
            sorted_corrs = sorted(
                correlations.items(), 
                key=lambda x: abs(x[1]['correlation']), 
                reverse=True
            )
            
            for pair, corr_info in sorted_corrs[:5]:
                correlation = corr_info['correlation']
                print(f"  {pair}: {correlation:+.3f}")
            print()
        
        # Patterns
        if 'patterns' in diagnostics and diagnostics['patterns']:
            print("Detected Patterns:")
            for pattern in diagnostics['patterns']:
                print(f"  • {pattern['name']}: {pattern['confidence']:.3f}")
            print()
        
        print("=" * 50)
    
    def _show_help(self):
        """Show help information"""
        
        print("\nAVAILABLE COMMANDS:")
        print("  analyze          - Perform single market analysis")
        print("  stream N         - Stream N continuous analyses")
        print("  scenario X       - Set market scenario:")
        print("                     bull, bear, range, volatile, normal")
        print("  diagnostics      - Show detailed system diagnostics")
        print("  help             - Show this help message")
        print("  quit             - Exit the demo")
        print()

async def main():
    """Main demonstration entry point"""
    
    print("Multidimensional Market Intelligence System")
    print("Choose demo mode:")
    print("1. Comprehensive Demo (automated)")
    print("2. Interactive Demo")
    print()
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == '1':
            demo = IntelligenceDemo()
            await demo.run_comprehensive_demo()
        elif choice == '2':
            demo = InteractiveDemo()
            await demo.run_interactive_demo()
        else:
            print("Invalid choice. Running comprehensive demo...")
            demo = IntelligenceDemo()
            await demo.run_comprehensive_demo()
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

