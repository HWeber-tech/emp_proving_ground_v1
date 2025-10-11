# Multidimensional Market Intelligence System

## Executive Summary

The Multidimensional Market Intelligence System represents a revolutionary approach to market analysis, implementing a sophisticated 5-dimensional sensory cortex that understands markets through interconnected layers of intelligence. This system transforms from structural blueprints into a fully functional, high-fidelity awareness engine capable of production-level market analysis.

## System Architecture

### 5-Dimensional Sensory Cortex

The system analyzes markets through five distinct but interconnected dimensions:

#### 1. WHY Dimension - Fundamental Intelligence
- **Purpose**: Understanding the fundamental forces driving market behavior
- **Components**: Economic data analysis, central bank policy tracking, geopolitical event assessment
- **Key Features**: 
  - Real-time economic calendar integration
  - Central bank sentiment analysis
  - Geopolitical risk assessment
  - Currency strength analysis

#### 2. HOW Dimension - Institutional Intelligence  
- **Purpose**: Analyzing institutional mechanics and execution patterns
- **Components**: Advanced ICT (Inner Circle Trader) pattern detection
- **Key Features**:
  - Order Block detection and analysis
  - Fair Value Gap (FVG) identification
  - Liquidity Sweep detection
  - Breaker Block analysis
  - Market Structure Shift recognition
  - Optimal Trade Entry (OTE) zones
  - Inducement pattern detection
  - Volume profile analysis

#### 3. WHAT Dimension - Technical Reality Engine
- **Purpose**: Pure price action analysis beyond traditional indicators
- **Components**: Advanced technical pattern recognition
- **Key Features**:
  - Momentum dynamics analysis
  - Structural analysis and regime detection
  - Price action pattern recognition
  - Support/resistance level identification
  - Trend strength measurement
  - Volatility regime analysis

#### 4. WHEN Dimension - Temporal Intelligence
- **Purpose**: Understanding timing patterns and session dynamics
- **Components**: Sophisticated temporal analysis
- **Key Features**:
  - Session overlap detection
  - Cyclical pattern analysis
  - Event-driven temporal intelligence
  - High/low activity period identification
  - Temporal regime classification
  - Session-specific behavior analysis

#### 5. ANOMALY Dimension - Chaos Intelligence
- **Purpose**: Detecting manipulation, chaos, and system stress
- **Components**: Self-refuting detection and antifragile design
- **Key Features**:
  - Statistical anomaly detection
  - Manipulation pattern recognition
  - Chaos theory analysis (Hurst exponent, Lyapunov exponent)
  - Self-refutation engine for meta-learning
  - System stress monitoring
  - Antifragility scoring

## Core Innovation: Contextual Fusion Engine

### Cross-Dimensional Awareness
The system's breakthrough innovation is its contextual fusion engine that creates cross-dimensional awareness:

- **Correlation Analysis**: Real-time detection of correlations between dimensions
- **Pattern Recognition**: Identification of cross-dimensional patterns
- **Adaptive Weights**: Dynamic weight adjustment based on performance and market regime
- **Narrative Generation**: Coherent market narrative creation from dimensional synthesis

### Intelligent Orchestration
Dimensions don't exist in isolation but are intelligently orchestrated:

- **Mutual Awareness**: Each dimension is aware of others' states
- **Influence Propagation**: Dimensions affect each other's analysis
- **Unified Understanding**: Synthesis creates holistic market comprehension
- **Emergent Intelligence**: System exhibits intelligence beyond individual components

## Key Features

### 1. Antifragile Design
- **Stress Adaptation**: System gets stronger from market stress and disorder
- **Self-Learning**: Continuous improvement through self-refutation
- **Robust Error Handling**: Graceful degradation under adverse conditions

### 2. Production-Ready Architecture
- **High Performance**: Optimized for real-time analysis
- **Scalable Design**: Modular architecture for easy extension
- **Comprehensive Testing**: Full test suite with integration tests
- **Memory Management**: Bounded data structures prevent memory leaks

### 3. Advanced Analytics
- **Bayesian Confidence Scoring**: Sophisticated confidence measurement
- **Meta-Cognition**: System awareness of its own perception quality
- **Adaptive Parameters**: Self-tuning based on market conditions
- **Performance Monitoring**: Real-time system diagnostics

## Installation and Setup

### Prerequisites
```bash
Python 3.11+
numpy >= 1.22.0
pandas >= 1.5.0
scipy >= 1.8.0
scikit-learn >= 1.2.0
matplotlib >= 3.5.0
```

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd emp_proving_ground_v1

# Install dependencies
pip install -r requirements/base.txt

# Install the package in editable mode so ``src`` is on the path
pip install -e .
```

### Quick Start
```python
import asyncio
from datetime import datetime

from src.core.base import MarketData
from src.orchestration.enhanced_understanding_engine import ContextualFusionEngine

# Initialize the system
engine = ContextualFusionEngine()

# Create market data
market_data = MarketData(
    timestamp=datetime.now(),
    symbol="EURUSD",
    bid=1.0950,
    ask=1.0952,
    volume=1000,
    spread=0.0002,
    volatility=0.005,
)


# Perform analysis
async def analyze() -> None:
    synthesis = await engine.analyze_market_understanding(market_data)
    print(f"Intelligence Level: {synthesis.intelligence_level.name}")
    print(f"Unified Score: {synthesis.unified_score:.3f}")
    print(f"Narrative: {synthesis.narrative_text}")


asyncio.run(analyze())
```

## Driver Formulas

### WHY Dimension (Fundamental Intelligence)
```python
# Economic surprise factor
surprise_factor = (actual - forecast) / historical_std_dev

# Policy rate differential impact
rate_impact = (current_rate - baseline_rate) * currency_weight

# Calendar proximity weighting
time_decay = exp(-time_to_event_hours / decay_constant)

# Final WHY signal
why_signal = (surprise_factor * 0.4) + (rate_impact * 0.3) + (calendar_proximity * 0.3)
```

### HOW Dimension (Institutional Flow)
```python
# Volume-weighted price impact
vwap_deviation = (current_price - vwap) / atr

# Order flow imbalance (Lee-Ready classification)
flow_imbalance = (buy_volume - sell_volume) / total_volume

# Spread regime analysis
spread_regime = log(current_spread / median_spread_20)

# Final HOW signal
how_signal = (vwap_deviation * 0.4) + (flow_imbalance * 0.4) + (spread_regime * 0.2)
```

### WHAT Dimension (Technical Analysis)
```python
# Multi-timeframe RSI divergence
rsi_divergence = (price_slope - rsi_slope) / price_volatility

# ATR volatility regime
vol_regime = current_atr / ema_atr_60

# Pattern confluence score
confluence = ma_cross_signal + bollinger_squeeze + inside_bar_breakout

# Final WHAT signal  
what_signal = (rsi_divergence * 0.3) + (vol_regime * 0.3) + (confluence * 0.4)
```

### WHEN Dimension (Temporal Intelligence)
```python
# Session overlap intensity
session_weight = london_weight + ny_weight + asian_weight

# Economic news proximity
news_proximity = exp(-minutes_to_news / 60)

# Option gamma exposure with strike analysis
gamma_summary = analyzer.summarise(option_positions, spot_price=current_price)
gamma_impact = gamma_summary.impact_score
pin_strike = (
    gamma_summary.primary_strike.strike if gamma_summary.primary_strike else None
)

# Final WHEN signal
when_signal = (session_weight * 0.5) + (news_proximity * 0.5) + (gamma_impact * 0.0)
```

### ANOMALY Dimension (Glitch Detection)
```python
# EWMA z-score of returns
ewma_mean = lambda * previous_mean + (1 - lambda) * current_return
ewma_var = lambda * previous_var + (1 - lambda) * (current_return - ewma_mean)^2
z_score = (current_return - ewma_mean) / sqrt(ewma_var)

# Anomaly signal with saturation
anomaly_signal = sign(z_score) * min(abs(z_score) / 6, 1.0)
anomaly_confidence = min(abs(z_score) / 4, 1.0)

# Downstream consumers now receive a boolean `is_anomaly` flag once
# `abs(z_score)` crosses the alert threshold (default 3.0).
# The raw z-score is also persisted inside the belief snapshot for
# auditability and regime calibration.
```

## CSV Data Dependencies

The sensory system requires three CSV data files in the `sensory/data/` directory:

### `yield_curve.csv`
Schema: `date,symbol,tenor,rate`
- **date**: ISO format date (YYYY-MM-DD)
- **symbol**: Currency pair (e.g., USD, EUR, GBP)
- **tenor**: Yield curve tenor (1M, 3M, 6M, 1Y, 2Y, 5Y, 10Y, 30Y)
- **rate**: Interest rate as decimal (e.g., 0.0525 for 5.25%)

Example:
```csv
date,symbol,tenor,rate
2024-01-15,USD,1M,0.0533
2024-01-15,USD,3M,0.0542
2024-01-15,USD,1Y,0.0485
```

### `risk_indexes.csv`
Schema: `date,index,value`
- **date**: ISO format date (YYYY-MM-DD)
- **index**: Risk index name (VIX, VSTOXX, MOVE, etc.)
- **value**: Index value as float

Example:
```csv
date,index,value
2024-01-15,VIX,13.45
2024-01-15,VSTOXX,15.23
2024-01-15,MOVE,89.12
```

### `policy_rates.csv`
Schema: `date,central_bank,rate,change`
- **date**: ISO format date (YYYY-MM-DD)
- **central_bank**: Central bank identifier (FED, ECB, BOE, BOJ, etc.)
- **rate**: Policy rate as decimal (e.g., 0.0525 for 5.25%)
- **change**: Rate change from previous meeting as decimal

Example:
```csv
date,central_bank,rate,change
2024-01-15,FED,0.0525,0.0000
2024-01-15,ECB,0.0400,-0.0025
2024-01-15,BOE,0.0525,0.0025
```

## Running the Demo

### Quick Smoke Test
```bash
cd /home/ubuntu/repos/emp_proving_ground_v1
python scripts/minimal_sensory_demo.py
```

This runs a simplified 100k tick demonstration that validates the fusion loop stays alive and produces non-zero dimensional readings.

### Full System Demo
```bash
cd /home/ubuntu/repos/emp_proving_ground_v1
python scripts/sensory_demo.py
```

This runs the complete sensory system with all dimensional engines and real data integration.

### Expected Output
```
🧠 Sensory Cortex Smoke Test Demo
==================================================
✓ Successfully imported sensory system components
Generating 100,000 ticks for EURUSD...
✓ Generated test tick generator
Initializing ContextualFusionEngine...
✓ ContextualFusionEngine initialized successfully

📊 Processing ticks (printing every 1000 ticks)...
Tick     | Time     | WHY    | HOW    | WHAT   | WHEN   | ANOMALY | Unified | Confidence
------------------------------------------------------------------------------------------
       0 | 19:57:16 | +0.71 | +1.00 | +1.00 | +1.00 | +1.00 | +0.94 | 0.67
    1000 | 20:13:56 | +0.62 | +1.00 | +1.00 | +1.00 | +1.00 | +0.92 | 0.70
    ...

==================================================
🎯 Demo Results:
✓ Total ticks processed: 100,000
✓ Successful updates: 100,000
✓ Errors encountered: 0
✓ Success rate: 100.0%
✓ Processing time: 15.5 seconds
✓ Ticks per second: 6457

🎉 SMOKE TEST PASSED: Fusion loop stayed alive for 100k+ ticks!
```

## Usage Examples

### 1. Programmatic fusion
```python
import asyncio
from datetime import datetime

from src.core.base import MarketData
from src.orchestration.enhanced_understanding_engine import ContextualFusionEngine


async def run() -> None:
    engine = ContextualFusionEngine()
    synthesis = await engine.analyze_market_understanding(
        MarketData(
            symbol="EURUSD",
            timestamp=datetime.utcnow(),
            open=1.0950,
            high=1.0962,
            low=1.0948,
            close=1.0957,
            bid=1.0956,
            ask=1.0958,
            volume=1_000_000,
            source="example",
        )
    )
    print(f"Narrative: {synthesis.narrative_text}")
    print(f"Unified score: {synthesis.unified_score:.3f}")


asyncio.run(run())
```

### 2. Integrate with the bootstrap pipeline
```python
import asyncio
from datetime import datetime

from src.core.base import MarketData
from src.data_foundation.fabric.market_data_fabric import MarketDataFabric
from src.orchestration.bootstrap_stack import BootstrapSensoryPipeline
from src.orchestration.enhanced_understanding_engine import ContextualFusionEngine


class StaticConnector:
    """Minimal connector that feeds a static market snapshot to the fabric."""

    name = "static"
    priority = 50

    async def fetch(self, symbol: str, *, as_of: datetime | None = None):
        return MarketData(
            symbol=symbol,
            timestamp=as_of or datetime.utcnow(),
            open=1.0950,
            high=1.0960,
            low=1.0940,
            close=1.0955,
            bid=1.0954,
            ask=1.0956,
            volume=1_000_000,
            source="static-demo",
        )


fabric = MarketDataFabric({"static": StaticConnector()})
pipeline = BootstrapSensoryPipeline(
    fabric=fabric, fusion_engine=ContextualFusionEngine()
)


async def main() -> None:
    snapshot = await pipeline.process_tick("EURUSD")
    print(f"Confidence: {snapshot.synthesis.confidence:.2f}")


asyncio.run(main())
```

### 3. Command-line demos
```bash
python scripts/minimal_sensory_demo.py
python scripts/sensory_demo.py
```

### 3. Custom Integration
```python
# Initialize individual dimensional engines
from sensory.dimensions.enhanced_what_dimension import TechnicalRealityEngine
from sensory.dimensions.enhanced_how_dimension import InstitutionalIntelligenceEngine

what_engine = TechnicalRealityEngine()
how_engine = InstitutionalIntelligenceEngine()

# Perform dimensional analysis
what_reading = await what_engine.analyze_technical_reality(market_data)
how_reading = await how_engine.analyze_institutional_intelligence(market_data)
```

## System Output

### Market Synthesis
The system produces comprehensive market synthesis including:

```python
MarketSynthesis(
    understanding_level=UnderstandingLevel.HIGH,
    narrative_coherence=NarrativeCoherence.COMPELLING,
    dominant_narrative=MarketNarrative.CONFLUENCE_SETUP,
    unified_score=0.742,  # -1 to 1 (bearish to bullish)
    confidence=0.856,     # 0 to 1
    narrative_text="Strong bullish confluence detected with positive fundamental backdrop supporting technical breakout during favorable session timing...",
    supporting_evidence=[...],
    contradicting_evidence=[...],
    risk_factors=[...],
    opportunity_factors=[...]
)
```

### Dimensional Readings
Each dimension provides detailed readings:

```python
DimensionalReading(
    dimension='WHAT',
    value=0.678,          # -1 to 1
    confidence=0.823,     # 0 to 1
    context={
        'market_regime': 'TRENDING_BULL',
        'momentum_strength': 0.745,
        'structural_integrity': 0.892,
        'pattern_confluence': ['ascending_triangle', 'bullish_flag']
    },
    timestamp=datetime.now()
)
```

## Performance Characteristics

### Throughput
- **Analysis Speed**: 10-50 analyses per second (depending on hardware)
- **Memory Usage**: Bounded growth with configurable limits
- **Latency**: Sub-100ms analysis time for real-time applications

### Accuracy Metrics
- **Confidence Calibration**: Self-monitoring confidence accuracy
- **Prediction Accuracy**: Tracked across all dimensions
- **Adaptive Performance**: Improves with market stress (antifragile)

## Testing

### Run Test Suite
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_integration.py::TestDimensionalEngines -v
python -m pytest tests/test_integration.py::TestContextualFusion -v
python -m pytest tests/test_integration.py::TestSystemIntegration -v
```

### Performance Benchmarks
```bash
# Run performance benchmarks
python -m pytest tests/test_integration.py::TestPerformanceBenchmarks -v
```

## Configuration

### Adaptive Weights
The system automatically adjusts dimensional weights based on:
- **Performance**: Historical accuracy of each dimension
- **Market Regime**: Current market conditions
- **Correlations**: Cross-dimensional relationships
- **Confidence**: Real-time confidence levels

### Customization
```python
# Customize weight manager
engine.weight_manager.weights['WHY'].base_weight = 0.3
engine.weight_manager.weights['WHAT'].base_weight = 0.25

# Customize correlation analyzer
engine.correlation_analyzer.lookback_periods = 300

# Customize anomaly detection
engine.anomaly_engine.statistical_detector.price_threshold = 2.5
```

## API Reference

### Core Classes

#### ContextualFusionEngine
Main orchestration engine that coordinates all dimensional analysis.

**Methods:**
- `analyze_market_understanding(market_data)`: Perform comprehensive analysis
- `get_diagnostic_information()`: Retrieve system diagnostics

#### DimensionalEngines
Individual engines for each dimension:
- `FundamentalIntelligenceEngine` (WHY)
- `InstitutionalIntelligenceEngine` (HOW)  
- `TechnicalRealityEngine` (WHAT)
- `ChronalIntelligenceEngine` (WHEN)
- `AnomalyIntelligenceEngine` (ANOMALY)

#### Support Classes
- `CorrelationAnalyzer`: Cross-dimensional correlation analysis
- `AdaptiveWeightManager`: Dynamic weight management
- `NarrativeGenerator`: Coherent narrative generation

## Advanced Features

### 1. Self-Refutation Engine
The system continuously validates its own predictions and learns from failures:

```python
# Record prediction for validation
engine.anomaly_engine.self_refutation_engine.record_prediction(
    'price', predicted_price, confidence, timestamp
)

# Get meta-learning insights
insights = engine.anomaly_engine.self_refutation_engine.get_meta_learning_insights()
```

### 2. Chaos Theory Analysis
Advanced chaos theory metrics for market understanding:

```python
# Access chaos metrics
chaos_metrics = engine.anomaly_engine.chaos_detector
print(f"Hurst Exponent: {chaos_metrics.hurst_exponent}")
print(f"Lyapunov Exponent: {chaos_metrics.lyapunov_exponent}")
```

### 3. Cross-Dimensional Patterns
Detection of complex patterns across multiple dimensions:

```python
# Get detected patterns
patterns = engine.correlation_analyzer.get_cross_dimensional_patterns()
for pattern in patterns:
    print(f"Pattern: {pattern.pattern_name}")
    print(f"Strength: {pattern.pattern_strength}")
    print(f"Involved Dimensions: {pattern.involved_dimensions}")
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed and Python path is correct
2. **Memory Usage**: Adjust lookback periods if memory usage is too high
3. **Performance**: Reduce analysis frequency or disable detailed logging for production

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed diagnostics
diagnostics = engine.get_diagnostic_information()
print(json.dumps(diagnostics, indent=2, default=str))
```

## Contributing

### Development Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run pre-commit hooks
pre-commit install

# Run tests before committing
python -m pytest tests/ -v
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints for all functions
- Document all public methods
- Maintain test coverage above 80%

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inner Circle Trader (ICT) concepts for institutional analysis
- Chaos theory research for anomaly detection
- Antifragile design principles from Nassim Taleb
- Modern portfolio theory for risk assessment

## Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact the development team
- Review the documentation and examples

---

**The Multidimensional Market Intelligence System represents the evolution from traditional single-dimensional analysis to true market understanding through interconnected intelligence. It embodies the principle that markets are complex adaptive systems requiring sophisticated, multi-faceted analysis for genuine comprehension.**

