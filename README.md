# EMP Proving Ground v1.0 - Evolutionary Trading System

A comprehensive evolutionary trading system that combines advanced market simulation, adversarial testing, and genetic algorithm optimization to evolve robust trading strategies.

## 🚀 Features

### Core Components
- **Data Pipeline**: Real-time market data ingestion and cleaning
- **Market Simulation**: Realistic trading environment with adversarial events
- **Sensory Cortex**: 4D+1 perception system for market analysis
- **Decision Genome**: Evolutionary decision trees for strategy representation
- **Fitness Evaluation**: Multi-objective fitness scoring across market regimes
- **Evolution Engine**: Population management and genetic algorithm optimization

### Advanced Features
- **Adversarial Testing**: Intelligent market manipulation and stop hunting
- **Regime Detection**: Automatic identification of market regimes
- **Multi-Objective Fitness**: Comprehensive evaluation across returns, robustness, adaptability, efficiency, and antifragility
- **Synthetic Data Generation**: Realistic market data for testing
- **Checkpoint System**: Save and resume evolution progress

## 📁 Project Structure

```
emp_proving_ground_v1/
├── emp/
│   ├── data/
│   │   ├── ingestion.py      # Dukascopy data ingestion
│   │   ├── cleaning.py       # Tick data cleaning
│   │   ├── storage.py        # Data storage and caching
│   │   └── regimes.py        # Market regime identification
│   ├── simulation/
│   │   ├── market.py         # Market simulator
│   │   └── adversary.py      # Adversarial engine
│   ├── agent/
│   │   ├── sensory.py        # 4D+1 sensory cortex
│   │   └── genome.py         # Decision genome
│   └── evolution/
│       ├── fitness.py        # Fitness evaluator
│       └── engine.py         # Evolution engine
├── run_evolution.py          # Main execution script
├── requirements.txt          # Dependencies
├── config.yaml              # Configuration file
└── README.md               # This file
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd emp_proving_ground_v1
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create necessary directories**:
   ```bash
   mkdir -p data results checkpoints logs
   ```

## 🚀 Quick Start

### 1. Test Components
```bash
python run_evolution.py --mode test
```

### 2. Download Market Data
```bash
python run_evolution.py --mode download --symbols EURUSD GBPUSD --years 2023
```

### 3. Identify Market Regimes
```bash
python run_evolution.py --mode regimes --symbols EURUSD --years 2023
```

### 4. Run Evolution
```bash
python run_evolution.py --mode evolution --symbols EURUSD --max-generations 50
```

## 📊 System Architecture

### Data Pipeline
- **DukascopyIngestor**: Downloads real market data from Dukascopy
- **TickDataCleaner**: Removes outliers and invalid data
- **TickDataStorage**: Efficient storage with LRU caching
- **MarketRegimeIdentifier**: Identifies trending, ranging, and volatile regimes

### Market Simulation
- **MarketSimulator**: Realistic PnL engine with spread, commission, and slippage
- **AdversarialEngine**: Intelligent market manipulation including:
  - Stop hunting
  - Spoofing attacks
  - News shocks
  - Flash crashes
  - Breakout traps

### Sensory Cortex (4D+1)
- **WHY**: Fundamental/macro momentum analysis
- **HOW**: Institutional footprint detection
- **WHAT**: Technical pattern recognition
- **WHEN**: Timing and session analysis
- **ANOMALY**: Deviation and outlier detection

### Decision Genome
- **Evolutionary Decision Trees**: Tree-based strategy representation
- **Mutation**: Random parameter changes
- **Crossover**: Strategy combination
- **Complexity Control**: Prevents overfitting

### Fitness Evaluation
- **Multi-Objective Scoring**:
  - Returns performance (Sharpe ratio, total return)
  - Risk management (max drawdown, Sortino ratio)
  - Adaptability (regime performance variance)
  - Efficiency (trade frequency, cost impact)
  - Antifragility (crisis resilience)

### Evolution Engine
- **Population Management**: Tournament selection, elitism
- **Generation Evolution**: Crossover, mutation, selection
- **Stagnation Detection**: Adaptive mutation rates
- **Checkpoint System**: Save/restore evolution progress

## ⚙️ Configuration

Create a `config.yaml` file to customize the system:

```yaml
# Data Configuration
data_dir: "data"
symbols: ["EURUSD", "GBPUSD"]
years: [2023]

# Simulation Configuration
initial_balance: 100000.0
leverage: 1.0
adversarial_intensity: 0.7
commission_rate: 0.0001
slippage_bps: 0.5

# Evolution Configuration
population_size: 100
elite_ratio: 0.1
crossover_ratio: 0.6
mutation_ratio: 0.3
mutation_rate: 0.1
max_stagnation: 20
max_generations: 50
target_fitness: 0.8

# Evaluation Configuration
evaluation_period_days: 30
calibration_days: 30
complexity_penalty: 0.01
```

## 📈 Usage Examples

### Basic Evolution Run
```bash
python run_evolution.py --mode evolution \
    --symbols EURUSD \
    --max-generations 100 \
    --log-level INFO
```

### Custom Configuration
```bash
python run_evolution.py --mode evolution \
    --config my_config.yaml \
    --log-level DEBUG
```

### Data Preparation
```bash
# Download data for multiple symbols and years
python run_evolution.py --mode download \
    --symbols EURUSD GBPUSD USDJPY \
    --years 2022 2023

# Identify regimes
python run_evolution.py --mode regimes \
    --symbols EURUSD \
    --years 2023
```

## 🔬 Advanced Features

### Adversarial Testing
The system includes sophisticated adversarial testing:
- **Intelligent Stop Hunting**: Targets liquidity zones
- **Breakout Traps**: Fake breakouts to trap traders
- **Spoofing Attacks**: Fake order book manipulation
- **News Shocks**: Sudden market movements
- **Flash Crashes**: Extreme volatility events

### Regime-Specific Evaluation
Genomes are evaluated across different market regimes:
- **Trending Markets**: Strong directional movement
- **Ranging Markets**: Sideways consolidation
- **Volatile Markets**: High volatility with choppy movement
- **Crisis Markets**: Extreme stress conditions

### Multi-Objective Fitness
Fitness evaluation considers multiple objectives:
1. **Returns Score**: Total return, Sharpe ratio, win rate
2. **Robustness Score**: Risk-adjusted returns, drawdown control
3. **Adaptability Score**: Performance consistency across regimes
4. **Efficiency Score**: Cost management, trade frequency
5. **Antifragility Score**: Crisis resilience, stress testing

## 📊 Results and Analysis

### Output Files
- `results/final_population_YYYYMMDD_HHMMSS.json`: Final evolved population
- `checkpoints/population_gen_X.json`: Evolution checkpoints
- `emp_evolution.log`: Detailed execution log
- `data/`: Downloaded and processed market data

### Performance Metrics
- **Fitness Scores**: Multi-objective fitness evaluation
- **Complexity Analysis**: Tree depth, node count, leaf analysis
- **Regime Performance**: Performance breakdown by market regime
- **Evolution Progress**: Generation-by-generation improvement

## 🔧 Development

### Adding New Components
1. Create new module in appropriate directory
2. Implement required interfaces
3. Add to main integration script
4. Update documentation

### Extending Fitness Functions
1. Modify `FitnessEvaluator` class
2. Add new fitness components
3. Update weight calculations
4. Test with validation data

### Custom Adversarial Events
1. Extend `AdversarialEngine` class
2. Implement new event types
3. Add event parameters
4. Test with market simulator

## 🐛 Troubleshooting

### Common Issues
1. **Data Download Failures**: Check internet connection and Dukascopy availability
2. **Memory Issues**: Reduce population size or evaluation period
3. **Slow Evolution**: Use smaller population or fewer generations for testing
4. **Import Errors**: Ensure all dependencies are installed

### Debug Mode
```bash
python run_evolution.py --mode evolution --log-level DEBUG
```

## 📚 References

- **Evolutionary Algorithms**: Genetic programming for trading strategies
- **Market Microstructure**: Realistic market simulation
- **Adversarial Testing**: Robustness evaluation techniques
- **Multi-Objective Optimization**: Pareto-optimal solution finding

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dukascopy for providing market data
- Scientific community for evolutionary algorithm research
- Open source contributors for supporting libraries

---

**EMP Proving Ground v1.0** - Pushing the boundaries of evolutionary trading systems. 