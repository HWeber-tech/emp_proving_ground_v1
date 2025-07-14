# EMP Proving Ground v2.0 - Modular Evolutionary Trading System

### An Adversarial Simulation Environment for Forging Antifragile Trading Organisms

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-85%25-blue)
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

The EMP Proving Ground is not just another backtesting engine. It is a **Darwinian crucible** designed to solve the fundamental problem of algorithmic trading: **static strategies fail in dynamic markets.**

Traditional trading bots are brittle; they are optimized for a specific set of historical patterns and break as soon as the market regime changes or they encounter sophisticated manipulation. The EMP project takes a different approach. We don't code trading strategies; we create the conditions for them to **emerge** through evolution.

## Core Philosophy: The Antifragile Predator

This system is built on a set of first principles derived from our collaborative AI brain trust:

1. **Evolution over Optimization:** We don't curve-fit. We use genetic programming to run a "digital natural selection" where populations of random trading strategies (genomes) compete. Only the fittest—those that are profitable, robust, and adaptive—survive to reproduce.

2. **Adversarial by Design:** The market is a hostile, adversarial environment. To prepare our organisms, the Proving Ground is not a peaceful garden but a deadly arena. A **Cunning Adversarial Engine** actively tries to trick, trap, and destroy the trading genomes by simulating real-world manipulation tactics like stop hunts and spoofing.

3. **Resilience as the Ultimate Goal:** A profitable but fragile strategy is useless. Our **Multi-Objective Fitness Function** rewards genomes not just for high returns, but for their ability to withstand attacks, perform consistently across different market conditions, and manage risk intelligently.

4. **Intelligence as an Emergent Property:** We do not hard-code trading patterns like the "London Sweep." Instead, we provide the EMP with the sensory tools to perceive the market (`5D Multidimensional Sensory Cortex`) and let it discover which patterns are truly effective through thousands of generations of trial and error.

---

## 🏛️ Architecture Overview

The EMP Proving Ground v2.0 is a modular system where each component plays a critical role in the evolutionary process.

```mermaid
graph TD
    subgraph "Modular System Architecture"
        A[src/core.py] --> B[src/simulation.py];
        C[src/risk.py] --> B;
        D[src/pnl.py] --> B;
        B --> E[src/sensory/];
        E --> F[src/evolution.py];
        F --> G[src/data.py];
        G --> A;
    end

    subgraph "5D Multidimensional Sensory Cortex"
        E1[src/sensory/dimensions/why_dimension.py] --> E0[src/sensory/orchestration/intelligence_engine.py];
        E2[src/sensory/dimensions/how_dimension.py] --> E0;
        E3[src/sensory/dimensions/what_dimension.py] --> E0;
        E4[src/sensory/dimensions/when_dimension.py] --> E0;
        E5[src/sensory/dimensions/anomaly_dimension.py] --> E0;
        E0 --> E;
    end

    A -- Core System --> B;
    C -- Risk Management --> B;
    D -- PnL Calculations --> B;
    B -- Market State --> E;
    E -- Sensory Input --> F;
    F -- Evolution --> G;
    G -- Data Pipeline --> A;
```

## 🚀 Features

### Core Components
- **Core System** (`src/core.py`): Central system components and utilities
- **Risk Management** (`src/risk.py`): Position sizing, risk limits, and validation
- **PnL Engine** (`src/pnl.py`): Profit/loss calculations and trade tracking
- **Data Pipeline** (`src/data.py`): Market data ingestion, cleaning, and storage
- **5D Sensory Cortex** (`src/sensory/`): Advanced multidimensional market intelligence system
- **Evolution Engine** (`src/evolution.py`): Genetic algorithm and population management
- **Market Simulation** (`src/simulation.py`): Realistic trading environment with adversarial events

### 🧠 5D Multidimensional Sensory Cortex

The system now features a **sophisticated 5-dimensional market intelligence engine** that understands markets through orchestrated dimensional awareness:

#### **Dimension 1: WHY - The Macro-Narrative Engine**
- **Economic Momentum Engine**: Measures health and direction of underlying economies
- **Central Bank Policy Analyzer**: Tracks flow and cost of money
- **Market Sentiment & Risk Flow Gauge**: Measures aggregate appetite for risk
- **Data Sources**: Real-time economic data feeds, central bank publications, futures market data
- **Key Metrics**: Economic Surprise Index, Policy Divergence Score, Risk-On/Risk-Off Score

#### **Dimension 2: HOW - The Institutional Mechanics Engine**
- **Order Flow & Volume Profiler**: Analyzes tape to see buyer/seller aggression
- **Market Depth & Liquidity Analyzer**: Examines order book liquidity patterns
- **Algorithmic Execution Detector**: Identifies institutional algorithm patterns
- **Data Sources**: Level 2 Order Book data, Time & Sales data, high-resolution tick data
- **Key Metrics**: Volume Delta, Book-Side Imbalance, Algorithmic Pattern Recognition

#### **Dimension 3: WHAT - The Technical Reality Engine**
- **Market Structure Analyzer**: ICT concepts and institutional order flow
- **Support/Resistance Detector**: Dynamic level identification
- **Momentum & Volatility Profiler**: Price action analysis
- **Data Sources**: OHLCV data, market structure patterns
- **Key Metrics**: Market Structure Score, Support/Resistance Score, Momentum Score

#### **Dimension 4: WHEN - The Temporal Intelligence Engine**
- **Session Analysis**: Trading session dynamics and overlaps
- **Intraday Pattern Recognition**: Time-based rhythms and patterns
- **Event Timing**: Scheduled releases and market events
- **Data Sources**: Time-based patterns, session characteristics
- **Key Metrics**: Session Analysis, Time Momentum, Event Proximity

#### **Dimension 5: ANOMALY - The Chaos Detection Engine**
- **Statistical Anomaly Detector**: Outlier identification and analysis
- **Manipulation Pattern Recognition**: Spoofing, wash trading, coordinated activity
- **Regime Change Detection**: Structural breaks and volatility shifts
- **Data Sources**: Statistical baselines, manipulation patterns
- **Key Metrics**: Anomaly Score, Manipulation Probability, Regime Stability

### Advanced Features
- **Cross-Dimensional Awareness**: Each dimension influences and is influenced by others
- **Adaptive Intelligence**: System adjusts dimensional weights based on market regime
- **Narrative Construction**: Coherent market narratives explaining behavior
- **Anti-Fragile Design**: System becomes stronger under market stress
- **Thread-Safe Operations**: Production-ready with comprehensive error handling
- **Adversarial Testing**: Intelligent market manipulation and stop hunting
- **Regime Detection**: Automatic identification of market regimes
- **Multi-Objective Fitness**: Comprehensive evaluation across returns, robustness, adaptability, efficiency, and antifragility
- **Synthetic Data Generation**: Realistic market data for testing
- **Checkpoint System**: Save and resume evolution progress

### ✨ Key Features
- **High-Fidelity Market Simulator**: Replays real historical tick data with realistic, dynamic spreads, commissions, slippage, and market impact modeling.
- **Cunning Adversarial Engine**: Goes beyond random noise to implement intelligent, context-aware manipulation tactics.
- **Liquidity Zone Hunter**: Identifies likely stop-loss clusters and executes targeted stop hunts.
- **Breakout Trap Spoofing**: Detects price consolidations and engineers fake breakouts to trap predictable algorithms.
- **"Triathlon" Fitness Evaluation**: A groundbreaking anti-overfitting mechanism. Every genome is tested across three distinct, pre-identified historical market regimes: Trending, Ranging, and Volatile/Crisis.
- **Multi-Objective Fitness Function**: The final fitness score is a sophisticated blend of Sortino Ratio, Calmar Ratio, Profit Factor, and a critical Robustness Score derived from performance under adversarial attack.
- **Genetic Programming Core**: Trading strategies are represented as evolvable Decision Trees, allowing for the emergence of complex, interpretable logic.
- **5D Multidimensional Sensory Cortex**: The "brain" of each organism, which perceives the market through five orchestrated dimensions with cross-dimensional awareness.

## 🛠️ Tech Stack

- **Python 3.10+**: Core programming language
- **Pandas & NumPy**: Data manipulation and numerical computing
- **SciPy & Scikit-learn**: Scientific computing and machine learning
- **PyYAML**: Configuration management
- **Matplotlib & Seaborn**: Data visualization
- **PyArrow**: High-performance data storage

## 📋 Prerequisites

- Python 3.10+
- Required packages (see `requirements.txt`)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test the Multidimensional Sensory System
```bash
python test_sensory_integration.py
```

### 3. Run the Complete System
```bash
python main.py
```

This will execute the complete EMP Proving Ground system with demonstrations of all components including the new 5D sensory cortex.

## ⚙️ Configuration

The system is configured through `config.yaml`:

```yaml
data:
  raw_dir: data/raw
  processed_dir: data/processed
  symbol: EURUSD
  start_year: 2018
  end_year: 2024

simulation:
  commission_per_trade: 0.0001  # 1 pip
  base_slippage_bps: 0.5
  size_impact_factor: 0.1
  volatility_factor: 0.2
  initial_balance: 100000
  leverage: 1.0

evolution:
  population_size: 50
  generations: 10
  elite_ratio: 0.1
  crossover_ratio: 0.8
  mutation_ratio: 0.1
  mutation_rate: 0.1

adversary:
  enabled: true
  intensity: 0.7

sensory:
  enable_cross_dimensional_awareness: true
  adaptive_weighting: true
  narrative_construction: true
```

## 📊 Project Structure

```
EMP/
├── src/                    # Core modular system
│   ├── __init__.py
│   ├── core.py            # Core system components
│   ├── risk.py            # Risk management
│   ├── pnl.py             # PnL calculations
│   ├── data.py            # Data handling
│   ├── sensory/           # 5D Multidimensional Sensory Cortex
│   │   ├── __init__.py
│   │   ├── core/          # Core sensory components
│   │   │   ├── base.py    # Foundational classes and data models
│   │   │   └── __init__.py
│   │   ├── dimensions/    # Five dimensional sensors
│   │   │   ├── why_dimension.py      # Fundamental forces
│   │   │   ├── how_dimension.py      # Institutional mechanics
│   │   │   ├── what_dimension.py     # Technical analysis
│   │   │   ├── when_dimension.py     # Temporal intelligence
│   │   │   ├── anomaly_dimension.py  # Chaos detection
│   │   │   └── __init__.py
│   │   ├── orchestration/ # Central synthesis
│   │   │   ├── intelligence_engine.py # Orchestration engine
│   │   │   └── __init__.py
│   │   └── examples/      # Usage examples
│   │       └── demo.py
│   ├── evolution.py       # Evolution engine
│   └── simulation.py      # Market simulation
├── main.py                 # Main entry point
├── config.yaml             # Configuration
├── configs/                # Configuration files
├── scripts/                # Utility scripts
├── data/                   # Data directory
├── experiments/            # Experiment results
├── archive/                # Legacy files (safely stored)
│   └── legacy_sensory/     # Previous sensory system
├── requirements.txt        # Dependencies
├── test_sensory_integration.py # Sensory system test
└── README.md              # This file
```

## 🔄 Development Workflow

1. **Data Preparation**: Use `scripts/create_test_data.py` to generate synthetic data
2. **Sensory System Testing**: Run `python test_sensory_integration.py` to test the 5D sensory cortex
3. **System Testing**: Run `python main.py` to test all components
4. **Configuration**: Modify `config.yaml` for different experiments
5. **Results**: Check `experiments/` directory for outputs

## 📈 Expected Outputs

The system generates comprehensive outputs including:
- Real-time fitness metrics
- Evolution progress tracking
- Performance analysis across market regimes
- Adversarial event statistics
- Final evolved genomes
- **5D Sensory Analysis**: Detailed dimensional readings and market narratives
- **Cross-Dimensional Correlations**: How dimensions influence each other
- **Market Intelligence Levels**: From CONFUSED to PRESCIENT understanding

## 🔬 Advanced Usage

### Custom Experiments
Modify `config.yaml` to adjust:
- Population size and generations
- Adversarial intensity levels
- Market regime datasets
- Fitness function weights
- Sensory system parameters

### Data Sources
- Use `scripts/download_data.py` for real market data
- Use `scripts/create_test_data.py` for synthetic data
- Configure data sources in `configs/`

### Sensory System Customization
The 5D sensory cortex can be customized:
- Adjust dimensional weights based on market conditions
- Add new data sources for each dimension
- Modify narrative construction templates
- Fine-tune anomaly detection thresholds

## 🗺️ Development Roadmap

- [x] **Phase 0: Modular Refactor** - Complete structural reorganization
- [x] **Phase 1: Core System** - Basic modular components
- [x] **Phase 2: 5D Sensory Cortex** - Advanced multidimensional market intelligence
- [ ] **Phase 3: Financial Core** - Robust risk management and PnL engine
- [ ] **Phase 4: Advanced Features** - Enhanced adversarial engine integration
- [ ] **Phase 5: Live Integration** - Paper trading and real-world validation

## 🤝 Contributing

This is a rigorously engineered system. Please adhere to the following standards:

* **Follow the Modular Architecture**: Keep logic within the appropriate components
* **Write Tests**: All new code must be accompanied by comprehensive tests
* **Maintain Code Quality**: Use proper formatting and type checking
* **Document Everything**: Update this README and add docstrings for all public methods
* **Sensory System Integration**: When adding new features, consider how they integrate with the 5D sensory cortex

## 📚 References

* **Evolutionary Algorithms**: Genetic programming for trading strategies
* **Market Microstructure**: Realistic market simulation
* **Adversarial Testing**: Robustness evaluation techniques
* **Multi-Objective Optimization**: Pareto-optimal solution finding
* **Multidimensional Market Intelligence**: Cross-dimensional awareness and synthesis

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

* Dukascopy for providing market data
* Scientific community for evolutionary algorithm research
* Open source contributors for supporting libraries
* The multidimensional market intelligence system creator for the advanced 5D sensory architecture

---

**EMP Proving Ground v2.0** - Pushing the boundaries of evolutionary trading systems with advanced 5D multidimensional market intelligence and a clean, modular architecture.
