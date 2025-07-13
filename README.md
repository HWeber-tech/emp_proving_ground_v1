# EMP Proving Ground v2.0 - Pilot Experiment
## Operation Phased Explosion - Phase 1: Shake-down Cruise

This repository contains the complete implementation of the EMP Proving Ground v2.0 pilot experiment as specified in the Prime Directive. The system implements an intelligent adversarial engine with triathlon evaluation across three distinct market regimes.

## 🎯 Objective

To validate the stability of our v2.0 system and confirm that our evolutionary process can generate a positive fitness trend from a random starting population before committing to a large-scale run.

## 🏗️ Architecture Overview

### Core Components
- **Intelligent Adversarial Engine**: Context-aware stop hunting and breakout trap spoofing
- **4D+1 Sensory Cortex**: Multi-dimensional market perception system
- **Decision Tree Genomes**: Evolvable trading logic representation
- **Triathlon Evaluation**: Testing across trending, ranging, and volatile market regimes
- **Multi-Objective Fitness**: Sortino, Calmar, profit factor, consistency, and robustness metrics

### v2.0 Features
- ✅ Liquidity zone detection for intelligent stop hunting
- ✅ Breakout trap spoofing with consolidation analysis
- ✅ Triathlon evaluation across three market regimes
- ✅ Multi-objective fitness with anti-overfitting penalties
- ✅ Parallel execution with checkpointing and real-time logging

## 📋 Prerequisites

- Python 3.8+
- Required packages (see `requirements.txt`)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test Configuration
```bash
python test_config.py
```
This validates that the configuration system loads correctly and creates experiment directories.

### 3. Run Pilot Experiment
```bash
python pilot_executor.py
```

## ⚙️ Configuration

The experiment is fully parameterized through `pilot_config.yaml`:

### Key Parameters
- **Population Size**: 100 genomes
- **Generations**: 50 evolutionary cycles
- **Adversarial Levels**: [0.3, 0.5, 0.7] (parallel execution)
- **Regime Datasets**: 
  - Trending (2022-09-01 to 2022-10-01)
  - Ranging (2021-08-01 to 2021-09-01)
  - Volatile (2020-03-01 to 2020-04-01)

### Evolution Hyperparameters
- Elite Ratio: 15%
- Crossover Ratio: 60%
- Mutation Ratio: 25%
- Tournament Size: 4
- Max Tree Depth: 6

## 📊 Expected Outputs

### Directory Structure
```
experiments/
└── Shake-down Cruise_YYYYMMDD_HHMMSS/
    ├── checkpoints/          # Generation checkpoints
    ├── logs/                 # Detailed execution logs
    ├── metrics/              # Real-time CSV metrics
    ├── results/              # Final results and summary
    └── config.yaml           # Saved configuration
```

### Key Files
- **`metrics_0.3.csv`**, **`metrics_0.5.csv`**, **`metrics_0.7.csv`**: Real-time metrics for each difficulty level
- **`final_results.json`**: Complete experiment results
- **`experiment_summary.txt`**: Human-readable summary
- **`checkpoint_*.pkl`**: Generation checkpoints for resumption

## 🔄 Resumption Capability

The system automatically saves checkpoints every 10 generations. If interrupted, the experiment will resume from the latest checkpoint:

```bash
python pilot_executor.py
```

## ✅ Success Criteria Validation

The system automatically validates three critical criteria:

1. **System Stability**: No unhandled exceptions or crashes
2. **Fitness Progression**: At least one difficulty level achieves statistically significant fitness improvement
3. **Behavioral Sanity**: Trap rates remain within [10%, 50%] range

## 📈 Metrics Tracked

### Per Generation
- Best/Average/Worst fitness scores
- Population diversity
- Trap rate (adversarial event frequency)
- Complexity statistics
- Elite count and new genomes

### Final Analysis
- Fitness improvement across difficulty levels
- Regime-specific performance
- Anti-overfitting consistency penalties
- Overall experiment success/failure status

## 🐛 Troubleshooting

### Common Issues
1. **Missing Dependencies**: Run `pip install -r requirements.txt`
2. **Configuration Errors**: Check `pilot_config.yaml` syntax
3. **Memory Issues**: Reduce population size in config
4. **Long Execution**: Reduce generations or use checkpointing

### Debug Mode
For detailed debugging, modify `pilot_config.yaml`:
```yaml
logging:
  level: "DEBUG"
```

## 🔬 Scientific Rigor

### Reproducibility
- Fixed random seeds for deterministic results
- Complete configuration capture
- Checkpoint-based resumption

### Validation Framework
- Multi-regime testing prevents overfitting
- Anti-overfitting penalties for regime inconsistency
- Automated success criteria validation

### Data Integrity
- Real-time logging prevents data loss
- Checkpointing preserves experiment state
- Comprehensive metrics collection

## 📝 File Descriptions

| File | Purpose |
|------|---------|
| `emp_proving_ground_unified.py` | Core simulation and evolution engine |
| `pilot_config.yaml` | Experiment configuration |
| `config.py` | Configuration loader and validator |
| `pilot_executor.py` | Main execution framework |
| `test_config.py` | Configuration validation test |
| `requirements.txt` | Python dependencies |

## 🎯 Next Steps

After successful pilot execution:
1. Review `experiment_summary.txt` for overall results
2. Analyze `final_results.json` for detailed metrics
3. Examine regime-specific performance in metrics CSV files
4. If successful, proceed to full-scale experiment
5. If issues found, adjust parameters and re-run

## 📞 Support

For technical issues or questions about the implementation, refer to the inline documentation in the source files or the configuration validation output.

---

**Status**: Ready for Lead Architect review and pilot execution
**Version**: v2.0
**Last Updated**: Current 