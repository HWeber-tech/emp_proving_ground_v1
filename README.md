# EMP Proving Ground v1

An algorithmic trading system framework in active development.

## Current Status: Development Phase

⚠️ **This system is currently in development and contains primarily mock implementations.**

## Architecture Components

- **FIX API Integration**: Basic connectivity framework (authentication working)
- **Genetic Evolution Engine**: Framework with abstract interfaces
- **Core Architecture**: Exception handling and validation frameworks
- **Risk Management**: Interface definitions and basic implementations
- **Data Processing**: Market data handling framework

## Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure development parameters
cp config/trading/icmarkets_config.py.example config/trading/icmarkets_config.py

# Run development tests
python -m pytest tests/

# Note: Main production entry point is for development testing only
python main_production.py
```

## Development Status

### Working Components
- FIX API authentication with IC Markets
- Basic market data connectivity
- Exception handling framework
- Abstract interface definitions

### In Development
- Genetic algorithm implementations
- Real trading strategy execution
- Risk management systems
- Production data integration

## Documentation

- [FIX API Development Guide](docs/fix_api/FIX_API_MASTER_GUIDE.md)
- [Architecture Framework](docs/architecture/)
- [Development API Reference](docs/api/)

## Development Notes

This is an active development project. The system architecture is designed for algorithmic trading but most components are currently framework implementations rather than production-ready trading systems.

## License

Proprietary - All rights reserved
