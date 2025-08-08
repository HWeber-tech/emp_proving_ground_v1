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

# Set up environment variables (copy template and edit values)
cp env_templates/.env.example .env
# Edit .env and provide real credentials before running anything live

# Configure development parameters
cp config/trading/icmarkets_config.py.example config/trading/icmarkets_config.py

# Run development tests
python -m pytest tests/

# Note: Main production entry point is for development testing only
python main_production.py
```

## Configuration

This project uses environment variables for configuration via a typed Pydantic `SystemConfig`.

1) Copy the example env file:

```bash
cp env_templates/.env.example .env
```

2) Edit `.env` and set your values. Key variables:

- `RUN_MODE` (default `paper`)
- `CONFIRM_LIVE` (must be `true` to enable live)
- `EMP_KILL_SWITCH` (path to kill-switch file)
- `EMP_ENVIRONMENT` (`demo` or `production`)
- `CONNECTION_PROTOCOL` (`fix` or `openapi`)
- `EMP_TIER` (`tier_0`, `tier_1`, `tier_2`)
- `ICMARKETS_ACCOUNT`, `ICMARKETS_PASSWORD`

3) Tier-0 ingest flags in `main.py`:

- `--symbols` Comma-separated symbols for ingest (default: `EURUSD,GBPUSD`)
- `--db` DuckDB path (default: `data/tier0.duckdb`)
- `--skip-ingest` Skip Tier-0 ingest at startup

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
