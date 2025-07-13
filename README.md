# EMP Proving Ground v1.0

A modular, production-grade evolutionary market proving ground for the development and testing of intelligent trading agents under adversarial conditions.

## Project Structure

- `emp/` - Core library modules
  - `data_pipeline/` - Data ingestion, cleaning, and storage
  - `simulation/`   - Market simulator and adversarial engine
  - `agent/`        - Agent intelligence and decision logic
  - `evolution/`    - Evolutionary engine and fitness evaluation
- `scripts/` - Standalone scripts for data download, regime identification, etc.
- `tests/`   - Unit and integration tests
- `config.yaml` - Central configuration file
- `requirements.txt` - Python dependencies

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Download data: `python scripts/download_data.py`
3. Identify regimes: `python scripts/identify_regimes.py`
4. Run simulation/evolution: see documentation in `README.md` and scripts. 