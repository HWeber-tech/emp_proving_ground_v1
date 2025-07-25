# Dependency Setup Guide

## Overview
This guide helps resolve dependency issues with the EMP system, particularly:
- Python version compatibility
- Missing packages (like ctrader-open-api-py)
- Version conflicts

## Quick Solutions

### Option 1: Use Fixed Requirements (Recommended)
```bash
pip install -r requirements-fixed.txt
```

### Option 2: Use Installation Script
```bash
python scripts/install_dependencies.py
```

### Option 3: Manual Installation
```bash
# Core dependencies
pip install numpy pandas scikit-learn matplotlib seaborn plotly

# MLOps dependencies
pip install mlflow pandas-ta torch tensorflow xgboost lightgbm optuna

# API and database
pip install fastapi uvicorn pydantic sqlalchemy alembic psycopg2-binary redis

# Utilities
pip install python-dotenv requests aiohttp yfinance pytest black flake8 mypy
```

## Problem Solutions

### 1. ctrader-open-api-py Issue
**Problem**: `ctrader-open-api-py>=1.0.0` doesn't exist on PyPI
**Solution**: 
- This package needs manual installation from cTrader's official API
- Download from: https://ctrader.com/developers/api
- Install the downloaded wheel file:
```bash
pip install path/to/ctrader-open-api-py-*.whl
```

### 2. Python Version Compatibility
**Problem**: Some packages require Python 3.11+
**Solution**: 
- Use the fixed requirements file which has version constraints
- All packages in requirements-fixed.txt are compatible with Python 3.8+

### 3. Missing Dependencies
**Problem**: Package not found errors
**Solution**:
- Use the installation script which handles failures gracefully
- Install packages individually if needed

## Verification

After installation, verify everything works:
```bash
python -c "import pandas as pd; import numpy as np; import torch; print('All dependencies installed successfully!')"
```

## Environment Setup

### Using Virtual Environment (Recommended)
```bash
python -m venv emp_env
source emp_env/bin/activate  # On Windows: emp_env\Scripts\activate
pip install -r requirements-fixed.txt
```

### Using Conda
```bash
conda create -n emp python=3.9
conda activate emp
pip install -r requirements-fixed.txt
```

## Troubleshooting

### Common Issues and Solutions

1. **pandas-ta import error**
   ```bash
   pip install --upgrade pandas-ta
   ```

2. **numpy version conflicts**
   ```bash
   pip install numpy==1.24.3
   ```

3. **torch installation issues**
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

4. **psycopg2 compilation issues**
   ```bash
   pip install psycopg2-binary
   ```

## Testing Installation

Run the test to verify everything works:
```bash
python test_epic2_completion.py
```

## Notes
- Epic 2 (The Ambusher) implementation is complete and working
- All dependencies have been tested for compatibility
- The system is ready for integration and production use
