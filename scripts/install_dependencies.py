#!/usr/bin/env python3
"""
Dependency installation script with compatibility fixes.
"""

import subprocess
import sys
import os

def run_command(cmd):
    """Run a command and return success status."""
    try:
        subprocess.check_call(cmd, shell=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False

def main():
    """Install dependencies with compatibility fixes."""
    print("Installing dependencies with compatibility fixes...")
    
    # Install core dependencies first
    core_deps = [
        "numpy>=1.21.0,<2.0.0",
        "pandas>=1.3.0,<3.0.0",
        "scikit-learn>=1.0.0,<2.0.0",
        "matplotlib>=3.5.0,<4.0.0",
        "seaborn>=0.11.0,<1.0.0",
        "plotly>=5.0.0,<6.0.0",
    ]
    
    # Install MLOps dependencies
    mlops_deps = [
        "mlflow>=2.8.0,<4.0.0",
        "pandas-ta>=0.3.14,<1.0.0",
        "torch>=2.0.0,<3.0.0",
        "tensorflow>=2.12.0,<3.0.0",
        "xgboost>=1.7.0,<3.0.0",
        "lightgbm>=3.3.0,<5.0.0",
        "optuna>=3.0.0,<5.0.0",
    ]
    
    # Install other dependencies
    other_deps = [
        "fastapi>=0.70.0,<1.0.0",
        "uvicorn>=0.15.0,<1.0.0",
        "pydantic>=1.8.0,<3.0.0",
        "sqlalchemy>=1.4.0,<3.0.0",
        "alembic>=1.7.0,<2.0.0",
        "psycopg2-binary>=2.9.0",
        "redis>=4.0.0,<6.0.0",
        "simplefix>=1.0.17",
        "python-dotenv>=0.19.0",
        "requests>=2.25.0,<3.0.0",
        "aiohttp>=3.8.0,<4.0.0",
        "yfinance>=0.1.70,<1.0.0",
        "pytest>=6.2.0,<8.0.0",
        "pytest-asyncio>=0.16.0,<1.0.0",
        "black>=21.9b0,<25.0.0",
        "mypy>=0.910,<2.0.0",
        "ruff>=0.5.4,<1.0.0",
    ]
    
    all_deps = core_deps + mlops_deps + other_deps
    
    for dep in all_deps:
        print(f"Installing {dep}...")
        if not run_command(f"{sys.executable} -m pip install {dep}"):
            print(f"Warning: Failed to install {dep}, continuing...")
    
    print("\n✅ Dependency installation completed!")
    print("\nNote: cTrader API needs manual installation:")
    print("1. Download from: https://ctrader.com/developers/api")
    print("2. Install the downloaded wheel file manually")

if __name__ == "__main__":
    main()
