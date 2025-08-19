"""
CLI Interface for EMP Ultimate Architecture v1.1
Production-ready command-line interface for system management
"""

from datetime import datetime
import logging
from pathlib import Path
from typing import Optional

import click
import pandas as pd

from src.data_integration.yfinance_gateway import YFinanceGateway

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """EMP Ultimate Architecture v1.1 CLI Interface"""
    pass


@cli.group()
def data():
    """Data management commands"""
    pass


@data.command()
@click.argument('symbol')
@click.option('--start-date', '-s', default='2024-01-01', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', '-e', default=None, help='End date (YYYY-MM-DD, default: today)')
@click.option('--interval', '-i', default='1h', help='Data interval (1m, 2m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)')
@click.option('--data-dir', '-d', default='data/historical', help='Data directory')
def download(symbol: str, start_date: str, end_date: Optional[str], interval: str, data_dir: str):
    """Download historical market data for a symbol"""
    try:
        gw = YFinanceGateway()
        click.echo(f"Downloading {symbol} data from {start_date} to {end_date or 'today'} ({interval})")

        # Fetch data via MarketDataGateway adapter
        df = gw.fetch_data(symbol=symbol, start=start_date, end=end_date, interval=interval)

        if df is None or getattr(df, "empty", True):
            click.echo(f"❌ Failed to download {symbol} data")
            return

        # Ensure data directory exists
        out_dir = Path(data_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save standardized parquet file
        filename = f"{symbol}_{interval}.parquet"
        filepath = out_dir / filename
        try:
            df.to_parquet(filepath, engine="pyarrow")
        except Exception:
            # Fallback to auto engine if pyarrow isn't available
            df.to_parquet(filepath)

        click.echo(f"✅ Successfully downloaded {symbol} data to {filepath}")

        # Basic validation (structure and NaN scan)
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_columns = [c for c in required_columns if c not in df.columns]
        if missing_columns:
            click.echo(f"⚠️ Missing columns: {missing_columns}")
        else:
            try:
                null_count = int(df[required_columns].isnull().to_numpy().sum())
                if null_count > 0:
                    click.echo(f"⚠️ Found {null_count} NaN values in required columns")
                else:
                    click.echo("✅ Data validation passed")
            except Exception:
                click.echo("⚠️ Validation step encountered an error")

    except Exception as e:
        click.echo(f"❌ Error: {e}")
        logger.error(f"Error downloading data: {e}")


@data.command()
@click.option('--data-dir', '-d', default='data/historical', help='Data directory')
def list(data_dir: str):
    """List available historical data files"""
    try:
        path = Path(data_dir)
        if not path.exists():
            click.echo("No historical data directory found")
            return

        files = sorted(path.glob("*.parquet"))
        if not files:
            click.echo("No historical data files found")
            return

        click.echo("Available historical data files:")
        click.echo("-" * 80)

        for file in files:
            try:
                stat = file.stat()
                df = pd.read_parquet(file)
                rows = len(df)
                start_date = df["timestamp"].min()
                end_date = df["timestamp"].max()
                # Infer symbol and interval from filename convention "<symbol>_<interval>.parquet"
                stem = file.stem
                parts = stem.split("_")
                symbol = parts[0] if parts else "UNKNOWN"
                interval = "_".join(parts[1:]) if len(parts) > 1 else "unknown"

                click.echo(f"{file.name}:")
                click.echo(f"  Symbol: {symbol}")
                click.echo(f"  Interval: {interval}")
                click.echo(f"  Rows: {rows}")
                click.echo(f"  Date Range: {getattr(start_date, 'strftime', lambda *_: str(start_date))('%Y-%m-%d') if hasattr(start_date, 'strftime') else start_date} to {getattr(end_date, 'strftime', lambda *_: str(end_date))('%Y-%m-%d') if hasattr(end_date, 'strftime') else end_date}")
                click.echo(f"  File Size: {stat.st_size / (1024 * 1024):.2f} MB")
                click.echo(f"  Last Modified: {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
                click.echo()
            except Exception as e:
                click.echo(f"⚠️ Error reading {file.name}: {e}")

    except Exception as e:
        click.echo(f"❌ Error: {e}")
        logger.error(f"Error listing data: {e}")


@cli.command()
@click.option('--config', '-c', default='config.yaml', help='Configuration file')
@click.option('--mode', '-m', default='simulation', type=click.Choice(['simulation', 'live']), help='Run mode')
def run(config: str, mode: str):
    """Run the EMP system"""
    try:
        click.echo(f"Starting EMP system in {mode} mode...")
        click.echo(f"Configuration: {config}")
        
        click.echo(f"🚀 {mode.title()} mode not yet implemented - using config: {config}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        logger.error(f"Error running system: {e}")


@cli.command()
@click.option('--config', '-c', default='config.yaml', help='Configuration file')
def monitor(config: str):
    """Start monitoring service"""
    try:
        click.echo("Starting monitoring service...")
        click.echo(f"Configuration: {config}")
        
        click.echo(f"📊 Monitoring service not yet implemented - using config: {config}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        logger.error(f"Error starting monitoring: {e}")


@cli.command()
def version():
    """Show system version"""
    click.echo("EMP Ultimate Architecture v1.1")
    click.echo("Production Ready System")


if __name__ == "__main__":
    cli()
