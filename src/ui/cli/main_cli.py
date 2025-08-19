"""
CLI Interface for EMP Ultimate Architecture v1.1
Production-ready command-line interface for system management
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import pandas as pd  # type: ignore[import-untyped]

from src.data_integration.yfinance_gateway import YFinanceGateway

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _fmt_date(val: object) -> str:
    """
    Best-effort formatting for pandas/pyarrow timestamps or plain values.
    """
    try:
        if hasattr(val, "strftime"):
            # mypy: treat as datetime-like
            return getattr(val, "strftime")("%Y-%m-%d")
        return str(val)
    except Exception:
        return str(val)


@click.group()
def cli() -> None:
    """EMP Ultimate Architecture v1.1 CLI Interface"""
    # Group entry point
    return None


@cli.group()
def data() -> None:
    """Data management commands"""
    return None


@data.command()
@click.argument("symbol")
@click.option("--start-date", "-s", default="2024-01-01", help="Start date (YYYY-MM-DD)")
@click.option("--end-date", "-e", default=None, help="End date (YYYY-MM-DD, default: today)")
@click.option(
    "--interval",
    "-i",
    default="1h",
    help="Data interval (1m, 2m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)",
)
@click.option("--data-dir", "-d", default="data/historical", help="Data directory")
def download(
    symbol: str,
    start_date: str,
    end_date: Optional[str],
    interval: str,
    data_dir: str,
) -> None:
    """Download historical market data for a symbol."""
    try:
        gw = YFinanceGateway()
        click.echo(f"Downloading {symbol} data from {start_date} to {end_date or 'today'} ({interval})")

        df = gw.fetch_data(symbol=symbol, start=start_date, end=end_date, interval=interval)

        if df is None or getattr(df, "empty", True):
            click.echo(f"Failed to download {symbol} data")
            return

        out_dir = Path(data_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{symbol}_{interval}.parquet"
        filepath = out_dir / filename
        try:
            df.to_parquet(filepath, engine="pyarrow")
        except Exception:
            df.to_parquet(filepath)

        click.echo(f"Saved {symbol} data to {filepath}")

        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_columns = [c for c in required_columns if c not in df.columns]
        if missing_columns:
            click.echo(f"Warning: missing columns: {missing_columns}")
        else:
            try:
                null_count = int(df[required_columns].isnull().to_numpy().sum())
                if null_count > 0:
                    click.echo(f"Warning: found {null_count} NaN values in required columns")
                else:
                    click.echo("Data validation passed")
            except Exception:
                click.echo("Warning: validation step encountered an error")
    except Exception as e:
        click.echo(f"Error: {e}")
        logger.error("Error downloading data", exc_info=True)


@data.command(name="list")
@click.option("--data-dir", "-d", default="data/historical", help="Data directory")
def list_data(data_dir: str) -> None:  # noqa: A001 - 'list' is the CLI command name
    """List available historical data files."""
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
                rows = int(len(df))
                start_value = df["timestamp"].min() if "timestamp" in df.columns else "N/A"
                end_value = df["timestamp"].max() if "timestamp" in df.columns else "N/A"
                stem = file.stem
                parts = stem.split("_")
                symbol = parts[0] if parts else "UNKNOWN"
                interval = "_".join(parts[1:]) if len(parts) > 1 else "unknown"

                click.echo(f"{file.name}:")
                click.echo(f"  Symbol: {symbol}")
                click.echo(f"  Interval: {interval}")
                click.echo(f"  Rows: {rows}")
                click.echo(f"  Date Range: {_fmt_date(start_value)} to {_fmt_date(end_value)}")
                click.echo(f"  File Size: {stat.st_size / (1024 * 1024):.2f} MB")
                click.echo(
                    f"  Last Modified: "
                    f"{datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}"
                )
                click.echo()
            except Exception as e:
                click.echo(f"Warning: error reading {file.name}: {e}")

    except Exception as e:
        click.echo(f"Error: {e}")
        logger.error("Error listing data", exc_info=True)


@cli.command()
@click.option("--config", "-c", default="config.yaml", help="Configuration file")
@click.option(
    "--mode",
    "-m",
    default="simulation",
    type=click.Choice(["simulation", "live"]),
    help="Run mode",
)
def run(config: str, mode: str) -> None:
    """Run the EMP system."""
    try:
        click.echo(f"Starting EMP system in {mode} mode...")
        click.echo(f"Configuration: {config}")
        click.echo(f"{mode.title()} mode not yet implemented - using config: {config}")
    except Exception as e:
        click.echo(f"Error: {e}")
        logger.error("Error running system", exc_info=True)


@cli.command()
@click.option("--config", "-c", default="config.yaml", help="Configuration file")
def monitor(config: str) -> None:
    """Start monitoring service."""
    try:
        click.echo("Starting monitoring service...")
        click.echo(f"Configuration: {config}")
        click.echo(f"Monitoring service not yet implemented - using config: {config}")
    except Exception as e:
        click.echo(f"Error: {e}")
        logger.error("Error starting monitoring", exc_info=True)


@cli.command()
def version() -> None:
    """Show system version."""
    click.echo("EMP Ultimate Architecture v1.1")
    click.echo("Production Ready System")


# Export a conventional alias for external importers if needed
app = cli


if __name__ == "__main__":
    cli()
