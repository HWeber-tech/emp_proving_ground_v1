"""
CLI Interface for EMP Ultimate Architecture v1.1
Production-ready command-line interface for system management
"""

import logging
from typing import Optional

import click

from src.sensory.organs.yahoo_finance_organ import YahooFinanceOrgan

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
        organ = YahooFinanceOrgan(data_dir)
        
        click.echo(f"Downloading {symbol} data from {start_date} to {end_date or 'today'} ({interval})")
        
        success = organ.download_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )
        
        if success:
            click.echo(f"✅ Successfully downloaded {symbol} data")
            
            # Validate the data
            if organ.validate_data(symbol, interval):
                click.echo("✅ Data validation passed")
            else:
                click.echo("⚠️ Data validation warnings")
                
        else:
            click.echo(f"❌ Failed to download {symbol} data")
            
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        logger.error(f"Error downloading data: {e}")


@data.command()
@click.option('--data-dir', '-d', default='data/historical', help='Data directory')
def list(data_dir: str):
    """List available historical data files"""
    try:
        organ = YahooFinanceOrgan(data_dir)
        available = organ.get_available_data()
        
        if not available:
            click.echo("No historical data files found")
            return
            
        click.echo("Available historical data files:")
        click.echo("-" * 80)
        
        for key, info in available.items():
            click.echo(f"{key}:")
            click.echo(f"  Symbol: {info['symbol']}")
            click.echo(f"  Interval: {info['interval']}")
            click.echo(f"  Rows: {info['rows']}")
            click.echo(f"  Date Range: {info['start_date']} to {info['end_date']}")
            click.echo(f"  File Size: {info['file_size_mb']:.2f} MB")
            click.echo(f"  Last Modified: {info['last_modified']}")
            click.echo()
            
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
