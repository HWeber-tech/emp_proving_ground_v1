"""
Main CLI Application - Ticket UI-01

Provides command-line interface for system control and monitoring.
"""

import asyncio
import typer
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# Import our UIManager
from ..ui_manager import ui_manager

# Create Typer app
app = typer.Typer(
    name="emp",
    help="EMP Trading System Command Line Interface",
    add_completion=False
)

console = Console()


@app.command()
def run(
    config: Optional[str] = typer.Option(
        None,
        "--config", "-c",
        help="Configuration file path"
    )
):
    """Start the main application event loop"""
    rprint(Panel.fit(
        "[bold green]🚀 Starting EMP Trading System[/bold green]",
        subtitle="Event-driven autonomous trading system"
    ))
    
    async def start_system():
        success = await ui_manager.initialize()
        if success:
            rprint("[green]✅ System initialized successfully[/green]")
            rprint("[yellow]🔄 Running event loop... Press Ctrl+C to stop[/yellow]")
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                rprint("\n[yellow]🛑 Shutting down...[/yellow]")
                await ui_manager.shutdown()
        else:
            rprint("[red]❌ Failed to initialize system[/red]")
    
    asyncio.run(start_system())


@app.command()
def status():
    """Query key services and print their status"""
    system_status = ui_manager.get_system_status()
    
    # Create status table
    table = Table(title="System Status")
    table.add_column("Service", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Value", style="yellow")
    
    table.add_row("Event Bus", 
                 "Connected" if system_status["event_bus_connected"] else "Disconnected",
                 str(system_status["event_bus_connected"]))
    table.add_row("Total Strategies", "Active", str(system_status["total_strategies"]))
    table.add_row("Active Strategies", "Active", str(system_status["active_strategies"]))
    
    console.print(table)
    
    # Additional info
    rprint(f"\n[dim]Last updated: {system_status['timestamp']}[/dim]")


@app.command()
def strategies():
    """Strategy management commands"""
    pass


@strategies.command("list")
def list_strategies():
    """List all strategies with their status"""
    strategies = ui_manager.list_strategies()
    
    if not strategies:
        rprint("[yellow]No strategies found[/yellow]")
        return
    
    # Create strategy table
    table = Table(title="Strategies")
    table.add_column("ID", style="cyan", max_width=30)
    table.add_column("Status", style="green")
    table.add_column("Created", style="yellow")
    table.add_column("Config", style="dim", max_width=40)
    
    for strategy in strategies:
        table.add_row(
            strategy.get("id", "N/A"),
            strategy.get("status", "unknown"),
            strategy.get("created_at", "N/A")[:19],
            str(strategy.get("config", {}))[:37] + "..."
        )
    
    console.print(table)
    rprint(f"\n[dim]Total: {len(strategies)} strategies[/dim]")


@strategies.command("approve")
def approve_strategy(
    strategy_id: str = typer.Argument(..., help="Strategy ID to approve")
):
    """Approve an evolved strategy for live trading"""
    success = ui_manager.approve_strategy(strategy_id)
    
    if success:
        rprint(f"[green]✅ Strategy {strategy_id} approved[/green]")
    else:
        rprint(f"[red]❌ Failed to approve strategy {strategy_id}[/red]")


@strategies.command("activate")
def activate_strategy(
    strategy_id: str = typer.Argument(..., help="Strategy ID to activate")
):
    """Activate an approved strategy for live trading"""
    success = ui_manager.activate_strategy(strategy_id)
    
    if success:
        rprint(f"[green]✅ Strategy {strategy_id} activated[/green]")
    else:
        rprint(f"[red]❌ Failed to activate strategy {strategy_id}[/red]")


@strategies.command("deactivate")
def deactivate_strategy(
    strategy_id: str = typer.Argument(..., help="Strategy ID to deactivate")
):
    """Deactivate an active strategy"""
    success = ui_manager.deactivate_strategy(strategy_id)
    
    if success:
        rprint(f"[green]✅ Strategy {strategy_id} deactivated[/green]")
    else:
        rprint(f"[red]❌ Failed to deactivate strategy {strategy_id}[/red]")


@strategies.command("details")
def strategy_details(
    strategy_id: str = typer.Argument(..., help="Strategy ID to inspect")
):
    """Get detailed information about a specific strategy"""
    details = ui_manager.get_strategy_details(strategy_id)
    
    if not details:
        rprint(f"[red]❌ Strategy {strategy_id} not found[/red]")
        return
    
    # Create details panel
    panel = Panel(
        f"""
[b]Strategy ID:[/b] {details.get('id', 'N/A')}
[b]Status:[/b] {details.get('status', 'N/A')}
[b]Created:[/b] {details.get('created_at', 'N/A')}
[b]Config:[/b] {json.dumps(details.get('config', {}), indent=2)}
        """,
        title=f"Strategy Details: {strategy_id}",
        border_style="blue"
    )
    
    console.print(panel)


@app.command()
def monitor(
    duration: int = typer.Option(
        10,
        "--duration", "-d",
        help="Monitoring duration in seconds"
    )
):
    """Monitor system events for a specified duration"""
    rprint(f"[yellow]🔍 Monitoring system for {duration} seconds...[/yellow]")
    
    async def monitor_system():
        await ui_manager.initialize()
        
        try:
            for i in range(duration):
                status = ui_manager.get_system_status()
                console.clear()
                
                # Display status
                table = Table(title=f"System Monitor - {i+1}/{duration}s")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")
                
                table.add_row("Event Bus", str(status["event_bus_connected"]))
                table.add_row("Total Strategies", str(status["total_strategies"]))
                table.add_row("Active Strategies", str(status["active_strategies"]))
                
                console.print(table)
                
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            rprint("\n[yellow]🛑 Monitoring stopped[/yellow]")
        finally:
            await ui_manager.shutdown()
    
    asyncio.run(monitor_system())


if __name__ == "__main__":
    app()
