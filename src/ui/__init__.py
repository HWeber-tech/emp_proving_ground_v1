"""
UI Layer - Human Interface Components

This package provides human interface capabilities for the EMP system:
- Command Line Interface (CLI) for system control
- Web API for real-time monitoring
- Event monitoring and broadcasting
"""

from .cli.main_cli import app as cli_app
from .ui_manager import UIManager

__all__ = ["UIManager", "cli_app"]
