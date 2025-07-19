"""
EMP UI Layer v1.1

The UI Layer provides user interfaces for monitoring, control, and interaction
with the EMP system. It includes web dashboards, CLI tools, and API endpoints
for system management and oversight.

Architecture:
- web/: Web dashboard and API endpoints
- cli/: Command-line interface tools
- models/: View models for UI data presentation
"""

from .web import *
from .cli import *
from .models import *

__version__ = "1.1.0"
__author__ = "EMP System"
__description__ = "UI Layer - User Interfaces and System Control" 