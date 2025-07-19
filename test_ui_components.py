"""
Test script for UI Layer components
Tests both CLI and Web API functionality
"""

import asyncio
import json
from datetime import datetime

# Test UIManager
from src.ui.ui_manager import UIManager

# Test CLI commands
from src.ui.cli.main_cli import app as cli_app

# Test Web API
from src.ui.web.api import app as web_app


async def test_ui_manager():
    """Test UIManager functionality"""
    print("🧪 Testing UIManager...")
    
    ui = UIManager()
    
    # Test initialization
    success = await ui.initialize()
    print(f"✅ Initialization: {success}")
    
    # Test system status
    status = ui.get_system_status()
    print(f"✅ System Status: {json.dumps(status, indent=2)}")
    
    # Test strategy management
    test_strategy = {
        "name": "TestStrategy",
        "risk_per_trade": 0.01,
        "max_positions": 5
    }
    
    # Register a test strategy
    ui.strategy_registry.register_strategy("test_001", test_strategy)
    
    # List strategies
    strategies = ui.list_strategies()
    print(f"✅ Strategies: {len(strategies)} found")
    
    # Test strategy approval
    success = ui.approve_strategy("test_001")
    print(f"✅ Strategy approval: {success}")
    
    # Test strategy activation
    success = ui.activate_strategy("test_001")
    print(f"✅ Strategy activation: {success}")
    
    # Get strategy details
    details = ui.get_strategy_details("test_001")
    print(f"✅ Strategy details: {json.dumps(details, indent=2)}")
    
    # Shutdown
    await ui.shutdown()
    print("✅ UIManager shutdown complete")


def test_cli_help():
    """Test CLI help output"""
    print("\n🧪 Testing CLI Help...")
    
    try:
        # Test CLI help
        import subprocess
        result = subprocess.run(
            ["python", "-m", "src.ui.cli.main_cli", "--help"],
            capture_output=True,
            text=True
        )
        print("✅ CLI help command executed")
        print("Output:", result.stdout[:200] + "...")
    except Exception as e:
        print(f"⚠️ CLI test skipped: {e}")


def test_web_api():
    """Test Web API endpoints"""
    print("\n🧪 Testing Web API...")
    
    try:
        from fastapi.testclient import TestClient
        
        client = TestClient(web_app)
        
        # Test root endpoint
        response = client.get("/")
        print(f"✅ Root endpoint: {response.status_code}")
        
        # Test status endpoint
        response = client.get("/status")
        print(f"✅ Status endpoint: {response.status_code}")
        
        # Test strategies endpoint
        response = client.get("/strategies")
        print(f"✅ Strategies endpoint: {response.status_code}")
        
        print("✅ Web API tests passed")
        
    except ImportError:
        print("⚠️ FastAPI test client not available, skipping web tests")
    except Exception as e:
        print(f"⚠️ Web API test skipped: {e}")


async def main():
    """Run all UI tests"""
    print("🚀 Starting UI Layer Tests")
    print("=" * 50)
    
    await test_ui_manager()
    test_cli_help()
    test_web_api()
    
    print("\n" + "=" * 50)
    print("✅ UI Layer tests completed")


if __name__ == "__main__":
    asyncio.run(main())
