#!/usr/bin/env python3
"""
Simple Master Switch Configuration Test

Tests the configuration system without complex imports.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_configuration():
    """Test the master switch configuration."""
    print("=" * 60)
    print("MASTER SWITCH CONFIGURATION TEST")
    print("=" * 60)
    
    try:
        # Test configuration loading
        from src.governance.system_config import SystemConfig
        
        # Test 1: Default configuration
        print("🔍 Testing default configuration...")
        config_default = SystemConfig()
        assert config_default.CONNECTION_PROTOCOL == "fix"
        print("✅ Default protocol is 'fix'")
        
        # Test 2: FIX protocol
        print("🔧 Testing FIX protocol...")
        config_fix = SystemConfig(CONNECTION_PROTOCOL="fix")
        assert config_fix.CONNECTION_PROTOCOL == "fix"
        print("✅ FIX protocol configured")
        
        # Test 3: OpenAPI protocol
        print("🔄 Testing OpenAPI protocol...")
        config_openapi = SystemConfig(CONNECTION_PROTOCOL="openapi")
        assert config_openapi.CONNECTION_PROTOCOL == "openapi"
        print("✅ OpenAPI protocol configured")
        
        # Test 4: Environment variable loading
        print("🌍 Testing environment variable loading...")
        os.environ['CONNECTION_PROTOCOL'] = 'openapi'
        config_env = SystemConfig()
        assert config_env.CONNECTION_PROTOCOL == "openapi"
        print("✅ Environment variable override works")
        
        # Reset environment
        os.environ.pop('CONNECTION_PROTOCOL', None)
        
        print("=" * 60)
        print("🎉 ALL CONFIGURATION TESTS PASSED!")
        print("=" * 60)
        
        print("\nConfiguration Summary:")
        print(f"  Default: {config_default.CONNECTION_PROTOCOL}")
        print(f"  FIX Mode: {config_fix.CONNECTION_PROTOCOL}")
        print(f"  OpenAPI Mode: {config_openapi.CONNECTION_PROTOCOL}")
        print(f"  Env Override: {config_env.CONNECTION_PROTOCOL}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_configuration()
    if success:
        print("\n✅ Master Switch configuration is ready for production!")
    else:
        print("\n❌ Master Switch configuration needs attention")
        sys.exit(1)
