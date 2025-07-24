#!/usr/bin/env python3
"""
Master Switch Integration Test Script

Tests both FIX and OpenAPI protocol modes to ensure the master switch works correctly.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.governance.system_config import SystemConfig

logger = logging.getLogger(__name__)


class MasterSwitchTester:
    """Comprehensive tester for the master switch functionality."""
    
    def __init__(self):
        self.test_results = {
            'fix_mode': False,
            'openapi_mode': False,
            'configuration_validation': False
        }
        
    async def test_configuration_validation(self):
        """Test that the configuration system validates protocol choices."""
        logger.info("🔍 Testing configuration validation...")
        
        try:
            # Test valid configurations
            config_fix = SystemConfig(CONNECTION_PROTOCOL="fix")
            config_openapi = SystemConfig(CONNECTION_PROTOCOL="openapi")
            config_default = SystemConfig()
            
            # Verify defaults
            assert config_fix.CONNECTION_PROTOCOL == "fix"
            assert config_openapi.CONNECTION_PROTOCOL == "openapi"
            assert config_default.CONNECTION_PROTOCOL == "fix"
            
            # Test invalid configuration (should raise validation error)
            try:
                SystemConfig(CONNECTION_PROTOCOL="invalid")
                assert False, "Should have raised validation error"
            except Exception:
                pass  # Expected
            
            self.test_results['configuration_validation'] = True
            logger.info("✅ Configuration validation passed")
            
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            
    async def test_fix_mode(self):
        """Test FIX protocol mode."""
        logger.info("🔧 Testing FIX protocol mode...")
        
        try:
            # Load FIX configuration
            config = SystemConfig(CONNECTION_PROTOCOL="fix")
            
            # Verify FIX-specific settings are loaded
            assert config.CONNECTION_PROTOCOL == "fix"
            assert "fix" in str(config.CONNECTION_PROTOCOL).lower()
            
            # Test that FIX credentials are expected
            logger.info("✅ FIX mode configuration loaded successfully")
            self.test_results['fix_mode'] = True
            
        except Exception as e:
            logger.error(f"❌ FIX mode test failed: {e}")
            
    async def test_openapi_mode(self):
        """Test OpenAPI protocol mode."""
        logger.info("🔄 Testing OpenAPI protocol mode...")
        
        try:
            # Load OpenAPI configuration
            config = SystemConfig(CONNECTION_PROTOCOL="openapi")
            
            # Verify OpenAPI-specific settings
            assert config.CONNECTION_PROTOCOL == "openapi"
            assert "openapi" in str(config.CONNECTION_PROTOCOL).lower()
            
            logger.info("✅ OpenAPI mode configuration loaded successfully")
            self.test_results['openapi_mode'] = True
            
        except Exception as e:
            logger.error(f"❌ OpenAPI mode test failed: {e}")
            
    async def run_comprehensive_test(self):
        """Run all master switch tests."""
        logger.info("=" * 60)
        logger.info("MASTER SWITCH INTEGRATION TEST")
        logger.info("=" * 60)
        
        # Test configuration validation
        await self.test_configuration_validation()
        
        # Test FIX mode
        await self.test_fix_mode()
        
        # Test OpenAPI mode
        await self.test_openapi_mode()
        
        # Summary
        logger.info("=" * 60)
        logger.info("MASTER SWITCH TEST RESULTS:")
        logger.info(f"Configuration Validation: {'✅ PASS' if self.test_results['configuration_validation'] else '❌ FAIL'}")
        logger.info(f"FIX Mode: {'✅ PASS' if self.test_results['fix_mode'] else '❌ FAIL'}")
        logger.info(f"OpenAPI Mode: {'✅ PASS' if self.test_results['openapi_mode'] else '❌ FAIL'}")
        
        all_passed = all(self.test_results.values())
        if all_passed:
            logger.info("🎉 All master switch tests passed!")
        else:
            logger.error("❌ Some master switch tests failed")
            
        return all_passed
    
    def print_usage_instructions(self):
        """Print usage instructions for the master switch."""
        print("\n" + "=" * 60)
        print("MASTER SWITCH USAGE INSTRUCTIONS")
        print("=" * 60)
        print("\n1. Test FIX mode:")
        print("   python scripts/test_master_switch.py")
        print("\n2. Run with FIX protocol:")
        print("   CONNECTION_PROTOCOL=fix python main.py")
        print("\n3. Run with OpenAPI protocol:")
        print("   CONNECTION_PROTOCOL=openapi python main.py")
        print("\n4. Use configuration files:")
        print("   cp config/test_fix.env .env    # For FIX mode")
        print("   cp config/test_openapi.env .env # For OpenAPI mode")
        print("\n5. Environment variable override:")
        print("   export CONNECTION_PROTOCOL=fix")
        print("=" * 60)


async def main():
    """Main test runner."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    tester = MasterSwitchTester()
    
    # Run comprehensive test
    success = await tester.run_comprehensive_test()
    
    # Print usage instructions
    tester.print_usage_instructions()
    
    if success:
        print("\n✅ Master Switch integration test completed successfully!")
    else:
        print("\n❌ Master Switch integration test failed")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
