#!/usr/bin/env python3
"""
Daily FIX API Verification Script
Run this script daily to ensure FIX API functionality remains intact
"""

import subprocess
import sys
from datetime import datetime

def verify_fix_api():
    """Run comprehensive FIX API verification"""
    print("FIX API Verification - {}".format(datetime.now()))
    
    # Test 1: Simple connection test
    try:
        result = subprocess.run([sys.executable, "scripts/test_simplefix.py"], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("Connection test: PASSED")
        else:
            print("Connection test: FAILED")
            print(result.stderr)
            return False
    except Exception as e:
        print("Connection test error: {}".format(e))
        return False
        
    # Test 2: Configuration validation
    try:
        from src.operational.icmarkets_simplefix_application import ICMarketsSimpleFix
        app = ICMarketsSimpleFix()
        if app.load_config():
            print("Configuration validation: PASSED")
        else:
            print("Configuration validation: FAILED")
            return False
    except Exception as e:
        print("Configuration validation error: {}".format(e))
        return False
        
    print("All FIX API verification tests PASSED")
    return True

if __name__ == "__main__":
    success = verify_fix_api()
    sys.exit(0 if success else 1)
