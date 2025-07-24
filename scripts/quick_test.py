#!/usr/bin/env python3
"""
Quick verification of the microstructure tool
"""

import os
import sys

print("Testing Microstructure Verification Tool Structure...")

# Check files exist
files_to_check = [
    'scripts/verify_microstructure.py',
    'docs/v4_reality_check_report.md'
]

all_exist = True
for file_path in files_to_check:
    if os.path.exists(file_path):
        print(f"✅ {file_path} exists")
    else:
        print(f"❌ {file_path} missing")
        all_exist = False

# Test imports
try:
    import pandas as pd
    print("✅ pandas available")
except ImportError:
    print("❌ pandas not available")
    all_exist = False

try:
    import argparse
    print("✅ argparse available")
except ImportError:
    print("❌ argparse not available")
    all_exist = False

try:
    from dotenv import load_dotenv
    print("✅ dotenv available")
except ImportError:
    print("❌ dotenv not available")
    all_exist = False

if all_exist:
    print("\n🎉 Tool structure verified successfully!")
    print("\n📋 Ready to use:")
    print("   1. Set up .env file with cTrader credentials")
    print("   2. Run: python scripts/verify_microstructure.py")
    print("   3. Check docs/v4_reality_check_report.md for results")
else:
    print("\n💥 Some components missing")
