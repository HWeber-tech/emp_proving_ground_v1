#!/usr/bin/env python3
"""
Quick verification of the microstructure tool
"""

import importlib.util
import os

print("Testing Microstructure Verification Tool Structure...")

# Check files exist
files_to_check = ["scripts/verify_microstructure.py", "docs/v4_reality_check_report.md"]

all_exist = True
for file_path in files_to_check:
    if os.path.exists(file_path):
        print(f"✅ {file_path} exists")
    else:
        print(f"❌ {file_path} missing")
        all_exist = False

# Test imports availability without importing modules (avoids F401)
if importlib.util.find_spec("pandas") is not None:
    print("✅ pandas available")
else:
    print("❌ pandas not available")
    all_exist = False

if importlib.util.find_spec("argparse") is not None:
    print("✅ argparse available")
else:
    print("❌ argparse not available")
    all_exist = False

if importlib.util.find_spec("dotenv") is not None:
    print("✅ dotenv available")
else:
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
