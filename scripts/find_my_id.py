#!/usr/bin/env python3
"""
Find your correct CTRADER_ACCOUNT_ID (ctidTraderAccountId)
"""

# Load from .env file
import os

import requests
from dotenv import load_dotenv

load_dotenv()

# Get access token from .env
ACCESS_TOKEN = os.getenv("CTRADER_ACCESS_TOKEN")

if not ACCESS_TOKEN or ACCESS_TOKEN == "your_access_token_here":
    print("❌ Please set your CTRADER_ACCESS_TOKEN in .env file")
    print("   Current value:", ACCESS_TOKEN)
    exit(1)

print("🔍 Finding your cTrader account ID...")
print("=" * 60)

# Use the correct IC Markets endpoint
response = requests.get(
    url="https://connect.icmarkets.com/api/v2/tradingaccounts",
    headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
)

if response.status_code == 200:
    accounts_data = response.json().get("data", [])

    if not accounts_data:
        print("❌ No accounts found with this access token")
        print("   Your token may not have 'trading' scope")
        exit(1)

    print("✅ SUCCESS! Found the following accounts:")
    print("-" * 60)

    for i, account in enumerate(accounts_data, 1):
        account_number = account.get("accountNumber")
        is_live = "LIVE" if account.get("isLive") else "DEMO"
        broker_name = account.get("brokerName")

        # This is the ID you need!
        ctid_trader_account_id = account.get("ctidTraderAccountId")

        print(f"ACCOUNT {i}:")
        print(f"  Broker:       {broker_name}")
        print(f"  Account #:    {account_number} ({is_live})")
        print(f"  Account ID:   {ctid_trader_account_id}  <-- COPY THIS NUMBER")
        print("-" * 60)

    # Find demo account
    demo_accounts = [a for a in accounts_data if not a.get("isLive")]
    if demo_accounts:
        demo_account = demo_accounts[0]
        print("🎯 RECOMMENDED DEMO ACCOUNT:")
        print(f"  CTRADER_ACCOUNT_ID={demo_account['ctidTraderAccountId']}")
        print()
        print("📋 UPDATE YOUR .env FILE:")
        print(f"  CTRADER_ACCOUNT_ID={demo_account['ctidTraderAccountId']}")

        # Save to file for easy copy
        with open(".env.update", "w") as f:
            f.write(f"CTRADER_ACCOUNT_ID={demo_account['ctidTraderAccountId']}\n")
        print("✅ Account ID saved to .env.update")

    else:
        print("❌ No demo accounts found")

else:
    print(f"❌ ERROR! Could not get accounts.")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    print("\nPossible issues:")
    print("1. Access token is invalid or expired")
    print("2. Token doesn't have 'trading' scope")
    print("3. No accounts are linked to this token")
    print("\nTry regenerating your access token with:")
    print(
        "   https://connect.icmarkets.com/oauth/authorize?client_id=12066_pp7glRUIsDmFaeXcBzMljuAz4083XEzL4GnegdcxiVKhshAXDt&redirect_uri=http://localhost/&scope=trading"
    )
