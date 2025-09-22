#!/usr/bin/env python3
"""
Simple account discovery tool - shows exactly what accounts are available
"""

import asyncio
import json
import os
import uuid

import websockets
from dotenv import load_dotenv

load_dotenv()


async def discover_accounts():
    """Discover available accounts without needing account ID"""

    print("=" * 80)
    print("CTRADER ACCOUNT DISCOVERY")
    print("=" * 80)

    # Load credentials
    client_id = os.getenv("CTRADER_CLIENT_ID")
    client_secret = os.getenv("CTRADER_CLIENT_SECRET")
    access_token = os.getenv("CTRADER_ACCESS_TOKEN")

    if not all([client_id, client_secret, access_token]):
        print("❌ Missing credentials in .env file")
        return

    print("🔍 Discovering available accounts...")

    try:
        # Connect to demo endpoint
        uri = "wss://demo.ctraderapi.com:5036"

        async with websockets.connect(uri) as websocket:
            print("✅ Connected to demo.ctraderapi.com")

            # Authenticate application
            app_auth_req = {
                "payloadType": 2100,
                "clientMsgId": str(uuid.uuid4()),
                "payload": {"clientId": client_id, "clientSecret": client_secret},
            }

            await websocket.send(json.dumps(app_auth_req))
            response = await websocket.recv()
            data = json.loads(response)

            if data.get("payloadType") != 2101:
                print("❌ Application authentication failed")
                return

            print("✅ Application authenticated")

            # Get account list
            account_list_req = {
                "payloadType": 2104,
                "clientMsgId": str(uuid.uuid4()),
                "payload": {"accessToken": access_token},
            }

            await websocket.send(json.dumps(account_list_req))
            response = await websocket.recv()
            data = json.loads(response)

            if data.get("payloadType") != 2105:
                print("❌ Account discovery failed")
                return

            accounts = data.get("payload", {}).get("ctidTraderAccount", [])

            if not accounts:
                print("❌ No accounts found")
                print("   Your access token needs to be regenerated with 'trading' scope")
                print("   Run: python scripts/fix_credentials.py")
                return

            print("✅ Found the following accounts:")
            print("-" * 80)

            for i, account in enumerate(accounts, 1):
                account_id = account.get("ctidTraderAccountId")
                account_number = account.get("accountNumber")
                is_live = account.get("isLive", False)
                balance = account.get("balance", 0)
                money_digits = account.get("moneyDigits", 2)

                real_balance = balance / (10**money_digits)

                print(f"ACCOUNT {i}:")
                print(f"  ctidTraderAccountId: {account_id}")
                print(f"  Account Number: {account_number}")
                print(f"  Type: {'LIVE' if is_live else 'DEMO'}")
                print(f"  Balance: {real_balance:.2f}")
                print("-" * 40)

            # Find demo account
            demo_accounts = [a for a in accounts if not a.get("isLive", False)]
            if demo_accounts:
                demo_account = demo_accounts[0]
                print("🎯 USE THIS FOR YOUR .env FILE:")
                print("-" * 40)
                print(f"CTRADER_ACCOUNT_ID={demo_account['ctidTraderAccountId']}")
                print("-" * 40)
            else:
                print("❌ No demo accounts found")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(discover_accounts())
