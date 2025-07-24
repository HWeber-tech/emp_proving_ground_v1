#!/usr/bin/env python3
"""
Get correct CTRADER_ACCOUNT_ID using WebSocket protocol
"""

import asyncio
import json
import os
import uuid
import websockets
from dotenv import load_dotenv

load_dotenv()

async def get_account_id():
    """Get account ID using WebSocket connection"""
    
    print("🔍 Getting account ID via WebSocket...")
    print("=" * 60)
    
    # Load credentials
    client_id = os.getenv('CTRADER_CLIENT_ID')
    client_secret = os.getenv('CTRADER_CLIENT_SECRET')
    access_token = os.getenv('CTRADER_ACCESS_TOKEN')
    
    if not all([client_id, client_secret, access_token]):
        print("❌ Missing credentials in .env file")
        return
    
    try:
        # Connect to demo endpoint
        uri = "wss://demo.ctraderapi.com:5036"
        
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to demo.ctraderapi.com")
            
            # Authenticate application
            app_auth_req = {
                "payloadType": 2100,
                "clientMsgId": str(uuid.uuid4()),
                "payload": {
                    "clientId": client_id,
                    "clientSecret": client_secret
                }
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
                "payload": {
                    "accessToken": access_token
                }
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
                return
            
            print("✅ Found the following accounts:")
            print("-" * 60)
            
            for i, account in enumerate(accounts, 1):
                account_id = account.get('ctidTraderAccountId')
                account_number = account.get('accountNumber')
                is_live = "LIVE" if account.get('isLive') else "DEMO"
                broker_name = account.get('brokerName')
                balance = account.get('balance', 0)
                money_digits = account.get('moneyDigits', 2)
                
                real_balance = balance / (10 ** money_digits)
                
                print(f"ACCOUNT {i}:")
                print(f"  ctidTraderAccountId: {account_id}")
                print(f"  Account Number: {account_number} ({is_live})")
                print(f"  Balance: {real_balance:.2f}")
                print("-" * 40)
            
            # Find demo account
            demo_accounts = [a for a in accounts if not a.get('isLive', False)]
            if demo_accounts:
                demo_account = demo_accounts[0]
                account_id = demo_account['ctidTraderAccountId']
                print("🎯 USE THIS FOR YOUR .env FILE:")
                print("-" * 40)
                print(f"CTRADER_ACCOUNT_ID={account_id}")
                print("-" * 40)
                
                # Update .env file
                with open('.env', 'r') as f:
                    env_content = f.read()
                
                updated_env = env_content.replace(
                    f"CTRADER_ACCOUNT_ID={os.getenv('CTRADER_ACCOUNT_ID')}",
                    f"CTRADER_ACCOUNT_ID={account_id}"
                )
                
                with open('.env', 'w') as f:
                    f.write(updated_env)
                
                print("✅ .env file updated with correct account ID")
                
            else:
                print("❌ No demo accounts found")
                
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(get_account_id())
