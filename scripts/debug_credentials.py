#!/usr/bin/env python3
"""
Debug credential issues with detailed logging
"""

import asyncio
import json
import os
import uuid
import websockets
from dotenv import load_dotenv

load_dotenv()

async def debug_credentials():
    """Debug credential issues with full response logging"""
    
    print("=" * 80)
    print("CREDENTIAL DEBUG - FULL RESPONSE LOGGING")
    print("=" * 80)
    
    # Load credentials
    client_id = os.getenv('CTRADER_CLIENT_ID')
    client_secret = os.getenv('CTRADER_CLIENT_SECRET')
    access_token = os.getenv('CTRADER_ACCESS_TOKEN')
    
    print(f"Client ID: {client_id}")
    print(f"Client Secret: {client_secret[:10]}..." if client_secret else "None")
    print(f"Access Token: {access_token[:20]}..." if access_token else "None")
    
    if not all([client_id, client_secret, access_token]):
        print("❌ Missing credentials")
        return
    
    try:
        uri = "wss://demo.ctraderapi.com:5036"
        
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to demo.ctraderapi.com:5036")
            
            # Step 1: Application Auth
            auth_req = {
                "payloadType": 2100,
                "clientMsgId": str(uuid.uuid4()),
                "payload": {
                    "clientId": client_id,
                    "clientSecret": client_secret
                }
            }
            
            print("\n📤 Sending app auth:")
            print(json.dumps(auth_req, indent=2))
            
            await websocket.send(json.dumps(auth_req))
            response = await websocket.recv()
            data = json.loads(response)
            
            print("\n📥 App auth response:")
            print(json.dumps(data, indent=2))
            
            if data.get("payloadType") != 2101:
                print("❌ Application auth failed")
                return
            
            print("✅ Application authenticated")
            
            # Step 2: Account Discovery
            accounts_req = {
                "payloadType": 2104,
                "clientMsgId": str(uuid.uuid4()),
                "payload": {
                    "accessToken": access_token
                }
            }
            
            print("\n📤 Sending account discovery:")
            print(json.dumps(accounts_req, indent=2))
            
            await websocket.send(json.dumps(accounts_req))
            response = await websocket.recv()
            data = json.loads(response)
            
            print("\n📥 Account discovery response:")
            print(json.dumps(data, indent=2))
            
            # Check for specific error codes
            if data.get("payloadType") == 50:  # Error response
                error_code = data.get("payload", {}).get("errorCode")
                error_desc = data.get("payload", {}).get("description")
                print(f"\n❌ ERROR: {error_code} - {error_desc}")
                return
            
            accounts = data.get("payload", {}).get("ctidTraderAccount", [])
            
            if accounts:
                print(f"\n✅ Found {len(accounts)} accounts:")
                for i, account in enumerate(accounts, 1):
                    print(f"\nAccount {i}:")
                    for key, value in account.items():
                        print(f"  {key}: {value}")
            else:
                print("\n❌ No accounts found")
                print("Response structure:", list(data.keys()))
                print("Payload:", data.get("payload", {}))
                
    except Exception as e:
        print(f"❌ Connection error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_credentials())
