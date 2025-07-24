#!/usr/bin/env python3
"""
Test script to validate real cTrader credentials and discover correct account ID
"""

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
import websockets
from dotenv import load_dotenv

load_dotenv()

async def test_credentials():
    """Test real credentials and discover correct account ID"""
    
    print("=" * 80)
    print("TESTING REAL CTRADER CREDENTIALS")
    print("=" * 80)
    
    # Load credentials
    client_id = os.getenv('CTRADER_CLIENT_ID')
    client_secret = os.getenv('CTRADER_CLIENT_SECRET')
    access_token = os.getenv('CTRADER_ACCESS_TOKEN')
    
    if not all([client_id, client_secret, access_token]):
        print("❌ Missing credentials in .env file")
        return
    
    print(f"✅ Credentials loaded:")
    print(f"   Client ID: {client_id[:15]}...")
    print(f"   Access Token: {access_token[:15]}...")
    
    try:
        # Connect to demo endpoint
        uri = "wss://demo.ctraderapi.com:5036"
        
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connection established")
            
            # Step 1: Authenticate application
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
            
            if data.get("payloadType") == 2101:
                print("✅ Application authentication successful")
            else:
                print(f"❌ Application auth failed: {data}")
                return
            
            # Step 2: Get account list
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
            
            if data.get("payloadType") == 2105:
                accounts = data.get("payload", {}).get("ctidTraderAccount", [])
                
                if not accounts:
                    print("❌ No accounts found for this access token")
                    print("   Possible causes:")
                    print("   1. Access token doesn't include trading scope")
                    print("   2. Account not selected during OAuth consent")
                    print("   3. Using wrong endpoint (demo vs live)")
                    return
                
                print("✅ Found accounts:")
                print("-" * 60)
                
                for account in accounts:
                    account_id = account.get("ctidTraderAccountId")
                    account_number = account.get("accountNumber")
                    is_live = account.get("isLive", False)
                    balance = account.get("balance", 0)
                    money_digits = account.get("moneyDigits", 2)
                    
                    real_balance = balance / (10 ** money_digits)
                    
                    print(f"Account ID: {account_id}")
                    print(f"Account Number: {account_number}")
                    print(f"Type: {'LIVE' if is_live else 'DEMO'}")
                    print(f"Balance: {real_balance:.2f}")
                    print("-" * 40)
                
                # Use the first demo account
                demo_accounts = [a for a in accounts if not a.get("isLive", False)]
                if demo_accounts:
                    correct_account_id = demo_accounts[0]["ctidTraderAccountId"]
                    print(f"✅ Recommended account ID: {correct_account_id}")
                    
                    # Test account authentication
                    account_auth_req = {
                        "payloadType": 2102,
                        "clientMsgId": str(uuid.uuid4()),
                        "payload": {
                            "ctidTraderAccountId": correct_account_id,
                            "accessToken": access_token
                        }
                    }
                    
                    await websocket.send(json.dumps(account_auth_req))
                    response = await websocket.recv()
                    data = json.loads(response)
                    
                    if data.get("payloadType") == 2103:
                        print("✅ Account authentication successful")
                        
                        # Test symbol loading
                        symbols_req = {
                            "payloadType": 2120,
                            "clientMsgId": str(uuid.uuid4()),
                            "payload": {
                                "ctidTraderAccountId": correct_account_id
                            }
                        }
                        
                        await websocket.send(json.dumps(symbols_req))
                        response = await websocket.recv()
                        data = json.loads(response)
                        
                        if data.get("payloadType") == 2121:
                            symbols = data.get("payload", {}).get("symbol", [])
                            eurusd_symbol = next((s for s in symbols if s.get("symbolName") == "EURUSD"), None)
                            
                            if eurusd_symbol:
                                symbol_id = eurusd_symbol.get("symbolId")
                                print(f"✅ EURUSD found with symbol ID: {symbol_id}")
                                
                                # Test depth subscription
                                subscribe_req = {
                                    "payloadType": 2125,
                                    "clientMsgId": str(uuid.uuid4()),
                                    "payload": {
                                        "ctidTraderAccountId": correct_account_id,
                                        "symbolId": [symbol_id]
                                    }
                                }
                                
                                await websocket.send(json.dumps(subscribe_req))
                                print("✅ Depth subscription request sent")
                                
                                # Wait for a few depth updates
                                print("✅ Waiting for depth data...")
                                for i in range(3):
                                    try:
                                        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                                        data = json.loads(response)
                                        if data.get("payloadType") == 2126:
                                            payload = data.get("payload", {})
                                            bids = len(payload.get("bid", []))
                                            asks = len(payload.get("ask", []))
                                            print(f"   Depth update: {bids} bid levels, {asks} ask levels")
                                    except asyncio.TimeoutError:
                                        break
                                
                                print("=" * 80)
                                print("✅ CREDENTIALS VALIDATION SUCCESSFUL")
                                print(f"   Use account ID: {correct_account_id}")
                                print(f"   EURUSD symbol ID: {symbol_id}")
                                print("=" * 80)
                                
                            else:
                                print("❌ EURUSD not found in symbols")
                        
                    else:
                        print(f"❌ Account auth failed: {data}")
                        
            else:
                print(f"❌ Account list request failed: {data}")
                
    except Exception as e:
        print(f"❌ Connection error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_credentials())
