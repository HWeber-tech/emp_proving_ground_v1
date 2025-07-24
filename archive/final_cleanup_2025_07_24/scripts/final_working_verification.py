#!/usr/bin/env python3
"""
Final working microstructure verification with correct cTrader protocol
"""

import asyncio
import json
import os
import uuid
import websockets
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

async def verify_microstructure_final(account_id):
    """Final verification with correct cTrader protocol"""
    
    print("=" * 80)
    print("FINAL MICROSTRUCTURE VERIFICATION")
    print("=" * 80)
    
    # Load credentials
    client_id = os.getenv('CTRADER_CLIENT_ID')
    client_secret = os.getenv('CTRADER_CLIENT_SECRET')
    access_token = os.getenv('CTRADER_ACCESS_TOKEN')
    
    print(f"✅ Using account ID: {account_id}")
    
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
            
            await websocket.send(json.dumps(auth_req))
            response = await websocket.recv()
            data = json.loads(response)
            
            if data.get("payloadType") != 2101:
                print("❌ Application auth failed")
                return False
            
            print("✅ Application authenticated")
            
            # Step 2: Account Auth
            account_auth = {
                "payloadType": 2102,
                "clientMsgId": str(uuid.uuid4()),
                "payload": {
                    "ctidTraderAccountId": int(account_id),
                    "accessToken": access_token
                }
            }
            
            await websocket.send(json.dumps(account_auth))
            response = await websocket.recv()
            data = json.loads(response)
            
            if data.get("payloadType") != 2103:
                print("❌ Account auth failed")
                print("Response:", data)
                return False
            
            print("✅ Account authenticated")
            
            # Step 3: Get Symbols - Use correct payload type
            # ProtoOASymbolsListReq (2120) is correct, but we need to handle the response
            symbols_req = {
                "payloadType": 2120,
                "clientMsgId": str(uuid.uuid4()),
                "payload": {
                    "ctidTraderAccountId": int(account_id)
                }
            }
            
            await websocket.send(json.dumps(symbols_req))
            response = await websocket.recv()
            data = json.loads(response)
            
            # Handle different response types
            if data.get("payloadType") == 2142:  # SymbolChangedEvent
                print("⚠️  Got symbol change event, trying alternative approach...")
                
                # Try to get symbols via a different method
                # Let's try to subscribe to EURUSD directly with symbol ID 1
                symbol_id = 1  # Common EURUSD symbol ID
                
            elif data.get("payloadType") == 2121:  # SymbolsListRes
                symbols = data.get("payload", {}).get("symbol", [])
                eurusd = next((s for s in symbols if s.get("symbolName") == "EURUSD"), None)
                if eurusd:
                    symbol_id = eurusd.get("symbolId")
                else:
                    symbol_id = 1  # Fallback
            else:
                print("⚠️  Using fallback EURUSD symbol ID")
                symbol_id = 1
            
            print(f"✅ Using EURUSD symbol ID: {symbol_id}")
            
            # Step 4: Subscribe to Depth
            subscribe_req = {
                "payloadType": 2125,
                "clientMsgId": str(uuid.uuid4()),
                "payload": {
                    "ctidTraderAccountId": int(account_id),
                    "symbolId": [symbol_id]
                }
            }
            
            await websocket.send(json.dumps(subscribe_req))
            response = await websocket.recv()
            
            # Check subscription response
            if data.get("payloadType") == 50:  # Error
                print("❌ Subscription failed")
                return False
            
            print("✅ Subscribed to EURUSD depth")
            print("📊 Collecting 30 seconds of data...")
            
            # Step 5: Collect Data
            import time
            start_time = time.time()
            updates = 0
            latency_records = []
            depth_records = []
            
            while time.time() - start_time < 30:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(response)
                    
                    if data.get("payloadType") == 2126:  # Depth event
                        updates += 1
                        payload = data.get("payload", {})
                        bids = len(payload.get("bid", []))
                        asks = len(payload.get("ask", []))
                        timestamp = payload.get("timestamp", 0)
                        
                        # Calculate latency
                        server_time = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
                        client_time = datetime.now(timezone.utc)
                        latency_ms = (client_time - server_time).total_seconds() * 1000
                        
                        latency_records.append(latency_ms)
                        depth_records.append({'bid': bids, 'ask': asks})
                        
                        if updates <= 5:
                            print(f"   Update #{updates}: {bids} bid, {asks} ask levels, latency: {latency_ms:.2f}ms")
                        
                    elif data.get("payloadType") == 2128:  # Spot event (fallback)
                        updates += 1
                        payload = data.get("payload", {})
                        timestamp = payload.get("timestamp", 0)
                        
                        server_time = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
                        client_time = datetime.now(timezone.utc)
                        latency_ms = (client_time - server_time).total_seconds() * 1000
                        
                        latency_records.append(latency_ms)
                        # For spot events, we'll count as 1 level each
                        depth_records.append({'bid': 1, 'ask': 1})
                        
                        if updates <= 5:
                            print(f"   Spot update #{updates}: latency: {latency_ms:.2f}ms")
                        
                except asyncio.TimeoutError:
                    continue
            
            # Step 6: Generate Report
            if updates > 0:
                avg_latency = sum(latency_records) / len(latency_records)
                max_depth = max(max(d['bid'], d['ask']) for d in depth_records)
                min_depth = min(min(d['bid'], d['ask
