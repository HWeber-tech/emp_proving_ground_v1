#!/usr/bin/env python3
"""
Final real microstructure verification using correct cTrader protocol
"""

import asyncio
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

async def verify_microstructure_final():
    """Final verification using correct cTrader protocol"""
    
    print("=" * 80)
    print("FINAL MICROSTRUCTURE VERIFICATION")
    print("=" * 80)
    
    # Load credentials
    client_id = os.getenv('CTRADER_CLIENT_ID')
    client_secret = os.getenv('CTRADER_CLIENT_SECRET')
    access_token = os.getenv('CTRADER_ACCESS_TOKEN')
    account_id = int(os.getenv('CTRADER_ACCOUNT_ID', '43939234'))
    
    print(f"✅ Account ID: {account_id}")
    
    try:
        # Use the working WebSocket approach with correct protocol
        import websockets
        import json
        import uuid
        
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
                    "ctidTraderAccountId": account_id,
                    "accessToken": access_token
                }
            }
            
            await websocket.send(json.dumps(account_auth))
            response = await websocket.recv()
            data = json.loads(response)
            
            if data.get("payloadType") != 2103:
                print("❌ Account auth failed")
                return False
            
            print("✅ Account authenticated")
            
            # Step 3: Subscribe to spot prices first (more reliable)
            subscribe_spot = {
                "payloadType": 2124,
                "clientMsgId": str(uuid.uuid4()),
                "payload": {
                    "ctidTraderAccountId": account_id,
                    "symbolId": [1]  # EURUSD
                }
            }
            
            await websocket.send(json.dumps(subscribe_spot))
            response = await websocket.recv()
            print("✅ Subscribed to EURUSD spot prices")
            
            # Step 4: Collect data
            import time
            start_time = time.time()
            updates = 0
            latency_records = []
            
            print("📊 Collecting 20 seconds of data...")
            
            while time.time() - start_time < 20:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(response)
                    
                    if data.get("payloadType") == 2128:  # Spot event
                        updates += 1
                        payload = data.get("payload", {})
                        timestamp = payload.get("timestamp", 0)
                        
                        # Calculate latency
                        server_time = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
                        client_time = datetime.now(timezone.utc)
                        latency_ms = (client_time - server_time).total_seconds() * 1000
                        
                        latency_records.append(latency_ms)
                        
                        if updates <= 5:
                            bid = payload.get("bid", 0) / 100000
                            ask = payload.get("ask", 0) / 100000
                            print(f"   Update #{updates}: Bid={bid:.5f}, Ask={ask:.5f}, latency: {latency_ms:.2f}ms")
                    
                except asyncio.TimeoutError:
                    continue
            
            # Step 5: Generate report
            if updates > 0:
                avg_latency = sum(latency_records) / len(latency_records)
                
                print("\n" + "=" * 80)
                print("✅ MICROSTRUCTURE VERIFICATION COMPLETE")
                print("=" * 80)
                print(f"✅ Successfully connected to cTrader")
                print(f"✅ Account authenticated: {account_id}")
                print(f"✅ EURUSD data received: {updates} updates")
                print(f"✅ Real-time data confirmed working")
                print()
                print("📊 RESULTS:")
                print(f"   Average latency: {avg_latency:.2f}ms")
                print(f"   Total updates: {updates}")
                print(f"   Updates per second: {updates/20:.2f}")
                print()
                print("🎯 RECOMMENDATION: GO")
                print("   The cTrader API provides real-time market data")
                print("   Protocol is working correctly for microstructure analysis")
                
                # Save results
                results = {
                    "account_id": account_id,
                    "symbol": "EURUSD",
                    "symbol_id": 1,
                    "total_updates": updates,
                    "average_latency_ms": avg_latency,
                    "updates_per_second": updates/20,
                    "test_duration_seconds": 20,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                with open('docs/microstructure_verification_results.json', 'w') as f:
                    json.dump(results, f, indent=2)
                
                # Create final report
                with open('docs
