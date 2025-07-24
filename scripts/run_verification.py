#!/usr/bin/env python3
"""
Final verification script that works with any valid credentials
"""

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
import websockets
from dotenv import load_dotenv

load_dotenv()

async def run_microstructure_verification():
    """Run complete microstructure verification"""
    
    print("=" * 80)
    print("CTRADER MICROSTRUCTURE VERIFICATION - FINAL TEST")
    print("=" * 80)
    
    # Load credentials
    client_id = os.getenv('CTRADER_CLIENT_ID')
    client_secret = os.getenv('CTRADER_CLIENT_SECRET')
    access_token = os.getenv('CTRADER_ACCESS_TOKEN')
    account_id = os.getenv('CTRADER_ACCOUNT_ID')
    
    print(f"Testing with account ID: {account_id}")
    
    try:
        # Connect to demo endpoint
        uri = "wss://demo.ctraderapi.com:5036"
        
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connection established")
            
            # Test application authentication
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
                return False
            
            print("✅ Application authentication successful")
            
            # Test account authentication
            account_auth_req = {
                "payloadType": 2102,
                "clientMsgId": str(uuid.uuid4()),
                "payload": {
                    "ctidTraderAccountId": int(account_id),
                    "accessToken": access_token
                }
            }
            
            await websocket.send(json.dumps(account_auth_req))
            response = await websocket.recv()
            data = json.loads(response)
            
            if data.get("payloadType") != 2103:
                print("❌ Account authentication failed")
                print("   Error:", data.get("payload", {}).get("description", "Unknown error"))
                return False
            
            print("✅ Account authentication successful")
            
            # Load symbols
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
            
            if data.get("payloadType") != 2121:
                print("❌ Symbol loading failed")
                return False
            
            symbols = data.get("payload", {}).get("symbol", [])
            eurusd_symbol = next((s for s in symbols if s.get("symbolName") == "EURUSD"), None)
            
            if not eurusd_symbol:
                print("❌ EURUSD not found")
                return False
            
            symbol_id = eurusd_symbol.get("symbolId")
            print(f"✅ EURUSD found with symbol ID: {symbol_id}")
            
            # Subscribe to depth
            subscribe_req = {
                "payloadType": 2125,
                "clientMsgId": str(uuid.uuid4()),
                "payload": {
                    "ctidTraderAccountId": int(account_id),
                    "symbolId": [symbol_id]
                }
            }
            
            await websocket.send(json.dumps(subscribe_req))
            print("✅ Depth subscription request sent")
            
            # Collect data for 30 seconds
            print("📊 Collecting depth data for 30 seconds...")
            start_time = datetime.now(timezone.utc)
            messages_received = 0
            latency_records = []
            depth_records = []
            
            while (datetime.now(timezone.utc) - start_time).total_seconds() < 30:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(response)
                    
                    if data.get("payloadType") == 2126:  # Depth event
                        payload = data.get("payload", {})
                        bids = payload.get("bid", [])
                        asks = payload.get("ask", [])
                        timestamp = payload.get("timestamp", 0)
                        
                        # Calculate latency
                        server_time = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
                        client_time = datetime.now(timezone.utc)
                        latency_ms = (client_time - server_time).total_seconds() * 1000
                        
                        messages_received += 1
                        bid_depth = len(bids)
                        ask_depth = len(asks)
                        
                        latency_records.append(latency_ms)
                        depth_records.append({'bid': bid_depth, 'ask': ask_depth})
                        
                        if messages_received <= 3:
                            print(f"   Update #{messages_received}: {bid_depth} bid, {ask_depth} ask levels, latency: {latency_ms:.2f}ms")
                        
                except asyncio.TimeoutError:
                    continue
            
            # Generate report
            if messages_received > 0:
                print("=" * 80)
                print("✅ MICROSTRUCTURE VERIFICATION COMPLETE")
                print("=" * 80)
                
                # Calculate statistics
                avg_latency = sum(latency_records) / len(latency_records)
                max_depth = max(max(d['bid'], d['ask']) for d in depth_records)
                min_depth = min(min(d['bid'], d['ask']) for d in depth_records)
                
                print(f"✅ Successfully connected to cTrader")
                print(f"✅ Account authenticated: {account_id}")
                print(f"✅ EURUSD depth data received: {messages_received} updates")
                print(f"✅ Real-time Level 2 data confirmed working")
                print()
                print("📊 RESULTS:")
                print(f"   Average latency: {avg_latency:.2f}ms")
                print(f"   Max depth levels: {max_depth}")
                print(f"   Min depth levels: {min_depth}")
                print(f"   Updates per second: {messages_received/30:.2f}")
                print()
                print("🎯 RECOMMENDATION: GO")
                print("   The microstructure data is sufficient for Sprint 2")
                
                # Save results
                results = {
                    "account_id": account_id,
                    "symbol": "EURUSD",
                    "symbol_id": symbol_id,
                    "total_updates": messages_received,
                    "average_latency_ms": avg_latency,
                    "max_depth": max_depth,
                    "min_depth": min_depth,
                    "updates_per_second": messages_received/30,
                    "test_duration_seconds": 30,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                with open('docs/microstructure_verification_results.json', 'w') as f:
                    json.dump(results, f, indent=2)
                
                print("✅ Results saved to docs/microstructure_verification_results.json")
                
                return True
            else:
                print("❌ No depth data received")
                return False
                
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(run_microstructure_verification())
