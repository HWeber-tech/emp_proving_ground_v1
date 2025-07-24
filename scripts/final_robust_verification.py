#!/usr/bin/env python3
"""
FINAL ROBUST MICROSTRUCTURE VERIFICATION
Race-condition-free, event-driven architecture
"""

import asyncio
import json
import os
import uuid
import websockets
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# Global state for async coordination
updates_received = 0
latency_records = []
depth_records = []
subscription_confirmed = asyncio.Event()
app_authenticated = asyncio.Event()
account_authenticated = asyncio.Event()

async def message_handler(websocket):
    """
    Dedicated message handler - processes ALL incoming messages
    regardless of order or timing
    """
    global updates_received
    
    async for message in websocket:
        try:
            data = json.loads(message)
            payload_type = data.get("payloadType")
            
            if payload_type == 2101:  # Application Auth Response
                print("✅ Application authentication successful")
                app_authenticated.set()
                
            elif payload_type == 2103:  # Account Auth Response
                print("✅ Account authentication successful")
                account_authenticated.set()
                
            elif payload_type == 2125:  # Subscribe Depth Response
                print("✅ Depth subscription confirmed")
                subscription_confirmed.set()
                
            elif payload_type == 2124:  # Subscribe Spot Response
                print("✅ Spot subscription confirmed")
                subscription_confirmed.set()
                
            elif payload_type == 2128:  # Spot Event (real-time data)
                updates_received += 1
                payload = data.get("payload", {})
                timestamp = payload.get("timestamp", 0)
                
                if timestamp > 0:
                    server_time = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
                    client_time = datetime.now(timezone.utc)
                    latency_ms = (client_time - server_time).total_seconds() * 1000
                    latency_records.append(latency_ms)
                    
                    bid = payload.get("bid", 0) / 100000
                    ask = payload.get("ask", 0) / 100000
                    
                    if updates_received <= 10:
                        print(f"   Update #{updates_received}: Bid={bid:.5f}, Ask={ask:.5f}, latency: {latency_ms:.2f}ms")
                        
            elif payload_type == 2126:  # Depth Event (Level 2 data)
                updates_received += 1
                payload = data.get("payload", {})
                timestamp = payload.get("timestamp", 0)
                
                if timestamp > 0:
                    server_time = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
                    client_time = datetime.now(timezone.utc)
                    latency_ms = (client_time - server_time).total_seconds() * 1000
                    latency_records.append(latency_ms)
                    
                    bids = len(payload.get("bid", []))
                    asks = len(payload.get("ask", []))
                    depth_records.append({'bid': bids, 'ask': asks})
                    
                    if updates_received <= 5:
                        print(f"   Update #{updates_received}: {bids} bid, {asks} ask levels, latency: {latency_ms:.2f}ms")
                        
                        # Show first few levels
                        for i, bid in enumerate(payload.get("bid", [])[:3]):
                            price = bid.get("price", 0) / 100000
                            volume = bid.get("volume", 0) / 100
                            print(f"      Bid {i+1}: {price:.5f} x {volume}")
                            
                        for i, ask in enumerate(payload.get("ask", [])[:3]):
                            price = ask.get("price", 0) / 100000
                            volume = ask.get("volume", 0) / 100
                            print(f"      Ask {i+1}: {price:.5f} x {volume}")
                            
            elif data.get("payload", {}).get("errorCode"):
                print(f"❌ API Error: {data['payload']['errorCode']} - {data['payload'].get('description', '')}")
                
        except Exception as e:
            print(f"❌ Message processing error: {e}")

async def heartbeat_sender(websocket):
    """Keeps connection alive with periodic heartbeats"""
    while True:
        try:
            heartbeat = {"payloadType": 51, "clientMsgId": str(uuid.uuid4())}
            await websocket.send(json.dumps(heartbeat))
            await asyncio.sleep(10)
        except websockets.exceptions.ConnectionClosed:
            break

async def main_logic(websocket, account_id, client_id, client_secret, access_token):
    """Main business logic - sends commands without blocking"""
    # Application authentication
    auth_req = {
        "payloadType": 2100,
        "clientMsgId": str(uuid.uuid4()),
        "payload": {"clientId": client_id, "clientSecret": client_secret}
    }
    await websocket.send(json.dumps(auth_req))
    await asyncio.wait_for(app_authenticated.wait(), timeout=10)

    # Account authentication
    account_auth = {
        "payloadType": 2102,
        "clientMsgId": str(uuid.uuid4()),
        "payload": {"ctidTraderAccountId": account_id, "accessToken": access_token}
    }
    await websocket.send(json.dumps(account_auth))
    await asyncio.wait_for(account_authenticated.wait(), timeout=10)

    # Subscribe to depth quotes (Level 2)
    subscribe_depth = {
        "payloadType": 2125,
        "clientMsgId": str(uuid.uuid4()),
        "payload": {"ctidTraderAccountId": account_id, "symbolId": [1]}  # EURUSD
    }
    await websocket.send(json.dumps(subscribe_depth))
    await asyncio.wait_for(subscription_confirmed.wait(), timeout=10)

async def verify_microstructure_complete():
    """Main verification function with proper async architecture"""
    print("=" * 80)
    print("FINAL ROBUST MICROSTRUCTURE VERIFICATION")
    print("=" * 80)
    
    # Load credentials
    client_id = os.getenv('CTRADER_CLIENT_ID')
    client_secret = os.getenv('CTRADER_CLIENT_SECRET')
    access_token = os.getenv('CTRADER_ACCESS_TOKEN')
    account_id = int(os.getenv('CTRADER_ACCOUNT_ID', '43939234'))
    duration_seconds = 30
    
    print(f"✅ Account ID: {account_id}")
    
    try:
        uri = "wss://demo.ctraderapi.com:5036"
        
        async with websockets.connect(uri) as websocket:
            print(f"✅ Connected to {uri}")
            
            # Run all tasks concurrently
            listener_task = asyncio.create_task(message_handler(websocket))
            heartbeat_task = asyncio.create_task(heartbeat_sender(websocket))
            main_task = asyncio.create_task(main_logic(websocket, account_id, client_id, client_secret, access_token))
            
            # Wait for setup to complete
            await asyncio.sleep(2)
            print("📊 Collecting real-time data...")
            
            # Collect data for specified duration
            await asyncio.sleep(duration_seconds)
            
            # Clean shutdown
            listener_task.cancel()
            heartbeat_task.cancel()
            
            # Generate final report
            if updates_received > 0:
                avg_latency = sum(latency_records) / len(latency_records)
                max_depth = max(max(d['bid'], d['ask']) for d in depth_records) if depth_records else 1
                min_depth = min(min(d['bid'], d['ask
