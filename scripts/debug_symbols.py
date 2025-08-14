#!/usr/bin/env python3
"""
Debug script to find correct EURUSD symbol ID and test depth data
Final version with race condition fix for message 2142
"""

import asyncio
import json
import os
import uuid

import websockets
from dotenv import load_dotenv

load_dotenv()

async def debug_symbols():
    client_id = os.getenv('CTRADER_CLIENT_ID')
    client_secret = os.getenv('CTRADER_CLIENT_SECRET')
    access_token = os.getenv('CTRADER_ACCESS_TOKEN')
    account_id = int(os.getenv('CTRADER_ACCOUNT_ID', '43939234'))

    uri = 'wss://demo.ctraderapi.com:5036'

    async with websockets.connect(uri) as ws:
        print('✅ Connected to demo.ctraderapi.com:5036')
        
        # Application authentication
        await ws.send(json.dumps({
            'payloadType': 2100,
            'clientMsgId': str(uuid.uuid4()),
            'payload': {'clientId': client_id, 'clientSecret': client_secret}
        }))
        resp_data = json.loads(await ws.recv())
        print(f"✅ App auth response: {resp_data.get('payloadType')}")

        # Account authentication
        await ws.send(json.dumps({
            'payloadType': 2102,
            'clientMsgId': str(uuid.uuid4()),
            'payload': {'ctidTraderAccountId': account_id, 'accessToken': access_token}
        }))
        resp_data = json.loads(await ws.recv())
        print(f"✅ Account auth response: {resp_data.get('payloadType')}")

        # --- FIX: THE FINAL AND MOST IMPORTANT STEP ---
        # After authentication, the server sends unsolicited messages (like 2142).
        # We must receive and "clear" these before making new requests.
        print("  Clearing initial message buffer from server...")
        try:
            while True:
                # Use a short timeout to quickly consume any messages the server sent automatically.
                resp = await asyncio.wait_for(ws.recv(), timeout=0.5)
                resp_data = json.loads(resp)
                print(f"  (Clearing initial message type: {resp_data.get('payloadType')})")
        except asyncio.TimeoutError:
            # A timeout here is GOOD. It means the buffer is clear and the server is waiting for our command.
            print("  Initial buffer clear. Server is ready.")
            pass
        
        # Now that the server is ready, request the symbol list.
        print("  Requesting symbol list...")
        await ws.send(json.dumps({
            'payloadType': 2120,
            'payload': {'ctidTraderAccountId': account_id}
        }))

        # Now, we listen specifically for the symbol list response.
        resp = await asyncio.wait_for(ws.recv(), timeout=5.0)
        data = json.loads(resp)
        symbols = []
        if data.get('payloadType') == 2121:
            symbols = data.get('payload', {}).get('symbol', [])
        
        print(f'✅ Found {len(symbols)} symbols')

        # Find EURUSD
        eurusd = next((s for s in symbols if s.get('symbolName') == 'EURUSD'), None)

        if eurusd:
            eurusd_id = eurusd.get("symbolId")
            print(f'✅ EURUSD Found: ID={eurusd_id}')

            # Subscribe to depth
            await ws.send(json.dumps({
                'payloadType': 2125,
                'payload': {'ctidTraderAccountId': account_id, 'symbolId': [eurusd_id]}
            }))
            
            # Now wait for data. The first message could be confirmation or a price tick.
            print('  Waiting for depth data...')
            resp = await asyncio.wait_for(ws.recv(), timeout=5.0)
            data = json.loads(resp)

            # If we got the confirmation first, get the next message which should be data.
            if data.get('payloadType') == 2125:
                 print(f'✅ Subscription confirmed. Waiting for next message...')
                 resp = await asyncio.wait_for(ws.recv(), timeout=5.0)
                 data = json.loads(resp)
            
            if data.get('payloadType') == 2126:  # Depth Event
                payload = data.get('payload', {})
                bids = len(payload.get('bid', []))
                asks = len(payload.get('ask', []))
                print(f'\n✅✅✅ SUCCESS! DEPTH DATA RECEIVED: {bids} bid, {asks} ask levels')
                
                # Show sample data
                for j, bid in enumerate(payload.get('bid', [])[:3]):
                    price = bid.get('price', 0) / 100000
                    volume = bid.get('volume', 0) / 100
                    print(f"   Bid {j+1}: {price:.5f} x {volume}")
                    
                for j, ask in enumerate(payload.get('ask', [])[:3]):
                    price = ask.get('price', 0) / 100000
                    volume = ask.get('volume', 0) / 100
                    print(f"   Ask {j+1}: {price:.5f} x {volume}")
                    
                return True
            else:
                 print(f"⚠️ Received unexpected message type after subscribing: {data.get('payloadType')}")
                 return False
        else:
            print('❌ EURUSD not found in the symbol list.')
            print('   Available symbols:')
            for symbol in symbols[:10]:
                print(f"   - {symbol.get('symbolName')}: ID={symbol.get('symbolId')}")
            return False

if __name__ == "__main__":
    asyncio.run(debug_symbols())
