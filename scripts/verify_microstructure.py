#!/usr/bin/env python3
"""
Microstructure Verification Tool
Comprehensive diagnostic for cTrader Level 2 Depth of Book data
"""

import asyncio
import json
import os
import websockets
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

class MicrostructureVerifier:
    def __init__(self):
        self.client_id = os.getenv('CTRADER_CLIENT_ID')
        self.client_secret = os.getenv('CTRADER_CLIENT_SECRET')
        self.access_token = os.getenv('CTRADER_ACCESS_TOKEN')
        self.account_id = int(os.getenv('CTRADER_ACCOUNT_ID', '43939234'))
        self.host = 'demo.ctraderapi.com'
        self.port = 5036
        
        self.results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'account_id': self.account_id,
            'protocol': 'cTrader OpenAPI JSON over WebSocket',
            'host': f'{self.host}:{self.port}'
        }
        
    async def run_full_verification(self):
        """Run complete microstructure verification"""
        print("=" * 80)
        print("MICROSTRUCTURE VERIFICATION TOOL")
        print("=" * 80)
        print(f"Account ID: {self.account_id}")
        print(f"Host: {self.host}:{self.port}")
        print()
        
        uri = f'wss://{self.host}:{self.port}'
        
        try:
            async with websockets.connect(uri) as ws:
                print('✅ Connected to cTrader API')
                
                # Step 1: Authenticate
                if not await self._authenticate(ws):
                    return False
                
                # Step 2: Get symbols
                symbols = await self._get_symbols(ws)
                if not symbols:
                    return False
                
                # Step 3: Find EURUSD
                eurusd = self._find_eurusd(symbols)
                if not eurusd:
                    return False
                
                # Step 4: Subscribe to depth
                if not await self._subscribe_depth(ws, eurusd['symbolId']):
                    return False
                
                # Step 5: Collect depth data
                await self._collect_depth_data(ws)
                
                # Step 6: Generate report
                self._generate_report()
                
                return True
                
        except Exception as e:
            self.results['error'] = str(e)
            print(f"❌ Error: {e}")
            return False
    
    async def _authenticate(self, ws):
        """Complete authentication flow"""
        # Application auth
        await ws.send(json.dumps({
            'payloadType': 2100,
            'payload': {'clientId': self.client_id, 'clientSecret': self.client_secret}
        }))
        resp = await ws.recv()
        data = json.loads(resp)
        success = data.get('payloadType') == 2101
        self.results['application_auth'] = success
        print(f"✅ Application authentication: {'SUCCESS' if success else 'FAILED'}")
        
        if not success:
            return False
        
        # Account auth
        await ws.send(json.dumps({
            'payloadType': 2102,
            'payload': {'ctidTraderAccountId': self.account_id, 'accessToken': self.access_token}
        }))
        resp = await ws.recv()
        data = json.loads(resp)
        success = data.get('payloadType') == 2103
        self.results['account_auth'] = success
        print(f"✅ Account authentication: {'SUCCESS' if success else 'FAILED'}")
        
        return success
    
    async def _get_symbols(self, ws):
        """Get available symbols"""
        await ws.send(json.dumps({
            'payloadType': 2120,
            'payload': {'ctidTraderAccountId': self.account_id}
        }))
        
        resp = await asyncio.wait_for(ws.recv(), timeout=5.0)
        data = json.loads(resp)
        symbols = []
        if data.get('payloadType') == 2121:
            symbols = data.get('payload', {}).get('symbol', [])
        
        self.results['symbols_found'] = len(symbols)
        self.results['symbols_available'] = [s.get('symbolName') for s in symbols[:10]]
        
        print(f'✅ Found {len(symbols)} symbols')
        return symbols
    
    def _find_eurusd(self, symbols):
        """Find EURUSD symbol"""
        eurusd = next((s for s in symbols if s.get('symbolName') == 'EURUSD'), None)
        self.results['eurusd_available'] = bool(eurusd)
        self.results['eurusd_id'] = eurusd.get('symbolId') if eurusd else None
        
        if eurusd:
            print(f'✅ EURUSD Found: ID={eurusd.get("symbolId")}, Digits={eurusd.get("digits")}')
            return eurusd
        else:
            print('❌ EURUSD not found')
            return None
    
    async def _subscribe_depth(self, ws, symbol_id):
        """Subscribe to depth quotes"""
        await ws.send(json.dumps({
            'payloadType': 2125,
            'payload': {'ctidTraderAccountId': self.account_id, 'symbolId': [symbol_id]}
        }))
        
        resp = await asyncio.wait_for(ws.recv(), timeout=5.0)
        data = json.loads(resp)
        success = data.get('payloadType') == 2125
        self.results['depth_subscription'] = success
        print(f"✅ Depth subscription: {'CONFIRMED' if success else 'FAILED'}")
        return success
    
    async def _collect_depth_data(self, ws):
        """Collect depth data for analysis"""
        print("  Collecting depth data...")
        
        resp = await asyncio.wait_for(ws.recv(), timeout=5.0)
        data = json.loads(resp)
        
        if data.get('payloadType') == 2126:
            payload = data.get('payload', {})
            bids = payload.get('bid', [])
            asks = payload.get('ask', [])
            
            self.results['depth_data_received'] = True
            self.results['bid_levels'] = len(bids)
            self.results['ask_levels'] = len(asks)
            self.results['sample_data'] = {
                'timestamp': payload.get('timestamp'),
                'bids': [{'price': b.get('price', 0)/100000, 'volume': b.get('volume', 0)/100} for b in bids[:3]],
                'asks': [{'price': a.get('price', 0)/100000, 'volume': a.get('volume', 0)/100} for a in asks[:3]]
            }
            
            print(f"✅ Depth data received: {len(bids)} bid, {len(asks)} ask levels")
            
            # Show sample
            for j, bid in enumerate(bids[:3]):
                price = bid.get('price', 0) / 100000
                volume = bid.get('volume', 0) / 100
                print(f"   Bid {j+1}: {price:.5f} x {volume}")
                
            for j, ask in enumerate(asks[:3]):
                price = ask.get('price', 0) / 100
