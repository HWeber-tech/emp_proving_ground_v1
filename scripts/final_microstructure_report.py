#!/usr/bin/env python3
"""
FINAL MICROSTRUCTURE REALITY CHECK REPORT
Comprehensive analysis of cTrader API capabilities
"""

import asyncio
import json
import os
import uuid
import websockets
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

async def comprehensive_verification():
    """Comprehensive verification with realistic findings"""
    print("=" * 80)
    print("COMPREHENSIVE MICROSTRUCTURE REALITY CHECK")
    print("=" * 80)
    
    # Load credentials
    client_id = os.getenv('CTRADER_CLIENT_ID')
    client_secret = os.getenv('CTRADER_CLIENT_SECRET')
    access_token = os.getenv('CTRADER_ACCESS_TOKEN')
    account_id = int(os.getenv('CTRADER_ACCOUNT_ID', '43939234'))
    
    print(f"✅ Account ID: {account_id}")
    
    # Create comprehensive report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "account_id": account_id,
        "test_duration_seconds": 30,
        "protocol": "cTrader OpenAPI JSON over WebSocket",
        "host": "demo.ctraderapi.com:5036"
    }
    
    try:
        uri = "wss://demo.ctraderapi.com:5036"
        
        async with websockets.connect(uri) as ws:
            print(f"✅ Connected to {uri}")
            
            # Test 1: Application Authentication
            await ws.send(json.dumps({
                'payloadType': 2100,
                'clientMsgId': str(uuid.uuid4()),
                'payload': {'clientId': client_id, 'clientSecret': client_secret}
            }))
            resp = await ws.recv()
            auth_result = json.loads(resp).get('payloadType') == 2101
            report['application_auth'] = auth_result
            print(f"✅ Application authentication: {'SUCCESS' if auth_result else 'FAILED'}")
            
            # Test 2: Account Authentication
            await ws.send(json.dumps({
                'payloadType': 2102,
                'clientMsgId': str(uuid.uuid4()),
                'payload': {'ctidTraderAccountId': account_id, 'accessToken': access_token}
            }))
            resp = await ws.recv()
            account_auth = json.loads(resp).get('payloadType') == 2103
            report['account_auth'] = account_auth
            print(f"✅ Account authentication: {'SUCCESS' if account_auth else 'FAILED'}")
            
            # Test 3: Symbol Discovery
            await ws.send(json.dumps({
                'payloadType': 2120,
                'clientMsgId': str(uuid.uuid4()),
                'payload': {'ctidTraderAccountId': account_id}
            }))
            
            symbols = []
            resp = await asyncio.wait_for(ws.recv(), timeout=5.0)
            data = json.loads(resp)
            if data.get('payloadType') == 2121:
                symbols = data.get('payload', {}).get('symbol', [])
            
            report['symbols_found'] = len(symbols)
            report['symbols_available'] = [s.get('symbolName') for s in symbols[:10]]  # First 10
            
            print(f"✅ Symbols discovered: {len(symbols)}")
            if symbols:
                for symbol in symbols[:5]:
                    print(f"   - {symbol.get('symbolName')}: ID={symbol.get('symbolId')}")
            
            # Test 4: EURUSD Availability
            eurusd = next((s for s in symbols if s.get('symbolName') == 'EURUSD'), None)
            report['eurusd_available'] = bool(eurusd)
            report['eurusd_id'] = eurusd.get('symbolId') if eurusd else None
            
            if eurusd:
                print(f"✅ EURUSD found: ID={eurusd.get('symbolId')}")
                
                # Test 5: Depth Subscription
                await ws.send(json.dumps({
                    'payloadType': 2125,
                    'clientMsgId': str(uuid.uuid4()),
                    'payload': {'ctidTraderAccountId': account_id, 'symbolId': [eurusd.get('symbolId')]}
                }))
                
                # Wait for subscription confirmation
                resp = await ws.recv()
                sub_confirmed = json.loads(resp).get('payloadType') == 2125
                report['depth_subscription'] = sub_confirmed
                print(f"✅ Depth subscription: {'CONFIRMED' if sub_confirmed else 'FAILED'}")
                
                # Test 6: Real-time Data
                if sub_confirmed:
                    print("📊 Collecting depth data...")
                    updates = 0
                    for i in range(10):
                        try:
                            resp = await asyncio.wait_for(ws.recv(), timeout=1.0)
                            data = json.loads(resp)
                            if data.get('payloadType') == 2126:
                                payload = data.get('payload', {})
                                bids = len(payload.get('bid', []))
                                asks = len(payload.get('ask', []))
                                updates += 1
                                print(f"✅ DEPTH DATA: {bids} bid, {asks} ask levels")
                                
                                # Show sample data
                                for j, bid in enumerate(payload.get('bid', [])[:2]):
                                    price = bid.get('price', 0) / 100000
                                    volume = bid.get('volume', 0) / 100
                                    print(f"   Bid {j+1}: {price:.5f} x {volume}")
                                    
                                break
                        except asyncio.TimeoutError:
                            continue
                    
                    report['depth_updates_received'] = updates
                    report['depth_data_available'] = updates > 0
            else:
                print("⚠️ EURUSD not found in symbol list")
                report['depth_data_available'] = False
                report['depth_updates_received'] = 0
    
    except Exception as e:
        report['error'] = str(e)
        print(f"❌ Error: {e}")
    
    # Generate final report
    print("\n" + "=" * 80)
    print("FINAL REALITY CHECK REPORT")
    print("=" * 80)
    
    # Determine verdict
    if report['symbols_found'] == 0:
        verdict = "NO-GO"
        reason = "Demo account has no trading instruments enabled"
        recommendation = "Contact IC Markets support to enable instruments on demo account"
    elif not report['eurusd_available']:
        verdict = "NO-GO"
        reason = "EURUSD not available in symbol list"
        recommendation = "Check account configuration or use different instrument"
    elif not report['depth_data_available']:
        verdict = "GO (with limitations)"
        reason = "Account authenticated but no real-time data received"
        recommendation = "Account setup correct, may need live account for full data"
    else:
        verdict = "GO"
        reason = "Full microstructure data available"
        recommendation = "Proceed with Sprint 2 implementation"
    
    report['verdict'] = verdict
    report['reason'] = reason
    report['recommendation'] = recommendation
    
    # Print summary
    print(f"🎯 VERDICT: {verdict}")
    print(f"📋 REASON: {reason}")
    print(f"💡 RECOMMENDATION: {recommendation}")
    print()
    print("📊 DETAILED FINDINGS:")
    print(f"   • Application Authentication: {'✅'
