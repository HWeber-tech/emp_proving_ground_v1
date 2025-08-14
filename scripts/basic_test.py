#!/usr/bin/env python3
# ruff: noqa
"""
The most basic cTrader API connection and authentication test.
"""

import asyncio
import logging
import os
import sys
from dotenv import load_dotenv

# --- Basic Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
load_dotenv()

# --- Credentials and Host from .env file ---
HOST = os.getenv('CTRADER_HOST', 'demo.ctraderapi.com')
PORT = int(os.getenv('CTRADER_PORT', 5035))
CLIENT_ID = os.getenv('CTRADER_CLIENT_ID')
CLIENT_SECRET = os.getenv('CTRADER_CLIENT_SECRET')
ACCESS_TOKEN = os.getenv('CTRADER_ACCESS_TOKEN')
ACCOUNT_ID = int(os.getenv('CTRADER_ACCOUNT_ID', 0))

# --- Import cTrader components ---
raise SystemExit("This script used legacy cTrader API and is disabled in FIX-only builds.")

async def main():
    """Main connection and authentication logic."""
    
    # Validate that credentials were loaded
    if not all([CLIENT_ID, CLIENT_SECRET, ACCESS_TOKEN, ACCOUNT_ID]):
        logging.error("FATAL: Missing credentials in .env file.")
        sys.exit(1)

    try:
        logging.info("=" * 60)
        logging.info("CTRADER BASIC CONNECTION TEST")
        logging.info("=" * 60)
        
        # Create and start the client with correct parameters
        client = Client(HOST, PORT, TcpProtocol)
        await client.start()
        logging.info(f"✅ Step 1: TCP Connection to {HOST}:{PORT} established.")
        
        # Set up message handler
        def on_message(message):
            try:
                if hasattr(message, 'payloadType'):
                    if message.payloadType == 2101:  # ProtoOAApplicationAuthRes
                        logging.info("✅ Step 2: Application authentication successful.")
                    elif message.payloadType == 2103:  # ProtoOAAccountAuthRes
                        logging.info("✅ Step 3: Account authentication successful.")
                    elif message.payloadType == 50:  # ProtoOAErrorRes
                        logging.error("🚨 API Error - check credentials")
            except Exception as e:
                logging.error(f"Message handler error: {e}")
        
        client.on_message = on_message

        # --- Application Authentication ---
        app_auth_req = messages.ProtoOAApplicationAuthReq()
        app_auth_req.clientId = CLIENT_ID
        app_auth_req.clientSecret = CLIENT_SECRET
        await client.send(app_auth_req)
        logging.info("Sent application authentication request...")

        # --- Account Authentication ---
        await asyncio.sleep(2)
        account_auth_req = messages.ProtoOAAccountAuthReq()
        account_auth_req.ctidTraderAccountId = ACCOUNT_ID
        account_auth_req.accessToken = ACCESS_TOKEN
        await client.send(account_auth_req)
        logging.info("Sent account authentication request...")

        # Wait for responses
        await asyncio.sleep(5)
        
        # Stop the client
        await client.stop()
        logging.info("Client stopped.")
        logging.info("\n--- TEST COMPLETED ---")

    except Exception as e:
        logging.error(f"Connection failed: {e}")
        logging.error("\n--- TEST FAILED ---")

if __name__ == "__main__":
    asyncio.run(main())
