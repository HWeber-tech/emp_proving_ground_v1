#!/usr/bin/env python3
"""
Diagnostic tool to find correct cTrader account ID
"""

import asyncio
import json
import logging
import os
import uuid

import websockets
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()


class AccountFinder:
    """Tool to find correct cTrader account ID."""

    def __init__(self):
        self.config = {
            "host": os.getenv("CTRADER_HOST", "demo.ctraderapi.com"),
            "port": 5036,
            "client_id": os.getenv("CTRADER_CLIENT_ID"),
            "client_secret": os.getenv("CTRADER_CLIENT_SECRET"),
            "access_token": os.getenv("CTRADER_ACCESS_TOKEN"),
            "account_id": int(os.getenv("CTRADER_ACCOUNT_ID", 0)),
        }

    async def find_accounts(self):
        """Find available trading accounts."""
        try:
            uri = f"wss://{self.config['host']}:{self.config['port']}"

            async with websockets.connect(uri) as websocket:
                logger.info("Connected to cTrader")

                # Authenticate application
                app_auth_req = {
                    "payloadType": 2100,
                    "clientMsgId": str(uuid.uuid4()),
                    "payload": {
                        "clientId": self.config["client_id"],
                        "clientSecret": self.config["client_secret"],
                    },
                }

                await websocket.send(json.dumps(app_auth_req))
                response = await websocket.recv()
                data = json.loads(response)

                if data.get("payloadType") == 2101:
                    logger.info("✅ Application authenticated")
                else:
                    logger.error("Application auth failed")
                    return

                # Get account list
                account_list_req = {
                    "payloadType": 2123,  # PROTO_OA_GET_ACCOUNT_LIST_BY_ACCESS_TOKEN_REQ
                    "clientMsgId": str(uuid.uuid4()),
                    "payload": {"accessToken": self.config["access_token"]},
                }

                await websocket.send(json.dumps(account_list_req))
                response = await websocket.recv()
                data = json.loads(response)

                if data.get("payloadType") == 2124:  # PROTO_OA_GET_ACCOUNT_LIST_BY_ACCESS_TOKEN_RES
                    accounts = data.get("payload", {}).get("ctidTraderAccount", [])

                    logger.info("=" * 60)
                    logger.info("AVAILABLE TRADING ACCOUNTS")
                    logger.info("=" * 60)

                    for account in accounts:
                        account_id = account.get("ctidTraderAccountId")
                        account_number = account.get("accountNumber")
                        is_live = account.get("isLive", False)
                        balance = account.get("balance", 0)
                        money_digits = account.get("moneyDigits", 2)

                        real_balance = balance / (10**money_digits)

                        logger.info(f"Account ID: {account_id}")
                        logger.info(f"Account Number: {account_number}")
                        logger.info(f"Type: {'LIVE' if is_live else 'DEMO'}")
                        logger.info(f"Balance: {real_balance:.2f}")
                        logger.info("-" * 40)

                    if accounts:
                        logger.info("✅ Update your .env file with the correct ACCOUNT_ID")
                        logger.info("Example: CTRADER_ACCOUNT_ID=12345678")
                    else:
                        logger.error("❌ No accounts found. Check your access token.")

                else:
                    logger.error(f"Failed to get accounts: {data}")

        except Exception as e:
            logger.error(f"Error finding accounts: {e}")


async def main():
    finder = AccountFinder()
    await finder.find_accounts()


if __name__ == "__main__":
    asyncio.run(main())
