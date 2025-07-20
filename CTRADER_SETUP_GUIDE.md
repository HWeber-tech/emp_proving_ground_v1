# cTrader API Integration Setup Guide

This guide provides step-by-step instructions to set up cTrader API integration for live trading.

## Prerequisites

1. **IC Markets cTrader Account**: You need a demo or live account
2. **Python 3.8+**: Ensure you have Python installed
3. **cTrader Open API Library**: Install with `pip install ctrader-open-api-py`

## Step 1: Create cTrader Application

1. Go to https://connect.icmarkets.com/
2. Log in with your cTID
3. Navigate to "Open API" in the left menu
4. Click "Create Application"
5. Fill out the form:
   - Application Name: "EMP Trading Bot"
   - Redirect URI(s): http://localhost/
   - Allowed Scopes: Ensure trading is checked
6. Save the application
7. Click the info icon to get your Client ID and Client Secret

## Step 2: Get Initial Tokens

1. Construct the authorization URL:
   ```
   https://connect.icmarkets.com/oauth/authorize?client_id=YOUR_CLIENT_ID&redirect_uri=http://localhost/&scope=trading
   ```

2. Open this URL in your browser and authorize the application
3. Copy the authorization code from the redirect URL
4. Exchange the code for tokens:

```python
import requests

CLIENT_ID = "YOUR_CLIENT_ID"
CLIENT_SECRET = "YOUR_CLIENT_SECRET"
AUTH_CODE = "CODE_FROM_URL"

response = requests.post(
    "https://connect.icmarkets.com/api/v2/oauth/token",
    data={
        "grant_type": "authorization_code",
        "code": AUTH_CODE,
        "redirect_uri": "http://localhost/",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }
)

if response.status_code == 200:
    token_data = response.json()
    print("Access Token:", token_data['access_token'])
    print("Refresh Token:", token_data['refresh_token'])
```

## Step 3: Get Account ID

```python
import requests

ACCESS_TOKEN = "YOUR_ACCESS_TOKEN"

response = requests.get(
    "https://connect.icmarkets.com/api/v2/tradingaccounts",
    headers={"Authorization": f"Bearer {ACCESS_TOKEN}"}
)

accounts = response.json().get('data', [])
for account in accounts:
    print(f"Account: {account['accountNumber']} -> ctidTraderAccountId: {account['ctidTraderAccountId']}")
```

## Step 4: Configure Environment

Create a `.env` file in your project root:

```bash
# cTrader API Configuration
CTRADER_CLIENT_ID=your_client_id_here
CTRADER_CLIENT_SECRET=your_client_secret_here
CTRADER_ACCESS_TOKEN=your_access_token_here
CTRADER_REFRESH_TOKEN=your_refresh_token_here
CTRADER_ACCOUNT_ID=your_account_id_here

# Database Configuration
DATABASE_URL=sqlite:///emp.db

# Logging Configuration
LOG_LEVEL=INFO
```

## Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

## Architecture Overview

### Components Created:

1. **CTraderDataOrgan** (`src/sensory/organs/ctrader_data_organ.py`)
   - Handles live market data streaming
   - Manages symbol mapping
   - Publishes MarketUnderstanding events

2. **CTraderBrokerInterface** (`src/trading/integration/ctrader_broker_interface.py`)
   - Handles order placement and management
   - Integrates with the data organ for shared connections
   - Supports market, limit, and stop orders

3. **TokenManager** (`src/governance/token_manager.py`)
   - Manages OAuth token refresh
   - Provides automatic token renewal
   - Handles token expiration

4. **SystemConfig** (`src/governance/system_config.py`)
   - Centralized configuration management
   - Loads from environment variables
   - Validates credentials

## Usage Example

```python
import asyncio
from src.core.events import EventBus
from src.sensory.organs.ctrader_data_organ import CTraderDataOrgan
from src.trading.integration.ctrader_broker_interface import CTraderBrokerInterface
from src.governance.token_manager import TokenManager

async def main():
    # Initialize components
    event_bus = EventBus()
    token_manager = TokenManager()
    
    # Start token auto-refresh
    await token_manager.start_auto_refresh()
    
    # Initialize data organ
    data_organ = CTraderDataOrgan(event_bus)
    await data_organ.start()
    
    # Initialize broker interface
    broker = CTraderBrokerInterface(event_bus)
    broker.set_data_organ(data_organ)
    await broker.start()
    
    # Your trading logic here...
    
    # Cleanup
    await data_organ.stop()
    await broker.stop()
    token_manager.stop_auto_refresh()

if __name__ == "__main__":
    asyncio.run(main())
```

## Testing

Run the integration test:
```bash
python test_ctrader_integration.py
```

## Security Notes

- Never commit your `.env` file to version control
- Store refresh tokens securely
- Use demo accounts for testing
- Rotate tokens regularly
- Monitor API usage and rate limits

## Troubleshooting

1. **Import Errors**: Ensure `ctrader-open-api-py` is installed
2. **Connection Issues**: Check firewall settings and API endpoints
3. **Authentication Errors**: Verify tokens and credentials
4. **Symbol Mapping**: Ensure symbols are available in your account
