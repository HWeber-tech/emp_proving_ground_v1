# [Legacy] IC Markets cTrader API Setup Guide (OpenAPI)

> **Status:** Legacy – deprecated in FIX-only builds.
> Follow the [Integration Policy](policies/integration_policy.md) and use
> the FIX guides in `docs/fix_api/` for supported connectivity.

## Prerequisites

1. **IC Markets cTrader Account**
   - Demo account (recommended for testing)
   - Live account (for production)

2. **Python Environment**
   - Python 3.8+
   - Required packages: `ctrader-open-api-py`, `pandas`, `python-dotenv`

## Step 1: Create cTrader Application

1. **Access cTID Portal**
   - Go to: https://connect.icmarkets.com/
   - Log in with your cTID credentials

2. **Create Application**
   - Navigate to "Open API" in the left menu
   - Click "Create Application"
   - Fill out the form:
     - **Application Name**: "EMP Microstructure Verification"
     - **Redirect URI**: `http://localhost/`
     - **Allowed Scopes**: Ensure "trading" is checked

3. **Get Credentials**
   - After creating the application, click the info (i) icon
   - Copy your **Client ID** and **Client Secret**
   - **Important**: The Client Secret is only shown once - save it securely!

## Step 2: Get Authorization Code

1. **Construct Authorization URL**
   ```
   https://connect.icmarkets.com/oauth/authorize?client_id=YOUR_CLIENT_ID&redirect_uri=http://localhost/&scope=trading
   ```

2. **Get Authorization Code**
   - Paste the URL in your browser
   - Log in if prompted
   - Click "Allow Access"
   - You'll be redirected to `http://localhost/?code=AUTH_CODE`
   - Copy the **AUTH_CODE** from the URL (valid for 60 seconds)

## Step 3: Exchange Code for Tokens

Run this Python script to get your access and refresh tokens:

```python
import requests

CLIENT_ID = "your_client_id_here"
CLIENT_SECRET = "your_client_secret_here"
AUTH_CODE = "your_auth_code_here"

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
else:
    print("Error:", response.status_code, response.text)
```

## Step 4: Get Trading Account ID

Run this script with your access token:

```python
import requests

ACCESS_TOKEN = "your_access_token_here"

response = requests.get(
    "https://connect.icmarkets.com/api/v2/tradingaccounts",
    headers={"Authorization": f"Bearer {ACCESS_TOKEN}"}
)

if response.status_code == 200:
    accounts = response.json().get('data', [])
    for account in accounts:
        account_id = account['ctidTraderAccountId']
        is_live = "LIVE" if account['isLive'] else "DEMO"
        print(f"Account: {account['accountNumber']} [{is_live}] -> ID: {account_id}")
else:
    print("Error:", response.status_code, response.text)
```

## Step 5: Configure Environment

Create a `.env` file in your project root with:

```bash
# cTrader API Credentials
CTRADER_CLIENT_ID=your_client_id_here
CTRADER_CLIENT_SECRET=your_client_secret_here
CTRADER_ACCESS_TOKEN=your_access_token_here
CTRADER_REFRESH_TOKEN=your_refresh_token_here
CTRADER_ACCOUNT_ID=your_account_id_here

# Host Configuration
CTRADER_DEMO_HOST=demo.ctraderapi.com
CTRADER_LIVE_HOST=live.ctraderapi.com
CTRADER_PORT=5035

# Default symbol for testing
DEFAULT_SYMBOLS=EURUSD
```

## Step 6: Test Connection

Run the verification tool:

```bash
# Basic test (30 minutes, EURUSD)
python scripts/verify_microstructure.py

# Custom parameters
python scripts/verify_microstructure.py --duration 15 --symbol GBPUSD
```

## Troubleshooting

### Common Issues

1. **"Invalid client credentials"**
   - Check Client ID and Client Secret are correct
   - Ensure no extra spaces in credentials

2. **"Authorization code expired"**
   - The code is only valid for 60 seconds
   - Get a new code and try again

3. **"Account not found"**
   - Ensure you're using the correct account ID
   - Use demo account ID for testing

4. **Connection timeouts**
   - Check firewall settings
   - Ensure port 5035 is open
   - Try different network

### Token Refresh

Access tokens expire after ~30 days. Use this script to refresh:

```python
import requests

CLIENT_ID = "your_client_id"
CLIENT_SECRET = "your_client_secret"
REFRESH_TOKEN = "your_refresh_token"

response = requests.post(
    "https://connect.icmarkets.com/api/v2/oauth/token",
    data={
        "grant_type": "refresh_token",
        "refresh_token": REFRESH_TOKEN,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }
)

if response.status_code == 200:
    token_data = response.json()
    print("New Access Token:", token_data['access_token'])
    print("New Refresh Token:", token_data['refresh_token'])
```

## Security Best Practices

1. **Never commit credentials** to version control
2. **Use environment variables** for sensitive data
3. **Rotate tokens** regularly
4. **Use demo accounts** for development
5. **Monitor API usage** for unusual activity

## Support

For API-specific issues:
- IC Markets Support: https://help.icmarkets.com/
- cTrader OpenAPI Documentation: consult the official cTrader developer portal via https://ctrader.com/developers/

For tool-specific issues:
- Check logs in `logs/microstructure_verification.log`
- Enable debug logging: `LOG_LEVEL=DEBUG python scripts/verify_microstructure.py`
