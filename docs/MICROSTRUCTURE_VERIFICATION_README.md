# IC Markets Microstructure Verification Tool

## Overview
This diagnostic tool connects to the IC Markets cTrader API to verify the quality, depth, and latency of Level 2 Depth-of-Book (DoB) market data. It produces a comprehensive report that serves as the basis for a go/no-go decision on microstructure-focused development.

## Quick Start

### Prerequisites
1. **cTrader Demo Account**: You need an IC Markets demo account
2. **API Credentials**: Follow the setup guide below to obtain credentials
3. **Python Environment**: Python 3.8+ with required packages

### Installation
```bash
# Install required packages
pip install ctrader-open-api-py pandas python-dotenv

# Ensure .env file exists with credentials
cp .env.example .env
# Edit .env with your actual credentials
```

### Running the Tool
```bash
# Basic usage (30 minutes, EURUSD)
python scripts/verify_microstructure.py

# Custom duration and symbol
python scripts/verify_microstructure.py --duration 60 --symbol GBPUSD

# Quick test (5 minutes)
python scripts/verify_microstructure.py --duration 5
```

## cTrader API Setup Guide

### Step 1: Create Application
1. Go to https://connect.icmarkets.com/
2. Log in with your cTID
3. Navigate to "Open API" → "Create Application"
4. Fill out the form:
   - Application Name: "Microstructure Verification"
   - Redirect URI: `http://localhost/`
   - Scopes: Ensure "trading" is checked

### Step 2: Get Authorization Code
1. Construct this URL (replace YOUR_CLIENT_ID):
   ```
   https://connect.icmarkets.com/oauth/authorize?client_id=YOUR_CLIENT_ID&redirect_uri=http://localhost/&scope=trading
   ```
2. Open in browser, log in, click "Allow"
3. Copy the `code` parameter from the redirect URL

### Step 3: Exchange Code for Tokens
```python
import requests

CLIENT_ID = "your_client_id"
CLIENT_SECRET = "your_client_secret"
AUTH_CODE = "code_from_url"

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

token_data = response.json()
print(f"Access Token: {token_data['access_token']}")
print(f"Refresh Token: {token_data['refresh_token']}")
```

### Step 4: Get Account ID
```python
import requests

ACCESS_TOKEN = "your_access_token"

response = requests.get(
    "https://connect.icmarkets.com/api/v2/tradingaccounts",
    headers={"Authorization": f"Bearer {ACCESS_TOKEN}"}
)

accounts = response.json()['data']
for account in accounts:
    print(f"Account: {account['accountNumber']} -> ID: {account['ctidTraderAccountId']}")
```

### Step 5: Configure .env File
Create `.env` file in project root:
```bash
# cTrader API Credentials
CTRADER_CLIENT_ID=your_client_id_here
CTRADER_CLIENT_SECRET=your_client_secret_here
CTRADER_ACCESS_TOKEN=your_access_token_here
CTRADER_REFRESH_TOKEN=your_refresh_token_here
CTRADER_ACCOUNT_ID=your_account_id_here

# Host Configuration
CTRADER_DEMO_HOST=demo.ctraderapi.com
CTRADER_PORT=5035

# Default symbol for testing
DEFAULT_SYMBOLS=EURUSD
```

## Understanding the Report

### Key Metrics Analyzed
1. **Data Latency**: Time between server timestamp and client receipt
2. **Depth Levels**: Number of bid/ask price levels available
3. **Update Frequency**: How often depth data changes
4. **Data Consistency**: Variability in depth levels over time

### Report Sections
- **Executive Summary**: Clear go/no-go recommendation
- **Quantitative Findings**: Detailed latency and depth statistics
- **Qualitative Observations**: Notable patterns and limitations
- **Raw Data Samples**: First few events for inspection
- **Strategic Recommendation**: Decision framework for MICRO-40

### Decision Criteria
- **GO**: >5 depth levels, <100ms latency
- **GO (with modifications)**: ≤5 levels, <200ms latency
- **NO-GO**: Only top-of-book or >500ms latency

## Best Practices

### Optimal Testing Times
- **London-New York overlap**: 14:00-16:00 GMT (highest liquidity)
- **Avoid weekends**: Markets closed
- **Avoid major news**: Volatility affects depth

### Troubleshooting
- **Connection issues**: Check credentials and network
- **Symbol not found**: Verify symbol name in cTrader platform
- **No depth data**: Some symbols may not support Level 2
- **High latency**: Check geographic location and network

### Expected Results
- **EURUSD**: Typically 5-10 depth levels, 50-200ms latency
- **GBPUSD**: Similar to EURUSD
- **Exotic pairs**: May have limited depth

## Output Files
- `docs/v4_reality_check_report.md`: Main report
- `docs/microstructure_raw_data.json`: Raw data for analysis
- `logs/microstructure_verification.log`: Detailed logs

## Next Steps After Verification
1. Review the generated report
2. Make go/no-go decision based on findings
3. If GO: Proceed with microstructure engine development
4. If NO-GO: Consider alternative strategies or brokers
