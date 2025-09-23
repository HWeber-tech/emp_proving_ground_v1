# IMMEDIATE FIX - Get Correct Account ID

> **Status:** Legacy – applies to the deprecated cTrader OpenAPI tooling.
> See the FIX-only [Integration Policy](policies/integration_policy.md)
> before attempting to reuse these steps.

## 🚨 The Issue
Your access token has no visibility to any accounts. This is a **scope issue**, not an account ID issue.

## 🎯 Exact Fix Steps

### Step 1: Regenerate Access Token
**Open this exact URL in your browser:**
```
https://connect.icmarkets.com/oauth/authorize?client_id=12066_pp7glRUIsDmFaeXcBzMljuAz4083XEzL4GnegdcxiVKhshAXDt&redirect_uri=http://localhost/&scope=trading
```

### Step 2: Complete OAuth Flow
1. **Log in with your IC Markets cTrader credentials**
2. **When the permission screen appears, SELECT YOUR DEMO ACCOUNT** (don't just view)
3. **Grant "trading" permissions**
4. **After authorization, you'll be redirected to http://localhost/?code=XXXXXXX**
5. **Copy the authorization code (the XXXXXXX part)**

### Step 3: Exchange Code for New Token
Run this Python script with your authorization code:

```python
import requests

CLIENT_ID = "12066_pp7glRUIsDmFaeXcBzMljuAz4083XEzL4GnegdcxiVKhshAXDt"
CLIENT_SECRET = "lbcjnJTdbd1I4QCeSArlQyh4zN8r84EnU78idxHzrbpYhHvGlv"
AUTH_CODE = "YOUR_CODE_HERE"

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
    data = response.json()
    print("New Access Token:", data['access_token'])
    print("New Refresh Token:", data['refresh_token'])
    print("Account ID will be discovered automatically")
else:
    print("Error:", response.status_code, response.text)
```

### Step 4: Update .env
Replace these values in your .env file:
```
CTRADER_ACCESS_TOKEN=your_new_access_token
CTRADER_REFRESH_TOKEN=your_new_refresh_token
```

### Step 5: Discover Account ID
After updating, run:
```bash
python scripts/discover_account.py
```

This will show you the exact ctidTraderAccountId to use.

## 🎯 Key Point
**You don't need to manually find the account ID** - the tools will discover it automatically once you have the correct access token with proper scope.
