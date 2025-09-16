# cTrader Credential Troubleshooting Guide

> **Status:** Legacy – this procedure targets the deprecated cTrader OpenAPI flow.
> The integration is blocked by the [Integration Policy](policies/integration_policy.md)
> and is preserved for historical reference only.

## Current Issue: No Accounts Found

**Error**: `No accounts found for this access token`

**Root Cause**: Your access token doesn't have the necessary scope or account permissions.

## Solution Steps

### Step 1: Regenerate Access Token with Correct Scope

You need to regenerate your access token with the `trading` scope. Here's the exact process:

#### 1.1 Generate New Authorization URL
```
https://connect.icmarkets.com/oauth/authorize?client_id=12066_pp7glRUIsDmFaeXcBzMljuAz4083XEzL4GnegdcxiVKhshAXDt&redirect_uri=http://localhost/&scope=trading
```

#### 1.2 Complete OAuth Flow
1. **Open the URL above in your browser**
2. **Log in with your IC Markets cTrader credentials**
3. **Select your demo account** when prompted for permissions
4. **Copy the authorization code** from the redirect URL (after `?code=`)

#### 1.3 Exchange Code for New Token
Run this Python script to get your new tokens:

```python
import requests

# Your credentials
CLIENT_ID = "12066_pp7glRUIsDmFaeXcBzMljuAz4083XEzL4GnegdcxiVKhshAXDt"
CLIENT_SECRET = "lbcjnJTdbd1I4QCeSArlQyh4zN8r84EnU78idxHzrbpYhHvGlv"
AUTH_CODE = "PASTE_THE_CODE_FROM_URL_HERE"

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
    print("New Access Token:", token_data['access_token'])
    print("New Refresh Token:", token_data['refresh_token'])
else:
    print("Error:", response.status_code, response.text)
```

### Step 2: Update Your .env File

Replace the old tokens with the new ones:

```bash
# Update these values in your .env file
CTRADER_ACCESS_TOKEN=your_new_access_token_here
CTRADER_REFRESH_TOKEN=your_new_refresh_token_here
```

### Step 3: Verify New Credentials

After updating, run the test again:
```bash
python scripts/test_real_credentials.py
```

## Expected Results

When successful, you should see:
```
✅ Found accounts:
------------------------------------------------------------
Account ID: 1234567890  (10-12 digit ctidTraderAccountId)
Account Number: 9509437  (Your human-readable login)
Type: DEMO
Balance: 10000.00
------------------------------------------------------------
✅ CREDENTIALS VALIDATION SUCCESSFUL
   Use account ID: 1234567890
   EURUSD symbol ID: 1
```

## Important Notes

1. **Account ID vs Login Number**: The 7-digit number (9509437) is your human login, but the API expects the 10-12 digit `ctidTraderAccountId`
2. **Demo vs Live**: Ensure you're using demo credentials for testing
3. **Token Scope**: Must include `trading` scope, not just `accounts`
4. **Token Expiry**: Access tokens expire in ~30 days, refresh tokens are long-lived

## Quick Verification Checklist

- [ ] Regenerated access token with `trading` scope
- [ ] Selected demo account during OAuth consent
- [ ] Updated .env file with new tokens
- [ ] Verified account discovery shows your account
- [ ] Confirmed EURUSD symbol ID

## Next Steps

Once you have the correct credentials, the microstructure verification tool will work automatically with the discovered account ID.
