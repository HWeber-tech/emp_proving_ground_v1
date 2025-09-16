# Get Your cTrader Account ID

> **Status:** Legacy – the runtime no longer consumes cTrader OpenAPI
> credentials. This lookup guide remains for archival purposes and is
> superseded by the FIX-only [Integration Policy](policies/integration_policy.md).

## Quick Method

1. **Log into IC Markets Client Portal**
   - Go to: https://secure.icmarkets.com/
   - Login with your credentials

2. **Find Your Demo Account**
   - Navigate to "Trading Accounts" or "cTrader Accounts"
   - Look for your **demo account** (not live)
   - Note the **account number** (usually 6-8 digits)

3. **Update Your .env File**
   ```bash
   CTRADER_ACCOUNT_ID=12345678  # Replace with your actual account number
   ```

## Alternative Method

If you can't find it in the portal:
1. **Contact IC Markets support** and ask for your "cTrader demo account ID"
2. **Check your cTrader platform** - the account number is displayed in the top-left corner

## Test Your Account ID

Once you have your account ID, run:

```bash
python scripts/working_verification.py 12345678
```

Replace `12345678` with your actual account ID.

## Expected Output

If successful, you'll see:
- ✅ Connection established
- ✅ Authentication successful
- ✅ EURUSD depth data streaming
- 📊 Real-time metrics
- 🎯 Final GO/NO-GO recommendation
