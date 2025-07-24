# EMP v4.0 FIX Protocol Setup Guide

## Overview
This guide provides step-by-step instructions for setting up FIX protocol connections to IC Markets cTrader for the EMP v4.0 Professional Predator.

## Architecture
- **Protocol**: FIX 4.4
- **Library**: simplefix (with quickfix compatibility layer)
- **Connections**: SSL-encrypted TCP
- **Environment**: Demo (demo-uk-eqx-01.p.c-trader.com)

## Connection Details
- **Price Feed**: demo-uk-eqx-01.p.c-trader.com:5211 (TargetSubID=QUOTE)
- **Trade Execution**: demo-uk-eqx-01.p.c-trader.com:5212 (TargetSubID=TRADE)

## Quick Start

### 1. Install Dependencies
```bash
pip install simplefix
```

### 2. Configure Credentials
Copy `.env.example` to `.env` and update with your IC Markets FIX credentials:

```bash
cp .env.example .env
```

Edit `.env` with your actual credentials:
```bash
# FIX Protocol Configuration (IC Markets cTrader FIX)
# Price Connection (Port 5211)
FIX_PRICE_SENDER_COMP_ID=your_price_sender_id
FIX_PRICE_USERNAME=your_price_username
FIX_PRICE_PASSWORD=your_price_password

# Trade Connection (Port 5212)
FIX_TRADE_SENDER_COMP_ID=your_trade_sender_id
FIX_TRADE_USERNAME=your_trade_username
FIX_TRADE_PASSWORD=your_trade_password
```

### 3. Verify Connections
Run the verification script:
```bash
python scripts/verify_fix_connection.py
```

## File Structure
```
config/fix/
├── ctrader_price_session.cfg    # Price connection config
├── ctrader_trade_session.cfg    # Trade connection config
└── FIX44.xml                   # FIX 4.4 data dictionary

src/operational/
├── enhanced_fix_application.py  # Enhanced FIX application
└── fix_connection_manager.py    # Connection manager (future)

scripts/
└── verify_fix_connection.py     # Connection verification
```

## Components

### EnhancedFIXApplication
The main FIX application class that handles:
- Connection establishment
- Logon/logout procedures
- Message routing
- Heartbeat management
- Error handling

### Configuration Files
- **Price Session**: Handles market data subscriptions
- **Trade Session**: Handles order placement and execution reports

## Testing

### Manual Testing
1. Run verification script
2. Check logs in `logs/fix/`
3. Verify successful logon messages

### Expected Output
```
SUCCESSFUL LOGON: price session
SUCCESSFUL LOGON: trade session
```

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Verify server address and port
   - Check firewall settings
   - Ensure credentials are correct

2. **Logon Rejected**
   - Verify SenderCompID, Username, and Password
   - Check account permissions
   - Ensure demo vs live environment

3. **SSL Certificate Issues**
   - SSL validation is disabled for demo environment
   - For live, ensure proper certificate chain

### Debug Mode
Enable debug logging:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Next Steps
After successful connection:
1. Implement market data subscriptions
2. Add order management
3. Integrate with trading strategies
4. Add risk management

## Support
For IC Markets FIX support:
- Email: support@icmarkets.com
- Documentation: IC Markets FIX API Guide
