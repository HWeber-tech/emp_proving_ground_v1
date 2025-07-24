# EMP v4.0 FIX Implementation Guide

## Overview
This guide provides comprehensive instructions for setting up and using the FIX protocol connection to IC Markets cTrader.

## Architecture

### Components
1. **FIXApplication** (`src/operational/fix_application.py`) - Core FIX protocol handler
2. **FIXConnectionManager** (`src/operational/fix_connection_manager.py`) - Session lifecycle manager
3. **FIXSensoryOrgan** (`src/sensory/organs/fix_sensory_organ.py`) - Market data ingestion
4. **Configuration Files** (`config/fix/`) - Session and symbol mappings

### Connection Types
- **Price Connection**: Port 5211 (Market Data)
- **Trade Connection**: Port 5212 (Order Management)

## Quick Start

### 1. Configure Credentials
Update your `.env` file with your IC Markets FIX credentials:

```bash
# FIX Protocol Configuration
FIX_PRICE_SENDER_COMP_ID=your_price_sender_id
FIX_PRICE_USERNAME=your_price_username
FIX_PRICE_PASSWORD=your_price_password

FIX_TRADE_SENDER_COMP_ID=your_trade_sender_id
FIX_TRADE_USERNAME=your_trade_username
FIX_TRADE_PASSWORD=your_trade_password
```

### 2. Verify Connection
Run the verification script:

```bash
python scripts/verify_fix_connection.py
```

Expected output:
```
✅ FIX credentials found in configuration
🚀 Starting FIX sessions...
⏳ Waiting for connections to establish...
✅ SUCCESSFUL LOGON: FIX price session
✅ SUCCESSFUL LOGON: FIX trade session
```

### 3. Test Market Data
Use the sensory organ to subscribe to market data:

```python
from src.governance.system_config import SystemConfig
from src.operational.fix_connection_manager import FIXConnectionManager
from src.sensory.organs.fix_sensory_organ import FIXSensoryOrgan

# Initialize
config = SystemConfig()
manager = FIXConnectionManager(config)
manager.start_sessions()

# Create sensory organ
fix_app = manager.get_application('price')
sensory_organ = FIXSensoryOrgan(event_bus, config, fix_app)

# Subscribe to EURUSD
await sensory_organ.subscribe_to_market_data("EURUSD")
```

## Configuration Files

### Session Configuration
- `config/fix/ctrader_price_session.cfg` - Price session settings
- `config/fix/ctrader_trade_session.cfg` - Trade session settings
- `config/fix/FIX44.xml` - FIX 4.4 data dictionary

### Symbol Mapping
- `config/fix/symbol_mapping.json` - Maps human-readable symbols to FIX symbol IDs

## Usage Examples

### Basic Market Data Subscription
```python
from src.sensory.organs.fix_sensory_organ import FIXSensoryOrgan

# Subscribe to full depth
await sensory_organ.subscribe_to_market_data("EURUSD")

# Subscribe with limited depth (fallback)
await sensory_organ.subscribe_to_market_data_limited("EURUSD", depth=10)
```

### CVD (Cumulative Volume Delta) Monitoring
```python
# Get current CVD state
cvd = sensory_organ.get_cvd_state("EURUSD")

# Get last trade info
price, size = sensory_organ.get_last_trade_info("EURUSD")

# Reset CVD state
sensory_organ.reset_cvd_state("EURUSD")
```

### Order Book Access
```python
# Get current order book
order_book = sensory_organ.get_current_order_book("EURUSD")
if order_book:
    print(f"Best Bid: {order_book.best_bid}")
    print(f"Best Ask: {order_book.best_ask}")
    print(f"Bid Levels: {len(order_book.bids)}")
    print(f"Ask Levels: {len(order_book.asks)}")
```

## Troubleshooting

### Common Issues

1. **Connection Failed**
   - Verify credentials in `.env`
   - Check network connectivity to `demo-uk-eqx-01.p.c-trader.com`
   - Ensure ports 5211/5212 are not blocked

2. **Market Data Rejection**
   - Try limited depth subscription: `subscribe_to_market_data_limited()`
   - Verify symbol mapping in `config/fix/symbol_mapping.json`

3. **No Data Received**
   - Check subscription status: `get_subscribed_symbols()`
   - Verify FIX credentials are correct

### Debug Mode
Enable debug logging:
```python
import logging
logging.getLogger('src.operational.fix_application').setLevel(logging.DEBUG)
logging.getLogger('src.operational.fix_connection_manager').setLevel(logging.DEBUG)
```

## Production Deployment

### Environment Variables
```bash
# Production settings
CONNECTION_PROTOCOL=fix
ENVIRONMENT=live
LOG_LEVEL=INFO

# Live FIX endpoints
FIX_PRICE_SENDER_COMP_ID=prod_price_id
FIX_PRICE_USERNAME=prod_price_user
FIX_PRICE_PASSWORD=prod_price_pass
FIX_TRADE_SENDER_COMP_ID=prod_trade_id
FIX_TRADE_USERNAME=prod_trade_user
FIX_TRADE_PASSWORD=prod_trade_pass
```

### Monitoring
- Check `logs/fix_verification.log` for connection status
- Monitor `logs/fix/price_log/` and `logs/fix/trade_log/` for session logs
- Use `get_connection_status()` for runtime monitoring

## Advanced Features

### CVD Divergence Detection
The system includes built-in CVD divergence detection:
- Tracks cumulative volume delta per symbol
- Identifies divergences between price and volume
- Provides early warning signals for trend changes

### Multi-Symbol Support
Subscribe to multiple symbols simultaneously:
```python
symbols = ["EURUSD", "GBPUSD", "XAUUSD"]
for symbol in symbols:
    await sensory_organ.subscribe_to_market_data(symbol)
```

### Real-time Processing
All market data is processed in real-time with:
- Microsecond-level timestamps
- Full order book reconstruction
- Trade print processing for CVD calculation

## API Reference

### FIXSensoryOrgan Methods
- `subscribe_to_market_data(symbol)` - Subscribe to full depth
- `subscribe_to_market_data_limited(symbol, depth)` - Subscribe with depth limit
- `get_current_order_book(symbol)` - Get current order book
- `get_cvd_state(symbol)` - Get CVD value
- `unsubscribe_all()` - Unsubscribe from all symbols

### FIXConnectionManager Methods
- `start_sessions()` - Start both price and trade sessions
- `stop_sessions()` - Stop all sessions
- `get_connection_status()` - Get connection status
- `is_connected(session_type)` - Check specific session status

## Support
For issues or questions:
1. Check this guide first
2. Review logs in `logs/fix/`
3. Run verification script: `python scripts/verify_fix_connection.py`
4. Contact support with log files
