# IC Markets FIX API Complete Setup & Troubleshooting Guide

**Version:** 1.0  
**Date:** 2025-07-29  
**Status:** Production-Ready Implementation  

---

## 🎯 **OVERVIEW**

This guide provides complete instructions for setting up, configuring, and troubleshooting the IC Markets FIX API implementation. It captures all the hard-won knowledge from getting the system working, including common pitfalls and their solutions.

---

## 📋 **TABLE OF CONTENTS**

1. [Quick Start](#quick-start)
2. [Complete Setup Process](#complete-setup-process)
3. [Configuration Details](#configuration-details)
4. [Common Issues & Solutions](#common-issues--solutions)
5. [Testing & Validation](#testing--validation)
6. [Troubleshooting Checklist](#troubleshooting-checklist)
7. [Advanced Debugging](#advanced-debugging)

---

## 🚀 **QUICK START**

### Prerequisites
- IC Markets demo account with FIX API access
- Python 3.11+ environment
- SSL/TLS support enabled

### Minimal Working Example
```python
from src.operational.icmarkets_api import ICMarketsFIXManager
from src.operational.icmarkets_config import ICMarketsConfig

# Create configuration
config = ICMarketsConfig(
    environment="demo",
    account_number="YOUR_ACCOUNT_NUMBER"
)

# Initialize and start
manager = ICMarketsFIXManager(config)
success = manager.start()

if success:
    # Subscribe to market data
    manager.subscribe_market_data([1])  # Symbol 1 = EURUSD
    
    # Place order
    order_id = manager.place_market_order(1, "BUY", 1000)
    print(f"Order placed: {order_id}")
    
    manager.stop()
```

---

## 🔧 **COMPLETE SETUP PROCESS**

### Step 1: Account Configuration

**IC Markets Account Requirements:**
- Demo account with FIX API enabled
- Account number (e.g., 9533708)
- Password for FIX API access
- Trading permissions enabled

**Critical Account Settings:**
```
Account Type: Demo
FIX API Access: Enabled
Trading Permissions: Enabled
Symbol Access: Verify EURUSD (Symbol 1) available
```

### Step 2: Network Configuration

**Connection Endpoints:**
```python
# WORKING CONFIGURATION (Account-Specific)
PRICE_HOST = "demo-uk-eqx-01.p.c-trader.com"
PRICE_PORT = 5211

TRADE_HOST = "demo-uk-eqx-01.p.c-trader.com"  
TRADE_PORT = 5212

# Alternative Configuration (Generic)
GENERIC_HOST = "h51.p.ctrader.com"
GENERIC_PORT = 5201
```

**SSL Configuration:**
```python
import ssl

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE
```

### Step 3: FIX Protocol Configuration

**Session Identifiers:**
```python
# CRITICAL: These must match exactly
SENDER_COMP_ID = "demo.icmarkets.{ACCOUNT_NUMBER}"
TARGET_COMP_ID = "cServer"

# Session-specific SubIDs
PRICE_TARGET_SUB_ID = "QUOTE"
PRICE_SENDER_SUB_ID = "QUOTE"

TRADE_TARGET_SUB_ID = "TRADE"  
TRADE_SENDER_SUB_ID = "TRADE"
```

**Message Structure:**
```python
# Logon Message Format
logon_fields = [
    ("8", "FIX.4.4"),           # BeginString
    ("35", "A"),                # MsgType (Logon)
    ("49", sender_comp_id),     # SenderCompID
    ("56", target_comp_id),     # TargetCompID
    ("57", target_sub_id),      # TargetSubID
    ("50", sender_sub_id),      # SenderSubID
    ("34", str(seq_num)),       # MsgSeqNum
    ("52", timestamp),          # SendingTime
    ("98", "0"),                # EncryptMethod
    ("108", "30"),              # HeartBtInt
    ("141", "Y"),               # ResetSeqNumFlag
    ("553", account_number),    # Username
    ("554", password)           # Password
]
```

### Step 4: Symbol Configuration

**Working Symbol IDs:**
```python
SYMBOL_MAP = {
    1: "EURUSD",    # Confirmed working
    2: "GBPUSD",    # Limited functionality
    3: "USDJPY",    # Limited functionality
    # Add more as discovered via SecurityListRequest
}
```

**Symbol Discovery Process:**
```python
# Send SecurityListRequest to get all available symbols
security_list_request = [
    ("8", "FIX.4.4"),
    ("35", "x"),                # MsgType (SecurityListRequest)
    ("49", sender_comp_id),
    ("56", target_comp_id),
    ("57", target_sub_id),
    ("50", sender_sub_id),
    ("34", str(seq_num)),
    ("52", timestamp),
    ("320", "0"),               # SecurityReqID
    ("559", "0")                # SecurityListRequestType
]
```

---

## ⚙️ **CONFIGURATION DETAILS**

### ICMarketsConfig Class
```python
class ICMarketsConfig:
    def __init__(self, environment="demo", account_number=None):
        self.environment = environment
        self.account_number = account_number
        
        # Connection settings
        if environment == "demo":
            self.price_host = "demo-uk-eqx-01.p.c-trader.com"
            self.price_port = 5211
            self.trade_host = "demo-uk-eqx-01.p.c-trader.com"
            self.trade_port = 5212
        
        # FIX settings
        self.sender_comp_id = f"demo.icmarkets.{account_number}"
        self.target_comp_id = "cServer"
        self.fix_version = "FIX.4.4"
        
        # Session settings
        self.heartbeat_interval = 30
        self.reset_seq_num = True
        self.ssl_enabled = True
```

### Connection Management
```python
def create_ssl_connection(host, port):
    """Create SSL connection with proper configuration."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    ssl_sock = ssl_context.wrap_socket(sock)
    ssl_sock.connect((host, port))
    
    return ssl_sock
```

---

## 🚨 **COMMON ISSUES & SOLUTIONS**

### Issue 1: Authentication Failure
**Symptoms:**
- Connection established but logon rejected
- "Invalid credentials" or similar error

**Solutions:**
1. **Verify Account Configuration:**
   ```python
   # Check account number format
   sender_comp_id = f"demo.icmarkets.{account_number}"
   # Must match exactly: demo.icmarkets.9533708
   ```

2. **Check Password:**
   ```python
   # Use FIX API password, not trading platform password
   password = "YOUR_FIX_API_PASSWORD"
   ```

3. **Verify Session IDs:**
   ```python
   # Price session
   target_sub_id = "QUOTE"
   sender_sub_id = "QUOTE"
   
   # Trade session  
   target_sub_id = "TRADE"
   sender_sub_id = "TRADE"
   ```

### Issue 2: Connection Timeout
**Symptoms:**
- Socket connection fails
- Timeout errors during connection

**Solutions:**
1. **Check Host Configuration:**
   ```python
   # Use account-specific host (preferred)
   host = "demo-uk-eqx-01.p.c-trader.com"
   
   # Alternative generic host
   host = "h51.p.ctrader.com"
   ```

2. **Verify Ports:**
   ```python
   # Account-specific ports
   price_port = 5211
   trade_port = 5212
   
   # Generic port
   port = 5201
   ```

3. **SSL Configuration:**
   ```python
   ssl_context = ssl.create_default_context()
   ssl_context.check_hostname = False
   ssl_context.verify_mode = ssl.CERT_NONE
   ```

### Issue 3: Market Data Rejection
**Symptoms:**
- "Tag not defined for this message type, field=55"
- "Required tag missing, field=146"

**Solutions:**
1. **Use Numeric Symbol IDs:**
   ```python
   # Correct: Use numeric symbol ID
   symbol = "1"  # For EURUSD
   
   # Wrong: Don't use text symbols
   symbol = "EURUSD"  # This will be rejected
   ```

2. **Proper MarketDataRequest Format:**
   ```python
   market_data_request = [
       ("8", "FIX.4.4"),
       ("35", "V"),                    # MsgType
       ("49", sender_comp_id),
       ("56", target_comp_id),
       ("57", "QUOTE"),               # TargetSubID
       ("50", "QUOTE"),               # SenderSubID
       ("34", str(seq_num)),
       ("52", timestamp),
       ("262", "MDR_" + str(int(time.time()))),  # MDReqID
       ("263", "1"),                  # SubscriptionRequestType
       ("264", "0"),                  # MarketDepth
       ("146", "1"),                  # NoRelatedSym (required!)
       ("55", str(symbol_id))         # Symbol (numeric)
   ]
   ```

### Issue 4: Order Placement Failure
**Symptoms:**
- "TRADING_DISABLED:Trading is disabled"
- Orders not appearing in account

**Solutions:**
1. **Verify Trading Permissions:**
   - Contact IC Markets to enable demo trading
   - Confirm account has trading permissions

2. **Use Trade Session:**
   ```python
   # Ensure using trade session, not price session
   target_sub_id = "TRADE"
   sender_sub_id = "TRADE"
   port = 5212  # Trade port
   ```

3. **Proper Order Format:**
   ```python
   new_order_single = [
       ("8", "FIX.4.4"),
       ("35", "D"),                   # MsgType (NewOrderSingle)
       ("49", sender_comp_id),
       ("56", target_comp_id),
       ("57", "TRADE"),              # TargetSubID
       ("50", "TRADE"),              # SenderSubID
       ("34", str(seq_num)),
       ("52", timestamp),
       ("11", order_id),             # ClOrdID
       ("55", str(symbol_id)),       # Symbol (numeric)
       ("54", "1" if side == "BUY" else "2"),  # Side
       ("38", str(quantity)),        # OrderQty
       ("40", "1"),                  # OrdType (Market)
       ("59", "0"),                  # TimeInForce (Day)
       ("60", timestamp)             # TransactTime
   ]
   ```

### Issue 5: Message Parsing Errors
**Symptoms:**
- Malformed FIX messages
- Checksum errors
- Field parsing failures

**Solutions:**
1. **Proper Message Construction:**
   ```python
   def build_fix_message(fields):
       """Build properly formatted FIX message."""
       # Build message body
       body = ""
       for tag, value in fields[2:]:  # Skip BeginString and BodyLength
           body += f"{tag}={value}\x01"
       
       # Calculate body length
       body_length = len(body)
       
       # Add BodyLength field
       message = f"{fields[0][1]}\x01{fields[1][0]}={body_length}\x01{body}"
       
       # Calculate checksum
       checksum = sum(ord(c) for c in message) % 256
       message += f"10={checksum:03d}\x01"
       
       return message.encode('ascii')
   ```

2. **Message Validation:**
   ```python
   def validate_fix_message(message):
       """Validate FIX message format."""
       if not message.startswith(b'8=FIX.4.4'):
           raise ValueError("Invalid BeginString")
       
       if not message.endswith(b'\x01'):
           raise ValueError("Message must end with SOH")
       
       # Additional validation as needed
   ```

---

## ✅ **TESTING & VALIDATION**

### Basic Connection Test
```python
def test_connection():
    """Test basic FIX API connection."""
    config = ICMarketsConfig(
        environment="demo",
        account_number="YOUR_ACCOUNT"
    )
    
    manager = ICMarketsFIXManager(config)
    
    # Test authentication
    success = manager.start()
    assert success, "Authentication failed"
    
    # Test market data
    manager.subscribe_market_data([1])
    time.sleep(5)  # Wait for data
    
    # Test order placement
    order_id = manager.place_market_order(1, "BUY", 1000)
    assert order_id is not None, "Order placement failed"
    
    manager.stop()
    print("All tests passed!")
```

### Market Data Validation
```python
def validate_market_data():
    """Validate market data reception."""
    manager = ICMarketsFIXManager(config)
    manager.start()
    
    # Subscribe and wait
    manager.subscribe_market_data([1])
    time.sleep(10)
    
    # Check if data received
    market_data = manager.get_latest_market_data(1)
    assert market_data is not None, "No market data received"
    
    print(f"Market data: {market_data}")
    manager.stop()
```

### Order Execution Validation
```python
def validate_order_execution():
    """Validate order execution capability."""
    manager = ICMarketsFIXManager(config)
    manager.start()
    
    # Place test order
    order_id = manager.place_market_order(1, "BUY", 1000)
    
    # Wait for execution report
    time.sleep(5)
    
    # Check order status
    order_status = manager.get_order_status(order_id)
    assert order_status is not None, "No order status received"
    
    print(f"Order {order_id} status: {order_status}")
    manager.stop()
```

---

## 🔍 **TROUBLESHOOTING CHECKLIST**

### Pre-Connection Checklist
- [ ] Account number correct and FIX API enabled
- [ ] Password is FIX API password (not trading password)
- [ ] Network connectivity to IC Markets servers
- [ ] SSL/TLS support available
- [ ] Python environment properly configured

### Connection Issues
- [ ] Host and port configuration correct
- [ ] SSL context properly configured
- [ ] SenderCompID format: `demo.icmarkets.{account_number}`
- [ ] TargetCompID set to `cServer`
- [ ] Session SubIDs correct (QUOTE/TRADE)

### Authentication Issues
- [ ] Sequence numbers starting from 1
- [ ] ResetSeqNumFlag set to Y
- [ ] Heartbeat interval set to 30
- [ ] Username field (553) contains account number
- [ ] Password field (554) contains correct password

### Market Data Issues
- [ ] Using price session (port 5211)
- [ ] TargetSubID and SenderSubID both set to QUOTE
- [ ] Symbol ID is numeric (1, 2, 3, etc.)
- [ ] NoRelatedSym field (146) included
- [ ] MarketDepth set appropriately

### Order Placement Issues
- [ ] Using trade session (port 5212)
- [ ] TargetSubID and SenderSubID both set to TRADE
- [ ] Trading permissions enabled on account
- [ ] Symbol ID is numeric and valid
- [ ] Order quantity within limits
- [ ] Order type and time-in-force valid

---

## 🛠️ **ADVANCED DEBUGGING**

### Message Logging
```python
import logging

# Enable detailed FIX message logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def log_fix_message(direction, message):
    """Log FIX messages for debugging."""
    readable = message.decode('ascii', errors='ignore').replace('\x01', '|')
    logger.debug(f"{direction}: {readable}")
```

### Network Debugging
```python
def test_network_connectivity():
    """Test basic network connectivity."""
    import socket
    
    hosts = [
        ("demo-uk-eqx-01.p.c-trader.com", 5211),
        ("demo-uk-eqx-01.p.c-trader.com", 5212),
        ("h51.p.ctrader.com", 5201)
    ]
    
    for host, port in hosts:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                print(f"✅ {host}:{port} - Reachable")
            else:
                print(f"❌ {host}:{port} - Unreachable")
        except Exception as e:
            print(f"❌ {host}:{port} - Error: {e}")
```

### Message Analysis
```python
def analyze_fix_message(message):
    """Analyze FIX message structure."""
    fields = message.decode('ascii').split('\x01')[:-1]
    
    print("FIX Message Analysis:")
    for field in fields:
        if '=' in field:
            tag, value = field.split('=', 1)
            print(f"  Tag {tag}: {value}")
```

---

## 📞 **SUPPORT CONTACTS**

### IC Markets Support
- **FIX API Support:** Contact IC Markets technical support
- **Account Issues:** Contact account management
- **Trading Permissions:** Verify with account manager

### Common Support Requests
1. **Enable FIX API Access:** Request FIX API credentials
2. **Trading Permissions:** Enable demo trading on account
3. **Symbol Access:** Verify available trading symbols
4. **Connection Issues:** Report connectivity problems

---

## 📚 **ADDITIONAL RESOURCES**

### FIX Protocol Documentation
- FIX 4.4 Specification
- IC Markets FIX API Documentation
- cTrader FIX API Reference

### Code Examples
- Complete working implementation in `src/operational/icmarkets_api.py`
- Configuration example in `src/operational/icmarkets_config.py`
- Test suite in `tests/integration/test_icmarkets_complete.py`

---

## 🔄 **VERSION HISTORY**

### Version 1.0 (2025-07-29)
- Initial comprehensive guide
- Based on working implementation
- Includes all discovered solutions
- Complete troubleshooting coverage

---

**This guide represents the complete knowledge gained from successfully implementing the IC Markets FIX API. Keep it updated as new issues and solutions are discovered.**

