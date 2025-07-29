# FIX API Troubleshooting Flowchart & Decision Tree

**Purpose:** Systematic approach to diagnosing and resolving FIX API issues  
**Version:** 1.0  
**Date:** 2025-07-29  

---

## 🔄 **TROUBLESHOOTING DECISION TREE**

### START HERE: FIX API Not Working

```
┌─────────────────────────────────────┐
│         FIX API ISSUE               │
│    What is the symptom?             │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  A) Cannot connect to server        │
│  B) Connection drops immediately    │
│  C) Authentication fails            │
│  D) Market data not received        │
│  E) Orders rejected/not executed    │
│  F) Messages malformed/corrupted    │
└─────────────────┬───────────────────┘
                  │
                  ▼
         [Choose path A-F below]
```

---

## 🔌 **PATH A: CONNECTION ISSUES**

### A1: Cannot Connect to Server

```
Connection Timeout/Refused
         │
         ▼
┌─────────────────┐    YES    ┌──────────────────┐
│ Network access  │ ────────► │ Check host/port  │
│ to internet?    │           │ configuration    │
└─────────────────┘           └─────────┬────────┘
         │ NO                           │
         ▼                              ▼
┌─────────────────┐           ┌──────────────────┐
│ Fix network     │           │ Try alternative  │
│ connectivity    │           │ endpoints        │
└─────────────────┘           └─────────┬────────┘
                                        │
                                        ▼
                              ┌──────────────────┐
                              │ Test with both:  │
                              │ Account-specific │
                              │ & Generic hosts  │
                              └──────────────────┘
```

**Host Configuration Check:**
```python
# Primary (Account-specific)
PRIMARY_HOSTS = {
    "price": ("demo-uk-eqx-01.p.c-trader.com", 5211),
    "trade": ("demo-uk-eqx-01.p.c-trader.com", 5212)
}

# Fallback (Generic)
FALLBACK_HOSTS = {
    "price": ("h51.p.ctrader.com", 5201),
    "trade": ("h51.p.ctrader.com", 5201)
}
```

### A2: SSL/TLS Issues

```
SSL Connection Error
         │
         ▼
┌─────────────────┐    NO     ┌──────────────────┐
│ SSL context     │ ────────► │ Create proper    │
│ configured?     │           │ SSL context      │
└─────────────────┘           └─────────┬────────┘
         │ YES                          │
         ▼                              ▼
┌─────────────────┐           ┌──────────────────┐
│ Check cert      │           │ Disable hostname │
│ verification    │           │ verification     │
│ settings        │           └──────────────────┘
└─────────────────┘
```

**SSL Fix:**
```python
import ssl

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE
```

---

## 🔐 **PATH B: AUTHENTICATION ISSUES**

### B1: Logon Rejected

```
Authentication Failed
         │
         ▼
┌─────────────────┐    NO     ┌──────────────────┐
│ Credentials     │ ────────► │ Verify account   │
│ correct?        │           │ number/password  │
└─────────────────┘           └─────────┬────────┘
         │ YES                          │
         ▼                              ▼
┌─────────────────┐           ┌──────────────────┐
│ Check session   │           │ Use FIX API      │
│ identifiers     │           │ password, not    │
└─────────┬───────┘           │ trading password │
          │                   └──────────────────┘
          ▼
┌─────────────────┐
│ Verify SubID    │
│ configuration   │
└─────────────────┘
```

**Session ID Verification:**
```python
# CRITICAL: Must match exactly
SENDER_COMP_ID = f"demo.icmarkets.{ACCOUNT_NUMBER}"
TARGET_COMP_ID = "cServer"

# Price Session
PRICE_TARGET_SUB_ID = "QUOTE"
PRICE_SENDER_SUB_ID = "QUOTE"

# Trade Session
TRADE_TARGET_SUB_ID = "TRADE"
TRADE_SENDER_SUB_ID = "TRADE"
```

### B2: Session Management Issues

```
Session Rejected/Dropped
         │
         ▼
┌─────────────────┐    NO     ┌──────────────────┐
│ Sequence nums   │ ────────► │ Reset sequence   │
│ starting at 1?  │           │ numbers to 1     │
└─────────────────┘           └─────────┬────────┘
         │ YES                          │
         ▼                              ▼
┌─────────────────┐           ┌──────────────────┐
│ ResetSeqNumFlag │           │ Set flag to Y    │
│ set to Y?       │           │ in logon message │
└─────────┬───────┘           └──────────────────┘
          │ YES
          ▼
┌─────────────────┐
│ Check heartbeat │
│ interval (30s)  │
└─────────────────┘
```

---

## 📊 **PATH C: MARKET DATA ISSUES**

### C1: Market Data Request Rejected

```
Market Data Rejected
         │
         ▼
┌─────────────────┐    NO     ┌──────────────────┐
│ Using numeric   │ ────────► │ Convert to       │
│ symbol IDs?     │           │ numeric IDs      │
└─────────────────┘           └─────────┬────────┘
         │ YES                          │
         ▼                              ▼
┌─────────────────┐           ┌──────────────────┐
│ NoRelatedSym    │           │ Symbol 1=EURUSD  │
│ field (146)     │           │ Symbol 2=GBPUSD  │
│ included?       │           │ Symbol 3=USDJPY  │
└─────────┬───────┘           └──────────────────┘
          │ YES
          ▼
┌─────────────────┐
│ Check session   │
│ type (QUOTE)    │
└─────────────────┘
```

**Market Data Request Format:**
```python
market_data_request = [
    ("8", "FIX.4.4"),
    ("35", "V"),                    # MsgType
    ("49", sender_comp_id),
    ("56", target_comp_id),
    ("57", "QUOTE"),               # Must be QUOTE
    ("50", "QUOTE"),               # Must be QUOTE
    ("34", str(seq_num)),
    ("52", timestamp),
    ("262", f"MDR_{int(time.time())}"),  # Unique MDReqID
    ("263", "1"),                  # SubscriptionRequestType
    ("264", "0"),                  # MarketDepth
    ("146", "1"),                  # NoRelatedSym (REQUIRED!)
    ("55", "1")                    # Symbol (numeric only!)
]
```

### C2: No Market Data Received

```
No Market Data Response
         │
         ▼
┌─────────────────┐    NO     ┌──────────────────┐
│ Subscription    │ ────────► │ Check symbol     │
│ successful?     │           │ availability     │
└─────────────────┘           └─────────┬────────┘
         │ YES                          │
         ▼                              ▼
┌─────────────────┐           ┌──────────────────┐
│ Wait longer     │           │ Try Symbol 1     │
│ (up to 30s)     │           │ (EURUSD) first   │
└─────────┬───────┘           └──────────────────┘
          │
          ▼
┌─────────────────┐
│ Check message   │
│ parsing logic   │
└─────────────────┘
```

---

## 💰 **PATH D: ORDER EXECUTION ISSUES**

### D1: Orders Rejected

```
Order Rejected
         │
         ▼
┌─────────────────┐    NO     ┌──────────────────┐
│ Trading enabled │ ────────► │ Contact IC       │
│ on account?     │           │ Markets support  │
└─────────────────┘           └─────────┬────────┘
         │ YES                          │
         ▼                              ▼
┌─────────────────┐           ┌──────────────────┐
│ Using TRADE     │           │ Enable demo      │
│ session?        │           │ trading on       │
└─────────┬───────┘           │ account          │
          │ YES               └──────────────────┘
          ▼
┌─────────────────┐
│ Check order     │
│ format/fields   │
└─────────────────┘
```

**Order Placement Checklist:**
```python
# 1. Use TRADE session
target_sub_id = "TRADE"
sender_sub_id = "TRADE"
port = 5212

# 2. Verify order format
new_order_single = [
    ("8", "FIX.4.4"),
    ("35", "D"),                   # NewOrderSingle
    ("49", sender_comp_id),
    ("56", target_comp_id),
    ("57", "TRADE"),              # CRITICAL: TRADE session
    ("50", "TRADE"),              # CRITICAL: TRADE session
    ("34", str(seq_num)),
    ("52", timestamp),
    ("11", order_id),             # Unique ClOrdID
    ("55", "1"),                  # Numeric symbol ID
    ("54", "1"),                  # Side: 1=BUY, 2=SELL
    ("38", "1000"),               # OrderQty
    ("40", "1"),                  # OrdType: 1=Market
    ("59", "0"),                  # TimeInForce: 0=Day
    ("60", timestamp)             # TransactTime
]

# 3. Wait for ExecutionReport (MsgType=8)
```

### D2: Orders Not Appearing in Account

```
No Order in Account
         │
         ▼
┌─────────────────┐    NO     ┌──────────────────┐
│ ExecutionReport │ ────────► │ Order was        │
│ received?       │           │ rejected         │
└─────────────────┘           └─────────┬────────┘
         │ YES                          │
         ▼                              ▼
┌─────────────────┐           ┌──────────────────┐
│ Check order     │           │ Check rejection  │
│ status in       │           │ reason in        │
│ ExecutionReport │           │ message          │
└─────────┬───────┘           └──────────────────┘
          │
          ▼
┌─────────────────┐
│ Verify account  │
│ synchronization │
└─────────────────┘
```

---

## 📝 **PATH E: MESSAGE FORMAT ISSUES**

### E1: Malformed Messages

```
Message Format Error
         │
         ▼
┌─────────────────┐    NO     ┌──────────────────┐
│ Proper SOH      │ ────────► │ Use \x01 as     │
│ delimiters?     │           │ field separator  │
└─────────────────┘           └─────────┬────────┘
         │ YES                          │
         ▼                              ▼
┌─────────────────┐           ┌──────────────────┐
│ Correct         │           │ Fix message      │
│ checksum?       │           │ construction     │
└─────────┬───────┘           └──────────────────┘
          │ YES
          ▼
┌─────────────────┐
│ Verify field    │
│ ordering        │
└─────────────────┘
```

**Message Construction Fix:**
```python
def build_fix_message(fields):
    """Build properly formatted FIX message."""
    # Build message body (exclude BeginString and BodyLength)
    body = ""
    for tag, value in fields[2:]:
        body += f"{tag}={value}\x01"
    
    # Calculate body length
    body_length = len(body)
    
    # Construct full message
    message = f"{fields[0][1]}\x01{fields[1][0]}={body_length}\x01{body}"
    
    # Calculate checksum
    checksum = sum(ord(c) for c in message) % 256
    message += f"10={checksum:03d}\x01"
    
    return message.encode('ascii')
```

### E2: Checksum Errors

```
Checksum Mismatch
         │
         ▼
┌─────────────────┐    NO     ┌──────────────────┐
│ Including all   │ ────────► │ Include entire   │
│ message chars   │           │ message in       │
│ in calculation? │           │ checksum calc    │
└─────────────────┘           └─────────┬────────┘
         │ YES                          │
         ▼                              ▼
┌─────────────────┐           ┌──────────────────┐
│ Modulo 256      │           │ Fix calculation  │
│ operation?      │           │ method           │
└─────────┬───────┘           └──────────────────┘
          │ YES
          ▼
┌─────────────────┐
│ Check for       │
│ encoding issues │
└─────────────────┘
```

---

## 🚨 **EMERGENCY RECOVERY PROCEDURES**

### Complete System Reset

```
1. Stop all FIX connections
2. Clear sequence numbers
3. Reset configuration to known working state
4. Test with minimal example
5. Gradually add functionality
```

### Known Working Configuration
```python
# EMERGENCY FALLBACK CONFIG
EMERGENCY_CONFIG = {
    "host": "demo-uk-eqx-01.p.c-trader.com",
    "price_port": 5211,
    "trade_port": 5212,
    "sender_comp_id": "demo.icmarkets.9533708",
    "target_comp_id": "cServer",
    "price_sub_id": "QUOTE",
    "trade_sub_id": "TRADE",
    "symbol_id": 1,  # EURUSD only
    "ssl_verify": False
}
```

### Minimal Test Script
```python
def emergency_test():
    """Minimal test to verify basic functionality."""
    try:
        # Test connection only
        config = ICMarketsConfig("demo", "9533708")
        manager = ICMarketsFIXManager(config)
        
        success = manager.start()
        if success:
            print("✅ Connection successful")
            manager.stop()
            return True
        else:
            print("❌ Connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
```

---

## 📋 **DIAGNOSTIC COMMANDS**

### Network Connectivity Test
```bash
# Test host reachability
ping demo-uk-eqx-01.p.c-trader.com
telnet demo-uk-eqx-01.p.c-trader.com 5211
telnet demo-uk-eqx-01.p.c-trader.com 5212

# Alternative hosts
ping h51.p.ctrader.com
telnet h51.p.ctrader.com 5201
```

### Python Environment Check
```python
import ssl
import socket
import time

print(f"SSL version: {ssl.OPENSSL_VERSION}")
print(f"Python version: {sys.version}")
print(f"Socket support: {hasattr(socket, 'AF_INET')}")
```

### Message Debugging
```python
def debug_message(message):
    """Debug FIX message content."""
    print("Raw bytes:", message)
    print("Decoded:", message.decode('ascii', errors='ignore'))
    print("Fields:", message.decode('ascii').replace('\x01', '|'))
    
    # Parse fields
    fields = message.decode('ascii').split('\x01')[:-1]
    for field in fields:
        if '=' in field:
            tag, value = field.split('=', 1)
            print(f"  Tag {tag}: {value}")
```

---

## 🔄 **ESCALATION PATHS**

### Level 1: Self-Service
1. Check this troubleshooting guide
2. Verify configuration against working examples
3. Test with minimal script
4. Check network connectivity

### Level 2: Code Review
1. Compare with working implementation
2. Validate message formats
3. Check sequence number management
4. Verify SSL configuration

### Level 3: External Support
1. Contact IC Markets FIX API support
2. Verify account configuration
3. Request trading permissions
4. Report connectivity issues

### Level 4: System Recovery
1. Restore from last working backup
2. Rebuild from minimal example
3. Implement step-by-step
4. Document new issues found

---

**This flowchart provides systematic diagnosis of FIX API issues. Follow the decision tree based on observed symptoms for fastest resolution.**

