# IC MARKETS FIX API DOCUMENTATION ANALYSIS
## Key Requirements and Implementation Details

**Analysis Date:** 2025-07-28  
**Source:** IC Markets FIX API Documentation  
**Protocol:** FIX 4.4 via cTrader Platform  

---

## CRITICAL FINDINGS

### Connection Configuration Requirements

**Host and Port Information:**
- **Host:** h51.p.ctrader.com (not demo-uk-eqx-01.p.c-trader.com)
- **Port:** 5201 (plain text), 5202 (SSL)
- **Protocol:** FIX 4.4
- **SSL:** Optional but recommended for production

**Message Format Requirements:**
- **SenderCompID:** `demo.icmarkets.YOUR_ACCOUNT_NUMBER`
- **TargetCompID:** `cServer`
- **SenderSubID:** `TRADE` or `QUOTE`
- **TargetSubID:** `TRADE` or `QUOTE`
- **HeartBeat:** 30 seconds

### Authentication Fields
- **Field 553 (Username):** Account number
- **Field 554 (Password):** FIX API password (not trading password)
- **Field 98 (EncryptMethod):** 0 (None)
- **Field 108 (HeartBtInt):** 30

---

## SYMBOL IDENTIFICATION METHOD

### FIX Symbol ID Discovery Process

**Critical Information:** Each trading symbol has a unique **FIX Symbol ID** that differs from standard symbol names.

**How to Find FIX Symbol IDs:**
1. Open cTrader or cAlgo
2. Right-click on desired symbol in Marketwatch
3. Select "Information" from popup menu
4. Scroll down to find "FIX Symbol ID"
5. Use this ID in FIX messages instead of regular symbol name

**Important Note:** FIX Symbol IDs vary between brokers and must be verified for each IC Markets account.

---

## CONNECTION TYPES

### Two Separate Connection Types Required

**Price Connection:**
- Purpose: Market data only
- Cannot be used for trading operations
- Separate credentials from trade connection

**Trade Connection:**
- Purpose: Trading operations only
- Cannot be used for price data
- Separate credentials from price connection

**Implication:** Our current implementation correctly uses separate sessions for price and trade operations.

---

## MESSAGE STRUCTURE REQUIREMENTS

### Logon Message Format
```
Field 8: FIX.4.4 (BeginString)
Field 35: A (MsgType = Logon)
Field 49: demo.icmarkets.YOUR_ACCOUNT (SenderCompID)
Field 56: cServer (TargetCompID)
Field 57: TRADE/QUOTE (TargetSubID)
Field 50: TRADE/QUOTE (SenderSubID)
Field 34: Sequence Number
Field 52: Timestamp
Field 98: 0 (EncryptMethod = None)
Field 108: 30 (HeartBtInt)
Field 553: Account Number (Username)
Field 554: FIX API Password
```

### Market Data Request Requirements

**Based on Documentation and Broker Feedback:**
- Field 146 (NoRelatedSym) is required
- Proper repeating group structure needed
- Symbol specification uses FIX Symbol ID (not standard Symbol field)
- Must follow cTrader-specific message format

---

## COMPARISON WITH CURRENT IMPLEMENTATION

### What We're Doing Correctly ✅

1. **FIX 4.4 Protocol:** Using correct protocol version
2. **Separate Sessions:** Price and trade connections separated
3. **Authentication Fields:** Using fields 553 and 554 correctly
4. **Message Structure:** Basic FIX message format is correct
5. **Heartbeat Handling:** 30-second interval implemented

### What Needs Correction ❌

1. **Host Address:** Using wrong host (demo-uk-eqx-01.p.c-trader.com vs h51.p.ctrader.com)
2. **Symbol Identification:** Using numeric IDs instead of FIX Symbol IDs
3. **Market Data Format:** Missing proper repeating group structure
4. **Symbol Discovery:** Need to use cTrader interface to get FIX Symbol IDs

---

## IMPLEMENTATION CORRECTIONS REQUIRED

### 1. Connection Configuration Update

**Current (Incorrect):**
```
Host: demo-uk-eqx-01.p.c-trader.com
Port: 5211/5212
```

**Required (Correct):**
```
Host: h51.p.ctrader.com
Port: 5201/5202
```

### 2. Symbol ID Resolution

**Current Approach:** Using SecurityListRequest to get numeric IDs
**Required Approach:** Use cTrader interface to get FIX Symbol IDs

**Process:**
1. Access cTrader platform
2. Get FIX Symbol ID for each required symbol (EURUSD, GBPUSD, etc.)
3. Use these IDs in MarketDataRequest messages
4. Implement proper repeating group structure for field 146

### 3. Market Data Request Structure

**Required Format:**
- Field 146 (NoRelatedSym): Count of symbols
- Proper repeating group with FIX Symbol IDs
- Correct message structure per cTrader requirements

---

## BROKER-SPECIFIC REQUIREMENTS

### IC Markets via cTrader Platform

**Key Differences from Standard FIX:**
1. **Symbol Identification:** Uses FIX Symbol IDs instead of standard symbols
2. **Message Format:** cTrader-specific requirements for repeating groups
3. **Connection Endpoints:** Specific hosts and ports for IC Markets
4. **Credential Format:** Specific SenderCompID format required

### Error Messages Explained

**"Tag not defined for this message type, field=55"**
- Standard Symbol field not accepted in MarketDataRequest
- Must use FIX Symbol ID in proper repeating group structure

**"Required tag missing, field=146"**
- NoRelatedSym field required for MarketDataRequest
- Must specify number of symbols in repeating group

**"Incorrect NumInGroup count for repeating group, field=146"**
- Repeating group structure incorrect
- Count in field 146 must match actual symbols provided

---

## NEXT STEPS FOR IMPLEMENTATION

### Phase 1: Connection Configuration
1. Update host to h51.p.ctrader.com
2. Update ports to 5201/5202
3. Test authentication with correct endpoints

### Phase 2: Symbol ID Resolution
1. Access cTrader platform to get FIX Symbol IDs
2. Create mapping of standard symbols to FIX Symbol IDs
3. Update symbol discovery process

### Phase 3: Market Data Request Format
1. Implement proper repeating group structure for field 146
2. Use FIX Symbol IDs instead of numeric IDs
3. Test MarketDataRequest with correct format

### Phase 4: Trading Enablement
1. Verify demo account trading permissions
2. Test order placement with correct symbol IDs
3. Validate ExecutionReport processing

---

## DOCUMENTATION INSIGHTS

### Library Recommendations

**simplefix (Current Choice):** Appropriate for learning and basic implementation
**ctrader-fix:** Official Spotware library with Twisted framework
**ejtraderCT:** High-level interface but reported connection issues
**QuickFIX:** Professional grade but complex setup

### Common Issues and Solutions

**Authentication Failures:**
- Verify FIX API password (not trading password)
- Ensure correct SenderCompID format
- Use correct host and port

**Message Format Issues:**
- Implement proper repeating group structure
- Use FIX Symbol IDs from cTrader interface
- Follow cTrader-specific message requirements

---

## CONCLUSION

**Root Cause Identified:** Using incorrect host/port and symbol identification method

**Primary Issues:**
1. Wrong connection endpoints (demo-uk-eqx-01.p.c-trader.com vs h51.p.ctrader.com)
2. Incorrect symbol identification (numeric IDs vs FIX Symbol IDs)
3. Improper MarketDataRequest structure (missing repeating groups)

**Solution Path:**
1. Update connection configuration to correct endpoints
2. Obtain FIX Symbol IDs from cTrader interface
3. Implement proper repeating group structure for MarketDataRequest
4. Test with correct symbol IDs and message format

**Expected Outcome:** Functional market data subscription and order placement with proper IC Markets FIX API integration.

