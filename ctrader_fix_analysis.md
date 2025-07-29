# cTrader FIX API Analysis - IC Markets Integration

## Key Findings from Official Documentation

### FIX Version and Protocol
- **Supported Version:** FIX 4.4 (confirmed)
- **Connection Types:** Internet, VPN tunnel, or cross-connect to UK data centers
- **Sequence Number Reset:** Required on establishing FIX session

### Authentication Requirements
- **SenderCompID Format:** `<Environment>.<BrokerUID>.<Trader Login>`
  - Environment: demo or live
  - BrokerUID: provided by cTrader
  - Trader Login: numeric identifier
- **TargetCompID:** Must be `CSERVER`
- **TargetSubID:** `QUOTE` for price data, `TRADE` for trading
- **Username (tag 553):** Numeric trader login value
- **Password (tag 554):** User password

### Connection Endpoints
- **Price Session:** TargetSubID = `QUOTE`
- **Trade Session:** TargetSubID = `TRADE`
- **Standard Header Requirements:** Specific format with mandatory fields

### Message Types Supported
**System Messages:**
- Heartbeat, Test Request, Logon, Logout, Resend Request, Reject, Sequence Reset

**Application Messages:**
- Market Data Request/Response
- New Order Single, Order Status Request, Execution Report
- Position Request/Report
- Order Cancel/Replace operations
- Security List operations

### Critical Implementation Requirements
1. **Standard Header:** Must include BeginString, BodyLength, MsgType, SenderCompID, TargetCompID, TargetSubID, MsgSeqNum, SendingTime
2. **Heartbeat Interval:** Default 30 seconds, configurable
3. **Encryption:** Currently only transport-level security (EncryptMethod=0)
4. **Sequence Numbers:** Must reset on session establishment
5. **Checksum:** Required in standard trailer



## IC Markets Specific Connection Details (From Community Forum)

### Live Server Endpoints
- **Host name:** h24.p.ctrader.com
- **IP Address:** 185.198.189.8 (can change without notice)
- **Trade Port:** 5212 (SSL), 5202 (Plain text)
- **Price Port:** 5211 (SSL), 5201 (Plain text)

### Credential Format for IC Markets
- **SenderCompID:** icmarkets.{account_number} (e.g., icmarkets.1044783)
- **TargetCompID:** cServer (note: lowercase 'c')
- **SenderSubID:** TRADE (for trading session)
- **Password:** Account password

### Known Issues from Community
1. **Session Conflicts:** When attempting to connect to Trading session, MD session may refuse to connect
2. **Connection Dependencies:** Price session must work before trade session can connect
3. **Credential Verification:** Must use exact credentials from cTrader platform
4. **Top of Book Limitations:** Level 1 quotes don't include sizes (tag 271), only prices (tag 270)

### Demo vs Live Endpoints
- **Demo:** demo-uk-eqx-01.p.c-trader.com (ports 5211/5212)
- **Live:** h24.p.ctrader.com (ports 5201/5202 plain, 5211/5212 SSL)

### Critical Implementation Notes
1. **Server Location:** IC Markets cTrader server located in LD5 IBX Equinix Data Centre, London
2. **SSL Recommended:** Use SSL ports (5211/5212) for production
3. **Session Management:** Price and Trade sessions are separate and require individual authentication
4. **Error Handling:** Connection failures often indicate credential or endpoint issues

