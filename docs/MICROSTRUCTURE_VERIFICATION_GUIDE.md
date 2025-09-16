# IC Markets Level 2 Depth Verification Guide

> **Status:** Legacy – maintained for context about the retired cTrader OpenAPI
> tooling. Active builds must adhere to the [Integration Policy](policies/integration_policy.md)
> and use FIX-based workflows instead.

## Overview
This guide provides step-by-step instructions for running the IC Markets cTrader Level 2 depth verification tool to assess data quality for microstructure strategies.

## Quick Start

### 1. Environment Setup
```bash
# Ensure dependencies are installed
pip install ctrader-open-api-py pandas python-dotenv

# Copy environment template
cp .env.example .env
```

### 2. Configure Credentials
Edit `.env` file with your IC Markets demo account credentials:
```bash
CTRADER_CLIENT_ID=your_client_id_here
CTRADER_CLIENT_SECRET=your_client_secret_here
CTRADER_ACCESS_TOKEN=your_access_token_here
CTRADER_ACCOUNT_ID=your_account_id_here
CTRADER_DEMO_HOST=demo.ctraderapi.com
CTRADER_PORT=5035
```

### 3. Run Verification
```bash
# Quick 1-minute test
python scripts/verify_microstructure_working.py --duration 1 --symbol EURUSD

# Comprehensive 30-minute test
python scripts/verify_microstructure_working.py --duration 30 --symbol EURUSD

# Test different symbol
python scripts/verify_microstructure_working.py --duration 5 --symbol GBPUSD
```

## Understanding the Output

### Files Generated
- `docs/v4_reality_check_report.md` - Main verification report
- `docs/microstructure_raw_data.json` - Raw collected data

### Report Sections
1. **Executive Summary** - Clear go/no-go recommendation
2. **Quantitative Findings** - Latency and depth metrics
3. **Qualitative Observations** - Data quality notes
4. **Strategic Recommendation** - Decision for MICRO-40 ticket

### Key Metrics
- **Latency**: <100ms = excellent, 100-500ms = acceptable, >500ms = poor
- **Depth**: >5 levels = excellent, 3-5 levels = limited, ≤2 levels = insufficient
- **Frequency**: Updates per second during active trading

## Troubleshooting

### Common Issues

**"Missing configuration" error**
- Check all required variables in `.env` file
- Ensure CTRADER_ACCOUNT_ID is a number, not string

**"Connection failed"**
- Verify internet connectivity
- Check if demo.ctraderapi.com:5035 is accessible
- Ensure credentials are for demo account

**"Symbol not found"**
- Verify symbol name (e.g., "EURUSD" not "EUR/USD")
- Check if symbol is available on your demo account

### Getting API Credentials

1. **Create Application**:
   - Login to https://connect.icmarkets.com/
   - Navigate to Open API section
   - Create new application with redirect URI: http://localhost/

2. **Get Tokens**:
   - Use OAuth flow to get access token
   - Save refresh token for long-term use

3. **Find Account ID**:
   - Use API call to list trading accounts
   - Use demo account ID for testing

## Best Practices

### Timing
- Run during active market hours (London/New York overlap)
- Avoid weekends and major holidays
- Consider different market sessions for comparison

### Duration
- **1-5 minutes**: Quick feasibility check
- **15-30 minutes**: Comprehensive analysis
- **Multiple runs**: Validate consistency

### Network
- Use wired connection when possible
- Close other bandwidth-intensive applications
- Consider geographic proximity to servers

## Decision Matrix

| Depth Levels | Latency | Recommendation | Action |
|--------------|---------|----------------|---------|
| >5 levels | <100ms | **GO** | Proceed with microstructure engine |
| 3-5 levels | <200ms | **GO (modified)** | Implement virtual depth model |
| ≤2 levels | Any | **NO-GO** | Focus on other features |
| Any | >500ms | **NO-GO** | Investigate latency issues |

## Next Steps After Verification

1. **Review report** in `docs/v4_reality_check_report.md`
2. **Make go/no-go decision** based on findings
3. **Update Sprint 2 planning** accordingly
4. **Proceed with implementation** if GO decision

## Support

For technical issues:
1. Check logs for specific error messages
2. Verify credentials and network connectivity
3. Review IC Markets API documentation
4. Test with minimal configuration first
