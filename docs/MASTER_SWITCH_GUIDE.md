# EMP v4.0 Master Switch Guide (FIX-Only)

## Overview
The Master Switch is the final integration component of Sprint 1: The Professional Upgrade. It provides a single, authoritative configuration setting that controls the entire operational mode of the EMP v4.0 Professional Predator.

## Architecture
- **Protocol Selection**: Single configuration field controls all components
- **Protocol-Agnostic Core**: Upper layers remain unchanged regardless of protocol
- **Clean Integration**: Protocol-specific setup encapsulated in dedicated methods
- **Policy**: OpenAPI is disabled. The system operates with FIX exclusively.

## Configuration

### Master Switch Setting
```bash
# In .env file or environment variable
CONNECTION_PROTOCOL=fix        # Professional FIX protocol (required)
```

### Configuration Files
- **Default**: `.env` (uses FIX protocol by default)
- **FIX Test**: `config/test_fix.env`
  (OpenAPI test file removed; FIX-only build)

## Usage

### Quick Start
```bash
# 1. Run with FIX protocol (default)
python main.py

# 2. Use configuration file
cp config/test_fix.env .env
python main.py
```

### Protocol-Specific Setup

#### FIX Protocol Mode
When `CONNECTION_PROTOCOL=fix`:
- ✅ Uses `FIXConnectionManager` for session management
- ✅ Instantiates `FIXSensoryOrgan` for market data
- ✅ Uses `FIXBrokerInterface` for trade execution
- ✅ Connects to IC Markets cTrader FIX gateways
- ✅ Professional-grade SSL-encrypted connections

#### OpenAPI Protocol Mode
Disabled. Set `CONNECTION_PROTOCOL=fix` and use IC Markets FIX per `docs/fix_api/`.

## Testing

### Automated Testing
```bash
# Run comprehensive master switch tests
python scripts/test_master_switch.py
```

### Manual Testing
```bash
# Test FIX mode
CONNECTION_PROTOCOL=fix python main.py
```

### Expected Output
```
🚀 Initializing EMP v4.0 Professional Predator
✅ Configuration loaded: EMP Professional Predator
🔧 Protocol: fix
✅ Event bus started
🔧 Setting up LIVE components using 'fix' protocol
🎯 Configuring FIX protocol components
✅ FIX components configured successfully
🎉 Professional Predator initialization complete
```

## File Structure
```
main.py                          # Main application with master switch
src/governance/system_config.py  # Configuration with CONNECTION_PROTOCOL
config/
└── test_fix.env                # FIX protocol test configuration
docs/
└── MASTER_SWITCH_GUIDE.md      # This documentation
```

## Integration Points

### Sensory Layer
- **FIXSensoryOrgan**: Professional market data via FIX

### Trading Layer
- **FIXBrokerInterface**: Professional trade execution via FIX

### Configuration Layer
- **SystemConfig**: Centralized configuration management
- **Environment Variables**: Runtime protocol selection

## Migration Guide

### From Legacy to Professional
1. **Backup**: Save current configuration
2. **Update**: Set `CONNECTION_PROTOCOL=fix` in .env
3. **Configure**: Add FIX credentials to .env
4. **Test**: Run `python scripts/test_master_switch.py`
5. **Deploy**: Use new main.py with master switch

### Rollback Procedure
Not available in FIX-only build. OpenAPI is disabled.

## Troubleshooting

### Common Issues

1. **Protocol Not Found**
   ```
   ValueError: Unsupported connection protocol: invalid
   ```
   **Solution**: Use only "fix" or "openapi" values

2. **Missing Components**
   ```
   ModuleNotFoundError: No module named 'src.sensory.fix_sensory_organ'
   ```
   **Solution**: Ensure FIX components are properly implemented

3. **Configuration Errors**
   ```
   KeyError: CONNECTION_PROTOCOL
   ```
   **Solution**: Update SystemConfig to include CONNECTION_PROTOCOL field

### Debug Mode
```bash
# Enable debug logging
LOG_LEVEL=DEBUG python main.py
```

## Next Steps

After successful master switch integration:
1. **Add FIX credentials** to .env file
2. **Test connections** with real credentials
3. **Monitor performance** of both protocols
4. FIX-only operations
5. **Production deployment** with FIX as default

## Support
For issues with the master switch:
- Check configuration validation with test script
- Verify protocol-specific components exist
- Review logs for connection issues
- Test both modes independently

The master switch is now complete and ready for production use.
