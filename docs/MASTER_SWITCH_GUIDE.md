# EMP v4.0 Master Switch Guide

## Overview
The Master Switch is the final integration component of Sprint 1: The Professional Upgrade. It provides a single, authoritative configuration setting that controls the entire operational mode of the EMP v4.0 Professional Predator.

## Architecture
- **Protocol Selection**: Single configuration field controls all components
- **Protocol-Agnostic Core**: Upper layers remain unchanged regardless of protocol
- **Clean Integration**: Protocol-specific setup encapsulated in dedicated methods
- **Backward Compatibility**: OpenAPI remains fully functional as fallback

## Configuration

### Master Switch Setting
```bash
# In .env file or environment variable
CONNECTION_PROTOCOL=fix        # Professional FIX protocol (default)
CONNECTION_PROTOCOL=openapi    # Legacy OpenAPI protocol (fallback)
```

### Configuration Files
- **Default**: `.env` (uses FIX protocol by default)
- **FIX Test**: `config/test_fix.env`
- **OpenAPI Test**: `config/test_openapi.env`

## Usage

### Quick Start
```bash
# 1. Test the master switch
python scripts/test_master_switch.py

# 2. Run with FIX protocol (default)
python main.py

# 3. Run with OpenAPI protocol
CONNECTION_PROTOCOL=openapi python main.py

# 4. Use configuration file
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
When `CONNECTION_PROTOCOL=openapi`:
- ✅ Uses `CTraderDataOrgan` for market data
- ✅ Uses `CTraderBrokerInterface` for trade execution
- ✅ Connects via cTrader OpenAPI
- ✅ Full backward compatibility maintained

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

# Test OpenAPI mode
CONNECTION_PROTOCOL=openapi python main.py
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
├── test_fix.env                # FIX protocol test configuration
├── test_openapi.env            # OpenAPI protocol test configuration
scripts/
├── test_master_switch.py       # Comprehensive testing script
docs/
└── MASTER_SWITCH_GUIDE.md      # This documentation
```

## Integration Points

### Sensory Layer
- **FIXSensoryOrgan**: Professional market data via FIX
- **CTraderDataOrgan**: Legacy market data via OpenAPI

### Trading Layer
- **FIXBrokerInterface**: Professional trade execution via FIX
- **CTraderBrokerInterface**: Legacy trade execution via OpenAPI

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
1. **Switch**: Set `CONNECTION_PROTOCOL=openapi`
2. **Verify**: Test with existing OpenAPI credentials
3. **Deploy**: Use same main.py with different configuration

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
4. **Gradual migration** from OpenAPI to FIX
5. **Production deployment** with FIX as default

## Support
For issues with the master switch:
- Check configuration validation with test script
- Verify protocol-specific components exist
- Review logs for connection issues
- Test both modes independently

The master switch is now complete and ready for production use.
