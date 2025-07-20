#!/usr/bin/env python3
"""
Phase 1 Validation Script
Validates that all stubs and mocks have been replaced with real implementations
"""

import os
import sys
import json
import sqlite3
import asyncio
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.trading.risk.real_risk_manager import RealRiskManager
from src.trading.portfolio.real_portfolio_monitor import RealPortfolioMonitor
from src.config.risk_config import RiskConfig
from src.config.portfolio_config import PortfolioConfig

class Phase1Validator:
    """Validates Phase 1 completion"""
    
    def __init__(self):
        self.validation_results = {
            'timestamp': str(datetime.now()),
            'phase1_complete': False,
            'stub_elimination': False,
            'mock_removal': False,
            'functionality': False,
            'integration': False,
            'tests_pass': False,
            'details': {}
        }
    
    async def run_validation(self) -> dict:
        """Run complete Phase 1 validation"""
        print("=== PHASE 1 VALIDATION ===")
        print("Validating Foundation Reality implementation...")
        
        # 1. Check stub elimination
        print("\n1. Checking stub elimination...")
        stub_check = await self._check_stub_elimination()
        self.validation_results['stub_elimination'] = stub_check['success']
        self.validation_results['details']['stub_elimination'] = stub_check
        
        # 2. Check mock removal
        print("\n2. Checking mock removal...")
        mock_check = await self._check_mock_removal()
        self.validation_results['mock_removal'] = mock_check['success']
        self.validation_results['details']['mock_removal'] = mock_check
        
        # 3. Check functionality
        print("\n3. Checking functionality...")
        func_check = await self._check_functionality()
        self.validation_results['functionality'] = func_check['success']
        self.validation_results['details']['functionality'] = func_check
        
        # 4. Check integration
        print("\n4. Checking integration...")
        int_check = await self._check_integration()
        self.validation_results['integration'] = int_check['success']
        self.validation_results['details']['integration'] = int_check
        
        # 5. Overall assessment
        all_checks = [
            self.validation_results['stub_elimination'],
            self.validation_results['mock_removal'],
            self.validation_results['functionality'],
            self.validation_results['integration']
        ]
        
        self.validation_results['phase1_complete'] = all(all_checks)
        
        # Save report
        self._save_report()
        
        return self.validation_results
    
    async def _check_stub_elimination(self) -> dict:
        """Check that all critical stubs have been eliminated"""
        critical_files = [
            'src/core/interfaces.py',
            'src/trading/strategies/base_strategy.py',
            'src/sensory/core/base.py',
            'src/trading/risk/risk_manager.py',
            'src/trading/portfolio/portfolio_monitor.py'
        ]
        
        stub_patterns = [
            'pass  # ← CRITICAL STUB',
            'pass  # ← STUB',
            'raise NotImplementedError',
            'NotImplementedError()',
            '# TODO: Implement',
            '# FIXME: Implement'
        ]
        
        stubs_found = []
        
        for file_path in critical_files:
            if not os.path.exists(file_path):
                continue
                
            with open(file_path, 'r') as f:
                content = f.read()
                
            for pattern in stub_patterns:
                if pattern in content:
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        if pattern in line:
                            stubs_found.append({
                                'file': file_path,
                                'line': i,
                                'pattern': pattern
                            })
        
        return {
            'success': len(stubs_found) == 0,
            'stubs_found': stubs_found,
            'total_stubs': len(stubs_found)
        }
    
    async def _check_mock_removal(self) -> dict:
        """Check that mock objects have been removed from production code"""
        mock_patterns = [
            'MockPortfolioMonitor',
            'MockRiskManager',
            'MockDataProvider',
            'MockBrokerConnector',
            'MockSensoryOrgan',
            'return 0.01  # Placeholder',
            'return {"risk_level": "low"}  # Mock'
        ]
        
        production_files = []
        for root, dirs, files in os.walk('src'):
            if 'test' not in root.lower():
                for file in files:
                    if file.endswith('.py'):
                        production_files.append(os.path.join(root, file))
        
        mocks_found = []
        
        for file_path in production_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                for pattern in mock_patterns:
                    if pattern in content:
                        lines = content.split('\n')
                        for i, line in enumerate(lines, 1):
                            if pattern in line:
                                mocks_found.append({
                                    'file': file_path,
                                    'line': i,
                                    'pattern': pattern
                                })
            except:
                continue
        
        return {
            'success': len(mocks_found) == 0,
            'mocks_found': mocks_found,
            'total_mocks': len(mocks_found)
        }
    
    async def _check_functionality(self) -> dict:
        """Check that core components are functional"""
        results = {
            'success': True,
            'components': {}
        }
        
        # Test Risk Manager
        try:
            from src.config.risk_config import RiskConfig
            config = RiskConfig()
            risk_manager = RealRiskManager(config)
            
            # Test basic functionality
            from src.trading.models import TradingSignal, SignalType
            signal = TradingSignal(
                symbol="EURUSD",
                signal_type=SignalType.BUY,
                price=1.1000,
                stop_loss=1.0950,
                take_profit=1.1100
            )
            
            position_size = risk_manager.calculate_position_size(signal, 10000, [])
            if position_size > 0:
                results['components']['risk_manager'] = 'PASS'
            else:
                results['components']['risk_manager'] = 'FAIL'
                results['success'] = False
                
        except Exception as e:
            results['components']['risk_manager'] = f'ERROR: {str(e)}'
            results['success'] = False
        
        # Test Portfolio Monitor
        try:
            from src.config.portfolio_config import PortfolioConfig
            config = PortfolioConfig()
            monitor = RealPortfolioMonitor(config)
            
            balance = monitor.get_balance()
            positions = monitor.get_positions()
            
            if balance == 10000.0 and isinstance(positions, list):
                results['components']['portfolio_monitor'] = 'PASS'
            else:
                results['components']['portfolio_monitor'] = 'FAIL'
                results['success'] = False
                
        except Exception as e:
            results['components']['portfolio_monitor'] = f'ERROR: {str(e)}'
            results['success'] = False
        
        return results
    
    async def _check_integration(self) -> dict:
        """Check basic integration between components"""
        results = {
            'success': True,
            'integration_tests': {}
        }
        
        try:
            # Test Risk Manager -> Portfolio Monitor integration
            from src.config.risk_config import RiskConfig
            from src.config.portfolio_config import PortfolioConfig
            
            risk_config = RiskConfig()
            portfolio_config = PortfolioConfig()
            
            risk_manager = RealRiskManager(risk_config)
            portfolio_monitor = RealPortfolioMonitor(portfolio_config)
            
            # Simulate basic flow
            from src.trading.models import TradingSignal, SignalType, Position
            
            signal = TradingSignal(
                symbol="EURUSD",
                signal_type=SignalType.BUY,
                price=1.1000,
                stop_loss=1.0950,
                take_profit=1.1100
            )
            
            # Calculate position size
            position_size = risk_manager.calculate_position_size(
                signal, 
                portfolio_monitor.get_balance(), 
                portfolio_monitor.get_positions()
            )
            
            # Create position
            position = Position(
                position_id="test_001",
                symbol="EURUSD",
                size=position_size,
                entry_price=1.1000,
                current_price=1.1000
            )
            
            # Add to portfolio
            success = portfolio_monitor.add_position(position)
            
            if success and position_size > 0:
                results['integration_tests']['basic_flow'] = 'PASS'
            else:
                results['integration_tests']['basic_flow'] = 'FAIL'
                results['success'] = False
                
        except Exception as e:
            results['integration_tests']['basic_flow'] = f'ERROR: {str(e)}'
            results['success'] = False
        
        return results
    
    def _save_report(self):
        """Save validation report"""
        with open('phase1_validation_report.json', 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        print("\n=== VALIDATION SUMMARY ===")
        print(f"Phase 1 Complete: {self.validation_results['phase1_complete']}")
        print(f"Stub Elimination: {self.validation_results['stub_elimination']}")
        print(f"Mock Removal: {self.validation_results['mock_removal']}")
        print(f"Functionality: {self.validation_results['functionality']}")
        print(f"Integration: {self.validation_results['integration']}")
        
        if self.validation_results['phase1_complete']:
            print("\n✅ PHASE 1 COMPLETE - Ready for Phase 2")
        else:
            print("\n❌ PHASE 1 INCOMPLETE - Please address issues above")
        
        print(f"\nDetailed report saved to: phase1_validation_report.json")

async def main():
    """Main validation function"""
    validator = Phase1Validator()
    results = await validator.run_validation()
    
    # Exit with appropriate code
    sys.exit(0 if results['phase1_complete'] else 1)

if __name__ == '__main__':
    asyncio.run(main())
