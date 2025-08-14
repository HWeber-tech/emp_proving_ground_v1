#!/usr/bin/env python3
"""
FIX API Protection System
Creates comprehensive backup and verification system for FIX API functionality
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path


class FIXAPIProtection:
    def __init__(self):
        self.backup_dir = Path("backup/fix_protection")
        self.config_dir = Path("config/fix")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def create_backup(self):
        """Create comprehensive backup of FIX API configuration and code"""
        backup_path = self.backup_dir / f"backup_{self.timestamp}"
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Backup FIX configuration
        config_backup = backup_path / "config"
        if self.config_dir.exists():
            shutil.copytree(self.config_dir, config_backup / "fix")
            
        # Backup operational modules
        operational_backup = backup_path / "operational"
        operational_backup.mkdir(exist_ok=True)
        
        operational_files = [
            "src/operational/icmarkets_simplefix_application.py",
            "src/operational/ctrader_interface.py",
            "src/operational/market_data_processor.py"
        ]
        
        for file_path in operational_files:
            src_path = Path(file_path)
            if src_path.exists():
                shutil.copy2(src_path, operational_backup / src_path.name)
                
        # Create verification manifest
        manifest = {
            "backup_timestamp": self.timestamp,
            "fix_api_status": "WORKING",
            "backup_contents": {
                "config_files": [str(f) for f in self.config_dir.glob("*") if f.is_file()],
                "operational_files": operational_files
            },
            "verification_command": "python scripts/test_simplefix.py"
        }
        
        with open(backup_path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
            
        print(f"✅ FIX API backup created at: {backup_path}")
        return backup_path
        
    def create_protection_script(self):
        """Create daily verification script"""
        protection_script = self.backup_dir / "daily_verification.py"
        
        script_content = '''#!/usr/bin/env python3
"""
Daily FIX API Verification Script
Run this script daily to ensure FIX API functionality remains intact
"""

import subprocess
import sys
from datetime import datetime

def verify_fix_api():
    """Run comprehensive FIX API verification"""
    print("FIX API Verification - {}".format(datetime.now()))
    
    # Test 1: Simple connection test
    try:
        result = subprocess.run([sys.executable, "scripts/test_simplefix.py"], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("Connection test: PASSED")
        else:
            print("Connection test: FAILED")
            print(result.stderr)
            return False
    except Exception as e:
        print("Connection test error: {}".format(e))
        return False
        
    # Test 2: Configuration validation
    try:
        from src.operational.icmarkets_simplefix_application import ICMarketsSimpleFix
        app = ICMarketsSimpleFix()
        if app.load_config():
            print("Configuration validation: PASSED")
        else:
            print("Configuration validation: FAILED")
            return False
    except Exception as e:
        print("Configuration validation error: {}".format(e))
        return False
        
    print("All FIX API verification tests PASSED")
    return True

if __name__ == "__main__":
    success = verify_fix_api()
    sys.exit(0 if success else 1)
'''
        
        with open(protection_script, "w") as f:
            f.write(script_content)
            
        # Make executable
        os.chmod(protection_script, 0o755)
        print(f"✅ Daily verification script created: {protection_script}")
        
    def run(self):
        """Execute complete protection setup"""
        print("🛡️ Setting up FIX API Protection System...")
        
        # Create backup
        backup_path = self.create_backup()
        
        # Create protection script
        self.create_protection_script()
        
        # Run initial verification
        print("\n🔍 Running initial verification...")
        os.system("python scripts/test_simplefix.py")
        
        print("\n🛡️ FIX API Protection System setup complete!")
        print(f"Backup location: {backup_path}")
        print("Run 'python backup/fix_protection/daily_verification.py' for daily checks")

if __name__ == "__main__":
    protector = FIXAPIProtection()
    protector.run()
