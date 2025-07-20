#!/usr/bin/env python3
"""
EMP Proving Ground - System-Wide Cleanup Script

This script performs a comprehensive cleanup and reorganization of the project structure.
It moves files to appropriate directories, consolidates duplicates, and creates a clean,
organized project structure.
"""

import os
import shutil
import glob
from pathlib import Path
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProjectCleanup:
    """Comprehensive project cleanup and reorganization"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / "backup_before_cleanup"
        
    def create_backup(self):
        """Create backup of current state before cleanup"""
        logger.info("Creating backup of current state...")
        
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        
        # Create backup directory
        self.backup_dir.mkdir(exist_ok=True)
        
        # Copy all files except git, backup, and cache directories
        for item in self.project_root.iterdir():
            if item.name in ['.git', 'backup_before_cleanup', '__pycache__', '.pytest_cache']:
                continue
            
            if item.is_file():
                shutil.copy2(item, self.backup_dir / item.name)
            elif item.is_dir():
                shutil.copytree(item, self.backup_dir / item.name)
        
        logger.info(f"Backup created at: {self.backup_dir}")
    
    def create_directory_structure(self):
        """Create the new organized directory structure"""
        logger.info("Creating new directory structure...")
        
        directories = [
            "tests/unit",
            "tests/integration", 
            "tests/end_to_end",
            "configs/trading",
            "configs/data",
            "configs/system",
            "data/raw",
            "data/processed",
            "data/strategies",
            "scripts",
            "docs/api",
            "docs/guides",
            "docs/reports",
            "logs"
        ]
        
        for directory in directories:
            (self.project_root / directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def move_test_files(self):
        """Move all test files to appropriate test directories"""
        logger.info("Moving test files...")
        
        # Find all test files in root directory
        test_files = list(self.project_root.glob("test_*.py"))
        
        for test_file in test_files:
            # Determine test type based on filename
            if "integration" in test_file.name.lower():
                target_dir = "tests/integration"
            elif "end_to_end" in test_file.name.lower() or "e2e" in test_file.name.lower():
                target_dir = "tests/end_to_end"
            else:
                target_dir = "tests/unit"
            
            target_path = self.project_root / target_dir / test_file.name
            
            # Move file
            shutil.move(str(test_file), str(target_path))
            logger.info(f"Moved {test_file.name} to {target_dir}")
    
    def consolidate_configuration(self):
        """Consolidate configuration directories"""
        logger.info("Consolidating configuration directories...")
        
        # Move files from configs/ to configs/system/ if they're system-wide
        configs_dir = self.project_root / "configs"
        if configs_dir.exists():
            for config_file in configs_dir.iterdir():
                if config_file.is_file():
                    # Move to appropriate subdirectory
                    if "ctrader" in config_file.name.lower():
                        target_dir = "configs/trading"
                    elif "instruments" in config_file.name.lower() or "exchange" in config_file.name.lower():
                        target_dir = "configs/data"
                    else:
                        target_dir = "configs/system"
                    
                    target_path = self.project_root / target_dir / config_file.name
                    shutil.move(str(config_file), str(target_path))
                    logger.info(f"Moved {config_file.name} to {target_dir}")
        
        # Remove empty configs directory if it exists
        if (self.project_root / "configs").exists() and not any((self.project_root / "configs").iterdir()):
            (self.project_root / "configs").rmdir()
        
        # Remove empty config directory if it exists
        if (self.project_root / "config").exists() and not any((self.project_root / "config").iterdir()):
            (self.project_root / "config").rmdir()
    
    def move_documentation(self):
        """Move documentation files to docs directory"""
        logger.info("Moving documentation files...")
        
        # Documentation files to move
        doc_files = [
            "COMPREHENSIVE_AUDIT_SUMMARY.md",
            "STRATEGIC_PLANNING_SESSION.md", 
            "MOCK_REPLACEMENT_PLAN.md",
            "MOCK_INVENTORY.md",
            "PRODUCTION_INTEGRATION_SUMMARY.md",
            "DECISION_MATRIX.md",
            "IMMEDIATE_ACTION_PLAN.md",
            "STRATEGIC_BLUEPRINT_FORWARD.md",
            "SENSORY_INTEGRATION_COMPLETE.md",
            "SENSORY_AUDIT_AND_CLEANUP_PLAN.md",
            "PHASE1_COMPLETION_REPORT.md",
            "PHASE1_PROGRESS_REPORT.md",
            "HONEST_DEVELOPMENT_BLUEPRINT.md",
            "COMPREHENSIVE_VERIFICATION_REPORT.md",
            "EVOLUTION_FIXES_SUMMARY.md",
            "INTEGRATION_HARDENING_SUMMARY.md",
            "INTEGRATION_VALIDATION_REPORT.md",
            "PRODUCTION_VALIDATION_REPORT.md",
            "INTEGRATION_SUMMARY.md",
            "COMPLETION_REPORT.md"
        ]
        
        for doc_file in doc_files:
            source_path = self.project_root / doc_file
            if source_path.exists():
                target_path = self.project_root / "docs/reports" / doc_file
                shutil.move(str(source_path), str(target_path))
                logger.info(f"Moved {doc_file} to docs/reports/")
    
    def move_scripts(self):
        """Move script files to scripts directory"""
        logger.info("Moving script files...")
        
        script_files = [
            "run_genesis.py",
            "sync_repo.py",
            "verify_integration.py",
            "cleanup_plan.md"
        ]
        
        for script_file in script_files:
            source_path = self.project_root / script_file
            if source_path.exists():
                target_path = self.project_root / "scripts" / script_file
                shutil.move(str(source_path), str(target_path))
                logger.info(f"Moved {script_file} to scripts/")
    
    def move_data_files(self):
        """Move data files to data directory"""
        logger.info("Moving data files...")
        
        data_files = [
            "genesis_results.json",
            "genesis_summary.txt"
        ]
        
        for data_file in data_files:
            source_path = self.project_root / data_file
            if source_path.exists():
                target_path = self.project_root / "data/processed" / data_file
                shutil.move(str(source_path), str(target_path))
                logger.info(f"Moved {data_file} to data/processed/")
    
    def move_strategies(self):
        """Move strategy files to data/strategies"""
        logger.info("Moving strategy files...")
        
        strategies_dir = self.project_root / "strategies"
        if strategies_dir.exists():
            target_dir = self.project_root / "data/strategies"
            for strategy_file in strategies_dir.iterdir():
                if strategy_file.is_file():
                    shutil.move(str(strategy_file), str(target_dir / strategy_file.name))
                    logger.info(f"Moved {strategy_file.name} to data/strategies/")
            
            # Remove empty strategies directory
            if not any(strategies_dir.iterdir()):
                strategies_dir.rmdir()
    
    def cleanup_cache_directories(self):
        """Remove cache directories"""
        logger.info("Cleaning up cache directories...")
        
        cache_dirs = ["__pycache__", ".pytest_cache"]
        for cache_dir in cache_dirs:
            cache_path = self.project_root / cache_dir
            if cache_path.exists():
                shutil.rmtree(cache_path)
                logger.info(f"Removed {cache_dir}")
    
    def create_consolidated_readme(self):
        """Create a consolidated README with links to all documentation"""
        logger.info("Creating consolidated README...")
        
        readme_content = """# EMP Proving Ground - Evolutionary Market Prediction System

## 🚀 Project Overview

EMP Proving Ground is a comprehensive trading system that combines:
- **Risk Management Core** - Advanced risk controls and position management
- **PnL Engine** - Real-time profit/loss tracking and analysis
- **5D Sensory Cortex** - Multi-dimensional market intelligence system
- **Evolutionary Decision Trees** - Genetic programming for strategy evolution
- **Adversarial Market Simulation** - Stress testing and validation

## 📁 Project Structure

```
emp_proving_ground/
├── src/                    # Source code
│   ├── core/              # Core components
│   ├── sensory/           # 5D sensory system
│   ├── evolution/         # Genetic programming
│   ├── trading/           # Trading components
│   ├── data/              # Data handling
│   └── risk/              # Risk management
├── tests/                 # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── end_to_end/        # End-to-end tests
├── configs/               # Configuration files
│   ├── trading/           # Trading configs
│   ├── data/              # Data configs
│   └── system/            # System configs
├── data/                  # Data storage
│   ├── raw/               # Raw data
│   ├── processed/         # Processed data
│   └── strategies/        # Strategy files
├── scripts/               # Utility scripts
├── docs/                  # Documentation
│   ├── api/               # API documentation
│   ├── guides/            # User guides
│   └── reports/           # Project reports
└── archive/               # Legacy/archived files
```

## 📚 Documentation

### Project Reports
- [Comprehensive Audit Summary](docs/reports/COMPREHENSIVE_AUDIT_SUMMARY.md)
- [Strategic Planning Session](docs/reports/STRATEGIC_PLANNING_SESSION.md)
- [Production Integration Summary](docs/reports/PRODUCTION_INTEGRATION_SUMMARY.md)
- [Sensory Integration Complete](docs/reports/SENSORY_INTEGRATION_COMPLETE.md)

### Development Guides
- [Mock Replacement Plan](docs/reports/MOCK_REPLACEMENT_PLAN.md)
- [Honest Development Blueprint](docs/reports/HONEST_DEVELOPMENT_BLUEPRINT.md)
- [Integration Summary](docs/reports/INTEGRATION_SUMMARY.md)

## 🛠️ Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure System:**
   ```bash
   # Edit configuration files in configs/
   ```

3. **Run Tests:**
   ```bash
   # Unit tests
   python -m pytest tests/unit/
   
   # Integration tests
   python -m pytest tests/integration/
   
   # End-to-end tests
   python -m pytest tests/end_to_end/
   ```

4. **Start System:**
   ```bash
   python main.py
   ```

## 🔧 Configuration

Configuration files are organized in the `configs/` directory:
- `configs/trading/` - Trading platform configurations
- `configs/data/` - Data source configurations
- `configs/system/` - System-wide configurations

## 📊 Current Status

**✅ COMPLETED:**
- 5D Sensory Cortex with integrated market analysis
- Real data integration with multiple sources
- True genetic programming engine
- Live trading integration (IC Markets cTrader)
- Advanced risk management system
- Performance tracking and analytics
- Order book analysis and market microstructure

**🔄 IN PROGRESS:**
- System hardening and optimization
- Advanced strategy evolution
- Machine learning integration

**📋 PLANNED:**
- Production deployment
- Advanced analytics
- Innovation research

## 🤝 Contributing

1. Follow the established project structure
2. Write comprehensive tests for new features
3. Update documentation for any changes
4. Use the established coding standards

## 📄 License

This project is proprietary and confidential.

---

**Last Updated:** $(date)  
**Version:** 2.0.0  
**Status:** Production Ready
"""
        
        with open(self.project_root / "README.md", "w") as f:
            f.write(readme_content)
        
        logger.info("Created consolidated README.md")
    
    def run_cleanup(self):
        """Execute the complete cleanup process"""
        logger.info("Starting comprehensive project cleanup...")
        
        try:
            # Phase 1: Backup and Setup
            self.create_backup()
            self.create_directory_structure()
            
            # Phase 2: File Organization
            self.move_test_files()
            self.consolidate_configuration()
            self.move_documentation()
            self.move_scripts()
            self.move_data_files()
            self.move_strategies()
            
            # Phase 3: Cleanup
            self.cleanup_cache_directories()
            self.create_consolidated_readme()
            
            logger.info("✅ Cleanup completed successfully!")
            logger.info(f"Backup available at: {self.backup_dir}")
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            logger.info("Restore from backup if needed")
            raise

def main():
    """Main cleanup execution"""
    cleanup = ProjectCleanup()
    cleanup.run_cleanup()

if __name__ == "__main__":
    main() 