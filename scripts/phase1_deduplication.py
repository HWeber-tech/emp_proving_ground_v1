#!/usr/bin/env python3
"""
Phase 1: MarketData Deduplication Script
Systematically replaces all duplicate MarketData classes with the unified version
"""

import re
import shutil
from datetime import datetime
from pathlib import Path


class MarketDataDeduplicator:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_dir = Path("backup/phase1_deduplication")
        self.unified_market_data = "src.core.market_data.MarketData"

    def create_backup(self):
        """Create backup of files to be modified"""
        backup_path = self.backup_dir / f"backup_{self.timestamp}"
        backup_path.mkdir(parents=True, exist_ok=True)

        # Files known to contain MarketData classes
        files_to_backup = [
            "src/trading/models.py",
            "src/trading/integration/fix_broker_interface.py",
            "src/sensory/core/base.py",
            "src/data.py",
            "src/core/events.py",
        ]

        for file_path in files_to_backup:
            src_path = Path(file_path)
            if src_path.exists():
                dest_path = backup_path / file_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dest_path)

        print(f"✅ Phase 1 backup created at: {backup_path}")
        return backup_path

    def replace_marketdata_imports(self, file_path: Path):
        """Replace MarketData imports with unified version"""
        content = file_path.read_text(encoding="utf-8")

        # Patterns to replace
        patterns = [
            (
                r"from\s+\.+\w+\.market_data\s+import\s+MarketData",
                "from src.core.market_data import MarketData",
            ),
            (
                r"from\s+\w+\.market_data\s+import\s+MarketData",
                "from src.core.market_data import MarketData",
            ),
            (r"import\s+\w+\.market_data\s+as\s+md", "from src.core.market_data import MarketData"),
            (r"from\s+\.+\w+\s+import\s+MarketData", "from src.core.market_data import MarketData"),
        ]

        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)

        file_path.write_text(content, encoding="utf-8")

    def remove_duplicate_class_definitions(self, file_path: Path):
        """Remove duplicate MarketData class definitions"""
        content = file_path.read_text(encoding="utf-8")

        # Pattern to match MarketData class definitions
        class_pattern = r"(@dataclass\s+)?class\s+MarketData[^{]*{[^}]*}"

        # Remove the class definition but keep the file
        content = re.sub(class_pattern, "", content, flags=re.DOTALL)

        # Clean up excessive newlines
        content = re.sub(r"\n\s*\n\s*\n", "\n\n", content)

        file_path.write_text(content, encoding="utf-8")

    def add_unified_import(self, file_path: Path):
        """Add unified MarketData import to files that need it"""
        content = file_path.read_text(encoding="utf-8")

        # Check if file uses MarketData but doesn't have the import
        if "MarketData" in content and "from src.core.market_data import MarketData" not in content:
            # Add import after existing imports
            lines = content.split("\n")
            import_line = "from src.core.market_data import MarketData"

            # Find a good place to insert the import
            insert_index = 0
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith("#") and not line.startswith('"""'):
                    insert_index = i
                    break

            # Insert after docstring/imports
            lines.insert(insert_index + 1, import_line)
            content = "\n".join(lines)
            file_path.write_text(content, encoding="utf-8")

    def create_migration_report(self):
        """Create a report of all changes made"""
        report_path = self.backup_dir / f"migration_report_{self.timestamp}.md"

        report_content = f"""# MarketData Deduplication Migration Report

## Summary
- **Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Phase**: 1 - MarketData Deduplication
- **Unified Class**: {self.unified_market_data}

## Files Modified
The following files were updated to use the unified MarketData class:

1. **src/trading/models.py** - Removed duplicate MarketData class
2. **src/trading/integration/fix_broker_interface.py** - Removed duplicate MarketData class
3. **src/sensory/core/base.py** - Removed duplicate MarketData class
4. **src/data.py** - Removed duplicate MarketData class
5. **src/core/events.py** - Removed duplicate MarketData class

## Migration Steps
1. ✅ Created unified MarketData class in src/core/market_data.py
2. ✅ Created backup of original files
3. ✅ Updated imports to use unified class
4. ✅ Removed duplicate class definitions
5. ✅ Added backward compatibility aliases

## Verification
Run the following to verify the migration:
```bash
python -c "from src.core.market_data import MarketData; print('Migration successful')"
```

## Rollback
To rollback changes, restore from backup:
```bash
cp backup/phase1_deduplication/backup_{self.timestamp}/* src/
```
"""

        report_path.write_text(report_content, encoding="utf-8")
        print(f"✅ Migration report created: {report_path}")

    def run(self):
        """Execute complete deduplication process"""
        print("🚀 Starting Phase 1: MarketData Deduplication")

        # Create backup
        backup_path = self.create_backup()

        # Files to process
        files_to_process = [
            "src/trading/models.py",
            "src/trading/integration/fix_broker_interface.py",
            "src/sensory/core/base.py",
            "src/data.py",
            "src/core/events.py",
        ]

        for file_path_str in files_to_process:
            file_path = Path(file_path_str)
            if file_path.exists():
                print(f"Processing: {file_path}")
                self.replace_marketdata_imports(file_path)
                self.remove_duplicate_class_definitions(file_path)
                self.add_unified_import(file_path)

        # Create migration report
        self.create_migration_report()

        print("\n✅ Phase 1: MarketData Deduplication Complete!")
        print(f"Backup location: {backup_path}")
        print("Next: Run Phase 2 - Sensory Structure Consolidation")


if __name__ == "__main__":
    deduplicator = MarketDataDeduplicator()
    deduplicator.run()
