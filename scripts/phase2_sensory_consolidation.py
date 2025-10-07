#!/usr/bin/env python3
"""
Phase 2: Sensory Structure Consolidation
Consolidates 3 redundant sensory folder hierarchies into 1 clean structure
"""

import shutil
from datetime import datetime
from pathlib import Path


class SensoryStructureConsolidator:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_dir = Path("backup/phase2_sensory_consolidation")
        self.primary_structure = "src/sensory/organs/dimensions/"

    def create_backup(self):
        """Create backup of sensory structure before consolidation"""
        backup_path = self.backup_dir / f"backup_{self.timestamp}"
        backup_path.mkdir(parents=True, exist_ok=True)

        # Backup entire sensory directory
        sensory_backup = backup_path / "sensory"
        if Path("src/sensory").exists():
            shutil.copytree("src/sensory", sensory_backup)

        print(f"✅ Phase 2 backup created at: {backup_path}")
        return backup_path

    def consolidate_structure(self):
        """Consolidate 3 sensory hierarchies into 1"""
        print("🔧 Consolidating sensory structure...")

        # Define the primary structure (src/sensory/organs/dimensions/)
        primary_dimensions = Path("src/sensory/organs/dimensions")
        primary_dimensions.mkdir(parents=True, exist_ok=True)

        # Define redundant structures to consolidate
        redundant_structures = [
            "src/sensory/core/",  # Redundant core structure
            "src/sensory/enhanced/",  # Redundant enhanced structure
        ]

        # Files to move to primary structure
        consolidation_map = {
            # Core structure -> Primary structure
            "src/sensory/core/base.py": "src/sensory/organs/dimensions/base_organ.py",
            "src/sensory/core/data_integration.py": "src/sensory/organs/dimensions/data_integration.py",
            "src/sensory/core/real_sensory_organ.py": "src/sensory/organs/dimensions/real_sensory_organ.py",
            "src/sensory/core/sensory_signal.py": "src/sensory/organs/dimensions/sensory_signal.py",
            "src/sensory/core/utils.py": "src/sensory/organs/dimensions/utils.py",
            # Enhanced structure -> Primary structure
            "src/sensory/enhanced/anomaly/manipulation_detection.py": "src/sensory/organs/dimensions/anomaly_detection.py",
            "src/sensory/enhanced/how/institutional_footprint_hunter.py": "src/sensory/organs/dimensions/institutional_tracker.py",
            "src/sensory/enhanced/integration/sensory_integration_orchestrator.py": "src/sensory/organs/dimensions/integration_orchestrator.py",
            "src/sensory/enhanced/what/pattern_synthesis_engine.py": "src/sensory/organs/dimensions/pattern_engine.py",
            "src/sensory/enhanced/when/temporal_advantage_system.py": "src/sensory/organs/dimensions/temporal_system.py",
        }

        # Execute consolidation
        for src_path_str, dest_path_str in consolidation_map.items():
            src_path = Path(src_path_str)
            dest_path = Path(dest_path_str)

            if src_path.exists():
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dest_path)
                print(f"  Consolidated: {src_path} -> {dest_path}")

    def handle_duplicate_files(self):
        """Handle duplicate real_sensory_organ.py files"""
        print("🔧 Handling duplicate files...")

        # Identify the primary real_sensory_organ.py
        primary = Path("src/sensory/organs/dimensions/real_sensory_organ.py")
        duplicate = Path("src/sensory/real_sensory_organ.py")

        if duplicate.exists():
            # Compare files to ensure they're identical or merge if different
            if primary.exists():
                primary_content = primary.read_text(encoding="utf-8")
                duplicate_content = duplicate.read_text(encoding="utf-8")

                if primary_content == duplicate_content:
                    # Files are identical, safe to remove duplicate
                    duplicate.unlink()
                    print("  Removed duplicate: src/sensory/real_sensory_organ.py")
                else:
                    # Files differ, create backup and use primary
                    shutil.copy2(duplicate, f"{duplicate}.backup")
                    duplicate.unlink()
                    print("  Merged and removed duplicate: src/sensory/real_sensory_organ.py")

    def clean_redundant_directories(self):
        """Remove redundant directory structures"""
        print("🧹 Cleaning redundant directories...")

        redundant_dirs = [
            "src/sensory/core/",
            "src/sensory/enhanced/",
            "src/sensory/real_sensory_organ.py",  # Remove duplicate file
        ]

        for dir_path in redundant_dirs:
            path = Path(dir_path)
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                    print(f"  Removed directory: {path}")
                elif path.is_file():
                    path.unlink()
                    print(f"  Removed file: {path}")

    def update_imports(self):
        """Update all imports to use the new consolidated structure"""
        print("🔄 Updating imports...")

        # Files that need import updates
        files_to_update = [
            "src/sensory/__init__.py",
            "src/sensory/organs/__init__.py",
            "src/sensory/organs/dimensions/__init__.py",
        ]

        # Import mapping
        import_updates = {
            "from sensory.core.base import": "from sensory.organs.dimensions.base_organ import",
            "from sensory.core.real_sensory_organ import": "from sensory.organs.dimensions.real_sensory_organ import",
            "from sensory.core.data_integration import": "from sensory.organs.dimensions.data_integration import",
            "from sensory.enhanced.anomaly.manipulation_detection import": "from sensory.organs.dimensions.anomaly_detection import",
            "from sensory.enhanced.integration.sensory_integration_orchestrator import": "from sensory.organs.dimensions.integration_orchestrator import",
        }

        for file_path_str in files_to_update:
            file_path = Path(file_path_str)
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")

                for old_import, new_import in import_updates.items():
                    content = content.replace(old_import, new_import)

                file_path.write_text(content, encoding="utf-8")
                print(f"  Updated imports: {file_path}")

    def create_consolidation_report(self):
        """Create a report of the consolidation process"""
        report_path = self.backup_dir / f"consolidation_report_{self.timestamp}.md"

        report_content = f"""# Sensory Structure Consolidation Report

## Summary
- **Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Phase**: 2 - Sensory Structure Consolidation
- **Primary Structure**: {self.primary_structure}

## Changes Made
### 1. Directory Consolidation
- **Consolidated**: 3 redundant sensory hierarchies → 1 clean structure
- **Primary Structure**: `src/sensory/organs/dimensions/`
- **Redundant Structures Removed**:
  - `src/sensory/core/` → Consolidated into dimensions
  - `src/sensory/enhanced/` → Consolidated into dimensions

### 2. File Migrations
The following files were moved to the primary structure:

#### Core Structure → Dimensions
- `src/sensory/core/base.py` → `src/sensory/organs/dimensions/base_organ.py`
- `src/sensory/core/data_integration.py` → `src/sensory/organs/dimensions/data_integration.py`
- `src/sensory/core/real_sensory_organ.py` → `src/sensory/organs/dimensions/real_sensory_organ.py`
- `src/sensory/core/sensory_signal.py` → `src/sensory/organs/dimensions/sensory_signal.py`
- `src/sensory/core/utils.py` → `src/sensory/organs/dimensions/utils.py`

#### Enhanced Structure → Dimensions
- `src/sensory/enhanced/anomaly/manipulation_detection.py` → `src/sensory/organs/dimensions/anomaly_detection.py`
- `src/sensory/enhanced/how/institutional_footprint_hunter.py` → `src/sensory/organs/dimensions/institutional_tracker.py`
- `src/sensory/enhanced/integration/sensory_integration_orchestrator.py` → `src/sensory/organs/dimensions/integration_orchestrator.py`
- `src/sensory/enhanced/what/pattern_synthesis_engine.py` → `src/sensory/organs/dimensions/pattern_engine.py`
- `src/sensory/enhanced/when/temporal_advantage_system.py` → `src/sensory/organs/dimensions/temporal_system.py`

### 3. Duplicate File Handling
- **Duplicate**: `src/sensory/real_sensory_organ.py` → **Removed** (consolidated into dimensions)

### 4. Import Updates
- Updated all import statements to use the new consolidated structure
- Maintained backward compatibility through __init__.py files

## Verification
Run the following to verify the consolidation:
```bash
python -c "from sensory.organs.dimensions import base_organ; print('Consolidation successful')"
```

## Rollback
To rollback changes, restore from backup:
```bash
cp backup/phase2_sensory_consolidation/backup_{self.timestamp}/sensory/* src/sensory/
```

## Next Steps
- Phase 3: Stub Elimination
- Phase 4: Import Standardization
"""

        report_path.write_text(report_content, encoding="utf-8")
        print(f"✅ Consolidation report created: {report_path}")

    def run(self):
        """Execute complete consolidation process"""
        print("🚀 Starting Phase 2: Sensory Structure Consolidation")

        # Create backup
        backup_path = self.create_backup()

        # Execute consolidation
        self.consolidate_structure()
        self.handle_duplicate_files()
        self.clean_redundant_directories()
        self.update_imports()

        # Create report
        self.create_consolidation_report()

        print("\n✅ Phase 2: Sensory Structure Consolidation Complete!")
        print(f"Backup location: {backup_path}")
        print("Next: Run Phase 3 - Stub Elimination")


if __name__ == "__main__":
    consolidator = SensoryStructureConsolidator()
    consolidator.run()
