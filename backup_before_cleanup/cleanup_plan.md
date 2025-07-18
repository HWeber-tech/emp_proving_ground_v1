# Repository Cleanup Plan

## Current State Analysis
- **Root directory**: Cluttered with test files and documentation
- **Test files**: 9 test files with overlapping purposes
- **Documentation**: 6 markdown files, some redundant
- **Archive**: Needs better organization

## Cleanup Strategy

### Phase 1: Root Test File Cleanup
**Keep in root:**
- test_integration_hardening.py (comprehensive new tests)
- verify_integration.py (system verification)

**Archive these:**
- test_enhanced_integration.py → archive/tests/
- test_sensory_integration.py → archive/tests/
- test_simple_imports.py → archive/tests/
- test_simple_integration.py → archive/tests/
- test_system_hardening.py → archive/tests/
- test_system_hardening_fixed.py → archive/tests/
- test_system_stabilization.py → archive/tests/

### Phase 2: Documentation Cleanup
**Keep in root:**
- README.md
- INTEGRATION_HARDENING_SUMMARY.md (current state)

**Archive these:**
- COMPLETION_REPORT.md → archive/docs/
- INTEGRATION_SUMMARY.md → archive/docs/
- INTEGRATION_VALIDATION_REPORT.md → archive/docs/
- PRODUCTION_INTEGRATION_SUMMARY.md → archive/docs/
- PRODUCTION_VALIDATION_REPORT.md → archive/docs/

### Phase 3: Archive Organization
**Create structure:**
- archive/tests/ (for old test files)
- archive/docs/ (for old documentation)
- archive/legacy/ (for deprecated files)

### Phase 4: Final Cleanup
- Remove duplicate/variant files
- Ensure only essential files remain in root
