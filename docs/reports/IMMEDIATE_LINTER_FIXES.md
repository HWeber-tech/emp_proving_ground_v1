# Immediate Linter Fixes - Critical Issues

## Top 10 Critical Issues Requiring Immediate Attention

### 1. Missing Type Stubs (CRITICAL)
```bash
pip install types-requests
```
**Impact**: 240+ mypy errors
**Files**: All trading and data modules

### 2. Unused Imports (HIGH)
**Files with most issues:**
- `src/trading/__init__.py` - 9 unused imports
- `src/trading/advanced_risk_manager.py` - 4 unused imports
- `src/trading/ctrader_interface.py` - 3 unused imports

**Quick fix:**
```python
# Remove these unused imports
from datetime import timedelta  # unused
from decimal import Decimal    # unused
import asyncio                 # unused
import requests               # unused
```

### 3. Missing Type Annotations (CRITICAL)
**Files needing immediate attention:**
- `src/sensory/dimensions/enhanced_*.py` - All dimension files
- `src/trading/*.py` - All trading files
- `src/evolution/real_genetic_engine.py`

**Example fixes:**
```python
# Before
price_history = []
volume_history = []
analysis_history = []

# After
price_history: List[float] = []
volume_history: List[float] = []
analysis_history: List[DimensionalReading] = []
```

### 4. Whitespace Issues (HIGH)
**Files with 100+ whitespace issues:**
- `src/simulation.py` - 690+ issues
- `src/trading/advanced_risk_manager.py` - 200+ issues
- `src/trading/ctrader_interface.py` - 100+ issues

**Quick fix:**
```bash
# Auto-fix whitespace issues
python -m autopep8 --in-place --recursive src/
```

### 5. Missing DimensionalReading Attributes (CRITICAL)
**Error**: `"DimensionalReading" has no attribute "value"`
**Files**: All sensory orchestration files

**Fix needed:**
```python
# Add missing attributes to DimensionalReading class
@dataclass
class DimensionalReading:
    value: float  # Missing attribute
    data_quality: float  # Missing attribute
    processing_time_ms: float  # Missing attribute
    # ... other attributes
```

### 6. Incompatible Type Assignments (CRITICAL)
**Files with type errors:**
- `src/evolution/real_genetic_engine.py`
- `src/decision_genome.py`
- `src/simulation.py`

**Example fixes:**
```python
# Before
fitness_score: int = 0.0  # Type mismatch

# After
fitness_score: float = 0.0  # Correct type
```

### 7. Undefined Names (CRITICAL)
**Files:**
- `src/data.py` - `RealDataIngestor` not defined
- `src/trading/strategy_manager.py` - `_prev_close` not defined

**Fixes needed:**
```python
# Add missing imports or definitions
from .data.real_data_ingestor import RealDataIngestor
```

### 8. Bare Except Statements (MEDIUM)
**Files:**
- `src/trading/real_ctrader_interface.py`
- `src/trading/strategy_manager.py`

**Fix:**
```python
# Before
except Exception as e:

# After
except (ConnectionError, TimeoutError) as e:
```

### 9. Missing Newlines (LOW)
**Files:**
- `src/trading/__init__.py`
- `src/trading/advanced_risk_manager.py`
- `src/trading/ctrader_interface.py`

**Fix:**
```bash
# Add newlines to end of files
echo "" >> src/trading/__init__.py
```

### 10. Duplicate Code Blocks (MEDIUM)
**Major duplicates:**
- Trading interface implementations
- Sensory dimension patterns
- Performance tracking structures

## Immediate Action Plan (Next 24 Hours)

### Hour 1-2: Install Dependencies
```bash
pip install types-requests black isort autopep8
```

### Hour 3-4: Auto-fix Whitespace
```bash
python -m autopep8 --in-place --recursive src/
python -m black src/
python -m isort src/
```

### Hour 5-6: Fix Missing Type Stubs
```bash
# Install all missing type stubs
pip install types-requests types-aiohttp types-websockets
```

### Hour 7-8: Fix Critical Type Errors
- Add missing DimensionalReading attributes
- Fix incompatible type assignments
- Add missing type annotations

### Hour 9-10: Remove Unused Imports
- Clean up all unused imports
- Fix import order issues
- Remove star imports

### Hour 11-12: Test and Validate
```bash
# Run linters
python -m flake8 src/ --max-line-length=120
python -m pylint src/ --disable=C0114,C0115,C0116
python -m mypy src/ --ignore-missing-imports

# Run tests
python -m pytest tests/ -v
```

## Expected Results After 24 Hours

- **Flake8 errors**: 4,321 → <500
- **Mypy errors**: 240 → <50
- **Pylint score**: 4.98/10 → 6.5+/10
- **All tests passing**: ✅

## Files Requiring Manual Review

1. `src/sensory/core/base.py` - DimensionalReading class definition
2. `src/trading/__init__.py` - Import structure
3. `src/evolution/real_genetic_engine.py` - Type assignments
4. `src/decision_genome.py` - Type system issues

## Success Criteria

- [ ] All critical type errors resolved
- [ ] All unused imports removed
- [ ] All whitespace issues fixed
- [ ] All tests passing
- [ ] Linter scores improved by 50%+ 