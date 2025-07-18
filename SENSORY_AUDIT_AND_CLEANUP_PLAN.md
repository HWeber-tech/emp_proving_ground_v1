# Sensory System Audit and Cleanup Plan

## Current State Analysis

### Duplicated Files Identified

#### Dimensions (src/sensory/dimensions/)
**Legacy Files (to be archived):**
- `what_engine.py` (34KB) - Legacy WHAT dimension
- `how_engine.py` (34KB) - Legacy HOW dimension  
- `when_engine.py` (44KB) - Legacy WHEN dimension
- `why_engine.py` (31KB) - Legacy WHY dimension
- `anomaly_engine.py` (41KB) - Legacy ANOMALY dimension

**Current Files (to keep):**
- `enhanced_what_dimension.py` (33KB) - Current WHAT dimension
- `enhanced_how_dimension.py` (33KB) - Current HOW dimension
- `enhanced_when_dimension.py` (42KB) - Current WHEN dimension
- `enhanced_why_dimension.py` (34KB) - Current WHY dimension
- `enhanced_anomaly_dimension.py` (67KB) - Current ANOMALY dimension

#### Orchestration (src/sensory/orchestration/)
**Legacy Files (to be archived):**
- None identified

**Current Files (to keep):**
- `enhanced_intelligence_engine.py` (64KB) - Advanced orchestration
- `master_orchestrator.py` (38KB) - Main orchestration (actively used)

### Import Analysis

#### Currently Active Imports
- **MasterOrchestrator**: Used in 15+ files (main orchestration)
- **Enhanced Intelligence Engine**: Used in 5+ files (advanced orchestration)
- **Legacy Engines**: Still imported in some test files

#### Integration Points
- **Live Trading Executor**: Uses MasterOrchestrator
- **Strategy Manager**: Uses MasterOrchestrator
- **Simulation**: Uses MasterOrchestrator
- **Main Application**: Uses MasterOrchestrator

## Advanced Analytics Integration Plan

### Current Duplication
- **Sensory System**: Sophisticated 5D market intelligence
- **Advanced Analytics**: Traditional technical indicators + sentiment

### Integration Strategy

#### 1. Merge Advanced Analytics into Sensory System

**WHAT Dimension (Technical Reality)**
- Add traditional technical indicators (RSI, MACD, Bollinger Bands, ATR)
- Integrate with existing price action analysis
- Keep advanced price action as primary, indicators as secondary

**WHY Dimension (Fundamental Intelligence)**
- Add sentiment analysis capabilities
- Integrate with existing economic data analysis
- Add news sentiment processing

**HOW Dimension (Institutional Flow)**
- Add market correlation analysis
- Integrate with existing order flow analysis
- Add beta, alpha, Sharpe ratio calculations

**WHEN Dimension (Temporal Intelligence)**
- Add temporal correlation analysis
- Integrate with existing session analysis
- Add time-based sentiment trends

**ANOMALY Dimension (Chaos Intelligence)**
- Add statistical anomaly detection from advanced analytics
- Integrate with existing chaos theory analysis
- Enhance self-refutation capabilities

#### 2. Remove Duplicate Advanced Analytics System
- Archive `src/analysis/advanced_analytics.py`
- Update live trading executor to use sensory system only
- Remove duplicate test files

## Cleanup Execution Plan

### Phase 1: Archive Legacy Files
```bash
# Create archive structure
mkdir -p archive/sensory/legacy_dimensions
mkdir -p archive/sensory/legacy_orchestration

# Move legacy files
mv src/sensory/dimensions/what_engine.py archive/sensory/legacy_dimensions/
mv src/sensory/dimensions/how_engine.py archive/sensory/legacy_dimensions/
mv src/sensory/dimensions/when_engine.py archive/sensory/legacy_dimensions/
mv src/sensory/dimensions/why_engine.py archive/sensory/legacy_dimensions/
mv src/sensory/dimensions/anomaly_engine.py archive/sensory/legacy_dimensions/
```

### Phase 2: Integrate Advanced Analytics
1. **Enhance WHAT Dimension**
   - Add technical indicators to `enhanced_what_dimension.py`
   - Integrate with existing price action analysis
   - Maintain architectural consistency

2. **Enhance WHY Dimension**
   - Add sentiment analysis to `enhanced_why_dimension.py`
   - Integrate with existing fundamental analysis
   - Add news processing capabilities

3. **Enhance HOW Dimension**
   - Add correlation analysis to `enhanced_how_dimension.py`
   - Integrate with existing institutional flow analysis
   - Add market microstructure metrics

4. **Enhance WHEN Dimension**
   - Add temporal analytics to `enhanced_when_dimension.py`
   - Integrate with existing session analysis
   - Add time-based pattern recognition

5. **Enhance ANOMALY Dimension**
   - Add statistical anomaly detection to `enhanced_anomaly_dimension.py`
   - Integrate with existing chaos theory
   - Enhance self-refutation capabilities

### Phase 3: Update Orchestration
1. **Update MasterOrchestrator**
   - Integrate new capabilities from advanced analytics
   - Maintain existing API compatibility
   - Add enhanced signal generation

2. **Update Enhanced Intelligence Engine**
   - Integrate cross-dimensional analytics
   - Enhance contextual fusion
   - Add comprehensive trading signals

### Phase 4: Update Integration Points
1. **Live Trading Executor**
   - Remove advanced analytics dependency
   - Use sensory system for all analytics
   - Update trading cycle

2. **Strategy Manager**
   - Use sensory system signals
   - Remove duplicate analytics

3. **Tests and Documentation**
   - Update all test files
   - Remove duplicate test suites
   - Update documentation

### Phase 5: Archive Advanced Analytics
```bash
# Archive duplicate system
mkdir -p archive/analysis
mv src/analysis/advanced_analytics.py archive/analysis/
mv test_advanced_analytics_*.py archive/analysis/
```

## Benefits of Integration

### 1. Architectural Consistency
- Single analytics engine instead of two
- Unified 5D market intelligence
- Consistent API and data flow

### 2. Enhanced Capabilities
- Cross-dimensional analysis
- Contextual fusion of all analytics
- Sophisticated signal generation

### 3. Maintainability
- One system to maintain
- Clear separation of concerns
- Reduced code duplication

### 4. Performance
- Optimized data flow
- Shared caching mechanisms
- Reduced computational overhead

## Implementation Priority

### High Priority
1. Archive legacy dimension files
2. Integrate technical indicators into WHAT dimension
3. Integrate sentiment analysis into WHY dimension
4. Update MasterOrchestrator

### Medium Priority
1. Integrate correlation analysis into HOW dimension
2. Integrate temporal analytics into WHEN dimension
3. Enhance ANOMALY dimension
4. Update Enhanced Intelligence Engine

### Low Priority
1. Archive advanced analytics system
2. Update all test files
3. Update documentation
4. Performance optimization

## Risk Mitigation

### 1. Backup Strategy
- Create git branch before cleanup
- Archive all files before deletion
- Maintain rollback capability

### 2. Testing Strategy
- Comprehensive integration tests
- Performance benchmarks
- Regression testing

### 3. Gradual Migration
- Phase-by-phase implementation
- Maintain backward compatibility
- Incremental testing

## Success Criteria

### 1. Functional
- All existing functionality preserved
- Enhanced analytics capabilities
- Improved signal quality

### 2. Performance
- No performance degradation
- Optimized data flow
- Reduced memory usage

### 3. Maintainability
- Single analytics system
- Clear architecture
- Comprehensive documentation

### 4. Integration
- Seamless live trading integration
- Consistent API
- Unified data model 