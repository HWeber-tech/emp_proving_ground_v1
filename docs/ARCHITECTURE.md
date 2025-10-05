# EMP Proving Ground - System Architecture

## Sensory Cortex Principle: All Analysis Belongs to a Sense

### **Core Rule: No Analysis Outside the Senses**

- **All technical analysis, indicators, pattern recognition, and regime detection must be implemented as part of a "sense" within the sensory cortex (`src/sensory/`).**
- **No such logic is permitted in the data integration layer (`src/data_integration/`) or elsewhere.**
- The data integration layer is strictly for data ingestion, harmonization, fusion, and validation—never for analysis or interpretation.

---

## Architectural Overview

### 1. Data Integration Layer (`src/data_integration/`)
- **Purpose:** Ingests, harmonizes, fuses, and validates raw data from multiple sources.
- **Responsibilities:**
  - Data ingestion (APIs, files, streams)
  - Data harmonization (format/time alignment)
  - Data fusion (cross-source aggregation, confidence scoring)
  - Data quality validation
- **Prohibited:**
  - No technical analysis, indicators, pattern recognition, or market logic

### 2. Sensory Cortex (`src/sensory/`)
- **Purpose:** Performs all higher-level analysis, interpretation, and signal generation.
- **Responsibilities:**
  - Each "sense" (e.g., how, what, when, why, anomaly) encapsulates a dimension of market perception
  - All technical indicators, pattern recognition, regime detection, and signal logic are implemented as part of a sense
  - Senses consume clean, validated data from the data integration layer
- **Extending the Cortex:**
  - When a new analysis is needed, allocate it to a sense (existing or new)
  - Implement it as a method/class/module within that sense
  - Integrate it into the sense’s workflow and the sensory cortex API
  - Document the mapping of each analysis to its sense

### 3. Core Engine & Simulation (`src/core/`, `src/simulation.py`)
- **Purpose:** Consumes signals and insights from the sensory cortex to drive strategy, risk, and simulation.
- **Responsibilities:**
  - Risk management
  - Position sizing
  - Performance tracking
  - Backtesting/forward testing

---

## Example: Adding a New Technical Indicator

1. **Decide the sense:**
   - E.g., RSI, MACD, and momentum indicators → "how" sense
2. **Implement in the sense:**
   - Add to `src/sensory/dimensions/enhanced_how_dimension.py`
3. **Integrate:**
   - Expose via the sense’s API and ensure it is used in the sensory cortex workflow
4. **Never add indicator logic to `data_integration/` or core engine**

---

## Rationale
- **Modularity:** Each sense is self-contained and easy to extend
- **Maintainability:** No duplicated or misplaced analysis logic
- **Clarity:** Data flow is clean: ingestion → fusion → perception (senses) → action
- **Extensibility:** New senses or analysis can be added without disrupting the architecture

---

## Enforcing the Rule
- **Code reviews must reject any analysis logic outside the senses.**
- **If analysis is found outside the sensory cortex, it must be refactored immediately.**
- **All contributors must follow this principle.**

---

**This rule is critical for the long-term health, clarity, and power of the EMP Proving Ground system.** 
