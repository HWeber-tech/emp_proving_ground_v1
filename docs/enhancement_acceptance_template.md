# Enhancement Acceptance Criteria Template

**Purpose**: This template ensures every enhancement is falsifiable, measurable, and safely deployed through a rigorous validation ladder.

**Usage**: Copy this template for each enhancement. Attach to PRs and promotion reviews.

---

## Enhancement: [Name]

**Category**: [ ] Forecasting  [ ] Execution  [ ] Risk  [ ] Evolution  [ ] Operational  
**Priority**: [ ] Critical  [ ] High  [ ] Medium  [ ] Low  
**Effort**: [X-Y hours]  
**Assigned to**: [Developer name]  
**Target completion**: [YYYY-MM-DD]

---

## 1. Problem & Baseline

### Current State
**Problem statement**: [Describe the specific problem this enhancement solves]

**Current performance**: [Quantify current state with metrics]
- Example: "Current implementation shortfall vs. arrival: 32 bps median, 45 bps p95"
- Example: "Current WHAT sensor RMSE: 0.0042 on EUR/USD 1-hour forecasts"

**Baseline methods**: [List current approaches being used]
- Example: "Almgren-Chriss + square-root impact model"
- Example: "LSTM with 128 hidden units, trained from scratch"

**Why this matters**: [Business impact of solving this problem]
- Example: "Every 10 bps reduction in shortfall = $50K annual savings at $10M daily volume"

---

## 2. Proposed Method

### Approach
**Method**: [Name and brief description]
- Example: "RL optimal execution with offline training on logged order telemetry"

**Model architecture**: [Technical details]
- Example: "DQN with 3-layer MLP (256-128-64 units), ReLU activation, Adam optimizer"
- Example: "iTransformer with 4 layers, 8 heads, 256 embedding dim, inverted attention"

**Training data**: [What data is used, how much, time period]
- Example: "6 months of logged orders (50K trades), EUR/USD and GBP/USD"
- Example: "3 years of 1-minute OHLCV data, 6 major FX pairs"

**Comparison methods**: [What this will be compared against]
- Example: "RL-IS vs. Almgren-Chriss vs. Square-root impact vs. TWAP"
- Example: "iTransformer vs. Ridge regression vs. GBM vs. Transformer-from-scratch"

### Why This Method
**Theoretical justification**: [Why this should work]
- Example: "RL can learn non-linear impact and adapt to liquidity cycles"

**Empirical evidence**: [Citations, prior results]
- Example: "arXiv:2411.06389 shows 20-40% shortfall reduction on simulated LOB"

**Risks**: [What could go wrong]
- Example: "RL may overfit to historical market microstructure"
- Example: "Transfer learning may not generalize if domain mismatch"

---

## 3. Validation Protocol

### Data Splits
**Training**: [X% or time period]  
**Validation**: [Y% or time period]  
**Test (holdout)**: [Z% or time period, never seen during training]

**Example**: 60% train (2020-2022), 20% val (2023 H1), 20% test (2023 H2)

### Cross-Validation
**Method**: CSCV (Combinatorially Symmetric Cross-Validation)  
**Folds**: [N folds, typically 10]  
**Purge**: [X days before and after test fold]  
**Embargo**: [Y days after test fold]

**Example**: 10 folds, 2-day purge, 1-day embargo

### Statistical Tests
**Deflated Sharpe Ratio (DSR)**: [Report DSR, not raw Sharpe]  
**Probability of Backtest Overfitting (PBO)**: [Must be < 50%]  
**Combinatorial Purged Cross-Validation (CPCV)**: [Report mean ± std across folds]

### Leakage Checks
- [ ] Timestamp hygiene verified (no look-ahead)
- [ ] Data provenance documented (reproducible)
- [ ] Feature sanity checks passed (no future information)

---

## 4. Target Metrics

### Primary Metric
**Metric**: [Main success criterion]  
**Baseline**: [Current performance]  
**Target**: [Required improvement]  
**Measurement**: [How it's calculated]

**Example**:
- Metric: Implementation shortfall (bps)
- Baseline: 32 bps median
- Target: ≤25 bps median (≥20% reduction)
- Measurement: (execution_price - arrival_price) / arrival_price * 10000

### Secondary Metrics
**Metric 1**: [Supporting metric]  
**Baseline**: [Current]  
**Target**: [Required]

**Metric 2**: [Supporting metric]  
**Baseline**: [Current]  
**Target**: [Required]

**Example**:
- Metric 1: Shortfall variance (must not increase >5%)
- Metric 2: Inference latency (must be ≤10ms per decision)

### Risk Constraints
- **Maximum drawdown**: [Must not exceed X%]
- **Volatility**: [Must not increase >Y%]
- **Tail risk (CVaR 95%)**: [Must not exceed Z]

---

## 5. Live Ramp Protocol

### Stage 1: Shadow Mode
**Duration**: [N days, typically 30]  
**Action**: Run enhancement alongside baseline, log decisions, measure divergence  
**No capital at risk**: Decisions logged but not executed

**Success criteria**:
- [ ] Paper↔live divergence < 5 bps over 100+ decisions
- [ ] No crashes, errors, or timeouts
- [ ] Latency ≤ target (e.g., 10ms)

**Go/No-Go**: If success criteria met, proceed to Canary. Otherwise, debug and repeat.

---

### Stage 2: Canary
**Duration**: [N days, typically 14]  
**Exposure**: [1-5% of volume]  
**Capital cap**: [$X, typically $5K-$10K]

**Success criteria**:
- [ ] Primary metric beats baseline by ≥[X%]
- [ ] No health score degradation (health ≥ 0.7)
- [ ] Paper↔live consistency maintained (divergence < 5 bps)
- [ ] No rollback triggers activated

**Monitoring**: Real-time dashboard, alerts on divergence/health degradation

**Go/No-Go**: If success criteria met over full duration, proceed to Promotion. Otherwise, rollback.

---

### Stage 3: Promotion
**Action**: Increase exposure to [X%] of volume (e.g., 10-25%)  
**Ramp schedule**: [Gradual increase over Y days]

**Example**: 5% → 10% → 25% over 30 days, with 7-day holds at each level

**Ongoing monitoring**:
- Daily health score checks
- Weekly performance reviews
- Monthly CSCV re-validation on new data

---

## 6. Acceptance Criteria (Go/No-Go)

### Validation Phase
- [ ] Clears CSCV validation (DSR > baseline DSR + 0.3)
- [ ] PBO < 50% (not overfit)
- [ ] Beats all comparison methods on test set
- [ ] Passes leakage checks (timestamp hygiene, data provenance, feature sanity)

### Shadow Mode
- [ ] Paper↔live divergence < 5 bps over 100+ decisions
- [ ] No errors, crashes, or timeouts
- [ ] Inference latency ≤ target

### Canary
- [ ] Primary metric beats baseline by ≥[X%] (e.g., ≥15%)
- [ ] Secondary metrics within acceptable ranges
- [ ] Health score ≥ 0.7 throughout canary period
- [ ] Paper↔live consistency maintained

### Promotion
- [ ] Sustained performance over 30+ days
- [ ] No rollback triggers activated
- [ ] Documented in strategy registry with capacity curves

---

## 7. Rollback Triggers (Auto-Demote)

### Immediate Rollback
- Health score < 0.5 (critical degradation)
- Single-day drawdown > 3x baseline
- System error or crash

### 3-Day Rollback
- Health score < 0.7 for 3 consecutive days
- Primary metric worse than baseline for 3 consecutive days

### 7-Day Rollback
- Paper↔live divergence > 10 bps over 50+ trades
- Cumulative drawdown > 2x baseline over 7 days

### Manual Rollback
- Regime change detected (market structure shift)
- Regulatory concern or compliance issue
- Strategic decision (reallocation, capacity constraints)

---

## 8. Documentation Requirements

### Code
- [ ] Implementation in appropriate module (e.g., `src/execution/rl_execution.py`)
- [ ] Unit tests (≥80% coverage)
- [ ] Integration tests (end-to-end)
- [ ] Docstrings (all public methods)

### Configuration
- [ ] Hyperparameters documented in config file
- [ ] Model artifacts versioned (MLflow, DVC, or similar)
- [ ] Reproducible training script

### Monitoring
- [ ] Metrics logged to PolicyLedger
- [ ] Dashboard created (Grafana, Streamlit, or similar)
- [ ] Alerts configured (Slack, email, or similar)

### Knowledge Transfer
- [ ] README updated with enhancement description
- [ ] Architecture diagram updated (if applicable)
- [ ] Runbook created (how to operate, troubleshoot, rollback)

---

## 9. Sign-Off

### Prepared By
**Name**: [Developer]  
**Date**: [YYYY-MM-DD]  
**Signature**: _______________

### Reviewed By
**Name**: [Tech Lead]  
**Date**: [YYYY-MM-DD]  
**Signature**: _______________

### Approved By
**Name**: [Boss/Project Manager]  
**Date**: [YYYY-MM-DD]  
**Signature**: _______________

---

## 10. Post-Deployment Review

**Review date**: [30 days after promotion]

### Actual Performance
**Primary metric**: [Actual result vs. target]  
**Secondary metrics**: [Actual results]  
**Unexpected issues**: [Any surprises]

### Lessons Learned
**What worked well**: [Successes]  
**What didn't work**: [Failures]  
**What would we do differently**: [Improvements]

### Next Steps
- [ ] Continue monitoring (business as usual)
- [ ] Tune hyperparameters (optimization)
- [ ] Expand to additional instruments (scaling)
- [ ] Deprecate and replace (if underperforming)

---

**Template Version**: 1.0  
**Last Updated**: October 25, 2025  
**Owner**: EMP Development Team

