# EMP Gap Roadmap

This document captures the missing or incomplete features across the EMP (Evolving Market Predator) codebase. The goal is to operationalize the system vision by building out the scaffolding and logic necessary for robust, adaptive, live-trading autonomy. Each task includes a brief explanation of what's expected for clear implementation.

---

## Perception, Invariants, and State Fidelity

- [ ] **Implement FIX W/X snapshot + update parsing**: Create a market data parser that supports full-depth order book construction from raw FIX messages.
- [ ] **Future-mask depth features**: Ensure features from future timesteps are zeroed out or excluded to preserve causal training.
- [ ] **Consistent feature scaling**: Normalize and scale features across assets and time to make models generalizable.
- [ ] **Snapshot-to-stream backfill**: Construct synthetic time series from raw snapshots for backward testing and time-consistency checks.
- [ ] **PSI gate and alerting**: Calculate Predictive Signal Index and alert when regime drift is detected.
- [ ] **Nightly invariant drift report**: Automate the generation of drift diagnostics across all invariants.
- [ ] **CI invariant enforcement**: Fail CI pipeline if invariant checks don't pass on new features or datasets.

## Model Architecture (Mamba-3, SSM, Memory)

- [ ] **Integrate Mamba-3-style SSD block**: Use scalar \( A_t \), vector \( B_t, C_t \), and Swish activation to build a lightweight structured state-space model. Start with the SSD PyTorch reference implementation.
- [ ] **Support blockwise streaming with chunked update**: Use intra-chunk SSD and inter-chunk decay to propagate memory efficiently.
- [ ] **Expose head and state size as config params**: Defaults should be (head_dim=64, state_dim=64) with safe fallbacks.
- [ ] **Add parallel projection logic**: Project A, B, C, and X in one shot using a shared linear layer for tensor parallel compatibility.
- [ ] **Provide drop-in wrapper for existing model backbone**: The SSD block must match MLP input/output shape and device semantics.
- [ ] **Add test comparing SSD vs LSTM baseline**: Sanity check on same synthetic price series.
- [ ] **CI gate for SSD latency and output consistency**: SSD output must be deterministic and match recurrence output within tolerance.

## Runtime & Observability

- [ ] **Migrate to `TaskGroup`**: Convert all background tasks to supervised, cancellable async task groups.
- [ ] **Shutdown coverage**: Test clean exits across all runtime components under normal and failure scenarios.
- [ ] **Governance summary export**: Add high-level runtime summary metadata for CLI dashboards.
- [ ] **Live audit CLI**: Enable full export of current strategy, planner, and policy state as a JSON/YAML artifact.
- [ ] **Health + latency metrics**: Export p99 latency, exception rates, and system health in Prometheus-compatible format.

## Policy, Risk, and Execution

- [ ] **Tiered risk limits**: Implement dynamic constraints on position sizing and leverage based on market volatility.
- [ ] **RiskManager throttles**: Add runtime hooks to prevent overexposure or double-down behavior in drawdowns.
- [ ] **Policy regret test**: Benchmark learned policy against an oracle with perfect hindsight to compute regret.
- [ ] **PnL attribution logger**: Break down performance into cross, post, and hold edge vs realized profit.

## Planning & Control

- [ ] **Disable tracking**: Log reason when planner chooses to disengage or fuse.
- [ ] **Planner SLA test**: Benchmark planner response under synthetic load and fail if latency exceeds threshold.
- [ ] **Imagined vs realized edge**: Correlate plannerâ€™s expected vs actual return distribution in CI.
- [ ] **Adversarial foresight**: Inject crafted inputs (e.g., spoof ladders, reversion bait) to test planner adaptability.

## Chaos and Fault Tolerance

- [ ] **Chaos test harness**: Build simulation layer to inject fault types like duplication, timeouts, or malformed orders.
- [ ] **Flattening latency validation**: Ensure pipeline can flatten (liquidate) in under 200ms under pressure.
- [ ] **Chaos log auditing**: Log failure context and time-to-recovery after injected chaos.
- [ ] **CI chaos pass/fail**: Fail PRs that regress flatten time, misfire recovery, or lose state.

## Evolutionary Learning (EMP Core)

- [ ] **EvolutionEngine scaffold**: Build orchestration layer for model mutation, evaluation, and promotion.
- [ ] **PolicyLedger tracking**: Maintain performance, regime, and promotion metadata for all trained models.
- [ ] **Self-play loops**: Run pairwise model battles over logged market states for dominance estimation.
- [ ] **Surrogate model testing**: Train proxy models to predict outcomes and validate them with A/B ground truth.
- [ ] **Alpha fidelity gate**: Validate surrogate models dont deviate more than 5% from true alpha.
- [ ] **Curriculum injection**: Schedule rare-event or edge-case slices into training for robustness.
- [ ] **Exploitability analysis**: Compute model weaknesses and known exploit patterns.
- [ ] **Replay + mutation viewer**: Interface to explore evolution history and lineage.

## CI/CD, Gates, and Governance

- [ ] **Auto-tag on gate pass**: Label approved policies as `APPROVED_DEFAULT` in CI.
- [ ] **Gate regression reverts**: Automatically roll back deployments that fail regression gates.
- [ ] **Artifact lineage**: Stamp models and binaries with promotion history and configuration.
- [ ] **README-paper parity**: Verify that documentation reflects actual runtime boundaries.
- [ ] **CI truth check**: Prevent merging if documentation falsely claims capabilities.

## Deployment & Infra

- [ ] **Live config overlays**: Use Kustomize to expose toggles between paper and live environments.
- [ ] **Boot flag support**: Add CLI/runtime flags for `--paper-mode` and `--live-mode` with distinct config outputs.
- [ ] **Docker reproducibility**: Publish images that boot with deterministic seeds and config snapshots.
- [ ] **Helm deployment**: Add Helm charts for production use, including healthcheck and autoscaling logic.
- [ ] **Market replay smoke test**: Build system test that simulates historical depth and verifies core loop integrity.

---

This roadmap is a living execution plan. Each checkbox moves EMP closer to becoming an autonomous, self-adaptive, risk-aware market predator. All contributors should understand the *intent* behind each taskâ€”not just what to build, but *why* it matters to the overall predator architecture.
