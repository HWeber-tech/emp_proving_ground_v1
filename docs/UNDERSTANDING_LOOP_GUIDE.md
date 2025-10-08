# Understanding Loop Phase I Guide

This guide documents the AlphaTrade understanding loop as delivered in the Phase I roadmap slice. It shows how live (or recorded) market data flows through the perceptual stack, how fast-weights bias the policy router, and how the decision diary captures evidence for governance.

The companion notebook `docs/examples/understanding_loop_demo.ipynb` walks through a runnable example using the real EURUSD sample under `docs/examples/data/eurusd_hourly_20240304_20240306.csv`.

## End-to-end Data Flow

1. **Ingest → Timescale/Kafka** – The operational backbone (`src/data_integration/real_data_integration.py`) orchestrates TimescaleDB reads, Redis caching, and Kafka fan-out so sensory organs receive fresh market frames.
2. **Sensory Fusion** – `RealSensoryOrgan` combines the WHAT/WHEN/HOW/WHY/ANOMALY organs into a single snapshot with lineage and audit trails (`src/sensory/real_sensory_organ.py`).
3. **Belief Formation** – `BeliefBuffer` applies the Hebbian covariance update to generate posterior beliefs, while `BeliefEmitter` publishes the state to the event bus (`src/understanding/belief.py`).
4. **Regime Classification** – `RegimeFSM` maps posterior strength/confidence into qualitative regimes (calm/balanced/bullish/bearish) and keeps the lineage metadata attached.
5. **Adaptation & Fast-Weights** – `UnderstandingRouter` wraps the `PolicyRouter`, applies feature-gated fast-weight adapters, and surfaces a deterministic decision bundle (`src/understanding/router.py`).
6. **Governance Evidence** – `DecisionDiaryStore` persists the decision, regime context, applied adapters, and probes so policy promotion reviews have machine-readable evidence (`src/understanding/decision_diary.py`).

The diagram below summarises the critical artifacts at each hop:

```
Timescale/Kafka frame → RealSensoryOrgan snapshot → BeliefState + RegimeSignal → UnderstandingRouter decision → DecisionDiaryEntry
```

## Fast-Weight Adaptation Cheatsheet

Fast-weights let short-lived signals bias tactic choices without retraining the base policy catalogue:

- **Feature gates** ensure adapters only fire when relevant sensory features are present and within bounds (`FeatureGate` in `src/understanding/router.py`).
- **Hebbian multipliers** roll the latest observation into a decaying multiplier so repeated evidence compounds while stale data fades.
- **Feature flags** (optional) can hard-disable adapters for compliance or A/B experimentation.
- **Summaries** are emitted with every routed decision so reviewers can see which adapters fired, the multiplier that was applied, and which feature triggered it.

Example (mirrored in the notebook): boost a breakout tactic when the HOW (liquidity) signal crosses 0.03, cap the multiplier at 3×, and decay toward 1× when the feature cools off.

## Decision Diary Interpretation

Every loop iteration emits a `DecisionDiaryEntry`. Key fields:

- `policy_id` / `tactic_id` – Strategy under evaluation.
- `regime_state` – The qualitative regime plus raw feature vector for audit.
- `decision.selected_weight` – Final fast-weight-adjusted score the router used when ranking tactics.
- `fast_weight_summary` – Per-adapter rationale and multipliers (on the mailbox event and inside the stored JSON).
- `belief_state` – Serialized posterior (means, covariance, confidence) so drift analysis can reproduce the inputs.
- `notes` / `metadata` – Free-form fields. The notebook stores `demo_run: true` here for traceability.

You can export the diary as Markdown or JSON for reviews via `DecisionDiaryStore.export_markdown()` or `DecisionDiaryStore.export_json()`.

## Running the Demo Notebook

1. Install project dependencies (`pip install -r requirements.txt`) if you have not already.
2. Open `docs/examples/understanding_loop_demo.ipynb` in Jupyter or VS Code.
3. Execute cells top-to-bottom:
   - Imports.
   - Load the EURUSD sample slice (recorded hourly candles from 2024‑03‑04/05).
   - Run `run_understanding_iteration(df)` to produce a fused snapshot, belief state, regime signal, and router decision.
   - Inspect the printed fast-weight summary and the Markdown export of the recorded diary entry.
4. Optional: Inspect `docs/examples/output/decision_diary_demo.json` after the run to see the structured record used by governance tooling.

## Troubleshooting & Next Steps

- If the notebook cannot import `src.*` modules, ensure you launch Jupyter from the project root so `PYTHONPATH` includes `.` or run `export PYTHONPATH=$(pwd)`.
- The sample dataset is intentionally small; swap in live data by pointing the notebook at a Timescale slice or by running `_execute_timescale_ingest` from `main.py` before executing the demo.
- Extend the demo by registering additional `FastWeightAdapter` instances (e.g. gating on `WHAT_signal` spikes) or by replaying the diary into `PolicyRouter.ingest_reflection_history` to simulate longer evaluation windows.

For deeper architectural context cross-reference:

- Roadmap context: `docs/roadmap.md`
- Sensory design notes: `docs/sensory/README.md`
- Governance/promotion flow: `docs/operations/runbooks/policy_promotion_governance.md`
