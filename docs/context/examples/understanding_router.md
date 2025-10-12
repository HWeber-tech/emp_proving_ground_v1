# Understanding Router Configuration Examples

This note accompanies the understanding loop sprint brief and documents type-checked
examples for the new `UnderstandingRouterConfig` schema. The schema is exposed via
`src/understanding/router_config.py` and validated under guardrail tests so routing
adapters stay auditable across bootstrap and institutional tiers.【F:src/understanding/router_config.py†L1-L320】【F:tests/understanding/test_understanding_router_config.py†L1-L88】

## Bootstrap tier (dry-run shadow)

```yaml
feature_flag: fast_weights_live
default_fast_weights_enabled: false
adapters:
  - adapter_id: liquidity_rescue
    tactic_id: alpha_strike
    rationale: "Lean into alpha tactic when liquidity is stressed"
    multiplier: 1.35
    feature_gates:
      - feature: liquidity_z
        maximum: -0.20
    required_flags:
      fast_weights_live: true
tier_defaults:
  bootstrap:
    fast_weights_enabled: false
    enabled_adapters:
      - liquidity_rescue
```

## Institutional tier (fast-weight experimentation)

```yaml
feature_flag: fast_weights_live
default_fast_weights_enabled: false
adapters:
  - adapter_id: liquidity_rescue
    tactic_id: alpha_strike
    rationale: "Lean into alpha tactic when liquidity is stressed"
    multiplier: 1.35
    feature_gates:
      - feature: liquidity_z
        maximum: -0.20
    required_flags:
      fast_weights_live: true
  - adapter_id: momentum_boost
    tactic_id: alpha_strike
    rationale: "Hebbian-style multiplier when momentum is strong"
    hebbian:
      feature: momentum
      learning_rate: 0.45
      decay: 0.15
      baseline: 1.00
      floor: 0.40
      ceiling: 2.25
    required_flags:
      fast_weights_live: true
      governance_signoff: true
tier_defaults:
  institutional:
    fast_weights_enabled: true
    enabled_adapters:
      - liquidity_rescue
      - momentum_boost
```

Each adapter definition declares feature-gate bounds, required flags, optional expiry, and
Hebbian parameters. Tier defaults keep bootstrap dry-runs conservative while institutional
runs enable both adapters by default. Update these examples alongside the sprint brief when
new adapters or tier policies are introduced.

Decision bundles now expose a `fast_weight_metrics` payload containing active counts, activation percentages, and sparsity ratios derived from the controller so governance briefs and diaries inherit the same health summary reviewed in loop telemetry.【F:src/thinking/adaptation/policy_router.py†L323-L411】【F:src/understanding/router.py†L260-L263】

## Fast-weight constraint overrides via `SystemConfig`

Bootstrap and institutional deployments can tune the fast-weight sparsity guardrails by
setting extras on `SystemConfig` (environment variables or config maps). The router parses the
following keys when constructed via `UnderstandingRouter.from_system_config(...)`:

- `FAST_WEIGHT_BASELINE` (float) – default multiplier for tactics with no fast-weight boost.
- `FAST_WEIGHT_MINIMUM_MULTIPLIER` (float) – floor applied before clamping values to non-negative.
- `FAST_WEIGHT_ACTIVATION_THRESHOLD` (float) – minimum multiplier required for a tactic to count as active.
- `FAST_WEIGHT_MAX_ACTIVE_FRACTION` (float) – maximum fraction of tactics allowed to remain boosted simultaneously.
- `FAST_WEIGHT_PRUNE_TOLERANCE` (float) – tolerance when comparing multipliers against the baseline during pruning.
- `FAST_WEIGHT_EXCITATORY_ONLY` (bool) – when true, multipliers never drop below the baseline, enforcing purely excitatory adjustments.

For example, the snippet below keeps at most 20 % of tactics boosted, raises the activation
threshold to 1.1×, and forces excitatory-only updates:

```python
from src.governance.system_config import SystemConfig
from src.understanding.router import UnderstandingRouter

config = SystemConfig(
    extras={
        "FAST_WEIGHT_MAX_ACTIVE_FRACTION": "0.2",
        "FAST_WEIGHT_ACTIVATION_THRESHOLD": "1.1",
        "FAST_WEIGHT_EXCITATORY_ONLY": "true",
    }
)

router = UnderstandingRouter.from_system_config(config)
```

All other router wiring (tactic registration, adapter setup) remains unchanged, but every
decision now carries sparsity metrics that reflect the configured thresholds.
