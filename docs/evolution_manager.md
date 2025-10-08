# Evolution Manager (Paper-Trade Auto Trials)

The evolution manager provides the first slice of the roadmap's Evolution Engine. It
monitors decision diary outcomes during paper-trade runs and automatically tweaks the
policy router when tactics underperform.

## Behaviour

- Keeps a rolling window of recent decisions (default 5) for each managed tactic.
- Derives a win/loss signal from the decision diary outcomes (`paper_pnl`, `paper_return`,
  or `win_rate`).
- When the observed win-rate drops below the configured threshold, the manager either:
  - registers a catalogue variant (e.g., an exploratory configuration), or
  - dampens the base tactic weight if no variants remain.
- Actions are feature-gated by `EVOLUTION_ENABLE_ADAPTIVE_RUNS` and only execute while the
  tactic remains in the `paper` ledger stage.

## Configuration Points

- `ManagedStrategyConfig` wires a base tactic to one or more `StrategyVariant` entries.
- Window size, win-rate threshold, and minimum observations are configurable per manager
  instance.
- Variant weights can be adjusted with `trial_weight_multiplier` to run more conservative
  experiments out of the box.

## Integration

`AlphaTradeLoopOrchestrator` now accepts an optional `EvolutionManager`. When supplied, the
manager is invoked after each iteration with the diary outcomes and can mutate the
`PolicyRouter` directly. Tests cover the following guarantees:

- Variants register after sustained losses and the base tactic weight is degraded.
- Feature flags gate all behaviour, ensuring a no-op when the override is disabled.
- Non-paper ledger stages are ignored to prevent unintended mutations in promoted phases.

This sets the groundwork for richer evolutionary logic (parameter perturbation, strategy
catalogue integration, and ledger-backed governance) in later roadmap phases.
