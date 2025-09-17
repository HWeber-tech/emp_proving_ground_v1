# Dead code audit triage – 2025-09-16

This checklist maps the outstanding findings from
[`docs/reports/dead_code_audit.md`](../reports/dead_code_audit.md) to concrete
actions. Items marked as **Keep** received inline comments so future Vulture
scans can skip them; **Delete** entries were removed in this change-set; and
**Refactor** entries should become follow-up tickets.

| Symbol | Location | Classification | Action | Notes |
| --- | --- | --- | --- | --- |
| `constraints` parameter | `src/core/interfaces/base.RiskManager.propose_rebalance` | Keep | Added inline comment confirming it is required for API compatibility with the legacy risk manager. | Protocol signature is part of the public surface; implementations rely on the optional `constraints` mapping. |
| `stream` parameter | `src/data_foundation/config/vol_config._YAMLProtocol.safe_load` | Keep | Documented that the parameter mirrors the PyYAML API. | Needed so tests can monkeypatch PyYAML-compatible objects during configuration loading. |
| `_nn` type alias | `src/intelligence/portfolio_evolution` | Delete | Removed unused `torch.nn` alias under `TYPE_CHECKING`. | No remaining references; future audits should no longer flag it. |
| `_MetricsSinkBase` shim | `src/operational/metrics` | Keep | Added comment describing the optional dependency fallback. | Maintains runtime compatibility when `src.core.telemetry` is absent. |
| `documentation` parameters | `src/operational/metrics_registry` constructors | Keep | Annotated that the arguments mirror `prometheus_client` signatures. | Required so the registry can swap between real and no-op sinks without signature drift. |
| `constraints` parameter | `src/risk/risk_manager_impl.RiskManagerImpl.propose_rebalance` | Keep | Added inline comment preserving interface parity. | Concrete implementation delegates directly to the protocol contract. |
| `parity_module` re-import | `tests/current/test_parity_checker.py` | Keep | Commented that the import intentionally replays module registration. | Ensures parity hooks execute on import without asserting direct usage. |
| `SupportsInt` typing helpers | `src/core/strategy/templates/{mean_reversion,moving_average}` | Keep | No action required; parameters are consumed when casting to `int`. | Treat audit entries as false positives. |
| `StrategyTestResult` / `AttackResult` / `ExploitResult` / `StrategyAnalysis` imports | `src/thinking/adversarial/{market_gan,red_team_ai}` | Refactor | Open follow-up ticket to replace `literal_eval` shims with structured dataclasses and import the types directly when available. | Dynamic imports remain, but the audit highlights the need for a cleaner interface. |
| `after return` block | `src/trading/portfolio/real_portfolio_monitor` | Refactor | Track follow-up work to prune the unreachable branch beyond the early `return`. | The monitor should either stream metrics incrementally or remove legacy accumulation logic. |

**Next steps**

1. File tracking tickets for the two **Refactor** rows above and link them back
   to this document.
2. Re-run `scripts/audit_dead_code.py` after each cleanup batch to confirm the
   annotations silence the intended findings and to catch any newly introduced
   candidates.
