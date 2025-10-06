1 - Produce comprehensive mypy remediation plan and sequencing strategy
2 - Create a long-form, high-impact TODO list to drive the cleanup effort
3 - Add strict mypy configuration to [pyproject.toml](pyproject.toml:53) with mypy_path including src and stubs, pydantic plugin enabled, and disallow-any settings preserved
4 - Add project-wide [mypy.ini](mypy.ini:1) with equivalent settings if pyproject is not used or for tool compatibility
5 - Add pre-commit hooks in [.pre-commit-config.yaml](.pre-commit-config.yaml:1) for mypy, ruff, and black
6 - Wire pre-commit into CI and developers’ local workflows (document in [docs/development/contributing.md](docs/development/contributing.md:1))
7 - Ensure all package directories have __init__.py files (audit src/* and add as needed)
8 - Add typed JSON aliases to [src/core/types.py](src/core/types.py:1) (e.g., JSONValue, JSONObject) and reuse across payloads
9 - Create local stubs tree root [stubs/README.md](stubs/README.md:1) explaining stub policy and maintenance
10 - Add stub package [stubs/duckdb/__init__.pyi](stubs/duckdb/__init__.pyi:1) with the minimal API used
11 - Add stub package [stubs/faiss/__init__.pyi](stubs/faiss/__init__.pyi:1) with the minimal API used
12 - Add stub package [stubs/simplefix/__init__.pyi](stubs/simplefix/__init__.pyi:1) with the minimal API used
13 - Add stub package [stubs/yfinance/__init__.pyi](stubs/yfinance/__init__.pyi:1) with the minimal API used
14 - Add stub package [stubs/torch/__init__.pyi](stubs/torch/__init__.pyi:1) for torch root imports
15 - Add stub package [stubs/torch/nn.pyi](stubs/torch/nn.pyi:1) for torch.nn symbols imported
16 - Add stub package [stubs/sklearn/__init__.pyi](stubs/sklearn/__init__.pyi:1) with minimal shapes and py.typed
17 - Add stub module [stubs/sklearn/cluster.pyi](stubs/sklearn/cluster.pyi:1) for used clustering classes
18 - Add stub module [stubs/sklearn/preprocessing.pyi](stubs/sklearn/preprocessing.pyi:1) for used preprocessing classes
19 - Ensure PyYAML types installed (types-PyYAML); confirm via mypy run
20 - Ensure pandas stubs installed (pandas-stubs); confirm via mypy run
21 - Ensure psutil types installed (types-psutil); confirm via mypy run
22 - Normalize internal imports to src.* consistently across codebase (no bare-package imports)
23 - Add typed re-export shim [src/core/event_bus.py](src/core/event_bus.py:1) if event bus has moved, mapping old import sites to actual implementation
24 - Legacy operational event bus path now aliases to [src/core/event_bus.py](src/core/event_bus.py:1) during `src.operational` import; keep the alias regression in [tests/operational/test_event_bus_alias.py](tests/operational/test_event_bus_alias.py:162) green
25 - Fix risk package public API: reconcile [src/risk/__init__.py](src/risk/__init__.py:1) to import the real module or add [src/risk/real_risk_manager.py](src/risk/real_risk_manager.py:1) if needed
26 - Audit orchestration imports in [src/orchestration/enhanced_intelligence_engine.py](src/orchestration/enhanced_intelligence_engine.py:1) and update to canonical src.market_intelligence.* paths
27 - Standardize imports in market intelligence dimension modules under [src/market_intelligence/dimensions/](src/market_intelligence/dimensions/)
28 - Add authoritative interface hub [src/core/interfaces/__init__.py](src/core/interfaces/__init__.py:1) describing Protocols for shared interfaces
29 - Define risk manager protocol and related config protocols in [src/core/interfaces/__init__.py](src/core/interfaces/__init__.py:1)
30 - Define configuration provider protocol in [src/core/interfaces/__init__.py](src/core/interfaces/__init__.py:1)
31 - Define market data gateway protocols in [src/core/interfaces/__init__.py](src/core/interfaces/__init__.py:1)
32 - Define regime classifier protocol in [src/core/interfaces/__init__.py](src/core/interfaces/__init__.py:1)
33 - Define metrics sink/registry protocols in [src/core/interfaces/__init__.py](src/core/interfaces/__init__.py:1)
34 - Define ecosystem optimizer and coordination engine protocols in [src/core/interfaces/__init__.py](src/core/interfaces/__init__.py:1)
35 - Rewire modules importing “src.core.interfaces” to align with the new Protocols in [src/core/interfaces/__init__.py](src/core/interfaces/__init__.py:1)
36 - Create canonical genome model alignment plan in [docs/development/DE-SHIMMIFICATION_PLAN.md](docs/development/DE-SHIMMIFICATION_PLAN.md:1) for DecisionGenome and adapters
37 - Align dataclass fields and constructor of genome in [src/genome/models/genome.py](src/genome/models/genome.py:1) with actual usage across ecosystem modules
38 - Replace Any-based helpers in [src/genome/models/genome.py](src/genome/models/genome.py:1) with concrete types or TypedDicts
39 - Fix adapters in [src/genome/models/adapters.py](src/genome/models/adapters.py:1) to be fully typed and to match canonical genome
40 - Clean up [src/genome/models/genome_adapter.py](src/genome/models/genome_adapter.py:1) by removing unused type: ignore comments and replacing Any with typed adapters
41 - Replace dynamic lazy indirections in [src/intelligence/__init__.py](src/intelligence/__init__.py:1) with either explicit imports under TYPE_CHECKING or module-level .pyi stub
42 - Provide module-level stubs [src/intelligence/__init__.pyi](src/intelligence/__init__.pyi:1) listing exported names with precise types
43 - Provide stubs for market intelligence dimension aggregators in [src/market_intelligence/dimensions/](src/market_intelligence/dimensions/) where __getattr__ was used
44 - Add stubs or explicit exports for orchestration adapters in [src/orchestration/compose.py](src/orchestration/compose.py:1) to eliminate Any
45 - Replace Any returns with TypedDict or dataclass payloads in [src/orchestration/compose.py](src/orchestration/compose.py:1)
46 - Fix pandas import typing in [src/orchestration/compose.py](src/orchestration/compose.py:1) by local import under TYPE_CHECKING and concrete signatures
47 - Remove unused type: ignore and fix generics for memoization in [src/operational/metrics.py](src/operational/metrics.py:1) and ensure get_registry returns protocol types
48 - Define CounterLike, GaugeLike, HistogramLike protocols in [src/operational/metrics_registry.py](src/operational/metrics_registry.py:1) or in [src/core/interfaces/__init__.py](src/core/interfaces/__init__.py:1)
49 - Type the memoization dictionaries in [src/operational/metrics_registry.py](src/operational/metrics_registry.py:1) with precise keys and values
50 - Ensure registry getters in [src/operational/metrics_registry.py](src/operational/metrics_registry.py:1) return protocol types, not Any
51 - Refactor [src/operational/icmarkets_robust_application.py](src/operational/icmarkets_robust_application.py:1) to type sockets, SSL sockets, threads, and message_queue properly
52 - Guard Optional socket/SSL fields before use in [src/operational/icmarkets_robust_application.py](src/operational/icmarkets_robust_application.py:1)
53 - Add missing returns and consistent bool return flow in [src/operational/icmarkets_robust_application.py](src/operational/icmarkets_robust_application.py:1)
54 - Replace Any payloads with small dataclasses or TypedDicts in [src/operational/icmarkets_robust_application.py](src/operational/icmarkets_robust_application.py:1)
55 - Replace anonymous type() constructs with typed dataclasses in [src/operational/mock_fix.py](src/operational/mock_fix.py:1)
56 - Fully type callbacks and queues in [src/operational/mock_fix.py](src/operational/mock_fix.py:1) and [src/operational/fix_connection_manager.py](src/operational/fix_connection_manager.py:1)
57 - Remove “unused type: ignore” and fix generics for asyncio.Queue in [src/operational/fix_connection_manager.py](src/operational/fix_connection_manager.py:1)
58 - Refactor message assembly payloads in [src/operational/fix_connection_manager.py](src/operational/fix_connection_manager.py:1) to TypedDicts
59 - Add return annotations and fix None misuse in [src/operational/state_store/adapters.py](src/operational/state_store/adapters.py:1)
60 - Add type annotation to health history and fully type health check results in [src/operational/health_monitor.py](src/operational/health_monitor.py:1)
61 - Use pandas and psutil stubs correctly within [src/operational/health_monitor.py](src/operational/health_monitor.py:1) and add concrete return types
62 - Fix _serialize_order_book and inner functions with concrete typed payloads in [src/operational/md_capture.py](src/operational/md_capture.py:1)
63 - Add return type annotations for dataclass post-init and correct Optional usage in [src/config/sensory_config.py](src/config/sensory_config.py:1)
64 - Fix mutable defaults via default_factory in [src/config/sensory_config.py](src/config/sensory_config.py:1)
65 - Replace Any in sensory utils with concrete numeric types in [src/sensory/organs/dimensions/utils.py](src/sensory/organs/dimensions/utils.py:1)
66 - Ensure numpy numerical returns are coerced to float where appropriate in [src/sensory/organs/dimensions/utils.py](src/sensory/organs/dimensions/utils.py:1)
67 - Fix pandas typing and imports in [src/sensory/organs/dimensions/why_organ.py](src/sensory/organs/dimensions/why_organ.py:1)
68 - Replace Any payloads in [src/sensory/organs/dimensions/why_organ.py](src/sensory/organs/dimensions/why_organ.py:1) with TypedDict or dataclasses
69 - Replace “return Any” shape with precise types in [src/sensory/organs/dimensions/why_organ.py](src/sensory/organs/dimensions/why_organ.py:1)
70 - Ensure core base types for sensory are imported from canonical modules (fix old src.sensory.core.base paths) in [src/sensory/organs/dimensions/why_organ.py](src/sensory/organs/dimensions/why_organ.py:1)
71 - Add return annotations and fix dict generics in [src/data_foundation/replay/multidim_replayer.py](src/data_foundation/replay/multidim_replayer.py:1)
72 - Replace dict with Dict[str, T] and TypedDict in replay callbacks within [src/data_foundation/replay/multidim_replayer.py](src/data_foundation/replay/multidim_replayer.py:1)
73 - Add concrete typing for parquet writer in [src/data_foundation/persist/parquet_writer.py](src/data_foundation/persist/parquet_writer.py:1)
74 - Add concrete typing for jsonl writer in [src/data_foundation/persist/jsonl_writer.py](src/data_foundation/persist/jsonl_writer.py:1)
75 - Provide duckdb stubs and correct imports in [src/data_foundation/ingest/yahoo_ingest.py](src/data_foundation/ingest/yahoo_ingest.py:1)
76 - Add return annotations for main() and IO types in [src/data_foundation/ingest/yahoo_ingest.py](src/data_foundation/ingest/yahoo_ingest.py:1)
77 - Type market data gateway interfaces in [src/core/market_data.py](src/core/market_data.py:1) without Any usage
78 - Replace Any metadata in [src/core/regime.py](src/core/regime.py:1) with precise Optional[Dict[str, str|float|…]] or TypedDicts
79 - Ensure detect_regime signatures are fully typed and async in [src/core/regime.py](src/core/regime.py:1)
80 - Add domain model types without Any in [src/domain/models.py](src/domain/models.py:1) (Pydantic models typed fields)
81 - Fix Pydantic BaseModel usage in [src/data_foundation/schemas.py](src/data_foundation/schemas.py:1) to avoid Any fields and dynamic tricks
82 - Replace Any in risk config model with precise types in [src/config/risk/risk_config.py](src/config/risk/risk_config.py:1)
83 - Normalize size config generics in [src/data_foundation/config/sizing_config.py](src/data_foundation/config/sizing_config.py:1) (dict generics and Optional defaults)
84 - Normalize vol config imports and typing in [src/data_foundation/config/vol_config.py](src/data_foundation/config/vol_config.py:1) (fix engine import path)
85 - Add explicit annotations for helper conversions in [src/data_foundation/config/vol_config.py](src/data_foundation/config/vol_config.py:1)
86 - Implement canonical risk manager config and imports in [src/risk/risk_manager_impl.py](src/risk/risk_manager_impl.py:1)
87 - Replace Decimal arguments with expected types or update config to accept Decimal in [src/risk/risk_manager_impl.py](src/risk/risk_manager_impl.py:1)
88 - Replace Any dicts for positions with TypedDicts or dataclasses in [src/risk/risk_manager_impl.py](src/risk/risk_manager_impl.py:1)
89 - Add asyncio import and correct run-time examples at bottom of [src/risk/risk_manager_impl.py](src/risk/risk_manager_impl.py:1)
90 - Correct references to underlying RealRiskManager methods in [src/risk/risk_manager_impl.py](src/risk/risk_manager_impl.py:1) or adjust adapter accordingly
91 - Remove unused type: ignore and duplicate StrEnum shim in [src/governance/system_config.py](src/governance/system_config.py:1)
92 - Add return annotation for with_updated and explicit coercion typing in [src/governance/system_config.py](src/governance/system_config.py:1)
93 - Type and fix DB initialization and query results in [src/governance/strategy_registry.py](src/governance/strategy_registry.py:1)
94 - Replace Any in registry summary and query output with precise dict types or TypedDicts in [src/governance/strategy_registry.py](src/governance/strategy_registry.py:1)
95 - Add return annotations and replace Any in [src/governance/audit_logger.py](src/governance/audit_logger.py:1); type event_types and strategies dicts
96 - Fix SafetyManager.from_config signature annotations in [src/governance/safety_manager.py](src/governance/safety_manager.py:1)
97 - Resolve core.events legacy imports by adding canonical module or shims used by thinking modules (place under [src/core/events.py](src/core/events.py:1))
98 - Fully type predictive modeling orchestrator in [src/thinking/prediction/predictive_market_modeler.py](src/thinking/prediction/predictive_market_modeler.py:1)
99 - Replace Any payloads with typed JSONDict/TypedDict in [src/thinking/prediction/predictive_market_modeler.py](src/thinking/prediction/predictive_market_modeler.py:1)
100 - Ensure numpy array types via NDArray[np.float64] in [src/thinking/prediction/predictive_market_modeler.py](src/thinking/prediction/predictive_market_modeler.py:1)
101 - Clean up literal_eval paths and types in [src/thinking/prediction/predictive_market_modeler.py](src/thinking/prediction/predictive_market_modeler.py:1)
102 - Provide competitive intelligence system typing in [src/thinking/competitive/competitive_intelligence_system.py](src/thinking/competitive/competitive_intelligence_system.py:1)
103 - Replace Any across behavior/signature/strategy payloads in [src/thinking/competitive/competitive_intelligence_system.py](src/thinking/competitive/competitive_intelligence_system.py:1)
104 - Fix float vs numpy scalar math consistency and type coercions in [src/thinking/competitive/competitive_intelligence_system.py](src/thinking/competitive/competitive_intelligence_system.py:1)
105 - Fully type red team engine in [src/thinking/adversarial/red_team_ai.py](src/thinking/adversarial/red_team_ai.py:1) with explicit dataclasses/TypedDicts
106 - Remove dynamic lazy proxies for numpy and type resolution in [src/intelligence/red_team_ai.py](src/intelligence/red_team_ai.py:1) by using TYPE_CHECKING and stubs
107 - Replace Any and legacy ignores in [src/intelligence/red_team_ai.py](src/intelligence/red_team_ai.py:1) with precise types and stubs
108 - Provide typed wrappers or stubs for sklearn in modules under [src/intelligence/](src/intelligence/)
109 - Replace Any payloads and missing returns in [src/thinking/adversarial/market_gan.py](src/thinking/adversarial/market_gan.py:1) and add typed results
110 - Normalize mypy treatment of torch imports via local stubs in [src/intelligence/portfolio_evolution.py](src/intelligence/portfolio_evolution.py:1)
111 - Replace Any and add precise ndarray typing in [src/intelligence/portfolio_evolution.py](src/intelligence/portfolio_evolution.py:1)
112 - Fix diversification and synergy computations to return float, not numpy scalars in [src/intelligence/portfolio_evolution.py](src/intelligence/portfolio_evolution.py:1)
113 - Type and fix coordination engine protocols in [src/ecosystem/coordination/coordination_engine.py](src/ecosystem/coordination/coordination_engine.py:1)
114 - Replace Any in prioritize_strategies payloads and scores in [src/ecosystem/coordination/coordination_engine.py](src/ecosystem/coordination/coordination_engine.py:1)
115 - Ensure defaultdict type parameters in [src/ecosystem/coordination/coordination_engine.py](src/ecosystem/coordination/coordination_engine.py:1)
116 - Replace Any in ecosystem optimizer public API in [src/ecosystem/optimization/ecosystem_optimizer.py](src/ecosystem/optimization/ecosystem_optimizer.py:1)
117 - Fix DecisionGenome constructor and field usage mismatches in [src/ecosystem/optimization/ecosystem_optimizer.py](src/ecosystem/optimization/ecosystem_optimizer.py:1)
118 - Replace Any in evaluation/niche detection with precise pandas/numpy types in [src/ecosystem/evaluation/niche_detector.py](src/ecosystem/evaluation/niche_detector.py:1)
119 - Type rolling function helpers in [src/ecosystem/evaluation/niche_detector.py](src/ecosystem/evaluation/niche_detector.py:1)
120 - Align specialist factories to canonical genome in [src/ecosystem/species/factories.py](src/ecosystem/species/factories.py:1)
121 - Remove inheritance from missing Any-typed interfaces by using Protocols from [src/core/interfaces/__init__.py](src/core/interfaces/__init__.py:1) in [src/ecosystem/species/factories.py](src/ecosystem/species/factories.py:1)
122 - Replace Any in specialized predator evolution module in [src/ecosystem/evolution/specialized_predator_evolution.py](src/ecosystem/evolution/specialized_predator_evolution.py:1)
123 - Fix imports and use typed ecosystem optimizer in [src/ecosystem/evolution/specialized_predator_evolution.py](src/ecosystem/evolution/specialized_predator_evolution.py:1)
124 - Replace Any in sentient adaptation with typed signals and controllers in [src/intelligence/sentient_adaptation.py](src/intelligence/sentient_adaptation.py:1)
125 - Ensure correct arguments passed to learning engine, pattern memory, and adaptation controller in [src/intelligence/sentient_adaptation.py](src/intelligence/sentient_adaptation.py:1)
126 - Implement missing core events or typed shims to satisfy imports in [src/thinking/phase3_orchestrator.py](src/thinking/phase3_orchestrator.py:1)
127 - Replace Any in orchestration results and summaries in [src/thinking/phase3_orchestrator.py](src/thinking/phase3_orchestrator.py:1)
128 - Eliminate unreachable code branches reported by mypy in [src/thinking/phase3_orchestrator.py](src/thinking/phase3_orchestrator.py:1)
129 - Replace Any and add return annotations in [src/phase3_integration.py](src/phase3_integration.py:1)
130 - Ensure no implicit Optional for config arguments in [src/phase3_integration.py](src/phase3_integration.py:1)
131 - Replace Any in integration component types and fix get_component_status signature in [src/integration/component_integrator_impl.py](src/integration/component_integrator_impl.py:1)
132 - Make base interface of integrator a Protocol in [src/integration/component_integrator.py](src/integration/component_integrator.py:1) rather than subclassing from Any
133 - Add canonical trading execution engine shim or adjust imports in [src/integration/component_integrator_impl.py](src/integration/component_integrator_impl.py:1)
134 - Replace Any return for “_” sentinel and remove undefined symbol in [src/integration/component_integrator.py](src/integration/component_integrator.py:1)
135 - Replace Any across UI manager (avoid redefining imported names) in [src/ui/ui_manager.py](src/ui/ui_manager.py:1)
136 - Rename local demo classes to avoid name collisions in [src/ui/ui_manager.py](src/ui/ui_manager.py:1)
137 - Add return annotations for all UI manager methods in [src/ui/ui_manager.py](src/ui/ui_manager.py:1)
138 - Add pandas stubs and fix types in CLI in [src/ui/cli/main_cli.py](src/ui/cli/main_cli.py:1)
139 - Replace Any in trading parity checker payloads and add annotations in [src/trading/monitoring/parity_checker.py](src/trading/monitoring/parity_checker.py:1)
140 - Ensure no Any in trading monitoring diff logic and helper closures in [src/trading/monitoring/parity_checker.py](src/trading/monitoring/parity_checker.py:1)
141 - Replace Any in performance cache and cache-market_data entries in [src/performance/__init__.py](src/performance/__init__.py:1)
142 - Use NDArray typing and typed dict returns in [src/performance/vectorized_indicators.py](src/performance/vectorized_indicators.py:1) (no bare dict)
143 - Replace Any in pnl models and annotate dataclass post-init in [src/pnl.py](src/pnl.py:1)
144 - Clarify instrument
145 - Fix broken sensory imports and explicit public API in [src/sensory/dimensions/__init__.py](src/sensory/dimensions/__init__.py:1)
146 - Replace explicit Any with precise types and add concrete dataclasses/Protocols in [src/core/instrument.py](src/core/instrument.py:1)
147 - Replace explicit Any and add precise types/TypedDicts in [src/core/anomaly.py](src/core/anomaly.py:1)
148 - Fix unreachable code and typing issues in [src/config/sensory_config.py](src/config/sensory_config.py:1) alongside return annotations and default factories
