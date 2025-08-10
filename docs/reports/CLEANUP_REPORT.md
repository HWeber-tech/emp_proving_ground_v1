# Cleanup Report

## Duplicates


Duplicate Definitions Summary
- classes:
  - evolution: 15
  - other: 76
  - risk: 3
  - strategy: 14
- functions:
  - evolution: 3
  - fitness: 8
  - other: 229
  - risk: 10
  - strategy: 8

Top duplicates (classes):
- instrument: 3 files
- currencyconverter: 2 files
- instrumentprovider: 2 files
- riskconfig: 4 files
- phase2dintegrationvalidator: 2 files
- validationresult: 3 files
- riskmanager: 2 files
- strategyregistry: 3 files
- systemconfig: 2 files
- config: 2 files
- learningsignal: 2 files
- realtimelearningengine: 3 files
- adaptationcontroller: 2 files
- faisspatternmemory: 3 files
- memoryentry: 3 files
- signaltype: 2 files
- tradingsignal: 2 files
- position: 2 files
- performancetracker: 3 files
- performancemetrics: 3 files


## Dependencies


Dependency Analysis
- modules: 100
- circulars: 0
- orphans: 99 (first 20)
  *  thinking.analysis.market_analyzer
  *  sensory.core.base
  *  sensory.organs.dimensions.macro_intelligence
  *  data_integration.real_data_integration
  *  data_integration.data_validation
  *  core.strategy.templates.mean_reversion
  *  ecosystem.coordination.coordination_engine
  *  operational.health_monitor
  *  validation.phase2d_simple_integration
  *  thinking.memory.pattern_memory
  *  thinking.competitive.competitive_intelligence_system
  *  trading.execution.execution_engine
  *  evolution.selection.tournament_selection
  *  sensory.anomaly.anomaly_sensor
  *  sensory.organs.dimensions.when_organ
  *  trading.monitoring.portfolio_monitor
  *  thinking.patterns.trend_detector
  *  operational.icmarkets_api
  *  risk
  *  thinking.phase3_orchestrator


## Dead Code


Dead code candidates (first 100):
-  src/core.py
-  src/phase2d_integration_validator.py
-  src/pnl.py
-  src/risk.py
-  src/phase3_integration.py
-  src/governance/strategy_registry.py
-  src/governance/token_manager.py
-  src/governance/human_gateway.py
-  src/governance/config_vault.py
-  src/governance/fitness_store.py
-  src/governance/models.py
-  src/governance/system_config.py
-  src/governance/safety_manager.py
-  src/governance/audit_logger.py
-  src/sentient/sentient_predator.py
-  src/sentient/learning/real_time_learning_engine.py
-  src/sentient/adaptation/adaptation_controller.py
-  src/sentient/memory/faiss_pattern_memory.py
-  src/domain/models.py
-  src/trading/trading_manager.py
-  src/trading/models.py
-  src/trading/monitoring/portfolio_monitor.py
-  src/trading/monitoring/parity_checker.py
-  src/trading/monitoring/performance_tracker.py
-  src/trading/monitoring/portfolio_tracker.py
-  src/trading/models/position.py
-  src/trading/models/order.py
-  src/trading/models/trade.py
-  src/trading/strategies/order_book_analyzer.py
-  src/trading/strategies/strategy_registry.py
-  src/trading/strategies/base_strategy.py
-  src/trading/strategies/real_base_strategy.py
-  src/trading/strategies/strategy_manager.py
-  src/trading/risk_management/assessment/dynamic_risk.py
-  src/trading/risk_management/position_sizing/kelly_criterion.py
-  src/trading/execution/liquidity_prober.py
-  src/trading/execution/execution_engine.py
-  src/trading/execution/fix_executor.py
-  src/trading/execution/execution_model.py
-  src/trading/execution/live_trading_executor.py
-  src/trading/strategy_engine/base_strategy.py
-  src/trading/strategy_engine/strategy_engine.py
-  src/trading/strategy_engine/backtesting/performance_analyzer.py
-  src/trading/strategy_engine/backtesting/backtest_engine.py
-  src/trading/strategy_engine/live_management/dynamic_adjustment.py
-  src/trading/strategy_engine/live_management/strategy_monitor.py
-  src/trading/strategy_engine/optimization/parameter_tuning.py
-  src/trading/strategy_engine/optimization/genetic_optimizer.py
-  src/trading/integration/fix_broker_interface.py
-  src/trading/integration/ctrader_broker_interface.py
-  src/trading/portfolio/real_portfolio_monitor.py
-  src/sensory/advanced_data_feeds.py
-  src/sensory/models.py
-  src/sensory/indicators.py
-  src/sensory/real_sensory_organ.py
-  src/sensory/signals.py
-  src/sensory/advanced_data_feeds_complete.py
-  src/sensory/orchestration/master_orchestrator.py
-  src/sensory/how/how_sensor.py
-  src/sensory/anomaly/anomaly_sensor.py
-  src/sensory/organs/ctrader_data_organ.py
-  src/sensory/organs/volume_organ.py
-  src/sensory/organs/sentiment_organ.py
-  src/sensory/organs/yahoo_finance_organ.py
-  src/sensory/organs/economic_organ.py
-  src/sensory/organs/price_organ.py
-  src/sensory/organs/orderbook_organ.py
-  src/sensory/organs/fix_sensory_organ.py
-  src/sensory/organs/news_organ.py
-  src/sensory/organs/analyzers/anomaly_organ.py
-  src/sensory/organs/dimensions/anomaly_dimension.py
-  src/sensory/organs/dimensions/pattern_engine.py
-  src/sensory/organs/dimensions/integration_orchestrator.py
-  src/sensory/organs/dimensions/when_organ.py
-  src/sensory/organs/dimensions/sensory_signal.py
-  src/sensory/organs/dimensions/how_organ.py
-  src/sensory/organs/dimensions/real_sensory_organ.py
-  src/sensory/organs/dimensions/anomaly_detection.py
-  src/sensory/organs/dimensions/temporal_system.py
-  src/sensory/organs/dimensions/chaos_adaptation.py
-  src/sensory/organs/dimensions/why_organ.py
-  src/sensory/organs/dimensions/macro_intelligence.py
-  src/sensory/organs/dimensions/chaos_dimension.py
-  src/sensory/organs/dimensions/institutional_tracker.py
-  src/sensory/organs/dimensions/what_organ.py
-  src/sensory/organs/dimensions/base_organ.py
-  src/sensory/organs/dimensions/utils.py
-  src/sensory/organs/dimensions/data_integration.py
-  src/sensory/what/what_sensor.py
-  src/sensory/why/why_sensor.py
-  src/sensory/tests/test_integration.py
-  src/sensory/integration/cross_modal.py
-  src/sensory/integration/sensory_cortex.py
-  src/sensory/when/when_sensor.py
-  src/sensory/examples/complete_demo.py
-  src/sensory/dimensions/indicators.py
-  src/sensory/dimensions/microstructure.py
-  src/sensory/dimensions/what/volatility_engine.py
-  src/sensory/dimensions/why/yield_signal.py
-  src/sensory/dimensions/why/macro_signal.py
Total candidates: 241
