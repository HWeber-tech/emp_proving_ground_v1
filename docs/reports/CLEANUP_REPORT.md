# Cleanup Report

## Duplicates


Duplicate Definitions Summary
- classes:
  - evolution: 15
  - other: 83
  - risk: 7
  - strategy: 23
- functions:
  - evolution: 8
  - fitness: 8
  - other: 265
  - risk: 12
  - strategy: 14

Top duplicates (classes):
- riskconfig: 4 files
- instrumentprovider: 2 files
- currencyconverter: 2 files
- instrument: 3 files
- phase2dintegrationvalidator: 2 files
- riskmanager: 4 files
- validationresult: 4 files
- evolutionconfig: 3 files
- contextpacket: 3 files
- eventbus: 4 files
- orderbook: 2 files
- executionreport: 2 files
- orderbooklevel: 4 files
- performancemetrics: 4 files
- riskmetrics: 3 files
- evolutionstats: 3 files
- realevolutionengine: 2 files
- marketdata: 3 files
- strategymodel: 2 files
- instrumentmeta: 3 files


## Dependencies


Dependency Analysis
- modules: 114
- circulars: 0
- orphans: 113 (first 20)
  *  sensory.integrate.bayesian_integrator
  *  evolution.engine.genetic_engine
  *  evolution.real_genetic_engine
  *  sensory.organs.dimensions.macro_intelligence
  *  trading.execution.execution_model
  *  validation.honest_validation_framework
  *  sensory.how.how_sensor
  *  governance.token_manager
  *  trading.strategies.strategy_manager
  *  thinking.sentient_adaptation_engine
  *  validation.phase2c_validation_suite
  *  sensory.organs.price_organ
  *  sensory.anomaly.anomaly_sensor
  *  core.interfaces_complete
  *  sensory.organs.dimensions.why_organ
  *  data_integration.alpha_vantage_integration
  *  sensory.organs.dimensions.sensory_signal
  *  validation.phase2d_simple_integration
  *  thinking.analysis.correlation_analyzer
  *  sensory.why.why_sensor


## Dead Code


Dead code candidates (first 100):
-  src\core.py
-  src\phase2d_integration_validator.py
-  src\phase3_integration.py
-  src\pnl.py
-  src\risk.py
-  src\config\evolution_config.py
-  src\config\portfolio_config.py
-  src\config\risk_config.py
-  src\config\sensory_config.py
-  src\core\configuration.py
-  src\core\context_packet.py
-  src\core\events.py
-  src\core\event_bus.py
-  src\core\evolution_engine.py
-  src\core\exceptions.py
-  src\core\instrument.py
-  src\core\interfaces.py
-  src\core\interfaces_complete.py
-  src\core\market_data.py
-  src\core\models.py
-  src\core\population_manager.py
-  src\core\risk_manager.py
-  src\core\sensory_organ.py
-  src\core\validation.py
-  src\core\evolution\engine.py
-  src\core\evolution\fitness.py
-  src\core\evolution\operators.py
-  src\core\evolution\population.py
-  src\core\performance\market_data_cache.py
-  src\core\performance\vectorized_indicators.py
-  src\core\risk\manager.py
-  src\core\risk\position_sizing.py
-  src\core\risk\stress_testing.py
-  src\core\risk\var_calculator.py
-  src\core\strategy\engine.py
-  src\core\strategy\templates\mean_reversion.py
-  src\core\strategy\templates\moving_average.py
-  src\core\strategy\templates\trend_strategies.py
-  src\data_foundation\schemas.py
-  src\data_foundation\config\execution_config.py
-  src\data_foundation\config\risk_portfolio_config.py
-  src\data_foundation\config\sizing_config.py
-  src\data_foundation\config\vol_config.py
-  src\data_foundation\config\why_config.py
-  src\data_foundation\ingest\fred_calendar.py
-  src\data_foundation\ingest\yahoo_ingest.py
-  src\data_foundation\persist\jsonl_writer.py
-  src\data_foundation\persist\parquet_writer.py
-  src\data_foundation\replay\multidim_replayer.py
-  src\data_integration\alpha_vantage_integration.py
-  src\data_integration\data_fusion.py
-  src\data_integration\data_validation.py
-  src\data_integration\dukascopy_ingestor.py
-  src\data_integration\fred_integration.py
-  src\data_integration\newsapi_integration.py
-  src\data_integration\openbb_integration.py
-  src\data_integration\real_data_ingestor.py
-  src\data_integration\real_data_integration.py
-  src\data_integration\real_time_streaming.py
-  src\domain\models.py
-  src\ecosystem\coordination\coordination_engine.py
-  src\ecosystem\evaluation\niche_detector.py
-  src\ecosystem\optimization\ecosystem_optimizer.py
-  src\ecosystem\species\factories.py
-  src\evolution\advanced_evolution_engine.py
-  src\evolution\episodic_memory_system.py
-  src\evolution\real_genetic_engine.py
-  src\evolution\ambusher\ambusher_fitness.py
-  src\evolution\ambusher\ambusher_orchestrator.py
-  src\evolution\ambusher\genetic_engine.py
-  src\evolution\crossover\uniform_crossover.py
-  src\evolution\engine\genetic_engine.py
-  src\evolution\engine\real_evolution_engine.py
-  src\evolution\fitness\adaptability_fitness.py
-  src\evolution\fitness\antifragility_fitness.py
-  src\evolution\fitness\base_fitness.py
-  src\evolution\fitness\base_fitness_broken.py
-  src\evolution\fitness\efficiency_fitness.py
-  src\evolution\fitness\innovation_fitness.py
-  src\evolution\fitness\multi_dimensional_fitness_evaluator.py
-  src\evolution\fitness\profit_fitness.py
-  src\evolution\fitness\real_trading_fitness_evaluator.py
-  src\evolution\fitness\robustness_fitness.py
-  src\evolution\fitness\survival_fitness.py
-  src\evolution\fitness\trading_fitness_evaluator.py
-  src\evolution\mutation\gaussian_mutation.py
-  src\evolution\selection\adversarial_selector.py
-  src\evolution\selection\selection_strategies.py
-  src\evolution\selection\tournament_selection.py
-  src\evolution\variation\variation_strategies.py
-  src\genome\models\genome.py
-  src\governance\audit_logger.py
-  src\governance\config_vault.py
-  src\governance\fitness_store.py
-  src\governance\human_gateway.py
-  src\governance\models.py
-  src\governance\safety_manager.py
-  src\governance\strategy_registry.py
-  src\governance\system_config.py
-  src\governance\token_manager.py
Total candidates: 274
