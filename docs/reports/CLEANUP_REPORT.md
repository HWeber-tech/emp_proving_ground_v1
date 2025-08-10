# Cleanup Report

## Duplicates


Duplicate Definitions Summary
- classes:
  - evolution: 15
  - other: 72
  - risk: 3
  - strategy: 10
- functions:
  - evolution: 5
  - fitness: 8
  - other: 216
  - risk: 10
  - strategy: 7

Top duplicates (classes):
- instrument: 3 files
- currencyconverter: 2 files
- riskconfig: 4 files
- instrumentprovider: 2 files
- phase2dintegrationvalidator: 2 files
- validationresult: 3 files
- riskmanager: 2 files
- evolutionconfig: 2 files
- eventbus: 9 files
- populationmanager: 3 files
- config: 2 files
- coordinationengine: 3 files
- ecosystemoptimizer: 3 files
- multidimensionalfitnessevaluator: 2 files
- uniformcrossover: 2 files
- adaptabilityfitness: 2 files
- antifragilityfitness: 2 files
- profitfitness: 3 files
- efficiencyfitness: 2 files
- innovationfitness: 2 files


## Dependencies


Dependency Analysis
- modules: 94
- circulars: 0
- orphans: 93 (first 20)
  *  operational.metrics_collector
  *  sensory.when.when_sensor
  *  sensory.dimensions.what.volatility_engine
  *  sensory.organs.dimensions.when_organ
  *  trading.monitoring.portfolio_tracker
  *  thinking.analysis.market_analyzer
  *  thinking.adaptation.tactical_adaptation_engine
  *  sensory.organs.price_organ
  *  sensory.organs.dimensions.institutional_tracker
  *  data_integration.data_fusion
  *  sensory.integration.sensory_cortex
  *  validation.phase2c_validation_suite
  *  thinking.competitive.competitive_intelligence_system
  *  trading.execution.fix_executor
  *  ecosystem.coordination.coordination_engine
  *  sensory.organs.orderbook_organ
  *  validation.phase2d_integration_validator
  *  sensory.core.sensory_signal
  *  thinking.patterns.trend_detector
  *  thinking.memory.faiss_memory


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
-  src\core\event_bus.py
-  src\core\exceptions.py
-  src\core\instrument.py
-  src\core\interfaces.py
-  src\core\population_manager.py
-  src\core\evolution\engine.py
-  src\core\evolution\fitness.py
-  src\core\evolution\operators.py
-  src\core\evolution\population.py
-  src\core\performance\market_data_cache.py
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
-  src\evolution\ambusher\ambusher_fitness.py
-  src\evolution\ambusher\ambusher_orchestrator.py
-  src\evolution\crossover\uniform_crossover.py
-  src\evolution\fitness\adaptability_fitness.py
-  src\evolution\fitness\antifragility_fitness.py
-  src\evolution\fitness\base_fitness.py
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
-  src\integration\component_integrator.py
-  src\integration\component_integrator_impl.py
-  src\intelligence\adversarial_training.py
-  src\intelligence\competitive_intelligence.py
-  src\intelligence\portfolio_evolution.py
-  src\intelligence\predictive_modeling.py
-  src\intelligence\red_team_ai.py
-  src\intelligence\sentient_adaptation.py
-  src\intelligence\specialized_predators.py
-  src\operational\event_bus.py
-  src\operational\fix_connection_manager.py
-  src\operational\health_monitor.py
-  src\operational\icmarkets_api.py
-  src\operational\icmarkets_config.py
Total candidates: 230
