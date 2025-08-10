# Cleanup Report

## Duplicates


Duplicate Definitions Summary
- classes:
  - evolution: 4
  - other: 62
  - risk: 3
  - strategy: 9
- functions:
  - evolution: 4
  - fitness: 5
  - other: 138
  - risk: 5
  - strategy: 6

Top duplicates (classes):
- riskconfig: 4 files
- instrument: 3 files
- instrumentprovider: 2 files
- currencyconverter: 2 files
- phase2dintegrationvalidator: 2 files
- validationresult: 2 files
- riskmanager: 2 files
- evolutionconfig: 2 files
- eventbus: 7 files
- config: 2 files
- coordinationengine: 3 files
- ecosystemoptimizer: 3 files
- strategyregistry: 3 files
- systemconfig: 2 files
- scenariovalidator: 2 files
- marketgan: 2 files
- marketdatagenerator: 2 files
- marketscenario: 2 files
- strategytester: 2 files
- adversarialtrainer: 2 files


## Dependencies


Dependency Analysis
- modules: 81
- circulars: 0
- orphans: 77 (first 20)
  *  sensory.organs.orderbook_organ
  *  sensory.core.sensory_signal
  *  trading.monitoring.portfolio_monitor
  *  thinking.analysis.market_analyzer
  *  thinking.patterns.cvd_divergence_detector
  *  thinking.patterns.anomaly_detector
  *  validation.honest_validation_framework
  *  data_integration.data_fusion
  *  evolution.ambusher.ambusher_orchestrator
  *  core.strategy.templates.mean_reversion
  *  validation.phase2c_validation_suite
  *  ecosystem.species.factories
  *  core.population_manager
  *  trading.monitoring.parity_checker
  *  trading.execution.execution_model
  *  ui.cli.main_cli
  *  pnl
  *  operational.health_monitor
  *  sensory.when.when_sensor
  *  thinking.ecosystem.specialized_predator_evolution


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
-  src\data_integration\data_fusion.py
-  src\data_integration\dukascopy_ingestor.py
-  src\domain\models.py
-  src\ecosystem\coordination\coordination_engine.py
-  src\ecosystem\evaluation\niche_detector.py
-  src\ecosystem\optimization\ecosystem_optimizer.py
-  src\ecosystem\species\factories.py
-  src\evolution\ambusher\ambusher_orchestrator.py
-  src\evolution\mutation\gaussian_mutation.py
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
-  src\operational\icmarkets_robust_application.py
-  src\operational\md_capture.py
-  src\operational\metrics.py
-  src\operational\mock_fix.py
-  src\operational\state_store.py
-  src\risk\risk_manager_impl.py
-  src\sensory\advanced_data_feeds.py
-  src\sensory\advanced_data_feeds_complete.py
-  src\sensory\indicators.py
-  src\sensory\models.py
-  src\sensory\real_sensory_organ.py
-  src\sensory\signals.py
-  src\sensory\anomaly\anomaly_sensor.py
-  src\sensory\core\base.py
-  src\sensory\core\data_integration.py
-  src\sensory\core\real_sensory_organ.py
-  src\sensory\core\sensory_signal.py
-  src\sensory\core\utils.py
-  src\sensory\dimensions\indicators.py
-  src\sensory\dimensions\microstructure.py
-  src\sensory\dimensions\what\volatility_engine.py
-  src\sensory\dimensions\why\macro_signal.py
-  src\sensory\dimensions\why\yield_signal.py
-  src\sensory\examples\complete_demo.py
-  src\sensory\how\how_sensor.py
-  src\sensory\integrate\bayesian_integrator.py
-  src\sensory\integration\cross_modal.py
-  src\sensory\integration\sensory_cortex.py
-  src\sensory\orchestration\master_orchestrator.py
Total candidates: 190
