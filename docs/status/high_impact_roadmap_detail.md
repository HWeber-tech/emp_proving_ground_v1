# High-impact roadmap status

## Stream A – Institutional data backbone

**Status:** Ready

**Summary:** Timescale ingest, Redis caching, Kafka streaming, and Spark exports ship with readiness telemetry and failover tooling.

**Next checkpoint:** Exercise cross-region failover and automated scheduler cutover using the readiness feeds.

**Evidence:**
- data_foundation.ingest.timescale_pipeline.TimescaleBackboneOrchestrator
- data_foundation.ingest.configuration.build_institutional_ingest_config
- data_foundation.ingest.quality.evaluate_ingest_quality
- data_foundation.cache.redis_cache.ManagedRedisCache
- data_foundation.streaming.kafka_stream.KafkaIngestEventPublisher
- data_foundation.streaming.kafka_stream.KafkaIngestQualityPublisher
- data_foundation.batch.spark_export.execute_spark_export_plan
- operations.data_backbone.evaluate_data_backbone_readiness
- operations.ingest_trends.evaluate_ingest_trends
- data_foundation.ingest.failover.decide_ingest_failover
- operations.cross_region_failover.evaluate_cross_region_failover
- operations.backup.evaluate_backup_readiness
- operations.spark_stress.execute_spark_stress_drill

## Stream B – Sensory cortex & evolution uplift

**Status:** Ready

**Summary:** All five sensory organs operate with drift telemetry and catalogue-backed evolution lineage exports.

**Next checkpoint:** Extend live-paper experiments and automated tuning loops using evolution telemetry.

**Deferred (Phase 3 backlog):** Deep ICT/ML sensory research – requirements captured for prioritisation once microstructure
streaming and production telemetry land.

**Evidence:**
- sensory.how.how_sensor.HowSensor
- sensory.anomaly.anomaly_sensor.AnomalySensor
- sensory.when.gamma_exposure.GammaExposureAnalyzer
- sensory.why.why_sensor.WhySensor
- sensory.what.what_sensor.WhatSensor
- operations.sensory_drift.evaluate_sensory_drift
- genome.catalogue.load_default_catalogue
- evolution.lineage_telemetry.EvolutionLineageSnapshot
- orchestration.evolution_cycle.EvolutionCycleOrchestrator
- operations.evolution_experiments.evaluate_evolution_experiments
- operations.evolution_tuning.evaluate_evolution_tuning

## Stream C – Execution, risk, compliance, ops readiness

**Status:** Ready

**Summary:** FIX pilots, risk/compliance workflows, ROI telemetry, and operational readiness publish evidence for operators.

**Next checkpoint:** Expand broker connectivity with drop-copy reconciliation and extend regulatory telemetry coverage.

**Evidence:**
- runtime.fix_pilot.FixIntegrationPilot
- operations.fix_pilot.evaluate_fix_pilot
- runtime.fix_dropcopy.FixDropcopyReconciler
- operations.execution.evaluate_execution_readiness
- operations.professional_readiness.evaluate_professional_readiness
- operations.roi.evaluate_roi_posture
- operations.strategy_performance.evaluate_strategy_performance
- compliance.workflow.evaluate_compliance_workflows
- operations.compliance_readiness.evaluate_compliance_readiness
- compliance.trade_compliance.TradeComplianceMonitor
- compliance.kyc.KycAmlMonitor
- operations.configuration_audit.evaluate_configuration_audit
- operations.system_validation.evaluate_system_validation
- operations.slo.evaluate_ingest_slos
- operations.event_bus_health.evaluate_event_bus_health
- operations.failover_drill.execute_failover_drill
- operations.alerts.build_default_alert_manager
- risk.analytics.var.compute_parametric_var
- risk.analytics.expected_shortfall.compute_historical_expected_shortfall
- risk.analytics.volatility_target.determine_target_allocation
- risk.analytics.volatility_regime.classify_volatility_regime
- trading.order_management.lifecycle_processor.OrderLifecycleProcessor
- trading.order_management.position_tracker.PositionTracker
- trading.order_management.event_journal.OrderEventJournal
- trading.order_management.reconciliation.replay_order_events
- trading.execution.market_regime.classify_market_regime
- scripts/order_lifecycle_dry_run.py
- scripts/reconcile_positions.py
- scripts/generate_risk_report.py
- docs/runbooks/execution_lifecycle.md
