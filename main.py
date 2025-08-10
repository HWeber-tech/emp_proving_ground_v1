#!/usr/bin/env python3
"""
EMP v4.0 Professional Predator - Master Switch Integration

Complete system with master switch (FIX-only)
This is the final integration of Sprint 1: The Professional Upgrade
"""

import asyncio
import logging
import sys
from pathlib import Path
from decimal import Decimal
from datetime import datetime
import os
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.governance.system_config import SystemConfig
from src.governance.safety_manager import SafetyManager
from src.operational.fix_connection_manager import FIXConnectionManager
from src.operational.event_bus import EventBus

# Protocol-specific components (FIX-only)
from src.sensory.organs.fix_sensory_organ import FIXSensoryOrgan
from src.trading.integration.fix_broker_interface import FIXBrokerInterface
from src.sensory.why.why_sensor import WhySensor
from src.sensory.how.how_sensor import HowSensor
from src.sensory.what.what_sensor import WhatSensor
from src.sensory.when.when_sensor import WhenSensor
from src.sensory.anomaly.anomaly_sensor import AnomalySensor
from src.sensory.integrate.bayesian_integrator import BayesianSignalIntegrator

logger = logging.getLogger(__name__)


class EMPProfessionalPredator:
    """Professional-grade trading system with configurable protocol selection."""
    
    def __init__(self):
        self.config = None
        self.event_bus = None
        self.fix_connection_manager = None
        self.sensory_organ = None
        self.broker_interface = None
        self.running = False
        # 4D+1 sensors and integrator
        self.why_sensor = None
        self.how_sensor = None
        self.what_sensor = None
        self.when_sensor = None
        self.anomaly_sensor = None
        self.signal_integrator = None
        
    async def initialize(self, config_path: str = None):
        """Initialize the professional predator system."""
        try:
            logger.info("🚀 Initializing EMP v4.0 Professional Predator")
            
            # Load configuration
            self.config = SystemConfig()
            logger.info(f"✅ Configuration loaded: EMP v4.0 Professional Predator")
            logger.info(f"🔧 Protocol: {self.config.connection_protocol}")
            logger.info(f"🧰 Run mode: {getattr(self.config, 'run_mode', 'paper')}")
            
            # Initialize event bus
            self.event_bus = EventBus()
            # Attach a risk manager instance for global checks (lazy/placeholder)
            try:
                from src.risk.risk_manager_impl import RiskManagerImpl
                self.event_bus.risk_manager = RiskManagerImpl()
                self.event_bus.risk_manager.event_bus = self.event_bus
                logger.info("✅ Risk manager attached to event bus (RiskManagerImpl)")
            except Exception as _:
                logger.warning("⚠️ Failed to attach risk manager; proceeding without")
            logger.info("✅ Event bus initialized")

            # Initialize 4D+1 sensors and integrator
            self.why_sensor = WhySensor()
            self.how_sensor = HowSensor()
            self.what_sensor = WhatSensor()
            self.when_sensor = WhenSensor()
            self.anomaly_sensor = AnomalySensor()
            self.signal_integrator = BayesianSignalIntegrator()
            # Expose on event bus if needed elsewhere
            self.event_bus.why_sensor = self.why_sensor
            self.event_bus.how_sensor = self.how_sensor
            self.event_bus.what_sensor = self.what_sensor
            self.event_bus.when_sensor = self.when_sensor
            self.event_bus.anomaly_sensor = self.anomaly_sensor
            self.event_bus.signal_integrator = self.signal_integrator
            
            # Safety guardrails
            SafetyManager.from_config(self.config).enforce()

            # Tier selection log
            logger.info(f"🏷️ Tier selected: {getattr(self.config, 'emp_tier', 'tier_0')}")

            # Setup protocol-specific components with safety guardrails
            await self._setup_live_components()
            
            logger.info("🎉 Professional Predator initialization complete")
            
        except Exception as e:
            logger.error(f"❌ Error initializing Professional Predator: {e}")
            raise
            
    async def _setup_live_components(self):
        """Dynamically sets up sensory and trading layers based on protocol."""
        logger.info(f"🔧 Setting up LIVE components using '{self.config.connection_protocol}' protocol")
        
        if self.config.connection_protocol == "fix":
            # --- SETUP FOR PROFESSIONAL FIX PROTOCOL ---
            logger.info("🎯 Configuring FIX protocol components")
            
            # 1. Start the FIX Connection Manager
            self.fix_connection_manager = FIXConnectionManager(self.config)
            self.fix_connection_manager.start_sessions()
            
            # 2. Create message queues for thread-safe communication
            price_queue = asyncio.Queue()
            trade_queue = asyncio.Queue()
            
            # 3. Configure FIX applications with queues
            price_app = self.fix_connection_manager.get_application("price")
            trade_app = self.fix_connection_manager.get_application("trade")
            
            if price_app:
                price_app.set_message_queue(price_queue)
            if trade_app:
                trade_app.set_message_queue(trade_queue)
            
            # 4. Instantiate FIX-based components
            self.sensory_organ = FIXSensoryOrgan(self.event_bus, price_queue, self.config)
            self.broker_interface = FIXBrokerInterface(
                self.event_bus, 
                trade_queue, 
                self.fix_connection_manager.get_initiator("trade")
            )
            
            logger.info("✅ FIX components configured successfully")
            
        elif self.config.connection_protocol == "openapi":
            # Only FIX is supported in this build
            raise ValueError(
                "Only FIX is supported. Set CONNECTION_PROTOCOL=fix and follow docs/fix_api guides."
            )
            
        else:
            raise ValueError(f"❌ Unsupported connection protocol: {self.config.connection_protocol}")
        
        # 5. Inject chosen components into protocol-agnostic managers
        # Note: These would be integrated with the actual system managers
        logger.info(f"✅ Successfully configured {self.sensory_organ.__class__.__name__} and {self.broker_interface.__class__.__name__}")
    
    def _enforce_guardrails(self) -> None:
        # Deprecated: guardrails moved to SafetyManager
        SafetyManager.from_config(self.config).enforce()
        
    async def run(self):
        """Run the professional predator system."""
        try:
            self.running = True
            logger.info("🎯 Professional Predator system started")
            
            # Display system status
            summary = await self.get_system_summary()
            logger.info("📊 System Summary:")
            for key, value in summary.items():
                logger.info(f"  {key}: {value}")
                
            logger.info("🎉 Professional Predator running successfully!")
            
            # Keep running for monitoring
            while self.running:
                await asyncio.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("⏹️ Received shutdown signal")
        except Exception as e:
            logger.error(f"❌ Professional Predator error: {e}")
        finally:
            await self.shutdown()
            
    async def shutdown(self):
        """Shutdown the professional predator system."""
        try:
            logger.info("🛑 Shutting down Professional Predator")
            self.running = False
            
            if self.fix_connection_manager:
                self.fix_connection_manager.stop_sessions()
                logger.info("✅ FIX connections stopped")
                
            if self.event_bus:
                logger.info("✅ Event bus shutdown")
                
            logger.info("✅ Professional Predator shutdown complete")
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")
            
    async def get_system_summary(self) -> dict:
        """Get comprehensive system summary."""
        return {
            'version': '4.0',
            'protocol': self.config.connection_protocol,
            'status': 'RUNNING',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'sensory_organ': self.sensory_organ.__class__.__name__ if self.sensory_organ else None,
                'broker_interface': self.broker_interface.__class__.__name__ if self.broker_interface else None,
                'fix_manager': 'FIXConnectionManager' if self.fix_connection_manager else None
            }
        }


async def main():
    """Main entry point for Professional Predator."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # CLI
    parser = argparse.ArgumentParser(description='EMP Professional Predator')
    parser.add_argument('--skip-ingest', action='store_true', help='Skip Tier-0 data ingestion at startup')
    parser.add_argument('--symbols', type=str, default='EURUSD,GBPUSD', help='Comma-separated symbols for Tier-0 ingest')
    parser.add_argument('--db', type=str, default='data/tier0.duckdb', help='DuckDB path for Tier-0 ingest')
    args, _ = parser.parse_known_args()

    # Create and run Professional Predator
    system = EMPProfessionalPredator()
    
    try:
        await system.initialize()

        # Branch on tier
        emp_tier = getattr(system.config, 'emp_tier', 'tier_0')
        logger.info(f"Tier behavior: {emp_tier}")
        if emp_tier == 'tier_0' and not args.skip_ingest:
            try:
                from src.data_foundation.ingest.yahoo_ingest import fetch_daily_bars, store_duckdb
                from pathlib import Path
                symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
                logger.info(f"📥 Tier-0 ingest for {symbols}")
                df = fetch_daily_bars(symbols)
                if not df.empty:
                    store_duckdb(df, Path(args.db))
                    logger.info(f"✅ Stored {len(df)} rows to {args.db}")
                    # Run 4D+1 sensors and integrate
                    try:
                        signals = []
                        for sensor in [system.why_sensor, system.how_sensor, system.what_sensor, system.when_sensor, system.anomaly_sensor]:
                            if sensor:
                                signals.extend(sensor.process(df))
                        integrated = await system.signal_integrator.integrate(signals)
                        logger.info(f"🧠 IntegratedSignal: dir={integrated.direction} strength={integrated.strength:.3f} conf={integrated.confidence:.2f} from={integrated.contributing}")
                        # Emit on event bus for monitoring/consumers
                        try:
                            system.event_bus.publish_sync('integrated_signal', {
                                'direction': integrated.direction,
                                'strength': integrated.strength,
                                'confidence': integrated.confidence,
                                'contributing': integrated.contributing,
                                'symbols': symbols,
                                'db_path': str(Path(args.db))
                            }, source='sensory_integrator')
                        except Exception:
                            pass
                    except Exception as e:
                        logger.warning(f"Sensor integration failed: {e}")
            except Exception as e:
                logger.warning(f"Tier-0 ingest failed (continuing): {e}")
        elif emp_tier == 'tier_1':
            logger.info("🧩 Tier-1 (Timescale/Redis) not implemented yet")
        elif emp_tier == 'tier_2':
            raise NotImplementedError("Tier-2 evolutionary mode is not yet supported")

        await system.run()
    except Exception as e:
        logger.error(f"❌ Professional Predator failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
