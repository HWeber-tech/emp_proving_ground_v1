#!/usr/bin/env python3
"""
Phase 2D: Real Integration & Testing Framework (Legacy Shim)
===========================================================

This legacy module previously imported sensory organs, trading risk components,
and data_integration modules directly. It now delegates to the modern
Phase2DIntegrationValidator that uses core ports and DI adapters assembled in
orchestration.compose.

- No direct imports from sensory, trading, or data_integration.
- Safe, non-raising behavior maintained.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List

from src.validation.phase2d_integration_validator import (
    Phase2DIntegrationValidator as ModernPhase2DIntegrationValidator,
)
from src.orchestration.compose import compose_validation_adapters
from src.core.market_data import MarketDataGateway, NoOpMarketDataGateway
from src.core.anomaly import AnomalyDetector, NoOpAnomalyDetector
from src.core.regime import RegimeClassifier, NoOpRegimeClassifier
from src.core.risk_ports import RiskManagerPort, NoOpRiskManager

logger = logging.getLogger(__name__)


class Phase2DIntegrationValidatorLegacy:
    """
    Legacy wrapper that delegates to the modern Phase2DIntegrationValidator
    wired with DI adapters from orchestration.compose.
    """

    def __init__(self) -> None:
        adapters = compose_validation_adapters()
        self.market_data: MarketDataGateway = adapters.get("market_data_gateway", NoOpMarketDataGateway())
        self.anomaly_detector: AnomalyDetector = adapters.get("anomaly_detector", NoOpAnomalyDetector())
        self.regime_classifier: RegimeClassifier = adapters.get("regime_classifier", NoOpRegimeClassifier())
        self.risk_manager: RiskManagerPort = adapters.get("risk_manager", NoOpRiskManager())
        self._modern = ModernPhase2DIntegrationValidator(
            market_data_gateway=self.market_data,
            anomaly_detector=self.anomaly_detector,
            regime_classifier=self.regime_classifier,
            risk_manager=self.risk_manager,
        )

    async def test_real_data_flow(self) -> Dict[str, Any]:
        """Delegate to modern test_real_data_flow."""
        try:
            return await self._modern.test_real_data_flow()
        except Exception as e:
            logger.error(f"Legacy real data flow test failed: {e}")
            return {
                "test_name": "real_data_flow",
                "passed": False,
                "error": str(e),
                "details": "Legacy wrapper execution failed",
            }

    async def test_strategy_performance_tracking(self) -> Dict[str, Any]:
        """Delegate to modern test_strategy_performance_tracking."""
        try:
            return await self._modern.test_strategy_performance_tracking()
        except Exception as e:
            logger.error(f"Legacy strategy performance tracking failed: {e}")
            return {
                "test_name": "strategy_performance_tracking",
                "passed": False,
                "error": str(e),
                "details": "Legacy wrapper execution failed",
            }

    async def test_concurrent_operations(self) -> Dict[str, Any]:
        """Delegate to modern test_concurrent_operations."""
        try:
            return await self._modern.test_concurrent_operations()
        except Exception as e:
            logger.error(f"Legacy concurrent operations test failed: {e}")
            return {
                "test_name": "concurrent_operations",
                "passed": False,
                "error": str(e),
                "details": "Legacy wrapper execution failed",
            }

    async def run_all_tests(self) -> List[Dict[str, Any]]:
        """Run all integration tests via modern validator and append a summary."""
        try:
            results = await self._modern.run_all_tests()
            # Ensure final summary exists
            if not any(r.get("test_name") == "phase2d_integration_summary" for r in results if isinstance(r, dict)):
                passed_count = sum(1 for r in results if isinstance(r, dict) and r.get("passed"))
                total_count = len([r for r in results if isinstance(r, dict) and "test_name" in r])
                results.append(
                    {
                        "test_name": "phase2d_integration_summary",
                        "passed": passed_count == total_count and total_count > 0,
                        "total_tests": total_count,
                        "passed_tests": passed_count,
                        "failed_tests": max(0, total_count - passed_count),
                        "details": f"Phase 2D integration tests: {passed_count}/{total_count} passed",
                    }
                )
            return results
        except Exception as e:
            logger.error(f"Legacy run_all_tests failed: {e}")
            return [
                {
                    "test_name": "phase2d_integration_summary",
                    "passed": False,
                    "error": str(e),
                    "details": "Legacy wrapper execution failed",
                }
            ]


async def main() -> int:
    """Entry point for running the legacy Phase 2D validation."""
    logging.basicConfig(level=logging.INFO)
    validator = Phase2DIntegrationValidatorLegacy()
    results = await validator.run_all_tests()

    # Save results (backward-compatible)
    out = {
        "timestamp": datetime.utcnow().isoformat(),
        "phase": "2D",
        "title": "Real Integration & Testing (Legacy Shim)",
        "tests": results,
    }
    try:
        with open("phase2d_integration_legacy_report.json", "w") as f:
            json.dump(out, f, indent=2)
    except Exception:
        pass

    summary = next((r for r in results if isinstance(r, dict) and r.get("test_name") == "phase2d_integration_summary"), None)
    ok = bool(summary and summary.get("passed"))
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
