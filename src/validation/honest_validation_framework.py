#!/usr/bin/env python3
"""
Honest Validation Framework - Phase 2A
======================================

Real validation framework that uses actual component testing and real data.

Refactor notes:
- Removed direct imports from data_integration, sensory, and trading.
- Depends on core ports only: MarketDataGateway and RegimeClassifier.
- Concrete implementations must be injected by orchestration.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, cast

import pandas as pd

from src.core.market_data import MarketDataGateway, NoOpMarketDataGateway
from src.core.regime import NoOpRegimeClassifier, RegimeClassifier, RegimeResult

if TYPE_CHECKING:
    from src.core.interfaces import DecisionGenome  # type: ignore
else:
    class DecisionGenome:  # minimal runtime placeholder
        pass


logger = logging.getLogger(__name__)


class HonestValidationResult:
    """Honest validation result with actual metrics"""

    def __init__(
        self,
        test_name: str,
        passed: bool,
        value: float,
        threshold: float,
        unit: str,
        details: str = "",
    ):
        self.test_name = test_name
        self.passed = passed
        self.value = value
        self.threshold = threshold
        self.unit = unit
        self.details = details
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "value": self.value,
            "threshold": self.threshold,
            "unit": self.unit,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


class HonestValidationFramework:
    """
    Honest validation framework that uses real components and actual data.
    Now decoupled from implementation packages via core ports.
    """

    def __init__(
        self,
        market_data_gateway: Optional[MarketDataGateway] = None,
        regime_classifier: Optional[RegimeClassifier] = None,
    ):
        self.results: List[HonestValidationResult] = []
        self.market_data: MarketDataGateway = market_data_gateway or NoOpMarketDataGateway()
        self.regime_classifier: RegimeClassifier = regime_classifier or NoOpRegimeClassifier()
        self.strategy_manager: Any | None = None  # Placeholder for StrategyManager or equivalent

    async def validate_data_integrity(self) -> HonestValidationResult:
        """Validate that real market data can be retrieved and processed"""
        try:
            # Test actual data retrieval
            start_time = time.time()
            data = self.market_data.fetch_data("EURUSD=X", period="1d", interval="1m")
            retrieval_time = time.time() - start_time

            if data is None:
                return HonestValidationResult(
                    test_name="data_integrity",
                    passed=False,
                    value=0.0,
                    threshold=1.0,
                    unit="boolean",
                    details="No data returned by MarketDataGateway",
                )

            # We expect a pandas DataFrame-like object
            if not isinstance(data, pd.DataFrame) or len(data) == 0:
                return HonestValidationResult(
                    test_name="data_integrity",
                    passed=False,
                    value=0.0,
                    threshold=1.0,
                    unit="boolean",
                    details="MarketDataGateway returned invalid or empty dataset",
                )

            # Validate data structure
            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in data.columns]

            if missing_columns:
                return HonestValidationResult(
                    test_name="data_integrity",
                    passed=False,
                    value=0.0,
                    threshold=1.0,
                    unit="boolean",
                    details=f"Missing required columns: {missing_columns}",
                )

            # Check data quality
            null_count = data[required_columns].isnull().to_numpy().sum()
            if null_count > 0:
                return HonestValidationResult(
                    test_name="data_integrity",
                    passed=False,
                    value=0.0,
                    threshold=1.0,
                    unit="boolean",
                    details=f"Found {null_count} null values in data",
                )

            return HonestValidationResult(
                test_name="data_integrity",
                passed=True,
                value=retrieval_time,
                threshold=5.0,
                unit="seconds",
                details=f"Successfully retrieved {len(data)} rows of EURUSD data in {retrieval_time:.2f}s",
            )

        except Exception as e:
            return HonestValidationResult(
                test_name="data_integrity",
                passed=False,
                value=0.0,
                threshold=1.0,
                unit="boolean",
                details=f"Data integrity validation failed: {str(e)}",
            )

    async def validate_regime_detection(self) -> HonestValidationResult:
        """Validate market regime detection with real data via the RegimeClassifier port"""
        try:
            # Get real market data
            data = self.market_data.fetch_data("EURUSD=X", period="1d", interval="1h")
            if not isinstance(data, pd.DataFrame) or len(data) < 20:
                return HonestValidationResult(
                    test_name="regime_detection",
                    passed=False,
                    value=0.0,
                    threshold=0.8,
                    unit="accuracy",
                    details="Insufficient or invalid data for regime detection",
                )

            # Test regime detection
            start_time = time.time()
            regime_result: Optional[RegimeResult] = await self.regime_classifier.detect_regime(
                cast(Mapping[str, object], data)
            )
            detection_time = time.time() - start_time

            if regime_result is None:
                return HonestValidationResult(
                    test_name="regime_detection",
                    passed=False,
                    value=0.0,
                    threshold=0.8,
                    unit="accuracy",
                    details="Regime detection returned None",
                )

            # Validate confidence
            confidence = float(regime_result.confidence or 0.0)
            passed = confidence >= 0.5  # Reasonable threshold for real data

            return HonestValidationResult(
                test_name="regime_detection",
                passed=passed,
                value=confidence,
                threshold=0.5,
                unit="confidence",
                details=f"Detected {regime_result.regime} with {confidence:.2f} confidence in {detection_time:.2f}s",
            )

        except Exception as e:
            return HonestValidationResult(
                test_name="regime_detection",
                passed=False,
                value=0.0,
                threshold=0.8,
                unit="accuracy",
                details=f"Regime detection failed: {str(e)}",
            )

    async def validate_strategy_integration(self) -> HonestValidationResult:
        """Validate strategy manager integration with real data (placeholder behavior if manager is None)"""
        try:
            # Create test strategy
            test_genome = cast(DecisionGenome, object())
            # Backfill expected attributes for legacy compatibility
            setattr(test_genome, "genome_id", "honest_test_strategy")
            setattr(test_genome, "generation", 1)
            setattr(test_genome, "fitness_score", 0.75)
            setattr(test_genome, "robustness_score", 0.8)

            if self.strategy_manager is None:
                return HonestValidationResult(
                    test_name="strategy_integration",
                    passed=False,
                    value=0.0,
                    threshold=1.0,
                    unit="boolean",
                    details="No strategy manager bound (DI required in orchestration)",
                )

            # Add strategy (implementation-specific)
            success = False
            try:
                success = bool(cast(Any, self.strategy_manager).add_strategy(test_genome))
            except Exception:
                success = False

            if not success:
                return HonestValidationResult(
                    test_name="strategy_integration",
                    passed=False,
                    value=0.0,
                    threshold=1.0,
                    unit="boolean",
                    details="Failed to add test strategy",
                )

            # Get real market data
            data = self.market_data.fetch_data("EURUSD=X", period="1d", interval="1h")
            if not isinstance(data, pd.DataFrame) or len(data) == 0:
                return HonestValidationResult(
                    test_name="strategy_integration",
                    passed=False,
                    value=0.0,
                    threshold=1.0,
                    unit="boolean",
                    details="No market data available",
                )

            # Create market data dict
            last = data.iloc[-1]
            market_data = {
                "symbol": "EURUSD",
                "timestamp": last.get("timestamp"),
                "open": last.get("open"),
                "high": last.get("high"),
                "low": last.get("low"),
                "close": last.get("close"),
                "volume": last.get("volume"),
            }

            # Test strategy evaluation
            start_time = time.time()
            try:
                signals = cast(Any, self.strategy_manager).evaluate_strategies(
                    "EURUSD", market_data
                )
            except Exception:
                signals = []
            evaluation_time = time.time() - start_time

            # Validate signals
            if not isinstance(signals, list):
                return HonestValidationResult(
                    test_name="strategy_integration",
                    passed=False,
                    value=0.0,
                    threshold=1.0,
                    unit="boolean",
                    details="Invalid signals format",
                )

            # Check if we got any signals
            signal_count = len(signals)
            passed = signal_count > 0

            return HonestValidationResult(
                test_name="strategy_integration",
                passed=passed,
                value=evaluation_time,
                threshold=2.0,
                unit="seconds",
                details=f"Strategy manager generated {signal_count} signals in {evaluation_time:.2f}s",
            )

        except Exception as e:
            return HonestValidationResult(
                test_name="strategy_integration",
                passed=False,
                value=0.0,
                threshold=1.0,
                unit="boolean",
                details=f"Strategy integration failed: {str(e)}",
            )

    async def validate_real_data_sources(self) -> HonestValidationResult:
        """Validate that the market data gateway is operational"""
        try:
            # Test sync fetch
            df = self.market_data.fetch_data("EURUSD=X", period="1d", interval="1h")
            sync_ok = isinstance(df, pd.DataFrame) and len(df) > 0

            # Test async fetch
            try:
                async_df = await cast(Any, self.market_data).get_market_data("EURUSD=X")
            except Exception:
                async_df = None
            async_ok = isinstance(async_df, pd.DataFrame) and len(async_df) > 0

            success = bool(sync_ok and async_ok)

            return HonestValidationResult(
                test_name="real_data_sources",
                passed=success,
                value=1.0 if success else 0.0,
                threshold=1.0,
                unit="boolean",
                details=f"MarketDataGateway sync: {'✅' if sync_ok else '❌'}, async: {'✅' if async_ok else '❌'}",
            )

        except Exception as e:
            return HonestValidationResult(
                test_name="real_data_sources",
                passed=False,
                value=0.0,
                threshold=1.0,
                unit="boolean",
                details=f"Real data sources validation failed: {str(e)}",
            )

    async def validate_no_synthetic_data(self) -> HonestValidationResult:
        """Validate that we're not using synthetic data (heuristics over gateway data)"""
        try:
            # Check if we're using real data
            data = self.market_data.fetch_data("EURUSD=X", period="1d", interval="1h")

            if not isinstance(data, pd.DataFrame) or len(data) == 0:
                return HonestValidationResult(
                    test_name="no_synthetic_data",
                    passed=False,
                    value=0.0,
                    threshold=1.0,
                    unit="boolean",
                    details="No real data available - system may be using synthetic data",
                )

            # Check for obvious synthetic patterns
            # Real data should have some randomness and not be perfectly smooth
            price_changes = data["close"].pct_change().dropna()
            volatility = float(price_changes.std())

            # Real EURUSD data typically has volatility > 0.0001
            is_real = volatility > 0.0001

            return HonestValidationResult(
                test_name="no_synthetic_data",
                passed=is_real,
                value=volatility,
                threshold=0.0001,
                unit="volatility",
                details=f"Data volatility: {volatility:.6f} - {'Real data' if is_real else 'Synthetic data detected'}",
            )

        except Exception as e:
            return HonestValidationResult(
                test_name="no_synthetic_data",
                passed=False,
                value=0.0,
                threshold=1.0,
                unit="boolean",
                details=f"Cannot determine data source: {str(e)}",
            )

    async def run_all_validations(self) -> Dict[str, Any]:
        """Run all honest validations"""
        logger.info("Starting honest validation framework...")

        # Run all validation tests
        validations = [
            self.validate_data_integrity(),
            self.validate_regime_detection(),
            self.validate_strategy_integration(),
            self.validate_real_data_sources(),
            self.validate_no_synthetic_data(),
        ]

        # Execute all validations
        results = await asyncio.gather(*validations)

        # Calculate summary
        passed = sum(1 for r in results if r.passed)
        total = len(results)

        # Create final report
        report = {
            "timestamp": datetime.now().isoformat(),
            "framework": "Honest Validation Framework",
            "version": "1.1.0",
            "total_tests": total,
            "passed_tests": passed,
            "failed_tests": total - passed,
            "success_rate": passed / total if total > 0 else 0.0,
            "results": [r.to_dict() for r in results],
            "summary": {
                "status": "PASSED" if passed == total else "FAILED",
                "message": f"{passed}/{total} validations passed",
            },
        }

        # Save results
        with open("honest_validation_report.json", "w") as f:
            json.dump(report, f, indent=2)

        return report

    def print_report(self, report: Dict[str, Any]) -> None:
        """Print comprehensive validation report"""
        print("\n" + "=" * 80)
        print("HONEST VALIDATION FRAMEWORK REPORT")
        print("=" * 80)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Framework: {report['framework']} v{report['version']}")
        print(f"Status: {report['summary']['status']}")
        print(f"Success Rate: {report['success_rate']:.2%}")
        print()

        print("VALIDATION RESULTS:")
        print("-" * 40)
        for result in report["results"]:
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            print(f"{status} {result['test_name']}: {result['details']}")
            print(f"  Value: {result['value']} {result['unit']}")
            print(f"  Threshold: {result['threshold']} {result['unit']}")
            print()

        print("=" * 80)
        print(report["summary"]["message"])
        print("=" * 80)


async def main() -> None:
    """Run honest validation framework"""
    logging.basicConfig(level=logging.INFO)

    framework = HonestValidationFramework()
    report = await framework.run_all_validations()
    framework.print_report(report)

    # Exit with appropriate code
    import sys

    sys.exit(0 if report["success_rate"] >= 0.8 else 1)


if __name__ == "__main__":
    asyncio.run(main())
