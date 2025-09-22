"""
Script to identify market regimes from historical data and output regimes.json.
"""

import logging
import os
import sys

import yaml

# Add the parent directory to the path so we can import emp modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emp.data_pipeline.regime_identifier import MarketRegimeIdentifier

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise


def identify_regimes(config: dict, output_path: str = "regimes.json") -> bool:
    """
    Identify market regimes using the MarketRegimeIdentifier.

    Args:
        config: Configuration dictionary
        output_path: Path to output regimes.json file

    Returns:
        True if successful, False otherwise
    """
    try:
        # Get configuration parameters
        symbol = config["data"]["symbol"]
        start_year = config["data"]["start_year"]
        end_year = config["data"]["end_year"]
        processed_dir = config["data"]["processed_dir"]

        logger.info(f"Starting regime identification for {symbol}")
        logger.info(f"Analysis period: {start_year} - {end_year}")
        logger.info(f"Data directory: {processed_dir}")

        # Initialize regime identifier
        regime_identifier = MarketRegimeIdentifier(processed_dir, symbol)

        # Identify regimes
        regimes_data = regime_identifier.identify_regimes(
            start_year=start_year, end_year=end_year, output_path=output_path
        )

        # Print summary
        print_regime_summary(regimes_data)

        logger.info(f"Regime identification completed successfully")
        logger.info(f"Results saved to: {output_path}")

        return True

    except Exception as e:
        logger.error(f"Regime identification failed: {e}")
        return False


def print_regime_summary(regimes_data: dict):
    """
    Print a summary of the identified regimes.

    Args:
        regimes_data: Regimes data dictionary
    """
    print("\n" + "=" * 60)
    print("MARKET REGIME ANALYSIS SUMMARY")
    print("=" * 60)

    # Print metadata
    metadata = regimes_data["metadata"]
    print(f"Symbol: {metadata['symbol']}")
    print(f"Analysis Period: {metadata['analysis_start_year']} - {metadata['analysis_end_year']}")
    print(f"Analysis Date: {metadata['analysis_date']}")
    print(f"Rolling Window: {metadata['rolling_window_days']} days")
    print(f"Minimum Regime Duration: {metadata['min_regime_days']} days")

    # Print regime periods
    regime_periods = regimes_data["regime_periods"]
    print(f"\nTotal Regime Periods: {len(regime_periods)}")

    # Group by regime type
    regime_counts = {}
    for period in regime_periods:
        regime_type = period["regime_type"]
        if regime_type not in regime_counts:
            regime_counts[regime_type] = 0
        regime_counts[regime_type] += 1

    print("\nRegime Distribution:")
    for regime_type, count in regime_counts.items():
        print(f"  {regime_type.capitalize()}: {count} periods")

    # Print triathlon datasets
    triathlon_datasets = regimes_data["triathlon_datasets"]
    print(f"\nTriathlon Datasets (Longest Periods):")
    for regime_type, dataset in triathlon_datasets.items():
        print(f"  {regime_type.capitalize()}:")
        print(f"    Period: {dataset['start_date']} to {dataset['end_date']}")
        print(f"    Duration: {dataset['duration_days']} days")
        print(f"    Avg Volatility: {dataset['avg_volatility']:.3f}")
        print(f"    Avg Hurst: {dataset['avg_hurst']:.3f}")
        print(f"    Avg Kurtosis: {dataset['avg_kurtosis']:.3f}")

    # Print statistics
    stats = regimes_data["regime_statistics"]
    print(f"\nRegime Statistics:")
    for regime_type, regime_stats in stats.items():
        print(f"  {regime_type.capitalize()}:")
        print(f"    Count: {regime_stats['count']}")
        print(f"    Total Days: {regime_stats['total_days']}")
        print(f"    Avg Duration: {regime_stats['avg_duration']:.1f} days")
        print(f"    Avg Volatility: {regime_stats['avg_volatility']:.3f}")
        print(f"    Avg Hurst: {regime_stats['avg_hurst']:.3f}")
        print(f"    Avg Kurtosis: {regime_stats['avg_kurtosis']:.3f}")

    print("=" * 60)


def validate_regimes_data(regimes_data: dict) -> bool:
    """
    Validate that the regimes data contains the required information.

    Args:
        regimes_data: Regimes data dictionary

    Returns:
        True if valid, False otherwise
    """
    try:
        # Check required keys
        required_keys = ["metadata", "regime_periods", "triathlon_datasets", "regime_statistics"]
        for key in required_keys:
            if key not in regimes_data:
                logger.error(f"Missing required key: {key}")
                return False

        # Check that we have at least one regime period
        if len(regimes_data["regime_periods"]) == 0:
            logger.error("No regime periods identified")
            return False

        # Check that we have triathlon datasets
        triathlon_datasets = regimes_data["triathlon_datasets"]
        if len(triathlon_datasets) == 0:
            logger.error("No triathlon datasets identified")
            return False

        # Check that we have the three main regime types
        expected_regimes = ["trending", "ranging", "volatile"]
        found_regimes = list(triathlon_datasets.keys())

        missing_regimes = [regime for regime in expected_regimes if regime not in found_regimes]
        if missing_regimes:
            logger.warning(f"Missing expected regime types: {missing_regimes}")
            logger.warning(f"Found regime types: {found_regimes}")

        logger.info("Regimes data validation passed")
        return True

    except Exception as e:
        logger.error(f"Regimes data validation failed: {e}")
        return False


def main():
    """
    Main function to identify market regimes.
    """
    try:
        # Load configuration
        config = load_config()

        # Define output path
        output_path = "regimes.json"

        logger.info("Starting market regime identification")

        # Identify regimes
        success = identify_regimes(config, output_path)

        if success:
            # Load and validate the results
            import json

            with open(output_path, "r") as f:
                regimes_data = json.load(f)

            if validate_regimes_data(regimes_data):
                logger.info("Market regime identification completed successfully!")
                return 0
            else:
                logger.error("Regimes data validation failed")
                return 1
        else:
            logger.error("Regime identification failed")
            return 1

    except Exception as e:
        logger.error(f"Regime identification script failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
