#!/usr/bin/env python3
"""
Test script for configuration loading
"""

from config import load_config, validate_config, create_experiment_directories

def test_config():
    """Test configuration loading and validation"""
    
    print("Testing configuration system...")
    
    try:
        # Load configuration
        config = load_config("pilot_config.yaml")
        print("✓ Configuration loaded successfully")
        
        # Validate configuration
        validate_config(config)
        print("✓ Configuration validation passed")
        
        # Create directories
        directories = create_experiment_directories(config)
        print("✓ Experiment directories created")
        
        # Print key configuration details
        print(f"\nConfiguration Summary:")
        print(f"  Experiment Name: {config.experiment_name}")
        print(f"  Population Size: {config.population_size}")
        print(f"  Generations: {config.generations}")
        print(f"  Difficulty Levels: {config.adversarial_sweep_levels}")
        print(f"  Regime Datasets: {list(config.regime_datasets.keys())}")
        
        print("\n✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_config() 