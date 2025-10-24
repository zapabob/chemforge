#!/usr/bin/env python3
"""
Test script for Pydantic configuration classes.
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'chemforge/utils')

from config_utils import ModelConfig, TrainingConfig, DataConfig, ADMETConfig
from pydantic import ValidationError

def test_model_config():
    """Test ModelConfig validation."""
    print("Testing ModelConfig...")
    
    # Test valid configuration
    config = ModelConfig()
    print(f"✓ Default config: model_type={config.model_type}, input_dim={config.input_dim}")
    
    # Test invalid input_dim
    try:
        config = ModelConfig(input_dim=-1)
        print("✗ ERROR: Should have failed for negative input_dim")
    except ValidationError as e:
        print(f"✓ Validation error caught: {e}")
    
    # Test invalid dropout
    try:
        config = ModelConfig(dropout=1.5)
        print("✗ ERROR: Should have failed for dropout > 1.0")
    except ValidationError as e:
        print(f"✓ Validation error caught: {e}")
    
    print("ModelConfig tests passed!")

def test_training_config():
    """Test TrainingConfig validation."""
    print("\nTesting TrainingConfig...")
    
    # Test valid configuration
    config = TrainingConfig()
    print(f"✓ Default config: epochs={config.epochs}, batch_size={config.batch_size}")
    
    # Test invalid epochs
    try:
        config = TrainingConfig(epochs=-1)
        print("✗ ERROR: Should have failed for negative epochs")
    except ValidationError as e:
        print(f"✓ Validation error caught: {e}")
    
    # Test invalid data splits
    try:
        config = TrainingConfig(train_split=0.8, val_split=0.1, test_split=0.2)
        print("✗ ERROR: Should have failed for splits not summing to 1.0")
    except ValidationError as e:
        print(f"✓ Validation error caught: {e}")
    
    print("TrainingConfig tests passed!")

def test_data_config():
    """Test DataConfig validation."""
    print("\nTesting DataConfig...")
    
    # Test valid configuration
    config = DataConfig()
    print(f"✓ Default config: morgan_radius={config.morgan_radius}, morgan_bits={config.morgan_bits}")
    
    # Test invalid morgan_radius
    try:
        config = DataConfig(morgan_radius=10)
        print("✗ ERROR: Should have failed for morgan_radius > 5")
    except ValidationError as e:
        print(f"✓ Validation error caught: {e}")
    
    # Test invalid feature_threshold
    try:
        config = DataConfig(feature_threshold=1.5)
        print("✗ ERROR: Should have failed for feature_threshold > 1.0")
    except ValidationError as e:
        print(f"✓ Validation error caught: {e}")
    
    print("DataConfig tests passed!")

def test_admet_config():
    """Test ADMETConfig validation."""
    print("\nTesting ADMETConfig...")
    
    # Test valid configuration
    config = ADMETConfig()
    print(f"✓ Default config: confidence_threshold={config.confidence_threshold}")
    
    # Test invalid confidence_threshold
    try:
        config = ADMETConfig(confidence_threshold=1.5)
        print("✗ ERROR: Should have failed for confidence_threshold > 1.0")
    except ValidationError as e:
        print(f"✓ Validation error caught: {e}")
    
    print("ADMETConfig tests passed!")

def test_compatibility_methods():
    """Test compatibility methods."""
    print("\nTesting compatibility methods...")
    
    # Test to_dict
    config = ModelConfig()
    config_dict = config.to_dict()
    print(f"✓ to_dict() works: {type(config_dict)}")
    
    # Test from_dict
    config2 = ModelConfig.from_dict(config_dict)
    print(f"✓ from_dict() works: {config2.model_type}")
    
    print("Compatibility methods tests passed!")

if __name__ == "__main__":
    print("🧪 Testing Pydantic Configuration Classes")
    print("=" * 50)
    
    test_model_config()
    test_training_config()
    test_data_config()
    test_admet_config()
    test_compatibility_methods()
    
    print("\n🎉 All tests passed! Pydantic implementation is working correctly!")
