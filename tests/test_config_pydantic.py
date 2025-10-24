"""
Tests for Pydantic configuration classes.

This module tests the Pydantic-based configuration classes to ensure
proper validation, type checking, and error handling.
"""

import pytest
import tempfile
import json
from pathlib import Path
from pydantic import ValidationError

from chemforge.utils.config_utils import (
    ModelConfig, TrainingConfig, DataConfig, ADMETConfig, ConfigManager
)
from chemforge.utils.validators import (
    validate_ensemble_weights, validate_positive_weights, validate_model_type,
    validate_optimizer, validate_scheduler, validate_log_level, validate_gnn_type,
    validate_scaffold_types, validate_admet_properties, validate_data_splits,
    validate_pwa_buckets, validate_ensemble_models, validate_confidence_threshold,
    validate_feature_threshold, validate_morgan_radius, validate_morgan_bits,
    validate_dropout_rate, validate_learning_rate, validate_weight_decay,
    validate_gradient_clip, validate_patience, validate_epochs, validate_batch_size,
    validate_dimensions, validate_layers, validate_heads, validate_curvature_reg,
    validate_checkpoint_interval, validate_log_interval
)


class TestModelConfig:
    """Test ModelConfig Pydantic validation."""
    
    def test_valid_model_config(self):
        """Test valid model configuration."""
        config = ModelConfig(
            model_type='transformer',
            input_dim=100,
            output_dim=3,
            hidden_dim=256,
            num_layers=6,
            num_heads=8,
            dropout=0.1
        )
        
        assert config.model_type == 'transformer'
        assert config.input_dim == 100
        assert config.output_dim == 3
        assert config.hidden_dim == 256
        assert config.num_layers == 6
        assert config.num_heads == 8
        assert config.dropout == 0.1
    
    def test_invalid_input_dim(self):
        """Test invalid input dimension."""
        with pytest.raises(ValidationError):
            ModelConfig(input_dim=-1)
    
    def test_invalid_dropout(self):
        """Test invalid dropout rate."""
        with pytest.raises(ValidationError):
            ModelConfig(dropout=1.5)
    
    def test_default_values(self):
        """Test default values are set correctly."""
        config = ModelConfig()
        
        assert config.pwa_buckets == {'trivial': 1, 'fund': 5, 'adj': 2}
        assert config.ensemble_models == ['transformer', 'gnn']
        assert config.ensemble_weights == [0.5, 0.5]
    
    def test_to_dict_compatibility(self):
        """Test to_dict method compatibility."""
        config = ModelConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'model_type' in config_dict
        assert 'input_dim' in config_dict
    
    def test_from_dict_compatibility(self):
        """Test from_dict method compatibility."""
        config_dict = {
            'model_type': 'transformer',
            'input_dim': 100,
            'output_dim': 3,
            'hidden_dim': 256,
            'num_layers': 6,
            'num_heads': 8,
            'dropout': 0.1
        }
        
        config = ModelConfig.from_dict(config_dict)
        
        assert config.model_type == 'transformer'
        assert config.input_dim == 100


class TestTrainingConfig:
    """Test TrainingConfig Pydantic validation."""
    
    def test_valid_training_config(self):
        """Test valid training configuration."""
        config = TrainingConfig(
            epochs=100,
            batch_size=32,
            learning_rate=1e-3,
            weight_decay=1e-4,
            optimizer='adam',
            scheduler='cosine'
        )
        
        assert config.epochs == 100
        assert config.batch_size == 32
        assert config.learning_rate == 1e-3
        assert config.weight_decay == 1e-4
        assert config.optimizer == 'adam'
        assert config.scheduler == 'cosine'
    
    def test_invalid_epochs(self):
        """Test invalid epochs."""
        with pytest.raises(ValidationError):
            TrainingConfig(epochs=-1)
    
    def test_invalid_learning_rate(self):
        """Test invalid learning rate."""
        with pytest.raises(ValidationError):
            TrainingConfig(learning_rate=2.0)
    
    def test_invalid_data_splits(self):
        """Test invalid data splits."""
        with pytest.raises(ValidationError):
            TrainingConfig(train_split=0.5, val_split=0.3, test_split=0.3)
    
    def test_valid_data_splits(self):
        """Test valid data splits."""
        config = TrainingConfig(train_split=0.8, val_split=0.1, test_split=0.1)
        
        assert config.train_split == 0.8
        assert config.val_split == 0.1
        assert config.test_split == 0.1


class TestDataConfig:
    """Test DataConfig Pydantic validation."""
    
    def test_valid_data_config(self):
        """Test valid data configuration."""
        config = DataConfig(
            data_path='./data',
            train_path='./data/train.csv',
            val_path='./data/val.csv',
            test_path='./data/test.csv',
            normalize=True,
            feature_selection=True,
            feature_threshold=0.01,
            use_rdkit=True,
            use_morgan=True,
            morgan_radius=2,
            morgan_bits=2048
        )
        
        assert config.data_path == './data'
        assert config.normalize == True
        assert config.feature_threshold == 0.01
        assert config.morgan_radius == 2
        assert config.morgan_bits == 2048
    
    def test_invalid_feature_threshold(self):
        """Test invalid feature threshold."""
        with pytest.raises(ValidationError):
            DataConfig(feature_threshold=1.5)
    
    def test_invalid_morgan_radius(self):
        """Test invalid Morgan radius."""
        with pytest.raises(ValidationError):
            DataConfig(morgan_radius=10)
    
    def test_default_scaffold_types(self):
        """Test default scaffold types."""
        config = DataConfig()
        
        assert config.scaffold_types == ['trivial', 'fund', 'adj']


class TestADMETConfig:
    """Test ADMETConfig Pydantic validation."""
    
    def test_valid_admet_config(self):
        """Test valid ADMET configuration."""
        config = ADMETConfig(
            properties=['absorption', 'distribution', 'metabolism'],
            use_cns_mpo=True,
            admet_model_type='transformer',
            admet_hidden_dim=128,
            admet_num_layers=3,
            confidence_threshold=0.7,
            ensemble_predictions=True
        )
        
        assert config.properties == ['absorption', 'distribution', 'metabolism']
        assert config.use_cns_mpo == True
        assert config.admet_model_type == 'transformer'
        assert config.admet_hidden_dim == 128
        assert config.admet_num_layers == 3
        assert config.confidence_threshold == 0.7
        assert config.ensemble_predictions == True
    
    def test_invalid_confidence_threshold(self):
        """Test invalid confidence threshold."""
        with pytest.raises(ValidationError):
            ADMETConfig(confidence_threshold=1.5)
    
    def test_default_properties(self):
        """Test default properties."""
        config = ADMETConfig()
        
        assert config.properties == ['absorption', 'distribution', 'metabolism', 
                                   'excretion', 'toxicity']


class TestConfigManager:
    """Test ConfigManager with Pydantic validation."""
    
    def test_get_model_config(self):
        """Test getting model configuration."""
        manager = ConfigManager()
        
        # Test with default config
        config = manager.get_model_config()
        assert isinstance(config, ModelConfig)
        assert config.model_type == 'transformer'
    
    def test_set_model_config(self):
        """Test setting model configuration."""
        manager = ConfigManager()
        
        config = ModelConfig(model_type='gnn', input_dim=200)
        manager.set_model_config(config)
        
        retrieved_config = manager.get_model_config()
        assert retrieved_config.model_type == 'gnn'
        assert retrieved_config.input_dim == 200
    
    def test_validation_error_handling(self):
        """Test validation error handling."""
        manager = ConfigManager()
        
        # Set invalid configuration
        manager.set_config('model', {'input_dim': -1})
        
        with pytest.raises(ValidationError):
            manager.get_model_config()
    
    def test_config_file_operations(self):
        """Test configuration file operations."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                'model': {
                    'model_type': 'transformer',
                    'input_dim': 100,
                    'output_dim': 3,
                    'hidden_dim': 256,
                    'num_layers': 6,
                    'num_heads': 8,
                    'dropout': 0.1
                },
                'training': {
                    'epochs': 100,
                    'batch_size': 32,
                    'learning_rate': 1e-3,
                    'weight_decay': 1e-4,
                    'optimizer': 'adam',
                    'scheduler': 'cosine'
                }
            }
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            manager = ConfigManager(config_path)
            
            model_config = manager.get_model_config()
            assert model_config.model_type == 'transformer'
            assert model_config.input_dim == 100
            
            training_config = manager.get_training_config()
            assert training_config.epochs == 100
            assert training_config.batch_size == 32
            
        finally:
            Path(config_path).unlink()


class TestValidators:
    """Test custom validator functions."""
    
    def test_validate_ensemble_weights(self):
        """Test ensemble weights validation."""
        # Valid weights
        weights = [0.5, 0.3, 0.2]
        result = validate_ensemble_weights(weights)
        assert result == weights
        
        # Invalid weights
        with pytest.raises(ValueError):
            validate_ensemble_weights([0.5, 0.3, 0.3])
    
    def test_validate_positive_weights(self):
        """Test positive weights validation."""
        # Valid weights
        weights = [0.5, 0.3, 0.2]
        result = validate_positive_weights(weights)
        assert result == weights
        
        # Invalid weights
        with pytest.raises(ValueError):
            validate_positive_weights([0.5, -0.3, 0.2])
    
    def test_validate_model_type(self):
        """Test model type validation."""
        # Valid types
        for model_type in ['transformer', 'gnn', 'ensemble', 'linear', 'mlp']:
            result = validate_model_type(model_type)
            assert result == model_type
        
        # Invalid type
        with pytest.raises(ValueError):
            validate_model_type('invalid')
    
    def test_validate_optimizer(self):
        """Test optimizer validation."""
        # Valid optimizers
        for optimizer in ['adam', 'adamw', 'sgd', 'rmsprop', 'adagrad']:
            result = validate_optimizer(optimizer)
            assert result == optimizer
        
        # Invalid optimizer
        with pytest.raises(ValueError):
            validate_optimizer('invalid')
    
    def test_validate_scheduler(self):
        """Test scheduler validation."""
        # Valid schedulers
        for scheduler in ['cosine', 'linear', 'step', 'exponential', 'plateau']:
            result = validate_scheduler(scheduler)
            assert result == scheduler
        
        # Invalid scheduler
        with pytest.raises(ValueError):
            validate_scheduler('invalid')
    
    def test_validate_log_level(self):
        """Test log level validation."""
        # Valid levels
        for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            result = validate_log_level(level)
            assert result == level.upper()
        
        # Invalid level
        with pytest.raises(ValueError):
            validate_log_level('invalid')
    
    def test_validate_gnn_type(self):
        """Test GNN type validation."""
        # Valid types
        for gnn_type in ['gat', 'gcn', 'sage', 'gin', 'graphsage']:
            result = validate_gnn_type(gnn_type)
            assert result == gnn_type
        
        # Invalid type
        with pytest.raises(ValueError):
            validate_gnn_type('invalid')
    
    def test_validate_scaffold_types(self):
        """Test scaffold types validation."""
        # Valid types
        scaffold_types = ['trivial', 'fund', 'adj']
        result = validate_scaffold_types(scaffold_types)
        assert result == scaffold_types
        
        # Invalid type
        with pytest.raises(ValueError):
            validate_scaffold_types(['trivial', 'invalid'])
    
    def test_validate_admet_properties(self):
        """Test ADMET properties validation."""
        # Valid properties
        properties = ['absorption', 'distribution', 'metabolism']
        result = validate_admet_properties(properties)
        assert result == properties
        
        # Invalid property
        with pytest.raises(ValueError):
            validate_admet_properties(['absorption', 'invalid'])
    
    def test_validate_data_splits(self):
        """Test data splits validation."""
        # Valid splits
        result = validate_data_splits(0.8, 0.1, 0.1)
        assert result == (0.8, 0.1, 0.1)
        
        # Invalid splits
        with pytest.raises(ValueError):
            validate_data_splits(0.8, 0.1, 0.2)
    
    def test_validate_pwa_buckets(self):
        """Test PWA buckets validation."""
        # Valid buckets
        buckets = {'trivial': 1, 'fund': 5, 'adj': 2}
        result = validate_pwa_buckets(buckets)
        assert result == buckets
        
        # Invalid buckets
        with pytest.raises(ValueError):
            validate_pwa_buckets({'trivial': 1, 'fund': 5})
    
    def test_validate_ensemble_models(self):
        """Test ensemble models validation."""
        # Valid models
        models = ['transformer', 'gnn']
        result = validate_ensemble_models(models)
        assert result == models
        
        # Invalid model
        with pytest.raises(ValueError):
            validate_ensemble_models(['transformer', 'invalid'])
    
    def test_validate_confidence_threshold(self):
        """Test confidence threshold validation."""
        # Valid threshold
        result = validate_confidence_threshold(0.7)
        assert result == 0.7
        
        # Invalid threshold
        with pytest.raises(ValueError):
            validate_confidence_threshold(1.5)
    
    def test_validate_feature_threshold(self):
        """Test feature threshold validation."""
        # Valid threshold
        result = validate_feature_threshold(0.01)
        assert result == 0.01
        
        # Invalid threshold
        with pytest.raises(ValueError):
            validate_feature_threshold(1.5)
    
    def test_validate_morgan_radius(self):
        """Test Morgan radius validation."""
        # Valid radius
        result = validate_morgan_radius(2)
        assert result == 2
        
        # Invalid radius
        with pytest.raises(ValueError):
            validate_morgan_radius(10)
    
    def test_validate_morgan_bits(self):
        """Test Morgan bits validation."""
        # Valid bits
        result = validate_morgan_bits(2048)
        assert result == 2048
        
        # Invalid bits
        with pytest.raises(ValueError):
            validate_morgan_bits(-1)
    
    def test_validate_dropout_rate(self):
        """Test dropout rate validation."""
        # Valid rate
        result = validate_dropout_rate(0.1)
        assert result == 0.1
        
        # Invalid rate
        with pytest.raises(ValueError):
            validate_dropout_rate(1.5)
    
    def test_validate_learning_rate(self):
        """Test learning rate validation."""
        # Valid rate
        result = validate_learning_rate(1e-3)
        assert result == 1e-3
        
        # Invalid rate
        with pytest.raises(ValueError):
            validate_learning_rate(2.0)
    
    def test_validate_weight_decay(self):
        """Test weight decay validation."""
        # Valid decay
        result = validate_weight_decay(1e-4)
        assert result == 1e-4
        
        # Invalid decay
        with pytest.raises(ValueError):
            validate_weight_decay(-1e-4)
    
    def test_validate_gradient_clip(self):
        """Test gradient clipping validation."""
        # Valid clip
        result = validate_gradient_clip(1.0)
        assert result == 1.0
        
        # Invalid clip
        with pytest.raises(ValueError):
            validate_gradient_clip(-1.0)
    
    def test_validate_patience(self):
        """Test patience validation."""
        # Valid patience
        result = validate_patience(10)
        assert result == 10
        
        # Invalid patience
        with pytest.raises(ValueError):
            validate_patience(-1)
    
    def test_validate_epochs(self):
        """Test epochs validation."""
        # Valid epochs
        result = validate_epochs(100)
        assert result == 100
        
        # Invalid epochs
        with pytest.raises(ValueError):
            validate_epochs(-1)
    
    def test_validate_batch_size(self):
        """Test batch size validation."""
        # Valid batch size
        result = validate_batch_size(32)
        assert result == 32
        
        # Invalid batch size
        with pytest.raises(ValueError):
            validate_batch_size(-1)
    
    def test_validate_dimensions(self):
        """Test dimensions validation."""
        # Valid dimension
        result = validate_dimensions(256, 'hidden_dim')
        assert result == 256
        
        # Invalid dimension
        with pytest.raises(ValueError):
            validate_dimensions(-1, 'hidden_dim')
    
    def test_validate_layers(self):
        """Test layers validation."""
        # Valid layers
        result = validate_layers(6, 'num_layers')
        assert result == 6
        
        # Invalid layers
        with pytest.raises(ValueError):
            validate_layers(-1, 'num_layers')
    
    def test_validate_heads(self):
        """Test heads validation."""
        # Valid heads
        result = validate_heads(8)
        assert result == 8
        
        # Invalid heads
        with pytest.raises(ValueError):
            validate_heads(-1)
    
    def test_validate_curvature_reg(self):
        """Test curvature regularization validation."""
        # Valid regularization
        result = validate_curvature_reg(1e-5)
        assert result == 1e-5
        
        # Invalid regularization
        with pytest.raises(ValueError):
            validate_curvature_reg(-1e-5)
    
    def test_validate_checkpoint_interval(self):
        """Test checkpoint interval validation."""
        # Valid interval
        result = validate_checkpoint_interval(10)
        assert result == 10
        
        # Invalid interval
        with pytest.raises(ValueError):
            validate_checkpoint_interval(-1)
    
    def test_validate_log_interval(self):
        """Test log interval validation."""
        # Valid interval
        result = validate_log_interval(10)
        assert result == 10
        
        # Invalid interval
        with pytest.raises(ValueError):
            validate_log_interval(-1)


if __name__ == '__main__':
    pytest.main([__file__])
