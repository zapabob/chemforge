"""
Unit tests for configuration utilities.
"""

import unittest
import tempfile
import os
import json
import yaml
from pathlib import Path

from chemforge.utils.config_utils import (
    ModelConfig,
    TrainingConfig,
    DataConfig,
    ADMETConfig,
    ConfigManager
)


class TestModelConfig(unittest.TestCase):
    """Test ModelConfig class."""
    
    def test_init_default(self):
        """Test ModelConfig initialization with default values."""
        config = ModelConfig()
        
        self.assertEqual(config.model_type, 'transformer')
        self.assertEqual(config.input_dim, 100)
        self.assertEqual(config.output_dim, 3)
        self.assertEqual(config.hidden_dim, 256)
        self.assertEqual(config.num_layers, 6)
        self.assertEqual(config.num_heads, 8)
        self.assertEqual(config.dropout, 0.1)
        self.assertTrue(config.use_pwa_pet)
        self.assertIsNotNone(config.pwa_buckets)
        self.assertTrue(config.use_rope)
        self.assertTrue(config.use_pet)
        self.assertEqual(config.pet_curv_reg, 1e-5)
    
    def test_init_custom(self):
        """Test ModelConfig initialization with custom values."""
        config = ModelConfig(
            model_type='gnn',
            input_dim=200,
            output_dim=5,
            hidden_dim=512,
            num_layers=8,
            num_heads=16,
            dropout=0.2,
            use_pwa_pet=False,
            pwa_buckets={'trivial': 2, 'fund': 10, 'adj': 4},
            use_rope=False,
            use_pet=False,
            pet_curv_reg=1e-4
        )
        
        self.assertEqual(config.model_type, 'gnn')
        self.assertEqual(config.input_dim, 200)
        self.assertEqual(config.output_dim, 5)
        self.assertEqual(config.hidden_dim, 512)
        self.assertEqual(config.num_layers, 8)
        self.assertEqual(config.num_heads, 16)
        self.assertEqual(config.dropout, 0.2)
        self.assertFalse(config.use_pwa_pet)
        self.assertEqual(config.pwa_buckets, {'trivial': 2, 'fund': 10, 'adj': 4})
        self.assertFalse(config.use_rope)
        self.assertFalse(config.use_pet)
        self.assertEqual(config.pet_curv_reg, 1e-4)
    
    def test_post_init(self):
        """Test ModelConfig post-initialization processing."""
        config = ModelConfig()
        
        # Check default values are set
        self.assertIsNotNone(config.pwa_buckets)
        self.assertIsNotNone(config.ensemble_models)
        self.assertIsNotNone(config.ensemble_weights)
        
        # Check ensemble weights sum to 1.0
        self.assertAlmostEqual(sum(config.ensemble_weights), 1.0, places=6)
    
    def test_to_dict(self):
        """Test ModelConfig to_dict method."""
        config = ModelConfig()
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['model_type'], 'transformer')
        self.assertEqual(config_dict['input_dim'], 100)
        self.assertEqual(config_dict['output_dim'], 3)
    
    def test_from_dict(self):
        """Test ModelConfig from_dict method."""
        config_dict = {
            'model_type': 'gnn',
            'input_dim': 200,
            'output_dim': 5,
            'hidden_dim': 512,
            'num_layers': 8,
            'num_heads': 16,
            'dropout': 0.2,
            'use_pwa_pet': False,
            'pwa_buckets': {'trivial': 2, 'fund': 10, 'adj': 4},
            'use_rope': False,
            'use_pet': False,
            'pet_curv_reg': 1e-4
        }
        
        config = ModelConfig.from_dict(config_dict)
        
        self.assertEqual(config.model_type, 'gnn')
        self.assertEqual(config.input_dim, 200)
        self.assertEqual(config.output_dim, 5)
        self.assertEqual(config.hidden_dim, 512)
        self.assertEqual(config.num_layers, 8)
        self.assertEqual(config.num_heads, 16)
        self.assertEqual(config.dropout, 0.2)
        self.assertFalse(config.use_pwa_pet)
        self.assertEqual(config.pwa_buckets, {'trivial': 2, 'fund': 10, 'adj': 4})
        self.assertFalse(config.use_rope)
        self.assertFalse(config.use_pet)
        self.assertEqual(config.pet_curv_reg, 1e-4)


class TestTrainingConfig(unittest.TestCase):
    """Test TrainingConfig class."""
    
    def test_init_default(self):
        """Test TrainingConfig initialization with default values."""
        config = TrainingConfig()
        
        self.assertEqual(config.epochs, 100)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.learning_rate, 1e-3)
        self.assertEqual(config.weight_decay, 1e-4)
        self.assertEqual(config.optimizer, 'adam')
        self.assertEqual(config.scheduler, 'cosine')
        self.assertEqual(config.train_split, 0.8)
        self.assertEqual(config.val_split, 0.1)
        self.assertEqual(config.test_split, 0.1)
        self.assertEqual(config.random_seed, 42)
        self.assertTrue(config.use_amp)
        self.assertEqual(config.gradient_clip, 1.0)
        self.assertTrue(config.early_stopping)
        self.assertEqual(config.patience, 10)
        self.assertEqual(config.checkpoint_interval, 10)
        self.assertTrue(config.save_best)
        self.assertTrue(config.save_last)
        self.assertEqual(config.log_interval, 10)
        self.assertEqual(config.log_level, 'INFO')
    
    def test_init_custom(self):
        """Test TrainingConfig initialization with custom values."""
        config = TrainingConfig(
            epochs=200,
            batch_size=64,
            learning_rate=1e-4,
            weight_decay=1e-5,
            optimizer='sgd',
            scheduler='step',
            train_split=0.7,
            val_split=0.15,
            test_split=0.15,
            random_seed=123,
            use_amp=False,
            gradient_clip=0.5,
            early_stopping=False,
            patience=20,
            checkpoint_interval=20,
            save_best=False,
            save_last=False,
            log_interval=20,
            log_level='DEBUG'
        )
        
        self.assertEqual(config.epochs, 200)
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.learning_rate, 1e-4)
        self.assertEqual(config.weight_decay, 1e-5)
        self.assertEqual(config.optimizer, 'sgd')
        self.assertEqual(config.scheduler, 'step')
        self.assertEqual(config.train_split, 0.7)
        self.assertEqual(config.val_split, 0.15)
        self.assertEqual(config.test_split, 0.15)
        self.assertEqual(config.random_seed, 123)
        self.assertFalse(config.use_amp)
        self.assertEqual(config.gradient_clip, 0.5)
        self.assertFalse(config.early_stopping)
        self.assertEqual(config.patience, 20)
        self.assertEqual(config.checkpoint_interval, 20)
        self.assertFalse(config.save_best)
        self.assertFalse(config.save_last)
        self.assertEqual(config.log_interval, 20)
        self.assertEqual(config.log_level, 'DEBUG')
    
    def test_to_dict(self):
        """Test TrainingConfig to_dict method."""
        config = TrainingConfig()
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['epochs'], 100)
        self.assertEqual(config_dict['batch_size'], 32)
        self.assertEqual(config_dict['learning_rate'], 1e-3)
    
    def test_from_dict(self):
        """Test TrainingConfig from_dict method."""
        config_dict = {
            'epochs': 200,
            'batch_size': 64,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'optimizer': 'sgd',
            'scheduler': 'step',
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15,
            'random_seed': 123,
            'use_amp': False,
            'gradient_clip': 0.5,
            'early_stopping': False,
            'patience': 20,
            'checkpoint_interval': 20,
            'save_best': False,
            'save_last': False,
            'log_interval': 20,
            'log_level': 'DEBUG'
        }
        
        config = TrainingConfig.from_dict(config_dict)
        
        self.assertEqual(config.epochs, 200)
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.learning_rate, 1e-4)
        self.assertEqual(config.weight_decay, 1e-5)
        self.assertEqual(config.optimizer, 'sgd')
        self.assertEqual(config.scheduler, 'step')
        self.assertEqual(config.train_split, 0.7)
        self.assertEqual(config.val_split, 0.15)
        self.assertEqual(config.test_split, 0.15)
        self.assertEqual(config.random_seed, 123)
        self.assertFalse(config.use_amp)
        self.assertEqual(config.gradient_clip, 0.5)
        self.assertFalse(config.early_stopping)
        self.assertEqual(config.patience, 20)
        self.assertEqual(config.checkpoint_interval, 20)
        self.assertFalse(config.save_best)
        self.assertFalse(config.save_last)
        self.assertEqual(config.log_interval, 20)
        self.assertEqual(config.log_level, 'DEBUG')


class TestDataConfig(unittest.TestCase):
    """Test DataConfig class."""
    
    def test_init_default(self):
        """Test DataConfig initialization with default values."""
        config = DataConfig()
        
        self.assertEqual(config.data_path, './data')
        self.assertEqual(config.train_path, './data/train.csv')
        self.assertEqual(config.val_path, './data/val.csv')
        self.assertEqual(config.test_path, './data/test.csv')
        self.assertTrue(config.normalize)
        self.assertTrue(config.feature_selection)
        self.assertEqual(config.feature_threshold, 0.01)
        self.assertTrue(config.use_rdkit)
        self.assertTrue(config.use_morgan)
        self.assertEqual(config.morgan_radius, 2)
        self.assertEqual(config.morgan_bits, 2048)
        self.assertTrue(config.use_scaffold)
        self.assertIsNotNone(config.scaffold_types)
    
    def test_init_custom(self):
        """Test DataConfig initialization with custom values."""
        config = DataConfig(
            data_path='/custom/data',
            train_path='/custom/train.csv',
            val_path='/custom/val.csv',
            test_path='/custom/test.csv',
            normalize=False,
            feature_selection=False,
            feature_threshold=0.05,
            use_rdkit=False,
            use_morgan=False,
            morgan_radius=3,
            morgan_bits=4096,
            use_scaffold=False,
            scaffold_types=['trivial', 'fund']
        )
        
        self.assertEqual(config.data_path, '/custom/data')
        self.assertEqual(config.train_path, '/custom/train.csv')
        self.assertEqual(config.val_path, '/custom/val.csv')
        self.assertEqual(config.test_path, '/custom/test.csv')
        self.assertFalse(config.normalize)
        self.assertFalse(config.feature_selection)
        self.assertEqual(config.feature_threshold, 0.05)
        self.assertFalse(config.use_rdkit)
        self.assertFalse(config.use_morgan)
        self.assertEqual(config.morgan_radius, 3)
        self.assertEqual(config.morgan_bits, 4096)
        self.assertFalse(config.use_scaffold)
        self.assertEqual(config.scaffold_types, ['trivial', 'fund'])
    
    def test_post_init(self):
        """Test DataConfig post-initialization processing."""
        config = DataConfig()
        
        # Check default scaffold types are set
        self.assertIsNotNone(config.scaffold_types)
        self.assertIn('trivial', config.scaffold_types)
        self.assertIn('fund', config.scaffold_types)
        self.assertIn('adj', config.scaffold_types)
    
    def test_to_dict(self):
        """Test DataConfig to_dict method."""
        config = DataConfig()
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['data_path'], './data')
        self.assertEqual(config_dict['train_path'], './data/train.csv')
        self.assertEqual(config_dict['val_path'], './data/val.csv')
        self.assertEqual(config_dict['test_path'], './data/test.csv')
    
    def test_from_dict(self):
        """Test DataConfig from_dict method."""
        config_dict = {
            'data_path': '/custom/data',
            'train_path': '/custom/train.csv',
            'val_path': '/custom/val.csv',
            'test_path': '/custom/test.csv',
            'normalize': False,
            'feature_selection': False,
            'feature_threshold': 0.05,
            'use_rdkit': False,
            'use_morgan': False,
            'morgan_radius': 3,
            'morgan_bits': 4096,
            'use_scaffold': False,
            'scaffold_types': ['trivial', 'fund']
        }
        
        config = DataConfig.from_dict(config_dict)
        
        self.assertEqual(config.data_path, '/custom/data')
        self.assertEqual(config.train_path, '/custom/train.csv')
        self.assertEqual(config.val_path, '/custom/val.csv')
        self.assertEqual(config.test_path, '/custom/test.csv')
        self.assertFalse(config.normalize)
        self.assertFalse(config.feature_selection)
        self.assertEqual(config.feature_threshold, 0.05)
        self.assertFalse(config.use_rdkit)
        self.assertFalse(config.use_morgan)
        self.assertEqual(config.morgan_radius, 3)
        self.assertEqual(config.morgan_bits, 4096)
        self.assertFalse(config.use_scaffold)
        self.assertEqual(config.scaffold_types, ['trivial', 'fund'])


class TestADMETConfig(unittest.TestCase):
    """Test ADMETConfig class."""
    
    def test_init_default(self):
        """Test ADMETConfig initialization with default values."""
        config = ADMETConfig()
        
        self.assertIsNotNone(config.properties)
        self.assertTrue(config.use_cns_mpo)
        self.assertEqual(config.admet_model_type, 'transformer')
        self.assertEqual(config.admet_hidden_dim, 128)
        self.assertEqual(config.admet_num_layers, 3)
        self.assertEqual(config.confidence_threshold, 0.7)
        self.assertTrue(config.ensemble_predictions)
    
    def test_init_custom(self):
        """Test ADMETConfig initialization with custom values."""
        config = ADMETConfig(
            properties=['absorption', 'distribution'],
            use_cns_mpo=False,
            admet_model_type='gnn',
            admet_hidden_dim=256,
            admet_num_layers=5,
            confidence_threshold=0.8,
            ensemble_predictions=False
        )
        
        self.assertEqual(config.properties, ['absorption', 'distribution'])
        self.assertFalse(config.use_cns_mpo)
        self.assertEqual(config.admet_model_type, 'gnn')
        self.assertEqual(config.admet_hidden_dim, 256)
        self.assertEqual(config.admet_num_layers, 5)
        self.assertEqual(config.confidence_threshold, 0.8)
        self.assertFalse(config.ensemble_predictions)
    
    def test_post_init(self):
        """Test ADMETConfig post-initialization processing."""
        config = ADMETConfig()
        
        # Check default properties are set
        self.assertIsNotNone(config.properties)
        self.assertIn('absorption', config.properties)
        self.assertIn('distribution', config.properties)
        self.assertIn('metabolism', config.properties)
        self.assertIn('excretion', config.properties)
        self.assertIn('toxicity', config.properties)
    
    def test_to_dict(self):
        """Test ADMETConfig to_dict method."""
        config = ADMETConfig()
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertIsNotNone(config_dict['properties'])
        self.assertTrue(config_dict['use_cns_mpo'])
        self.assertEqual(config_dict['admet_model_type'], 'transformer')
    
    def test_from_dict(self):
        """Test ADMETConfig from_dict method."""
        config_dict = {
            'properties': ['absorption', 'distribution'],
            'use_cns_mpo': False,
            'admet_model_type': 'gnn',
            'admet_hidden_dim': 256,
            'admet_num_layers': 5,
            'confidence_threshold': 0.8,
            'ensemble_predictions': False
        }
        
        config = ADMETConfig.from_dict(config_dict)
        
        self.assertEqual(config.properties, ['absorption', 'distribution'])
        self.assertFalse(config.use_cns_mpo)
        self.assertEqual(config.admet_model_type, 'gnn')
        self.assertEqual(config.admet_hidden_dim, 256)
        self.assertEqual(config.admet_num_layers, 5)
        self.assertEqual(config.confidence_threshold, 0.8)
        self.assertFalse(config.ensemble_predictions)


class TestConfigManager(unittest.TestCase):
    """Test ConfigManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.json')
        self.config_manager = ConfigManager()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test ConfigManager initialization."""
        self.assertIsNotNone(self.config_manager.logger)
        self.assertEqual(self.config_manager.config, {})
        self.assertIsNone(self.config_manager.config_path)
    
    def test_init_with_path(self):
        """Test ConfigManager initialization with config path."""
        config_manager = ConfigManager(self.config_path)
        self.assertEqual(config_manager.config_path, Path(self.config_path))
    
    def test_create_default_config(self):
        """Test default configuration creation."""
        default_config = self.config_manager.create_default_config()
        
        self.assertIsInstance(default_config, dict)
        self.assertIn('model', default_config)
        self.assertIn('training', default_config)
        self.assertIn('data', default_config)
        self.assertIn('admet', default_config)
        self.assertIn('metadata', default_config)
        
        # Check model config
        model_config = default_config['model']
        self.assertEqual(model_config['model_type'], 'transformer')
        self.assertEqual(model_config['input_dim'], 100)
        self.assertEqual(model_config['output_dim'], 3)
        
        # Check training config
        training_config = default_config['training']
        self.assertEqual(training_config['epochs'], 100)
        self.assertEqual(training_config['batch_size'], 32)
        self.assertEqual(training_config['learning_rate'], 1e-3)
        
        # Check data config
        data_config = default_config['data']
        self.assertEqual(data_config['data_path'], './data')
        self.assertEqual(data_config['train_path'], './data/train.csv')
        
        # Check ADMET config
        admet_config = default_config['admet']
        self.assertIsNotNone(admet_config['properties'])
        self.assertTrue(admet_config['use_cns_mpo'])
        
        # Check metadata
        metadata = default_config['metadata']
        self.assertIn('created_at', metadata)
        self.assertEqual(metadata['version'], '1.0.0')
        self.assertEqual(metadata['description'], 'ChemForge default configuration')
    
    def test_save_config_json(self):
        """Test configuration saving to JSON."""
        self.config_manager.create_default_config()
        result_path = self.config_manager.save_config(self.config_path)
        
        self.assertEqual(result_path, Path(self.config_path))
        self.assertTrue(Path(self.config_path).exists())
        
        # Verify content
        with open(self.config_path, 'r') as f:
            saved_config = json.load(f)
        
        self.assertEqual(saved_config['model']['model_type'], 'transformer')
        self.assertEqual(saved_config['training']['epochs'], 100)
    
    def test_save_config_yaml(self):
        """Test configuration saving to YAML."""
        yaml_path = os.path.join(self.temp_dir, 'test_config.yaml')
        self.config_manager.create_default_config()
        result_path = self.config_manager.save_config(yaml_path)
        
        self.assertEqual(result_path, Path(yaml_path))
        self.assertTrue(Path(yaml_path).exists())
    
    def test_load_config_json(self):
        """Test configuration loading from JSON."""
        # Create test configuration
        test_config = {
            'model': {'model_type': 'gnn', 'input_dim': 200},
            'training': {'epochs': 200, 'batch_size': 64}
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(test_config, f)
        
        # Load configuration
        loaded_config = self.config_manager.load_config(self.config_path)
        
        self.assertEqual(loaded_config['model']['model_type'], 'gnn')
        self.assertEqual(loaded_config['model']['input_dim'], 200)
        self.assertEqual(loaded_config['training']['epochs'], 200)
        self.assertEqual(loaded_config['training']['batch_size'], 64)
    
    def test_load_config_yaml(self):
        """Test configuration loading from YAML."""
        yaml_path = os.path.join(self.temp_dir, 'test_config.yaml')
        
        # Create test configuration
        test_config = {
            'model': {'model_type': 'gnn', 'input_dim': 200},
            'training': {'epochs': 200, 'batch_size': 64}
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(test_config, f)
        
        # Load configuration
        loaded_config = self.config_manager.load_config(yaml_path)
        
        self.assertEqual(loaded_config['model']['model_type'], 'gnn')
        self.assertEqual(loaded_config['model']['input_dim'], 200)
        self.assertEqual(loaded_config['training']['epochs'], 200)
        self.assertEqual(loaded_config['training']['batch_size'], 64)
    
    def test_load_config_nonexistent(self):
        """Test configuration loading from nonexistent file."""
        nonexistent_path = os.path.join(self.temp_dir, 'nonexistent.json')
        with self.assertRaises(FileNotFoundError):
            self.config_manager.load_config(nonexistent_path)
    
    def test_get_config(self):
        """Test configuration value retrieval."""
        self.config_manager.create_default_config()
        
        # Test simple key
        model_type = self.config_manager.get_config('model.model_type')
        self.assertEqual(model_type, 'transformer')
        
        # Test nested key
        input_dim = self.config_manager.get_config('model.input_dim')
        self.assertEqual(input_dim, 100)
        
        # Test default value
        nonexistent = self.config_manager.get_config('nonexistent.key', 'default')
        self.assertEqual(nonexistent, 'default')
    
    def test_set_config(self):
        """Test configuration value setting."""
        self.config_manager.create_default_config()
        
        # Test simple key
        self.config_manager.set_config('model.model_type', 'gnn')
        model_type = self.config_manager.get_config('model.model_type')
        self.assertEqual(model_type, 'gnn')
        
        # Test nested key
        self.config_manager.set_config('model.input_dim', 200)
        input_dim = self.config_manager.get_config('model.input_dim')
        self.assertEqual(input_dim, 200)
        
        # Test new key creation
        self.config_manager.set_config('new.key', 'value')
        new_value = self.config_manager.get_config('new.key')
        self.assertEqual(new_value, 'value')
    
    def test_update_config(self):
        """Test configuration update with dictionary."""
        self.config_manager.create_default_config()
        
        updates = {
            'model.model_type': 'gnn',
            'model.input_dim': 200,
            'training.epochs': 200,
            'training.batch_size': 64
        }
        
        self.config_manager.update_config(updates)
        
        self.assertEqual(self.config_manager.get_config('model.model_type'), 'gnn')
        self.assertEqual(self.config_manager.get_config('model.input_dim'), 200)
        self.assertEqual(self.config_manager.get_config('training.epochs'), 200)
        self.assertEqual(self.config_manager.get_config('training.batch_size'), 64)
    
    def test_get_model_config(self):
        """Test model configuration retrieval."""
        self.config_manager.create_default_config()
        
        model_config = self.config_manager.get_model_config()
        
        self.assertIsInstance(model_config, ModelConfig)
        self.assertEqual(model_config.model_type, 'transformer')
        self.assertEqual(model_config.input_dim, 100)
        self.assertEqual(model_config.output_dim, 3)
    
    def test_set_model_config(self):
        """Test model configuration setting."""
        custom_model_config = ModelConfig(
            model_type='gnn',
            input_dim=200,
            output_dim=5,
            hidden_dim=512
        )
        
        self.config_manager.set_model_config(custom_model_config)
        
        retrieved_config = self.config_manager.get_model_config()
        self.assertEqual(retrieved_config.model_type, 'gnn')
        self.assertEqual(retrieved_config.input_dim, 200)
        self.assertEqual(retrieved_config.output_dim, 5)
        self.assertEqual(retrieved_config.hidden_dim, 512)
    
    def test_get_training_config(self):
        """Test training configuration retrieval."""
        self.config_manager.create_default_config()
        
        training_config = self.config_manager.get_training_config()
        
        self.assertIsInstance(training_config, TrainingConfig)
        self.assertEqual(training_config.epochs, 100)
        self.assertEqual(training_config.batch_size, 32)
        self.assertEqual(training_config.learning_rate, 1e-3)
    
    def test_set_training_config(self):
        """Test training configuration setting."""
        custom_training_config = TrainingConfig(
            epochs=200,
            batch_size=64,
            learning_rate=1e-4
        )
        
        self.config_manager.set_training_config(custom_training_config)
        
        retrieved_config = self.config_manager.get_training_config()
        self.assertEqual(retrieved_config.epochs, 200)
        self.assertEqual(retrieved_config.batch_size, 64)
        self.assertEqual(retrieved_config.learning_rate, 1e-4)
    
    def test_get_data_config(self):
        """Test data configuration retrieval."""
        self.config_manager.create_default_config()
        
        data_config = self.config_manager.get_data_config()
        
        self.assertIsInstance(data_config, DataConfig)
        self.assertEqual(data_config.data_path, './data')
        self.assertEqual(data_config.train_path, './data/train.csv')
    
    def test_set_data_config(self):
        """Test data configuration setting."""
        custom_data_config = DataConfig(
            data_path='/custom/data',
            train_path='/custom/train.csv',
            normalize=False
        )
        
        self.config_manager.set_data_config(custom_data_config)
        
        retrieved_config = self.config_manager.get_data_config()
        self.assertEqual(retrieved_config.data_path, '/custom/data')
        self.assertEqual(retrieved_config.train_path, '/custom/train.csv')
        self.assertFalse(retrieved_config.normalize)
    
    def test_get_admet_config(self):
        """Test ADMET configuration retrieval."""
        self.config_manager.create_default_config()
        
        admet_config = self.config_manager.get_admet_config()
        
        self.assertIsInstance(admet_config, ADMETConfig)
        self.assertIsNotNone(admet_config.properties)
        self.assertTrue(admet_config.use_cns_mpo)
    
    def test_set_admet_config(self):
        """Test ADMET configuration setting."""
        custom_admet_config = ADMETConfig(
            properties=['absorption', 'distribution'],
            use_cns_mpo=False,
            confidence_threshold=0.8
        )
        
        self.config_manager.set_admet_config(custom_admet_config)
        
        retrieved_config = self.config_manager.get_admet_config()
        self.assertEqual(retrieved_config.properties, ['absorption', 'distribution'])
        self.assertFalse(retrieved_config.use_cns_mpo)
        self.assertEqual(retrieved_config.confidence_threshold, 0.8)
    
    def test_validate_config(self):
        """Test configuration validation."""
        self.config_manager.create_default_config()
        
        errors = self.config_manager.validate_config()
        
        # Default configuration should be valid
        self.assertEqual(len(errors), 0)
    
    def test_validate_config_invalid(self):
        """Test configuration validation with invalid values."""
        # Set invalid values
        self.config_manager.create_default_config()
        self.config_manager.set_config('model.input_dim', -1)
        self.config_manager.set_config('model.output_dim', 0)
        self.config_manager.set_config('training.epochs', -1)
        self.config_manager.set_config('training.batch_size', 0)
        self.config_manager.set_config('training.learning_rate', -1)
        
        errors = self.config_manager.validate_config()
        
        # Should have validation errors
        self.assertGreater(len(errors), 0)
        self.assertTrue(any('input_dim must be positive' in error for error in errors))
        self.assertTrue(any('output_dim must be positive' in error for error in errors))
        self.assertTrue(any('epochs must be positive' in error for error in errors))
        self.assertTrue(any('batch_size must be positive' in error for error in errors))
        self.assertTrue(any('learning_rate must be positive' in error for error in errors))
    
    def test_merge_configs(self):
        """Test configuration merging."""
        self.config_manager.create_default_config()
        
        other_config = {
            'model': {'model_type': 'gnn', 'input_dim': 200},
            'training': {'epochs': 200, 'batch_size': 64},
            'new_section': {'new_key': 'new_value'}
        }
        
        self.config_manager.merge_configs(other_config)
        
        # Check merged values
        self.assertEqual(self.config_manager.get_config('model.model_type'), 'gnn')
        self.assertEqual(self.config_manager.get_config('model.input_dim'), 200)
        self.assertEqual(self.config_manager.get_config('training.epochs'), 200)
        self.assertEqual(self.config_manager.get_config('training.batch_size'), 64)
        self.assertEqual(self.config_manager.get_config('new_section.new_key'), 'new_value')
    
    def test_export_config(self):
        """Test configuration export."""
        self.config_manager.create_default_config()
        
        # Export to JSON
        json_path = os.path.join(self.temp_dir, 'exported_config.json')
        result_path = self.config_manager.export_config(json_path, 'json')
        
        self.assertEqual(result_path, Path(json_path))
        self.assertTrue(Path(json_path).exists())
        
        # Export to YAML
        yaml_path = os.path.join(self.temp_dir, 'exported_config.yaml')
        result_path = self.config_manager.export_config(yaml_path, 'yaml')
        
        self.assertEqual(result_path, Path(yaml_path))
        self.assertTrue(Path(yaml_path).exists())
    
    def test_export_config_invalid_format(self):
        """Test configuration export with invalid format."""
        self.config_manager.create_default_config()
        
        invalid_path = os.path.join(self.temp_dir, 'invalid.txt')
        with self.assertRaises(ValueError):
            self.config_manager.export_config(invalid_path, 'invalid')


if __name__ == '__main__':
    unittest.main()
