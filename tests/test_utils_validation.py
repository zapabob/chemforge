"""
Unit tests for validation utilities.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from chemforge.utils.validation import (
    DataValidator,
    ModelValidator,
    PredictionValidator
)


class TestDataValidator(unittest.TestCase):
    """Test DataValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = DataValidator()
        
        # Create test molecular data
        self.molecular_data = pd.DataFrame({
            'smiles': ['CCO', 'CCN', 'CC(C)O', 'CC(C)N'],
            'mol_weight': [46.07, 45.08, 60.10, 59.11],
            'logp': [0.31, 0.16, 0.05, 0.10],
            'hbd': [1, 2, 1, 2],
            'hba': [1, 1, 1, 1],
            'tpsa': [20.23, 26.02, 20.23, 26.02],
            'rotatable_bonds': [0, 0, 0, 0],
            'aromatic_rings': [0, 0, 0, 0],
            'heavy_atoms': [2, 2, 3, 3]
        })
        
        # Create test activity data
        self.activity_data = pd.DataFrame({
            'molecule_id': [1, 2, 3, 4],
            'target_id': [1, 1, 2, 2],
            'activity_value': [5.0, 6.0, 4.5, 5.5],
            'activity_type': ['IC50', 'IC50', 'IC50', 'IC50'],
            'activity_unit': ['nM', 'nM', 'nM', 'nM'],
            'activity_relation': ['=', '=', '=', '=']
        })
    
    def test_init(self):
        """Test DataValidator initialization."""
        self.assertIsNotNone(self.validator.logger)
    
    def test_validate_molecular_data(self):
        """Test molecular data validation."""
        results = self.validator.validate_molecular_data(self.molecular_data)
        
        self.assertIsInstance(results, dict)
        self.assertIn('valid', results)
        self.assertIn('errors', results)
        self.assertIn('warnings', results)
        self.assertIn('statistics', results)
        
        self.assertTrue(results['valid'])
        self.assertEqual(len(results['errors']), 0)
        self.assertGreater(len(results['statistics']), 0)
    
    def test_validate_molecular_data_empty(self):
        """Test molecular data validation with empty data."""
        empty_data = pd.DataFrame()
        results = self.validator.validate_molecular_data(empty_data)
        
        self.assertFalse(results['valid'])
        self.assertIn('Data is empty', results['errors'])
    
    def test_validate_molecular_data_missing_columns(self):
        """Test molecular data validation with missing columns."""
        data_no_smiles = pd.DataFrame({
            'mol_weight': [46.07, 45.08],
            'logp': [0.31, 0.16]
        })
        results = self.validator.validate_molecular_data(data_no_smiles)
        
        self.assertFalse(results['valid'])
        self.assertIn('Missing required columns', results['errors'])
    
    def test_validate_molecular_data_duplicate_smiles(self):
        """Test molecular data validation with duplicate SMILES."""
        data_with_duplicates = pd.DataFrame({
            'smiles': ['CCO', 'CCO', 'CCN'],
            'mol_weight': [46.07, 46.07, 45.08],
            'logp': [0.31, 0.31, 0.16]
        })
        results = self.validator.validate_molecular_data(data_with_duplicates)
        
        self.assertTrue(results['valid'])
        self.assertIn('duplicate SMILES', results['warnings'][0])
    
    def test_validate_molecular_data_invalid_smiles(self):
        """Test molecular data validation with invalid SMILES."""
        data_with_nan = pd.DataFrame({
            'smiles': ['CCO', None, 'CCN'],
            'mol_weight': [46.07, 45.08, 45.08],
            'logp': [0.31, 0.16, 0.16]
        })
        results = self.validator.validate_molecular_data(data_with_nan)
        
        self.assertTrue(results['valid'])
        self.assertIn('invalid SMILES', results['warnings'][0])
    
    def test_validate_molecular_data_extreme_values(self):
        """Test molecular data validation with extreme values."""
        data_with_extreme = pd.DataFrame({
            'smiles': ['CCO', 'CCN'],
            'mol_weight': [46.07, 1500.0],  # Extreme MW
            'logp': [0.31, 15.0],  # Extreme logP
            'hbd': [1, 2],
            'hba': [1, 1],
            'tpsa': [20.23, 26.02],
            'rotatable_bonds': [0, 0],
            'aromatic_rings': [0, 0],
            'heavy_atoms': [2, 2]
        })
        results = self.validator.validate_molecular_data(data_with_extreme)
        
        self.assertTrue(results['valid'])
        self.assertIn('extreme', results['warnings'][0])
    
    def test_validate_activity_data(self):
        """Test activity data validation."""
        results = self.validator.validate_activity_data(self.activity_data)
        
        self.assertIsInstance(results, dict)
        self.assertIn('valid', results)
        self.assertIn('errors', results)
        self.assertIn('warnings', results)
        self.assertIn('statistics', results)
        
        self.assertTrue(results['valid'])
        self.assertEqual(len(results['errors']), 0)
    
    def test_validate_activity_data_empty(self):
        """Test activity data validation with empty data."""
        empty_data = pd.DataFrame()
        results = self.validator.validate_activity_data(empty_data)
        
        self.assertFalse(results['valid'])
        self.assertIn('Data is empty', results['errors'])
    
    def test_validate_activity_data_missing_columns(self):
        """Test activity data validation with missing columns."""
        data_no_molecule_id = pd.DataFrame({
            'target_id': [1, 2],
            'activity_value': [5.0, 6.0]
        })
        results = self.validator.validate_activity_data(data_no_molecule_id)
        
        self.assertFalse(results['valid'])
        self.assertIn('Missing required columns', results['errors'])
    
    def test_validate_activity_data_invalid_values(self):
        """Test activity data validation with invalid values."""
        data_with_nan = pd.DataFrame({
            'molecule_id': [1, 2],
            'target_id': [1, 2],
            'activity_value': [5.0, None]
        })
        results = self.validator.validate_activity_data(data_with_nan)
        
        self.assertTrue(results['valid'])
        self.assertIn('invalid activity values', results['warnings'][0])
    
    def test_validate_activity_data_extreme_values(self):
        """Test activity data validation with extreme values."""
        data_with_extreme = pd.DataFrame({
            'molecule_id': [1, 2],
            'target_id': [1, 2],
            'activity_value': [5.0, 25.0]  # Extreme value
        })
        results = self.validator.validate_activity_data(data_with_extreme)
        
        self.assertTrue(results['valid'])
        self.assertIn('extreme activity values', results['warnings'][0])
    
    def test_validate_activity_data_duplicates(self):
        """Test activity data validation with duplicates."""
        data_with_duplicates = pd.DataFrame({
            'molecule_id': [1, 1],
            'target_id': [1, 1],
            'activity_value': [5.0, 6.0]
        })
        results = self.validator.validate_activity_data(data_with_duplicates)
        
        self.assertTrue(results['valid'])
        self.assertIn('duplicate activities', results['warnings'][0])
    
    def test_validate_features(self):
        """Test feature matrix validation."""
        features = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        feature_names = ['feat1', 'feat2', 'feat3']
        
        results = self.validator.validate_features(features, feature_names)
        
        self.assertIsInstance(results, dict)
        self.assertIn('valid', results)
        self.assertIn('errors', results)
        self.assertIn('warnings', results)
        self.assertIn('statistics', results)
        
        self.assertTrue(results['valid'])
        self.assertEqual(len(results['errors']), 0)
    
    def test_validate_features_with_nan(self):
        """Test feature matrix validation with NaN values."""
        features = np.array([[1, 2, 3], [4, np.nan, 6], [7, 8, 9]])
        feature_names = ['feat1', 'feat2', 'feat3']
        
        results = self.validator.validate_features(features, feature_names)
        
        self.assertTrue(results['valid'])
        self.assertIn('NaN values', results['warnings'][0])
    
    def test_validate_features_with_inf(self):
        """Test feature matrix validation with infinite values."""
        features = np.array([[1, 2, 3], [4, np.inf, 6], [7, 8, 9]])
        feature_names = ['feat1', 'feat2', 'feat3']
        
        results = self.validator.validate_features(features, feature_names)
        
        self.assertTrue(results['valid'])
        self.assertIn('infinite values', results['warnings'][0])
    
    def test_validate_features_constant(self):
        """Test feature matrix validation with constant features."""
        features = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        feature_names = ['feat1', 'feat2', 'feat3']
        
        results = self.validator.validate_features(features, feature_names)
        
        self.assertTrue(results['valid'])
        self.assertIn('constant features', results['warnings'][0])
    
    def test_validate_features_high_correlation(self):
        """Test feature matrix validation with highly correlated features."""
        features = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
        feature_names = ['feat1', 'feat2', 'feat3']
        
        results = self.validator.validate_features(features, feature_names)
        
        self.assertTrue(results['valid'])
        self.assertIn('highly correlated', results['warnings'][0])


class TestModelValidator(unittest.TestCase):
    """Test ModelValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = ModelValidator()
    
    def test_init(self):
        """Test ModelValidator initialization."""
        self.assertIsNotNone(self.validator.logger)
    
    def test_validate_model_config(self):
        """Test model configuration validation."""
        config = {
            'model_type': 'transformer',
            'input_dim': 100,
            'output_dim': 3,
            'hidden_dim': 256,
            'num_layers': 6,
            'num_heads': 8,
            'dropout': 0.1
        }
        
        results = self.validator.validate_model_config(config)
        
        self.assertIsInstance(results, dict)
        self.assertIn('valid', results)
        self.assertIn('errors', results)
        self.assertIn('warnings', results)
        self.assertIn('statistics', results)
        
        self.assertTrue(results['valid'])
        self.assertEqual(len(results['errors']), 0)
    
    def test_validate_model_config_missing_params(self):
        """Test model configuration validation with missing parameters."""
        config = {
            'model_type': 'transformer'
        }
        
        results = self.validator.validate_model_config(config)
        
        self.assertFalse(results['valid'])
        self.assertIn('Missing required parameters', results['errors'][0])
    
    def test_validate_model_config_invalid_type(self):
        """Test model configuration validation with invalid model type."""
        config = {
            'model_type': 'invalid_type',
            'input_dim': 100,
            'output_dim': 3
        }
        
        results = self.validator.validate_model_config(config)
        
        self.assertFalse(results['valid'])
        self.assertIn('Invalid model type', results['errors'][0])
    
    def test_validate_model_config_invalid_dimensions(self):
        """Test model configuration validation with invalid dimensions."""
        config = {
            'model_type': 'transformer',
            'input_dim': -1,
            'output_dim': 0
        }
        
        results = self.validator.validate_model_config(config)
        
        self.assertFalse(results['valid'])
        self.assertIn('input_dim must be a positive integer', results['errors'][0])
        self.assertIn('output_dim must be a positive integer', results['errors'][0])
    
    def test_validate_model_config_pwa_pet(self):
        """Test model configuration validation with PWA+PET parameters."""
        config = {
            'model_type': 'transformer',
            'input_dim': 100,
            'output_dim': 3,
            'use_pwa_pet': True,
            'pwa_buckets': {'trivial': 1, 'fund': 5, 'adj': 2},
            'pet_curv_reg': 1e-5
        }
        
        results = self.validator.validate_model_config(config)
        
        self.assertTrue(results['valid'])
        self.assertEqual(len(results['errors']), 0)
    
    def test_validate_model_config_pwa_pet_missing_buckets(self):
        """Test model configuration validation with PWA+PET missing buckets."""
        config = {
            'model_type': 'transformer',
            'input_dim': 100,
            'output_dim': 3,
            'use_pwa_pet': True
        }
        
        results = self.validator.validate_model_config(config)
        
        self.assertTrue(results['valid'])
        self.assertIn('pwa_buckets not specified', results['warnings'][0])
    
    def test_validate_model_config_pwa_pet_invalid_curv_reg(self):
        """Test model configuration validation with invalid PWA+PET curvature regularization."""
        config = {
            'model_type': 'transformer',
            'input_dim': 100,
            'output_dim': 3,
            'use_pwa_pet': True,
            'pet_curv_reg': -1.0
        }
        
        results = self.validator.validate_model_config(config)
        
        self.assertFalse(results['valid'])
        self.assertIn('pet_curv_reg must be a non-negative number', results['errors'][0])
    
    def test_validate_model_config_gnn(self):
        """Test model configuration validation with GNN parameters."""
        config = {
            'model_type': 'gnn',
            'input_dim': 100,
            'output_dim': 3,
            'gnn_type': 'gat',
            'gnn_layers': 3
        }
        
        results = self.validator.validate_model_config(config)
        
        self.assertTrue(results['valid'])
        self.assertEqual(len(results['errors']), 0)
    
    def test_validate_model_config_gnn_missing_type(self):
        """Test model configuration validation with GNN missing type."""
        config = {
            'model_type': 'gnn',
            'input_dim': 100,
            'output_dim': 3
        }
        
        results = self.validator.validate_model_config(config)
        
        self.assertTrue(results['valid'])
        self.assertIn('GNN model type not specified', results['warnings'][0])
    
    def test_validate_model_config_gnn_invalid_layers(self):
        """Test model configuration validation with invalid GNN layers."""
        config = {
            'model_type': 'gnn',
            'input_dim': 100,
            'output_dim': 3,
            'gnn_layers': 0
        }
        
        results = self.validator.validate_model_config(config)
        
        self.assertFalse(results['valid'])
        self.assertIn('gnn_layers must be a positive integer', results['errors'][0])
    
    def test_validate_model_config_ensemble(self):
        """Test model configuration validation with ensemble parameters."""
        config = {
            'model_type': 'ensemble',
            'input_dim': 100,
            'output_dim': 3,
            'ensemble_models': ['transformer', 'gnn'],
            'ensemble_weights': [0.5, 0.5]
        }
        
        results = self.validator.validate_model_config(config)
        
        self.assertTrue(results['valid'])
        self.assertEqual(len(results['errors']), 0)
    
    def test_validate_model_config_ensemble_missing_models(self):
        """Test model configuration validation with ensemble missing models."""
        config = {
            'model_type': 'ensemble',
            'input_dim': 100,
            'output_dim': 3
        }
        
        results = self.validator.validate_model_config(config)
        
        self.assertFalse(results['valid'])
        self.assertIn('Ensemble models not specified', results['errors'][0])
    
    def test_validate_model_config_ensemble_insufficient_models(self):
        """Test model configuration validation with ensemble insufficient models."""
        config = {
            'model_type': 'ensemble',
            'input_dim': 100,
            'output_dim': 3,
            'ensemble_models': ['transformer']
        }
        
        results = self.validator.validate_model_config(config)
        
        self.assertFalse(results['valid'])
        self.assertIn('Ensemble must have at least 2 models', results['errors'][0])
    
    def test_validate_model_config_ensemble_invalid_weights(self):
        """Test model configuration validation with invalid ensemble weights."""
        config = {
            'model_type': 'ensemble',
            'input_dim': 100,
            'output_dim': 3,
            'ensemble_models': ['transformer', 'gnn'],
            'ensemble_weights': [0.5, 0.5, 0.5]  # Wrong length
        }
        
        results = self.validator.validate_model_config(config)
        
        self.assertFalse(results['valid'])
        self.assertIn('ensemble_weights length must match ensemble_models length', results['errors'][0])
    
    def test_validate_model_config_ensemble_weights_not_sum_to_one(self):
        """Test model configuration validation with ensemble weights not summing to one."""
        config = {
            'model_type': 'ensemble',
            'input_dim': 100,
            'output_dim': 3,
            'ensemble_models': ['transformer', 'gnn'],
            'ensemble_weights': [0.3, 0.3]  # Sums to 0.6, not 1.0
        }
        
        results = self.validator.validate_model_config(config)
        
        self.assertTrue(results['valid'])
        self.assertIn('ensemble_weights do not sum to 1.0', results['warnings'][0])
    
    def test_validate_training_config(self):
        """Test training configuration validation."""
        config = {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'optimizer': 'adam',
            'scheduler': 'cosine',
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1
        }
        
        results = self.validator.validate_training_config(config)
        
        self.assertIsInstance(results, dict)
        self.assertIn('valid', results)
        self.assertIn('errors', results)
        self.assertIn('warnings', results)
        self.assertIn('statistics', results)
        
        self.assertTrue(results['valid'])
        self.assertEqual(len(results['errors']), 0)
    
    def test_validate_training_config_missing_params(self):
        """Test training configuration validation with missing parameters."""
        config = {
            'epochs': 100
        }
        
        results = self.validator.validate_training_config(config)
        
        self.assertFalse(results['valid'])
        self.assertIn('Missing required parameters', results['errors'][0])
    
    def test_validate_training_config_invalid_epochs(self):
        """Test training configuration validation with invalid epochs."""
        config = {
            'epochs': -1,
            'batch_size': 32,
            'learning_rate': 1e-3
        }
        
        results = self.validator.validate_training_config(config)
        
        self.assertFalse(results['valid'])
        self.assertIn('epochs must be a positive integer', results['errors'][0])
    
    def test_validate_training_config_invalid_batch_size(self):
        """Test training configuration validation with invalid batch size."""
        config = {
            'epochs': 100,
            'batch_size': 0,
            'learning_rate': 1e-3
        }
        
        results = self.validator.validate_training_config(config)
        
        self.assertFalse(results['valid'])
        self.assertIn('batch_size must be a positive integer', results['errors'][0])
    
    def test_validate_training_config_invalid_learning_rate(self):
        """Test training configuration validation with invalid learning rate."""
        config = {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': -1e-3
        }
        
        results = self.validator.validate_training_config(config)
        
        self.assertFalse(results['valid'])
        self.assertIn('learning_rate must be a positive number', results['errors'][0])
    
    def test_validate_training_config_high_learning_rate(self):
        """Test training configuration validation with high learning rate."""
        config = {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 2.0
        }
        
        results = self.validator.validate_training_config(config)
        
        self.assertTrue(results['valid'])
        self.assertIn('learning_rate is very high', results['warnings'][0])
    
    def test_validate_training_config_invalid_splits(self):
        """Test training configuration validation with invalid data splits."""
        config = {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'train_split': 1.5,  # Invalid split
            'val_split': 0.1,
            'test_split': 0.1
        }
        
        results = self.validator.validate_training_config(config)
        
        self.assertFalse(results['valid'])
        self.assertIn('train_split must be between 0 and 1', results['errors'][0])
    
    def test_validate_training_config_splits_not_sum_to_one(self):
        """Test training configuration validation with splits not summing to one."""
        config = {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'train_split': 0.5,
            'val_split': 0.3,
            'test_split': 0.3  # Sums to 1.1, not 1.0
        }
        
        results = self.validator.validate_training_config(config)
        
        self.assertTrue(results['valid'])
        self.assertIn('Data splits sum to', results['warnings'][0])
    
    def test_validate_training_config_unknown_optimizer(self):
        """Test training configuration validation with unknown optimizer."""
        config = {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'optimizer': 'unknown_optimizer'
        }
        
        results = self.validator.validate_training_config(config)
        
        self.assertTrue(results['valid'])
        self.assertIn('Unknown optimizer', results['warnings'][0])
    
    def test_validate_training_config_unknown_scheduler(self):
        """Test training configuration validation with unknown scheduler."""
        config = {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'scheduler': 'unknown_scheduler'
        }
        
        results = self.validator.validate_training_config(config)
        
        self.assertTrue(results['valid'])
        self.assertIn('Unknown scheduler', results['warnings'][0])
    
    def test_validate_model_weights(self):
        """Test model weights validation."""
        weights = {
            'layer1.weight': np.array([[1, 2], [3, 4]]),
            'layer1.bias': np.array([1, 2]),
            'layer2.weight': np.array([[5, 6], [7, 8]]),
            'layer2.bias': np.array([3, 4])
        }
        
        results = self.validator.validate_model_weights(weights)
        
        self.assertIsInstance(results, dict)
        self.assertIn('valid', results)
        self.assertIn('errors', results)
        self.assertIn('warnings', results)
        self.assertIn('statistics', results)
        
        self.assertTrue(results['valid'])
        self.assertEqual(len(results['errors']), 0)
    
    def test_validate_model_weights_empty(self):
        """Test model weights validation with empty weights."""
        weights = {}
        
        results = self.validator.validate_model_weights(weights)
        
        self.assertFalse(results['valid'])
        self.assertIn('Model weights are empty', results['errors'][0])
    
    def test_validate_model_weights_with_nan(self):
        """Test model weights validation with NaN values."""
        weights = {
            'layer1.weight': np.array([[1, 2], [np.nan, 4]]),
            'layer1.bias': np.array([1, 2])
        }
        
        results = self.validator.validate_model_weights(weights)
        
        self.assertFalse(results['valid'])
        self.assertIn('NaN values in model weights', results['errors'][0])
    
    def test_validate_model_weights_with_inf(self):
        """Test model weights validation with infinite values."""
        weights = {
            'layer1.weight': np.array([[1, 2], [np.inf, 4]]),
            'layer1.bias': np.array([1, 2])
        }
        
        results = self.validator.validate_model_weights(weights)
        
        self.assertFalse(results['valid'])
        self.assertIn('infinite values in model weights', results['errors'][0])
    
    def test_validate_model_weights_high_zero_count(self):
        """Test model weights validation with high zero count."""
        weights = {
            'layer1.weight': np.zeros((10, 10)),
            'layer1.bias': np.zeros(10)
        }
        
        results = self.validator.validate_model_weights(weights)
        
        self.assertTrue(results['valid'])
        self.assertIn('High number of zero weights', results['warnings'][0])


class TestPredictionValidator(unittest.TestCase):
    """Test PredictionValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = PredictionValidator()
    
    def test_init(self):
        """Test PredictionValidator initialization."""
        self.assertIsNotNone(self.validator.logger)
    
    def test_validate_predictions(self):
        """Test predictions validation."""
        predictions = np.array([5.0, 6.0, 4.5, 5.5])
        confidence = np.array([0.8, 0.9, 0.7, 0.85])
        
        results = self.validator.validate_predictions(predictions, confidence)
        
        self.assertIsInstance(results, dict)
        self.assertIn('valid', results)
        self.assertIn('errors', results)
        self.assertIn('warnings', results)
        self.assertIn('statistics', results)
        
        self.assertTrue(results['valid'])
        self.assertEqual(len(results['errors']), 0)
    
    def test_validate_predictions_with_nan(self):
        """Test predictions validation with NaN values."""
        predictions = np.array([5.0, np.nan, 4.5, 5.5])
        confidence = np.array([0.8, 0.9, 0.7, 0.85])
        
        results = self.validator.validate_predictions(predictions, confidence)
        
        self.assertFalse(results['valid'])
        self.assertIn('NaN values in predictions', results['errors'][0])
    
    def test_validate_predictions_with_inf(self):
        """Test predictions validation with infinite values."""
        predictions = np.array([5.0, np.inf, 4.5, 5.5])
        confidence = np.array([0.8, 0.9, 0.7, 0.85])
        
        results = self.validator.validate_predictions(predictions, confidence)
        
        self.assertFalse(results['valid'])
        self.assertIn('infinite values in predictions', results['errors'][0])
    
    def test_validate_predictions_extreme_values(self):
        """Test predictions validation with extreme values."""
        predictions = np.array([5.0, 25.0, 4.5, 5.5])  # Extreme value
        confidence = np.array([0.8, 0.9, 0.7, 0.85])
        
        results = self.validator.validate_predictions(predictions, confidence)
        
        self.assertTrue(results['valid'])
        self.assertIn('extreme prediction values', results['warnings'][0])
    
    def test_validate_predictions_confidence_mismatch(self):
        """Test predictions validation with confidence shape mismatch."""
        predictions = np.array([5.0, 6.0, 4.5, 5.5])
        confidence = np.array([0.8, 0.9])  # Wrong shape
        
        results = self.validator.validate_predictions(predictions, confidence)
        
        self.assertFalse(results['valid'])
        self.assertIn('Confidence shape does not match predictions shape', results['errors'][0])
    
    def test_validate_predictions_confidence_out_of_range(self):
        """Test predictions validation with confidence out of range."""
        predictions = np.array([5.0, 6.0, 4.5, 5.5])
        confidence = np.array([0.8, 1.5, 0.7, 0.85])  # Out of range
        
        results = self.validator.validate_predictions(predictions, confidence)
        
        self.assertTrue(results['valid'])
        self.assertIn('Confidence values outside [0, 1] range', results['warnings'][0])
    
    def test_validate_predictions_low_confidence(self):
        """Test predictions validation with low confidence."""
        predictions = np.array([5.0, 6.0, 4.5, 5.5])
        confidence = np.array([0.8, 0.3, 0.7, 0.85])  # Low confidence
        
        results = self.validator.validate_predictions(predictions, confidence)
        
        self.assertTrue(results['valid'])
        self.assertIn('low confidence predictions', results['warnings'][0])
    
    def test_validate_predictions_no_confidence(self):
        """Test predictions validation without confidence."""
        predictions = np.array([5.0, 6.0, 4.5, 5.5])
        
        results = self.validator.validate_predictions(predictions)
        
        self.assertTrue(results['valid'])
        self.assertEqual(len(results['errors']), 0)
        self.assertFalse(results['statistics']['has_confidence'])
    
    def test_validate_admet_predictions(self):
        """Test ADMET predictions validation."""
        predictions = {
            'absorption': np.array([0.8, 0.7, 0.9, 0.6]),
            'distribution': np.array([0.7, 0.8, 0.6, 0.9]),
            'metabolism': np.array([0.6, 0.7, 0.8, 0.5]),
            'excretion': np.array([0.9, 0.8, 0.7, 0.6]),
            'toxicity': np.array([0.3, 0.4, 0.2, 0.5])
        }
        
        results = self.validator.validate_admet_predictions(predictions)
        
        self.assertIsInstance(results, dict)
        self.assertIn('valid', results)
        self.assertIn('errors', results)
        self.assertIn('warnings', results)
        self.assertIn('statistics', results)
        
        self.assertTrue(results['valid'])
        self.assertEqual(len(results['errors']), 0)
    
    def test_validate_admet_predictions_missing_properties(self):
        """Test ADMET predictions validation with missing properties."""
        predictions = {
            'absorption': np.array([0.8, 0.7, 0.9, 0.6]),
            'distribution': np.array([0.7, 0.8, 0.6, 0.9])
            # Missing metabolism, excretion, toxicity
        }
        
        results = self.validator.validate_admet_predictions(predictions)
        
        self.assertTrue(results['valid'])
        self.assertIn('Missing ADMET properties', results['warnings'][0])
    
    def test_validate_admet_predictions_invalid_property(self):
        """Test ADMET predictions validation with invalid property type."""
        predictions = {
            'absorption': [0.8, 0.7, 0.9, 0.6],  # Not numpy array
            'distribution': np.array([0.7, 0.8, 0.6, 0.9])
        }
        
        results = self.validator.validate_admet_predictions(predictions)
        
        self.assertFalse(results['valid'])
        self.assertIn('ADMET property absorption is not a numpy array', results['errors'][0])
    
    def test_validate_admet_predictions_with_nan(self):
        """Test ADMET predictions validation with NaN values."""
        predictions = {
            'absorption': np.array([0.8, np.nan, 0.9, 0.6]),
            'distribution': np.array([0.7, 0.8, 0.6, 0.9])
        }
        
        results = self.validator.validate_admet_predictions(predictions)
        
        self.assertTrue(results['valid'])
        self.assertIn('NaN values in absorption', results['warnings'][0])
    
    def test_validate_admet_predictions_out_of_range(self):
        """Test ADMET predictions validation with values out of range."""
        predictions = {
            'absorption': np.array([0.8, 1.5, 0.9, 0.6]),  # Out of range
            'distribution': np.array([0.7, 0.8, 0.6, 0.9])
        }
        
        results = self.validator.validate_admet_predictions(predictions)
        
        self.assertTrue(results['valid'])
        self.assertIn('values outside [0, 1] range in absorption', results['warnings'][0])
    
    def test_validate_cns_mpo_scores(self):
        """Test CNS-MPO scores validation."""
        scores = np.array([4.5, 3.8, 5.2, 2.1])
        
        results = self.validator.validate_cns_mpo_scores(scores)
        
        self.assertIsInstance(results, dict)
        self.assertIn('valid', results)
        self.assertIn('errors', results)
        self.assertIn('warnings', results)
        self.assertIn('statistics', results)
        
        self.assertTrue(results['valid'])
        self.assertEqual(len(results['errors']), 0)
    
    def test_validate_cns_mpo_scores_with_nan(self):
        """Test CNS-MPO scores validation with NaN values."""
        scores = np.array([4.5, np.nan, 5.2, 2.1])
        
        results = self.validator.validate_cns_mpo_scores(scores)
        
        self.assertTrue(results['valid'])
        self.assertIn('NaN values in CNS-MPO scores', results['warnings'][0])
    
    def test_validate_cns_mpo_scores_out_of_range(self):
        """Test CNS-MPO scores validation with values out of range."""
        scores = np.array([4.5, 7.0, 5.2, 2.1])  # Out of range
        
        results = self.validator.validate_cns_mpo_scores(scores)
        
        self.assertTrue(results['valid'])
        self.assertIn('CNS-MPO scores outside [0, 6] range', results['warnings'][0])
    
    def test_validate_cns_mpo_scores_low_scores(self):
        """Test CNS-MPO scores validation with low scores."""
        scores = np.array([4.5, 1.5, 5.2, 2.1])  # Low score
        
        results = self.validator.validate_cns_mpo_scores(scores)
        
        self.assertTrue(results['valid'])
        self.assertIn('low CNS-MPO scores', results['warnings'][0])


if __name__ == '__main__':
    unittest.main()
