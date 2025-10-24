"""
Unit tests for pre-trained model trainer.
"""

import unittest
import tempfile
import os
import torch
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path

from chemforge.pretrained.model_trainer import PreTrainer


class TestPreTrainer(unittest.TestCase):
    """Test PreTrainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, 'output')
        self.log_dir = os.path.join(self.temp_dir, 'logs')
        
        self.trainer = PreTrainer(
            output_dir=self.output_dir,
            log_dir=self.log_dir,
            device='cpu'
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test PreTrainer initialization."""
        self.assertEqual(self.trainer.output_dir, Path(self.output_dir))
        self.assertEqual(self.trainer.log_dir, Path(self.log_dir))
        self.assertEqual(self.trainer.device, torch.device('cpu'))
        self.assertIsNotNone(self.trainer.logger)
        self.assertIsNotNone(self.trainer.data_validator)
        self.assertIsNotNone(self.trainer.model_validator)
    
    def test_get_device_auto_cpu(self):
        """Test device selection with auto on CPU."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=False):
                device = self.trainer._get_device('auto')
                self.assertEqual(device, torch.device('cpu'))
    
    def test_get_device_auto_cuda(self):
        """Test device selection with auto on CUDA."""
        with patch('torch.cuda.is_available', return_value=True):
            device = self.trainer._get_device('auto')
            self.assertEqual(device, torch.device('cuda'))
    
    def test_get_device_auto_mps(self):
        """Test device selection with auto on MPS."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=True):
                device = self.trainer._get_device('auto')
                self.assertEqual(device, torch.device('mps'))
    
    def test_get_device_specific(self):
        """Test device selection with specific device."""
        device = self.trainer._get_device('cpu')
        self.assertEqual(device, torch.device('cpu'))
    
    @patch('chemforge.pretrained.model_trainer.ChEMBLLoader')
    def test_prepare_chembl_data(self, mock_chembl_loader):
        """Test ChEMBL data preparation."""
        # Mock ChEMBL loader
        mock_loader_instance = MagicMock()
        mock_chembl_loader.return_value = mock_loader_instance
        
        # Mock data
        mock_molecules = pd.DataFrame({
            'molecule_id': [1, 2, 3],
            'smiles': ['CCO', 'CCN', 'CC(C)O'],
            'mol_weight': [46.07, 45.08, 60.10]
        })
        
        mock_targets = pd.DataFrame({
            'target_id': [1, 2],
            'target_name': ['5-HT2A', 'D2R']
        })
        
        mock_activities = pd.DataFrame({
            'molecule_id': [1, 2, 3],
            'target_id': [1, 1, 2],
            'activity_value': [5.0, 6.0, 4.5]
        })
        
        mock_loader_instance.load_molecules.return_value = mock_molecules
        mock_loader_instance.load_targets.return_value = mock_targets
        mock_loader_instance.load_activities.return_value = mock_activities
        
        # Mock molecular features
        with patch('chemforge.pretrained.model_trainer.MolecularFeatures') as mock_mol_features:
            mock_features_instance = MagicMock()
            mock_mol_features.return_value = mock_features_instance
            mock_features_instance.extract_features.return_value = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            
            # Test data preparation
            targets = ['CHEMBL1234', 'CHEMBL5678']
            data_info, split_data = self.trainer.prepare_chembl_data(targets)
            
            # Verify results
            self.assertIsInstance(data_info, dict)
            self.assertIn('total_molecules', data_info)
            self.assertIn('total_activities', data_info)
            self.assertIn('num_targets', data_info)
            self.assertIn('targets', data_info)
            
            self.assertIsInstance(split_data, dict)
            self.assertIn('train', split_data)
            self.assertIn('val', split_data)
            self.assertIn('test', split_data)
    
    def test_create_dataset(self):
        """Test dataset creation."""
        molecules = pd.DataFrame({
            'molecule_id': [1, 2, 3],
            'smiles': ['CCO', 'CCN', 'CC(C)O']
        })
        
        activities = pd.DataFrame({
            'molecule_id': [1, 2, 3],
            'target_id': [1, 1, 2],
            'activity_value': [5.0, 6.0, 4.5]
        })
        
        features = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        dataset = self.trainer._create_dataset(molecules, activities, features)
        
        self.assertIsInstance(dataset, pd.DataFrame)
        self.assertGreater(len(dataset), 0)
        self.assertIn('molecule_id', dataset.columns)
        self.assertIn('activity_value', dataset.columns)
    
    @patch('chemforge.pretrained.model_trainer.TransformerModel')
    @patch('chemforge.pretrained.model_trainer.Trainer')
    def test_train_transformer_model(self, mock_trainer, mock_transformer):
        """Test Transformer model training."""
        # Mock model
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        
        # Mock trainer
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.train.return_value = {'train_loss': 0.5, 'val_loss': 0.6}
        mock_trainer_instance.evaluate.return_value = {'test_loss': 0.7}
        
        # Mock data
        data_info = {
            'feature_dim': 200,
            'targets': ['5-HT2A', 'D2R']
        }
        
        split_data = {
            'train': pd.DataFrame({'features': [[1, 2, 3]], 'targets': [[5.0, 6.0]]}),
            'val': pd.DataFrame({'features': [[4, 5, 6]], 'targets': [[7.0, 8.0]]}),
            'test': pd.DataFrame({'features': [[7, 8, 9]], 'targets': [[9.0, 10.0]]})
        }
        
        model_config = {
            'model_type': 'transformer',
            'input_dim': 200,
            'output_dim': 2,
            'hidden_dim': 256,
            'num_layers': 6,
            'num_heads': 8,
            'dropout': 0.1
        }
        
        training_config = {
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'optimizer': 'adam',
            'scheduler': 'cosine'
        }
        
        # Mock data loaders
        with patch.object(self.trainer, '_prepare_data_loaders') as mock_prepare_loaders:
            mock_train_loader = MagicMock()
            mock_val_loader = MagicMock()
            mock_test_loader = MagicMock()
            mock_prepare_loaders.return_value = (mock_train_loader, mock_val_loader, mock_test_loader)
            
            # Test training
            results = self.trainer.train_transformer_model(
                data_info, split_data, model_config, training_config, save_model=False
            )
            
            # Verify results
            self.assertIsInstance(results, dict)
            self.assertIn('train_loss', results)
            self.assertIn('val_loss', results)
            self.assertIn('test_results', results)
    
    @patch('chemforge.pretrained.model_trainer.GNNModel')
    @patch('chemforge.pretrained.model_trainer.Trainer')
    def test_train_gnn_model(self, mock_trainer, mock_gnn):
        """Test GNN model training."""
        # Mock model
        mock_model = MagicMock()
        mock_gnn.return_value = mock_model
        
        # Mock trainer
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.train.return_value = {'train_loss': 0.5, 'val_loss': 0.6}
        mock_trainer_instance.evaluate.return_value = {'test_loss': 0.7}
        
        # Mock data
        data_info = {
            'feature_dim': 200,
            'targets': ['5-HT2A', 'D2R']
        }
        
        split_data = {
            'train': pd.DataFrame({'features': [[1, 2, 3]], 'targets': [[5.0, 6.0]]}),
            'val': pd.DataFrame({'features': [[4, 5, 6]], 'targets': [[7.0, 8.0]]}),
            'test': pd.DataFrame({'features': [[7, 8, 9]], 'targets': [[9.0, 10.0]]})
        }
        
        model_config = {
            'model_type': 'gnn',
            'input_dim': 200,
            'output_dim': 2,
            'gnn_type': 'gat',
            'gnn_layers': 3
        }
        
        training_config = {
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'optimizer': 'adam',
            'scheduler': 'cosine'
        }
        
        # Mock data loaders
        with patch.object(self.trainer, '_prepare_data_loaders') as mock_prepare_loaders:
            mock_train_loader = MagicMock()
            mock_val_loader = MagicMock()
            mock_test_loader = MagicMock()
            mock_prepare_loaders.return_value = (mock_train_loader, mock_val_loader, mock_test_loader)
            
            # Test training
            results = self.trainer.train_gnn_model(
                data_info, split_data, model_config, training_config, save_model=False
            )
            
            # Verify results
            self.assertIsInstance(results, dict)
            self.assertIn('train_loss', results)
            self.assertIn('val_loss', results)
            self.assertIn('test_results', results)
    
    @patch('chemforge.pretrained.model_trainer.EnsembleModel')
    @patch('chemforge.pretrained.model_trainer.Trainer')
    def test_train_ensemble_model(self, mock_trainer, mock_ensemble):
        """Test ensemble model training."""
        # Mock model
        mock_model = MagicMock()
        mock_ensemble.return_value = mock_model
        
        # Mock trainer
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.train.return_value = {'train_loss': 0.5, 'val_loss': 0.6}
        mock_trainer_instance.evaluate.return_value = {'test_loss': 0.7}
        
        # Mock data
        data_info = {
            'feature_dim': 200,
            'targets': ['5-HT2A', 'D2R']
        }
        
        split_data = {
            'train': pd.DataFrame({'features': [[1, 2, 3]], 'targets': [[5.0, 6.0]]}),
            'val': pd.DataFrame({'features': [[4, 5, 6]], 'targets': [[7.0, 8.0]]}),
            'test': pd.DataFrame({'features': [[7, 8, 9]], 'targets': [[9.0, 10.0]]})
        }
        
        model_config = {
            'model_type': 'ensemble',
            'input_dim': 200,
            'output_dim': 2,
            'ensemble_models': ['transformer', 'gnn'],
            'ensemble_weights': [0.5, 0.5]
        }
        
        training_config = {
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'optimizer': 'adam',
            'scheduler': 'cosine'
        }
        
        # Mock data loaders
        with patch.object(self.trainer, '_prepare_data_loaders') as mock_prepare_loaders:
            mock_train_loader = MagicMock()
            mock_val_loader = MagicMock()
            mock_test_loader = MagicMock()
            mock_prepare_loaders.return_value = (mock_train_loader, mock_val_loader, mock_test_loader)
            
            # Test training
            results = self.trainer.train_ensemble_model(
                data_info, split_data, model_config, training_config, save_model=False
            )
            
            # Verify results
            self.assertIsInstance(results, dict)
            self.assertIn('train_loss', results)
            self.assertIn('val_loss', results)
            self.assertIn('test_results', results)
    
    def test_prepare_data_loaders(self):
        """Test data loader preparation."""
        split_data = {
            'train': pd.DataFrame({'features': [[1, 2, 3]], 'targets': [[5.0, 6.0]]}),
            'val': pd.DataFrame({'features': [[4, 5, 6]], 'targets': [[7.0, 8.0]]}),
            'test': pd.DataFrame({'features': [[7, 8, 9]], 'targets': [[9.0, 10.0]]})
        }
        
        batch_size = 32
        
        train_loader, val_loader, test_loader = self.trainer._prepare_data_loaders(split_data, batch_size)
        
        # Verify loaders are created
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)
    
    def test_train_all_models(self):
        """Test training all model types."""
        # Mock data preparation
        with patch.object(self.trainer, 'prepare_chembl_data') as mock_prepare_data:
            mock_data_info = {
                'feature_dim': 200,
                'targets': ['5-HT2A', 'D2R']
            }
            mock_split_data = {
                'train': pd.DataFrame({'features': [[1, 2, 3]], 'targets': [[5.0, 6.0]]}),
                'val': pd.DataFrame({'features': [[4, 5, 6]], 'targets': [[7.0, 8.0]]}),
                'test': pd.DataFrame({'features': [[7, 8, 9]], 'targets': [[9.0, 10.0]]})
            }
            mock_prepare_data.return_value = (mock_data_info, mock_split_data)
            
            # Mock individual training methods
            with patch.object(self.trainer, 'train_transformer_model') as mock_transformer:
                with patch.object(self.trainer, 'train_gnn_model') as mock_gnn:
                    with patch.object(self.trainer, 'train_ensemble_model') as mock_ensemble:
                        mock_transformer.return_value = {'transformer_results': 'success'}
                        mock_gnn.return_value = {'gnn_results': 'success'}
                        mock_ensemble.return_value = {'ensemble_results': 'success'}
                        
                        # Test training all models
                        targets = ['CHEMBL1234', 'CHEMBL5678']
                        model_configs = {
                            'transformer': {'model_type': 'transformer'},
                            'gnn': {'model_type': 'gnn'},
                            'ensemble': {'model_type': 'ensemble'}
                        }
                        training_config = {
                            'epochs': 10,
                            'batch_size': 32,
                            'learning_rate': 1e-3
                        }
                        
                        results = self.trainer.train_all_models(
                            targets, model_configs, training_config, save_models=False
                        )
                        
                        # Verify results
                        self.assertIsInstance(results, dict)
                        self.assertIn('transformer', results)
                        self.assertIn('gnn', results)
                        self.assertIn('ensemble', results)
    
    def test_get_training_history(self):
        """Test getting training history."""
        # Add some mock history
        self.trainer.training_history = {
            'transformer': {'train_loss': 0.5, 'val_loss': 0.6},
            'gnn': {'train_loss': 0.4, 'val_loss': 0.5}
        }
        
        history = self.trainer.get_training_history()
        
        self.assertIsInstance(history, dict)
        self.assertIn('transformer', history)
        self.assertIn('gnn', history)
    
    def test_get_best_models(self):
        """Test getting best models."""
        # Add some mock models
        mock_model1 = MagicMock()
        mock_model2 = MagicMock()
        self.trainer.best_models = {
            'transformer': mock_model1,
            'gnn': mock_model2
        }
        
        models = self.trainer.get_best_models()
        
        self.assertIsInstance(models, dict)
        self.assertIn('transformer', models)
        self.assertIn('gnn', models)
    
    def test_save_training_state(self):
        """Test saving training state."""
        # Add some mock state
        self.trainer.training_history = {'transformer': {'loss': 0.5}}
        self.trainer.best_models = {'transformer': MagicMock()}
        
        state_path = os.path.join(self.temp_dir, 'training_state.pkl')
        self.trainer.save_training_state(state_path)
        
        # Verify file was created
        self.assertTrue(os.path.exists(state_path))
    
    def test_load_training_state(self):
        """Test loading training state."""
        # Create a mock state file
        mock_state = {
            'training_history': {'transformer': {'loss': 0.5}},
            'best_models': {'transformer': MagicMock()},
            'output_dir': self.output_dir,
            'log_dir': self.log_dir,
            'device': 'cpu'
        }
        
        state_path = os.path.join(self.temp_dir, 'training_state.pkl')
        with open(state_path, 'wb') as f:
            import pickle
            pickle.dump(mock_state, f)
        
        # Load state
        self.trainer.load_training_state(state_path)
        
        # Verify state was loaded
        self.assertIn('transformer', self.trainer.training_history)
        self.assertIn('transformer', self.trainer.best_models)


if __name__ == '__main__':
    unittest.main()
