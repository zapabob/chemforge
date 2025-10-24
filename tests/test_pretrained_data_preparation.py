"""
Unit tests for pre-trained data preparation.
"""

import unittest
import tempfile
import os
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path

from chemforge.pretrained.data_preparation import DataPreparator


class TestDataPreparator(unittest.TestCase):
    """Test DataPreparator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, 'output')
        self.log_dir = os.path.join(self.temp_dir, 'logs')
        
        self.preparator = DataPreparator(
            output_dir=self.output_dir,
            log_dir=self.log_dir
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test DataPreparator initialization."""
        self.assertEqual(self.preparator.output_dir, Path(self.output_dir))
        self.assertEqual(self.preparator.log_dir, Path(self.log_dir))
        self.assertIsNotNone(self.preparator.logger)
        self.assertIsNotNone(self.preparator.data_validator)
        self.assertIsNotNone(self.preparator.mol_features)
        self.assertIsNotNone(self.preparator.rdkit_descriptors)
        self.assertIsNotNone(self.preparator.preprocessor)
    
    @patch('chemforge.pretrained.data_preparation.ChEMBLLoader')
    def test_prepare_chembl_dataset(self, mock_chembl_loader):
        """Test ChEMBL dataset preparation."""
        # Mock ChEMBL loader
        mock_loader_instance = MagicMock()
        mock_chembl_loader.return_value = mock_loader_instance
        
        # Mock data
        mock_molecules = pd.DataFrame({
            'molecule_id': [1, 2, 3, 4, 5],
            'smiles': ['CCO', 'CCN', 'CC(C)O', 'CC(C)N', 'CC(C)(C)O'],
            'mol_weight': [46.07, 45.08, 60.10, 59.11, 74.12]
        })
        
        mock_targets = pd.DataFrame({
            'target_id': [1, 2],
            'target_name': ['5-HT2A', 'D2R']
        })
        
        mock_activities = pd.DataFrame({
            'molecule_id': [1, 2, 3, 4, 5],
            'target_id': [1, 1, 2, 2, 1],
            'activity_value': [5.0, 6.0, 4.5, 5.5, 4.0]
        })
        
        mock_loader_instance.load_molecules.return_value = mock_molecules
        mock_loader_instance.load_targets.return_value = mock_targets
        mock_loader_instance.load_activities.return_value = mock_activities
        
        # Mock molecular features
        with patch.object(self.preparator, '_extract_molecular_features') as mock_extract_features:
            mock_extract_features.return_value = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
            
            # Test dataset preparation
            targets = ['CHEMBL1234', 'CHEMBL5678']
            dataset = self.preparator.prepare_chembl_dataset(
                targets, min_activities=2, save_data=False
            )
            
            # Verify results
            self.assertIsInstance(dataset, dict)
            self.assertIn('data_info', dataset)
            self.assertIn('split_data', dataset)
            self.assertIn('validation_results', dataset)
            self.assertIn('preparation_metadata', dataset)
            
            # Verify data info
            data_info = dataset['data_info']
            self.assertIn('total_molecules', data_info)
            self.assertIn('total_activities', data_info)
            self.assertIn('num_targets', data_info)
            self.assertIn('targets', data_info)
            self.assertIn('feature_dim', data_info)
            
            # Verify split data
            split_data = dataset['split_data']
            self.assertIn('train', split_data)
            self.assertIn('val', split_data)
            self.assertIn('test', split_data)
    
    def test_extract_molecular_features(self):
        """Test molecular feature extraction."""
        molecules = pd.DataFrame({
            'molecule_id': [1, 2, 3],
            'smiles': ['CCO', 'CCN', 'CC(C)O']
        })
        
        # Mock feature extraction
        with patch.object(self.preparator.mol_features, 'extract_features') as mock_basic_features:
            with patch.object(self.preparator.rdkit_descriptors, 'calculate_descriptors') as mock_rdkit_features:
                mock_basic_features.return_value = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
                mock_rdkit_features.return_value = np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]])
                
                features = self.preparator._extract_molecular_features(molecules)
                
                # Verify features
                self.assertIsInstance(features, np.ndarray)
                self.assertEqual(features.shape[0], 3)  # 3 molecules
                self.assertEqual(features.shape[1], 6)  # 3 basic + 3 RDKit features
    
    def test_create_unified_dataset(self):
        """Test unified dataset creation."""
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
        
        dataset = self.preparator._create_unified_dataset(molecules, activities, features)
        
        # Verify dataset
        self.assertIsInstance(dataset, pd.DataFrame)
        self.assertGreater(len(dataset), 0)
        self.assertIn('molecule_id', dataset.columns)
        self.assertIn('activity_value', dataset.columns)
        self.assertIn('feature_0', dataset.columns)
        self.assertIn('feature_1', dataset.columns)
        self.assertIn('feature_2', dataset.columns)
    
    def test_normalize_features(self):
        """Test feature normalization."""
        split_data = {
            'train': pd.DataFrame({
                'feature_0': [1, 2, 3],
                'feature_1': [4, 5, 6],
                'feature_2': [7, 8, 9],
                'other_col': ['a', 'b', 'c']
            }),
            'val': pd.DataFrame({
                'feature_0': [2, 3, 4],
                'feature_1': [5, 6, 7],
                'feature_2': [8, 9, 10],
                'other_col': ['d', 'e', 'f']
            }),
            'test': pd.DataFrame({
                'feature_0': [3, 4, 5],
                'feature_1': [6, 7, 8],
                'feature_2': [9, 10, 11],
                'other_col': ['g', 'h', 'i']
            })
        }
        
        normalized_data = self.preparator._normalize_features(split_data)
        
        # Verify normalization
        self.assertIsInstance(normalized_data, dict)
        self.assertIn('train', normalized_data)
        self.assertIn('val', normalized_data)
        self.assertIn('test', normalized_data)
        
        # Verify scaler was stored
        self.assertIn('features', self.preparator.scalers)
        self.assertIsNotNone(self.preparator.scalers['features'])
    
    def test_prepare_custom_dataset(self):
        """Test custom dataset preparation."""
        molecules = pd.DataFrame({
            'molecule_id': [1, 2, 3, 4, 5],
            'smiles': ['CCO', 'CCN', 'CC(C)O', 'CC(C)N', 'CC(C)(C)O']
        })
        
        activities = pd.DataFrame({
            'molecule_id': [1, 2, 3, 4, 5],
            'target_id': [1, 1, 2, 2, 1],
            'activity_value': [5.0, 6.0, 4.5, 5.5, 4.0]
        })
        
        targets = ['5-HT2A', 'D2R']
        
        # Mock feature extraction
        with patch.object(self.preparator, '_extract_molecular_features') as mock_extract_features:
            mock_extract_features.return_value = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
            
            dataset = self.preparator.prepare_custom_dataset(
                molecules, activities, targets, save_data=False
            )
            
            # Verify results
            self.assertIsInstance(dataset, dict)
            self.assertIn('data_info', dataset)
            self.assertIn('split_data', dataset)
            self.assertIn('validation_results', dataset)
            self.assertIn('preparation_metadata', dataset)
            
            # Verify dataset type
            self.assertEqual(dataset['data_info']['dataset_type'], 'custom')
    
    def test_get_data_statistics(self):
        """Test data statistics generation."""
        # Create mock dataset
        dataset = {
            'data_info': {
                'total_molecules': 100,
                'total_activities': 200,
                'num_targets': 2,
                'targets': ['5-HT2A', 'D2R']
            },
            'split_data': {
                'train': pd.DataFrame({
                    'feature_0': [1, 2, 3, 4, 5],
                    'feature_1': [6, 7, 8, 9, 10],
                    'target_5-HT2A': [5.0, 6.0, 7.0, 8.0, 9.0],
                    'target_D2R': [4.0, 5.0, 6.0, 7.0, 8.0]
                }),
                'val': pd.DataFrame({
                    'feature_0': [2, 3, 4],
                    'feature_1': [7, 8, 9],
                    'target_5-HT2A': [6.0, 7.0, 8.0],
                    'target_D2R': [5.0, 6.0, 7.0]
                }),
                'test': pd.DataFrame({
                    'feature_0': [3, 4, 5],
                    'feature_1': [8, 9, 10],
                    'target_5-HT2A': [7.0, 8.0, 9.0],
                    'target_D2R': [6.0, 7.0, 8.0]
                })
            }
        }
        
        statistics = self.preparator.get_data_statistics(dataset)
        
        # Verify statistics
        self.assertIsInstance(statistics, dict)
        self.assertIn('dataset_info', statistics)
        self.assertIn('split_sizes', statistics)
        self.assertIn('feature_statistics', statistics)
        self.assertIn('target_statistics', statistics)
        
        # Verify split sizes
        self.assertEqual(statistics['split_sizes']['train'], 5)
        self.assertEqual(statistics['split_sizes']['val'], 3)
        self.assertEqual(statistics['split_sizes']['test'], 3)
        
        # Verify feature statistics
        self.assertIn('mean', statistics['feature_statistics'])
        self.assertIn('std', statistics['feature_statistics'])
        self.assertIn('min', statistics['feature_statistics'])
        self.assertIn('max', statistics['feature_statistics'])
        
        # Verify target statistics
        self.assertIn('5-HT2A', statistics['target_statistics'])
        self.assertIn('D2R', statistics['target_statistics'])
    
    def test_export_dataset_csv(self):
        """Test dataset export to CSV."""
        # Create mock dataset
        dataset = {
            'data_info': {'total_molecules': 100},
            'split_data': {
                'train': pd.DataFrame({'feature_0': [1, 2, 3], 'target': [5.0, 6.0, 7.0]}),
                'val': pd.DataFrame({'feature_0': [4, 5, 6], 'target': [8.0, 9.0, 10.0]}),
                'test': pd.DataFrame({'feature_0': [7, 8, 9], 'target': [11.0, 12.0, 13.0]})
            }
        }
        
        output_path = self.preparator.export_dataset(dataset, 'csv')
        
        # Verify files were created
        self.assertTrue(os.path.exists(output_path.replace('.csv', '_train.csv')))
        self.assertTrue(os.path.exists(output_path.replace('.csv', '_val.csv')))
        self.assertTrue(os.path.exists(output_path.replace('.csv', '_test.csv')))
    
    def test_export_dataset_json(self):
        """Test dataset export to JSON."""
        # Create mock dataset
        dataset = {
            'data_info': {'total_molecules': 100},
            'split_data': {
                'train': pd.DataFrame({'feature_0': [1, 2, 3], 'target': [5.0, 6.0, 7.0]}),
                'val': pd.DataFrame({'feature_0': [4, 5, 6], 'target': [8.0, 9.0, 10.0]}),
                'test': pd.DataFrame({'feature_0': [7, 8, 9], 'target': [11.0, 12.0, 13.0]})
            }
        }
        
        output_path = self.preparator.export_dataset(dataset, 'json')
        
        # Verify file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Verify JSON content
        import json
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        self.assertIn('data_info', data)
        self.assertIn('split_data', data)
        self.assertIn('train', data['split_data'])
        self.assertIn('val', data['split_data'])
        self.assertIn('test', data['split_data'])
    
    def test_export_dataset_parquet(self):
        """Test dataset export to Parquet."""
        # Create mock dataset
        dataset = {
            'data_info': {'total_molecules': 100},
            'split_data': {
                'train': pd.DataFrame({'feature_0': [1, 2, 3], 'target': [5.0, 6.0, 7.0]}),
                'val': pd.DataFrame({'feature_0': [4, 5, 6], 'target': [8.0, 9.0, 10.0]}),
                'test': pd.DataFrame({'feature_0': [7, 8, 9], 'target': [11.0, 12.0, 13.0]})
            }
        }
        
        output_path = self.preparator.export_dataset(dataset, 'parquet')
        
        # Verify files were created
        self.assertTrue(os.path.exists(output_path.replace('.parquet', '_train.parquet')))
        self.assertTrue(os.path.exists(output_path.replace('.parquet', '_val.parquet')))
        self.assertTrue(os.path.exists(output_path.replace('.parquet', '_test.parquet')))
    
    def test_export_dataset_unsupported_format(self):
        """Test dataset export with unsupported format."""
        dataset = {
            'data_info': {'total_molecules': 100},
            'split_data': {
                'train': pd.DataFrame({'feature_0': [1, 2, 3]})
            }
        }
        
        with self.assertRaises(ValueError):
            self.preparator.export_dataset(dataset, 'unsupported_format')
    
    def test_get_available_datasets(self):
        """Test getting available datasets."""
        # Create some mock dataset files
        dataset_file1 = os.path.join(self.output_dir, 'chembl_dataset_5-HT2A_D2R.pkl')
        dataset_file2 = os.path.join(self.output_dir, 'custom_dataset_5-HT2A.pkl')
        
        # Create the files
        with open(dataset_file1, 'w') as f:
            f.write('mock data')
        with open(dataset_file2, 'w') as f:
            f.write('mock data')
        
        datasets = self.preparator.get_available_datasets()
        
        # Verify datasets
        self.assertIsInstance(datasets, list)
        self.assertIn('chembl_dataset_5-HT2A_D2R', datasets)
        self.assertIn('custom_dataset_5-HT2A', datasets)
    
    def test_load_prepared_data(self):
        """Test loading prepared data."""
        # Create mock dataset file
        dataset_file = os.path.join(self.output_dir, 'chembl_dataset_5-HT2A_D2R.pkl')
        mock_dataset = {
            'data_info': {'total_molecules': 100},
            'split_data': {
                'train': pd.DataFrame({'feature_0': [1, 2, 3]}),
                'val': pd.DataFrame({'feature_0': [4, 5, 6]}),
                'test': pd.DataFrame({'feature_0': [7, 8, 9]})
            }
        }
        
        # Save mock dataset
        import pickle
        with open(dataset_file, 'wb') as f:
            pickle.dump(mock_dataset, f)
        
        # Load dataset
        targets = ['5-HT2A', 'D2R']
        dataset = self.preparator.load_prepared_data(targets)
        
        # Verify dataset
        self.assertIsInstance(dataset, dict)
        self.assertIn('data_info', dataset)
        self.assertIn('split_data', dataset)
    
    def test_load_prepared_data_not_found(self):
        """Test loading prepared data when file doesn't exist."""
        targets = ['NONEXISTENT']
        
        with self.assertRaises(FileNotFoundError):
            self.preparator.load_prepared_data(targets)


if __name__ == '__main__':
    unittest.main()
