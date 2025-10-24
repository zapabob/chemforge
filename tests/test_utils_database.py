"""
Unit tests for database utilities.
"""

import unittest
import tempfile
import os
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from chemforge.utils.database import DatabaseManager, ChEMBLDatabase, LocalDatabase


class TestDatabaseManager(unittest.TestCase):
    """Test DatabaseManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test.db')
        self.db_manager = DatabaseManager(self.db_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_init(self):
        """Test DatabaseManager initialization."""
        self.assertEqual(self.db_manager.db_path, Path(self.db_path))
        self.assertIsNone(self.db_manager.connection)
    
    def test_connect(self):
        """Test database connection."""
        self.db_manager.connect()
        self.assertIsNotNone(self.db_manager.connection)
        self.db_manager.disconnect()
    
    def test_disconnect(self):
        """Test database disconnection."""
        self.db_manager.connect()
        self.db_manager.disconnect()
        self.assertIsNone(self.db_manager.connection)
    
    def test_get_connection(self):
        """Test connection context manager."""
        with self.db_manager.get_connection() as conn:
            self.assertIsNotNone(conn)
    
    def test_execute_query(self):
        """Test query execution."""
        self.db_manager.connect()
        
        # Create test table
        self.db_manager.execute_update(
            "CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)"
        )
        
        # Insert test data
        self.db_manager.execute_update(
            "INSERT INTO test (name) VALUES (?)", ("test_name",)
        )
        
        # Query test data
        results = self.db_manager.execute_query("SELECT * FROM test")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['name'], 'test_name')
        
        self.db_manager.disconnect()
    
    def test_execute_update(self):
        """Test update execution."""
        self.db_manager.connect()
        
        # Create test table
        self.db_manager.execute_update(
            "CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)"
        )
        
        # Insert test data
        rows_affected = self.db_manager.execute_update(
            "INSERT INTO test (name) VALUES (?)", ("test_name",)
        )
        self.assertEqual(rows_affected, 1)
        
        self.db_manager.disconnect()


class TestChEMBLDatabase(unittest.TestCase):
    """Test ChEMBLDatabase class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'chembl.db')
        self.chembl_db = ChEMBLDatabase(self.db_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_init(self):
        """Test ChEMBLDatabase initialization."""
        self.assertEqual(self.chembl_db.db_path, Path(self.db_path))
        self.assertIsNotNone(self.chembl_db.chembl_loader)
    
    def test_create_tables(self):
        """Test table creation."""
        self.chembl_db.connect()
        self.chembl_db.create_tables()
        
        # Check if tables exist
        tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
        tables = self.chembl_db.execute_query(tables_query)
        table_names = [table['name'] for table in tables]
        
        expected_tables = ['molecules', 'activities', 'targets', 'predictions']
        for table in expected_tables:
            self.assertIn(table, table_names)
        
        self.chembl_db.disconnect()
    
    def test_insert_molecule(self):
        """Test molecule insertion."""
        self.chembl_db.connect()
        self.chembl_db.create_tables()
        
        molecule_data = {
            'chembl_id': 'CHEMBL123',
            'smiles': 'CCO',
            'mol_weight': 46.07,
            'logp': 0.31,
            'hbd': 1,
            'hba': 1,
            'tpsa': 20.23,
            'rotatable_bonds': 0,
            'aromatic_rings': 0,
            'heavy_atoms': 2
        }
        
        molecule_id = self.chembl_db.insert_molecule(molecule_data)
        self.assertIsInstance(molecule_id, int)
        
        # Verify insertion
        molecules = self.chembl_db.get_molecules()
        self.assertEqual(len(molecules), 1)
        self.assertEqual(molecules.iloc[0]['chembl_id'], 'CHEMBL123')
        
        self.chembl_db.disconnect()
    
    def test_insert_target(self):
        """Test target insertion."""
        self.chembl_db.connect()
        self.chembl_db.create_tables()
        
        target_data = {
            'chembl_id': 'CHEMBL1234',
            'target_name': '5-HT2A',
            'target_type': 'SINGLE PROTEIN',
            'organism': 'Homo sapiens',
            'uniprot_id': 'P28223',
            'gene_name': 'HTR2A'
        }
        
        target_id = self.chembl_db.insert_target(target_data)
        self.assertIsInstance(target_id, int)
        
        # Verify insertion
        targets = self.chembl_db.get_targets()
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets.iloc[0]['target_name'], '5-HT2A')
        
        self.chembl_db.disconnect()
    
    def test_insert_activity(self):
        """Test activity insertion."""
        self.chembl_db.connect()
        self.chembl_db.create_tables()
        
        # Insert molecule and target first
        molecule_id = self.chembl_db.insert_molecule({
            'chembl_id': 'CHEMBL123',
            'smiles': 'CCO'
        })
        
        target_id = self.chembl_db.insert_target({
            'chembl_id': 'CHEMBL1234',
            'target_name': '5-HT2A'
        })
        
        activity_data = {
            'molecule_id': molecule_id,
            'target_id': target_id,
            'activity_type': 'IC50',
            'activity_value': 5.0,
            'activity_unit': 'nM',
            'activity_relation': '='
        }
        
        activity_id = self.chembl_db.insert_activity(activity_data)
        self.assertIsInstance(activity_id, int)
        
        # Verify insertion
        activities = self.chembl_db.get_activities()
        self.assertEqual(len(activities), 1)
        self.assertEqual(activities.iloc[0]['activity_value'], 5.0)
        
        self.chembl_db.disconnect()
    
    def test_insert_prediction(self):
        """Test prediction insertion."""
        self.chembl_db.connect()
        self.chembl_db.create_tables()
        
        # Insert molecule and target first
        molecule_id = self.chembl_db.insert_molecule({
            'chembl_id': 'CHEMBL123',
            'smiles': 'CCO'
        })
        
        target_id = self.chembl_db.insert_target({
            'chembl_id': 'CHEMBL1234',
            'target_name': '5-HT2A'
        })
        
        prediction_data = {
            'molecule_id': molecule_id,
            'target_id': target_id,
            'prediction_value': 5.5,
            'confidence': 0.85,
            'model_name': 'test_model',
            'model_version': '1.0'
        }
        
        prediction_id = self.chembl_db.insert_prediction(prediction_data)
        self.assertIsInstance(prediction_id, int)
        
        # Verify insertion
        predictions = self.chembl_db.get_predictions()
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions.iloc[0]['prediction_value'], 5.5)
        
        self.chembl_db.disconnect()
    
    def test_get_molecules(self):
        """Test molecule retrieval."""
        self.chembl_db.connect()
        self.chembl_db.create_tables()
        
        # Insert test molecules
        for i in range(3):
            self.chembl_db.insert_molecule({
                'chembl_id': f'CHEMBL{i}',
                'smiles': f'C{"C" * i}O'
            })
        
        molecules = self.chembl_db.get_molecules()
        self.assertEqual(len(molecules), 3)
        
        # Test limit
        molecules_limited = self.chembl_db.get_molecules(limit=2)
        self.assertEqual(len(molecules_limited), 2)
        
        self.chembl_db.disconnect()
    
    def test_get_targets(self):
        """Test target retrieval."""
        self.chembl_db.connect()
        self.chembl_db.create_tables()
        
        # Insert test targets
        targets_data = [
            {'chembl_id': 'CHEMBL1', 'target_name': '5-HT2A'},
            {'chembl_id': 'CHEMBL2', 'target_name': 'D2'},
            {'chembl_id': 'CHEMBL3', 'target_name': 'CB1'}
        ]
        
        for target_data in targets_data:
            self.chembl_db.insert_target(target_data)
        
        targets = self.chembl_db.get_targets()
        self.assertEqual(len(targets), 3)
        
        self.chembl_db.disconnect()
    
    def test_get_activities(self):
        """Test activity retrieval."""
        self.chembl_db.connect()
        self.chembl_db.create_tables()
        
        # Insert test data
        molecule_id = self.chembl_db.insert_molecule({
            'chembl_id': 'CHEMBL123',
            'smiles': 'CCO'
        })
        
        target_id = self.chembl_db.insert_target({
            'chembl_id': 'CHEMBL1234',
            'target_name': '5-HT2A'
        })
        
        # Insert activities
        for i in range(3):
            self.chembl_db.insert_activity({
                'molecule_id': molecule_id,
                'target_id': target_id,
                'activity_type': 'IC50',
                'activity_value': 5.0 + i,
                'activity_unit': 'nM',
                'activity_relation': '='
            })
        
        activities = self.chembl_db.get_activities()
        self.assertEqual(len(activities), 3)
        
        # Test filtering
        activities_filtered = self.chembl_db.get_activities(molecule_id=molecule_id)
        self.assertEqual(len(activities_filtered), 3)
        
        self.chembl_db.disconnect()
    
    def test_get_predictions(self):
        """Test prediction retrieval."""
        self.chembl_db.connect()
        self.chembl_db.create_tables()
        
        # Insert test data
        molecule_id = self.chembl_db.insert_molecule({
            'chembl_id': 'CHEMBL123',
            'smiles': 'CCO'
        })
        
        target_id = self.chembl_db.insert_target({
            'chembl_id': 'CHEMBL1234',
            'target_name': '5-HT2A'
        })
        
        # Insert predictions
        for i in range(3):
            self.chembl_db.insert_prediction({
                'molecule_id': molecule_id,
                'target_id': target_id,
                'prediction_value': 5.0 + i,
                'confidence': 0.8 + i * 0.05,
                'model_name': 'test_model',
                'model_version': '1.0'
            })
        
        predictions = self.chembl_db.get_predictions()
        self.assertEqual(len(predictions), 3)
        
        # Test filtering
        predictions_filtered = self.chembl_db.get_predictions(molecule_id=molecule_id)
        self.assertEqual(len(predictions_filtered), 3)
        
        self.chembl_db.disconnect()
    
    @patch('chemforge.utils.database.ChEMBLLoader')
    def test_load_chembl_data(self, mock_loader):
        """Test ChEMBL data loading."""
        # Mock ChEMBL loader
        mock_loader_instance = MagicMock()
        mock_loader.return_value = mock_loader_instance
        
        # Mock target info
        mock_loader_instance.get_target_info.return_value = {
            'chembl_id': 'CHEMBL1234',
            'target_name': '5-HT2A',
            'target_type': 'SINGLE PROTEIN',
            'organism': 'Homo sapiens',
            'uniprot_id': 'P28223',
            'gene_name': 'HTR2A'
        }
        
        # Mock activities
        mock_loader_instance.get_activities.return_value = [
            {
                'molecule_chembl_id': 'CHEMBL123',
                'canonical_smiles': 'CCO',
                'molecular_weight': 46.07,
                'alogp': 0.31,
                'hbd': 1,
                'hba': 1,
                'tpsa': 20.23,
                'rotatable_bonds': 0,
                'aromatic_rings': 0,
                'heavy_atoms': 2,
                'standard_type': 'IC50',
                'standard_value': 5.0,
                'standard_units': 'nM',
                'standard_relation': '='
            }
        ]
        
        self.chembl_db.connect()
        self.chembl_db.create_tables()
        
        # Load data
        self.chembl_db.load_chembl_data(['CHEMBL1234'], limit=1)
        
        # Verify data was loaded
        molecules = self.chembl_db.get_molecules()
        targets = self.chembl_db.get_targets()
        activities = self.chembl_db.get_activities()
        
        self.assertEqual(len(molecules), 1)
        self.assertEqual(len(targets), 1)
        self.assertEqual(len(activities), 1)
        
        self.chembl_db.disconnect()


class TestLocalDatabase(unittest.TestCase):
    """Test LocalDatabase class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'local.db')
        self.local_db = LocalDatabase(self.db_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_init(self):
        """Test LocalDatabase initialization."""
        self.assertEqual(self.local_db.db_path, Path(self.db_path))
    
    def test_create_tables(self):
        """Test table creation."""
        self.local_db.connect()
        self.local_db.create_tables()
        
        # Check if tables exist
        tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
        tables = self.local_db.execute_query(tables_query)
        table_names = [table['name'] for table in tables]
        
        expected_tables = ['custom_molecules', 'custom_predictions', 'model_checkpoints']
        for table in expected_tables:
            self.assertIn(table, table_names)
        
        self.local_db.disconnect()
    
    def test_insert_custom_molecule(self):
        """Test custom molecule insertion."""
        self.local_db.connect()
        self.local_db.create_tables()
        
        molecule_data = {
            'molecule_id': 'mol_123',
            'smiles': 'CCO',
            'features': {'mol_weight': 46.07, 'logp': 0.31},
            'properties': {'cns_mpo': 4.5}
        }
        
        molecule_id = self.local_db.insert_custom_molecule(molecule_data)
        self.assertIsInstance(molecule_id, int)
        
        # Verify insertion
        molecules = self.local_db.get_custom_molecules()
        self.assertEqual(len(molecules), 1)
        self.assertEqual(molecules.iloc[0]['molecule_id'], 'mol_123')
        
        self.local_db.disconnect()
    
    def test_insert_custom_prediction(self):
        """Test custom prediction insertion."""
        self.local_db.connect()
        self.local_db.create_tables()
        
        prediction_data = {
            'molecule_id': 'mol_123',
            'target_name': '5-HT2A',
            'prediction_value': 5.5,
            'confidence': 0.85,
            'model_name': 'test_model'
        }
        
        prediction_id = self.local_db.insert_custom_prediction(prediction_data)
        self.assertIsInstance(prediction_id, int)
        
        # Verify insertion
        predictions = self.local_db.get_custom_predictions()
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions.iloc[0]['prediction_value'], 5.5)
        
        self.local_db.disconnect()
    
    def test_insert_model_checkpoint(self):
        """Test model checkpoint insertion."""
        self.local_db.connect()
        self.local_db.create_tables()
        
        checkpoint_data = {
            'model_name': 'test_model',
            'model_version': '1.0',
            'checkpoint_path': '/path/to/checkpoint.pt',
            'metrics': {'accuracy': 0.85, 'loss': 0.15}
        }
        
        checkpoint_id = self.local_db.insert_model_checkpoint(checkpoint_data)
        self.assertIsInstance(checkpoint_id, int)
        
        # Verify insertion
        checkpoints = self.local_db.get_model_checkpoints()
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints.iloc[0]['model_name'], 'test_model')
        
        self.local_db.disconnect()
    
    def test_get_custom_molecules(self):
        """Test custom molecule retrieval."""
        self.local_db.connect()
        self.local_db.create_tables()
        
        # Insert test molecules
        for i in range(3):
            self.local_db.insert_custom_molecule({
                'molecule_id': f'mol_{i}',
                'smiles': f'C{"C" * i}O',
                'features': {'mol_weight': 46.07 + i},
                'properties': {'cns_mpo': 4.5 + i}
            })
        
        molecules = self.local_db.get_custom_molecules()
        self.assertEqual(len(molecules), 3)
        
        # Test limit
        molecules_limited = self.local_db.get_custom_molecules(limit=2)
        self.assertEqual(len(molecules_limited), 2)
        
        self.local_db.disconnect()
    
    def test_get_custom_predictions(self):
        """Test custom prediction retrieval."""
        self.local_db.connect()
        self.local_db.create_tables()
        
        # Insert test predictions
        for i in range(3):
            self.local_db.insert_custom_prediction({
                'molecule_id': f'mol_{i}',
                'target_name': '5-HT2A',
                'prediction_value': 5.0 + i,
                'confidence': 0.8 + i * 0.05,
                'model_name': 'test_model'
            })
        
        predictions = self.local_db.get_custom_predictions()
        self.assertEqual(len(predictions), 3)
        
        # Test filtering
        predictions_filtered = self.local_db.get_custom_predictions(molecule_id='mol_0')
        self.assertEqual(len(predictions_filtered), 1)
        
        self.local_db.disconnect()
    
    def test_get_model_checkpoints(self):
        """Test model checkpoint retrieval."""
        self.local_db.connect()
        self.local_db.create_tables()
        
        # Insert test checkpoints
        for i in range(3):
            self.local_db.insert_model_checkpoint({
                'model_name': f'model_{i}',
                'model_version': '1.0',
                'checkpoint_path': f'/path/to/checkpoint_{i}.pt',
                'metrics': {'accuracy': 0.8 + i * 0.05, 'loss': 0.2 - i * 0.05}
            })
        
        checkpoints = self.local_db.get_model_checkpoints()
        self.assertEqual(len(checkpoints), 3)
        
        # Test filtering
        checkpoints_filtered = self.local_db.get_model_checkpoints(model_name='model_0')
        self.assertEqual(len(checkpoints_filtered), 1)
        
        self.local_db.disconnect()


if __name__ == '__main__':
    unittest.main()
