"""
Unit tests for GUI utilities.
"""

import unittest
import tempfile
import os
import pandas as pd
import numpy as np
import json
from unittest.mock import patch, MagicMock
from pathlib import Path

from chemforge.gui.gui_utils import GUIUtils


class TestGUIUtils(unittest.TestCase):
    """Test GUIUtils class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.utils = GUIUtils()
    
    def test_init(self):
        """Test GUIUtils initialization."""
        self.assertIsNotNone(self.utils.logger)
        self.assertIsInstance(self.utils, GUIUtils)
    
    def test_process_uploaded_file_csv(self):
        """Test CSV file processing."""
        # Create mock CSV content
        csv_content = "smiles\nCCO\nCCN\nCC(C)O"
        contents = f"data:text/csv;base64,{csv_content.encode().decode()}"
        
        # Test CSV processing
        molecules = self.utils.process_uploaded_file(contents, "test.csv")
        
        # Verify molecules were extracted
        self.assertEqual(len(molecules), 3)
        self.assertIn('CCO', molecules)
        self.assertIn('CCN', molecules)
        self.assertIn('CC(C)O', molecules)
    
    def test_process_uploaded_file_txt(self):
        """Test TXT file processing."""
        # Create mock TXT content
        txt_content = "CCO\nCCN\nCC(C)O"
        contents = f"data:text/plain;base64,{txt_content.encode().decode()}"
        
        # Test TXT processing
        molecules = self.utils.process_uploaded_file(contents, "test.txt")
        
        # Verify molecules were extracted
        self.assertEqual(len(molecules), 3)
        self.assertIn('CCO', molecules)
        self.assertIn('CCN', molecules)
        self.assertIn('CC(C)O', molecules)
    
    def test_process_uploaded_file_json(self):
        """Test JSON file processing."""
        # Create mock JSON content
        json_content = ["CCO", "CCN", "CC(C)O"]
        contents = f"data:application/json;base64,{json.dumps(json_content).encode().decode()}"
        
        # Test JSON processing
        molecules = self.utils.process_uploaded_file(contents, "test.json")
        
        # Verify molecules were extracted
        self.assertEqual(len(molecules), 3)
        self.assertIn('CCO', molecules)
        self.assertIn('CCN', molecules)
        self.assertIn('CC(C)O', molecules)
    
    def test_process_uploaded_file_json_with_molecules_key(self):
        """Test JSON file processing with molecules key."""
        # Create mock JSON content with molecules key
        json_content = {"molecules": ["CCO", "CCN", "CC(C)O"]}
        contents = f"data:application/json;base64,{json.dumps(json_content).encode().decode()}"
        
        # Test JSON processing
        molecules = self.utils.process_uploaded_file(contents, "test.json")
        
        # Verify molecules were extracted
        self.assertEqual(len(molecules), 3)
        self.assertIn('CCO', molecules)
        self.assertIn('CCN', molecules)
        self.assertIn('CC(C)O', molecules)
    
    def test_process_uploaded_file_invalid(self):
        """Test invalid file processing."""
        # Test with invalid content
        molecules = self.utils.process_uploaded_file("invalid_content", "test.xyz")
        
        # Verify empty list is returned
        self.assertEqual(len(molecules), 0)
    
    def test_create_molecular_properties_plot(self):
        """Test molecular properties plot creation."""
        # Create test data
        df = pd.DataFrame({
            'SMILES': ['CCO', 'CCN', 'CC(C)O'],
            'MW': [46.07, 45.08, 60.10],
            'LogP': [0.31, 0.16, 0.31],
            'HBD': [1, 1, 1],
            'HBA': [1, 1, 1],
            'TPSA': [20.23, 12.03, 20.23]
        })
        
        # Test plot creation
        fig = self.utils.create_molecular_properties_plot(df)
        
        # Verify plot was created
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, go.Figure)
    
    def test_create_similarity_heatmap(self):
        """Test similarity heatmap creation."""
        # Create test data
        similarity_matrix = np.array([[1.0, 0.8, 0.6], [0.8, 1.0, 0.7], [0.6, 0.7, 1.0]])
        molecule_labels = ['Molecule_1', 'Molecule_2', 'Molecule_3']
        
        # Test heatmap creation
        fig = self.utils.create_similarity_heatmap(similarity_matrix, molecule_labels)
        
        # Verify heatmap was created
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, go.Figure)
    
    def test_create_prediction_results_plot(self):
        """Test prediction results plot creation."""
        # Create test data
        df = pd.DataFrame({
            'Molecule': ['Molecule_1', 'Molecule_2', 'Molecule_1', 'Molecule_2'],
            'SMILES': ['CCO', 'CCN', 'CCO', 'CCN'],
            'Target': ['5-HT2A', '5-HT2A', 'D2R', 'D2R'],
            'pIC50': [6.5, 5.8, 7.2, 6.9]
        })
        
        # Test plot creation
        fig = self.utils.create_prediction_results_plot(df)
        
        # Verify plot was created
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, go.Figure)
    
    def test_create_admet_radar_chart(self):
        """Test ADMET radar chart creation."""
        # Create test data
        admet_data = {
            'MW': 200.0,
            'LogP': 2.5,
            'HBD': 2,
            'HBA': 4,
            'TPSA': 80.0
        }
        
        # Test radar chart creation
        fig = self.utils.create_admet_radar_chart(admet_data)
        
        # Verify radar chart was created
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, go.Figure)
    
    def test_create_generation_history_plot(self):
        """Test generation history plot creation."""
        # Create test data
        history = {
            'fitness': [0.5, 0.6, 0.7, 0.8],
            'diversity': [0.8, 0.7, 0.6, 0.5]
        }
        
        # Test plot creation
        fig = self.utils.create_generation_history_plot(history)
        
        # Verify plot was created
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, go.Figure)
    
    def test_create_descriptors_correlation_plot(self):
        """Test descriptors correlation plot creation."""
        # Create test data
        df = pd.DataFrame({
            'SMILES': ['CCO', 'CCN', 'CC(C)O'],
            'Descriptor_1': [0.1, 0.2, 0.3],
            'Descriptor_2': [0.4, 0.5, 0.6],
            'Descriptor_3': [0.7, 0.8, 0.9]
        })
        
        # Test plot creation
        fig = self.utils.create_descriptors_correlation_plot(df)
        
        # Verify plot was created
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, go.Figure)
    
    def test_create_descriptors_correlation_plot_insufficient_data(self):
        """Test descriptors correlation plot with insufficient data."""
        # Create test data with insufficient numeric columns
        df = pd.DataFrame({
            'SMILES': ['CCO', 'CCN', 'CC(C)O'],
            'Descriptor_1': [0.1, 0.2, 0.3]
        })
        
        # Test plot creation
        fig = self.utils.create_descriptors_correlation_plot(df)
        
        # Verify empty figure is returned
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, go.Figure)
    
    def test_create_ensemble_prediction_plot(self):
        """Test ensemble prediction plot creation."""
        # Create test data
        predictions = {
            'Model_1': {'5-HT2A': 6.5, 'D2R': 7.2},
            'Model_2': {'5-HT2A': 6.8, 'D2R': 7.0},
            'Model_3': {'5-HT2A': 6.2, 'D2R': 7.5}
        }
        
        # Test plot creation
        fig = self.utils.create_ensemble_prediction_plot(predictions)
        
        # Verify plot was created
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, go.Figure)
    
    def test_create_cns_mpo_plot(self):
        """Test CNS-MPO plot creation."""
        # Create test data
        cns_mpo_scores = [0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.3, 0.2]
        
        # Test plot creation
        fig = self.utils.create_cns_mpo_plot(cns_mpo_scores)
        
        # Verify plot was created
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, go.Figure)
    
    def test_create_optimization_progress_plot(self):
        """Test optimization progress plot creation."""
        # Create test data
        progress_data = {
            'fitness': [0.5, 0.6, 0.7, 0.8],
            'diversity': [0.8, 0.7, 0.6, 0.5]
        }
        
        # Test plot creation
        fig = self.utils.create_optimization_progress_plot(progress_data)
        
        # Verify plot was created
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, go.Figure)
    
    def test_export_results_to_csv(self):
        """Test CSV export."""
        # Create test data
        data = [
            {'Molecule': 'CCO', 'MW': 46.07, 'LogP': 0.31},
            {'Molecule': 'CCN', 'MW': 45.08, 'LogP': 0.16}
        ]
        
        # Test CSV export
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            output_path = self.utils.export_results_to_csv(data, tmp.name)
            
            # Verify file was created
            self.assertTrue(os.path.exists(output_path))
            
            # Verify content
            df = pd.read_csv(output_path)
            self.assertEqual(len(df), 2)
            self.assertIn('Molecule', df.columns)
            self.assertIn('MW', df.columns)
            self.assertIn('LogP', df.columns)
            
            # Clean up
            os.unlink(output_path)
    
    def test_export_results_to_json(self):
        """Test JSON export."""
        # Create test data
        data = {
            'molecules': ['CCO', 'CCN', 'CC(C)O'],
            'properties': {'MW': [46.07, 45.08, 60.10]}
        }
        
        # Test JSON export
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            output_path = self.utils.export_results_to_json(data, tmp.name)
            
            # Verify file was created
            self.assertTrue(os.path.exists(output_path))
            
            # Verify content
            with open(output_path, 'r') as f:
                loaded_data = json.load(f)
            self.assertEqual(loaded_data, data)
            
            # Clean up
            os.unlink(output_path)
    
    def test_create_summary_statistics(self):
        """Test summary statistics creation."""
        # Create test data
        data = [
            {'MW': 46.07, 'LogP': 0.31, 'HBD': 1},
            {'MW': 45.08, 'LogP': 0.16, 'HBD': 1},
            {'MW': 60.10, 'LogP': 0.31, 'HBD': 1}
        ]
        
        # Test summary statistics creation
        summary = self.utils.create_summary_statistics(data)
        
        # Verify summary was created
        self.assertIsInstance(summary, dict)
        self.assertIn('MW', summary)
        self.assertIn('LogP', summary)
        self.assertIn('HBD', summary)
        
        # Verify summary contains expected keys
        for col in ['MW', 'LogP', 'HBD']:
            self.assertIn('mean', summary[col])
            self.assertIn('std', summary[col])
            self.assertIn('min', summary[col])
            self.assertIn('max', summary[col])
            self.assertIn('median', summary[col])
    
    def test_create_summary_statistics_empty(self):
        """Test summary statistics with empty data."""
        # Test with empty data
        summary = self.utils.create_summary_statistics([])
        
        # Verify empty summary is returned
        self.assertEqual(summary, {})
    
    def test_validate_smiles(self):
        """Test SMILES validation."""
        # Test valid SMILES
        valid_smiles = ['CCO', 'CCN', 'CC(C)O', 'C1=CC=CC=C1']
        for smiles in valid_smiles:
            self.assertTrue(self.utils.validate_smiles(smiles))
        
        # Test invalid SMILES
        invalid_smiles = ['', 'invalid', 'C@C', 'C#C#C']
        for smiles in invalid_smiles:
            self.assertFalse(self.utils.validate_smiles(smiles))
    
    def test_format_molecular_data(self):
        """Test molecular data formatting."""
        # Create test data
        molecules = ['CCO', 'CCN', 'CC(C)O']
        
        # Test formatting
        df = self.utils.format_molecular_data(molecules)
        
        # Verify formatting
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertIn('ID', df.columns)
        self.assertIn('SMILES', df.columns)
        self.assertIn('Length', df.columns)
        self.assertIn('Valid', df.columns)
        
        # Verify data
        self.assertEqual(df.iloc[0]['SMILES'], 'CCO')
        self.assertEqual(df.iloc[0]['Length'], 3)
        self.assertTrue(df.iloc[0]['Valid'])
    
    def test_create_download_link_csv(self):
        """Test CSV download link creation."""
        # Create test data
        data = [
            {'Molecule': 'CCO', 'MW': 46.07, 'LogP': 0.31},
            {'Molecule': 'CCN', 'MW': 45.08, 'LogP': 0.16}
        ]
        
        # Test download link creation
        link = self.utils.create_download_link(data, 'test.csv', 'csv')
        
        # Verify link was created
        self.assertIsInstance(link, str)
        self.assertIn('data:file/csv;base64,', link)
        self.assertIn('Download test.csv', link)
    
    def test_create_download_link_json(self):
        """Test JSON download link creation."""
        # Create test data
        data = {'molecules': ['CCO', 'CCN', 'CC(C)O']}
        
        # Test download link creation
        link = self.utils.create_download_link(data, 'test.json', 'json')
        
        # Verify link was created
        self.assertIsInstance(link, str)
        self.assertIn('data:file/json;base64,', link)
        self.assertIn('Download test.json', link)
    
    def test_create_download_link_invalid_type(self):
        """Test download link creation with invalid type."""
        # Test with invalid file type
        link = self.utils.create_download_link({'test': 'data'}, 'test.txt', 'txt')
        
        # Verify empty link is returned
        self.assertEqual(link, "")


if __name__ == '__main__':
    unittest.main()
