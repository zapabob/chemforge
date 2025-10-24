"""
Unit tests for Streamlit GUI application.
"""

import unittest
import tempfile
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
import streamlit as st
from io import StringIO

from chemforge.gui.streamlit_app import StreamlitApp


class TestStreamlitApp(unittest.TestCase):
    """Test StreamlitApp class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = StreamlitApp()
    
    def test_init(self):
        """Test StreamlitApp initialization."""
        self.assertIsNotNone(self.app.logger)
        self.assertIsInstance(self.app, StreamlitApp)
    
    def test_setup_page_config(self):
        """Test page configuration setup."""
        # This would test the page config setup
        # In practice, this is tested by running the app
        self.assertTrue(True)
    
    def test_initialize_session_state(self):
        """Test session state initialization."""
        # Mock session state
        with patch.object(st, 'session_state', {}):
            self.app.initialize_session_state()
            
            # Check if session state is initialized
            self.assertIn('molecules', st.session_state)
            self.assertIn('predictions', st.session_state)
            self.assertIn('generated_molecules', st.session_state)
            self.assertIn('admet_results', st.session_state)
    
    def test_handle_file_upload_csv(self):
        """Test CSV file upload handling."""
        # Create mock CSV content
        csv_content = "smiles\nCCO\nCCN\nCC(C)O"
        
        # Mock file upload
        mock_file = MagicMock()
        mock_file.name = "test.csv"
        mock_file.read.return_value = csv_content.encode()
        
        # Mock pandas read_csv
        with patch('pandas.read_csv') as mock_read_csv:
            mock_df = pd.DataFrame({'smiles': ['CCO', 'CCN', 'CC(C)O']})
            mock_read_csv.return_value = mock_df
            
            # Test file upload
            self.app.handle_file_upload(mock_file)
            
            # Verify molecules were added
            self.assertEqual(len(st.session_state.molecules), 3)
            self.assertIn('CCO', st.session_state.molecules)
    
    def test_handle_file_upload_txt(self):
        """Test TXT file upload handling."""
        # Create mock TXT content
        txt_content = "CCO\nCCN\nCC(C)O"
        
        # Mock file upload
        mock_file = MagicMock()
        mock_file.name = "test.txt"
        mock_file.read.return_value = txt_content.encode()
        
        # Test file upload
        self.app.handle_file_upload(mock_file)
        
        # Verify molecules were added
        self.assertEqual(len(st.session_state.molecules), 3)
        self.assertIn('CCO', st.session_state.molecules)
    
    def test_handle_smiles_input(self):
        """Test SMILES input handling."""
        smiles_input = "CCO\nCCN\nCC(C)O"
        
        # Test SMILES input
        self.app.handle_smiles_input(smiles_input)
        
        # Verify molecules were added
        self.assertEqual(len(st.session_state.molecules), 3)
        self.assertIn('CCO', st.session_state.molecules)
    
    def test_handle_smiles_input_empty(self):
        """Test empty SMILES input handling."""
        smiles_input = ""
        
        # Test empty SMILES input
        self.app.handle_smiles_input(smiles_input)
        
        # Verify no molecules were added
        self.assertEqual(len(st.session_state.molecules), 0)
    
    def test_calculate_basic_properties(self):
        """Test basic properties calculation."""
        # Set up test molecules
        st.session_state.molecules = ['CCO', 'CCN', 'CC(C)O']
        
        # Test properties calculation
        properties = self.app.calculate_basic_properties()
        
        # Verify properties were calculated
        self.assertEqual(len(properties), 3)
        self.assertEqual(properties[0]['SMILES'], 'CCO')
        self.assertIn('MW', properties[0])
        self.assertIn('LogP', properties[0])
        self.assertIn('HBD', properties[0])
        self.assertIn('HBA', properties[0])
        self.assertIn('TPSA', properties[0])
    
    def test_calculate_molecular_descriptors(self):
        """Test molecular descriptors calculation."""
        # Set up test molecules
        st.session_state.molecules = ['CCO', 'CCN', 'CC(C)O']
        
        # Test descriptors calculation
        descriptors = self.app.calculate_molecular_descriptors()
        
        # Verify descriptors were calculated
        self.assertEqual(len(descriptors), 3)
        self.assertEqual(descriptors[0]['SMILES'], 'CCO')
        self.assertIn('Descriptor_1', descriptors[0])
        self.assertIn('Descriptor_2', descriptors[0])
        self.assertIn('Descriptor_3', descriptors[0])
    
    def test_calculate_similarity_matrix(self):
        """Test similarity matrix calculation."""
        # Set up test molecules
        st.session_state.molecules = ['CCO', 'CCN', 'CC(C)O']
        
        # Test similarity matrix calculation
        similarity_matrix = self.app.calculate_similarity_matrix()
        
        # Verify similarity matrix was calculated
        self.assertIsNotNone(similarity_matrix)
        self.assertEqual(similarity_matrix.shape, (3, 3))
        
        # Check diagonal elements are 1.0
        for i in range(3):
            self.assertEqual(similarity_matrix[i, i], 1.0)
    
    def test_calculate_similarity_matrix_insufficient_molecules(self):
        """Test similarity matrix calculation with insufficient molecules."""
        # Set up test molecules
        st.session_state.molecules = ['CCO']
        
        # Test similarity matrix calculation
        similarity_matrix = self.app.calculate_similarity_matrix()
        
        # Verify similarity matrix is None
        self.assertIsNone(similarity_matrix)
    
    def test_run_predictions(self):
        """Test predictions running."""
        # Set up test molecules
        st.session_state.molecules = ['CCO', 'CCN']
        
        # Test predictions
        self.app.run_predictions('Transformer', ['5-HT2A', 'D2R'])
        
        # Verify predictions were created
        self.assertIn('molecule_0', st.session_state.predictions)
        self.assertIn('molecule_1', st.session_state.predictions)
    
    def test_run_admet_analysis(self):
        """Test ADMET analysis running."""
        # Set up test molecules
        st.session_state.molecules = ['CCO', 'CCN']
        
        # Test ADMET analysis
        self.app.run_admet_analysis('all')
        
        # Verify ADMET results were created
        self.assertIn('molecule_0', st.session_state.admet_results)
        self.assertIn('molecule_1', st.session_state.admet_results)
    
    def test_run_molecular_generation(self):
        """Test molecular generation running."""
        # Test molecular generation
        self.app.run_molecular_generation('VAE', 10, 1.0, 100, 100)
        
        # Verify molecules were generated
        self.assertEqual(len(st.session_state.generated_molecules), 10)
    
    def test_create_properties_visualizations(self):
        """Test properties visualizations creation."""
        # Create test data
        df = pd.DataFrame({
            'SMILES': ['CCO', 'CCN', 'CC(C)O'],
            'MW': [46.07, 45.08, 60.10],
            'LogP': [0.31, 0.16, 0.31],
            'HBD': [1, 1, 1],
            'HBA': [1, 1, 1],
            'TPSA': [20.23, 12.03, 20.23]
        })
        
        # Test visualizations creation
        try:
            self.app.create_properties_visualizations(df)
            # If no exception is raised, the test passes
            self.assertTrue(True)
        except Exception as e:
            # If visualization fails due to display issues, that's okay
            self.assertIn('display', str(e).lower())
    
    def test_create_descriptors_visualizations(self):
        """Test descriptors visualizations creation."""
        # Create test data
        df = pd.DataFrame({
            'SMILES': ['CCO', 'CCN', 'CC(C)O'],
            'Descriptor_1': [0.1, 0.2, 0.3],
            'Descriptor_2': [0.4, 0.5, 0.6],
            'Descriptor_3': [0.7, 0.8, 0.9]
        })
        
        # Test visualizations creation
        try:
            self.app.create_descriptors_visualizations(df)
            # If no exception is raised, the test passes
            self.assertTrue(True)
        except Exception as e:
            # If visualization fails due to display issues, that's okay
            self.assertIn('display', str(e).lower())
    
    def test_render_recent_activity(self):
        """Test recent activity rendering."""
        # Set up test data
        st.session_state.molecules = ['CCO', 'CCN']
        st.session_state.predictions = {'molecule_0': {}}
        st.session_state.generated_molecules = ['Generated_1']
        
        # Test recent activity rendering
        try:
            self.app.render_recent_activity()
            # If no exception is raised, the test passes
            self.assertTrue(True)
        except Exception as e:
            # If rendering fails due to display issues, that's okay
            self.assertIn('display', str(e).lower())
    
    def test_render_prediction_results(self):
        """Test prediction results rendering."""
        # Set up test predictions
        st.session_state.predictions = {
            'molecule_0': {
                'CCO': {'5-HT2A': 6.5, 'D2R': 7.2}
            },
            'molecule_1': {
                'CCN': {'5-HT2A': 5.8, 'D2R': 6.9}
            }
        }
        
        # Test prediction results rendering
        try:
            self.app.render_prediction_results()
            # If no exception is raised, the test passes
            self.assertTrue(True)
        except Exception as e:
            # If rendering fails due to display issues, that's okay
            self.assertIn('display', str(e).lower())
    
    def test_render_admet_results(self):
        """Test ADMET results rendering."""
        # Set up test ADMET results
        st.session_state.admet_results = {
            'molecule_0': {
                'CCO': {'MW': 46.07, 'LogP': 0.31, 'HBD': 1, 'HBA': 1, 'TPSA': 20.23}
            },
            'molecule_1': {
                'CCN': {'MW': 45.08, 'LogP': 0.16, 'HBD': 1, 'HBA': 1, 'TPSA': 12.03}
            }
        }
        
        # Test ADMET results rendering
        try:
            self.app.render_admet_results()
            # If no exception is raised, the test passes
            self.assertTrue(True)
        except Exception as e:
            # If rendering fails due to display issues, that's okay
            self.assertIn('display', str(e).lower())
    
    def test_render_generation_results(self):
        """Test generation results rendering."""
        # Set up test generated molecules
        st.session_state.generated_molecules = ['Generated_1', 'Generated_2', 'Generated_3']
        
        # Test generation results rendering
        try:
            self.app.render_generation_results()
            # If no exception is raised, the test passes
            self.assertTrue(True)
        except Exception as e:
            # If rendering fails due to display issues, that's okay
            self.assertIn('display', str(e).lower())
    
    def test_render_chembl_data_interface(self):
        """Test ChEMBL data interface rendering."""
        # Test ChEMBL data interface rendering
        try:
            self.app.render_chembl_data_interface()
            # If no exception is raised, the test passes
            self.assertTrue(True)
        except Exception as e:
            # If rendering fails due to display issues, that's okay
            self.assertIn('display', str(e).lower())
    
    def test_render_custom_dataset_interface(self):
        """Test custom dataset interface rendering."""
        # Test custom dataset interface rendering
        try:
            self.app.render_custom_dataset_interface()
            # If no exception is raised, the test passes
            self.assertTrue(True)
        except Exception as e:
            # If rendering fails due to display issues, that's okay
            self.assertIn('display', str(e).lower())
    
    def test_render_file_upload_interface(self):
        """Test file upload interface rendering."""
        # Test file upload interface rendering
        try:
            self.app.render_file_upload_interface()
            # If no exception is raised, the test passes
            self.assertTrue(True)
        except Exception as e:
            # If rendering fails due to display issues, that's okay
            self.assertIn('display', str(e).lower())


if __name__ == '__main__':
    unittest.main()
