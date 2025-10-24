"""
Unit tests for visualization utilities.
"""

import unittest
import tempfile
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from chemforge.utils.visualization import (
    MolecularVisualizer,
    ADMETVisualizer,
    CNSMPOVisualizer,
    ScaffoldVisualizer,
    TrainingVisualizer
)


class TestMolecularVisualizer(unittest.TestCase):
    """Test MolecularVisualizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.visualizer = MolecularVisualizer()
        
        # Create test data
        self.test_data = pd.DataFrame({
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
    
    def test_init(self):
        """Test MolecularVisualizer initialization."""
        self.assertIsNotNone(self.visualizer.logger)
    
    def test_plot_molecular_properties(self):
        """Test molecular properties plotting."""
        fig = self.visualizer.plot_molecular_properties(self.test_data)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_molecular_properties_custom(self):
        """Test molecular properties plotting with custom properties."""
        properties = ['mol_weight', 'logp', 'hbd']
        fig = self.visualizer.plot_molecular_properties(self.test_data, properties=properties)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_molecular_scatter(self):
        """Test molecular scatter plot."""
        fig = self.visualizer.plot_molecular_scatter(
            self.test_data, 'mol_weight', 'logp'
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_molecular_scatter_with_color(self):
        """Test molecular scatter plot with color property."""
        fig = self.visualizer.plot_molecular_scatter(
            self.test_data, 'mol_weight', 'logp', color_prop='hbd'
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_molecular_correlation(self):
        """Test molecular correlation matrix."""
        fig = self.visualizer.plot_molecular_correlation(self.test_data)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_molecular_correlation_custom(self):
        """Test molecular correlation matrix with custom properties."""
        properties = ['mol_weight', 'logp', 'hbd', 'hba']
        fig = self.visualizer.plot_molecular_correlation(self.test_data, properties=properties)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_molecular_properties_empty_data(self):
        """Test molecular properties plotting with empty data."""
        empty_data = pd.DataFrame()
        with self.assertRaises(ValueError):
            self.visualizer.plot_molecular_properties(empty_data)
    
    def test_plot_molecular_properties_no_properties(self):
        """Test molecular properties plotting with no valid properties."""
        data_no_props = pd.DataFrame({'smiles': ['CCO']})
        with self.assertRaises(ValueError):
            self.visualizer.plot_molecular_properties(data_no_props)


class TestADMETVisualizer(unittest.TestCase):
    """Test ADMETVisualizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.visualizer = ADMETVisualizer()
        
        # Create test data
        self.test_data = pd.DataFrame({
            'molecule_id': ['mol_1', 'mol_2', 'mol_3', 'mol_4'],
            'absorption': [0.8, 0.7, 0.9, 0.6],
            'distribution': [0.7, 0.8, 0.6, 0.9],
            'metabolism': [0.6, 0.7, 0.8, 0.5],
            'excretion': [0.9, 0.8, 0.7, 0.6],
            'toxicity': [0.3, 0.4, 0.2, 0.5]
        })
    
    def test_init(self):
        """Test ADMETVisualizer initialization."""
        self.assertIsNotNone(self.visualizer.logger)
    
    def test_plot_admet_radar(self):
        """Test ADMET radar chart."""
        fig = self.visualizer.plot_admet_radar(self.test_data, 'mol_1')
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_admet_radar_custom_properties(self):
        """Test ADMET radar chart with custom properties."""
        properties = ['absorption', 'distribution', 'metabolism']
        fig = self.visualizer.plot_admet_radar(
            self.test_data, 'mol_1', admet_props=properties
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_admet_radar_molecule_not_found(self):
        """Test ADMET radar chart with molecule not found."""
        with self.assertRaises(ValueError):
            self.visualizer.plot_admet_radar(self.test_data, 'nonexistent_mol')
    
    def test_plot_admet_radar_no_properties(self):
        """Test ADMET radar chart with no properties."""
        data_no_props = pd.DataFrame({'molecule_id': ['mol_1']})
        with self.assertRaises(ValueError):
            self.visualizer.plot_admet_radar(data_no_props, 'mol_1')
    
    def test_plot_admet_distribution(self):
        """Test ADMET distribution plot."""
        fig = self.visualizer.plot_admet_distribution(self.test_data)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_admet_distribution_custom_properties(self):
        """Test ADMET distribution plot with custom properties."""
        properties = ['absorption', 'distribution', 'metabolism']
        fig = self.visualizer.plot_admet_distribution(self.test_data, properties=properties)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_admet_heatmap(self):
        """Test ADMET heatmap."""
        fig = self.visualizer.plot_admet_heatmap(self.test_data)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_admet_heatmap_custom_properties(self):
        """Test ADMET heatmap with custom properties."""
        properties = ['absorption', 'distribution', 'metabolism']
        fig = self.visualizer.plot_admet_heatmap(self.test_data, properties=properties)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)


class TestCNSMPOVisualizer(unittest.TestCase):
    """Test CNSMPOVisualizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.visualizer = CNSMPOVisualizer()
        
        # Create test data
        self.test_data = pd.DataFrame({
            'molecule_id': ['mol_1', 'mol_2', 'mol_3', 'mol_4'],
            'cns_mpo_score': [4.5, 3.8, 5.2, 2.1],
            'mol_weight': [300.5, 250.3, 350.7, 200.1],
            'logp': [2.5, 1.8, 3.2, 0.9],
            'hbd': [2, 1, 3, 0],
            'hba': [4, 3, 5, 2],
            'tpsa': [60.5, 45.2, 75.8, 30.1]
        })
    
    def test_init(self):
        """Test CNSMPOVisualizer initialization."""
        self.assertIsNotNone(self.visualizer.logger)
    
    def test_plot_cns_mpo_scores(self):
        """Test CNS-MPO scores plotting."""
        fig = self.visualizer.plot_cns_mpo_scores(self.test_data)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_cns_mpo_scores_no_column(self):
        """Test CNS-MPO scores plotting with no CNS-MPO column."""
        data_no_cns_mpo = pd.DataFrame({'molecule_id': ['mol_1']})
        with self.assertRaises(ValueError):
            self.visualizer.plot_cns_mpo_scores(data_no_cns_mpo)
    
    def test_plot_cns_mpo_vs_properties(self):
        """Test CNS-MPO vs properties plotting."""
        fig = self.visualizer.plot_cns_mpo_vs_properties(self.test_data)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_cns_mpo_vs_properties_custom(self):
        """Test CNS-MPO vs properties plotting with custom properties."""
        properties = ['mol_weight', 'logp', 'hbd']
        fig = self.visualizer.plot_cns_mpo_vs_properties(self.test_data, properties=properties)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_cns_mpo_vs_properties_no_cns_mpo(self):
        """Test CNS-MPO vs properties plotting with no CNS-MPO column."""
        data_no_cns_mpo = pd.DataFrame({'mol_weight': [300.5]})
        with self.assertRaises(ValueError):
            self.visualizer.plot_cns_mpo_vs_properties(data_no_cns_mpo)


class TestScaffoldVisualizer(unittest.TestCase):
    """Test ScaffoldVisualizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.visualizer = ScaffoldVisualizer()
        
        # Create test data
        self.test_data = pd.DataFrame({
            'molecule_id': ['mol_1', 'mol_2', 'mol_3', 'mol_4', 'mol_5'],
            'scaffold_type': ['trivial', 'fund', 'adj', 'trivial', 'fund'],
            'mol_weight': [300.5, 250.3, 350.7, 200.1, 400.2],
            'logp': [2.5, 1.8, 3.2, 0.9, 4.1],
            'hbd': [2, 1, 3, 0, 4],
            'hba': [4, 3, 5, 2, 6],
            'tpsa': [60.5, 45.2, 75.8, 30.1, 90.3]
        })
    
    def test_init(self):
        """Test ScaffoldVisualizer initialization."""
        self.assertIsNotNone(self.visualizer.logger)
    
    def test_plot_scaffold_distribution(self):
        """Test scaffold distribution plotting."""
        fig = self.visualizer.plot_scaffold_distribution(self.test_data)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_scaffold_distribution_no_column(self):
        """Test scaffold distribution plotting with no scaffold column."""
        data_no_scaffold = pd.DataFrame({'molecule_id': ['mol_1']})
        with self.assertRaises(ValueError):
            self.visualizer.plot_scaffold_distribution(data_no_scaffold)
    
    def test_plot_scaffold_properties(self):
        """Test scaffold properties plotting."""
        fig = self.visualizer.plot_scaffold_properties(self.test_data)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_scaffold_properties_custom(self):
        """Test scaffold properties plotting with custom properties."""
        properties = ['mol_weight', 'logp', 'hbd']
        fig = self.visualizer.plot_scaffold_properties(self.test_data, properties=properties)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_scaffold_properties_no_scaffold(self):
        """Test scaffold properties plotting with no scaffold column."""
        data_no_scaffold = pd.DataFrame({'mol_weight': [300.5]})
        with self.assertRaises(ValueError):
            self.visualizer.plot_scaffold_properties(data_no_scaffold)


class TestTrainingVisualizer(unittest.TestCase):
    """Test TrainingVisualizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.visualizer = TrainingVisualizer()
        
        # Create test training data
        self.training_data = {
            'train_loss': [1.0, 0.8, 0.6, 0.4, 0.2],
            'val_loss': [1.1, 0.9, 0.7, 0.5, 0.3],
            'train_acc': [0.5, 0.6, 0.7, 0.8, 0.9],
            'val_acc': [0.4, 0.5, 0.6, 0.7, 0.8]
        }
        
        # Create test learning rate schedule
        self.lr_schedule = [0.001, 0.0008, 0.0006, 0.0004, 0.0002]
        
        # Create test model results
        self.model_results = {
            'Model A': {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.88},
            'Model B': {'accuracy': 0.87, 'precision': 0.85, 'recall': 0.89},
            'Model C': {'accuracy': 0.83, 'precision': 0.80, 'recall': 0.86}
        }
    
    def test_init(self):
        """Test TrainingVisualizer initialization."""
        self.assertIsNotNone(self.visualizer.logger)
    
    def test_plot_training_curves(self):
        """Test training curves plotting."""
        fig = self.visualizer.plot_training_curves(self.training_data)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_training_curves_partial_data(self):
        """Test training curves plotting with partial data."""
        partial_data = {'train_loss': [1.0, 0.8, 0.6]}
        fig = self.visualizer.plot_training_curves(partial_data)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_learning_rate_schedule(self):
        """Test learning rate schedule plotting."""
        fig = self.visualizer.plot_learning_rate_schedule(self.lr_schedule)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_model_comparison(self):
        """Test model comparison plotting."""
        fig = self.visualizer.plot_model_comparison(self.model_results)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_model_comparison_single_metric(self):
        """Test model comparison plotting with single metric."""
        single_metric_results = {
            'Model A': {'accuracy': 0.85},
            'Model B': {'accuracy': 0.87},
            'Model C': {'accuracy': 0.83}
        }
        fig = self.visualizer.plot_model_comparison(single_metric_results)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()

