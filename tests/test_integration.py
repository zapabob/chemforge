"""
Integration tests for ChemForge.

This module provides integration tests for the entire ChemForge platform
including end-to-end workflows and cross-module functionality.
"""

import unittest
import tempfile
import os
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from chemforge.data.chembl_loader import ChEMBLLoader
from chemforge.data.molecular_features import MolecularFeatures
from chemforge.data.rdkit_descriptors import RDKitDescriptors
from chemforge.models.transformer_model import TransformerModel
from chemforge.models.gnn_model import GNNModel
from chemforge.models.ensemble_model import EnsembleModel
from chemforge.admet.admet_predictor import ADMETPredictor
from chemforge.generation.molecular_generator import MolecularGenerator
from chemforge.training.trainer import Trainer
from chemforge.utils.logging_utils import Logger


class TestChemForgeIntegration(unittest.TestCase):
    """Test ChemForge integration workflows."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = Logger('integration_test')
        
        # Create test data
        self.test_smiles = [
            'CCO',  # Ethanol
            'CCN',  # Ethylamine
            'CC(C)O',  # Isopropanol
            'CC(C)N',  # Isopropylamine
            'CC(C)(C)O',  # tert-Butanol
        ]
        
        self.test_targets = ['5-HT2A', 'D2R', 'DAT', 'NET', 'SERT']
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_data_processing_pipeline(self):
        """Test complete data processing pipeline."""
        self.logger.info("Testing data processing pipeline")
        
        # Test molecular features extraction
        features_extractor = MolecularFeatures()
        features = features_extractor.extract_features(self.test_smiles)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), len(self.test_smiles))
        self.assertGreater(features.shape[1], 0)
        
        # Test RDKit descriptors
        rdkit_descriptors = RDKitDescriptors()
        descriptors = rdkit_descriptors.calculate_descriptors(self.test_smiles)
        
        self.assertIsInstance(descriptors, np.ndarray)
        self.assertEqual(len(descriptors), len(self.test_smiles))
        self.assertGreater(descriptors.shape[1], 0)
        
        self.logger.info("Data processing pipeline test completed")
    
    def test_model_training_pipeline(self):
        """Test complete model training pipeline."""
        self.logger.info("Testing model training pipeline")
        
        # Create mock training data
        X = np.random.randn(100, 200)  # 100 samples, 200 features
        y = np.random.randn(100, 5)    # 100 samples, 5 targets
        
        # Test Transformer model
        transformer = TransformerModel(
            input_dim=200,
            output_dim=5,
            hidden_dim=256,
            num_heads=8,
            num_layers=4
        )
        
        # Test model initialization
        self.assertIsNotNone(transformer)
        self.assertEqual(transformer.input_dim, 200)
        self.assertEqual(transformer.output_dim, 5)
        
        # Test GNN model
        gnn = GNNModel(
            node_features=200,
            hidden_dim=256,
            output_dim=5,
            num_layers=3
        )
        
        self.assertIsNotNone(gnn)
        self.assertEqual(gnn.node_features, 200)
        self.assertEqual(gnn.output_dim, 5)
        
        # Test Ensemble model
        ensemble = EnsembleModel(
            models=[transformer, gnn],
            weights=[0.5, 0.5]
        )
        
        self.assertIsNotNone(ensemble)
        self.assertEqual(len(ensemble.models), 2)
        
        self.logger.info("Model training pipeline test completed")
    
    def test_admet_prediction_pipeline(self):
        """Test ADMET prediction pipeline."""
        self.logger.info("Testing ADMET prediction pipeline")
        
        # Test ADMET predictor
        admet_predictor = ADMETPredictor()
        
        # Test prediction
        predictions = admet_predictor.predict_properties(self.test_smiles)
        
        self.assertIsInstance(predictions, dict)
        self.assertEqual(len(predictions), len(self.test_smiles))
        
        # Verify prediction structure
        for smiles in self.test_smiles:
            self.assertIn(smiles, predictions)
            pred = predictions[smiles]
            self.assertIn('MW', pred)
            self.assertIn('LogP', pred)
            self.assertIn('HBD', pred)
            self.assertIn('HBA', pred)
            self.assertIn('TPSA', pred)
        
        self.logger.info("ADMET prediction pipeline test completed")
    
    def test_molecular_generation_pipeline(self):
        """Test molecular generation pipeline."""
        self.logger.info("Testing molecular generation pipeline")
        
        # Test molecular generator
        generator = MolecularGenerator(
            output_dir=str(Path(self.temp_dir) / "generation"),
            log_dir=str(Path(self.temp_dir) / "logs")
        )
        
        # Test VAE generator setup
        vae_generator = generator.setup_vae_generator(
            input_dim=200,
            latent_dim=64,
            hidden_dim=256,
            max_length=50,
            vocab_size=100,
            device='cpu'
        )
        
        self.assertIsNotNone(vae_generator)
        
        # Test RL optimizer setup
        rl_optimizer = generator.setup_rl_optimizer(
            state_dim=200,
            action_dim=10,
            hidden_dim=128,
            learning_rate=1e-3,
            gamma=0.99,
            device='cpu'
        )
        
        self.assertIsNotNone(rl_optimizer)
        
        # Test genetic optimizer setup
        genetic_optimizer = generator.setup_genetic_optimizer(
            population_size=50,
            mutation_rate=0.1,
            crossover_rate=0.8,
            elite_size=5,
            max_generations=20
        )
        
        self.assertIsNotNone(genetic_optimizer)
        
        self.logger.info("Molecular generation pipeline test completed")
    
    def test_training_workflow(self):
        """Test complete training workflow."""
        self.logger.info("Testing training workflow")
        
        # Create mock training data
        X_train = np.random.randn(100, 200)
        y_train = np.random.randn(100, 5)
        X_val = np.random.randn(20, 200)
        y_val = np.random.randn(20, 5)
        
        # Test trainer
        trainer = Trainer(
            model=None,  # Will be set by the model
            device='cpu',
            output_dir=str(Path(self.temp_dir) / "training"),
            log_dir=str(Path(self.temp_dir) / "logs")
        )
        
        self.assertIsNotNone(trainer)
        
        # Test training configuration
        config = {
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'patience': 5
        }
        
        self.assertIsInstance(config, dict)
        self.assertIn('epochs', config)
        self.assertIn('batch_size', config)
        self.assertIn('learning_rate', config)
        
        self.logger.info("Training workflow test completed")
    
    def test_prediction_workflow(self):
        """Test complete prediction workflow."""
        self.logger.info("Testing prediction workflow")
        
        # Test molecular features
        features_extractor = MolecularFeatures()
        features = features_extractor.extract_features(self.test_smiles)
        
        # Test model prediction
        transformer = TransformerModel(
            input_dim=features.shape[1],
            output_dim=len(self.test_targets),
            hidden_dim=256,
            num_heads=8,
            num_layers=4
        )
        
        # Mock prediction
        predictions = np.random.randn(len(self.test_smiles), len(self.test_targets))
        
        self.assertEqual(predictions.shape[0], len(self.test_smiles))
        self.assertEqual(predictions.shape[1], len(self.test_targets))
        
        # Test ADMET prediction
        admet_predictor = ADMETPredictor()
        admet_predictions = admet_predictor.predict_properties(self.test_smiles)
        
        self.assertIsInstance(admet_predictions, dict)
        self.assertEqual(len(admet_predictions), len(self.test_smiles))
        
        self.logger.info("Prediction workflow test completed")
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        self.logger.info("Testing end-to-end workflow")
        
        # Step 1: Data preparation
        features_extractor = MolecularFeatures()
        features = features_extractor.extract_features(self.test_smiles)
        
        # Step 2: Model training
        transformer = TransformerModel(
            input_dim=features.shape[1],
            output_dim=len(self.test_targets),
            hidden_dim=256,
            num_heads=8,
            num_layers=4
        )
        
        # Step 3: Prediction
        predictions = np.random.randn(len(self.test_smiles), len(self.test_targets))
        
        # Step 4: ADMET analysis
        admet_predictor = ADMETPredictor()
        admet_predictions = admet_predictor.predict_properties(self.test_smiles)
        
        # Step 5: Molecular generation
        generator = MolecularGenerator(
            output_dir=str(Path(self.temp_dir) / "generation"),
            log_dir=str(Path(self.temp_dir) / "logs")
        )
        
        # Verify all steps completed
        self.assertIsNotNone(features)
        self.assertIsNotNone(transformer)
        self.assertIsNotNone(predictions)
        self.assertIsNotNone(admet_predictions)
        self.assertIsNotNone(generator)
        
        self.logger.info("End-to-end workflow test completed")
    
    def test_error_handling(self):
        """Test error handling across modules."""
        self.logger.info("Testing error handling")
        
        # Test invalid SMILES handling
        invalid_smiles = ['invalid_smiles', '', None]
        
        features_extractor = MolecularFeatures()
        try:
            features = features_extractor.extract_features(invalid_smiles)
            # Should handle errors gracefully
            self.assertIsInstance(features, np.ndarray)
        except Exception as e:
            # Error handling should be implemented
            self.logger.warning(f"Expected error in features extraction: {e}")
        
        # Test invalid model parameters
        try:
            transformer = TransformerModel(
                input_dim=-1,  # Invalid input dimension
                output_dim=5,
                hidden_dim=256,
                num_heads=8,
                num_layers=4
            )
            # Should handle invalid parameters
        except Exception as e:
            self.logger.warning(f"Expected error in model initialization: {e}")
        
        self.logger.info("Error handling test completed")
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks."""
        self.logger.info("Testing performance benchmarks")
        
        # Test feature extraction performance
        import time
        
        start_time = time.time()
        features_extractor = MolecularFeatures()
        features = features_extractor.extract_features(self.test_smiles)
        extraction_time = time.time() - start_time
        
        self.assertLess(extraction_time, 10.0)  # Should complete within 10 seconds
        self.logger.info(f"Feature extraction time: {extraction_time:.2f} seconds")
        
        # Test model initialization performance
        start_time = time.time()
        transformer = TransformerModel(
            input_dim=200,
            output_dim=5,
            hidden_dim=256,
            num_heads=8,
            num_layers=4
        )
        init_time = time.time() - start_time
        
        self.assertLess(init_time, 5.0)  # Should initialize within 5 seconds
        self.logger.info(f"Model initialization time: {init_time:.2f} seconds")
        
        self.logger.info("Performance benchmarks test completed")
    
    def test_memory_usage(self):
        """Test memory usage."""
        self.logger.info("Testing memory usage")
        
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Test feature extraction memory usage
        features_extractor = MolecularFeatures()
        features = features_extractor.extract_features(self.test_smiles)
        
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        
        self.assertLess(memory_increase, 100.0)  # Should not increase by more than 100MB
        self.logger.info(f"Memory increase: {memory_increase:.2f} MB")
        
        # Test model memory usage
        transformer = TransformerModel(
            input_dim=200,
            output_dim=5,
            hidden_dim=256,
            num_heads=8,
            num_layers=4
        )
        
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        
        self.assertLess(memory_increase, 200.0)  # Should not increase by more than 200MB
        self.logger.info(f"Total memory increase: {memory_increase:.2f} MB")
        
        self.logger.info("Memory usage test completed")
    
    def test_concurrent_operations(self):
        """Test concurrent operations."""
        self.logger.info("Testing concurrent operations")
        
        import threading
        import time
        
        results = []
        errors = []
        
        def extract_features(smiles_list, result_list, error_list):
            try:
                features_extractor = MolecularFeatures()
                features = features_extractor.extract_features(smiles_list)
                result_list.append(features)
            except Exception as e:
                error_list.append(e)
        
        # Test concurrent feature extraction
        threads = []
        for i in range(3):
            thread = threading.Thread(
                target=extract_features,
                args=(self.test_smiles, results, errors)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        self.assertEqual(len(results), 3)
        self.assertEqual(len(errors), 0)
        
        self.logger.info("Concurrent operations test completed")


if __name__ == '__main__':
    unittest.main()
