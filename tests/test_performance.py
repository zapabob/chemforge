"""
Performance tests for ChemForge.

This module provides performance benchmarks and tests for ChemForge components.
"""

import unittest
import time
import psutil
import os
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from chemforge.data.molecular_features import MolecularFeatures
from chemforge.data.rdkit_descriptors import RDKitDescriptors
from chemforge.models.transformer_model import TransformerModel
from chemforge.models.gnn_model import GNNModel
from chemforge.admet.admet_predictor import ADMETPredictor
from chemforge.utils.logging_utils import Logger


class TestChemForgePerformance(unittest.TestCase):
    """Test ChemForge performance benchmarks."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = Logger('performance_test')
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test data
        self.test_smiles = [
            'CCO', 'CCN', 'CC(C)O', 'CC(C)N', 'CC(C)(C)O',
            'C1=CC=CC=C1', 'C1=CC=CC=C1O', 'C1=CC=CC=C1N',
            'C1=CC=CC=C1C', 'C1=CC=CC=C1CC'
        ] * 10  # 100 molecules
        
        self.large_smiles = [
            'CCO', 'CCN', 'CC(C)O', 'CC(C)N', 'CC(C)(C)O'
        ] * 100  # 500 molecules
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_molecular_features_performance(self):
        """Test molecular features extraction performance."""
        self.logger.info("Testing molecular features performance")
        
        # Test with small dataset
        start_time = time.time()
        features_extractor = MolecularFeatures()
        features = features_extractor.extract_features(self.test_smiles)
        small_time = time.time() - start_time
        
        self.assertLess(small_time, 5.0)  # Should complete within 5 seconds
        self.logger.info(f"Small dataset ({len(self.test_smiles)} molecules): {small_time:.2f}s")
        
        # Test with large dataset
        start_time = time.time()
        features = features_extractor.extract_features(self.large_smiles)
        large_time = time.time() - start_time
        
        self.assertLess(large_time, 30.0)  # Should complete within 30 seconds
        self.logger.info(f"Large dataset ({len(self.large_smiles)} molecules): {large_time:.2f}s")
        
        # Test scalability
        scalability_ratio = large_time / small_time
        expected_ratio = len(self.large_smiles) / len(self.test_smiles)
        self.assertLess(scalability_ratio, expected_ratio * 2)  # Should scale reasonably
        
        self.logger.info(f"Scalability ratio: {scalability_ratio:.2f} (expected: {expected_ratio:.2f})")
    
    def test_rdkit_descriptors_performance(self):
        """Test RDKit descriptors performance."""
        self.logger.info("Testing RDKit descriptors performance")
        
        # Test with small dataset
        start_time = time.time()
        rdkit_descriptors = RDKitDescriptors()
        descriptors = rdkit_descriptors.calculate_descriptors(self.test_smiles)
        small_time = time.time() - start_time
        
        self.assertLess(small_time, 10.0)  # Should complete within 10 seconds
        self.logger.info(f"Small dataset ({len(self.test_smiles)} molecules): {small_time:.2f}s")
        
        # Test with large dataset
        start_time = time.time()
        descriptors = rdkit_descriptors.calculate_descriptors(self.large_smiles)
        large_time = time.time() - start_time
        
        self.assertLess(large_time, 60.0)  # Should complete within 60 seconds
        self.logger.info(f"Large dataset ({len(self.large_smiles)} molecules): {large_time:.2f}s")
    
    def test_model_initialization_performance(self):
        """Test model initialization performance."""
        self.logger.info("Testing model initialization performance")
        
        # Test Transformer model
        start_time = time.time()
        transformer = TransformerModel(
            input_dim=200,
            output_dim=5,
            hidden_dim=256,
            num_heads=8,
            num_layers=4
        )
        transformer_time = time.time() - start_time
        
        self.assertLess(transformer_time, 2.0)  # Should initialize within 2 seconds
        self.logger.info(f"Transformer initialization: {transformer_time:.2f}s")
        
        # Test GNN model
        start_time = time.time()
        gnn = GNNModel(
            node_features=200,
            hidden_dim=256,
            output_dim=5,
            num_layers=3
        )
        gnn_time = time.time() - start_time
        
        self.assertLess(gnn_time, 2.0)  # Should initialize within 2 seconds
        self.logger.info(f"GNN initialization: {gnn_time:.2f}s")
    
    def test_admet_prediction_performance(self):
        """Test ADMET prediction performance."""
        self.logger.info("Testing ADMET prediction performance")
        
        # Test with small dataset
        start_time = time.time()
        admet_predictor = ADMETPredictor()
        predictions = admet_predictor.predict_properties(self.test_smiles)
        small_time = time.time() - start_time
        
        self.assertLess(small_time, 5.0)  # Should complete within 5 seconds
        self.logger.info(f"Small dataset ({len(self.test_smiles)} molecules): {small_time:.2f}s")
        
        # Test with large dataset
        start_time = time.time()
        predictions = admet_predictor.predict_properties(self.large_smiles)
        large_time = time.time() - start_time
        
        self.assertLess(large_time, 30.0)  # Should complete within 30 seconds
        self.logger.info(f"Large dataset ({len(self.large_smiles)} molecules): {large_time:.2f}s")
    
    def test_memory_usage(self):
        """Test memory usage."""
        self.logger.info("Testing memory usage")
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Test feature extraction memory usage
        features_extractor = MolecularFeatures()
        features = features_extractor.extract_features(self.test_smiles)
        
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        
        self.assertLess(memory_increase, 100.0)  # Should not increase by more than 100MB
        self.logger.info(f"Feature extraction memory increase: {memory_increase:.2f} MB")
        
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
    
    def test_concurrent_performance(self):
        """Test concurrent operations performance."""
        self.logger.info("Testing concurrent operations performance")
        
        import threading
        import queue
        
        results = queue.Queue()
        errors = queue.Queue()
        
        def extract_features(smiles_list, result_queue, error_queue):
            try:
                features_extractor = MolecularFeatures()
                features = features_extractor.extract_features(smiles_list)
                result_queue.put(features)
            except Exception as e:
                error_queue.put(e)
        
        # Test concurrent feature extraction
        start_time = time.time()
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
        
        concurrent_time = time.time() - start_time
        
        # Verify results
        self.assertEqual(results.qsize(), 3)
        self.assertEqual(errors.qsize(), 0)
        
        self.assertLess(concurrent_time, 15.0)  # Should complete within 15 seconds
        self.logger.info(f"Concurrent operations time: {concurrent_time:.2f}s")
    
    def test_batch_processing_performance(self):
        """Test batch processing performance."""
        self.logger.info("Testing batch processing performance")
        
        # Test different batch sizes
        batch_sizes = [1, 5, 10, 20, 50]
        times = []
        
        for batch_size in batch_sizes:
            start_time = time.time()
            
            # Process in batches
            for i in range(0, len(self.test_smiles), batch_size):
                batch = self.test_smiles[i:i+batch_size]
                features_extractor = MolecularFeatures()
                features = features_extractor.extract_features(batch)
            
            batch_time = time.time() - start_time
            times.append(batch_time)
            
            self.logger.info(f"Batch size {batch_size}: {batch_time:.2f}s")
        
        # Verify that larger batch sizes are more efficient
        self.assertLess(times[-1], times[0])  # Largest batch should be faster than smallest
    
    def test_scalability(self):
        """Test scalability with increasing dataset sizes."""
        self.logger.info("Testing scalability")
        
        dataset_sizes = [10, 50, 100, 200, 500]
        times = []
        
        for size in dataset_sizes:
            smiles = self.test_smiles[:size]
            
            start_time = time.time()
            features_extractor = MolecularFeatures()
            features = features_extractor.extract_features(smiles)
            extraction_time = time.time() - start_time
            
            times.append(extraction_time)
            self.logger.info(f"Dataset size {size}: {extraction_time:.2f}s")
        
        # Verify reasonable scalability
        for i in range(1, len(times)):
            ratio = times[i] / times[i-1]
            size_ratio = dataset_sizes[i] / dataset_sizes[i-1]
            self.assertLess(ratio, size_ratio * 2)  # Should scale reasonably
    
    def test_error_handling_performance(self):
        """Test error handling performance."""
        self.logger.info("Testing error handling performance")
        
        # Test with invalid inputs
        invalid_smiles = ['invalid', '', None, 'C@C', 'C#C#C']
        
        start_time = time.time()
        features_extractor = MolecularFeatures()
        try:
            features = features_extractor.extract_features(invalid_smiles)
            # Should handle errors gracefully
        except Exception as e:
            # Error handling should be implemented
            pass
        
        error_time = time.time() - start_time
        
        self.assertLess(error_time, 5.0)  # Should handle errors quickly
        self.logger.info(f"Error handling time: {error_time:.2f}s")
    
    def test_benchmark_comparison(self):
        """Test benchmark comparison with baseline."""
        self.logger.info("Testing benchmark comparison")
        
        # Baseline performance
        baseline_time = 1.0  # 1 second baseline
        
        start_time = time.time()
        features_extractor = MolecularFeatures()
        features = features_extractor.extract_features(self.test_smiles)
        actual_time = time.time() - start_time
        
        # Should be within reasonable range of baseline
        self.assertLess(actual_time, baseline_time * 10)  # Within 10x of baseline
        self.logger.info(f"Baseline: {baseline_time:.2f}s, Actual: {actual_time:.2f}s")
    
    def test_resource_utilization(self):
        """Test resource utilization."""
        self.logger.info("Testing resource utilization")
        
        # Test CPU usage
        start_time = time.time()
        features_extractor = MolecularFeatures()
        features = features_extractor.extract_features(self.test_smiles)
        processing_time = time.time() - start_time
        
        # Test memory usage
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        self.assertLess(memory_usage, 1000.0)  # Should not use more than 1GB
        self.logger.info(f"Memory usage: {memory_usage:.2f} MB")
        self.logger.info(f"Processing time: {processing_time:.2f}s")


if __name__ == '__main__':
    unittest.main()
