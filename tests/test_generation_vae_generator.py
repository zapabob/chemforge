"""
Unit tests for VAE generator.
"""

import unittest
import tempfile
import os
import torch
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path

from chemforge.generation.vae_generator import VAEGenerator, MolecularDataset, VAEEncoder, VAEDecoder


class TestMolecularDataset(unittest.TestCase):
    """Test MolecularDataset class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.smiles = ['CCO', 'CCN', 'CC(C)O', 'CC(C)N', 'CC(C)(C)O']
        self.features = np.random.randn(5, 10)
        self.max_length = 50
        
        self.dataset = MolecularDataset(
            smiles=self.smiles,
            features=self.features,
            max_length=self.max_length
        )
    
    def test_init(self):
        """Test MolecularDataset initialization."""
        self.assertEqual(len(self.dataset), 5)
        self.assertEqual(self.dataset.max_length, 50)
        self.assertIsNotNone(self.dataset.vocab)
        self.assertGreater(self.dataset.vocab_size, 0)
    
    def test_create_vocabulary(self):
        """Test vocabulary creation."""
        vocab = self.dataset._create_vocabulary()
        
        self.assertIn('<PAD>', vocab)
        self.assertIn('<START>', vocab)
        self.assertIn('<END>', vocab)
        self.assertIn('C', vocab)
        self.assertIn('O', vocab)
        self.assertIn('N', vocab)
    
    def test_smiles_to_indices(self):
        """Test SMILES to indices conversion."""
        indices = self.dataset._smiles_to_indices()
        
        self.assertEqual(len(indices), 5)
        self.assertEqual(len(indices[0]), self.max_length)
        
        # Check that all indices are valid
        for smile_indices in indices:
            for idx in smile_indices:
                self.assertGreaterEqual(idx, 0)
                self.assertLess(idx, self.dataset.vocab_size)
    
    def test_getitem(self):
        """Test dataset item retrieval."""
        item = self.dataset[0]
        
        self.assertIn('smiles_idx', item)
        self.assertIn('features', item)
        self.assertIn('smiles', item)
        
        self.assertEqual(item['smiles'], 'CCO')
        self.assertEqual(len(item['smiles_idx']), self.max_length)
        self.assertEqual(len(item['features']), 10)
    
    def test_getitem_without_features(self):
        """Test dataset item retrieval without features."""
        dataset_no_features = MolecularDataset(
            smiles=self.smiles,
            features=None,
            max_length=self.max_length
        )
        
        item = dataset_no_features[0]
        
        self.assertIn('smiles_idx', item)
        self.assertIn('smiles', item)
        self.assertNotIn('features', item)


class TestVAEEncoder(unittest.TestCase):
    """Test VAEEncoder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 100
        self.hidden_dim = 256
        self.latent_dim = 64
        
        self.encoder = VAEEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim
        )
    
    def test_init(self):
        """Test VAEEncoder initialization."""
        self.assertEqual(self.encoder.input_dim, self.input_dim)
        self.assertEqual(self.encoder.hidden_dim, self.hidden_dim)
        self.assertEqual(self.encoder.latent_dim, self.latent_dim)
    
    def test_forward(self):
        """Test VAEEncoder forward pass."""
        batch_size = 32
        x = torch.randn(batch_size, self.input_dim)
        
        mu, logvar = self.encoder(x)
        
        self.assertEqual(mu.shape, (batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (batch_size, self.latent_dim))
    
    def test_reparameterize(self):
        """Test reparameterization trick."""
        batch_size = 32
        mu = torch.randn(batch_size, self.latent_dim)
        logvar = torch.randn(batch_size, self.latent_dim)
        
        z = self.encoder.reparameterize(mu, logvar)
        
        self.assertEqual(z.shape, (batch_size, self.latent_dim))


class TestVAEDecoder(unittest.TestCase):
    """Test VAEDecoder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.latent_dim = 64
        self.hidden_dim = 256
        self.output_dim = 50
        self.vocab_size = 100
        
        self.decoder = VAEDecoder(
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            vocab_size=self.vocab_size
        )
    
    def test_init(self):
        """Test VAEDecoder initialization."""
        self.assertEqual(self.decoder.latent_dim, self.latent_dim)
        self.assertEqual(self.decoder.hidden_dim, self.hidden_dim)
        self.assertEqual(self.decoder.output_dim, self.output_dim)
        self.assertEqual(self.decoder.vocab_size, self.vocab_size)
    
    def test_forward(self):
        """Test VAEDecoder forward pass."""
        batch_size = 32
        z = torch.randn(batch_size, self.latent_dim)
        
        output = self.decoder(z)
        
        self.assertEqual(output.shape, (batch_size, self.output_dim, self.vocab_size))


class TestVAEGenerator(unittest.TestCase):
    """Test VAEGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, 'output')
        self.log_dir = os.path.join(self.temp_dir, 'logs')
        
        self.generator = VAEGenerator(
            input_dim=100,
            latent_dim=64,
            hidden_dim=256,
            max_length=50,
            vocab_size=100,
            device='cpu',
            output_dir=self.output_dir,
            log_dir=self.log_dir
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test VAEGenerator initialization."""
        self.assertEqual(self.generator.input_dim, 100)
        self.assertEqual(self.generator.latent_dim, 64)
        self.assertEqual(self.generator.hidden_dim, 256)
        self.assertEqual(self.generator.max_length, 50)
        self.assertEqual(self.generator.vocab_size, 100)
        self.assertEqual(self.generator.device, torch.device('cpu'))
        self.assertIsNotNone(self.generator.encoder)
        self.assertIsNotNone(self.generator.decoder)
    
    def test_get_device_auto_cpu(self):
        """Test device selection with auto on CPU."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=False):
                device = self.generator._get_device('auto')
                self.assertEqual(device, torch.device('cpu'))
    
    def test_get_device_auto_cuda(self):
        """Test device selection with auto on CUDA."""
        with patch('torch.cuda.is_available', return_value=True):
            device = self.generator._get_device('auto')
            self.assertEqual(device, torch.device('cuda'))
    
    def test_prepare_data(self):
        """Test data preparation."""
        smiles = ['CCO', 'CCN', 'CC(C)O', 'CC(C)N', 'CC(C)(C)O']
        features = np.random.randn(5, 100)
        
        train_loader, val_loader, test_loader = self.generator.prepare_data(
            smiles, features, test_size=0.2, val_size=0.1
        )
        
        # Verify loaders are created
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)
        
        # Verify vocabulary is created
        self.assertIsNotNone(self.generator.vocab)
        self.assertGreater(self.generator.vocab_size, 0)
    
    def test_train(self):
        """Test VAE training."""
        # Create mock data
        smiles = ['CCO', 'CCN', 'CC(C)O', 'CC(C)N', 'CC(C)(C)O'] * 10  # 50 molecules
        features = np.random.randn(50, 100)
        
        train_loader, val_loader, test_loader = self.generator.prepare_data(
            smiles, features, test_size=0.2, val_size=0.1
        )
        
        # Train for a few epochs
        training_results = self.generator.train(
            train_loader, val_loader, epochs=2, save_model=False
        )
        
        # Verify training results
        self.assertIsInstance(training_results, dict)
        self.assertIn('train_losses', training_results)
        self.assertIn('val_losses', training_results)
        self.assertEqual(len(training_results['train_losses']), 2)
        self.assertEqual(len(training_results['val_losses']), 2)
    
    def test_generate_molecules(self):
        """Test molecule generation."""
        # Create mock data and train briefly
        smiles = ['CCO', 'CCN', 'CC(C)O', 'CC(C)N', 'CC(C)(C)O'] * 10
        features = np.random.randn(50, 100)
        
        train_loader, val_loader, test_loader = self.generator.prepare_data(
            smiles, features, test_size=0.2, val_size=0.1
        )
        
        # Train briefly
        self.generator.train(train_loader, val_loader, epochs=1, save_model=False)
        
        # Generate molecules
        generated_smiles = self.generator.generate_molecules(num_molecules=10)
        
        # Verify generation
        self.assertIsInstance(generated_smiles, list)
        self.assertEqual(len(generated_smiles), 10)
        self.assertTrue(all(isinstance(s, str) for s in generated_smiles))
    
    def test_interpolate_molecules(self):
        """Test molecule interpolation."""
        # Create mock data and train briefly
        smiles = ['CCO', 'CCN', 'CC(C)O', 'CC(C)N', 'CC(C)(C)O'] * 10
        features = np.random.randn(50, 100)
        
        train_loader, val_loader, test_loader = self.generator.prepare_data(
            smiles, features, test_size=0.2, val_size=0.1
        )
        
        # Train briefly
        self.generator.train(train_loader, val_loader, epochs=1, save_model=False)
        
        # Test interpolation
        interpolated_smiles = self.generator.interpolate_molecules('CCO', 'CCN', num_steps=5)
        
        # Verify interpolation
        self.assertIsInstance(interpolated_smiles, list)
        self.assertEqual(len(interpolated_smiles), 5)
        self.assertTrue(all(isinstance(s, str) for s in interpolated_smiles))
    
    def test_tensor_to_smiles(self):
        """Test tensor to SMILES conversion."""
        # Create mock vocabulary
        self.generator.vocab = {'<PAD>': 0, '<START>': 1, '<END>': 2, 'C': 3, 'O': 4, 'N': 5}
        
        # Create mock tensor
        tensor = torch.tensor([
            [0, 0, 0, 0, 0],  # <PAD> tokens
            [1, 0, 0, 0, 0],  # <START> token
            [0, 0, 0, 0, 0],  # <PAD> tokens
            [0, 0, 0, 0, 0],  # <PAD> tokens
            [0, 0, 0, 0, 0]   # <PAD> tokens
        ])
        
        smiles = self.generator._tensor_to_smiles(tensor)
        
        # Verify conversion
        self.assertIsInstance(smiles, str)
    
    def test_save_model(self):
        """Test model saving."""
        # Create mock training history
        self.generator.training_history = {'train_losses': [1.0, 0.8]}
        
        # Save model
        model_path = os.path.join(self.temp_dir, 'test_model.pt')
        self.generator.save_model(model_path)
        
        # Verify file was created
        self.assertTrue(os.path.exists(model_path))
    
    def test_load_model(self):
        """Test model loading."""
        # Create mock checkpoint
        checkpoint = {
            'encoder_state_dict': self.generator.encoder.state_dict(),
            'decoder_state_dict': self.generator.decoder.state_dict(),
            'vocab': {'<PAD>': 0, '<START>': 1, '<END>': 2},
            'vocab_size': 100,
            'max_length': 50,
            'latent_dim': 64,
            'hidden_dim': 256,
            'training_history': {'train_losses': [1.0, 0.8]}
        }
        
        model_path = os.path.join(self.temp_dir, 'test_model.pt')
        torch.save(checkpoint, model_path)
        
        # Load model
        self.generator.load_model(model_path)
        
        # Verify loading
        self.assertIsNotNone(self.generator.vocab)
        self.assertEqual(self.generator.vocab_size, 100)
        self.assertEqual(self.generator.max_length, 50)
    
    def test_get_training_history(self):
        """Test getting training history."""
        # Set mock training history
        self.generator.training_history = {'train_losses': [1.0, 0.8]}
        
        history = self.generator.get_training_history()
        
        self.assertIsInstance(history, dict)
        self.assertIn('train_losses', history)


if __name__ == '__main__':
    unittest.main()
