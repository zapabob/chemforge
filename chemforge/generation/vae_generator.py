"""
Variational Autoencoder (VAE) for molecular generation.

This module provides VAE-based molecular generation capabilities
for creating novel molecules with desired properties.
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from chemforge.utils.logging_utils import Logger
from chemforge.utils.validation import DataValidator


class MolecularDataset(Dataset):
    """
    Dataset class for molecular data.
    
    This class handles molecular data for VAE training,
    including SMILES strings and molecular features.
    """
    
    def __init__(
        self,
        smiles: List[str],
        features: Optional[np.ndarray] = None,
        max_length: int = 100
    ):
        """
        Initialize the molecular dataset.
        
        Args:
            smiles: List of SMILES strings
            features: Optional molecular features
            max_length: Maximum SMILES length for padding
        """
        self.smiles = smiles
        self.features = features
        self.max_length = max_length
        
        # Create vocabulary from SMILES
        self.vocab = self._create_vocabulary()
        self.vocab_size = len(self.vocab)
        
        # Convert SMILES to indices
        self.smiles_indices = self._smiles_to_indices()
        
        self.logger = Logger('molecular_dataset')
        self.logger.info(f"Created dataset with {len(smiles)} molecules")
        self.logger.info(f"Vocabulary size: {self.vocab_size}")
    
    def _create_vocabulary(self) -> Dict[str, int]:
        """Create vocabulary from SMILES strings."""
        vocab = {'<PAD>': 0, '<START>': 1, '<END>': 2}
        
        for smiles in self.smiles:
            for char in smiles:
                if char not in vocab:
                    vocab[char] = len(vocab)
        
        return vocab
    
    def _smiles_to_indices(self) -> List[List[int]]:
        """Convert SMILES strings to indices."""
        indices = []
        
        for smiles in self.smiles:
            # Add start token
            smile_indices = [self.vocab['<START>']]
            
            # Add character indices
            for char in smiles:
                if char in self.vocab:
                    smile_indices.append(self.vocab[char])
                else:
                    smile_indices.append(self.vocab['<PAD>'])
            
            # Add end token
            smile_indices.append(self.vocab['<END>'])
            
            # Pad to max_length
            while len(smile_indices) < self.max_length:
                smile_indices.append(self.vocab['<PAD>'])
            
            # Truncate if too long
            smile_indices = smile_indices[:self.max_length]
            
            indices.append(smile_indices)
        
        return indices
    
    def __len__(self):
        """Return the number of molecules."""
        return len(self.smiles)
    
    def __getitem__(self, idx):
        """Get a molecule by index."""
        smiles_idx = torch.tensor(self.smiles_indices[idx], dtype=torch.long)
        
        if self.features is not None:
            features = torch.tensor(self.features[idx], dtype=torch.float32)
            return {
                'smiles_idx': smiles_idx,
                'features': features,
                'smiles': self.smiles[idx]
            }
        else:
            return {
                'smiles_idx': smiles_idx,
                'smiles': self.smiles[idx]
            }


class VAEEncoder(nn.Module):
    """
    VAE Encoder for molecular generation.
    
    This encoder takes molecular representations and encodes them
    into a latent space for generation.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        latent_dim: int = 128,
        dropout: float = 0.1
    ):
        """
        Initialize the VAE encoder.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            latent_dim: Latent space dimension
            dropout: Dropout rate
        """
        super(VAEEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Mean and log variance layers
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        
        self.logger = Logger('vae_encoder')
        self.logger.info(f"VAE Encoder initialized: {input_dim} -> {latent_dim}")
    
    def forward(self, x):
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (mu, logvar) for latent space
        """
        # Encode to hidden representation
        hidden = self.encoder(x)
        
        # Get mean and log variance
        mu = self.mu_layer(hidden)
        logvar = self.logvar_layer(hidden)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for VAE.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class VAEDecoder(nn.Module):
    """
    VAE Decoder for molecular generation.
    
    This decoder takes latent representations and decodes them
    into molecular representations.
    """
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 100,
        vocab_size: int = 100,
        dropout: float = 0.1
    ):
        """
        Initialize the VAE decoder.
        
        Args:
            latent_dim: Latent space dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output sequence length
            vocab_size: Vocabulary size
            dropout: Dropout rate
        """
        super(VAEDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.vocab_size = vocab_size
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim * vocab_size)
        
        self.logger = Logger('vae_decoder')
        self.logger.info(f"VAE Decoder initialized: {latent_dim} -> {output_dim}x{vocab_size}")
    
    def forward(self, z):
        """
        Forward pass through the decoder.
        
        Args:
            z: Latent tensor
            
        Returns:
            Decoded molecular representation
        """
        # Decode from latent space
        hidden = self.decoder(z)
        
        # Reshape to sequence
        output = self.output_layer(hidden)
        output = output.view(-1, self.output_dim, self.vocab_size)
        
        return output


class VAEGenerator:
    """
    VAE-based molecular generator.
    
    This class provides VAE-based molecular generation capabilities
    for creating novel molecules with desired properties.
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 128,
        hidden_dim: int = 512,
        max_length: int = 100,
        vocab_size: int = 100,
        device: str = 'auto',
        output_dir: str = "./vae_models",
        log_dir: str = "./logs"
    ):
        """
        Initialize the VAE generator.
        
        Args:
            input_dim: Input dimension
            latent_dim: Latent space dimension
            hidden_dim: Hidden layer dimension
            max_length: Maximum sequence length
            vocab_size: Vocabulary size
            device: Device to use for training
            output_dir: Directory to save models
            log_dir: Directory for logs
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.device = self._get_device(device)
        
        # Create directories
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.logger = Logger('vae_generator', log_dir=str(self.log_dir))
        self.data_validator = DataValidator()
        
        # Initialize VAE components
        self.encoder = VAEEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        ).to(self.device)
        
        self.decoder = VAEDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=max_length,
            vocab_size=vocab_size
        ).to(self.device)
        
        # Training state
        self.training_history = {}
        self.vocab = None
        self.scaler = None
        
        self.logger.info(f"VAE Generator initialized with device: {self.device}")
    
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device for training."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def prepare_data(
        self,
        smiles: List[str],
        features: Optional[np.ndarray] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare data for VAE training.
        
        Args:
            smiles: List of SMILES strings
            features: Optional molecular features
            test_size: Fraction of data for testing
            val_size: Fraction of data for validation
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        self.logger.info("Preparing data for VAE training")
        
        # Create dataset
        dataset = MolecularDataset(smiles, features, self.max_length)
        self.vocab = dataset.vocab
        self.vocab_size = dataset.vocab_size
        
        # Update decoder vocabulary size
        self.decoder.vocab_size = self.vocab_size
        self.decoder.output_layer = nn.Linear(
            self.hidden_dim, self.max_length * self.vocab_size
        ).to(self.device)
        
        # Split dataset
        from sklearn.model_selection import train_test_split
        
        train_data, test_data = train_test_split(
            dataset, test_size=test_size, random_state=random_state
        )
        train_data, val_data = train_test_split(
            train_data, test_size=val_size, random_state=random_state
        )
        
        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        
        self.logger.info(f"Data prepared: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        return train_loader, val_loader, test_loader
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        beta: float = 1.0,
        save_model: bool = True
    ) -> Dict[str, Any]:
        """
        Train the VAE model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate
            beta: Beta parameter for VAE loss
            save_model: Whether to save the trained model
            
        Returns:
            Training results dictionary
        """
        self.logger.info("Training VAE model")
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate
        )
        
        # Training history
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.encoder.train()
            self.decoder.train()
            
            train_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Get batch data
                smiles_idx = batch['smiles_idx'].to(self.device)
                
                # Forward pass
                mu, logvar = self.encoder(smiles_idx.float())
                z = self.encoder.reparameterize(mu, logvar)
                recon = self.decoder(z)
                
                # Calculate loss
                recon_loss = F.cross_entropy(
                    recon.view(-1, self.vocab_size),
                    smiles_idx.view(-1),
                    ignore_index=0  # Ignore padding tokens
                )
                
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kl_loss = kl_loss / smiles_idx.size(0)
                
                total_loss = recon_loss + beta * kl_loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                train_loss += total_loss.item()
            
            # Validation
            self.encoder.eval()
            self.decoder.eval()
            
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    smiles_idx = batch['smiles_idx'].to(self.device)
                    
                    mu, logvar = self.encoder(smiles_idx.float())
                    z = self.encoder.reparameterize(mu, logvar)
                    recon = self.decoder(z)
                    
                    recon_loss = F.cross_entropy(
                        recon.view(-1, self.vocab_size),
                        smiles_idx.view(-1),
                        ignore_index=0
                    )
                    
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    kl_loss = kl_loss / smiles_idx.size(0)
                    
                    total_loss = recon_loss + beta * kl_loss
                    val_loss += total_loss.item()
            
            # Store losses
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss / len(val_loader))
            
            # Log progress
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: Train Loss = {train_losses[-1]:.4f}, Val Loss = {val_losses[-1]:.4f}")
        
        # Store training history
        self.training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'beta': beta
        }
        
        # Save model if requested
        if save_model:
            self.save_model()
        
        self.logger.info("VAE training completed")
        
        return self.training_history
    
    def generate_molecules(
        self,
        num_molecules: int = 100,
        temperature: float = 1.0
    ) -> List[str]:
        """
        Generate new molecules using the trained VAE.
        
        Args:
            num_molecules: Number of molecules to generate
            temperature: Sampling temperature
            
        Returns:
            List of generated SMILES strings
        """
        self.logger.info(f"Generating {num_molecules} molecules")
        
        self.encoder.eval()
        self.decoder.eval()
        
        generated_smiles = []
        
        with torch.no_grad():
            for _ in range(num_molecules):
                # Sample from latent space
                z = torch.randn(1, self.latent_dim).to(self.device)
                
                # Decode to molecular representation
                output = self.decoder(z)
                
                # Convert to SMILES
                smiles = self._tensor_to_smiles(output[0], temperature)
                generated_smiles.append(smiles)
        
        self.logger.info(f"Generated {len(generated_smiles)} molecules")
        
        return generated_smiles
    
    def _tensor_to_smiles(self, tensor: torch.Tensor, temperature: float = 1.0) -> str:
        """Convert tensor to SMILES string."""
        if self.vocab is None:
            return ""
        
        # Get indices with temperature sampling
        probs = F.softmax(tensor / temperature, dim=-1)
        indices = torch.multinomial(probs, 1).squeeze(-1)
        
        # Convert indices to characters
        chars = []
        for idx in indices:
            if idx.item() == self.vocab['<END>']:
                break
            elif idx.item() == self.vocab['<PAD>']:
                continue
            elif idx.item() == self.vocab['<START>']:
                continue
            else:
                for char, char_idx in self.vocab.items():
                    if char_idx == idx.item():
                        chars.append(char)
                        break
        
        return ''.join(chars)
    
    def interpolate_molecules(
        self,
        smiles1: str,
        smiles2: str,
        num_steps: int = 10
    ) -> List[str]:
        """
        Interpolate between two molecules in latent space.
        
        Args:
            smiles1: First SMILES string
            smiles2: Second SMILES string
            num_steps: Number of interpolation steps
            
        Returns:
            List of interpolated SMILES strings
        """
        self.logger.info(f"Interpolating between {smiles1} and {smiles2}")
        
        self.encoder.eval()
        self.decoder.eval()
        
        # Encode molecules to latent space
        z1 = self._encode_smiles(smiles1)
        z2 = self._encode_smiles(smiles2)
        
        # Interpolate in latent space
        interpolated_smiles = []
        
        with torch.no_grad():
            for i in range(num_steps):
                alpha = i / (num_steps - 1)
                z_interp = (1 - alpha) * z1 + alpha * z2
                
                # Decode interpolated latent vector
                output = self.decoder(z_interp)
                smiles = self._tensor_to_smiles(output[0])
                interpolated_smiles.append(smiles)
        
        return interpolated_smiles
    
    def _encode_smiles(self, smiles: str) -> torch.Tensor:
        """Encode a SMILES string to latent space."""
        if self.vocab is None:
            return torch.randn(1, self.latent_dim).to(self.device)
        
        # Convert SMILES to indices
        indices = [self.vocab['<START>']]
        for char in smiles:
            if char in self.vocab:
                indices.append(self.vocab[char])
        indices.append(self.vocab['<END>'])
        
        # Pad to max_length
        while len(indices) < self.max_length:
            indices.append(self.vocab['<PAD>'])
        indices = indices[:self.max_length]
        
        # Convert to tensor
        smiles_tensor = torch.tensor(indices, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Encode to latent space
        with torch.no_grad():
            mu, logvar = self.encoder(smiles_tensor)
            z = self.encoder.reparameterize(mu, logvar)
        
        return z
    
    def save_model(self, filepath: Optional[str] = None):
        """Save the trained VAE model."""
        if filepath is None:
            filepath = self.output_dir / "vae_model.pt"
        
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'vocab': self.vocab,
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'latent_dim': self.latent_dim,
            'hidden_dim': self.hidden_dim,
            'training_history': self.training_history
        }, filepath)
        
        self.logger.info(f"Saved VAE model to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained VAE model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.vocab = checkpoint['vocab']
        self.vocab_size = checkpoint['vocab_size']
        self.max_length = checkpoint['max_length']
        self.training_history = checkpoint.get('training_history', {})
        
        self.logger.info(f"Loaded VAE model from {filepath}")
    
    def get_training_history(self) -> Dict[str, Any]:
        """Get the training history."""
        return self.training_history
