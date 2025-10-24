"""
Training utilities for ChemForge platform.

This module provides training functionality including model training,
validation, and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Trainer:
    """Base trainer class for model training."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config: Training configuration
        """
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        optimizer_type = self.config.get('optimizer', 'adam')
        lr = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        scheduler_type = self.config.get('scheduler', None)
        if scheduler_type is None:
            return None
        
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.get('epochs', 100)
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
            else:
                batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(batch)
            
            # Compute loss
            if isinstance(output, dict):
                if 'loss' not in output:
                    raise ValueError("Model output dictionary must contain 'loss' key")
                loss = output['loss']
            else:
                loss = output
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                else:
                    batch = batch.to(self.device)
                
                # Forward pass
                output = self.model(batch)
                
                # Compute loss
                if isinstance(output, dict):
                    loss = output.get('loss', torch.tensor(0.0))
                else:
                    loss = output
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """
        Train model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Training history
        """
        epochs = self.config.get('epochs', 100)
        history = {'train_loss': [], 'val_loss': []}
        
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            # Training
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics['train_loss'])
            
            # Validation
            val_metrics = self.validate(val_loader)
            history['val_loss'].append(val_metrics['val_loss'])
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Log progress
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: train_loss={train_metrics['train_loss']:.4f}, "
                           f"val_loss={val_metrics['val_loss']:.4f}")
        
        logger.info("Training completed")
        return history
    
    def save_checkpoint(self, path: Path, epoch: int, metrics: Dict[str, float]):
        """
        Save model checkpoint.
        
        Args:
            path: Checkpoint path
            epoch: Current epoch
            metrics: Training metrics
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: Path):
        """
        Load model checkpoint.
        
        Args:
            path: Checkpoint path
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded: {path}")
        return checkpoint['epoch'], checkpoint['metrics']