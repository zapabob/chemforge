"""
Pre-trained model trainer for ChemForge.

This module provides functionality for training pre-trained models
using ChEMBL data and other molecular datasets.
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from chemforge.data.chembl_loader import ChEMBLLoader
from chemforge.data.molecular_features import MolecularFeatures
from chemforge.data.data_preprocessor import DataPreprocessor
from chemforge.models.transformer_model import TransformerModel
from chemforge.models.gnn_model import GNNModel
from chemforge.models.ensemble_model import EnsembleModel
from chemforge.training.trainer import Trainer
from chemforge.training.loss_functions import MultiTargetLoss
from chemforge.training.metrics import MultiTargetMetrics
from chemforge.training.optimizer import OptimizerManager
from chemforge.training.scheduler import SchedulerManager
from chemforge.training.checkpoint import CheckpointManager
from chemforge.utils.logging_utils import TrainingLogger
from chemforge.utils.validation import DataValidator, ModelValidator


class PreTrainer:
    """
    Pre-trained model trainer for ChemForge.
    
    This class handles the training of pre-trained models using ChEMBL data
    and other molecular datasets for CNS drug discovery.
    """
    
    def __init__(
        self,
        output_dir: str = "./pretrained_models",
        log_dir: str = "./logs",
        device: str = "auto"
    ):
        """
        Initialize the pre-trainer.
        
        Args:
            output_dir: Directory to save pre-trained models
            log_dir: Directory for training logs
            device: Device to use for training ('auto', 'cpu', 'cuda')
        """
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.device = self._get_device(device)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.logger = TrainingLogger('pretrainer', log_dir=str(self.log_dir))
        self.data_validator = DataValidator()
        self.model_validator = ModelValidator()
        
        # Training state
        self.current_model = None
        self.training_history = {}
        self.best_models = {}
        
        self.logger.info(f"PreTrainer initialized with device: {self.device}")
    
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
    
    def prepare_chembl_data(
        self,
        targets: List[str],
        min_activities: int = 100,
        activity_types: List[str] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Prepare ChEMBL data for pre-training.
        
        Args:
            targets: List of target ChEMBL IDs
            min_activities: Minimum number of activities per target
            activity_types: List of activity types to include
            test_size: Fraction of data for testing
            val_size: Fraction of data for validation
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (data_info, split_data)
        """
        self.logger.info("Preparing ChEMBL data for pre-training")
        
        # Initialize data loader
        chembl_loader = ChEMBLLoader()
        
        # Load molecular data
        molecules = chembl_loader.load_molecules()
        self.logger.info(f"Loaded {len(molecules)} molecules from ChEMBL")
        
        # Load target data
        targets_data = chembl_loader.load_targets(targets)
        self.logger.info(f"Loaded {len(targets_data)} targets")
        
        # Load activity data
        activities = chembl_loader.load_activities(
            targets=targets,
            min_activities=min_activities,
            activity_types=activity_types
        )
        self.logger.info(f"Loaded {len(activities)} activities")
        
        # Prepare molecular features
        mol_features = MolecularFeatures()
        features_data = mol_features.extract_features(molecules)
        
        # Create dataset
        dataset = self._create_dataset(molecules, activities, features_data)
        
        # Split data
        train_data, test_data = train_test_split(
            dataset, test_size=test_size, random_state=random_state
        )
        train_data, val_data = train_test_split(
            train_data, test_size=val_size, random_state=random_state
        )
        
        # Prepare data info
        data_info = {
            'total_molecules': len(molecules),
            'total_activities': len(activities),
            'num_targets': len(targets),
            'targets': targets,
            'feature_dim': features_data.shape[1] if len(features_data) > 0 else 0,
            'train_size': len(train_data),
            'val_size': len(val_data),
            'test_size': len(test_data)
        }
        
        split_data = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
        self.logger.info(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        return data_info, split_data
    
    def _create_dataset(
        self,
        molecules: pd.DataFrame,
        activities: pd.DataFrame,
        features: np.ndarray
    ) -> pd.DataFrame:
        """Create a unified dataset from molecules, activities, and features."""
        # Merge molecules and activities
        dataset = activities.merge(molecules, on='molecule_id', how='inner')
        
        # Add features
        if len(features) > 0:
            feature_cols = [f'feature_{i}' for i in range(features.shape[1])]
            feature_df = pd.DataFrame(features, columns=feature_cols)
            feature_df['molecule_id'] = molecules['molecule_id'].values
            dataset = dataset.merge(feature_df, on='molecule_id', how='inner')
        
        return dataset
    
    def train_transformer_model(
        self,
        data_info: Dict[str, Any],
        split_data: Dict[str, Any],
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        save_model: bool = True
    ) -> Dict[str, Any]:
        """
        Train a Transformer model for pre-training.
        
        Args:
            data_info: Information about the dataset
            split_data: Split dataset (train, val, test)
            model_config: Model configuration
            training_config: Training configuration
            save_model: Whether to save the trained model
            
        Returns:
            Training results dictionary
        """
        self.logger.info("Training Transformer model for pre-training")
        
        # Validate configurations
        model_validation = self.model_validator.validate_model_config(model_config)
        if not model_validation['valid']:
            raise ValueError(f"Invalid model configuration: {model_validation['errors']}")
        
        training_validation = self.model_validator.validate_training_config(training_config)
        if not training_validation['valid']:
            raise ValueError(f"Invalid training configuration: {training_validation['errors']}")
        
        # Prepare data loaders
        train_loader, val_loader, test_loader = self._prepare_data_loaders(
            split_data, training_config['batch_size']
        )
        
        # Initialize model
        model = TransformerModel(
            input_dim=data_info['feature_dim'],
            output_dim=len(data_info['targets']),
            **model_config
        ).to(self.device)
        
        # Initialize training components
        optimizer = OptimizerManager.get_optimizer(
            model.parameters(),
            training_config['optimizer'],
            training_config['learning_rate'],
            training_config.get('weight_decay', 1e-4)
        )
        
        scheduler = SchedulerManager.get_scheduler(
            optimizer,
            training_config['scheduler'],
            training_config.get('scheduler_params', {})
        )
        
        loss_fn = MultiTargetLoss(
            targets=data_info['targets'],
            loss_type=training_config.get('loss_type', 'mse')
        )
        
        metrics = MultiTargetMetrics(targets=data_info['targets'])
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            metrics=metrics,
            device=self.device,
            log_dir=str(self.log_dir)
        )
        
        # Train model
        training_results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=training_config['epochs'],
            save_best=True,
            save_last=True,
            checkpoint_interval=training_config.get('checkpoint_interval', 10)
        )
        
        # Evaluate on test set
        test_results = trainer.evaluate(test_loader)
        training_results['test_results'] = test_results
        
        # Save model if requested
        if save_model:
            model_path = self.output_dir / f"transformer_pretrained_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': model_config,
                'training_config': training_config,
                'data_info': data_info,
                'training_results': training_results
            }, model_path)
            self.logger.info(f"Saved pre-trained Transformer model to {model_path}")
        
        # Update training history
        self.training_history['transformer'] = training_results
        self.best_models['transformer'] = model
        
        return training_results
    
    def train_gnn_model(
        self,
        data_info: Dict[str, Any],
        split_data: Dict[str, Any],
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        save_model: bool = True
    ) -> Dict[str, Any]:
        """
        Train a GNN model for pre-training.
        
        Args:
            data_info: Information about the dataset
            split_data: Split dataset (train, val, test)
            model_config: Model configuration
            training_config: Training configuration
            save_model: Whether to save the trained model
            
        Returns:
            Training results dictionary
        """
        self.logger.info("Training GNN model for pre-training")
        
        # Validate configurations
        model_validation = self.model_validator.validate_model_config(model_config)
        if not model_validation['valid']:
            raise ValueError(f"Invalid model configuration: {model_validation['errors']}")
        
        training_validation = self.model_validator.validate_training_config(training_config)
        if not training_validation['valid']:
            raise ValueError(f"Invalid training configuration: {training_validation['errors']}")
        
        # Prepare data loaders
        train_loader, val_loader, test_loader = self._prepare_data_loaders(
            split_data, training_config['batch_size']
        )
        
        # Initialize model
        model = GNNModel(
            input_dim=data_info['feature_dim'],
            output_dim=len(data_info['targets']),
            **model_config
        ).to(self.device)
        
        # Initialize training components
        optimizer = OptimizerManager.get_optimizer(
            model.parameters(),
            training_config['optimizer'],
            training_config['learning_rate'],
            training_config.get('weight_decay', 1e-4)
        )
        
        scheduler = SchedulerManager.get_scheduler(
            optimizer,
            training_config['scheduler'],
            training_config.get('scheduler_params', {})
        )
        
        loss_fn = MultiTargetLoss(
            targets=data_info['targets'],
            loss_type=training_config.get('loss_type', 'mse')
        )
        
        metrics = MultiTargetMetrics(targets=data_info['targets'])
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            metrics=metrics,
            device=self.device,
            log_dir=str(self.log_dir)
        )
        
        # Train model
        training_results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=training_config['epochs'],
            save_best=True,
            save_last=True,
            checkpoint_interval=training_config.get('checkpoint_interval', 10)
        )
        
        # Evaluate on test set
        test_results = trainer.evaluate(test_loader)
        training_results['test_results'] = test_results
        
        # Save model if requested
        if save_model:
            model_path = self.output_dir / f"gnn_pretrained_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': model_config,
                'training_config': training_config,
                'data_info': data_info,
                'training_results': training_results
            }, model_path)
            self.logger.info(f"Saved pre-trained GNN model to {model_path}")
        
        # Update training history
        self.training_history['gnn'] = training_results
        self.best_models['gnn'] = model
        
        return training_results
    
    def train_ensemble_model(
        self,
        data_info: Dict[str, Any],
        split_data: Dict[str, Any],
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        save_model: bool = True
    ) -> Dict[str, Any]:
        """
        Train an ensemble model for pre-training.
        
        Args:
            data_info: Information about the dataset
            split_data: Split dataset (train, val, test)
            model_config: Model configuration
            training_config: Training configuration
            save_model: Whether to save the trained model
            
        Returns:
            Training results dictionary
        """
        self.logger.info("Training ensemble model for pre-training")
        
        # Validate configurations
        model_validation = self.model_validator.validate_model_config(model_config)
        if not model_validation['valid']:
            raise ValueError(f"Invalid model configuration: {model_validation['errors']}")
        
        training_validation = self.model_validator.validate_training_config(training_config)
        if not training_validation['valid']:
            raise ValueError(f"Invalid training configuration: {training_validation['errors']}")
        
        # Prepare data loaders
        train_loader, val_loader, test_loader = self._prepare_data_loaders(
            split_data, training_config['batch_size']
        )
        
        # Initialize model
        model = EnsembleModel(
            input_dim=data_info['feature_dim'],
            output_dim=len(data_info['targets']),
            **model_config
        ).to(self.device)
        
        # Initialize training components
        optimizer = OptimizerManager.get_optimizer(
            model.parameters(),
            training_config['optimizer'],
            training_config['learning_rate'],
            training_config.get('weight_decay', 1e-4)
        )
        
        scheduler = SchedulerManager.get_scheduler(
            optimizer,
            training_config['scheduler'],
            training_config.get('scheduler_params', {})
        )
        
        loss_fn = MultiTargetLoss(
            targets=data_info['targets'],
            loss_type=training_config.get('loss_type', 'mse')
        )
        
        metrics = MultiTargetMetrics(targets=data_info['targets'])
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            metrics=metrics,
            device=self.device,
            log_dir=str(self.log_dir)
        )
        
        # Train model
        training_results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=training_config['epochs'],
            save_best=True,
            save_last=True,
            checkpoint_interval=training_config.get('checkpoint_interval', 10)
        )
        
        # Evaluate on test set
        test_results = trainer.evaluate(test_loader)
        training_results['test_results'] = test_results
        
        # Save model if requested
        if save_model:
            model_path = self.output_dir / f"ensemble_pretrained_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': model_config,
                'training_config': training_config,
                'data_info': data_info,
                'training_results': training_results
            }, model_path)
            self.logger.info(f"Saved pre-trained ensemble model to {model_path}")
        
        # Update training history
        self.training_history['ensemble'] = training_results
        self.best_models['ensemble'] = model
        
        return training_results
    
    def _prepare_data_loaders(
        self,
        split_data: Dict[str, Any],
        batch_size: int
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data loaders for training."""
        # This is a simplified implementation
        # In practice, you would need to implement proper data loading
        # with molecular graphs, features, and targets
        
        train_loader = DataLoader(
            split_data['train'], batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            split_data['val'], batch_size=batch_size, shuffle=False
        )
        test_loader = DataLoader(
            split_data['test'], batch_size=batch_size, shuffle=False
        )
        
        return train_loader, val_loader, test_loader
    
    def train_all_models(
        self,
        targets: List[str],
        model_configs: Dict[str, Dict[str, Any]],
        training_config: Dict[str, Any],
        save_models: bool = True
    ) -> Dict[str, Any]:
        """
        Train all model types for pre-training.
        
        Args:
            targets: List of target ChEMBL IDs
            model_configs: Model configurations for each model type
            training_config: Training configuration
            save_models: Whether to save the trained models
            
        Returns:
            Dictionary of training results for all models
        """
        self.logger.info("Training all model types for pre-training")
        
        # Prepare data
        data_info, split_data = self.prepare_chembl_data(targets)
        
        results = {}
        
        # Train Transformer model
        if 'transformer' in model_configs:
            try:
                transformer_results = self.train_transformer_model(
                    data_info, split_data, model_configs['transformer'], training_config, save_models
                )
                results['transformer'] = transformer_results
            except Exception as e:
                self.logger.error(f"Failed to train Transformer model: {e}")
                results['transformer'] = {'error': str(e)}
        
        # Train GNN model
        if 'gnn' in model_configs:
            try:
                gnn_results = self.train_gnn_model(
                    data_info, split_data, model_configs['gnn'], training_config, save_models
                )
                results['gnn'] = gnn_results
            except Exception as e:
                self.logger.error(f"Failed to train GNN model: {e}")
                results['gnn'] = {'error': str(e)}
        
        # Train ensemble model
        if 'ensemble' in model_configs:
            try:
                ensemble_results = self.train_ensemble_model(
                    data_info, split_data, model_configs['ensemble'], training_config, save_models
                )
                results['ensemble'] = ensemble_results
            except Exception as e:
                self.logger.error(f"Failed to train ensemble model: {e}")
                results['ensemble'] = {'error': str(e)}
        
        # Save training summary
        summary_path = self.output_dir / f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'data_info': data_info,
                'model_configs': model_configs,
                'training_config': training_config,
                'results': results
            }, f, indent=2)
        
        self.logger.info(f"Saved training summary to {summary_path}")
        
        return results
    
    def get_training_history(self) -> Dict[str, Any]:
        """Get the training history."""
        return self.training_history
    
    def get_best_models(self) -> Dict[str, Any]:
        """Get the best trained models."""
        return self.best_models
    
    def save_training_state(self, filepath: str):
        """Save the current training state."""
        state = {
            'training_history': self.training_history,
            'best_models': self.best_models,
            'output_dir': str(self.output_dir),
            'log_dir': str(self.log_dir),
            'device': str(self.device)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        self.logger.info(f"Saved training state to {filepath}")
    
    def load_training_state(self, filepath: str):
        """Load a training state."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.training_history = state['training_history']
        self.best_models = state['best_models']
        
        self.logger.info(f"Loaded training state from {filepath}")
