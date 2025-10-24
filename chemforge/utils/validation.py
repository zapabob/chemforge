"""
Validation utilities for ChemForge platform.

This module provides validation functionality for data, models, and predictions
to ensure data quality and model reliability.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from pathlib import Path
import json
from datetime import datetime
import warnings


class DataValidator:
    """Data validation utilities."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize data validator.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def validate_molecular_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate molecular data.
        
        Args:
            data: DataFrame containing molecular data
            
        Returns:
            Validation results dictionary
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check required columns
        required_columns = ['smiles']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            results['valid'] = False
            results['errors'].append(f"Missing required columns: {missing_columns}")
        
        # Check for empty data
        if data.empty:
            results['valid'] = False
            results['errors'].append("Data is empty")
            return results
        
        # Check for duplicate SMILES
        if 'smiles' in data.columns:
            duplicate_smiles = data['smiles'].duplicated().sum()
            if duplicate_smiles > 0:
                results['warnings'].append(f"Found {duplicate_smiles} duplicate SMILES")
        
        # Check for invalid SMILES
        if 'smiles' in data.columns:
            invalid_smiles = data['smiles'].isna().sum()
            if invalid_smiles > 0:
                results['warnings'].append(f"Found {invalid_smiles} invalid SMILES (NaN)")
        
        # Validate molecular properties
        property_columns = ['mol_weight', 'logp', 'hbd', 'hba', 'tpsa']
        for col in property_columns:
            if col in data.columns:
                # Check for negative values where not allowed
                if col in ['mol_weight', 'logp', 'hbd', 'hba', 'tpsa']:
                    negative_count = (data[col] < 0).sum()
                    if negative_count > 0:
                        results['warnings'].append(f"Found {negative_count} negative values in {col}")
                
                # Check for extreme values
                if col == 'mol_weight':
                    extreme_count = (data[col] > 1000).sum()
                    if extreme_count > 0:
                        results['warnings'].append(f"Found {extreme_count} molecules with MW > 1000")
                
                if col == 'logp':
                    extreme_count = ((data[col] < -5) | (data[col] > 10)).sum()
                    if extreme_count > 0:
                        results['warnings'].append(f"Found {extreme_count} molecules with extreme logP values")
        
        # Calculate statistics
        results['statistics'] = {
            'total_molecules': len(data),
            'valid_smiles': data['smiles'].notna().sum() if 'smiles' in data.columns else 0,
            'duplicate_smiles': duplicate_smiles if 'smiles' in data.columns else 0,
            'columns': list(data.columns),
            'data_types': data.dtypes.to_dict()
        }
        
        self.logger.info(f"Data validation completed: {len(results['errors'])} errors, {len(results['warnings'])} warnings")
        return results
    
    def validate_activity_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate activity data.
        
        Args:
            data: DataFrame containing activity data
            
        Returns:
            Validation results dictionary
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check required columns
        required_columns = ['molecule_id', 'target_id', 'activity_value']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            results['valid'] = False
            results['errors'].append(f"Missing required columns: {missing_columns}")
        
        # Check for empty data
        if data.empty:
            results['valid'] = False
            results['errors'].append("Data is empty")
            return results
        
        # Check for invalid activity values
        if 'activity_value' in data.columns:
            invalid_activities = data['activity_value'].isna().sum()
            if invalid_activities > 0:
                results['warnings'].append(f"Found {invalid_activities} invalid activity values (NaN)")
            
            # Check for extreme values
            extreme_count = ((data['activity_value'] < -20) | (data['activity_value'] > 20)).sum()
            if extreme_count > 0:
                results['warnings'].append(f"Found {extreme_count} extreme activity values")
        
        # Check for duplicate activities
        if all(col in data.columns for col in ['molecule_id', 'target_id']):
            duplicate_activities = data.duplicated(subset=['molecule_id', 'target_id']).sum()
            if duplicate_activities > 0:
                results['warnings'].append(f"Found {duplicate_activities} duplicate activities")
        
        # Calculate statistics
        results['statistics'] = {
            'total_activities': len(data),
            'unique_molecules': data['molecule_id'].nunique() if 'molecule_id' in data.columns else 0,
            'unique_targets': data['target_id'].nunique() if 'target_id' in data.columns else 0,
            'valid_activities': data['activity_value'].notna().sum() if 'activity_value' in data.columns else 0,
            'activity_range': {
                'min': data['activity_value'].min() if 'activity_value' in data.columns else None,
                'max': data['activity_value'].max() if 'activity_value' in data.columns else None,
                'mean': data['activity_value'].mean() if 'activity_value' in data.columns else None
            }
        }
        
        self.logger.info(f"Activity data validation completed: {len(results['errors'])} errors, {len(results['warnings'])} warnings")
        return results
    
    def validate_features(self, features: np.ndarray, feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Validate feature matrix.
        
        Args:
            features: Feature matrix
            feature_names: List of feature names
            
        Returns:
            Validation results dictionary
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check for NaN values
        nan_count = np.isnan(features).sum()
        if nan_count > 0:
            results['warnings'].append(f"Found {nan_count} NaN values in features")
        
        # Check for infinite values
        inf_count = np.isinf(features).sum()
        if inf_count > 0:
            results['warnings'].append(f"Found {inf_count} infinite values in features")
        
        # Check for constant features
        constant_features = []
        for i in range(features.shape[1]):
            if np.std(features[:, i]) == 0:
                constant_features.append(i)
        
        if constant_features:
            results['warnings'].append(f"Found {len(constant_features)} constant features")
        
        # Check for highly correlated features
        if features.shape[1] > 1:
            correlation_matrix = np.corrcoef(features.T)
            high_corr_pairs = []
            for i in range(correlation_matrix.shape[0]):
                for j in range(i + 1, correlation_matrix.shape[1]):
                    if abs(correlation_matrix[i, j]) > 0.95:
                        high_corr_pairs.append((i, j))
            
            if high_corr_pairs:
                results['warnings'].append(f"Found {len(high_corr_pairs)} highly correlated feature pairs")
        
        # Calculate statistics
        results['statistics'] = {
            'shape': features.shape,
            'nan_count': nan_count,
            'inf_count': inf_count,
            'constant_features': len(constant_features),
            'high_corr_pairs': len(high_corr_pairs) if 'high_corr_pairs' in locals() else 0,
            'feature_names': feature_names,
            'mean': np.mean(features, axis=0).tolist(),
            'std': np.std(features, axis=0).tolist()
        }
        
        self.logger.info(f"Feature validation completed: {len(results['errors'])} errors, {len(results['warnings'])} warnings")
        return results


class ModelValidator:
    """Model validation utilities."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize model validator.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def validate_model_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate model configuration.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            Validation results dictionary
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check required parameters
        required_params = ['model_type', 'input_dim', 'output_dim']
        missing_params = [param for param in required_params if param not in config]
        
        if missing_params:
            results['valid'] = False
            results['errors'].append(f"Missing required parameters: {missing_params}")
        
        # Validate model type
        if 'model_type' in config:
            valid_types = ['transformer', 'gnn', 'ensemble']
            if config['model_type'] not in valid_types:
                results['errors'].append(f"Invalid model type: {config['model_type']}")
        
        # Validate dimensions
        if 'input_dim' in config:
            if not isinstance(config['input_dim'], int) or config['input_dim'] <= 0:
                results['errors'].append("input_dim must be a positive integer")
        
        if 'output_dim' in config:
            if not isinstance(config['output_dim'], int) or config['output_dim'] <= 0:
                results['errors'].append("output_dim must be a positive integer")
        
        # Validate PWA+PET parameters
        if config.get('use_pwa_pet', False):
            if 'pwa_buckets' not in config:
                results['warnings'].append("PWA+PET enabled but pwa_buckets not specified")
            
            if 'pet_curv_reg' in config:
                if not isinstance(config['pet_curv_reg'], (int, float)) or config['pet_curv_reg'] < 0:
                    results['errors'].append("pet_curv_reg must be a non-negative number")
        
        # Validate GNN parameters
        if config.get('model_type') == 'gnn':
            if 'gnn_type' not in config:
                results['warnings'].append("GNN model type not specified")
            
            if 'gnn_layers' in config:
                if not isinstance(config['gnn_layers'], int) or config['gnn_layers'] <= 0:
                    results['errors'].append("gnn_layers must be a positive integer")
        
        # Validate ensemble parameters
        if config.get('model_type') == 'ensemble':
            if 'ensemble_models' not in config:
                results['errors'].append("Ensemble models not specified")
            elif not isinstance(config['ensemble_models'], list) or len(config['ensemble_models']) < 2:
                results['errors'].append("Ensemble must have at least 2 models")
            
            if 'ensemble_weights' in config:
                if not isinstance(config['ensemble_weights'], list):
                    results['errors'].append("ensemble_weights must be a list")
                elif len(config['ensemble_weights']) != len(config.get('ensemble_models', [])):
                    results['errors'].append("ensemble_weights length must match ensemble_models length")
                elif abs(sum(config['ensemble_weights']) - 1.0) > 1e-6:
                    results['warnings'].append("ensemble_weights do not sum to 1.0")
        
        # Calculate statistics
        results['statistics'] = {
            'model_type': config.get('model_type'),
            'input_dim': config.get('input_dim'),
            'output_dim': config.get('output_dim'),
            'use_pwa_pet': config.get('use_pwa_pet', False),
            'use_gnn': config.get('model_type') == 'gnn',
            'use_ensemble': config.get('model_type') == 'ensemble'
        }
        
        self.logger.info(f"Model config validation completed: {len(results['errors'])} errors, {len(results['warnings'])} warnings")
        return results
    
    def validate_training_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate training configuration.
        
        Args:
            config: Training configuration dictionary
            
        Returns:
            Validation results dictionary
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check required parameters
        required_params = ['epochs', 'batch_size', 'learning_rate']
        missing_params = [param for param in required_params if param not in config]
        
        if missing_params:
            results['valid'] = False
            results['errors'].append(f"Missing required parameters: {missing_params}")
        
        # Validate epochs
        if 'epochs' in config:
            if not isinstance(config['epochs'], int) or config['epochs'] <= 0:
                results['errors'].append("epochs must be a positive integer")
        
        # Validate batch size
        if 'batch_size' in config:
            if not isinstance(config['batch_size'], int) or config['batch_size'] <= 0:
                results['errors'].append("batch_size must be a positive integer")
        
        # Validate learning rate
        if 'learning_rate' in config:
            if not isinstance(config['learning_rate'], (int, float)) or config['learning_rate'] <= 0:
                results['errors'].append("learning_rate must be a positive number")
            elif config['learning_rate'] > 1.0:
                results['warnings'].append("learning_rate is very high (> 1.0)")
        
        # Validate data splits
        split_params = ['train_split', 'val_split', 'test_split']
        for param in split_params:
            if param in config:
                if not isinstance(config[param], (int, float)) or not (0 <= config[param] <= 1):
                    results['errors'].append(f"{param} must be between 0 and 1")
        
        # Check split sum
        if all(param in config for param in split_params):
            split_sum = config['train_split'] + config['val_split'] + config['test_split']
            if abs(split_sum - 1.0) > 1e-6:
                results['warnings'].append(f"Data splits sum to {split_sum:.6f}, not 1.0")
        
        # Validate optimizer
        if 'optimizer' in config:
            valid_optimizers = ['adam', 'sgd', 'adamw', 'rmsprop']
            if config['optimizer'] not in valid_optimizers:
                results['warnings'].append(f"Unknown optimizer: {config['optimizer']}")
        
        # Validate scheduler
        if 'scheduler' in config:
            valid_schedulers = ['cosine', 'step', 'exponential', 'plateau']
            if config['scheduler'] not in valid_schedulers:
                results['warnings'].append(f"Unknown scheduler: {config['scheduler']}")
        
        # Calculate statistics
        results['statistics'] = {
            'epochs': config.get('epochs'),
            'batch_size': config.get('batch_size'),
            'learning_rate': config.get('learning_rate'),
            'optimizer': config.get('optimizer'),
            'scheduler': config.get('scheduler'),
            'use_amp': config.get('use_amp', False),
            'early_stopping': config.get('early_stopping', False)
        }
        
        self.logger.info(f"Training config validation completed: {len(results['errors'])} errors, {len(results['warnings'])} warnings")
        return results
    
    def validate_model_weights(self, model_weights: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate model weights.
        
        Args:
            model_weights: Model weights dictionary
            
        Returns:
            Validation results dictionary
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        if not model_weights:
            results['valid'] = False
            results['errors'].append("Model weights are empty")
            return results
        
        # Check for NaN values in weights
        nan_count = 0
        inf_count = 0
        total_params = 0
        
        for name, weight in model_weights.items():
            if hasattr(weight, 'isnan'):
                nan_count += weight.isnan().sum().item()
                inf_count += weight.isinf().sum().item()
                total_params += weight.numel()
            elif isinstance(weight, np.ndarray):
                nan_count += np.isnan(weight).sum()
                inf_count += np.isinf(weight).sum()
                total_params += weight.size
        
        if nan_count > 0:
            results['errors'].append(f"Found {nan_count} NaN values in model weights")
        
        if inf_count > 0:
            results['errors'].append(f"Found {inf_count} infinite values in model weights")
        
        # Check for zero weights
        zero_count = 0
        for name, weight in model_weights.items():
            if hasattr(weight, 'eq'):
                zero_count += (weight == 0).sum().item()
            elif isinstance(weight, np.ndarray):
                zero_count += (weight == 0).sum()
        
        if zero_count > total_params * 0.5:
            results['warnings'].append(f"High number of zero weights: {zero_count}/{total_params}")
        
        # Calculate statistics
        results['statistics'] = {
            'total_parameters': total_params,
            'nan_count': nan_count,
            'inf_count': inf_count,
            'zero_count': zero_count,
            'weight_layers': len(model_weights)
        }
        
        self.logger.info(f"Model weights validation completed: {len(results['errors'])} errors, {len(results['warnings'])} warnings")
        return results


class PredictionValidator:
    """Prediction validation utilities."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize prediction validator.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def validate_predictions(self, predictions: np.ndarray, 
                           confidence: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Validate predictions.
        
        Args:
            predictions: Prediction array
            confidence: Confidence array (optional)
            
        Returns:
            Validation results dictionary
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check for NaN values
        nan_count = np.isnan(predictions).sum()
        if nan_count > 0:
            results['errors'].append(f"Found {nan_count} NaN values in predictions")
        
        # Check for infinite values
        inf_count = np.isinf(predictions).sum()
        if inf_count > 0:
            results['errors'].append(f"Found {inf_count} infinite values in predictions")
        
        # Check for extreme values
        extreme_count = ((predictions < -20) | (predictions > 20)).sum()
        if extreme_count > 0:
            results['warnings'].append(f"Found {extreme_count} extreme prediction values")
        
        # Validate confidence if provided
        if confidence is not None:
            if confidence.shape != predictions.shape:
                results['errors'].append("Confidence shape does not match predictions shape")
            else:
                # Check confidence range
                if np.any(confidence < 0) or np.any(confidence > 1):
                    results['warnings'].append("Confidence values outside [0, 1] range")
                
                # Check for low confidence predictions
                low_conf_count = (confidence < 0.5).sum()
                if low_conf_count > 0:
                    results['warnings'].append(f"Found {low_conf_count} low confidence predictions (< 0.5)")
        
        # Calculate statistics
        results['statistics'] = {
            'shape': predictions.shape,
            'nan_count': nan_count,
            'inf_count': inf_count,
            'extreme_count': extreme_count,
            'mean': np.mean(predictions),
            'std': np.std(predictions),
            'min': np.min(predictions),
            'max': np.max(predictions),
            'has_confidence': confidence is not None
        }
        
        if confidence is not None:
            results['statistics']['confidence_mean'] = np.mean(confidence)
            results['statistics']['confidence_std'] = np.std(confidence)
            results['statistics']['low_conf_count'] = low_conf_count
        
        self.logger.info(f"Prediction validation completed: {len(results['errors'])} errors, {len(results['warnings'])} warnings")
        return results
    
    def validate_admet_predictions(self, predictions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Validate ADMET predictions.
        
        Args:
            predictions: Dictionary containing ADMET predictions
            
        Returns:
            Validation results dictionary
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check required ADMET properties
        required_properties = ['absorption', 'distribution', 'metabolism', 'excretion', 'toxicity']
        missing_properties = [prop for prop in required_properties if prop not in predictions]
        
        if missing_properties:
            results['warnings'].append(f"Missing ADMET properties: {missing_properties}")
        
        # Validate each property
        for prop, values in predictions.items():
            if not isinstance(values, np.ndarray):
                results['errors'].append(f"ADMET property {prop} is not a numpy array")
                continue
            
            # Check for NaN values
            nan_count = np.isnan(values).sum()
            if nan_count > 0:
                results['warnings'].append(f"Found {nan_count} NaN values in {prop}")
            
            # Check for values outside [0, 1] range
            out_of_range = ((values < 0) | (values > 1)).sum()
            if out_of_range > 0:
                results['warnings'].append(f"Found {out_of_range} values outside [0, 1] range in {prop}")
        
        # Calculate statistics
        results['statistics'] = {
            'properties': list(predictions.keys()),
            'missing_properties': missing_properties,
            'property_stats': {}
        }
        
        for prop, values in predictions.items():
            if isinstance(values, np.ndarray):
                results['statistics']['property_stats'][prop] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'nan_count': np.isnan(values).sum()
                }
        
        self.logger.info(f"ADMET prediction validation completed: {len(results['errors'])} errors, {len(results['warnings'])} warnings")
        return results
    
    def validate_cns_mpo_scores(self, scores: np.ndarray) -> Dict[str, Any]:
        """
        Validate CNS-MPO scores.
        
        Args:
            scores: CNS-MPO scores array
            
        Returns:
            Validation results dictionary
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check for NaN values
        nan_count = np.isnan(scores).sum()
        if nan_count > 0:
            results['warnings'].append(f"Found {nan_count} NaN values in CNS-MPO scores")
        
        # Check for values outside [0, 6] range
        out_of_range = ((scores < 0) | (scores > 6)).sum()
        if out_of_range > 0:
            results['warnings'].append(f"Found {out_of_range} CNS-MPO scores outside [0, 6] range")
        
        # Check for low scores
        low_scores = (scores < 2).sum()
        if low_scores > 0:
            results['warnings'].append(f"Found {low_scores} low CNS-MPO scores (< 2)")
        
        # Calculate statistics
        results['statistics'] = {
            'shape': scores.shape,
            'nan_count': nan_count,
            'out_of_range': out_of_range,
            'low_scores': low_scores,
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores)
        }
        
        self.logger.info(f"CNS-MPO score validation completed: {len(results['errors'])} errors, {len(results['warnings'])} warnings")
        return results
