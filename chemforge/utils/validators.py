"""
Custom validators for ChemForge configuration classes.

This module provides custom validation functions for Pydantic models
to ensure configuration integrity and business logic compliance.
"""

from typing import Any, Dict, List, Optional
from pydantic import Field, model_validator
import logging

logger = logging.getLogger(__name__)


def validate_ensemble_weights(weights: List[float]) -> List[float]:
    """
    Validate that ensemble weights sum to 1.0.
    
    Args:
        weights: List of ensemble weights
        
    Returns:
        Validated weights
        
    Raises:
        ValueError: If weights don't sum to 1.0
    """
    if not weights:
        return weights
    
    total_weight = sum(weights)
    if not abs(total_weight - 1.0) < 1e-6:
        raise ValueError(f"Ensemble weights must sum to 1.0, got {total_weight}")
    
    return weights


def validate_positive_weights(weights: List[float]) -> List[float]:
    """
    Validate that all ensemble weights are positive.
    
    Args:
        weights: List of ensemble weights
        
    Returns:
        Validated weights
        
    Raises:
        ValueError: If any weight is negative
    """
    if not weights:
        return weights
    
    for i, weight in enumerate(weights):
        if weight < 0:
            raise ValueError(f"Ensemble weight {i} must be non-negative, got {weight}")
    
    return weights


def validate_model_type(model_type: str) -> str:
    """
    Validate model type is supported.
    
    Args:
        model_type: Model type string
        
    Returns:
        Validated model type
        
    Raises:
        ValueError: If model type is not supported
    """
    supported_types = ['transformer', 'gnn', 'ensemble', 'linear', 'mlp']
    
    if model_type not in supported_types:
        raise ValueError(f"Model type must be one of {supported_types}, got {model_type}")
    
    return model_type


def validate_optimizer(optimizer: str) -> str:
    """
    Validate optimizer is supported.
    
    Args:
        optimizer: Optimizer string
        
    Returns:
        Validated optimizer
        
    Raises:
        ValueError: If optimizer is not supported
    """
    supported_optimizers = ['adam', 'adamw', 'sgd', 'rmsprop', 'adagrad']
    
    if optimizer not in supported_optimizers:
        raise ValueError(f"Optimizer must be one of {supported_optimizers}, got {optimizer}")
    
    return optimizer


def validate_scheduler(scheduler: str) -> str:
    """
    Validate scheduler is supported.
    
    Args:
        scheduler: Scheduler string
        
    Returns:
        Validated scheduler
        
    Raises:
        ValueError: If scheduler is not supported
    """
    supported_schedulers = ['cosine', 'linear', 'step', 'exponential', 'plateau']
    
    if scheduler not in supported_schedulers:
        raise ValueError(f"Scheduler must be one of {supported_schedulers}, got {scheduler}")
    
    return scheduler


def validate_log_level(log_level: str) -> str:
    """
    Validate log level is supported.
    
    Args:
        log_level: Log level string
        
    Returns:
        Validated log level
        
    Raises:
        ValueError: If log level is not supported
    """
    supported_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    
    if log_level.upper() not in supported_levels:
        raise ValueError(f"Log level must be one of {supported_levels}, got {log_level}")
    
    return log_level.upper()


def validate_gnn_type(gnn_type: str) -> str:
    """
    Validate GNN type is supported.
    
    Args:
        gnn_type: GNN type string
        
    Returns:
        Validated GNN type
        
    Raises:
        ValueError: If GNN type is not supported
    """
    supported_types = ['gat', 'gcn', 'sage', 'gin', 'graphsage']
    
    if gnn_type not in supported_types:
        raise ValueError(f"GNN type must be one of {supported_types}, got {gnn_type}")
    
    return gnn_type


def validate_scaffold_types(scaffold_types: List[str]) -> List[str]:
    """
    Validate scaffold types are supported.
    
    Args:
        scaffold_types: List of scaffold type strings
        
    Returns:
        Validated scaffold types
        
    Raises:
        ValueError: If any scaffold type is not supported
    """
    supported_types = ['trivial', 'fund', 'adj', 'murcko', 'bemis']
    
    for scaffold_type in scaffold_types:
        if scaffold_type not in supported_types:
            raise ValueError(f"Scaffold type must be one of {supported_types}, got {scaffold_type}")
    
    return scaffold_types


def validate_admet_properties(properties: List[str]) -> List[str]:
    """
    Validate ADMET properties are supported.
    
    Args:
        properties: List of ADMET property strings
        
    Returns:
        Validated ADMET properties
        
    Raises:
        ValueError: If any property is not supported
    """
    supported_properties = [
        'absorption', 'distribution', 'metabolism', 'excretion', 'toxicity',
        'solubility', 'permeability', 'bioavailability', 'clearance',
        'half_life', 'vd', 'fup', 'clint', 'clhep'
    ]
    
    for property_name in properties:
        if property_name not in supported_properties:
            raise ValueError(f"ADMET property must be one of {supported_properties}, got {property_name}")
    
    return properties


def validate_data_splits(train_split: float, val_split: float, test_split: float) -> tuple:
    """
    Validate that data splits are valid and sum to 1.0.
    
    Args:
        train_split: Training split ratio
        val_split: Validation split ratio
        test_split: Test split ratio
        
    Returns:
        Validated splits tuple
        
    Raises:
        ValueError: If splits are invalid
    """
    splits = [train_split, val_split, test_split]
    
    # Check all splits are non-negative
    for i, split in enumerate(['train', 'val', 'test']):
        if splits[i] < 0:
            raise ValueError(f"{split}_split must be non-negative, got {splits[i]}")
    
    # Check splits sum to 1.0
    total_split = sum(splits)
    if not abs(total_split - 1.0) < 1e-6:
        raise ValueError(f"Data splits must sum to 1.0, got {total_split}")
    
    return train_split, val_split, test_split


def validate_pwa_buckets(buckets: Dict[str, int]) -> Dict[str, int]:
    """
    Validate PWA buckets configuration.
    
    Args:
        buckets: PWA buckets dictionary
        
    Returns:
        Validated buckets dictionary
        
    Raises:
        ValueError: If buckets configuration is invalid
    """
    if not buckets:
        return buckets
    
    required_keys = ['trivial', 'fund', 'adj']
    for key in required_keys:
        if key not in buckets:
            raise ValueError(f"PWA buckets must contain '{key}' key")
    
    for key, value in buckets.items():
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"PWA bucket '{key}' must be a non-negative integer, got {value}")
    
    return buckets


def validate_ensemble_models(models: List[str]) -> List[str]:
    """
    Validate ensemble models are supported.
    
    Args:
        models: List of ensemble model strings
        
    Returns:
        Validated ensemble models
        
    Raises:
        ValueError: If any model is not supported
    """
    supported_models = ['transformer', 'gnn', 'linear', 'mlp', 'ensemble']
    
    for model in models:
        if model not in supported_models:
            raise ValueError(f"Ensemble model must be one of {supported_models}, got {model}")
    
    return models


def validate_confidence_threshold(threshold: float) -> float:
    """
    Validate confidence threshold is in valid range.
    
    Args:
        threshold: Confidence threshold value
        
    Returns:
        Validated threshold
        
    Raises:
        ValueError: If threshold is not in valid range
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Confidence threshold must be between 0.0 and 1.0, got {threshold}")
    
    return threshold


def validate_feature_threshold(threshold: float) -> float:
    """
    Validate feature threshold is in valid range.
    
    Args:
        threshold: Feature threshold value
        
    Returns:
        Validated threshold
        
    Raises:
        ValueError: If threshold is not in valid range
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Feature threshold must be between 0.0 and 1.0, got {threshold}")
    
    return threshold


def validate_morgan_radius(radius: int) -> int:
    """
    Validate Morgan radius is in valid range.
    
    Args:
        radius: Morgan radius value
        
    Returns:
        Validated radius
        
    Raises:
        ValueError: If radius is not in valid range
    """
    if not 1 <= radius <= 5:
        raise ValueError(f"Morgan radius must be between 1 and 5, got {radius}")
    
    return radius


def validate_morgan_bits(bits: int) -> int:
    """
    Validate Morgan bits is positive.
    
    Args:
        bits: Morgan bits value
        
    Returns:
        Validated bits
        
    Raises:
        ValueError: If bits is not positive
    """
    if bits <= 0:
        raise ValueError(f"Morgan bits must be positive, got {bits}")
    
    return bits


def validate_dropout_rate(dropout: float) -> float:
    """
    Validate dropout rate is in valid range.
    
    Args:
        dropout: Dropout rate value
        
    Returns:
        Validated dropout rate
        
    Raises:
        ValueError: If dropout rate is not in valid range
    """
    if not 0.0 <= dropout <= 1.0:
        raise ValueError(f"Dropout rate must be between 0.0 and 1.0, got {dropout}")
    
    return dropout


def validate_learning_rate(lr: float) -> float:
    """
    Validate learning rate is in valid range.
    
    Args:
        lr: Learning rate value
        
    Returns:
        Validated learning rate
        
    Raises:
        ValueError: If learning rate is not in valid range
    """
    if not 0.0 < lr < 1.0:
        raise ValueError(f"Learning rate must be between 0.0 and 1.0, got {lr}")
    
    return lr


def validate_weight_decay(wd: float) -> float:
    """
    Validate weight decay is non-negative.
    
    Args:
        wd: Weight decay value
        
    Returns:
        Validated weight decay
        
    Raises:
        ValueError: If weight decay is negative
    """
    if wd < 0:
        raise ValueError(f"Weight decay must be non-negative, got {wd}")
    
    return wd


def validate_gradient_clip(clip: float) -> float:
    """
    Validate gradient clipping is positive.
    
    Args:
        clip: Gradient clipping value
        
    Returns:
        Validated gradient clipping
        
    Raises:
        ValueError: If gradient clipping is not positive
    """
    if clip <= 0:
        raise ValueError(f"Gradient clipping must be positive, got {clip}")
    
    return clip


def validate_patience(patience: int) -> int:
    """
    Validate patience is positive.
    
    Args:
        patience: Patience value
        
    Returns:
        Validated patience
        
    Raises:
        ValueError: If patience is not positive
    """
    if patience <= 0:
        raise ValueError(f"Patience must be positive, got {patience}")
    
    return patience


def validate_epochs(epochs: int) -> int:
    """
    Validate epochs is positive.
    
    Args:
        epochs: Number of epochs
        
    Returns:
        Validated epochs
        
    Raises:
        ValueError: If epochs is not positive
    """
    if epochs <= 0:
        raise ValueError(f"Epochs must be positive, got {epochs}")
    
    return epochs


def validate_batch_size(batch_size: int) -> int:
    """
    Validate batch size is positive.
    
    Args:
        batch_size: Batch size value
        
    Returns:
        Validated batch size
        
    Raises:
        ValueError: If batch size is not positive
    """
    if batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got {batch_size}")
    
    return batch_size


def validate_dimensions(dim: int, name: str) -> int:
    """
    Validate dimension is positive.
    
    Args:
        dim: Dimension value
        name: Dimension name for error message
        
    Returns:
        Validated dimension
        
    Raises:
        ValueError: If dimension is not positive
    """
    if dim <= 0:
        raise ValueError(f"{name} must be positive, got {dim}")
    
    return dim


def validate_layers(layers: int, name: str) -> int:
    """
    Validate number of layers is positive.
    
    Args:
        layers: Number of layers
        name: Layer name for error message
        
    Returns:
        Validated layers
        
    Raises:
        ValueError: If layers is not positive
    """
    if layers <= 0:
        raise ValueError(f"{name} must be positive, got {layers}")
    
    return layers


def validate_heads(heads: int) -> int:
    """
    Validate number of attention heads is positive.
    
    Args:
        heads: Number of attention heads
        
    Returns:
        Validated heads
        
    Raises:
        ValueError: If heads is not positive
    """
    if heads <= 0:
        raise ValueError(f"Number of attention heads must be positive, got {heads}")
    
    return heads


def validate_curvature_reg(reg: float) -> float:
    """
    Validate curvature regularization is positive.
    
    Args:
        reg: Curvature regularization value
        
    Returns:
        Validated curvature regularization
        
    Raises:
        ValueError: If curvature regularization is not positive
    """
    if reg <= 0:
        raise ValueError(f"Curvature regularization must be positive, got {reg}")
    
    return reg


def validate_checkpoint_interval(interval: int) -> int:
    """
    Validate checkpoint interval is positive.
    
    Args:
        interval: Checkpoint interval value
        
    Returns:
        Validated checkpoint interval
        
    Raises:
        ValueError: If checkpoint interval is not positive
    """
    if interval <= 0:
        raise ValueError(f"Checkpoint interval must be positive, got {interval}")
    
    return interval


def validate_log_interval(interval: int) -> int:
    """
    Validate log interval is positive.
    
    Args:
        interval: Log interval value
        
    Returns:
        Validated log interval
        
    Raises:
        ValueError: If log interval is not positive
    """
    if interval <= 0:
        raise ValueError(f"Log interval must be positive, got {interval}")
    
    return interval
