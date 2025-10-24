"""
Configuration utilities for ChemForge platform.

This module provides configuration management functionality including model
configuration, training configuration, and settings management.
"""

import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import os


@dataclass
class ModelConfig:
    """Model configuration class."""
    
    # Model architecture
    model_type: str = 'transformer'
    input_dim: int = 100
    output_dim: int = 3
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    
    # PWA+PET specific
    use_pwa_pet: bool = True
    pwa_buckets: Dict[str, int] = None
    use_rope: bool = True
    use_pet: bool = True
    pet_curv_reg: float = 1e-5
    
    # GNN specific
    gnn_type: str = 'gat'
    gnn_layers: int = 3
    gnn_hidden_dim: int = 128
    
    # Ensemble specific
    ensemble_models: List[str] = None
    ensemble_weights: List[float] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.pwa_buckets is None:
            self.pwa_buckets = {'trivial': 1, 'fund': 5, 'adj': 2}
        
        if self.ensemble_models is None:
            self.ensemble_models = ['transformer', 'gnn']
        
        if self.ensemble_weights is None:
            self.ensemble_weights = [0.5, 0.5]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create from dictionary."""
        return cls(**config_dict)


@dataclass
class TrainingConfig:
    """Training configuration class."""
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = 'adam'
    scheduler: str = 'cosine'
    
    # Data parameters
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42
    
    # Training options
    use_amp: bool = True
    gradient_clip: float = 1.0
    early_stopping: bool = True
    patience: int = 10
    
    # Checkpointing
    checkpoint_interval: int = 10
    save_best: bool = True
    save_last: bool = True
    
    # Logging
    log_interval: int = 10
    log_level: str = 'INFO'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create from dictionary."""
        return cls(**config_dict)


@dataclass
class DataConfig:
    """Data configuration class."""
    
    # Data paths
    data_path: str = './data'
    train_path: str = './data/train.csv'
    val_path: str = './data/val.csv'
    test_path: str = './data/test.csv'
    
    # Data processing
    normalize: bool = True
    feature_selection: bool = True
    feature_threshold: float = 0.01
    
    # Molecular features
    use_rdkit: bool = True
    use_morgan: bool = True
    morgan_radius: int = 2
    morgan_bits: int = 2048
    
    # Scaffold analysis
    use_scaffold: bool = True
    scaffold_types: List[str] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.scaffold_types is None:
            self.scaffold_types = ['trivial', 'fund', 'adj']
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataConfig':
        """Create from dictionary."""
        return cls(**config_dict)


@dataclass
class ADMETConfig:
    """ADMET configuration class."""
    
    # ADMET properties
    properties: List[str] = None
    use_cns_mpo: bool = True
    
    # Model settings
    admet_model_type: str = 'transformer'
    admet_hidden_dim: int = 128
    admet_num_layers: int = 3
    
    # Prediction settings
    confidence_threshold: float = 0.7
    ensemble_predictions: bool = True
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.properties is None:
            self.properties = ['absorption', 'distribution', 'metabolism', 
                             'excretion', 'toxicity']
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ADMETConfig':
        """Create from dictionary."""
        return cls(**config_dict)


class ConfigManager:
    """Configuration manager class."""
    
    def __init__(self, config_path: Optional[str] = None, 
                 logger: Optional[logging.Logger] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.config_path = Path(config_path) if config_path else None
        self.config = {}
        
        if self.config_path and self.config_path.exists():
            self.load_config()
    
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        if config_path:
            self.config_path = Path(config_path)
        
        if not self.config_path or not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        # Determine file format
        if self.config_path.suffix == '.json':
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        elif self.config_path.suffix in ['.yaml', '.yml']:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {self.config_path.suffix}")
        
        self.logger.info(f"Loaded configuration from: {self.config_path}")
        return self.config
    
    def save_config(self, config_path: Optional[str] = None) -> Path:
        """
        Save configuration to file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration file path
        """
        if config_path:
            self.config_path = Path(config_path)
        
        if not self.config_path:
            raise ValueError("Configuration file path not specified")
        
        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save based on file format
        if self.config_path.suffix == '.json':
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2, default=str)
        elif self.config_path.suffix in ['.yaml', '.yml']:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported configuration file format: {self.config_path.suffix}")
        
        self.logger.info(f"Saved configuration to: {self.config_path}")
        return self.config_path
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set_config(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Configuration value
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set value
        config[keys[-1]] = value
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with dictionary.
        
        Args:
            updates: Dictionary of configuration updates
        """
        for key, value in updates.items():
            self.set_config(key, value)
    
    def get_model_config(self) -> ModelConfig:
        """
        Get model configuration.
        
        Returns:
            Model configuration object
        """
        model_config_dict = self.get_config('model', {})
        return ModelConfig.from_dict(model_config_dict)
    
    def set_model_config(self, model_config: ModelConfig) -> None:
        """
        Set model configuration.
        
        Args:
            model_config: Model configuration object
        """
        self.set_config('model', model_config.to_dict())
    
    def get_training_config(self) -> TrainingConfig:
        """
        Get training configuration.
        
        Returns:
            Training configuration object
        """
        training_config_dict = self.get_config('training', {})
        return TrainingConfig.from_dict(training_config_dict)
    
    def set_training_config(self, training_config: TrainingConfig) -> None:
        """
        Set training configuration.
        
        Args:
            training_config: Training configuration object
        """
        self.set_config('training', training_config.to_dict())
    
    def get_data_config(self) -> DataConfig:
        """
        Get data configuration.
        
        Returns:
            Data configuration object
        """
        data_config_dict = self.get_config('data', {})
        return DataConfig.from_dict(data_config_dict)
    
    def set_data_config(self, data_config: DataConfig) -> None:
        """
        Set data configuration.
        
        Args:
            data_config: Data configuration object
        """
        self.set_config('data', data_config.to_dict())
    
    def get_admet_config(self) -> ADMETConfig:
        """
        Get ADMET configuration.
        
        Returns:
            ADMET configuration object
        """
        admet_config_dict = self.get_config('admet', {})
        return ADMETConfig.from_dict(admet_config_dict)
    
    def set_admet_config(self, admet_config: ADMETConfig) -> None:
        """
        Set ADMET configuration.
        
        Args:
            admet_config: ADMET configuration object
        """
        self.set_config('admet', admet_config.to_dict())
    
    def create_default_config(self) -> Dict[str, Any]:
        """
        Create default configuration.
        
        Returns:
            Default configuration dictionary
        """
        default_config = {
            'model': ModelConfig().to_dict(),
            'training': TrainingConfig().to_dict(),
            'data': DataConfig().to_dict(),
            'admet': ADMETConfig().to_dict(),
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'version': '1.0.0',
                'description': 'ChemForge default configuration'
            }
        }
        
        self.config = default_config
        return default_config
    
    def validate_config(self) -> List[str]:
        """
        Validate configuration.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Validate model configuration
        try:
            model_config = self.get_model_config()
            if model_config.input_dim <= 0:
                errors.append("Model input_dim must be positive")
            if model_config.output_dim <= 0:
                errors.append("Model output_dim must be positive")
            if model_config.hidden_dim <= 0:
                errors.append("Model hidden_dim must be positive")
            if model_config.num_layers <= 0:
                errors.append("Model num_layers must be positive")
            if model_config.num_heads <= 0:
                errors.append("Model num_heads must be positive")
            if not 0 <= model_config.dropout <= 1:
                errors.append("Model dropout must be between 0 and 1")
        except Exception as e:
            errors.append(f"Model configuration error: {e}")
        
        # Validate training configuration
        try:
            training_config = self.get_training_config()
            if training_config.epochs <= 0:
                errors.append("Training epochs must be positive")
            if training_config.batch_size <= 0:
                errors.append("Training batch_size must be positive")
            if training_config.learning_rate <= 0:
                errors.append("Training learning_rate must be positive")
            if not 0 <= training_config.train_split <= 1:
                errors.append("Training train_split must be between 0 and 1")
            if not 0 <= training_config.val_split <= 1:
                errors.append("Training val_split must be between 0 and 1")
            if not 0 <= training_config.test_split <= 1:
                errors.append("Training test_split must be between 0 and 1")
        except Exception as e:
            errors.append(f"Training configuration error: {e}")
        
        # Validate data configuration
        try:
            data_config = self.get_data_config()
            if not 0 <= data_config.feature_threshold <= 1:
                errors.append("Data feature_threshold must be between 0 and 1")
            if data_config.morgan_radius <= 0:
                errors.append("Data morgan_radius must be positive")
            if data_config.morgan_bits <= 0:
                errors.append("Data morgan_bits must be positive")
        except Exception as e:
            errors.append(f"Data configuration error: {e}")
        
        # Validate ADMET configuration
        try:
            admet_config = self.get_admet_config()
            if not 0 <= admet_config.confidence_threshold <= 1:
                errors.append("ADMET confidence_threshold must be between 0 and 1")
            if admet_config.admet_hidden_dim <= 0:
                errors.append("ADMET admet_hidden_dim must be positive")
            if admet_config.admet_num_layers <= 0:
                errors.append("ADMET admet_num_layers must be positive")
        except Exception as e:
            errors.append(f"ADMET configuration error: {e}")
        
        return errors
    
    def merge_configs(self, other_config: Dict[str, Any]) -> None:
        """
        Merge configuration with another configuration.
        
        Args:
            other_config: Other configuration dictionary
        """
        def merge_dicts(dict1, dict2):
            """Recursively merge dictionaries."""
            for key, value in dict2.items():
                if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
                    merge_dicts(dict1[key], value)
                else:
                    dict1[key] = value
        
        merge_dicts(self.config, other_config)
        self.logger.info("Merged configuration with other config")
    
    def export_config(self, export_path: str, format: str = 'json') -> Path:
        """
        Export configuration to file.
        
        Args:
            export_path: Export file path
            format: Export format ('json' or 'yaml')
            
        Returns:
            Export file path
        """
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(export_path, 'w') as f:
                json.dump(self.config, f, indent=2, default=str)
        elif format == 'yaml':
            with open(export_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Exported configuration to: {export_path}")
        return export_path
