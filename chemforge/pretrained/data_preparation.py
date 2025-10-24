"""
Data preparation for pre-trained models.

This module provides functionality for preparing and processing data
for pre-trained model training and evaluation.
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

from chemforge.data.chembl_loader import ChEMBLLoader
from chemforge.data.molecular_features import MolecularFeatures
from chemforge.data.rdkit_descriptors import RDKitDescriptors
from chemforge.data.data_preprocessor import DataPreprocessor
from chemforge.utils.validation import DataValidator
from chemforge.utils.logging_utils import Logger


class DataPreparator:
    """
    Data preparation for pre-trained models.
    
    This class handles the preparation and processing of data
    for pre-trained model training and evaluation.
    """
    
    def __init__(
        self,
        output_dir: str = "./prepared_data",
        log_dir: str = "./logs"
    ):
        """
        Initialize the data preparator.
        
        Args:
            output_dir: Directory to save prepared data
            log_dir: Directory for logs
        """
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.logger = Logger('data_preparator', log_dir=str(self.log_dir))
        self.data_validator = DataValidator()
        
        # Data processing components
        self.mol_features = MolecularFeatures()
        self.rdkit_descriptors = RDKitDescriptors()
        self.preprocessor = DataPreprocessor()
        
        # Scalers for normalization
        self.scalers = {}
        
        self.logger.info("DataPreparator initialized")
    
    def prepare_chembl_dataset(
        self,
        targets: List[str],
        min_activities: int = 100,
        activity_types: List[str] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        save_data: bool = True
    ) -> Dict[str, Any]:
        """
        Prepare ChEMBL dataset for pre-training.
        
        Args:
            targets: List of target ChEMBL IDs
            min_activities: Minimum number of activities per target
            activity_types: List of activity types to include
            test_size: Fraction of data for testing
            val_size: Fraction of data for validation
            random_state: Random state for reproducibility
            save_data: Whether to save the prepared data
            
        Returns:
            Dictionary containing prepared dataset information
        """
        self.logger.info("Preparing ChEMBL dataset for pre-training")
        
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
        
        # Extract molecular features
        features_data = self._extract_molecular_features(molecules)
        
        # Create unified dataset
        dataset = self._create_unified_dataset(molecules, activities, features_data)
        
        # Validate dataset
        validation_results = self.data_validator.validate_molecular_data(dataset)
        if not validation_results['valid']:
            self.logger.warning(f"Dataset validation issues: {validation_results['warnings']}")
        
        # Split dataset
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
            'test_size': len(test_data),
            'activity_types': activity_types,
            'min_activities': min_activities
        }
        
        # Prepare split data
        split_data = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
        # Normalize features
        normalized_data = self._normalize_features(split_data)
        
        # Prepare final dataset
        prepared_dataset = {
            'data_info': data_info,
            'split_data': normalized_data,
            'validation_results': validation_results,
            'preparation_metadata': {
                'preparation_date': pd.Timestamp.now().isoformat(),
                'random_state': random_state,
                'test_size': test_size,
                'val_size': val_size
            }
        }
        
        # Save data if requested
        if save_data:
            self._save_prepared_data(prepared_dataset, targets)
        
        self.logger.info(f"Dataset prepared: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        return prepared_dataset
    
    def _extract_molecular_features(self, molecules: pd.DataFrame) -> np.ndarray:
        """Extract molecular features from molecules."""
        self.logger.info("Extracting molecular features")
        
        # Extract basic molecular descriptors
        basic_features = self.mol_features.extract_features(molecules)
        
        # Extract RDKit descriptors
        rdkit_features = self.rdkit_descriptors.calculate_descriptors(molecules)
        
        # Combine features
        if len(basic_features) > 0 and len(rdkit_features) > 0:
            features = np.hstack([basic_features, rdkit_features])
        elif len(basic_features) > 0:
            features = basic_features
        elif len(rdkit_features) > 0:
            features = rdkit_features
        else:
            features = np.array([])
        
        self.logger.info(f"Extracted {features.shape[1]} molecular features")
        
        return features
    
    def _create_unified_dataset(
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
    
    def _normalize_features(self, split_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Normalize features in the split data."""
        self.logger.info("Normalizing features")
        
        # Get feature columns
        feature_cols = [col for col in split_data['train'].columns if col.startswith('feature_')]
        
        if len(feature_cols) == 0:
            self.logger.warning("No feature columns found for normalization")
            return split_data
        
        # Fit scaler on training data
        scaler = StandardScaler()
        train_features = split_data['train'][feature_cols].values
        scaler.fit(train_features)
        
        # Store scaler
        self.scalers['features'] = scaler
        
        # Transform all splits
        normalized_data = {}
        for split_name, data in split_data.items():
            normalized_data[split_name] = data.copy()
            normalized_data[split_name][feature_cols] = scaler.transform(data[feature_cols].values)
        
        self.logger.info("Features normalized using StandardScaler")
        
        return normalized_data
    
    def _save_prepared_data(self, prepared_dataset: Dict[str, Any], targets: List[str]):
        """Save the prepared dataset."""
        # Create filename based on targets
        targets_str = "_".join(targets[:3])  # Use first 3 targets for filename
        if len(targets) > 3:
            targets_str += f"_and_{len(targets)-3}_more"
        
        # Save dataset
        dataset_path = self.output_dir / f"chembl_dataset_{targets_str}.pkl"
        with open(dataset_path, 'wb') as f:
            pickle.dump(prepared_dataset, f)
        
        # Save scalers
        scalers_path = self.output_dir / f"scalers_{targets_str}.pkl"
        with open(scalers_path, 'wb') as f:
            pickle.dump(self.scalers, f)
        
        # Save metadata
        metadata_path = self.output_dir / f"metadata_{targets_str}.json"
        with open(metadata_path, 'w') as f:
            json.dump(prepared_dataset['preparation_metadata'], f, indent=2)
        
        self.logger.info(f"Saved prepared data to {dataset_path}")
        self.logger.info(f"Saved scalers to {scalers_path}")
        self.logger.info(f"Saved metadata to {metadata_path}")
    
    def load_prepared_data(self, targets: List[str]) -> Dict[str, Any]:
        """
        Load prepared data for the specified targets.
        
        Args:
            targets: List of target ChEMBL IDs
            
        Returns:
            Dictionary containing prepared dataset
        """
        # Create filename based on targets
        targets_str = "_".join(targets[:3])
        if len(targets) > 3:
            targets_str += f"_and_{len(targets)-3}_more"
        
        # Load dataset
        dataset_path = self.output_dir / f"chembl_dataset_{targets_str}.pkl"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Prepared data not found: {dataset_path}")
        
        with open(dataset_path, 'rb') as f:
            prepared_dataset = pickle.load(f)
        
        # Load scalers
        scalers_path = self.output_dir / f"scalers_{targets_str}.pkl"
        if scalers_path.exists():
            with open(scalers_path, 'rb') as f:
                self.scalers = pickle.load(f)
        
        self.logger.info(f"Loaded prepared data from {dataset_path}")
        
        return prepared_dataset
    
    def prepare_custom_dataset(
        self,
        molecules: pd.DataFrame,
        activities: pd.DataFrame,
        targets: List[str],
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        save_data: bool = True
    ) -> Dict[str, Any]:
        """
        Prepare custom dataset for pre-training.
        
        Args:
            molecules: DataFrame containing molecular data
            activities: DataFrame containing activity data
            targets: List of target names
            test_size: Fraction of data for testing
            val_size: Fraction of data for validation
            random_state: Random state for reproducibility
            save_data: Whether to save the prepared data
            
        Returns:
            Dictionary containing prepared dataset information
        """
        self.logger.info("Preparing custom dataset for pre-training")
        
        # Extract molecular features
        features_data = self._extract_molecular_features(molecules)
        
        # Create unified dataset
        dataset = self._create_unified_dataset(molecules, activities, features_data)
        
        # Validate dataset
        validation_results = self.data_validator.validate_molecular_data(dataset)
        if not validation_results['valid']:
            self.logger.warning(f"Dataset validation issues: {validation_results['warnings']}")
        
        # Split dataset
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
            'test_size': len(test_data),
            'dataset_type': 'custom'
        }
        
        # Prepare split data
        split_data = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
        # Normalize features
        normalized_data = self._normalize_features(split_data)
        
        # Prepare final dataset
        prepared_dataset = {
            'data_info': data_info,
            'split_data': normalized_data,
            'validation_results': validation_results,
            'preparation_metadata': {
                'preparation_date': pd.Timestamp.now().isoformat(),
                'random_state': random_state,
                'test_size': test_size,
                'val_size': val_size,
                'dataset_type': 'custom'
            }
        }
        
        # Save data if requested
        if save_data:
            self._save_custom_data(prepared_dataset, targets)
        
        self.logger.info(f"Custom dataset prepared: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        return prepared_dataset
    
    def _save_custom_data(self, prepared_dataset: Dict[str, Any], targets: List[str]):
        """Save the prepared custom dataset."""
        # Create filename based on targets
        targets_str = "_".join(targets[:3])
        if len(targets) > 3:
            targets_str += f"_and_{len(targets)-3}_more"
        
        # Save dataset
        dataset_path = self.output_dir / f"custom_dataset_{targets_str}.pkl"
        with open(dataset_path, 'wb') as f:
            pickle.dump(prepared_dataset, f)
        
        # Save scalers
        scalers_path = self.output_dir / f"custom_scalers_{targets_str}.pkl"
        with open(scalers_path, 'wb') as f:
            pickle.dump(self.scalers, f)
        
        # Save metadata
        metadata_path = self.output_dir / f"custom_metadata_{targets_str}.json"
        with open(metadata_path, 'w') as f:
            json.dump(prepared_dataset['preparation_metadata'], f, indent=2)
        
        self.logger.info(f"Saved custom dataset to {dataset_path}")
        self.logger.info(f"Saved scalers to {scalers_path}")
        self.logger.info(f"Saved metadata to {metadata_path}")
    
    def get_data_statistics(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get statistics for the prepared dataset.
        
        Args:
            dataset: Prepared dataset dictionary
            
        Returns:
            Dictionary containing dataset statistics
        """
        data_info = dataset['data_info']
        split_data = dataset['split_data']
        
        statistics = {
            'dataset_info': data_info,
            'split_sizes': {
                'train': len(split_data['train']),
                'val': len(split_data['val']),
                'test': len(split_data['test'])
            },
            'feature_statistics': {},
            'target_statistics': {}
        }
        
        # Feature statistics
        feature_cols = [col for col in split_data['train'].columns if col.startswith('feature_')]
        if len(feature_cols) > 0:
            train_features = split_data['train'][feature_cols]
            statistics['feature_statistics'] = {
                'mean': train_features.mean().to_dict(),
                'std': train_features.std().to_dict(),
                'min': train_features.min().to_dict(),
                'max': train_features.max().to_dict()
            }
        
        # Target statistics
        for target in data_info['targets']:
            target_col = f'target_{target}'
            if target_col in split_data['train'].columns:
                target_data = split_data['train'][target_col]
                statistics['target_statistics'][target] = {
                    'mean': target_data.mean(),
                    'std': target_data.std(),
                    'min': target_data.min(),
                    'max': target_data.max(),
                    'count': target_data.count()
                }
        
        return statistics
    
    def export_dataset(
        self,
        dataset: Dict[str, Any],
        export_format: str = 'csv',
        output_path: Optional[str] = None
    ) -> str:
        """
        Export the prepared dataset to a file.
        
        Args:
            dataset: Prepared dataset dictionary
            export_format: Format to export ('csv', 'json', 'parquet')
            output_path: Path to save the exported file
            
        Returns:
            Path to the exported file
        """
        if output_path is None:
            output_path = self.output_dir / f"exported_dataset.{export_format}"
        else:
            output_path = Path(output_path)
        
        split_data = dataset['split_data']
        
        if export_format == 'csv':
            # Export each split as a separate CSV
            for split_name, data in split_data.items():
                split_path = output_path.parent / f"{output_path.stem}_{split_name}.csv"
                data.to_csv(split_path, index=False)
                self.logger.info(f"Exported {split_name} data to {split_path}")
        
        elif export_format == 'json':
            # Export as JSON
            export_data = {
                'data_info': dataset['data_info'],
                'split_data': {name: data.to_dict('records') for name, data in split_data.items()}
            }
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            self.logger.info(f"Exported dataset to {output_path}")
        
        elif export_format == 'parquet':
            # Export each split as a separate Parquet file
            for split_name, data in split_data.items():
                split_path = output_path.parent / f"{output_path.stem}_{split_name}.parquet"
                data.to_parquet(split_path, index=False)
                self.logger.info(f"Exported {split_name} data to {split_path}")
        
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
        
        return str(output_path)
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available prepared datasets."""
        dataset_files = list(self.output_dir.glob("chembl_dataset_*.pkl"))
        custom_dataset_files = list(self.output_dir.glob("custom_dataset_*.pkl"))
        
        datasets = []
        for file_path in dataset_files + custom_dataset_files:
            datasets.append(file_path.stem)
        
        return datasets
