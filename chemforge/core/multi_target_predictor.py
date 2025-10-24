"""
Multi-Target pIC50 Predictor

Main predictor class for multi-target pIC50 prediction using correct ChEMBL database IDs.
Integrated with scaffold features and ADMET prediction.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import json
import yaml

from .transformer_model import TransformerRegressor
from .gnn_model import GNNRegressor
from .ensemble_model import EnsembleRegressor
from ..targets import get_chembl_targets, get_target_info
from ..data.molecular_preprocessor import MolecularPreprocessor
from ..data.scaffold_detector import ScaffoldDetector
from ..data.admet_predictor import ADMETPredictor
from ..training.metrics import PredictionMetrics


class MultiTargetPredictor:
    """Multi-target pIC50 predictor with correct ChEMBL database IDs."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        targets: Optional[List[str]] = None,
        device: str = 'auto',
        config_path: Optional[str] = None,
        use_scaffold_features: bool = True,
        use_admet: bool = True
    ):
        """
        Initialize multi-target predictor.
        
        Args:
            model_path: Path to trained model
            targets: List of target names or ChEMBL IDs
            device: Device to use ('auto', 'cpu', 'cuda')
            config_path: Path to configuration file
            use_scaffold_features: Use scaffold features
            use_admet: Use ADMET prediction
        """
        self.device = self._get_device(device)
        self.targets = targets or ['DAT', '5HT2A', 'CB1', 'CB2', 'MOR']
        self.chembl_targets = get_chembl_targets()
        self.metrics = PredictionMetrics()
        
        # Initialize molecular preprocessor
        self.molecular_preprocessor = MolecularPreprocessor(
            use_scaffold_features=use_scaffold_features,
            use_admet=use_admet,
            verbose=True
        )
        
        # Initialize scaffold detector
        self.scaffold_detector = ScaffoldDetector(verbose=True)
        
        # Initialize ADMET predictor
        self.admet_predictor = ADMETPredictor(verbose=True)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize models
        self.models = {}
        self._load_models(model_path)
        
        # Target mapping
        self.target_mapping = self._create_target_mapping()
        
    def _get_device(self, device: str) -> torch.device:
        """Get device for computation."""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'model': {
                'type': 'ensemble',
                'transformer_weight': 0.6,
                'gnn_weight': 0.4,
                'uncertainty_threshold': 0.3
            },
            'features': {
                'rdkit_descriptors': True,
                'morgan_fingerprints': True,
                'scaffold_features': True,
                'admet_features': True,
                'cns_mpo': True
            },
            'prediction': {
                'batch_size': 32,
                'uncertainty_estimation': True,
                'confidence_threshold': 0.8
            }
        }
    
    def _load_models(self, model_path: Optional[str]):
        """Load trained models."""
        if model_path and Path(model_path).exists():
            # Load ensemble model
            self.models['ensemble'] = EnsembleRegressor()
            self.models['ensemble'].load_state_dict(torch.load(model_path, map_location=self.device))
            self.models['ensemble'].eval()
        else:
            # Initialize default models with 2279 input dimensions
            self.models['transformer'] = TransformerRegressor(
                input_dim=2279,  # Combined features: RDKit + Fingerprint + Scaffold + ADMET + CNS-MPO
                hidden_dim=512,
                num_layers=4,
                num_heads=8,
                num_targets=len(self.targets)
            ).to(self.device)
            
            self.models['gnn'] = GNNRegressor(
                node_features=78,  # Atomic features
                edge_features=12,  # Bond features
                hidden_dim=256,
                num_layers=3,
                num_targets=len(self.targets),
                use_attention=True,
                use_scaffold_features=True
            ).to(self.device)
            
            self.models['ensemble'] = EnsembleRegressor(
                transformer_model=self.models['transformer'],
                gnn_model=self.models['gnn'],
                weights=[0.6, 0.4],
                num_targets=len(self.targets),
                use_uncertainty=True
            ).to(self.device)
    
    def _create_target_mapping(self) -> Dict[str, str]:
        """Create mapping between target names and ChEMBL IDs."""
        mapping = {}
        for target in self.targets:
            if target in self.chembl_targets.targets:
                mapping[target] = self.chembl_targets.get_chembl_id(target)
            else:
                # Assume it's already a ChEMBL ID
                mapping[target] = target
        return mapping
    
    def predict(
        self,
        smiles: str,
        target: Optional[str] = None,
        return_uncertainty: bool = True
    ) -> Dict:
        """
        Predict pIC50 for a single molecule.
        
        Args:
            smiles: SMILES string
            target: Target name or ChEMBL ID (if None, predicts for all targets)
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Dictionary with predictions
        """
        # Extract molecular features
        features = self._extract_features(smiles)
        
        if target:
            # Single target prediction
            target_id = self.target_mapping.get(target, target)
            prediction = self._predict_single_target(features, target_id, return_uncertainty)
            return {target: prediction}
        else:
            # Multi-target prediction
            predictions = {}
            for target_name, target_id in self.target_mapping.items():
                prediction = self._predict_single_target(features, target_id, return_uncertainty)
                predictions[target_name] = prediction
            return predictions
    
    def predict_multi_target(
        self,
        smiles: str,
        targets: List[str],
        return_uncertainty: bool = True
    ) -> Dict:
        """
        Predict pIC50 for multiple targets.
        
        Args:
            smiles: SMILES string
            targets: List of target names or ChEMBL IDs
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Dictionary with predictions for each target
        """
        predictions = {}
        for target in targets:
            prediction = self.predict(smiles, target, return_uncertainty)
            predictions.update(prediction)
        return predictions
    
    def predict_batch(
        self,
        smiles_list: List[str],
        target: Optional[str] = None,
        return_uncertainty: bool = True
    ) -> List[Dict]:
        """
        Predict pIC50 for a batch of molecules.
        
        Args:
            smiles_list: List of SMILES strings
            target: Target name or ChEMBL ID (if None, predicts for all targets)
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            List of dictionaries with predictions
        """
        predictions = []
        for smiles in smiles_list:
            prediction = self.predict(smiles, target, return_uncertainty)
            predictions.append(prediction)
        return predictions
    
    def _extract_features(self, smiles: str) -> Dict:
        """Extract molecular features from SMILES."""
        # Use molecular preprocessor for integrated feature extraction
        features = self.molecular_preprocessor.preprocess(smiles)
        
        # Extract combined features for Transformer
        combined_features = self.molecular_preprocessor.extract_combined_features(smiles)
        features['combined'] = combined_features
        
        return features
    
    def _predict_single_target(
        self,
        features: Dict,
        target_id: str,
        return_uncertainty: bool
    ) -> Dict:
        """Predict pIC50 for a single target."""
        # Get target index
        target_idx = self.targets.index(target_id) if target_id in self.targets else 0
        
        # Make prediction
        with torch.no_grad():
            if 'ensemble' in self.models:
                # Ensemble prediction with uncertainty
                if return_uncertainty:
                    prediction, uncertainty, confidence = self.models['ensemble'].predict_with_uncertainty(
                        features['combined'].unsqueeze(0).to(self.device),
                        num_samples=10
                    )
                    pIC50 = prediction[0, target_idx].item()
                    uncertainty_val = uncertainty[0, target_idx].item()
                    confidence_val = confidence[0, target_idx].item()
                else:
                    prediction = self.models['ensemble'](features['combined'].unsqueeze(0).to(self.device))
                    pIC50 = prediction[0, target_idx].item()
                    uncertainty_val = None
                    confidence_val = 0.8
            else:
                # Single model prediction
                if 'transformer' in self.models:
                    prediction = self.models['transformer'](features['combined'].unsqueeze(0).to(self.device))
                elif 'gnn' in self.models:
                    # For GNN, we need graph features
                    if 'node_features' in features and 'edge_features' in features and 'adj_matrix' in features:
                        prediction = self.models['gnn'](
                            features['node_features'].unsqueeze(0).to(self.device),
                            features['edge_features'].unsqueeze(0).to(self.device),
                            features['adj_matrix'].unsqueeze(0).to(self.device),
                            features.get('scaffold_features', None).unsqueeze(0).to(self.device) if 'scaffold_features' in features else None
                        )
                    else:
                        # Fallback to combined features
                        prediction = self.models['gnn'](features['combined'].unsqueeze(0).to(self.device))
                else:
                    raise ValueError("No trained models available")
                
                pIC50 = prediction[0, target_idx].item()
                uncertainty_val = None
                confidence_val = 0.8
        
        return {
            'pIC50': pIC50,
            'uncertainty': uncertainty_val,
            'target_id': target_id,
            'confidence': confidence_val
        }
    
    def _prepare_input_tensors(self, features: Dict) -> Dict:
        """Prepare input tensors for models."""
        tensors = {}
        
        for feature_type, feature_data in features.items():
            if isinstance(feature_data, np.ndarray):
                tensors[feature_type] = torch.tensor(feature_data, dtype=torch.float32).to(self.device)
            elif isinstance(feature_data, torch.Tensor):
                tensors[feature_type] = feature_data.to(self.device)
            else:
                # Convert to tensor
                tensors[feature_type] = torch.tensor(feature_data, dtype=torch.float32).to(self.device)
        
        return tensors
    
    def get_scaffold_info(self, smiles: str) -> Dict:
        """Get scaffold information for a molecule."""
        return self.scaffold_detector.get_scaffold_summary(smiles)
    
    def get_admet_info(self, smiles: str) -> Dict:
        """Get ADMET information for a molecule."""
        return self.admet_predictor.get_admet_summary(smiles)
    
    def get_molecular_info(self, smiles: str) -> Dict:
        """Get comprehensive molecular information."""
        info = {
            'smiles': smiles,
            'scaffold_info': self.get_scaffold_info(smiles),
            'admet_info': self.get_admet_info(smiles),
            'features': self._extract_features(smiles)
        }
        return info
    
    def get_target_info(self, target: str) -> Dict:
        """Get information about a target."""
        return get_target_info(target)
    
    def get_available_targets(self) -> List[str]:
        """Get list of available targets."""
        return self.chembl_targets.get_available_targets()
    
    def save_model(self, path: str):
        """Save trained model."""
        if 'ensemble' in self.models:
            torch.save(self.models['ensemble'].state_dict(), path)
        else:
            raise ValueError("No ensemble model to save")
    
    def load_model(self, path: str):
        """Load trained model."""
        if Path(path).exists():
            self.models['ensemble'] = EnsembleRegressor()
            self.models['ensemble'].load_state_dict(torch.load(path, map_location=self.device))
            self.models['ensemble'].eval()
        else:
            raise FileNotFoundError(f"Model file not found: {path}")
    
    def evaluate(
        self,
        test_data: pd.DataFrame,
        target: str,
        metrics: List[str] = ['mse', 'mae', 'r2']
    ) -> Dict:
        """
        Evaluate model performance on test data.
        
        Args:
            test_data: Test dataset
            target: Target name or ChEMBL ID
            metrics: List of metrics to calculate
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = []
        true_values = []
        
        for _, row in test_data.iterrows():
            smiles = row['smiles']
            true_pIC50 = row['pIC50']
            
            prediction = self.predict(smiles, target)
            pred_pIC50 = prediction[target]['pIC50']
            
            predictions.append(pred_pIC50)
            true_values.append(true_pIC50)
        
        # Calculate metrics
        results = {}
        for metric in metrics:
            results[metric] = self.metrics.calculate_metric(
                true_values, predictions, metric
            )
        
        return results
