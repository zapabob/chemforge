"""
Multi-Target pIC50 Predictor Library

Advanced multi-target pIC50 prediction platform for psychoactive and drug-like compounds.
Supports DAT, 5HT2A, CB1, CB2, and opioid receptors with correct ChEMBL database IDs.
"""

__version__ = "1.0.0"
__author__ = "AI Assistant (なんｊ風)"
__email__ = "ai@example.com"
__license__ = "MIT"

# Core imports
from .core.transformer_model import TransformerRegressor
from .core.gnn_model import GNNRegressor
from .core.ensemble_model import EnsembleRegressor

# Data imports
from .data.chembl_loader import ChEMBLLoader
from .data.molecular_features import MolecularFeatures
from .data.rdkit_descriptors import RDKitDescriptors
# from .data.smarts_scaffolds import SMARTSScaffolds

# Model imports
from .models.transformer import MolecularTransformer
from .models.gnn_model import GNNRegressor
from .models.ensemble_model import EnsembleRegressor

# Training imports
from .training.trainer import Trainer
# from .training.losses import MultiTaskLoss
# from .training.metrics import PredictionMetrics

# Optimization imports
# from .optimization.optuna_optimizer import OptunaOptimizer
# from .optimization.hyperparameter_tuning import HyperparameterTuning

# Targets imports
from .targets.chembl_targets import ChEMBLTargets, get_chembl_targets
# from .targets.target_utils import TargetUtils

# GUI imports
from .gui.main_window import MainWindow
from .gui.prediction_widget import PredictionWidget
from .gui.visualization_widget import VisualizationWidget
from .gui.chat_widget import ChatWidget

# CLI imports
from .cli.main import main
from .cli.train import train_cli
from .cli.predict import predict_cli
from .cli.chat import chat_cli

# Main predictor class
from .core.multi_target_predictor import MultiTargetPredictor

# Constants
from .targets.chembl_targets import CHEMBL_TARGETS, TARGET_CONFIGS

__all__ = [
    # Core
    'MolecularTransformer',
    'GNNRegressor',
    'EnsembleRegressor',
    
    # Data
    'ChEMBLLoader',
    'MolecularFeatures',
    'RDKitDescriptors',
    # 'SMARTSScaffolds',
    
    # Models
    'MolecularTransformer',
    'GNNRegressor',
    'EnsembleRegressor',
    
    # Training
    'Trainer',
    # 'MultiTaskLoss',
    # 'PredictionMetrics',
    
    # Optimization
    'OptunaOptimizer',
    'HyperparameterTuning',
    
    # Targets
    'ChEMBLTargets',
    'get_chembl_targets',
    'TargetUtils',
    
    # GUI
    'MainWindow',
    'PredictionWidget',
    'VisualizationWidget',
    'ChatWidget',
    
    # CLI
    'main',
    'train_cli',
    'predict_cli',
    'chat_cli',
    
    # Main predictor
    'MultiTargetPredictor',
    
    # Constants
    'CHEMBL_TARGETS',
    'TARGET_CONFIGS',
]