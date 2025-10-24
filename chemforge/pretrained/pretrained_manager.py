"""
Pretrained Manager Module

事前学習管理モジュール
改良されたPWA+PET Transformerを活用した効率的な事前学習システム
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from pathlib import Path
import json
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 既存モジュール活用
from chemforge.models.transformer import PotencyTransformer
from chemforge.models.gnn import PotencyGNN
from chemforge.models.ensemble import PotencyEnsemble
from chemforge.training.training_manager import TrainingManager
from chemforge.training.inference_manager import InferenceManager
from chemforge.data.chembl_loader import ChEMBLLoader
from chemforge.data.molecular_features import MolecularFeatures
from chemforge.data.rdkit_descriptors import RDKitDescriptors
from chemforge.utils.config_utils import ConfigManager
from chemforge.utils.logging_utils import Logger
from chemforge.utils.validation import DataValidator

logger = logging.getLogger(__name__)

class PretrainedManager:
    """
    事前学習管理クラス
    
    改良されたPWA+PET Transformerを活用した効率的な事前学習システム
    """
    
    def __init__(self, config_path: Optional[str] = None, cache_dir: str = "cache"):
        """
        初期化
        
        Args:
            config_path: 設定ファイルパス
            cache_dir: キャッシュディレクトリ
        """
        self.config_path = config_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 既存モジュール活用
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load_config()
        self.logger = Logger("PretrainedManager")
        self.validator = DataValidator()
        
        # データローダー
        self.chembl_loader = ChEMBLLoader(config_path, cache_dir)
        self.molecular_features = MolecularFeatures(config_path, cache_dir)
        self.rdkit_descriptors = RDKitDescriptors(config_path, cache_dir)
        
        # 学習・推論管理
        self.training_manager = TrainingManager(config_path, cache_dir)
        self.inference_manager = InferenceManager(config_path, cache_dir)
        
        # 事前学習設定
        self.pretrained_config = self.config.get('pretrained', {})
        self.model_config = self.pretrained_config.get('model', {})
        self.training_config = self.pretrained_config.get('training', {})
        self.data_config = self.pretrained_config.get('data', {})
        
        # モデルレジストリ
        self.model_registry = {}
        self.model_paths = {}
        
        logger.info("PretrainedManager initialized")
    
    def create_pretrained_models(self, target_chembl_ids: List[str],
                               model_types: List[str] = ['transformer', 'gnn', 'ensemble']) -> Dict[str, nn.Module]:
        """
        事前学習モデル作成
        
        Args:
            target_chembl_ids: ターゲットChEMBL IDリスト
            model_types: モデルタイプリスト
            
        Returns:
            作成済みモデル辞書
        """
        logger.info(f"Creating pretrained models for {len(target_chembl_ids)} targets")
        
        models = {}
        
        for model_type in model_types:
            if model_type == 'transformer':
                model = self._create_transformer_model()
            elif model_type == 'gnn':
                model = self._create_gnn_model()
            elif model_type == 'ensemble':
                model = self._create_ensemble_model()
            else:
                logger.warning(f"Unsupported model type: {model_type}")
                continue
            
            models[model_type] = model
            self.model_registry[model_type] = model
        
        logger.info(f"Created {len(models)} pretrained models")
        return models
    
    def _create_transformer_model(self) -> PotencyTransformer:
        """
        Transformerモデル作成
        
        Returns:
            PotencyTransformer
        """
        model_config = self.model_config.get('transformer', {})
        
        model = PotencyTransformer(
            d_model=model_config.get('d_model', 512),
            n_layers=model_config.get('n_layers', 8),
            n_heads=model_config.get('n_heads', 8),
            d_ff=model_config.get('d_ff', 2048),
            dropout=model_config.get('dropout', 0.0),
            max_len=model_config.get('max_len', 256),
            vocab_size=model_config.get('vocab_size', 1000),
            num_tasks=model_config.get('num_tasks', 2),
            task_types=model_config.get('task_types', ['regression', 'classification'])
        )
        
        logger.info("Created PotencyTransformer model")
        return model
    
    def _create_gnn_model(self) -> PotencyGNN:
        """
        GNNモデル作成
        
        Returns:
            PotencyGNN
        """
        model_config = self.model_config.get('gnn', {})
        
        model = PotencyGNN(
            node_features=model_config.get('node_features', 78),
            edge_features=model_config.get('edge_features', 4),
            hidden_dim=model_config.get('hidden_dim', 256),
            num_layers=model_config.get('num_layers', 4),
            dropout=model_config.get('dropout', 0.0),
            num_tasks=model_config.get('num_tasks', 2),
            task_types=model_config.get('task_types', ['regression', 'classification'])
        )
        
        logger.info("Created PotencyGNN model")
        return model
    
    def _create_ensemble_model(self) -> PotencyEnsemble:
        """
        Ensembleモデル作成
        
        Returns:
            PotencyEnsemble
        """
        model_config = self.model_config.get('ensemble', {})
        
        # ベースモデル作成
        transformer_model = self._create_transformer_model()
        gnn_model = self._create_gnn_model()
        
        model = PotencyEnsemble(
            models=[transformer_model, gnn_model],
            ensemble_method=model_config.get('ensemble_method', 'weighted_average'),
            num_tasks=model_config.get('num_tasks', 2),
            task_types=model_config.get('task_types', ['regression', 'classification'])
        )
        
        logger.info("Created PotencyEnsemble model")
        return model
    
    def train_pretrained_models(self, models: Dict[str, nn.Module],
                               train_loader: DataLoader,
                               val_loader: DataLoader,
                               num_epochs: int = 100) -> Dict[str, Dict[str, Any]]:
        """
        事前学習モデル学習
        
        Args:
            models: モデル辞書
            train_loader: 学習データローダー
            val_loader: 検証データローダー
            num_epochs: エポック数
            
        Returns:
            学習結果辞書
        """
        logger.info(f"Training {len(models)} pretrained models for {num_epochs} epochs")
        
        training_results = {}
        
        for model_name, model in models.items():
            logger.info(f"Training {model_name} model...")
            
            # 学習実行
            result = self.training_manager.train_model(
                model, train_loader, val_loader, num_epochs
            )
            
            training_results[model_name] = result
            
            # モデル保存
            model_path = self.cache_dir / f"{model_name}_pretrained.pth"
            self._save_model(model, result, model_path)
            self.model_paths[model_name] = model_path
            
            logger.info(f"{model_name} model training completed")
        
        logger.info("All pretrained models training completed")
        return training_results
    
    def _save_model(self, model: nn.Module, training_result: Dict[str, Any], 
                   model_path: Path):
        """
        モデル保存
        
        Args:
            model: モデル
            training_result: 学習結果
            model_path: 保存パス
        """
        try:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'training_result': training_result,
                'model_config': self.model_config,
                'timestamp': time.time()
            }
            
            torch.save(checkpoint, model_path)
            logger.info(f"Model saved to: {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_pretrained_model(self, model_name: str, 
                             model_path: Optional[str] = None) -> nn.Module:
        """
        事前学習モデル読み込み
        
        Args:
            model_name: モデル名
            model_path: モデルパス
            
        Returns:
            読み込み済みモデル
        """
        if model_path is None:
            model_path = self.model_paths.get(model_name)
            if model_path is None:
                raise ValueError(f"Model path not found for {model_name}")
        
        logger.info(f"Loading pretrained model: {model_name}")
        
        # チェックポイント読み込み
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # モデル作成
        if model_name == 'transformer':
            model = self._create_transformer_model()
        elif model_name == 'gnn':
            model = self._create_gnn_model()
        elif model_name == 'ensemble':
            model = self._create_ensemble_model()
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
        # 状態読み込み
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logger.info(f"Pretrained model {model_name} loaded successfully")
        return model
    
    def evaluate_pretrained_models(self, models: Dict[str, nn.Module],
                                  test_loader: DataLoader) -> Dict[str, Dict[str, float]]:
        """
        事前学習モデル評価
        
        Args:
            models: モデル辞書
            test_loader: テストデータローダー
            
        Returns:
            評価結果辞書
        """
        logger.info(f"Evaluating {len(models)} pretrained models")
        
        evaluation_results = {}
        
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name} model...")
            
            # 評価実行
            metrics = self.inference_manager.evaluate_model(model, test_loader)
            evaluation_results[model_name] = metrics
            
            logger.info(f"{model_name} model evaluation completed")
        
        logger.info("All pretrained models evaluation completed")
        return evaluation_results
    
    def predict_with_pretrained_models(self, models: Dict[str, nn.Module],
                                      data_loader: DataLoader) -> Dict[str, Dict[str, np.ndarray]]:
        """
        事前学習モデル予測
        
        Args:
            models: モデル辞書
            data_loader: データローダー
            
        Returns:
            予測結果辞書
        """
        logger.info(f"Predicting with {len(models)} pretrained models")
        
        prediction_results = {}
        
        for model_name, model in models.items():
            logger.info(f"Predicting with {model_name} model...")
            
            # 予測実行
            predictions = self.inference_manager.predict(model, data_loader)
            prediction_results[model_name] = predictions
            
            logger.info(f"{model_name} model prediction completed")
        
        logger.info("All pretrained models prediction completed")
        return prediction_results
    
    def create_pretrained_dataset(self, target_chembl_ids: List[str],
                                 include_features: bool = True,
                                 include_descriptors: bool = True) -> pd.DataFrame:
        """
        事前学習データセット作成
        
        Args:
            target_chembl_ids: ターゲットChEMBL IDリスト
            include_features: 分子特徴量含むフラグ
            include_descriptors: RDKit記述子含むフラグ
            
        Returns:
            事前学習データセット
        """
        logger.info(f"Creating pretrained dataset for {len(target_chembl_ids)} targets")
        
        # ChEMBLデータロード
        chembl_data = self.chembl_loader.load_data(target_chembl_ids)
        
        if chembl_data.empty:
            logger.warning("No ChEMBL data loaded")
            return pd.DataFrame()
        
        logger.info(f"Loaded {len(chembl_data)} ChEMBL entries")
        
        # 分子特徴量追加
        if include_features:
            logger.info("Adding molecular features...")
            chembl_data = self.molecular_features.featurize_dataframe(
                chembl_data, smiles_col='smiles', include_3d=True
            )
            logger.info(f"Added molecular features. Shape: {chembl_data.shape}")
        
        # RDKit記述子追加
        if include_descriptors:
            logger.info("Adding RDKit descriptors...")
            chembl_data = self.rdkit_descriptors.featurize_dataframe(
                chembl_data, smiles_col='smiles',
                include_morgan=True, include_maccs=True, include_2d_descriptors=True
            )
            logger.info(f"Added RDKit descriptors. Shape: {chembl_data.shape}")
        
        # データセット保存
        dataset_path = self.cache_dir / "pretrained_dataset.csv"
        chembl_data.to_csv(dataset_path, index=False)
        logger.info(f"Pretrained dataset saved to: {dataset_path}")
        
        return chembl_data
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        モデルサマリー取得
        
        Returns:
            モデルサマリー辞書
        """
        summary = {
            'total_models': len(self.model_registry),
            'model_names': list(self.model_registry.keys()),
            'model_paths': self.model_paths,
            'cache_dir': str(self.cache_dir),
            'config_path': self.config_path
        }
        
        return summary
    
    def export_pretrained_models(self, output_dir: str) -> bool:
        """
        事前学習モデルエクスポート
        
        Args:
            output_dir: 出力ディレクトリ
            
        Returns:
            成功フラグ
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # モデルファイルコピー
            for model_name, model_path in self.model_paths.items():
                if model_path.exists():
                    import shutil
                    shutil.copy2(model_path, output_dir / f"{model_name}_pretrained.pth")
            
            # 設定ファイル保存
            config_path = output_dir / "pretrained_config.json"
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            # サマリーファイル保存
            summary_path = output_dir / "model_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(self.get_model_summary(), f, indent=2)
            
            logger.info(f"Pretrained models exported to: {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting pretrained models: {e}")
            return False

def create_pretrained_manager(config_path: Optional[str] = None, 
                            cache_dir: str = "cache") -> PretrainedManager:
    """
    事前学習管理器作成
    
    Args:
        config_path: 設定ファイルパス
        cache_dir: キャッシュディレクトリ
        
    Returns:
        PretrainedManager
    """
    return PretrainedManager(config_path, cache_dir)

if __name__ == "__main__":
    # テスト実行
    pretrained_manager = PretrainedManager()
    
    print(f"PretrainedManager created: {pretrained_manager}")
    print(f"Cache directory: {pretrained_manager.cache_dir}")
    print(f"Model config: {pretrained_manager.model_config}")
    print(f"Training config: {pretrained_manager.training_config}")
