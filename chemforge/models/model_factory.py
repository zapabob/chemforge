"""
Model Factory Module

モデルファクトリー・設定管理・モデル構築の統合
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import yaml
from pathlib import Path

from .transformer import MolecularTransformer
from .gnn_model import GNNRegressor, MolecularGNN, MultiScaleGNN
from .ensemble_model import EnsembleRegressor, HybridEnsemble, AdaptiveEnsemble

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    モデルファクトリー
    
    設定に基づいてモデルを構築・管理
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        モデルファクトリーを初期化
        
        Args:
            config_path: 設定ファイルパス
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        logger.info("ModelFactory initialized")
    
    def _load_config(self) -> Dict:
        """
        設定を読み込み
        
        Returns:
            設定辞書
        """
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = self._get_default_config()
        
        return config
    
    def _get_default_config(self) -> Dict:
        """
        デフォルト設定を取得
        
        Returns:
            デフォルト設定辞書
        """
        return {
            "transformer": {
                "input_dim": 2279,
                "hidden_dim": 512,
                "num_layers": 6,
                "num_heads": 8,
                "num_targets": 13,
                "use_pwa_pet": True,
                "buckets": {"trivial": 2, "fund": 4, "adj": 2},
                "use_rope": True,
                "use_pet": True,
                "pet_curv_reg": 1e-5,
                "dropout": 0.1
            },
            "gnn": {
                "input_dim": 2279,
                "hidden_dim": 512,
                "num_layers": 6,
                "num_heads": 8,
                "num_targets": 13,
                "gnn_type": "gat",
                "dropout": 0.1,
                "use_batch_norm": True,
                "use_residual": True,
                "use_attention": True,
                "use_global_pooling": "mean"
            },
            "ensemble": {
                "models": ["transformer", "gnn"],
                "ensemble_method": "weighted_average",
                "weights": [0.6, 0.4],
                "use_meta_learning": False,
                "meta_hidden_dim": 256
            }
        }
    
    def create_transformer(
        self,
        input_dim: int = 2279,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        num_targets: int = 13,
        use_pwa_pet: bool = True,
        buckets: Dict = None,
        use_rope: bool = True,
        use_pet: bool = True,
        pet_curv_reg: float = 1e-5,
        dropout: float = 0.1
    ) -> MolecularTransformer:
        """
        Transformerモデルを作成
        
        Args:
            input_dim: 入力次元
            hidden_dim: 隠れ次元
            num_layers: レイヤー数
            num_heads: アテンションヘッド数
            num_targets: ターゲット数
            use_pwa_pet: PWA+PETを使用するか
            buckets: バケット設定
            use_rope: RoPEを使用するか
            use_pet: PETを使用するか
            pet_curv_reg: PET正則化強度
            dropout: ドロップアウト率
        
        Returns:
            Transformerモデル
        """
        if buckets is None:
            buckets = {"trivial": 2, "fund": 4, "adj": 2}
        
        model = MolecularTransformer(
            input_dim=input_dim,      # 特徴量次元数（連続値）
            hidden_dim=hidden_dim,    # 隠れ層次元数
            num_layers=num_layers,    # レイヤー数
            num_heads=num_heads,      # アテンションヘッド数
            num_targets=num_targets,  # 出力次元数
            use_pwa_pet=use_pwa_pet,
            buckets=buckets,
            use_rope=use_rope,
            use_pet=use_pet,
            pet_curv_reg=pet_curv_reg,
            dropout=dropout
        )
        
        logger.info(f"Transformer model created: PWA+PET={use_pwa_pet}, {num_layers} layers")
        return model
    
    def create_gnn(
        self,
        input_dim: int = 2279,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        num_targets: int = 13,
        gnn_type: str = "gat",
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        use_residual: bool = True,
        use_attention: bool = True,
        use_global_pooling: str = "mean"
    ) -> GNNRegressor:
        """
        GNNモデルを作成
        
        Args:
            input_dim: 入力次元
            hidden_dim: 隠れ次元
            num_layers: レイヤー数
            num_heads: アテンションヘッド数
            num_targets: ターゲット数
            gnn_type: GNNタイプ
            dropout: ドロップアウト率
            use_batch_norm: バッチ正規化を使用するか
            use_residual: 残差接続を使用するか
            use_attention: アテンションを使用するか
            use_global_pooling: グローバルプーリング
        
        Returns:
            GNNモデル
        """
        model = GNNRegressor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_targets=num_targets,
            gnn_type=gnn_type,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            use_residual=use_residual,
            use_attention=use_attention,
            use_global_pooling=use_global_pooling
        )
        
        logger.info(f"GNN model created: {gnn_type}, {num_layers} layers")
        return model
    
    def create_molecular_gnn(
        self,
        atom_dim: int = 100,
        bond_dim: int = 50,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_targets: int = 13,
        gnn_type: str = "gat",
        dropout: float = 0.1
    ) -> MolecularGNN:
        """
        分子特化GNNモデルを作成
        
        Args:
            atom_dim: 原子特徴量次元
            bond_dim: 結合特徴量次元
            hidden_dim: 隠れ次元
            num_layers: レイヤー数
            num_targets: ターゲット数
            gnn_type: GNNタイプ
            dropout: ドロップアウト率
        
        Returns:
            分子特化GNNモデル
        """
        model = MolecularGNN(
            atom_dim=atom_dim,
            bond_dim=bond_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_targets=num_targets,
            gnn_type=gnn_type,
            dropout=dropout
        )
        
        logger.info(f"MolecularGNN model created: {gnn_type}, {num_layers} layers")
        return model
    
    def create_multiscale_gnn(
        self,
        input_dim: int = 2279,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_targets: int = 13,
        scales: List[int] = [1, 2, 3],
        dropout: float = 0.1
    ) -> MultiScaleGNN:
        """
        マルチスケールGNNモデルを作成
        
        Args:
            input_dim: 入力次元
            hidden_dim: 隠れ次元
            num_layers: レイヤー数
            num_targets: ターゲット数
            scales: スケールリスト
            dropout: ドロップアウト率
        
        Returns:
            マルチスケールGNNモデル
        """
        model = MultiScaleGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_targets=num_targets,
            scales=scales,
            dropout=dropout
        )
        
        logger.info(f"MultiScaleGNN model created: {len(scales)} scales, {num_layers} layers")
        return model
    
    def create_ensemble(
        self,
        models: List[nn.Module],
        ensemble_method: str = "weighted_average",
        weights: Optional[List[float]] = None,
        use_meta_learning: bool = False,
        meta_hidden_dim: int = 256
    ) -> EnsembleRegressor:
        """
        アンサンブルモデルを作成
        
        Args:
            models: モデルリスト
            ensemble_method: アンサンブル方法
            weights: モデル重み
            use_meta_learning: メタ学習を使用するか
            meta_hidden_dim: メタ学習隠れ次元
        
        Returns:
            アンサンブルモデル
        """
        model = EnsembleRegressor(
            models=models,
            ensemble_method=ensemble_method,
            weights=weights,
            use_meta_learning=use_meta_learning,
            meta_hidden_dim=meta_hidden_dim
        )
        
        logger.info(f"Ensemble model created: {len(models)} models, {ensemble_method} method")
        return model
    
    def create_hybrid_ensemble(
        self,
        deep_models: List[nn.Module],
        ml_models: List[object],
        fusion_method: str = "attention",
        hidden_dim: int = 256
    ) -> HybridEnsemble:
        """
        ハイブリッドアンサンブルモデルを作成
        
        Args:
            deep_models: 深層学習モデルリスト
            ml_models: 機械学習モデルリスト
            fusion_method: 融合方法
            hidden_dim: 隠れ次元
        
        Returns:
            ハイブリッドアンサンブルモデル
        """
        model = HybridEnsemble(
            deep_models=deep_models,
            ml_models=ml_models,
            fusion_method=fusion_method,
            hidden_dim=hidden_dim
        )
        
        logger.info(f"HybridEnsemble model created: {len(deep_models)} deep models, {len(ml_models)} ML models")
        return model
    
    def create_adaptive_ensemble(
        self,
        models: List[nn.Module],
        input_dim: int,
        hidden_dim: int = 256,
        num_targets: int = 13
    ) -> AdaptiveEnsemble:
        """
        適応的アンサンブルモデルを作成
        
        Args:
            models: モデルリスト
            input_dim: 入力次元
            hidden_dim: 隠れ次元
            num_targets: ターゲット数
        
        Returns:
            適応的アンサンブルモデル
        """
        model = AdaptiveEnsemble(
            models=models,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_targets=num_targets
        )
        
        logger.info(f"AdaptiveEnsemble model created: {len(models)} models")
        return model
    
    def create_model_from_config(
        self,
        model_type: str,
        config: Optional[Dict] = None
    ) -> nn.Module:
        """
        設定からモデルを作成
        
        Args:
            model_type: モデルタイプ
            config: 設定辞書
        
        Returns:
            モデル
        """
        if config is None:
            config = self.config.get(model_type, {})
        
        if model_type == "transformer":
            return self.create_transformer(**config)
        elif model_type == "gnn":
            return self.create_gnn(**config)
        elif model_type == "molecular_gnn":
            return self.create_molecular_gnn(**config)
        elif model_type == "multiscale_gnn":
            return self.create_multiscale_gnn(**config)
        elif model_type == "ensemble":
            # アンサンブルモデルを作成
            models = []
            for model_name in config.get("models", ["transformer", "gnn"]):
                model = self.create_model_from_config(model_name)
                models.append(model)
            
            return self.create_ensemble(
                models=models,
                ensemble_method=config.get("ensemble_method", "weighted_average"),
                weights=config.get("weights"),
                use_meta_learning=config.get("use_meta_learning", False),
                meta_hidden_dim=config.get("meta_hidden_dim", 256)
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def get_model_info(self, model: nn.Module) -> Dict:
        """
        モデル情報を取得
        
        Args:
            model: モデル
        
        Returns:
            モデル情報辞書
        """
        info = {
            "model_type": type(model).__name__,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        }
        
        return info
    
    def save_model_config(self, model: nn.Module, config_path: str):
        """
        モデル設定を保存
        
        Args:
            model: モデル
            config_path: 設定ファイルパス
        """
        config = {
            "model_type": type(model).__name__,
            "model_info": self.get_model_info(model)
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Model config saved to: {config_path}")
    
    def load_model_config(self, config_path: str) -> Dict:
        """
        モデル設定を読み込み
        
        Args:
            config_path: 設定ファイルパス
        
        Returns:
            設定辞書
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Model config loaded from: {config_path}")
        return config
