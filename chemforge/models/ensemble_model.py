"""
Ensemble Model Module

アンサンブル学習モデル実装
Transformer, GNN, その他のモデルの統合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class EnsembleRegressor(nn.Module):
    """
    アンサンブル回帰モデル
    
    複数のモデルを統合したアンサンブル学習
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        ensemble_method: str = "weighted_average",
        weights: Optional[List[float]] = None,
        use_meta_learning: bool = False,
        meta_hidden_dim: int = 256
    ):
        """
        アンサンブル回帰モデルを初期化
        
        Args:
            models: モデルリスト
            ensemble_method: アンサンブル方法 ("weighted_average", "stacking", "voting")
            weights: モデル重み
            use_meta_learning: メタ学習を使用するか
            meta_hidden_dim: メタ学習隠れ次元
        """
        super(EnsembleRegressor, self).__init__()
        
        self.models = nn.ModuleList(models)
        self.ensemble_method = ensemble_method
        self.use_meta_learning = use_meta_learning
        self.meta_hidden_dim = meta_hidden_dim
        
        # 重み設定
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = weights
        
        # メタ学習器
        if use_meta_learning:
            self.meta_learner = nn.Sequential(
                nn.Linear(len(models) * self.models[0].num_targets, meta_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(meta_hidden_dim, meta_hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(meta_hidden_dim // 2, self.models[0].num_targets)
            )
        
        logger.info(f"EnsembleRegressor initialized: {len(models)} models, {ensemble_method} method")
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        フォワードパス
        
        Args:
            x: 入力特徴量
            **kwargs: 追加引数
        
        Returns:
            アンサンブル予測値
        """
        # 各モデルの予測を取得
        predictions = []
        for model in self.models:
            if hasattr(model, 'forward'):
                pred = model(x, **kwargs)
                if isinstance(pred, tuple):
                    pred = pred[0]  # 正則化損失がある場合は予測値のみ
                predictions.append(pred)
            else:
                # 非PyTorchモデルの場合
                with torch.no_grad():
                    pred = model.predict(x.cpu().numpy())
                    pred = torch.tensor(pred, dtype=x.dtype, device=x.device)
                predictions.append(pred)
        
        # アンサンブル方法に応じて統合
        if self.ensemble_method == "weighted_average":
            return self._weighted_average(predictions)
        elif self.ensemble_method == "stacking":
            return self._stacking(predictions)
        elif self.ensemble_method == "voting":
            return self._voting(predictions)
        else:
            raise ValueError(f"Unsupported ensemble method: {self.ensemble_method}")
    
    def _weighted_average(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """
        重み付き平均
        
        Args:
            predictions: 予測値リスト
        
        Returns:
            重み付き平均予測値
        """
        weighted_sum = torch.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            weighted_sum += weight * pred
        
        return weighted_sum
    
    def _stacking(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """
        スタッキング
        
        Args:
            predictions: 予測値リスト
        
        Returns:
            スタッキング予測値
        """
        if self.use_meta_learning:
            # 予測値を結合
            stacked = torch.cat(predictions, dim=1)
            return self.meta_learner(stacked)
        else:
            # 単純な平均
            return torch.stack(predictions, dim=0).mean(dim=0)
    
    def _voting(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """
        投票
        
        Args:
            predictions: 予測値リスト
        
        Returns:
            投票予測値
        """
        return torch.stack(predictions, dim=0).mean(dim=0)
    
    def get_model_weights(self) -> List[float]:
        """
        モデル重みを取得
        
        Returns:
            重みリスト
        """
        return self.weights
    
    def set_model_weights(self, weights: List[float]):
        """
        モデル重みを設定
        
        Args:
            weights: 重みリスト
        """
        if len(weights) != len(self.models):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({len(self.models)})")
        
        self.weights = weights
    
    def get_model_predictions(self, x: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        """
        各モデルの予測を取得
        
        Args:
            x: 入力特徴量
            **kwargs: 追加引数
        
        Returns:
            予測値リスト
        """
        predictions = []
        for model in self.models:
            if hasattr(model, 'forward'):
                pred = model(x, **kwargs)
                if isinstance(pred, tuple):
                    pred = pred[0]
                predictions.append(pred)
            else:
                with torch.no_grad():
                    pred = model.predict(x.cpu().numpy())
                    pred = torch.tensor(pred, dtype=x.dtype, device=x.device)
                predictions.append(pred)
        
        return predictions


class HybridEnsemble(nn.Module):
    """
    ハイブリッドアンサンブルモデル
    
    深層学習モデルと機械学習モデルの統合
    """
    
    def __init__(
        self,
        deep_models: List[nn.Module],
        ml_models: List[object],
        fusion_method: str = "attention",
        hidden_dim: int = 256
    ):
        """
        ハイブリッドアンサンブルを初期化
        
        Args:
            deep_models: 深層学習モデルリスト
            ml_models: 機械学習モデルリスト
            fusion_method: 融合方法 ("attention", "concat", "weighted")
            hidden_dim: 隠れ次元
        """
        super(HybridEnsemble, self).__init__()
        
        self.deep_models = nn.ModuleList(deep_models)
        self.ml_models = ml_models
        self.fusion_method = fusion_method
        self.hidden_dim = hidden_dim
        
        # 融合層
        if fusion_method == "attention":
            self.fusion_layer = nn.MultiheadAttention(
                hidden_dim, num_heads=8, dropout=0.1, batch_first=True
            )
        elif fusion_method == "concat":
            self.fusion_layer = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2)
            )
        elif fusion_method == "weighted":
            self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # 出力層
        self.output_layer = nn.Linear(hidden_dim, deep_models[0].num_targets)
        
        logger.info(f"HybridEnsemble initialized: {len(deep_models)} deep models, {len(ml_models)} ML models")
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        フォワードパス
        
        Args:
            x: 入力特徴量
            **kwargs: 追加引数
        
        Returns:
            ハイブリッド予測値
        """
        # 深層学習モデルの予測
        deep_predictions = []
        for model in self.deep_models:
            pred = model(x, **kwargs)
            if isinstance(pred, tuple):
                pred = pred[0]
            deep_predictions.append(pred)
        
        # 機械学習モデルの予測
        ml_predictions = []
        for model in self.ml_models:
            with torch.no_grad():
                pred = model.predict(x.cpu().numpy())
                pred = torch.tensor(pred, dtype=x.dtype, device=x.device)
            ml_predictions.append(pred)
        
        # 予測を統合
        deep_avg = torch.stack(deep_predictions, dim=0).mean(dim=0)
        ml_avg = torch.stack(ml_predictions, dim=0).mean(dim=0)
        
        # 融合
        if self.fusion_method == "attention":
            # アテンション融合
            combined = torch.stack([deep_avg, ml_avg], dim=1)
            attended, _ = self.fusion_layer(combined, combined, combined)
            fused = attended.mean(dim=1)
        elif self.fusion_method == "concat":
            # 結合融合
            combined = torch.cat([deep_avg, ml_avg], dim=1)
            fused = self.fusion_layer(combined)
        elif self.fusion_method == "weighted":
            # 重み付き融合
            combined = torch.cat([deep_avg, ml_avg], dim=1)
            fused = self.fusion_layer(combined)
        
        # 出力
        return self.output_layer(fused)


class AdaptiveEnsemble(nn.Module):
    """
    適応的アンサンブルモデル
    
    入力に応じてモデル重みを動的に調整
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        input_dim: int,
        hidden_dim: int = 256,
        num_targets: int = 13
    ):
        """
        適応的アンサンブルを初期化
        
        Args:
            models: モデルリスト
            input_dim: 入力次元
            hidden_dim: 隠れ次元
            num_targets: ターゲット数
        """
        super(AdaptiveEnsemble, self).__init__()
        
        self.models = nn.ModuleList(models)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_targets = num_targets
        
        # 重み予測器
        self.weight_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, len(models))
        )
        
        # ソフトマックス層
        self.softmax = nn.Softmax(dim=1)
        
        logger.info(f"AdaptiveEnsemble initialized: {len(models)} models")
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        フォワードパス
        
        Args:
            x: 入力特徴量
            **kwargs: 追加引数
        
        Returns:
            適応的予測値
        """
        # 重みを予測
        weights = self.weight_predictor(x)
        weights = self.softmax(weights)
        
        # 各モデルの予測を取得
        predictions = []
        for model in self.models:
            pred = model(x, **kwargs)
            if isinstance(pred, tuple):
                pred = pred[0]
            predictions.append(pred)
        
        # 重み付き平均
        weighted_sum = torch.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights.T):
            weighted_sum += weight.unsqueeze(-1) * pred
        
        return weighted_sum
    
    def get_adaptive_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        適応的重みを取得
        
        Args:
            x: 入力特徴量
        
        Returns:
            重みテンソル
        """
        weights = self.weight_predictor(x)
        return self.softmax(weights)
