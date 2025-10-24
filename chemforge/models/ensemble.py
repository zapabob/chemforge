"""
Ensemble Model Module

Ensemble深層学習モデル実装
既存EnsembleRegressorを活用した効率的なEnsemble実装
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from chemforge.core.ensemble_model import EnsembleRegressor
from chemforge.models.transformer import PotencyTransformer
from chemforge.models.gnn import PotencyGNN
from chemforge.utils.config_utils import ConfigManager
from chemforge.utils.logging_utils import Logger
from chemforge.utils.validation import DataValidator

logger = logging.getLogger(__name__)

class MolecularEnsemble(nn.Module):
    """
    Molecular Ensemble Model
    
    複数のモデルを組み合わせたEnsembleモデル
    """
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None,
                 fusion_method: str = "weighted_average", use_uncertainty: bool = True):
        """
        初期化
        
        Args:
            models: モデルリスト
            weights: 重みリスト
            fusion_method: 融合方法
            use_uncertainty: 不確実性使用フラグ
        """
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.n_models = len(models)
        self.fusion_method = fusion_method
        self.use_uncertainty = use_uncertainty
        
        # 重み設定
        if weights is None:
            self.weights = nn.Parameter(torch.ones(self.n_models) / self.n_models)
        else:
            self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32))
        
        # 不確実性重み
        if use_uncertainty:
            self.uncertainty_weights = nn.Parameter(torch.ones(self.n_models))
        
        # 融合層
        if fusion_method == "mlp":
            self.fusion_layer = nn.Sequential(
                nn.Linear(self.n_models, self.n_models * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.n_models * 2, 1)
            )
        
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        フォワードパス
        
        Args:
            *args: 入力引数
            **kwargs: 入力キーワード引数
            
        Returns:
            出力辞書
        """
        # 各モデルの予測
        predictions = []
        uncertainties = []
        
        for model in self.models:
            with torch.no_grad() if not self.training else torch.enable_grad():
                output = model(*args, **kwargs)
                
                if isinstance(output, dict):
                    pred = output.get('regression', output.get('prediction', output))
                    unc = output.get('uncertainty', torch.ones_like(pred))
                else:
                    pred = output
                    unc = torch.ones_like(pred)
                
                predictions.append(pred)
                uncertainties.append(unc)
        
        # 予測統合
        predictions = torch.stack(predictions, dim=1)  # [batch_size, n_models]
        uncertainties = torch.stack(uncertainties, dim=1)  # [batch_size, n_models]
        
        # 融合
        if self.fusion_method == "weighted_average":
            # 重み付き平均
            weights = F.softmax(self.weights, dim=0)
            if self.use_uncertainty:
                # 不確実性重み
                unc_weights = F.softmax(self.uncertainty_weights, dim=0)
                weights = weights * unc_weights
                weights = weights / weights.sum()
            
            fused_pred = torch.sum(predictions * weights.unsqueeze(0), dim=1)
            
        elif self.fusion_method == "uncertainty_weighted":
            # 不確実性重み付き平均
            weights = 1.0 / (uncertainties + 1e-8)
            weights = weights / weights.sum(dim=1, keepdim=True)
            fused_pred = torch.sum(predictions * weights, dim=1)
            
        elif self.fusion_method == "mlp":
            # MLP融合
            fused_pred = self.fusion_layer(predictions).squeeze(-1)
            
        else:
            # デフォルトは重み付き平均
            weights = F.softmax(self.weights, dim=0)
            fused_pred = torch.sum(predictions * weights.unsqueeze(0), dim=1)
        
        # 不確実性計算
        if self.use_uncertainty:
            # 予測分散
            pred_var = torch.var(predictions, dim=1)
            # 不確実性重み分散
            unc_var = torch.var(uncertainties, dim=1)
            # 統合不確実性
            fused_uncertainty = torch.sqrt(pred_var + unc_var)
        else:
            fused_uncertainty = torch.ones_like(fused_pred)
        
        return {
            'prediction': fused_pred,
            'uncertainty': fused_uncertainty,
            'individual_predictions': predictions,
            'individual_uncertainties': uncertainties
        }

class PotencyEnsemble(nn.Module):
    """
    Potency Prediction Ensemble
    
    力価予測用のEnsembleモデル
    """
    
    def __init__(self, transformer_model: PotencyTransformer, gnn_model: PotencyGNN,
                 ensemble_weights: Optional[List[float]] = None,
                 fusion_method: str = "weighted_average", use_uncertainty: bool = True):
        """
        初期化
        
        Args:
            transformer_model: Transformerモデル
            gnn_model: GNNモデル
            ensemble_weights: アンサンブル重み
            fusion_method: 融合方法
            use_uncertainty: 不確実性使用フラグ
        """
        super().__init__()
        
        self.transformer_model = transformer_model
        self.gnn_model = gnn_model
        self.fusion_method = fusion_method
        self.use_uncertainty = use_uncertainty
        
        # 重み設定
        if ensemble_weights is None:
            self.ensemble_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
        else:
            self.ensemble_weights = nn.Parameter(torch.tensor(ensemble_weights, dtype=torch.float32))
        
        # 不確実性重み
        if use_uncertainty:
            self.uncertainty_weights = nn.Parameter(torch.ones(2))
        
        # 融合層
        if fusion_method == "mlp":
            self.fusion_layer = nn.Sequential(
                nn.Linear(4, 8),  # 2 models * 2 tasks
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(8, 2)  # 2 tasks
            )
    
    def forward(self, input_ids: torch.Tensor, x: torch.Tensor, adj: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        フォワードパス
        
        Args:
            input_ids: 入力トークンID [batch_size, seq_len]
            x: ノード特徴量 [batch_size, num_nodes, node_features]
            adj: 隣接行列 [batch_size, num_nodes, num_nodes]
            attention_mask: アテンションマスク [batch_size, seq_len]
            batch: バッチインデックス
            
        Returns:
            出力辞書
        """
        # Transformer予測
        transformer_output = self.transformer_model(input_ids, attention_mask)
        
        # GNN予測
        gnn_output = self.gnn_model(x, adj, batch)
        
        # 予測統合
        regression_preds = torch.stack([
            transformer_output['regression'],
            gnn_output['regression']
        ], dim=1)  # [batch_size, 2]
        
        classification_preds = torch.stack([
            transformer_output['classification'],
            gnn_output['classification']
        ], dim=1)  # [batch_size, 2]
        
        # 融合
        if self.fusion_method == "weighted_average":
            # 重み付き平均
            weights = F.softmax(self.ensemble_weights, dim=0)
            
            fused_regression = torch.sum(regression_preds * weights.unsqueeze(0), dim=1)
            fused_classification = torch.sum(classification_preds * weights.unsqueeze(0), dim=1)
            
        elif self.fusion_method == "mlp":
            # MLP融合
            combined_input = torch.cat([regression_preds, classification_preds], dim=1)
            fused_output = self.fusion_layer(combined_input)
            fused_regression = fused_output[:, 0]
            fused_classification = torch.sigmoid(fused_output[:, 1])
            
        else:
            # デフォルトは重み付き平均
            weights = F.softmax(self.ensemble_weights, dim=0)
            fused_regression = torch.sum(regression_preds * weights.unsqueeze(0), dim=1)
            fused_classification = torch.sum(classification_preds * weights.unsqueeze(0), dim=1)
        
        # 不確実性計算
        if self.use_uncertainty:
            regression_uncertainty = torch.std(regression_preds, dim=1)
            classification_uncertainty = torch.std(classification_preds, dim=1)
        else:
            regression_uncertainty = torch.ones_like(fused_regression)
            classification_uncertainty = torch.ones_like(fused_classification)
        
        return {
            'regression': fused_regression,
            'classification': fused_classification,
            'regression_uncertainty': regression_uncertainty,
            'classification_uncertainty': classification_uncertainty,
            'transformer_regression': transformer_output['regression'],
            'transformer_classification': transformer_output['classification'],
            'gnn_regression': gnn_output['regression'],
            'gnn_classification': gnn_output['classification']
        }

class MultiTaskEnsemble(nn.Module):
    """
    Multi-Task Ensemble Model
    
    マルチタスク用のEnsembleモデル
    """
    
    def __init__(self, models: List[nn.Module], task_weights: Optional[Dict[str, float]] = None,
                 fusion_method: str = "weighted_average", use_uncertainty: bool = True):
        """
        初期化
        
        Args:
            models: モデルリスト
            task_weights: タスク重み
            fusion_method: 融合方法
            use_uncertainty: 不確実性使用フラグ
        """
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.n_models = len(models)
        self.fusion_method = fusion_method
        self.use_uncertainty = use_uncertainty
        
        # タスク重み
        if task_weights is None:
            self.task_weights = nn.Parameter(torch.ones(2))  # regression, classification
        else:
            self.task_weights = nn.Parameter(torch.tensor([
                task_weights.get('regression', 1.0),
                task_weights.get('classification', 1.0)
            ]))
        
        # モデル重み
        self.model_weights = nn.Parameter(torch.ones(self.n_models) / self.n_models)
        
        # 不確実性重み
        if use_uncertainty:
            self.uncertainty_weights = nn.Parameter(torch.ones(self.n_models))
    
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        フォワードパス
        
        Args:
            *args: 入力引数
            **kwargs: 入力キーワード引数
            
        Returns:
            出力辞書
        """
        # 各モデルの予測
        all_predictions = []
        
        for model in self.models:
            output = model(*args, **kwargs)
            all_predictions.append(output)
        
        # タスク別統合
        regression_preds = torch.stack([pred['regression'] for pred in all_predictions], dim=1)
        classification_preds = torch.stack([pred['classification'] for pred in all_predictions], dim=1)
        
        # 重み計算
        model_weights = F.softmax(self.model_weights, dim=0)
        task_weights = F.softmax(self.task_weights, dim=0)
        
        # 融合
        if self.fusion_method == "weighted_average":
            # 重み付き平均
            fused_regression = torch.sum(regression_preds * model_weights.unsqueeze(0), dim=1)
            fused_classification = torch.sum(classification_preds * model_weights.unsqueeze(0), dim=1)
            
        elif self.fusion_method == "uncertainty_weighted":
            # 不確実性重み付き平均
            if self.use_uncertainty:
                regression_uncertainties = torch.stack([pred.get('regression_uncertainty', torch.ones_like(pred['regression'])) for pred in all_predictions], dim=1)
                classification_uncertainties = torch.stack([pred.get('classification_uncertainty', torch.ones_like(pred['classification'])) for pred in all_predictions], dim=1)
                
                reg_weights = 1.0 / (regression_uncertainties + 1e-8)
                reg_weights = reg_weights / reg_weights.sum(dim=1, keepdim=True)
                
                cls_weights = 1.0 / (classification_uncertainties + 1e-8)
                cls_weights = cls_weights / cls_weights.sum(dim=1, keepdim=True)
                
                fused_regression = torch.sum(regression_preds * reg_weights, dim=1)
                fused_classification = torch.sum(classification_preds * cls_weights, dim=1)
            else:
                fused_regression = torch.sum(regression_preds * model_weights.unsqueeze(0), dim=1)
                fused_classification = torch.sum(classification_preds * model_weights.unsqueeze(0), dim=1)
        else:
            # デフォルトは重み付き平均
            fused_regression = torch.sum(regression_preds * model_weights.unsqueeze(0), dim=1)
            fused_classification = torch.sum(classification_preds * model_weights.unsqueeze(0), dim=1)
        
        # 不確実性計算
        if self.use_uncertainty:
            regression_uncertainty = torch.std(regression_preds, dim=1)
            classification_uncertainty = torch.std(classification_preds, dim=1)
        else:
            regression_uncertainty = torch.ones_like(fused_regression)
            classification_uncertainty = torch.ones_like(fused_classification)
        
        return {
            'regression': fused_regression,
            'classification': fused_classification,
            'regression_uncertainty': regression_uncertainty,
            'classification_uncertainty': classification_uncertainty,
            'individual_predictions': all_predictions
        }

def create_molecular_ensemble(models: List[nn.Module], weights: Optional[List[float]] = None,
                             fusion_method: str = "weighted_average", 
                             use_uncertainty: bool = True) -> MolecularEnsemble:
    """
    分子Ensemble作成
    
    Args:
        models: モデルリスト
        weights: 重みリスト
        fusion_method: 融合方法
        use_uncertainty: 不確実性使用フラグ
        
    Returns:
        MolecularEnsemble
    """
    return MolecularEnsemble(
        models=models,
        weights=weights,
        fusion_method=fusion_method,
        use_uncertainty=use_uncertainty
    )

def create_potency_ensemble(transformer_model: PotencyTransformer, gnn_model: PotencyGNN,
                           ensemble_weights: Optional[List[float]] = None,
                           fusion_method: str = "weighted_average",
                           use_uncertainty: bool = True) -> PotencyEnsemble:
    """
    力価予測Ensemble作成
    
    Args:
        transformer_model: Transformerモデル
        gnn_model: GNNモデル
        ensemble_weights: アンサンブル重み
        fusion_method: 融合方法
        use_uncertainty: 不確実性使用フラグ
        
    Returns:
        PotencyEnsemble
    """
    return PotencyEnsemble(
        transformer_model=transformer_model,
        gnn_model=gnn_model,
        ensemble_weights=ensemble_weights,
        fusion_method=fusion_method,
        use_uncertainty=use_uncertainty
    )

def create_multi_task_ensemble(models: List[nn.Module], 
                              task_weights: Optional[Dict[str, float]] = None,
                              fusion_method: str = "weighted_average",
                              use_uncertainty: bool = True) -> MultiTaskEnsemble:
    """
    マルチタスクEnsemble作成
    
    Args:
        models: モデルリスト
        task_weights: タスク重み
        fusion_method: 融合方法
        use_uncertainty: 不確実性使用フラグ
        
    Returns:
        MultiTaskEnsemble
    """
    return MultiTaskEnsemble(
        models=models,
        task_weights=task_weights,
        fusion_method=fusion_method,
        use_uncertainty=use_uncertainty
    )

if __name__ == "__main__":
    # テスト実行
    vocab_size = 1000
    node_features = 100
    hidden_features = 128
    
    # モデル作成
    transformer = PotencyTransformer(vocab_size=vocab_size, d_model=512, n_heads=8, n_layers=6)
    gnn = PotencyGNN(node_features=node_features, hidden_features=hidden_features, n_layers=3)
    
    # Ensemble作成
    ensemble = create_potency_ensemble(transformer, gnn)
    
    print(f"Potency Ensemble created: {ensemble}")
    print(f"Parameters: {sum(p.numel() for p in ensemble.parameters()):,}")
    
    # テスト実行
    batch_size = 2
    seq_len = 128
    num_nodes = 50
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    x = torch.randn(batch_size, num_nodes, node_features)
    adj = torch.randn(batch_size, num_nodes, num_nodes)
    adj = torch.sigmoid(adj)
    
    # Ensemble予測
    with torch.no_grad():
        outputs = ensemble(input_ids, x, adj, attention_mask)
        print(f"Ensemble outputs: {outputs.keys()}")
        print(f"Regression output shape: {outputs['regression'].shape}")
        print(f"Classification output shape: {outputs['classification'].shape}")
        print(f"Regression uncertainty shape: {outputs['regression_uncertainty'].shape}")
        print(f"Classification uncertainty shape: {outputs['classification_uncertainty'].shape}")
