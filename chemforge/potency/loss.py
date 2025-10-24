"""
Potency Loss Functions

pIC50/pKi力価回帰用の損失関数
Huber/MSE、BCE、不確実性重み、マスク対応を実装
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, Union
import math

class PotencyLoss(nn.Module):
    """pIC50/pKi力価回帰用損失関数"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初期化
        
        Args:
            config: 損失関数設定
        """
        super().__init__()
        
        self.config = config or {}
        
        # 回帰損失設定
        self.reg_loss_type = self.config.get('reg', 'huber')  # 'huber' or 'mse'
        self.huber_delta = self.config.get('huber_delta', 1.0)
        
        # 分類損失設定
        self.cls_loss_type = self.config.get('cls', 'bce')  # 'bce' or 'focal'
        self.label_smoothing = self.config.get('label_smoothing', 0.05)
        
        # タスク重み設定
        self.lambda_reg = self.config.get('lambda_reg', 1.0)
        self.lambda_cls = self.config.get('lambda_cls', 0.3)
        self.use_uncertainty_weight = self.config.get('use_uncertainty_weight', True)
        
        # 閾値設定
        self.pIC50_threshold = self.config.get('pIC50_threshold', 6.0)  # pIC50>6.0で活性
        self.pKi_threshold = self.config.get('pKi_threshold', 7.0)  # pKi>7.0で活性
        
        # 損失関数初期化
        if self.reg_loss_type == 'huber':
            self.reg_loss_fn = nn.HuberLoss(delta=self.huber_delta)
        else:
            self.reg_loss_fn = nn.MSELoss()
        
        self.cls_loss_fn = nn.BCEWithLogitsLoss(label_smoothing=self.label_smoothing)
        
        # 不確実性重み（Kendall et al.）
        if self.use_uncertainty_weight:
            self.log_vars = nn.Parameter(torch.zeros(4))  # reg_pIC50, reg_pKi, cls_pIC50, cls_pKi
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor],
                masks: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        損失計算
        
        Args:
            predictions: 予測結果辞書
            targets: ターゲット辞書
            masks: マスク辞書（外れ値等）
            
        Returns:
            損失辞書
        """
        device = next(iter(predictions.values())).device
        
        # マスク設定（デフォルトは全て有効）
        if masks is None:
            masks = {
                'reg_pIC50': torch.ones_like(predictions['reg_pIC50'], dtype=torch.bool, device=device),
                'reg_pKi': torch.ones_like(predictions['reg_pKi'], dtype=torch.bool, device=device),
                'cls_pIC50': torch.ones_like(predictions['cls_pIC50'], dtype=torch.bool, device=device),
                'cls_pKi': torch.ones_like(predictions['cls_pKi'], dtype=torch.bool, device=device)
            }
        
        losses = {}
        
        # 回帰損失計算
        reg_pIC50_loss = self._compute_regression_loss(
            predictions['reg_pIC50'], targets['reg_pIC50'], masks['reg_pIC50']
        )
        reg_pKi_loss = self._compute_regression_loss(
            predictions['reg_pKi'], targets['reg_pKi'], masks['reg_pKi']
        )
        
        # 分類損失計算
        cls_pIC50_loss = self._compute_classification_loss(
            predictions['cls_pIC50'], targets['cls_pIC50'], masks['cls_pIC50']
        )
        cls_pKi_loss = self._compute_classification_loss(
            predictions['cls_pKi'], targets['cls_pKi'], masks['cls_pKi']
        )
        
        # 正則化損失
        reg_loss = predictions.get('regularization_loss', torch.tensor(0.0, device=device))
        
        # 不確実性重み適用
        if self.use_uncertainty_weight:
            # Kendall et al. の不確実性重み
            precision_reg_pIC50 = torch.exp(-self.log_vars[0])
            precision_reg_pKi = torch.exp(-self.log_vars[1])
            precision_cls_pIC50 = torch.exp(-self.log_vars[2])
            precision_cls_pKi = torch.exp(-self.log_vars[3])
            
            weighted_reg_pIC50_loss = precision_reg_pIC50 * reg_pIC50_loss + self.log_vars[0]
            weighted_reg_pKi_loss = precision_reg_pKi * reg_pKi_loss + self.log_vars[1]
            weighted_cls_pIC50_loss = precision_cls_pIC50 * cls_pIC50_loss + self.log_vars[2]
            weighted_cls_pKi_loss = precision_cls_pKi * cls_pKi_loss + self.log_vars[3]
        else:
            # 手動重み
            weighted_reg_pIC50_loss = self.lambda_reg * reg_pIC50_loss
            weighted_reg_pKi_loss = self.lambda_reg * reg_pKi_loss
            weighted_cls_pIC50_loss = self.lambda_cls * cls_pIC50_loss
            weighted_cls_pKi_loss = self.lambda_cls * cls_pKi_loss
        
        # 総損失
        total_loss = (weighted_reg_pIC50_loss + weighted_reg_pKi_loss + 
                     weighted_cls_pIC50_loss + weighted_cls_pKi_loss + reg_loss)
        
        # 損失辞書作成
        losses = {
            'total_loss': total_loss,
            'reg_pIC50_loss': reg_pIC50_loss,
            'reg_pKi_loss': reg_pKi_loss,
            'cls_pIC50_loss': cls_pIC50_loss,
            'cls_pKi_loss': cls_pKi_loss,
            'regularization_loss': reg_loss,
            'weighted_reg_pIC50_loss': weighted_reg_pIC50_loss,
            'weighted_reg_pKi_loss': weighted_reg_pKi_loss,
            'weighted_cls_pIC50_loss': weighted_cls_pIC50_loss,
            'weighted_cls_pKi_loss': weighted_cls_pKi_loss
        }
        
        # 不確実性重み情報追加
        if self.use_uncertainty_weight:
            losses.update({
                'log_var_reg_pIC50': self.log_vars[0],
                'log_var_reg_pKi': self.log_vars[1],
                'log_var_cls_pIC50': self.log_vars[2],
                'log_var_cls_pKi': self.log_vars[3]
            })
        
        return losses
    
    def _compute_regression_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                                mask: torch.Tensor) -> torch.Tensor:
        """回帰損失計算"""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # マスク適用
        pred_masked = pred[mask].squeeze()
        target_masked = target[mask].squeeze()
        
        if len(pred_masked) == 0:
            return torch.tensor(0.0, device=pred.device)
        
        return self.reg_loss_fn(pred_masked, target_masked)
    
    def _compute_classification_loss(self, pred: torch.Tensor, target: torch.Tensor,
                                   mask: torch.Tensor) -> torch.Tensor:
        """分類損失計算"""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # マスク適用
        pred_masked = pred[mask].squeeze()
        target_masked = target[mask].squeeze()
        
        if len(pred_masked) == 0:
            return torch.tensor(0.0, device=pred.device)
        
        return self.cls_loss_fn(pred_masked, target_masked)
    
    def create_targets_from_values(self, pIC50_values: torch.Tensor, 
                                 pKi_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        pIC50/pKi値から分類ターゲット作成
        
        Args:
            pIC50_values: pIC50値
            pKi_values: pKi値
            
        Returns:
            分類ターゲット辞書
        """
        device = pIC50_values.device
        
        # 分類ターゲット作成
        cls_pIC50_targets = (pIC50_values > self.pIC50_threshold).float()
        cls_pKi_targets = (pKi_values > self.pKi_threshold).float()
        
        return {
            'reg_pIC50': pIC50_values,
            'reg_pKi': pKi_values,
            'cls_pIC50': cls_pIC50_targets,
            'cls_pKi': cls_pKi_targets
        }

class FocalLoss(nn.Module):
    """Focal Loss実装（分類用）"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        初期化
        
        Args:
            alpha: 重み係数
            gamma: フォーカス係数
            reduction: 削減方法
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal Loss計算"""
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class UncertaintyWeightedLoss(nn.Module):
    """不確実性重み付き損失（Kendall et al.）"""
    
    def __init__(self, num_tasks: int = 4):
        """
        初期化
        
        Args:
            num_tasks: タスク数
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        """
        不確実性重み付き損失計算
        
        Args:
            losses: 各タスクの損失 [num_tasks]
            
        Returns:
            重み付き総損失
        """
        precision = torch.exp(-self.log_vars)
        weighted_losses = precision * losses + self.log_vars
        return weighted_losses.sum()

class MaskedLoss(nn.Module):
    """マスク対応損失関数"""
    
    def __init__(self, base_loss_fn: nn.Module):
        """
        初期化
        
        Args:
            base_loss_fn: ベース損失関数
        """
        super().__init__()
        self.base_loss_fn = base_loss_fn
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                mask: torch.Tensor) -> torch.Tensor:
        """
        マスク適用損失計算
        
        Args:
            pred: 予測値
            target: ターゲット
            mask: マスク
            
        Returns:
            マスク適用損失
        """
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        pred_masked = pred[mask]
        target_masked = target[mask]
        
        return self.base_loss_fn(pred_masked, target_masked)

def create_loss_function(config: Dict) -> PotencyLoss:
    """
    損失関数作成
    
    Args:
        config: 設定辞書
        
    Returns:
        PotencyLoss
    """
    return PotencyLoss(config)

def compute_loss_weights(losses: Dict[str, float], method: str = 'uncertainty') -> Dict[str, float]:
    """
    損失重み計算
    
    Args:
        losses: 損失辞書
        method: 重み計算方法
        
    Returns:
        重み辞書
    """
    if method == 'uncertainty':
        # 不確実性重み（Kendall et al.）
        weights = {}
        for key, loss in losses.items():
            if loss > 0:
                weights[key] = 1.0 / (loss + 1e-8)
            else:
                weights[key] = 1.0
        return weights
    
    elif method == 'equal':
        # 等重み
        return {key: 1.0 for key in losses.keys()}
    
    elif method == 'inverse':
        # 逆数重み
        weights = {}
        for key, loss in losses.items():
            if loss > 0:
                weights[key] = 1.0 / loss
            else:
                weights[key] = 1.0
        return weights
    
    else:
        raise ValueError(f"Unknown weight method: {method}")
