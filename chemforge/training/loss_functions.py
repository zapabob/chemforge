"""
Loss Functions Module

包括的な損失関数実装
回帰・分類・マルチタスク対応
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class LossFunctions:
    """
    包括的な損失関数実装
    
    回帰・分類・マルチタスク対応
    """
    
    def __init__(self):
        """損失関数を初期化"""
        self.loss_functions = {
            'mse': nn.MSELoss(),
            'mae': nn.L1Loss(),
            'smooth_l1': nn.SmoothL1Loss(),
            'huber': nn.HuberLoss(),
            'cross_entropy': nn.CrossEntropyLoss(),
            'binary_cross_entropy': nn.BCELoss(),
            'binary_cross_entropy_with_logits': nn.BCEWithLogitsLoss(),
            'focal_loss': self._focal_loss,
            'dice_loss': self._dice_loss,
            'focal_tversky_loss': self._focal_tversky_loss,
            'multi_task_loss': self._multi_task_loss,
            'weighted_mse': self._weighted_mse,
            'weighted_mae': self._weighted_mae,
            'quantile_loss': self._quantile_loss,
            'pinball_loss': self._pinball_loss
        }
        
        logger.info("LossFunctions initialized")
    
    def get_loss_function(
        self,
        loss_type: str = "mse",
        **kwargs
    ) -> nn.Module:
        """
        損失関数を取得
        
        Args:
            loss_type: 損失関数タイプ
            **kwargs: 追加引数
        
        Returns:
            損失関数
        """
        if loss_type in self.loss_functions:
            if loss_type in ['focal_loss', 'dice_loss', 'focal_tversky_loss', 
                           'multi_task_loss', 'weighted_mse', 'weighted_mae',
                           'quantile_loss', 'pinball_loss']:
                return self.loss_functions[loss_type](**kwargs)
            else:
                return self.loss_functions[loss_type]
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def _focal_loss(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ) -> nn.Module:
        """
        Focal Lossを実装
        
        Args:
            alpha: 重み係数
            gamma: フォーカス係数
            reduction: 削減方法
        
        Returns:
            Focal Loss関数
        """
        class FocalLoss(nn.Module):
            def __init__(self, alpha, gamma, reduction):
                super(FocalLoss, self).__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.reduction = reduction
            
            def forward(self, inputs, targets):
                ce_loss = F.cross_entropy(inputs, targets, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                
                if self.reduction == 'mean':
                    return focal_loss.mean()
                elif self.reduction == 'sum':
                    return focal_loss.sum()
                else:
                    return focal_loss
        
        return FocalLoss(alpha, gamma, reduction)
    
    def _dice_loss(
        self,
        smooth: float = 1e-5,
        reduction: str = 'mean'
    ) -> nn.Module:
        """
        Dice Lossを実装
        
        Args:
            smooth: スムージング係数
            reduction: 削減方法
        
        Returns:
            Dice Loss関数
        """
        class DiceLoss(nn.Module):
            def __init__(self, smooth, reduction):
                super(DiceLoss, self).__init__()
                self.smooth = smooth
                self.reduction = reduction
            
            def forward(self, inputs, targets):
                inputs = torch.sigmoid(inputs)
                inputs = inputs.view(-1)
                targets = targets.view(-1)
                
                intersection = (inputs * targets).sum()
                dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
                
                if self.reduction == 'mean':
                    return 1 - dice.mean()
                elif self.reduction == 'sum':
                    return 1 - dice.sum()
                else:
                    return 1 - dice
        
        return DiceLoss(smooth, reduction)
    
    def _focal_tversky_loss(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.0,
        smooth: float = 1e-5,
        reduction: str = 'mean'
    ) -> nn.Module:
        """
        Focal Tversky Lossを実装
        
        Args:
            alpha: 偽陽性重み
            beta: 偽陰性重み
            gamma: フォーカス係数
            smooth: スムージング係数
            reduction: 削減方法
        
        Returns:
            Focal Tversky Loss関数
        """
        class FocalTverskyLoss(nn.Module):
            def __init__(self, alpha, beta, gamma, smooth, reduction):
                super(FocalTverskyLoss, self).__init__()
                self.alpha = alpha
                self.beta = beta
                self.gamma = gamma
                self.smooth = smooth
                self.reduction = reduction
            
            def forward(self, inputs, targets):
                inputs = torch.sigmoid(inputs)
                inputs = inputs.view(-1)
                targets = targets.view(-1)
                
                true_pos = (inputs * targets).sum()
                false_neg = (targets * (1 - inputs)).sum()
                false_pos = ((1 - targets) * inputs).sum()
                
                tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth)
                focal_tversky = (1 - tversky) ** self.gamma
                
                if self.reduction == 'mean':
                    return focal_tversky.mean()
                elif self.reduction == 'sum':
                    return focal_tversky.sum()
                else:
                    return focal_tversky
        
        return FocalTverskyLoss(alpha, beta, gamma, smooth, reduction)
    
    def _multi_task_loss(
        self,
        task_weights: List[float] = None,
        loss_types: List[str] = None
    ) -> nn.Module:
        """
        マルチタスク損失を実装
        
        Args:
            task_weights: タスク重み
            loss_types: 損失関数タイプ
        
        Returns:
            マルチタスク損失関数
        """
        class MultiTaskLoss(nn.Module):
            def __init__(self, task_weights, loss_types):
                super(MultiTaskLoss, self).__init__()
                self.task_weights = task_weights or [1.0]
                self.loss_types = loss_types or ['mse']
                self.loss_functions = []
                
                for loss_type in self.loss_types:
                    if loss_type == 'mse':
                        self.loss_functions.append(nn.MSELoss())
                    elif loss_type == 'mae':
                        self.loss_functions.append(nn.L1Loss())
                    elif loss_type == 'cross_entropy':
                        self.loss_functions.append(nn.CrossEntropyLoss())
                    else:
                        self.loss_functions.append(nn.MSELoss())
            
            def forward(self, outputs, targets):
                if isinstance(outputs, list) and isinstance(targets, list):
                    total_loss = 0.0
                    for i, (output, target, weight, loss_fn) in enumerate(zip(outputs, targets, self.task_weights, self.loss_functions)):
                        total_loss += weight * loss_fn(output, target)
                    return total_loss
                else:
                    # 単一タスクの場合
                    return self.loss_functions[0](outputs, targets)
        
        return MultiTaskLoss(task_weights, loss_types)
    
    def _weighted_mse(
        self,
        weights: torch.Tensor = None,
        reduction: str = 'mean'
    ) -> nn.Module:
        """
        重み付きMSE損失を実装
        
        Args:
            weights: 重みテンソル
            reduction: 削減方法
        
        Returns:
            重み付きMSE損失関数
        """
        class WeightedMSELoss(nn.Module):
            def __init__(self, weights, reduction):
                super(WeightedMSELoss, self).__init__()
                self.weights = weights
                self.reduction = reduction
            
            def forward(self, inputs, targets):
                if self.weights is not None:
                    loss = F.mse_loss(inputs, targets, reduction='none')
                    weighted_loss = loss * self.weights
                    
                    if self.reduction == 'mean':
                        return weighted_loss.mean()
                    elif self.reduction == 'sum':
                        return weighted_loss.sum()
                    else:
                        return weighted_loss
                else:
                    return F.mse_loss(inputs, targets, reduction=self.reduction)
        
        return WeightedMSELoss(weights, reduction)
    
    def _weighted_mae(
        self,
        weights: torch.Tensor = None,
        reduction: str = 'mean'
    ) -> nn.Module:
        """
        重み付きMAE損失を実装
        
        Args:
            weights: 重みテンソル
            reduction: 削減方法
        
        Returns:
            重み付きMAE損失関数
        """
        class WeightedMAELoss(nn.Module):
            def __init__(self, weights, reduction):
                super(WeightedMAELoss, self).__init__()
                self.weights = weights
                self.reduction = reduction
            
            def forward(self, inputs, targets):
                if self.weights is not None:
                    loss = F.l1_loss(inputs, targets, reduction='none')
                    weighted_loss = loss * self.weights
                    
                    if self.reduction == 'mean':
                        return weighted_loss.mean()
                    elif self.reduction == 'sum':
                        return weighted_loss.sum()
                    else:
                        return weighted_loss
                else:
                    return F.l1_loss(inputs, targets, reduction=self.reduction)
        
        return WeightedMAELoss(weights, reduction)
    
    def _quantile_loss(
        self,
        quantile: float = 0.5,
        reduction: str = 'mean'
    ) -> nn.Module:
        """
        分位点損失を実装
        
        Args:
            quantile: 分位点
            reduction: 削減方法
        
        Returns:
            分位点損失関数
        """
        class QuantileLoss(nn.Module):
            def __init__(self, quantile, reduction):
                super(QuantileLoss, self).__init__()
                self.quantile = quantile
                self.reduction = reduction
            
            def forward(self, inputs, targets):
                errors = targets - inputs
                loss = torch.max((self.quantile - 1) * errors, self.quantile * errors)
                
                if self.reduction == 'mean':
                    return loss.mean()
                elif self.reduction == 'sum':
                    return loss.sum()
                else:
                    return loss
        
        return QuantileLoss(quantile, reduction)
    
    def _pinball_loss(
        self,
        quantile: float = 0.5,
        reduction: str = 'mean'
    ) -> nn.Module:
        """
        Pinball損失を実装
        
        Args:
            quantile: 分位点
            reduction: 削減方法
        
        Returns:
            Pinball損失関数
        """
        class PinballLoss(nn.Module):
            def __init__(self, quantile, reduction):
                super(PinballLoss, self).__init__()
                self.quantile = quantile
                self.reduction = reduction
            
            def forward(self, inputs, targets):
                errors = targets - inputs
                loss = torch.max((self.quantile - 1) * errors, self.quantile * errors)
                
                if self.reduction == 'mean':
                    return loss.mean()
                elif self.reduction == 'sum':
                    return loss.sum()
                else:
                    return loss
        
        return PinballLoss(quantile, reduction)
    
    def create_custom_loss(
        self,
        loss_function: callable,
        **kwargs
    ) -> nn.Module:
        """
        カスタム損失関数を作成
        
        Args:
            loss_function: 損失関数
            **kwargs: 追加引数
        
        Returns:
            カスタム損失関数
        """
        class CustomLoss(nn.Module):
            def __init__(self, loss_function, **kwargs):
                super(CustomLoss, self).__init__()
                self.loss_function = loss_function
                self.kwargs = kwargs
            
            def forward(self, inputs, targets):
                return self.loss_function(inputs, targets, **self.kwargs)
        
        return CustomLoss(loss_function, **kwargs)
    
    def get_available_losses(self) -> List[str]:
        """
        利用可能な損失関数を取得
        
        Returns:
            損失関数リスト
        """
        return list(self.loss_functions.keys())
    
    def get_loss_info(self, loss_type: str) -> Dict[str, Any]:
        """
        損失関数情報を取得
        
        Args:
            loss_type: 損失関数タイプ
        
        Returns:
            損失関数情報
        """
        info = {
            'name': loss_type,
            'description': self._get_loss_description(loss_type),
            'parameters': self._get_loss_parameters(loss_type),
            'use_cases': self._get_loss_use_cases(loss_type)
        }
        
        return info
    
    def _get_loss_description(self, loss_type: str) -> str:
        """損失関数の説明を取得"""
        descriptions = {
            'mse': 'Mean Squared Error - 回帰問題に適した損失関数',
            'mae': 'Mean Absolute Error - 外れ値に頑健な損失関数',
            'smooth_l1': 'Smooth L1 Loss - MSEとMAEの組み合わせ',
            'huber': 'Huber Loss - 外れ値に頑健な損失関数',
            'cross_entropy': 'Cross Entropy - 分類問題に適した損失関数',
            'binary_cross_entropy': 'Binary Cross Entropy - 二値分類に適した損失関数',
            'focal_loss': 'Focal Loss - 不均衡データに適した損失関数',
            'dice_loss': 'Dice Loss - セグメンテーションに適した損失関数',
            'multi_task_loss': 'Multi Task Loss - マルチタスク学習に適した損失関数'
        }
        
        return descriptions.get(loss_type, 'Unknown loss function')
    
    def _get_loss_parameters(self, loss_type: str) -> List[str]:
        """損失関数のパラメータを取得"""
        parameters = {
            'mse': ['reduction'],
            'mae': ['reduction'],
            'smooth_l1': ['beta', 'reduction'],
            'huber': ['delta', 'reduction'],
            'cross_entropy': ['weight', 'reduction', 'label_smoothing'],
            'binary_cross_entropy': ['weight', 'reduction'],
            'focal_loss': ['alpha', 'gamma', 'reduction'],
            'dice_loss': ['smooth', 'reduction'],
            'multi_task_loss': ['task_weights', 'loss_types']
        }
        
        return parameters.get(loss_type, [])
    
    def _get_loss_use_cases(self, loss_type: str) -> List[str]:
        """損失関数の使用例を取得"""
        use_cases = {
            'mse': ['回帰問題', '連続値予測'],
            'mae': ['回帰問題', '外れ値が多いデータ'],
            'smooth_l1': ['回帰問題', 'ロバストな予測'],
            'huber': ['回帰問題', '外れ値に頑健な予測'],
            'cross_entropy': ['分類問題', '多クラス分類'],
            'binary_cross_entropy': ['二値分類', '確率予測'],
            'focal_loss': ['不均衡データ', 'ハードサンプル重視'],
            'dice_loss': ['セグメンテーション', 'オーバーラップ重視'],
            'multi_task_loss': ['マルチタスク学習', '複数ターゲット予測']
        }
        
        return use_cases.get(loss_type, [])

