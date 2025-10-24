"""
Metrics Module

包括的な評価指標実装
回帰・分類・マルチタスク対応
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve
)
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class Metrics:
    """
    包括的な評価指標実装
    
    回帰・分類・マルチタスク対応
    """
    
    def __init__(self):
        """評価指標を初期化"""
        self.regression_metrics = [
            'mse', 'rmse', 'mae', 'r2', 'mape', 'smape',
            'pearson', 'spearman', 'max_error', 'explained_variance'
        ]
        
        self.classification_metrics = [
            'accuracy', 'precision', 'recall', 'f1', 'f1_macro', 'f1_weighted',
            'auc', 'ap', 'log_loss', 'matthews_corrcoef'
        ]
        
        logger.info("Metrics initialized")
    
    def calculate_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        task_type: str = "regression"
    ) -> Dict[str, float]:
        """
        評価指標を計算
        
        Args:
            predictions: 予測値
            targets: 実際の値
            task_type: タスクタイプ ("regression", "classification")
            
        Returns:
            評価指標辞書
        """
        # テンソルをnumpy配列に変換
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        metrics = {}
        
        if task_type == "regression":
            metrics.update(self._calculate_regression_metrics(predictions, targets))
        elif task_type == "classification":
            metrics.update(self._calculate_classification_metrics(predictions, targets))
        else:
            # 自動判定
            if self._is_classification_task(predictions, targets):
                metrics.update(self._calculate_classification_metrics(predictions, targets))
            else:
                metrics.update(self._calculate_regression_metrics(predictions, targets))
        
        return metrics
    
    def _calculate_regression_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """
        回帰指標を計算
        
        Args:
            predictions: 予測値
            targets: 実際の値
            
        Returns:
            回帰指標辞書
        """
        metrics = {}
        
        # 基本回帰指標
        metrics['mse'] = mean_squared_error(targets, predictions)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(targets, predictions)
        metrics['r2'] = r2_score(targets, predictions)
        
        # 相関係数
        if len(predictions.shape) == 1:
            # 単一ターゲット
            metrics['pearson'] = np.corrcoef(predictions, targets)[0, 1]
            metrics['spearman'] = self._spearman_correlation(predictions, targets)
        else:
            # マルチターゲット
            metrics['pearson'] = np.mean([np.corrcoef(predictions[:, i], targets[:, i])[0, 1] 
                                        for i in range(predictions.shape[1])])
            metrics['spearman'] = np.mean([self._spearman_correlation(predictions[:, i], targets[:, i]) 
                                         for i in range(predictions.shape[1])])
        
        # その他の指標
        metrics['mape'] = self._mean_absolute_percentage_error(predictions, targets)
        metrics['smape'] = self._symmetric_mean_absolute_percentage_error(predictions, targets)
        metrics['max_error'] = np.max(np.abs(predictions - targets))
        metrics['explained_variance'] = self._explained_variance_score(predictions, targets)
        
        return metrics
    
    def _calculate_classification_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """
        分類指標を計算
        
        Args:
            predictions: 予測値
            targets: 実際の値
        
        Returns:
            分類指標辞書
        """
        metrics = {}
        
        # 予測値をクラスに変換
        if len(predictions.shape) > 1:
            pred_classes = np.argmax(predictions, axis=1)
        else:
            pred_classes = (predictions > 0.5).astype(int)
        
        # 基本分類指標
        metrics['accuracy'] = accuracy_score(targets, pred_classes)
        metrics['precision'] = precision_score(targets, pred_classes, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(targets, pred_classes, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(targets, pred_classes, average='weighted', zero_division=0)
        metrics['f1_macro'] = f1_score(targets, pred_classes, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(targets, pred_classes, average='weighted', zero_division=0)
        
        # 確率ベースの指標
        if len(predictions.shape) > 1:
            # マルチクラス分類
            try:
                metrics['auc'] = roc_auc_score(targets, predictions, multi_class='ovr', average='weighted')
            except:
                metrics['auc'] = 0.0
            
            try:
                metrics['ap'] = average_precision_score(targets, predictions, average='weighted')
            except:
                metrics['ap'] = 0.0
        else:
            # 二値分類
            try:
                metrics['auc'] = roc_auc_score(targets, predictions)
            except:
                metrics['auc'] = 0.0
            
            try:
                metrics['ap'] = average_precision_score(targets, predictions)
            except:
                metrics['ap'] = 0.0
        
        # その他の指標
        metrics['log_loss'] = self._log_loss(predictions, targets)
        metrics['matthews_corrcoef'] = self._matthews_correlation_coefficient(targets, pred_classes)
        
        return metrics
    
    def _is_classification_task(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> bool:
        """
        タスクタイプを自動判定
        
        Args:
            predictions: 予測値
            targets: 実際の値
            
        Returns:
            分類タスクかどうか
        """
        # ターゲットが離散値かどうか
        if len(targets.shape) == 1:
            unique_values = np.unique(targets)
            if len(unique_values) <= 10 and all(val == int(val) for val in unique_values):
                return True
        
        # 予測値が確率分布かどうか
        if len(predictions.shape) > 1:
            if predictions.shape[1] > 1:
                return True
        
        return False
    
    def _spearman_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> float:
        """
        Spearman相関係数を計算
        
        Args:
            x: 変数1
            y: 変数2
        
        Returns:
            Spearman相関係数
        """
        try:
            from scipy.stats import spearmanr
            return spearmanr(x, y)[0]
        except:
            # フォールバック実装
            return np.corrcoef(x, y)[0, 1]
    
    def _mean_absolute_percentage_error(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """
        平均絶対パーセント誤差を計算
        
        Args:
            predictions: 予測値
            targets: 実際の値
            
        Returns:
            MAPE
        """
        mask = targets != 0
        if np.sum(mask) == 0:
            return 0.0
        
        return np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
    
    def _symmetric_mean_absolute_percentage_error(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """
        対称平均絶対パーセント誤差を計算
        
        Args:
            predictions: 予測値
            targets: 実際の値
        
        Returns:
            SMAPE
        """
        return np.mean(2 * np.abs(predictions - targets) / (np.abs(predictions) + np.abs(targets))) * 100
    
    def _explained_variance_score(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """
        説明分散スコアを計算
        
        Args:
            predictions: 予測値
            targets: 実際の値
            
        Returns:
            説明分散スコア
        """
        var_y = np.var(targets)
        if var_y == 0:
            return 0.0
        
        return 1 - np.var(targets - predictions) / var_y
    
    def _log_loss(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """
        ログ損失を計算
        
        Args:
            predictions: 予測値
            targets: 実際の値
        
        Returns:
            ログ損失
        """
        try:
            from sklearn.metrics import log_loss
            return log_loss(targets, predictions)
        except:
            # フォールバック実装
            epsilon = 1e-15
            predictions = np.clip(predictions, epsilon, 1 - epsilon)
            return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
    
    def _matthews_correlation_coefficient(
        self,
        targets: np.ndarray,
        predictions: np.ndarray
    ) -> float:
        """
        Matthews相関係数を計算
        
        Args:
            targets: 実際の値
            predictions: 予測値
            
        Returns:
            Matthews相関係数
        """
        try:
            from sklearn.metrics import matthews_corrcoef
            return matthews_corrcoef(targets, predictions)
        except:
            # フォールバック実装
            tp = np.sum((targets == 1) & (predictions == 1))
            tn = np.sum((targets == 0) & (predictions == 0))
            fp = np.sum((targets == 0) & (predictions == 1))
            fn = np.sum((targets == 1) & (predictions == 0))
            
            if tp + tn + fp + fn == 0:
                return 0.0
            
            return (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    def calculate_multi_task_metrics(
        self,
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor],
        task_types: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        マルチタスク指標を計算
        
        Args:
            predictions: 予測値リスト
            targets: 実際の値リスト
            task_types: タスクタイプリスト
        
        Returns:
            マルチタスク指標辞書
        """
        if task_types is None:
            task_types = ['regression'] * len(predictions)
        
        multi_task_metrics = {}
        
        for i, (pred, target, task_type) in enumerate(zip(predictions, targets, task_types)):
            task_metrics = self.calculate_metrics(pred, target, task_type)
            multi_task_metrics[f'task_{i}'] = task_metrics
        
        # 全体の指標
        all_metrics = {}
        for task_metrics in multi_task_metrics.values():
            for metric_name, metric_value in task_metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(metric_value)
        
        # 平均指標
        for metric_name, values in all_metrics.items():
            multi_task_metrics[f'mean_{metric_name}'] = np.mean(values)
            multi_task_metrics[f'std_{metric_name}'] = np.std(values)
        
        return multi_task_metrics
    
    def get_confusion_matrix(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        normalize: bool = False
    ) -> np.ndarray:
        """
        混同行列を取得
        
        Args:
            predictions: 予測値
            targets: 実際の値
            normalize: 正規化するか
        
        Returns:
            混同行列
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        if len(predictions.shape) > 1:
            pred_classes = np.argmax(predictions, axis=1)
        else:
            pred_classes = (predictions > 0.5).astype(int)
        
        cm = confusion_matrix(targets, pred_classes)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        return cm
    
    def get_classification_report(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> str:
        """
        分類レポートを取得
        
        Args:
            predictions: 予測値
            targets: 実際の値
        
        Returns:
            分類レポート
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        if len(predictions.shape) > 1:
            pred_classes = np.argmax(predictions, axis=1)
        else:
            pred_classes = (predictions > 0.5).astype(int)
        
        return classification_report(targets, pred_classes)
    
    def get_available_metrics(self, task_type: str = "regression") -> List[str]:
        """
        利用可能な指標を取得
        
        Args:
            task_type: タスクタイプ
        
        Returns:
            指標リスト
        """
        if task_type == "regression":
            return self.regression_metrics
        elif task_type == "classification":
            return self.classification_metrics
        else:
            return self.regression_metrics + self.classification_metrics
    
    def get_metric_info(self, metric_name: str) -> Dict[str, Any]:
        """
        指標情報を取得
        
        Args:
            metric_name: 指標名
        
        Returns:
            指標情報
        """
        info = {
            'name': metric_name,
            'description': self._get_metric_description(metric_name),
            'range': self._get_metric_range(metric_name),
            'interpretation': self._get_metric_interpretation(metric_name)
        }
        
        return info
    
    def _get_metric_description(self, metric_name: str) -> str:
        """指標の説明を取得"""
        descriptions = {
            'mse': 'Mean Squared Error - 平均二乗誤差',
            'rmse': 'Root Mean Squared Error - 平方根平均二乗誤差',
            'mae': 'Mean Absolute Error - 平均絶対誤差',
            'r2': 'R-squared - 決定係数',
            'accuracy': 'Accuracy - 正解率',
            'precision': 'Precision - 適合率',
            'recall': 'Recall - 再現率',
            'f1': 'F1-score - F1スコア',
            'auc': 'Area Under Curve - ROC曲線下面積',
            'ap': 'Average Precision - 平均適合率'
        }
        
        return descriptions.get(metric_name, 'Unknown metric')
    
    def _get_metric_range(self, metric_name: str) -> Tuple[float, float]:
        """指標の範囲を取得"""
        ranges = {
            'mse': (0, float('inf')),
            'rmse': (0, float('inf')),
            'mae': (0, float('inf')),
            'r2': (-float('inf'), 1),
            'accuracy': (0, 1),
            'precision': (0, 1),
            'recall': (0, 1),
            'f1': (0, 1),
            'auc': (0, 1),
            'ap': (0, 1)
        }
        
        return ranges.get(metric_name, (0, 1))
    
    def _get_metric_interpretation(self, metric_name: str) -> str:
        """指標の解釈を取得"""
        interpretations = {
            'mse': '低いほど良い（0が最良）',
            'rmse': '低いほど良い（0が最良）',
            'mae': '低いほど良い（0が最良）',
            'r2': '高いほど良い（1が最良）',
            'accuracy': '高いほど良い（1が最良）',
            'precision': '高いほど良い（1が最良）',
            'recall': '高いほど良い（1が最良）',
            'f1': '高いほど良い（1が最良）',
            'auc': '高いほど良い（1が最良）',
            'ap': '高いほど良い（1が最良）'
        }
        
        return interpretations.get(metric_name, 'Unknown interpretation')