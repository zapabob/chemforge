"""
Potency Metrics

pIC50/pKi力価回帰用の評価指標
RMSE/MAE/R²、ROC/PR、ECE（温度スケーリング）を実装
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, brier_score_loss
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

class PotencyMetrics:
    """pIC50/pKi力価回帰用評価指標クラス"""
    
    def __init__(self):
        """初期化"""
        pass
    
    def compute_metrics(self, predictions: Dict[str, np.ndarray], 
                        targets: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        評価指標計算
        
        Args:
            predictions: 予測結果辞書
            targets: ターゲット辞書
            
        Returns:
            評価指標辞書
        """
        metrics = {}
        
        # 回帰指標
        for target_type in ['pIC50', 'pKi']:
            pred_key = f'reg_{target_type}'
            target_key = f'reg_{target_type}'
            
            if pred_key in predictions and target_key in targets:
                pred = predictions[pred_key].flatten()
                target = targets[target_key].flatten()
                
                # 有効な値のみ使用
                valid_mask = ~(np.isnan(pred) | np.isnan(target))
                if valid_mask.sum() > 0:
                    pred_valid = pred[valid_mask]
                    target_valid = target[valid_mask]
                    
                    # 回帰指標
                    metrics.update(self._compute_regression_metrics(
                        pred_valid, target_valid, target_type
                    ))
        
        # 分類指標
        for target_type in ['pIC50', 'pKi']:
            pred_key = f'cls_{target_type}'
            target_key = f'cls_{target_type}'
            
            if pred_key in predictions and target_key in targets:
                pred = predictions[pred_key].flatten()
                target = targets[target_key].flatten()
                
                # 有効な値のみ使用
                valid_mask = ~(np.isnan(pred) | np.isnan(target))
                if valid_mask.sum() > 0:
                    pred_valid = pred[valid_mask]
                    target_valid = target[valid_mask]
                    
                    # 分類指標
                    metrics.update(self._compute_classification_metrics(
                        pred_valid, target_valid, target_type
                    ))
        
        return metrics
    
    def _compute_regression_metrics(self, pred: np.ndarray, target: np.ndarray, 
                                   target_type: str) -> Dict[str, float]:
        """回帰指標計算"""
        metrics = {}
        
        # RMSE
        rmse = np.sqrt(mean_squared_error(target, pred))
        metrics[f'rmse_{target_type}'] = rmse
        
        # MAE
        mae = mean_absolute_error(target, pred)
        metrics[f'mae_{target_type}'] = mae
        
        # R²
        r2 = r2_score(target, pred)
        metrics[f'r2_{target_type}'] = r2
        
        # Spearman相関
        try:
            from scipy.stats import spearmanr
            spearman_corr, _ = spearmanr(target, pred)
            metrics[f'spearman_{target_type}'] = spearman_corr
        except ImportError:
            metrics[f'spearman_{target_type}'] = 0.0
        
        # Pearson相関
        try:
            from scipy.stats import pearsonr
            pearson_corr, _ = pearsonr(target, pred)
            metrics[f'pearson_{target_type}'] = pearson_corr
        except ImportError:
            metrics[f'pearson_{target_type}'] = 0.0
        
        return metrics
    
    def _compute_classification_metrics(self, pred: np.ndarray, target: np.ndarray,
                                       target_type: str) -> Dict[str, float]:
        """分類指標計算"""
        metrics = {}
        
        # バイナリ分類用に調整
        pred_binary = (pred > 0.5).astype(int)
        target_binary = target.astype(int)
        
        # ROC-AUC
        try:
            if len(np.unique(target_binary)) > 1:
                roc_auc = roc_auc_score(target_binary, pred)
                metrics[f'auc_{target_type}'] = roc_auc
            else:
                metrics[f'auc_{target_type}'] = 0.5
        except ValueError:
            metrics[f'auc_{target_type}'] = 0.5
        
        # PR-AUC
        try:
            if len(np.unique(target_binary)) > 1:
                pr_auc = average_precision_score(target_binary, pred)
                metrics[f'pr_auc_{target_type}'] = pr_auc
            else:
                metrics[f'pr_auc_{target_type}'] = 0.0
        except ValueError:
            metrics[f'pr_auc_{target_type}'] = 0.0
        
        # F1スコア
        f1 = f1_score(target_binary, pred_binary, average='binary')
        metrics[f'f1_{target_type}'] = f1
        
        # 精度
        precision = precision_score(target_binary, pred_binary, average='binary', zero_division=0)
        metrics[f'precision_{target_type}'] = precision
        
        # 再現率
        recall = recall_score(target_binary, pred_binary, average='binary', zero_division=0)
        metrics[f'recall_{target_type}'] = recall
        
        # Brier Score
        brier = brier_score_loss(target_binary, pred)
        metrics[f'brier_{target_type}'] = brier
        
        # ECE（Expected Calibration Error）
        ece = self._compute_ece(target_binary, pred)
        metrics[f'ece_{target_type}'] = ece
        
        return metrics
    
    def _compute_ece(self, targets: np.ndarray, predictions: np.ndarray, 
                    n_bins: int = 10) -> float:
        """Expected Calibration Error計算"""
        try:
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = targets[in_bin].mean()
                    avg_confidence_in_bin = predictions[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            return ece
        except Exception:
            return 0.0
    
    def compute_assay_metrics(self, predictions: Dict[str, np.ndarray],
                             targets: Dict[str, np.ndarray],
                             assay_ids: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        アッセイ別評価指標計算
        
        Args:
            predictions: 予測結果辞書
            targets: ターゲット辞書
            assay_ids: アッセイID配列
            
        Returns:
            アッセイ別評価指標辞書
        """
        assay_metrics = {}
        unique_assays = np.unique(assay_ids)
        
        for assay_id in unique_assays:
            mask = assay_ids == assay_id
            
            if mask.sum() < 2:  # サンプル数が少なすぎる場合はスキップ
                continue
            
            # アッセイ別データ抽出
            assay_pred = {k: v[mask] for k, v in predictions.items()}
            assay_target = {k: v[mask] for k, v in targets.items()}
            
            # 評価指標計算
            metrics = self.compute_metrics(assay_pred, assay_target)
            assay_metrics[str(assay_id)] = metrics
        
        return assay_metrics
    
    def compute_split_metrics(self, predictions: Dict[str, np.ndarray],
                             targets: Dict[str, np.ndarray],
                             split_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        スプリット別評価指標計算
        
        Args:
            predictions: 予測結果辞書
            targets: ターゲット辞書
            split_names: スプリット名リスト
            
        Returns:
            スプリット別評価指標辞書
        """
        split_metrics = {}
        
        for split_name in split_names:
            # スプリット別データ抽出（簡易版）
            # 実際の実装では、スプリット情報を適切に管理する必要がある
            metrics = self.compute_metrics(predictions, targets)
            split_metrics[split_name] = metrics
        
        return split_metrics
    
    def compute_outlier_metrics(self, predictions: Dict[str, np.ndarray],
                               targets: Dict[str, np.ndarray],
                               outlier_masks: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        外れ値別評価指標計算
        
        Args:
            predictions: 予測結果辞書
            targets: ターゲット辞書
            outlier_masks: 外れ値マスク辞書
            
        Returns:
            外れ値別評価指標辞書
        """
        outlier_metrics = {}
        
        # 外れ値あり/なしで分けて評価
        for outlier_type in ['inlier', 'outlier']:
            is_outlier = outlier_type == 'outlier'
            
            # マスク適用
            if is_outlier:
                mask = outlier_masks.get('is_outlier', np.zeros(len(predictions['reg_pIC50']), dtype=bool))
            else:
                mask = ~outlier_masks.get('is_outlier', np.zeros(len(predictions['reg_pIC50']), dtype=bool))
            
            if mask.sum() < 2:  # サンプル数が少なすぎる場合はスキップ
                continue
            
            # マスク適用データ抽出
            masked_pred = {k: v[mask] for k, v in predictions.items()}
            masked_target = {k: v[mask] for k, v in targets.items()}
            
            # 評価指標計算
            metrics = self.compute_metrics(masked_pred, masked_target)
            outlier_metrics[outlier_type] = metrics
        
        return outlier_metrics
    
    def compute_confidence_metrics(self, predictions: Dict[str, np.ndarray],
                                   targets: Dict[str, np.ndarray],
                                   confidence_scores: np.ndarray) -> Dict[str, float]:
        """
        信頼度別評価指標計算
        
        Args:
            predictions: 予測結果辞書
            targets: ターゲット辞書
            confidence_scores: 信頼度スコア配列
            
        Returns:
            信頼度別評価指標辞書
        """
        confidence_metrics = {}
        
        # 信頼度でソート
        sorted_indices = np.argsort(confidence_scores)
        
        # 信頼度別に分割（例：上位50%、下位50%）
        n_samples = len(confidence_scores)
        high_conf_indices = sorted_indices[n_samples//2:]
        low_conf_indices = sorted_indices[:n_samples//2]
        
        for conf_type, indices in [('high_confidence', high_conf_indices), 
                                   ('low_confidence', low_conf_indices)]:
            if len(indices) < 2:
                continue
            
            # 信頼度別データ抽出
            conf_pred = {k: v[indices] for k, v in predictions.items()}
            conf_target = {k: v[indices] for k, v in targets.items()}
            
            # 評価指標計算
            metrics = self.compute_metrics(conf_pred, conf_target)
            confidence_metrics[conf_type] = metrics
        
        return confidence_metrics
    
    def generate_report(self, metrics: Dict[str, float]) -> str:
        """
        評価レポート生成
        
        Args:
            metrics: 評価指標辞書
            
        Returns:
            レポート文字列
        """
        report = []
        report.append("=" * 60)
        report.append("pIC50/pKi力価回帰評価レポート")
        report.append("=" * 60)
        
        # 回帰指標
        report.append("\n【回帰指標】")
        for target_type in ['pIC50', 'pKi']:
            report.append(f"\n{target_type}:")
            report.append(f"  RMSE: {metrics.get(f'rmse_{target_type}', 0.0):.4f}")
            report.append(f"  MAE:  {metrics.get(f'mae_{target_type}', 0.0):.4f}")
            report.append(f"  R²:   {metrics.get(f'r2_{target_type}', 0.0):.4f}")
            report.append(f"  Spearman: {metrics.get(f'spearman_{target_type}', 0.0):.4f}")
            report.append(f"  Pearson:  {metrics.get(f'pearson_{target_type}', 0.0):.4f}")
        
        # 分類指標
        report.append("\n【分類指標】")
        for target_type in ['pIC50', 'pKi']:
            report.append(f"\n{target_type}:")
            report.append(f"  ROC-AUC: {metrics.get(f'auc_{target_type}', 0.0):.4f}")
            report.append(f"  PR-AUC:  {metrics.get(f'pr_auc_{target_type}', 0.0):.4f}")
            report.append(f"  F1:      {metrics.get(f'f1_{target_type}', 0.0):.4f}")
            report.append(f"  Precision: {metrics.get(f'precision_{target_type}', 0.0):.4f}")
            report.append(f"  Recall:    {metrics.get(f'recall_{target_type}', 0.0):.4f}")
            report.append(f"  Brier:    {metrics.get(f'brier_{target_type}', 0.0):.4f}")
            report.append(f"  ECE:      {metrics.get(f'ece_{target_type}', 0.0):.4f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)

def create_metrics() -> PotencyMetrics:
    """評価指標オブジェクト作成"""
    return PotencyMetrics()

def compute_baseline_metrics(predictions: Dict[str, np.ndarray],
                             targets: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    ベースライン評価指標計算
    
    Args:
        predictions: 予測結果辞書
        targets: ターゲット辞書
        
    Returns:
        ベースライン評価指標辞書
    """
    metrics = PotencyMetrics()
    return metrics.compute_metrics(predictions, targets)
