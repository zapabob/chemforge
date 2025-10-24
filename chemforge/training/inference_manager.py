"""
Inference Manager Module

推論管理モジュール
既存InferenceManagerを活用した効率的な推論管理
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import logging
from pathlib import Path
import json
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 既存モジュール活用
from chemforge.potency.metrics import MultiTaskMetrics
from chemforge.utils.config_utils import ConfigManager
from chemforge.utils.logging_utils import Logger
from chemforge.utils.validation import DataValidator

logger = logging.getLogger(__name__)

class InferenceManager:
    """
    推論管理クラス
    
    既存InferenceManagerを活用した効率的な推論管理
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
        self.logger = Logger("InferenceManager")
        self.validator = DataValidator()
        
        # 推論設定
        self.inference_config = self.config.get('inference', {})
        self.batch_size = self.inference_config.get('batch_size', 32)
        self.use_amp = self.inference_config.get('use_amp', True)
        self.num_workers = self.inference_config.get('num_workers', 4)
        
        # デバイス設定
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"InferenceManager initialized on device: {self.device}")
    
    def load_model(self, model_path: str, model_class: nn.Module, 
                  map_location: Optional[str] = None) -> nn.Module:
        """
        モデル読み込み
        
        Args:
            model_path: モデルパス
            model_class: モデルクラス
            map_location: デバイスマッピング
            
        Returns:
            読み込み済みモデル
        """
        logger.info(f"Loading model from: {model_path}")
        
        # デバイス設定
        if map_location is None:
            map_location = self.device
        
        # チェックポイント読み込み
        checkpoint = torch.load(model_path, map_location=map_location)
        
        # モデル状態辞書取得
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
        else:
            model_state_dict = checkpoint
        
        # モデル作成・状態読み込み
        model = model_class()
        model.load_state_dict(model_state_dict)
        model = model.to(self.device)
        model.eval()
        
        logger.info("Model loaded successfully")
        return model
    
    def predict(self, model: nn.Module, data_loader: DataLoader,
               return_uncertainty: bool = False, num_samples: int = 1) -> Dict[str, np.ndarray]:
        """
        推論実行
        
        Args:
            model: 推論対象モデル
            data_loader: データローダー
            return_uncertainty: 不確実性返却フラグ
            num_samples: サンプル数（不確実性計算用）
            
        Returns:
            推論結果辞書
        """
        logger.info(f"Starting inference on {len(data_loader)} batches")
        
        model.eval()
        all_predictions = []
        all_uncertainties = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Inference")):
                # データをデバイスに移動
                batch = self._move_batch_to_device(batch)
                
                if return_uncertainty and num_samples > 1:
                    # 不確実性計算（複数サンプル）
                    predictions, uncertainties = self._predict_with_uncertainty(
                        model, batch, num_samples
                    )
                else:
                    # 通常推論
                    predictions = self._predict_single(model, batch)
                    uncertainties = None
                
                all_predictions.append(predictions)
                if uncertainties is not None:
                    all_uncertainties.append(uncertainties)
        
        # 結果統合
        results = self._combine_predictions(all_predictions)
        if all_uncertainties:
            results['uncertainty'] = self._combine_predictions(all_uncertainties)
        
        logger.info(f"Inference completed: {len(results)} predictions")
        return results
    
    def _predict_single(self, model: nn.Module, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        単一推論
        
        Args:
            model: モデル
            batch: バッチデータ
            
        Returns:
            推論結果
        """
        if self.use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(batch['input_ids'], batch.get('attention_mask'))
        else:
            outputs = model(batch['input_ids'], batch.get('attention_mask'))
        
        return outputs
    
    def _predict_with_uncertainty(self, model: nn.Module, batch: Dict, 
                                 num_samples: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        不確実性付き推論
        
        Args:
            model: モデル
            batch: バッチデータ
            num_samples: サンプル数
            
        Returns:
            推論結果・不確実性
        """
        predictions_list = []
        
        for _ in range(num_samples):
            # ドロップアウト有効化（不確実性計算）
            model.train()
            pred = self._predict_single(model, batch)
            predictions_list.append(pred)
            model.eval()
        
        # 平均・分散計算
        predictions = {}
        uncertainties = {}
        
        for key in predictions_list[0].keys():
            preds = torch.stack([pred[key] for pred in predictions_list])
            predictions[key] = preds.mean(dim=0)
            uncertainties[key] = preds.std(dim=0)
        
        return predictions, uncertainties
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """
        バッチをデバイスに移動
        
        Args:
            batch: バッチデータ
            
        Returns:
            デバイス移動済みバッチ
        """
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            elif isinstance(value, dict):
                device_batch[key] = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                   for k, v in value.items()}
            else:
                device_batch[key] = value
        
        return device_batch
    
    def _combine_predictions(self, predictions_list: List[Dict[str, torch.Tensor]]) -> Dict[str, np.ndarray]:
        """
        推論結果統合
        
        Args:
            predictions_list: 推論結果リスト
            
        Returns:
            統合推論結果
        """
        if not predictions_list:
            return {}
        
        combined = {}
        for key in predictions_list[0].keys():
            tensors = [pred[key] for pred in predictions_list]
            combined[key] = torch.cat(tensors, dim=0).cpu().numpy()
        
        return combined
    
    def evaluate_model(self, model: nn.Module, data_loader: DataLoader,
                      metrics_function: Optional[MultiTaskMetrics] = None) -> Dict[str, float]:
        """
        モデル評価
        
        Args:
            model: 評価対象モデル
            data_loader: データローダー
            metrics_function: 評価指標関数
            
        Returns:
            評価結果
        """
        logger.info("Starting model evaluation")
        
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluation"):
                # データをデバイスに移動
                batch = self._move_batch_to_device(batch)
                
                # 推論
                predictions = self._predict_single(model, batch)
                all_predictions.append(predictions)
                all_targets.append(batch['targets'])
        
        # 結果統合
        predictions = self._combine_predictions(all_predictions)
        targets = self._combine_predictions(all_targets)
        
        # 評価指標計算
        if metrics_function:
            metrics = metrics_function.compute_metrics(predictions, targets)
        else:
            # 基本評価指標
            metrics = self._compute_basic_metrics(predictions, targets)
        
        logger.info(f"Evaluation completed: {metrics}")
        return metrics
    
    def _compute_basic_metrics(self, predictions: Dict[str, np.ndarray], 
                              targets: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        基本評価指標計算
        
        Args:
            predictions: 予測結果
            targets: 正解ラベル
            
        Returns:
            評価指標
        """
        metrics = {}
        
        for key in predictions.keys():
            if key in targets:
                pred = predictions[key]
                target = targets[key]
                
                if key == 'regression':
                    # 回帰指標
                    mse = np.mean((pred - target) ** 2)
                    rmse = np.sqrt(mse)
                    mae = np.mean(np.abs(pred - target))
                    r2 = 1 - np.sum((target - pred) ** 2) / np.sum((target - np.mean(target)) ** 2)
                    
                    metrics[f'{key}_mse'] = mse
                    metrics[f'{key}_rmse'] = rmse
                    metrics[f'{key}_mae'] = mae
                    metrics[f'{key}_r2'] = r2
                
                elif key == 'classification':
                    # 分類指標
                    pred_binary = (pred > 0.5).astype(int)
                    target_binary = target.astype(int)
                    
                    accuracy = np.mean(pred_binary == target_binary)
                    precision = np.sum((pred_binary == 1) & (target_binary == 1)) / (np.sum(pred_binary == 1) + 1e-8)
                    recall = np.sum((pred_binary == 1) & (target_binary == 1)) / (np.sum(target_binary == 1) + 1e-8)
                    f1 = 2 * precision * recall / (precision + recall + 1e-8)
                    
                    metrics[f'{key}_accuracy'] = accuracy
                    metrics[f'{key}_precision'] = precision
                    metrics[f'{key}_recall'] = recall
                    metrics[f'{key}_f1'] = f1
        
        return metrics
    
    def predict_single(self, model: nn.Module, input_data: Dict) -> Dict[str, float]:
        """
        単一データ推論
        
        Args:
            model: モデル
            input_data: 入力データ
            
        Returns:
            推論結果
        """
        model.eval()
        
        # データをテンソルに変換
        input_tensor = self._prepare_input_tensor(input_data)
        
        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(input_tensor['input_ids'], input_tensor.get('attention_mask'))
            else:
                outputs = model(input_tensor['input_ids'], input_tensor.get('attention_mask'))
        
        # 結果をCPUに移動・numpy変換
        results = {}
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                results[key] = value.cpu().numpy().item()
            else:
                results[key] = value
        
        return results
    
    def _prepare_input_tensor(self, input_data: Dict) -> Dict[str, torch.Tensor]:
        """
        入力データテンソル準備
        
        Args:
            input_data: 入力データ
            
        Returns:
            テンソル化済み入力データ
        """
        tensor_data = {}
        
        for key, value in input_data.items():
            if isinstance(value, (list, np.ndarray)):
                tensor_data[key] = torch.tensor(value, dtype=torch.long if key == 'input_ids' else torch.float32)
            else:
                tensor_data[key] = value
        
        return tensor_data
    
    def save_predictions(self, predictions: Dict[str, np.ndarray], 
                        output_path: str, format: str = "csv") -> bool:
        """
        推論結果保存
        
        Args:
            predictions: 推論結果
            output_path: 出力パス
            format: 出力形式
            
        Returns:
            成功フラグ
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == "csv":
                import pandas as pd
                df = pd.DataFrame(predictions)
                df.to_csv(output_path, index=False)
            elif format.lower() == "json":
                # numpy配列をリストに変換
                json_predictions = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                  for k, v in predictions.items()}
                with open(output_path, 'w') as f:
                    json.dump(json_predictions, f, indent=2)
            elif format.lower() == "npy":
                np.save(output_path, predictions)
            else:
                logger.error(f"Unsupported format: {format}")
                return False
            
            logger.info(f"Predictions saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")
            return False

def create_inference_manager(config_path: Optional[str] = None, 
                           cache_dir: str = "cache") -> InferenceManager:
    """
    推論管理器作成
    
    Args:
        config_path: 設定ファイルパス
        cache_dir: キャッシュディレクトリ
        
    Returns:
        InferenceManager
    """
    return InferenceManager(config_path, cache_dir)

if __name__ == "__main__":
    # テスト実行
    inference_manager = InferenceManager()
    
    print(f"InferenceManager created: {inference_manager}")
    print(f"Device: {inference_manager.device}")
    print(f"Use AMP: {inference_manager.use_amp}")
    print(f"Batch size: {inference_manager.batch_size}")
