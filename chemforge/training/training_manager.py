"""
Training Manager Module

学習管理モジュール
既存Trainer・損失関数・評価指標を活用した効率的な学習管理
"""

import torch
import torch.nn as nn
import torch.optim as optim
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
from chemforge.potency.trainer import PotencyTrainer
from chemforge.potency.loss import PotencyLoss
from chemforge.potency.metrics import PotencyMetrics
from chemforge.utils.config_utils import ConfigManager
from chemforge.utils.logging_utils import Logger
from chemforge.utils.validation import DataValidator

logger = logging.getLogger(__name__)

class TrainingManager:
    """
    学習管理クラス
    
    既存Trainer・損失関数・評価指標を活用した効率的な学習管理
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
        self.logger = Logger("TrainingManager")
        self.validator = DataValidator()
        
        # 学習設定
        self.training_config = self.config.get('training', {})
        self.model_config = self.training_config.get('model', {})
        self.optimizer_config = self.training_config.get('optimizer', {})
        self.scheduler_config = self.training_config.get('scheduler', {})
        self.loss_config = self.training_config.get('loss', {})
        self.metrics_config = self.training_config.get('metrics', {})
        
        # デバイス設定
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_amp = self.training_config.get('use_amp', True)
        self.grad_clip = self.training_config.get('grad_clip', 1.0)
        
        logger.info(f"TrainingManager initialized on device: {self.device}")
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, 
                   val_loader: DataLoader, num_epochs: int = 100,
                   save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        モデル学習
        
        Args:
            model: 学習対象モデル
            train_loader: 学習データローダー
            val_loader: 検証データローダー
            num_epochs: エポック数
            save_path: 保存パス
            
        Returns:
            学習結果辞書
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        # モデルをデバイスに移動
        model = model.to(self.device)
        
        # 最適化器設定
        optimizer = self._create_optimizer(model)
        
        # スケジューラー設定
        scheduler = self._create_scheduler(optimizer)
        
        # 損失関数設定
        loss_function = self._create_loss_function()
        
        # 評価指標設定
        metrics_function = self._create_metrics_function()
        
        # AMP設定
        scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # 学習履歴
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
        # 最良モデル保存
        best_val_loss = float('inf')
        best_model_state = None
        
        # 学習ループ
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # 学習
            train_loss, train_metrics = self._train_epoch(
                model, train_loader, optimizer, loss_function, 
                metrics_function, scaler
            )
            
            # 検証
            val_loss, val_metrics = self._validate_epoch(
                model, val_loader, loss_function, metrics_function
            )
            
            # スケジューラー更新
            if scheduler:
                scheduler.step()
            
            # 履歴更新
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_metrics'].append(train_metrics)
            history['val_metrics'].append(val_metrics)
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # 最良モデル保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                logger.info(f"New best validation loss: {best_val_loss:.4f}")
            
            # ログ出力
            self._log_epoch_results(epoch, train_loss, val_loss, train_metrics, val_metrics)
        
        # 最良モデル復元
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        # モデル保存
        if save_path:
            self._save_model(model, optimizer, scheduler, history, save_path)
        
        logger.info("Training completed")
        
        return {
            'model': model,
            'history': history,
            'best_val_loss': best_val_loss,
            'final_metrics': history['val_metrics'][-1] if history['val_metrics'] else {}
        }
    
    def _create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """
        最適化器作成
        
        Args:
            model: モデル
            
        Returns:
            最適化器
        """
        optimizer_type = self.optimizer_config.get('type', 'adamw')
        lr = self.optimizer_config.get('lr', 3e-4)
        weight_decay = self.optimizer_config.get('weight_decay', 0.05)
        betas = self.optimizer_config.get('betas', (0.9, 0.95))
        
        if optimizer_type.lower() == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
                fused=True
            )
        elif optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=betas
            )
        elif optimizer_type.lower() == 'sgd':
            momentum = self.optimizer_config.get('momentum', 0.9)
            optimizer = optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        logger.info(f"Created {optimizer_type} optimizer with lr={lr}")
        return optimizer
    
    def _create_scheduler(self, optimizer: optim.Optimizer) -> Optional[optim.lr_scheduler._LRScheduler]:
        """
        スケジューラー作成
        
        Args:
            optimizer: 最適化器
            
        Returns:
            スケジューラー
        """
        scheduler_type = self.scheduler_config.get('type', 'cosine')
        warmup_steps = self.scheduler_config.get('warmup_steps', 2000)
        total_steps = self.scheduler_config.get('total_steps', 100000)
        
        if scheduler_type.lower() == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=1e-6
            )
        elif scheduler_type.lower() == 'cosine_warmup':
            # カスタムコサインウォームアップスケジューラー
            scheduler = self._create_cosine_warmup_scheduler(optimizer, warmup_steps, total_steps)
        elif scheduler_type.lower() == 'step':
            step_size = self.scheduler_config.get('step_size', 30)
            gamma = self.scheduler_config.get('gamma', 0.1)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        else:
            scheduler = None
        
        if scheduler:
            logger.info(f"Created {scheduler_type} scheduler")
        
        return scheduler
    
    def _create_cosine_warmup_scheduler(self, optimizer: optim.Optimizer, 
                                      warmup_steps: int, total_steps: int):
        """
        コサインウォームアップスケジューラー作成
        
        Args:
            optimizer: 最適化器
            warmup_steps: ウォームアップステップ数
            total_steps: 総ステップ数
            
        Returns:
            スケジューラー
        """
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def _create_loss_function(self) -> nn.Module:
        """
        損失関数作成
        
        Returns:
            損失関数
        """
        loss_type = self.loss_config.get('type', 'multi_task')
        
        if loss_type == 'multi_task':
            loss_function = PotencyLoss(
                regression_loss=self.loss_config.get('regression_loss', 'huber'),
                classification_loss=self.loss_config.get('classification_loss', 'bce'),
                uncertainty_weighting=self.loss_config.get('uncertainty_weighting', True),
                label_smoothing=self.loss_config.get('label_smoothing', 0.05)
            )
        elif loss_type == 'mse':
            loss_function = nn.MSELoss()
        elif loss_type == 'bce':
            loss_function = nn.BCELoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        logger.info(f"Created {loss_type} loss function")
        return loss_function
    
    def _create_metrics_function(self) -> PotencyMetrics:
        """
        評価指標作成
        
        Returns:
            評価指標
        """
        metrics_function = PotencyMetrics(
            regression_metrics=self.metrics_config.get('regression_metrics', ['rmse', 'mae', 'r2', 'spearman']),
            classification_metrics=self.metrics_config.get('classification_metrics', ['roc_auc', 'pr_auc', 'f1', 'brier', 'ece']),
            temperature_scaling=self.metrics_config.get('temperature_scaling', True)
        )
        
        logger.info("Created multi-task metrics function")
        return metrics_function
    
    def _train_epoch(self, model: nn.Module, train_loader: DataLoader,
                    optimizer: optim.Optimizer, loss_function: nn.Module,
                    metrics_function: PotencyMetrics, scaler: Optional[torch.cuda.amp.GradScaler]) -> Tuple[float, Dict]:
        """
        1エポック学習
        
        Args:
            model: モデル
            train_loader: 学習データローダー
            optimizer: 最適化器
            loss_function: 損失関数
            metrics_function: 評価指標
            scaler: AMPスケーラー
            
        Returns:
            学習損失・評価指標
        """
        model.train()
        total_loss = 0.0
        all_metrics = []
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            # データをデバイスに移動
            batch = self._move_batch_to_device(batch)
            
            # 勾配リセット
            optimizer.zero_grad()
            
            # フォワードパス
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(batch['input_ids'], batch.get('attention_mask'))
                    loss = loss_function(outputs, batch['targets'])
            else:
                outputs = model(batch['input_ids'], batch.get('attention_mask'))
                loss = loss_function(outputs, batch['targets'])
            
            # バックワードパス
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                optimizer.step()
            
            # 損失・評価指標記録
            total_loss += loss.item()
            
            with torch.no_grad():
                metrics = metrics_function.compute_metrics(outputs, batch['targets'])
                all_metrics.append(metrics)
        
        # 平均計算
        avg_loss = total_loss / len(train_loader)
        avg_metrics = self._average_metrics(all_metrics)
        
        return avg_loss, avg_metrics
    
    def _validate_epoch(self, model: nn.Module, val_loader: DataLoader,
                       loss_function: nn.Module, metrics_function: PotencyMetrics) -> Tuple[float, Dict]:
        """
        1エポック検証
        
        Args:
            model: モデル
            val_loader: 検証データローダー
            loss_function: 損失関数
            metrics_function: 評価指標
            
        Returns:
            検証損失・評価指標
        """
        model.eval()
        total_loss = 0.0
        all_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # データをデバイスに移動
                batch = self._move_batch_to_device(batch)
                
                # フォワードパス
                outputs = model(batch['input_ids'], batch.get('attention_mask'))
                loss = loss_function(outputs, batch['targets'])
                
                # 損失・評価指標記録
                total_loss += loss.item()
                metrics = metrics_function.compute_metrics(outputs, batch['targets'])
                all_metrics.append(metrics)
        
        # 平均計算
        avg_loss = total_loss / len(val_loader)
        avg_metrics = self._average_metrics(all_metrics)
        
        return avg_loss, avg_metrics
    
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
    
    def _average_metrics(self, metrics_list: List[Dict]) -> Dict:
        """
        評価指標平均計算
        
        Args:
            metrics_list: 評価指標リスト
            
        Returns:
            平均評価指標
        """
        if not metrics_list:
            return {}
        
        avg_metrics = {}
        for key in metrics_list[0].keys():
            values = [metrics[key] for metrics in metrics_list if key in metrics]
            if values:
                avg_metrics[key] = np.mean(values)
        
        return avg_metrics
    
    def _log_epoch_results(self, epoch: int, train_loss: float, val_loss: float,
                          train_metrics: Dict, val_metrics: Dict):
        """
        エポック結果ログ出力
        
        Args:
            epoch: エポック数
            train_loss: 学習損失
            val_loss: 検証損失
            train_metrics: 学習評価指標
            val_metrics: 検証評価指標
        """
        logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if train_metrics:
            train_metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
            logger.info(f"Train Metrics - {train_metrics_str}")
        
        if val_metrics:
            val_metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
            logger.info(f"Val Metrics - {val_metrics_str}")
    
    def _save_model(self, model: nn.Module, optimizer: optim.Optimizer,
                   scheduler: Optional[optim.lr_scheduler._LRScheduler],
                   history: Dict, save_path: str):
        """
        モデル保存
        
        Args:
            model: モデル
            optimizer: 最適化器
            scheduler: スケジューラー
            history: 学習履歴
            save_path: 保存パス
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'history': history,
            'config': self.config,
            'timestamp': time.time()
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Model saved to: {save_path}")

def create_training_manager(config_path: Optional[str] = None, 
                          cache_dir: str = "cache") -> TrainingManager:
    """
    学習管理器作成
    
    Args:
        config_path: 設定ファイルパス
        cache_dir: キャッシュディレクトリ
        
    Returns:
        TrainingManager
    """
    return TrainingManager(config_path, cache_dir)

if __name__ == "__main__":
    # テスト実行
    training_manager = TrainingManager()
    
    print(f"TrainingManager created: {training_manager}")
    print(f"Device: {training_manager.device}")
    print(f"Use AMP: {training_manager.use_amp}")
    print(f"Grad Clip: {training_manager.grad_clip}")
