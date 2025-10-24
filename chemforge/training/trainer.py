"""
Trainer Module

包括的な学習・推論システム
AMP・チェックポイント・メトリクス統合対応
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import time
import os
from pathlib import Path
import json
from tqdm import tqdm
import warnings

from .loss_functions import LossFunctions
from .metrics import Metrics
from .optimizer import OptimizerManager
from .scheduler import SchedulerManager
from .checkpoint import CheckpointManager

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class Trainer:
    """
    包括的な学習・推論システム
    
    AMP・チェックポイント・メトリクス統合対応
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        use_amp: bool = True,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        save_best: bool = True,
        patience: int = 10,
        min_delta: float = 1e-4
    ):
        """
        トレーナーを初期化
        
        Args:
            model: 学習対象モデル
            device: デバイス
            use_amp: AMPを使用するか
            checkpoint_dir: チェックポイントディレクトリ
            log_dir: ログディレクトリ
            save_best: 最良モデルを保存するか
            patience: 早期停止のパティエンス
            min_delta: 最小改善量
        """
        self.model = model
        self.device = device
        self.use_amp = use_amp
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.save_best = save_best
        self.patience = patience
        self.min_delta = min_delta
        
        # ディレクトリ作成
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 初期化
        self.model.to(device)
        self.scaler = GradScaler() if use_amp else None
        self.loss_functions = LossFunctions()
        self.metrics = Metrics()
        self.optimizer_manager = OptimizerManager()
        self.scheduler_manager = SchedulerManager()
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        
        # 学習状態
        self.current_epoch = 0
        self.best_score = float('inf')
        self.patience_counter = 0
        self.training_history = []
        
        logger.info(f"Trainer initialized: device={device}, amp={use_amp}")
    
    def setup_optimizer(
        self,
        optimizer_type: str = "adamw",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        **kwargs
    ):
        """
        オプティマイザーを設定
        
        Args:
            optimizer_type: オプティマイザータイプ
            learning_rate: 学習率
            weight_decay: 重み減衰
            **kwargs: 追加引数
        """
        self.optimizer = self.optimizer_manager.create_optimizer(
            model=self.model,
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
        
        logger.info(f"Optimizer setup: {optimizer_type}, lr={learning_rate}")
    
    def setup_scheduler(
        self,
        scheduler_type: str = "cosine",
        **kwargs
    ):
        """
        スケジューラーを設定
        
        Args:
            scheduler_type: スケジューラータイプ
            **kwargs: 追加引数
        """
        self.scheduler = self.scheduler_manager.create_scheduler(
            optimizer=self.optimizer,
            scheduler_type=scheduler_type,
            **kwargs
        )
        
        logger.info(f"Scheduler setup: {scheduler_type}")
    
    def setup_loss_function(
        self,
        loss_type: str = "mse",
        **kwargs
    ):
        """
        損失関数を設定
        
        Args:
            loss_type: 損失関数タイプ
            **kwargs: 追加引数
        """
        self.loss_function = self.loss_functions.get_loss_function(
            loss_type=loss_type,
            **kwargs
        )
        
        logger.info(f"Loss function setup: {loss_type}")
    
    def train_epoch(
        self,
        train_loader,
        loss_function: Optional[nn.Module] = None
    ) -> Dict[str, float]:
        """
        1エポックの学習を実行
        
        Args:
            train_loader: 学習データローダー
            loss_function: 損失関数
            
        Returns:
            学習メトリクス
        """
        if loss_function is None:
            loss_function = self.loss_function
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # メトリクス初期化
        epoch_metrics = {}
        
        # プログレスバー
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
            for batch_idx, batch in enumerate(pbar):
            # データをデバイスに移動
            if isinstance(batch, dict):
                inputs = batch['features'].to(self.device)
                targets = batch['targets'].to(self.device)
            else:
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                
            # 勾配をリセット
                self.optimizer.zero_grad()
                
            # フォワードパス
                if self.use_amp:
                    with autocast():
                    outputs = self.model(inputs)
                    if isinstance(outputs, tuple):
                        outputs, reg_loss = outputs
                        loss = loss_function(outputs, targets) + reg_loss
                    else:
                        loss = loss_function(outputs, targets)
            else:
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    outputs, reg_loss = outputs
                    loss = loss_function(outputs, targets) + reg_loss
                else:
                    loss = loss_function(outputs, targets)
                
            # バックワードパス
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                
            # メトリクス更新
                total_loss += loss.item()
                num_batches += 1
                
            # バッチメトリクス計算
            batch_metrics = self.metrics.calculate_metrics(outputs, targets)
            for key, value in batch_metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = 0.0
                epoch_metrics[key] += value
            
            # プログレスバー更新
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/num_batches:.4f}"
            })
        
        # エポックメトリクス計算
        epoch_metrics = {key: value/num_batches for key, value in epoch_metrics.items()}
        epoch_metrics['loss'] = total_loss / num_batches
        
        return epoch_metrics
    
    def validate_epoch(
        self,
        val_loader,
        loss_function: Optional[nn.Module] = None
    ) -> Dict[str, float]:
        """
        1エポックの検証を実行
        
        Args:
            val_loader: 検証データローダー
            loss_function: 損失関数
            
        Returns:
            検証メトリクス
        """
        if loss_function is None:
            loss_function = self.loss_function
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # メトリクス初期化
        epoch_metrics = {}
        
        with torch.no_grad():
            for batch in val_loader:
                # データをデバイスに移動
                if isinstance(batch, dict):
                    inputs = batch['features'].to(self.device)
                    targets = batch['targets'].to(self.device)
                else:
                        inputs, targets = batch
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                    
                # フォワードパス
                    if self.use_amp:
                        with autocast():
                        outputs = self.model(inputs)
                        if isinstance(outputs, tuple):
                            outputs, reg_loss = outputs
                            loss = loss_function(outputs, targets) + reg_loss
                        else:
                            loss = loss_function(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    if isinstance(outputs, tuple):
                        outputs, reg_loss = outputs
                        loss = loss_function(outputs, targets) + reg_loss
                    else:
                        loss = loss_function(outputs, targets)
                    
                # メトリクス更新
                    total_loss += loss.item()
                    num_batches += 1
                    
                # バッチメトリクス計算
                batch_metrics = self.metrics.calculate_metrics(outputs, targets)
                for key, value in batch_metrics.items():
                    if key not in epoch_metrics:
                        epoch_metrics[key] = 0.0
                    epoch_metrics[key] += value
        
        # エポックメトリクス計算
        epoch_metrics = {key: value/num_batches for key, value in epoch_metrics.items()}
        epoch_metrics['loss'] = total_loss / num_batches
        
        return epoch_metrics
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 100,
        loss_function: Optional[nn.Module] = None,
        save_frequency: int = 10
    ) -> Dict[str, List[float]]:
        """
        学習を実行
        
        Args:
            train_loader: 学習データローダー
            val_loader: 検証データローダー
            num_epochs: エポック数
            loss_function: 損失関数
            save_frequency: 保存頻度
            
        Returns:
            学習履歴
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        # 学習履歴初期化
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # 学習
            train_metrics = self.train_epoch(train_loader, loss_function)
            
            # 検証
            val_metrics = self.validate_epoch(val_loader, loss_function)
            
            # スケジューラー更新
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # 履歴更新
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['train_metrics'].append(train_metrics)
            history['val_metrics'].append(val_metrics)
            
            # ログ出力
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
            logger.info(f"  Val Loss: {val_metrics['loss']:.4f}")
            
            # 最良モデル保存
            if self.save_best:
                if val_metrics['loss'] < self.best_score - self.min_delta:
                    self.best_score = val_metrics['loss']
                    self.patience_counter = 0
                self.checkpoint_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        score=val_metrics['loss'],
                        is_best=True
                    )
                    logger.info(f"  New best model saved (score: {val_metrics['loss']:.4f})")
                else:
                    self.patience_counter += 1
            
            # 定期保存
            if (epoch + 1) % save_frequency == 0:
                self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    score=val_metrics['loss'],
                    is_best=False
                )
                logger.info(f"  Checkpoint saved at epoch {epoch+1}")
            
            # 早期停止チェック
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        # 学習履歴保存
        self.training_history = history
        self._save_training_history(history)
        
        logger.info("Training completed")
        return history
    
    def predict(
        self,
        test_loader,
        return_probabilities: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        予測を実行
        
        Args:
            test_loader: テストデータローダー
            return_probabilities: 確率を返すか
        
        Returns:
            予測結果
        """
        self.model.eval()
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                # データをデバイスに移動
                if isinstance(batch, dict):
                    inputs = batch['features'].to(self.device)
                else:
                    inputs, _ = batch
                    inputs = inputs.to(self.device)
                
                # フォワードパス
                if self.use_amp:
                    with autocast():
                        outputs = self.model(inputs)
                        if isinstance(outputs, tuple):
                            outputs, _ = outputs
                else:
                    outputs = self.model(inputs)
                    if isinstance(outputs, tuple):
                        outputs, _ = outputs
                
                # 予測結果を保存
                predictions.append(outputs.cpu().numpy())
                
                if return_probabilities:
                    # 確率計算（必要に応じて）
                    probs = torch.softmax(outputs, dim=1)
                    probabilities.append(probs.cpu().numpy())
        
        # 結果を結合
        predictions = np.concatenate(predictions, axis=0)
        
        if return_probabilities:
            probabilities = np.concatenate(probabilities, axis=0)
            return predictions, probabilities
        else:
            return predictions
    
    def evaluate(
        self,
        test_loader,
        loss_function: Optional[nn.Module] = None
    ) -> Dict[str, float]:
        """
        評価を実行
        
        Args:
            test_loader: テストデータローダー
            loss_function: 損失関数
        
        Returns:
            評価メトリクス
        """
        if loss_function is None:
            loss_function = self.loss_function
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # メトリクス初期化
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                # データをデバイスに移動
                if isinstance(batch, dict):
                    inputs = batch['features'].to(self.device)
                    targets = batch['targets'].to(self.device)
                else:
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
    
                # フォワードパス
                if self.use_amp:
                    with autocast():
                        outputs = self.model(inputs)
                        if isinstance(outputs, tuple):
                            outputs, reg_loss = outputs
                            loss = loss_function(outputs, targets) + reg_loss
                        else:
                            loss = loss_function(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    if isinstance(outputs, tuple):
                        outputs, reg_loss = outputs
                        loss = loss_function(outputs, targets) + reg_loss
                    else:
                        loss = loss_function(outputs, targets)
    
                # メトリクス更新
                total_loss += loss.item()
                num_batches += 1
                
                # 予測結果を保存
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # 全体のメトリクス計算
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        metrics = self.metrics.calculate_metrics(
            torch.tensor(all_predictions),
            torch.tensor(all_targets)
        )
        metrics['loss'] = total_loss / num_batches
        
        return metrics
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        load_optimizer: bool = True
    ):
        """
        チェックポイントを読み込み
        
        Args:
            checkpoint_path: チェックポイントパス
            load_optimizer: オプティマイザーを読み込むか
        """
        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        
        # モデル読み込み
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
        # オプティマイザー読み込み
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # エポック情報読み込み
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_score = checkpoint.get('score', float('inf'))
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def _save_training_history(self, history: Dict[str, List[float]]):
        """
        学習履歴を保存
        
        Args:
            history: 学習履歴
        """
        history_path = self.log_dir / "training_history.json"
        
        # 履歴をJSON形式で保存
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Training history saved to {history_path}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        学習要約を取得
        
        Returns:
            学習要約
        """
        summary = {
            "current_epoch": self.current_epoch,
            "best_score": self.best_score,
            "patience_counter": self.patience_counter,
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "device": self.device,
            "use_amp": self.use_amp
        }
        
        return summary