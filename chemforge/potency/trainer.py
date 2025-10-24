"""
Potency Trainer

pIC50/pKi力価回帰専用のトレーナー
AMP、AdamW、scheduler、early stoppingを実装
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import time
import os
from tqdm import tqdm
import json
import math

from .potency_model import PotencyPWAPETModel
from .loss import PotencyLoss
from .metrics import PotencyMetrics

class PotencyTrainer:
    """pIC50/pKi力価回帰専用トレーナー"""
    
    def __init__(self, model: PotencyPWAPETModel, config: Dict):
        """
        初期化
        
        Args:
            model: 学習するモデル
            config: 学習設定
        """
        self.model = model
        self.config = config
        
        # デバイス設定
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)
        
        # 学習設定
        self.epochs = config.get('epochs', 50)
        self.batch_size = config.get('batch_size', 128)
        self.grad_accum_steps = config.get('grad_accum_steps', 1)
        self.lr = config.get('lr', 3e-4)
        self.weight_decay = config.get('weight_decay', 0.05)
        self.grad_clip = config.get('grad_clip', 1.0)
        
        # AMP設定
        self.amp = config.get('amp', True)
        self.scaler = GradScaler() if self.amp else None
        
        # オプティマイザー設定
        self.optimizer = self._create_optimizer()
        
        # スケジューラー設定
        self.scheduler = self._create_scheduler()
        
        # 損失関数
        self.criterion = PotencyLoss(config.get('loss', {}))
        
        # 評価指標
        self.metrics = PotencyMetrics()
        
        # Early stopping設定
        self.early_stopping = config.get('early_stopping', True)
        self.patience = config.get('patience', 10)
        self.min_delta = config.get('min_delta', 1e-4)
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        
        # チェックポイント設定
        self.checkpointing = config.get('checkpointing', True)
        self.save_dir = config.get('save_dir', './checkpoints')
        self.save_every = config.get('save_every', 5)
        
        # ログ設定
        self.log_every = config.get('log_every', 50)
        self.training_history = []
        
        # 高速化設定
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    def _create_optimizer(self) -> optim.Optimizer:
        """オプティマイザー作成"""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.95),
            weight_decay=self.weight_decay,
            fused=True if torch.cuda.is_available() else False
        )
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """スケジューラー作成"""
        warmup_steps = self.config.get('warmup_steps', 2000)
        total_steps = self.epochs * self.config.get('steps_per_epoch', 1000)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        1エポック学習
        
        Args:
            train_loader: 学習データローダー
            epoch: エポック数
            
        Returns:
            学習メトリクス
        """
        self.model.train()
        
        total_loss = 0.0
        total_reg_pIC50_loss = 0.0
        total_reg_pKi_loss = 0.0
        total_cls_pIC50_loss = 0.0
        total_cls_pKi_loss = 0.0
        total_reg_loss = 0.0
        
        num_batches = len(train_loader)
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # データをデバイスに移動
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # ターゲット作成
            targets = self._create_targets(batch)
            
            # AMP使用
            if self.amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    predictions = self.model(batch)
                    losses = self.criterion(predictions, targets, batch.get('mask'))
            else:
                predictions = self.model(batch)
                losses = self.criterion(predictions, targets, batch.get('mask'))
            
            # 損失正規化
            loss = losses['total_loss'] / self.grad_accum_steps
            
            # 勾配計算
            if self.amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 勾配蓄積
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                # 勾配クリッピング
                if self.amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.scheduler.step()
            
            # 損失累積
            total_loss += losses['total_loss'].item()
            total_reg_pIC50_loss += losses['reg_pIC50_loss'].item()
            total_reg_pKi_loss += losses['reg_pKi_loss'].item()
            total_cls_pIC50_loss += losses['cls_pIC50_loss'].item()
            total_cls_pKi_loss += losses['cls_pKi_loss'].item()
            total_reg_loss += losses['regularization_loss'].item()
            
            # プログレスバー更新
            progress_bar.set_postfix({
                'Loss': f"{losses['total_loss'].item():.4f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # ログ出力
            if batch_idx % self.log_every == 0:
                print(f"[Epoch {epoch+1}, Batch {batch_idx}] "
                      f"Loss: {losses['total_loss'].item():.4f}, "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        
        # 平均損失計算
        avg_losses = {
            'total_loss': total_loss / num_batches,
            'reg_pIC50_loss': total_reg_pIC50_loss / num_batches,
            'reg_pKi_loss': total_reg_pKi_loss / num_batches,
            'cls_pIC50_loss': total_cls_pIC50_loss / num_batches,
            'cls_pKi_loss': total_cls_pKi_loss / num_batches,
            'regularization_loss': total_reg_loss / num_batches
        }
        
        return avg_losses
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        検証
        
        Args:
            val_loader: 検証データローダー
            
        Returns:
            検証メトリクス
        """
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = {'reg_pIC50': [], 'reg_pKi': [], 'cls_pIC50': [], 'cls_pKi': []}
        all_targets = {'reg_pIC50': [], 'reg_pKi': [], 'cls_pIC50': [], 'cls_pKi': []}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # データをデバイスに移動
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # ターゲット作成
                targets = self._create_targets(batch)
                
                # 予測
                if self.amp:
                    with autocast(device_type='cuda', dtype=torch.float16):
                        predictions = self.model(batch)
                        losses = self.criterion(predictions, targets, batch.get('mask'))
                else:
                    predictions = self.model(batch)
                    losses = self.criterion(predictions, targets, batch.get('mask'))
                
                total_loss += losses['total_loss'].item()
                
                # 予測結果とターゲットを保存
                for key in ['reg_pIC50', 'reg_pKi', 'cls_pIC50', 'cls_pKi']:
                    all_predictions[key].append(predictions[key].cpu().numpy())
                    all_targets[key].append(targets[key].cpu().numpy())
        
        # 配列結合
        for key in all_predictions:
            all_predictions[key] = np.concatenate(all_predictions[key])
            all_targets[key] = np.concatenate(all_targets[key])
        
        # メトリクス計算
        metrics = self.metrics.compute_metrics(all_predictions, all_targets)
        metrics['val_loss'] = total_loss / len(val_loader)
        
        return metrics
    
    def _create_targets(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """バッチからターゲット作成"""
        targets = {}
        
        # 回帰ターゲット（実際のpIC50/pKi値）
        if 'target' in batch:
            # 単一ターゲットの場合
            targets['reg_pIC50'] = batch['target']
            targets['reg_pKi'] = batch['target']
        else:
            # マルチターゲットの場合
            targets['reg_pIC50'] = batch.get('pIC50_target', torch.zeros_like(batch['tokens'][:, 0:1]))
            targets['reg_pKi'] = batch.get('pKi_target', torch.zeros_like(batch['tokens'][:, 0:1]))
        
        # 分類ターゲット（閾値ベース）
        pIC50_threshold = self.config.get('pIC50_threshold', 6.0)
        pKi_threshold = self.config.get('pKi_threshold', 7.0)
        
        targets['cls_pIC50'] = (targets['reg_pIC50'] > pIC50_threshold).float()
        targets['cls_pKi'] = (targets['reg_pKi'] > pKi_threshold).float()
        
        return targets
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """
        学習実行
        
        Args:
            train_loader: 学習データローダー
            val_loader: 検証データローダー
            
        Returns:
            学習履歴
        """
        print("=" * 60)
        print("pIC50/pKi力価回帰学習開始")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Model size: {self.model.get_model_size()}")
        print(f"AMP: {self.amp}")
        print(f"Epochs: {self.epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.lr}")
        print("=" * 60)
        
        # 学習履歴
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_rmse_pIC50': [],
            'val_rmse_pKi': [],
            'val_r2_pIC50': [],
            'val_r2_pKi': [],
            'val_auc_pIC50': [],
            'val_auc_pKi': []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            start_time = time.time()
            
            # 学習
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # 検証
            val_metrics = self.validate(val_loader)
            
            # 履歴更新
            history['train_loss'].append(train_metrics['total_loss'])
            history['val_loss'].append(val_metrics['val_loss'])
            history['val_rmse_pIC50'].append(val_metrics.get('rmse_pIC50', 0.0))
            history['val_rmse_pKi'].append(val_metrics.get('rmse_pKi', 0.0))
            history['val_r2_pIC50'].append(val_metrics.get('r2_pIC50', 0.0))
            history['val_r2_pKi'].append(val_metrics.get('r2_pKi', 0.0))
            history['val_auc_pIC50'].append(val_metrics.get('auc_pIC50', 0.0))
            history['val_auc_pKi'].append(val_metrics.get('auc_pKi', 0.0))
            
            # ログ出力
            epoch_time = time.time() - start_time
            print(f"\n[Epoch {epoch+1}/{self.epochs}] "
                  f"Train Loss: {train_metrics['total_loss']:.4f}, "
                  f"Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"Time: {epoch_time:.2f}s")
            
            # ベストモデル保存
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                self.best_val_loss = best_val_loss
                self.early_stop_counter = 0
                
                if self.checkpointing:
                    self._save_checkpoint(epoch, val_metrics['val_loss'], is_best=True)
            else:
                self.early_stop_counter += 1
            
            # 定期保存
            if self.checkpointing and (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(epoch, val_metrics['val_loss'], is_best=False)
            
            # Early stopping
            if self.early_stopping and self.early_stop_counter >= self.patience:
                print(f"\n[Early Stopping] No improvement for {self.patience} epochs")
                break
        
        print("=" * 60)
        print("学習完了")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print("=" * 60)
        
        return history
    
    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """チェックポイント保存"""
        os.makedirs(self.save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        if is_best:
            save_path = os.path.join(self.save_dir, 'best_model.pt')
        else:
            save_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pt')
        
        torch.save(checkpoint, save_path)
        print(f"[INFO] Checkpoint saved: {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """チェックポイント読み込み"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"[INFO] Checkpoint loaded: {checkpoint_path}")
        return checkpoint['epoch'], checkpoint['val_loss']
