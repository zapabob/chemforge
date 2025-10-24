"""
Checkpoint Manager Module

包括的なチェックポイント管理システム
自動保存・復元・バックアップ対応
"""

import torch
import torch.nn as nn
import os
import shutil
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import pickle
import hashlib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    包括的なチェックポイント管理システム
    
    自動保存・復元・バックアップ対応
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        max_checkpoints: int = 10,
        save_best: bool = True,
        save_frequency: int = 10,
        backup_frequency: int = 100
    ):
        """
        チェックポイントマネージャーを初期化
        
        Args:
            checkpoint_dir: チェックポイントディレクトリ
            max_checkpoints: 最大チェックポイント数
            save_best: 最良モデルを保存するか
            save_frequency: 保存頻度
            backup_frequency: バックアップ頻度
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.save_frequency = save_frequency
        self.backup_frequency = backup_frequency
        
        # ディレクトリ作成
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # チェックポイント履歴
        self.checkpoint_history = []
        self.best_score = float('inf')
        self.best_checkpoint_path = None
        
        logger.info(f"CheckpointManager initialized: {checkpoint_dir}")
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        score: float,
        is_best: bool = False,
        additional_info: Dict[str, Any] = None
    ) -> str:
        """
        チェックポイントを保存
        
        Args:
            model: モデル
            optimizer: オプティマイザー
            epoch: エポック数
            score: スコア
            is_best: 最良モデルかどうか
            additional_info: 追加情報
        
        Returns:
            チェックポイントパス
        """
        # チェックポイント情報を準備
        checkpoint_info = {
            'epoch': epoch,
            'score': score,
            'timestamp': time.time(),
            'is_best': is_best,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'additional_info': additional_info or {}
        }
        
        # チェックポイントファイル名を生成
        checkpoint_filename = f"checkpoint_epoch_{epoch:04d}_score_{score:.4f}.pt"
        if is_best:
            checkpoint_filename = f"best_{checkpoint_filename}"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_filename
        
        # チェックポイントを保存
        torch.save(checkpoint_info, checkpoint_path)
        
        # 履歴を更新
        self.checkpoint_history.append({
            'path': str(checkpoint_path),
            'epoch': epoch,
            'score': score,
            'timestamp': time.time(),
            'is_best': is_best
        })
        
        # 最良モデルを更新
        if is_best and score < self.best_score:
            self.best_score = score
            self.best_checkpoint_path = str(checkpoint_path)
        
        # 古いチェックポイントを削除
        self._cleanup_old_checkpoints()
        
        # バックアップを作成
        if epoch % self.backup_frequency == 0:
            self._create_backup()
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module = None,
        optimizer: torch.optim.Optimizer = None
    ) -> Dict[str, Any]:
        """
        チェックポイントを読み込み
        
        Args:
            checkpoint_path: チェックポイントパス
            model: モデル
            optimizer: オプティマイザー
        
        Returns:
            チェックポイント情報
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # チェックポイントを読み込み
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # モデルを読み込み
        if model is not None:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # オプティマイザーを読み込み
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint
    
    def load_best_checkpoint(
        self,
        model: nn.Module = None,
        optimizer: torch.optim.Optimizer = None
    ) -> Dict[str, Any]:
        """
        最良チェックポイントを読み込み
        
        Args:
            model: モデル
            optimizer: オプティマイザー
        
        Returns:
            チェックポイント情報
        """
        if self.best_checkpoint_path is None:
            raise ValueError("No best checkpoint found")
        
        return self.load_checkpoint(self.best_checkpoint_path, model, optimizer)
    
    def load_latest_checkpoint(
        self,
        model: nn.Module = None,
        optimizer: torch.optim.Optimizer = None
    ) -> Dict[str, Any]:
        """
        最新チェックポイントを読み込み
        
        Args:
            model: モデル
            optimizer: オプティマイザー
        
        Returns:
            チェックポイント情報
        """
        if not self.checkpoint_history:
            raise ValueError("No checkpoints found")
        
        # 最新のチェックポイントを取得
        latest_checkpoint = max(self.checkpoint_history, key=lambda x: x['timestamp'])
        
        return self.load_checkpoint(latest_checkpoint['path'], model, optimizer)
    
    def get_checkpoint_list(self) -> List[Dict[str, Any]]:
        """
        チェックポイントリストを取得
        
        Returns:
            チェックポイントリスト
        """
        return self.checkpoint_history.copy()
    
    def get_best_checkpoint_info(self) -> Dict[str, Any]:
        """
        最良チェックポイント情報を取得
        
        Returns:
            最良チェックポイント情報
        """
        if self.best_checkpoint_path is None:
            return None
        
        return {
            'path': self.best_checkpoint_path,
            'score': self.best_score,
            'timestamp': self._get_checkpoint_timestamp(self.best_checkpoint_path)
        }
    
    def delete_checkpoint(self, checkpoint_path: str) -> bool:
        """
        チェックポイントを削除
        
        Args:
            checkpoint_path: チェックポイントパス
        
        Returns:
            削除成功かどうか
        """
        try:
            checkpoint_path = Path(checkpoint_path)
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                
                # 履歴から削除
                self.checkpoint_history = [
                    cp for cp in self.checkpoint_history 
                    if cp['path'] != str(checkpoint_path)
                ]
                
                logger.info(f"Checkpoint deleted: {checkpoint_path}")
                return True
            else:
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                return False
        except Exception as e:
            logger.error(f"Error deleting checkpoint: {e}")
            return False
    
    def _cleanup_old_checkpoints(self):
        """古いチェックポイントを削除"""
        if len(self.checkpoint_history) <= self.max_checkpoints:
            return
        
        # 最良チェックポイントを除外
        non_best_checkpoints = [
            cp for cp in self.checkpoint_history 
            if not cp['is_best']
        ]
        
        # 古いチェックポイントを削除
        if len(non_best_checkpoints) > self.max_checkpoints - 1:
            # スコアでソート（最良を除く）
            non_best_checkpoints.sort(key=lambda x: x['score'])
            
            # 古いチェックポイントを削除
            for cp in non_best_checkpoints[:len(non_best_checkpoints) - (self.max_checkpoints - 1)]:
                self.delete_checkpoint(cp['path'])
    
    def _create_backup(self):
        """バックアップを作成"""
        try:
            backup_dir = self.checkpoint_dir / "backups"
            backup_dir.mkdir(exist_ok=True)
            
            # バックアップファイル名を生成
            timestamp = int(time.time())
            backup_filename = f"checkpoint_backup_{timestamp}.tar.gz"
            backup_path = backup_dir / backup_filename
            
            # チェックポイントディレクトリを圧縮
            import tarfile
            with tarfile.open(backup_path, "w:gz") as tar:
                tar.add(self.checkpoint_dir, arcname="checkpoints")
            
            logger.info(f"Backup created: {backup_path}")
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
    
    def _get_checkpoint_timestamp(self, checkpoint_path: str) -> float:
        """チェックポイントのタイムスタンプを取得"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            return checkpoint.get('timestamp', 0.0)
        except:
            return 0.0
    
    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """
        チェックポイント要約を取得
        
        Returns:
            チェックポイント要約
        """
        summary = {
            'total_checkpoints': len(self.checkpoint_history),
            'best_score': self.best_score,
            'best_checkpoint_path': self.best_checkpoint_path,
            'checkpoint_dir': str(self.checkpoint_dir),
            'max_checkpoints': self.max_checkpoints,
            'save_best': self.save_best,
            'save_frequency': self.save_frequency,
            'backup_frequency': self.backup_frequency
        }
        
        return summary
    
    def export_checkpoint(
        self,
        checkpoint_path: str,
        export_path: str,
        include_optimizer: bool = True,
        include_additional_info: bool = True
    ) -> str:
        """
        チェックポイントをエクスポート
        
        Args:
            checkpoint_path: チェックポイントパス
            export_path: エクスポートパス
            include_optimizer: オプティマイザーを含むか
            include_additional_info: 追加情報を含むか
        
        Returns:
            エクスポートパス
        """
        # チェックポイントを読み込み
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # エクスポート用のチェックポイントを作成
        export_checkpoint = {
            'epoch': checkpoint['epoch'],
            'score': checkpoint['score'],
            'timestamp': checkpoint['timestamp'],
            'is_best': checkpoint['is_best'],
            'model_state_dict': checkpoint['model_state_dict']
        }
        
        if include_optimizer:
            export_checkpoint['optimizer_state_dict'] = checkpoint['optimizer_state_dict']
        
        if include_additional_info:
            export_checkpoint['additional_info'] = checkpoint.get('additional_info', {})
        
        # エクスポート
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(export_checkpoint, export_path)
        
        logger.info(f"Checkpoint exported: {export_path}")
        return str(export_path)
    
    def import_checkpoint(
        self,
        import_path: str,
        model: nn.Module = None,
        optimizer: torch.optim.Optimizer = None
    ) -> Dict[str, Any]:
        """
        チェックポイントをインポート
        
        Args:
            import_path: インポートパス
            model: モデル
            optimizer: オプティマイザー
        
        Returns:
            チェックポイント情報
        """
        return self.load_checkpoint(import_path, model, optimizer)
    
    def get_checkpoint_metadata(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        チェックポイントメタデータを取得
        
        Args:
            checkpoint_path: チェックポイントパス
        
        Returns:
            メタデータ
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            metadata = {
                'epoch': checkpoint.get('epoch', 0),
                'score': checkpoint.get('score', 0.0),
                'timestamp': checkpoint.get('timestamp', 0.0),
                'is_best': checkpoint.get('is_best', False),
                'file_size': Path(checkpoint_path).stat().st_size,
                'additional_info': checkpoint.get('additional_info', {})
            }
            
            return metadata
        except Exception as e:
            logger.error(f"Error getting checkpoint metadata: {e}")
            return {}
    
    def validate_checkpoint(self, checkpoint_path: str) -> bool:
        """
        チェックポイントを検証
        
        Args:
            checkpoint_path: チェックポイントパス
        
        Returns:
            有効かどうか
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 必要なキーをチェック
            required_keys = ['epoch', 'score', 'model_state_dict']
            for key in required_keys:
                if key not in checkpoint:
                    return False
            
            # モデル状態辞書をチェック
            if not isinstance(checkpoint['model_state_dict'], dict):
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating checkpoint: {e}")
            return False
