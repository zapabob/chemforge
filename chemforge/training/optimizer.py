"""
Optimizer Manager Module

包括的なオプティマイザー管理システム
Adam, AdamW, SGD, RMSprop, Adagrad対応
"""

import torch
import torch.optim as optim
from torch.optim import Adam, AdamW, SGD, RMSprop, Adagrad, Adadelta, Adamax
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class OptimizerManager:
    """
    包括的なオプティマイザー管理システム
    
    Adam, AdamW, SGD, RMSprop, Adagrad対応
    """
    
    def __init__(self):
        """オプティマイザーマネージャーを初期化"""
        self.optimizers = {
            'adam': Adam,
            'adamw': AdamW,
            'sgd': SGD,
            'rmsprop': RMSprop,
            'adagrad': Adagrad,
            'adadelta': Adadelta,
            'adamax': Adamax
        }
        
        # デフォルトパラメータ
        self.default_params = {
            'adam': {'lr': 1e-3, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0},
            'adamw': {'lr': 1e-3, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 1e-2},
            'sgd': {'lr': 1e-2, 'momentum': 0.9, 'weight_decay': 0, 'nesterov': False},
            'rmsprop': {'lr': 1e-2, 'alpha': 0.99, 'eps': 1e-8, 'weight_decay': 0, 'momentum': 0},
            'adagrad': {'lr': 1e-2, 'lr_decay': 0, 'weight_decay': 0, 'eps': 1e-10},
            'adadelta': {'lr': 1.0, 'rho': 0.9, 'eps': 1e-6, 'weight_decay': 0},
            'adamax': {'lr': 2e-3, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0}
        }
        
        logger.info("OptimizerManager initialized")
    
    def create_optimizer(
        self,
        model: torch.nn.Module,
        optimizer_type: str = "adamw",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        **kwargs
    ) -> torch.optim.Optimizer:
        """
        オプティマイザーを作成
        
        Args:
            model: モデル
            optimizer_type: オプティマイザータイプ
            learning_rate: 学習率
            weight_decay: 重み減衰
            **kwargs: 追加引数
        
        Returns:
            オプティマイザー
        """
        if optimizer_type not in self.optimizers:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        # パラメータを準備
        params = self._prepare_optimizer_params(
            model, optimizer_type, learning_rate, weight_decay, **kwargs
        )
        
        # オプティマイザーを作成
        optimizer_class = self.optimizers[optimizer_type]
        optimizer = optimizer_class(model.parameters(), **params)
        
        logger.info(f"Optimizer created: {optimizer_type}, lr={learning_rate}, wd={weight_decay}")
        return optimizer
    
    def _prepare_optimizer_params(
        self,
        model: torch.nn.Module,
        optimizer_type: str,
        learning_rate: float,
        weight_decay: float,
        **kwargs
    ) -> Dict[str, Any]:
        """
        オプティマイザーパラメータを準備
        
        Args:
            model: モデル
            optimizer_type: オプティマイザータイプ
            learning_rate: 学習率
            weight_decay: 重み減衰
            **kwargs: 追加引数
        
        Returns:
            オプティマイザーパラメータ
        """
        # デフォルトパラメータを取得
        params = self.default_params[optimizer_type].copy()
        
        # 基本パラメータを設定
        params['lr'] = learning_rate
        params['weight_decay'] = weight_decay
        
        # 追加引数を適用
        params.update(kwargs)
        
        # オプティマイザー固有の調整
        if optimizer_type == 'adamw':
            # AdamWの重み減衰は別途設定
            params['weight_decay'] = weight_decay
        elif optimizer_type == 'sgd':
            # SGDのモメンタム設定
            if 'momentum' not in params:
                params['momentum'] = 0.9
        elif optimizer_type == 'rmsprop':
            # RMSpropのアルファ設定
            if 'alpha' not in params:
                params['alpha'] = 0.99
        
        return params
    
    def create_optimizer_with_different_lr(
        self,
        model: torch.nn.Module,
        optimizer_type: str = "adamw",
        base_lr: float = 1e-4,
        weight_decay: float = 1e-5,
        lr_multipliers: Dict[str, float] = None,
        **kwargs
    ) -> torch.optim.Optimizer:
        """
        異なる学習率でオプティマイザーを作成
        
        Args:
            model: モデル
            optimizer_type: オプティマイザータイプ
            base_lr: ベース学習率
            weight_decay: 重み減衰
            lr_multipliers: 学習率乗数
            **kwargs: 追加引数
        
        Returns:
            オプティマイザー
        """
        if lr_multipliers is None:
            lr_multipliers = {}
        
        # パラメータグループを作成
        param_groups = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # 学習率乗数を取得
            lr_mult = lr_multipliers.get(name, 1.0)
            lr = base_lr * lr_mult
            
            param_groups.append({
                'params': param,
                'lr': lr,
                'weight_decay': weight_decay
            })
        
        # オプティマイザーを作成
        optimizer_class = self.optimizers[optimizer_type]
        optimizer = optimizer_class(param_groups, **kwargs)
        
        logger.info(f"Optimizer created with different LR: {optimizer_type}, base_lr={base_lr}")
        return optimizer
    
    def create_optimizer_with_lr_schedule(
        self,
        model: torch.nn.Module,
        optimizer_type: str = "adamw",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        lr_schedule: str = "cosine",
        **kwargs
    ) -> torch.optim.Optimizer:
        """
        学習率スケジュール付きオプティマイザーを作成
        
        Args:
            model: モデル
            optimizer_type: オプティマイザータイプ
            learning_rate: 学習率
            weight_decay: 重み減衰
            lr_schedule: 学習率スケジュール
            **kwargs: 追加引数
        
        Returns:
            オプティマイザー
        """
        # 基本オプティマイザーを作成
        optimizer = self.create_optimizer(
            model, optimizer_type, learning_rate, weight_decay, **kwargs
        )
        
        # 学習率スケジュールを設定
        if lr_schedule == "cosine":
            # コサインアニーリング
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=100, eta_min=1e-6
            )
        elif lr_schedule == "step":
            # ステップスケジュール
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )
        elif lr_schedule == "exponential":
            # 指数スケジュール
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=0.95
            )
        else:
            scheduler = None
        
        # スケジューラーをオプティマイザーに添付
        if scheduler is not None:
            optimizer.scheduler = scheduler
        
        logger.info(f"Optimizer created with LR schedule: {optimizer_type}, schedule={lr_schedule}")
        return optimizer
    
    def create_optimizer_with_warmup(
        self,
        model: torch.nn.Module,
        optimizer_type: str = "adamw",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_steps: int = 1000,
        **kwargs
    ) -> torch.optim.Optimizer:
        """
        ウォームアップ付きオプティマイザーを作成
        
        Args:
            model: モデル
            optimizer_type: オプティマイザータイプ
            learning_rate: 学習率
            weight_decay: 重み減衰
            warmup_steps: ウォームアップステップ数
            **kwargs: 追加引数
        
        Returns:
            オプティマイザー
        """
        # 基本オプティマイザーを作成
        optimizer = self.create_optimizer(
            model, optimizer_type, learning_rate, weight_decay, **kwargs
        )
        
        # ウォームアップスケジューラーを作成
        def warmup_lr_schedule(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return 1.0
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_lr_schedule)
        optimizer.scheduler = scheduler
        
        logger.info(f"Optimizer created with warmup: {optimizer_type}, warmup_steps={warmup_steps}")
        return optimizer
    
    def create_optimizer_with_gradient_clipping(
        self,
        model: torch.nn.Module,
        optimizer_type: str = "adamw",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        max_grad_norm: float = 1.0,
        **kwargs
    ) -> torch.optim.Optimizer:
        """
        勾配クリッピング付きオプティマイザーを作成
        
        Args:
            model: モデル
            optimizer_type: オプティマイザータイプ
            learning_rate: 学習率
            weight_decay: 重み減衰
            max_grad_norm: 最大勾配ノルム
            **kwargs: 追加引数
        
        Returns:
            オプティマイザー
        """
        # 基本オプティマイザーを作成
        optimizer = self.create_optimizer(
            model, optimizer_type, learning_rate, weight_decay, **kwargs
        )
        
        # 勾配クリッピング関数を設定
        def clip_gradients():
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.clip_gradients = clip_gradients
        
        logger.info(f"Optimizer created with gradient clipping: {optimizer_type}, max_grad_norm={max_grad_norm}")
        return optimizer
    
    def get_optimizer_info(self, optimizer_type: str) -> Dict[str, Any]:
        """
        オプティマイザー情報を取得
        
        Args:
            optimizer_type: オプティマイザータイプ
        
        Returns:
            オプティマイザー情報
        """
        info = {
            'name': optimizer_type,
            'description': self._get_optimizer_description(optimizer_type),
            'parameters': self.default_params[optimizer_type],
            'use_cases': self._get_optimizer_use_cases(optimizer_type),
            'advantages': self._get_optimizer_advantages(optimizer_type),
            'disadvantages': self._get_optimizer_disadvantages(optimizer_type)
        }
        
        return info
    
    def _get_optimizer_description(self, optimizer_type: str) -> str:
        """オプティマイザーの説明を取得"""
        descriptions = {
            'adam': 'Adam - 適応的学習率オプティマイザー',
            'adamw': 'AdamW - 重み減衰付きAdam',
            'sgd': 'SGD - 確率的勾配降下法',
            'rmsprop': 'RMSprop - 二乗平均平方根伝播',
            'adagrad': 'Adagrad - 適応的勾配アルゴリズム',
            'adadelta': 'Adadelta - 適応的学習率アルゴリズム',
            'adamax': 'Adamax - 無限ノルムAdam'
        }
        
        return descriptions.get(optimizer_type, 'Unknown optimizer')
    
    def _get_optimizer_use_cases(self, optimizer_type: str) -> List[str]:
        """オプティマイザーの使用例を取得"""
        use_cases = {
            'adam': ['深層学習', '一般的な最適化', '収束が早い'],
            'adamw': ['Transformer', '重み減衰が必要', '正則化'],
            'sgd': ['古典的機械学習', '安定した学習', '解釈しやすい'],
            'rmsprop': ['RNN', '非定常問題', '勾配の変動が大きい'],
            'adagrad': ['スパースデータ', '特徴量の頻度が異なる', '自然言語処理'],
            'adadelta': ['Adagradの改良版', '学習率の調整不要', '収束が早い'],
            'adamax': ['Adamの改良版', '無限ノルム', '安定した学習']
        }
        
        return use_cases.get(optimizer_type, [])
    
    def _get_optimizer_advantages(self, optimizer_type: str) -> List[str]:
        """オプティマイザーの利点を取得"""
        advantages = {
            'adam': ['適応的学習率', '収束が早い', 'パラメータ調整が少ない'],
            'adamw': ['重み減衰', '正則化効果', 'Transformerに適している'],
            'sgd': ['シンプル', '安定', '解釈しやすい'],
            'rmsprop': ['非定常問題に適している', '勾配の変動に対応'],
            'adagrad': ['スパースデータに適している', '特徴量の頻度に対応'],
            'adadelta': ['学習率の調整不要', '収束が早い'],
            'adamax': ['無限ノルム', '安定した学習']
        }
        
        return advantages.get(optimizer_type, [])
    
    def _get_optimizer_disadvantages(self, optimizer_type: str) -> List[str]:
        """オプティマイザーの欠点を取得"""
        disadvantages = {
            'adam': ['メモリ使用量が多い', '過学習しやすい'],
            'adamw': ['パラメータが多い', '調整が複雑'],
            'sgd': ['収束が遅い', '学習率の調整が必要'],
            'rmsprop': ['局所最適解に陥りやすい'],
            'adagrad': ['学習率が小さくなりすぎる', '収束が遅い'],
            'adadelta': ['パラメータが多い'],
            'adamax': ['メモリ使用量が多い']
        }
        
        return disadvantages.get(optimizer_type, [])
    
    def get_available_optimizers(self) -> List[str]:
        """
        利用可能なオプティマイザーを取得
        
        Returns:
            オプティマイザーリスト
        """
        return list(self.optimizers.keys())
    
    def create_custom_optimizer(
        self,
        model: torch.nn.Module,
        optimizer_class: torch.optim.Optimizer,
        **kwargs
    ) -> torch.optim.Optimizer:
        """
        カスタムオプティマイザーを作成
        
        Args:
            model: モデル
            optimizer_class: オプティマイザークラス
            **kwargs: 追加引数
        
        Returns:
            カスタムオプティマイザー
        """
        optimizer = optimizer_class(model.parameters(), **kwargs)
        
        logger.info(f"Custom optimizer created: {optimizer_class.__name__}")
        return optimizer
