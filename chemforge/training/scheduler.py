"""
Scheduler Manager Module

包括的な学習率スケジューラー管理システム
Cosine, Step, Exponential, ReduceLROnPlateau対応
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, StepLR, ExponentialLR, ReduceLROnPlateau,
    CosineAnnealingWarmRestarts, OneCycleLR, CyclicLR,
    LambdaLR, MultiStepLR, LinearLR, PolynomialLR
)
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class SchedulerManager:
    """
    包括的な学習率スケジューラー管理システム
    
    Cosine, Step, Exponential, ReduceLROnPlateau対応
    """
    
    def __init__(self):
        """スケジューラーマネージャーを初期化"""
        self.schedulers = {
            'cosine': CosineAnnealingLR,
            'step': StepLR,
            'exponential': ExponentialLR,
            'reduce_on_plateau': ReduceLROnPlateau,
            'cosine_warm_restarts': CosineAnnealingWarmRestarts,
            'one_cycle': OneCycleLR,
            'cyclic': CyclicLR,
            'lambda': LambdaLR,
            'multi_step': MultiStepLR,
            'linear': LinearLR,
            'polynomial': PolynomialLR
        }
        
        # デフォルトパラメータ
        self.default_params = {
            'cosine': {'T_max': 100, 'eta_min': 1e-6},
            'step': {'step_size': 30, 'gamma': 0.1},
            'exponential': {'gamma': 0.95},
            'reduce_on_plateau': {'mode': 'min', 'factor': 0.5, 'patience': 10, 'threshold': 1e-4},
            'cosine_warm_restarts': {'T_0': 10, 'T_mult': 1, 'eta_min': 1e-6},
            'one_cycle': {'max_lr': 1e-3, 'total_steps': 1000, 'pct_start': 0.3},
            'cyclic': {'base_lr': 1e-6, 'max_lr': 1e-3, 'step_size_up': 500},
            'lambda': {'lr_lambda': lambda epoch: 0.95 ** epoch},
            'multi_step': {'milestones': [30, 60, 90], 'gamma': 0.1},
            'linear': {'start_factor': 0.1, 'total_iters': 100},
            'polynomial': {'total_iters': 100, 'power': 2.0}
        }
        
        logger.info("SchedulerManager initialized")
    
    def create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_type: str = "cosine",
        **kwargs
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """
        スケジューラーを作成
        
        Args:
            optimizer: オプティマイザー
            scheduler_type: スケジューラータイプ
            **kwargs: 追加引数
        
        Returns:
            スケジューラー
        """
        if scheduler_type not in self.schedulers:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
        # パラメータを準備
        params = self._prepare_scheduler_params(scheduler_type, **kwargs)
        
        # スケジューラーを作成
        scheduler_class = self.schedulers[scheduler_type]
        scheduler = scheduler_class(optimizer, **params)
        
        logger.info(f"Scheduler created: {scheduler_type}")
        return scheduler
    
    def _prepare_scheduler_params(
        self,
        scheduler_type: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        スケジューラーパラメータを準備
        
        Args:
            scheduler_type: スケジューラータイプ
            **kwargs: 追加引数
        
        Returns:
            スケジューラーパラメータ
        """
        # デフォルトパラメータを取得
        params = self.default_params[scheduler_type].copy()
        
        # 追加引数を適用
        params.update(kwargs)
        
        return params
    
    def create_cosine_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        T_max: int = 100,
        eta_min: float = 1e-6
    ) -> CosineAnnealingLR:
        """
        コサインアニーリングスケジューラーを作成
        
        Args:
            optimizer: オプティマイザー
            T_max: 最大エポック数
            eta_min: 最小学習率
        
        Returns:
            コサインアニーリングスケジューラー
        """
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        
        logger.info(f"Cosine scheduler created: T_max={T_max}, eta_min={eta_min}")
        return scheduler
    
    def create_step_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        step_size: int = 30,
        gamma: float = 0.1
    ) -> StepLR:
        """
        ステップスケジューラーを作成
        
        Args:
            optimizer: オプティマイザー
            step_size: ステップサイズ
            gamma: 学習率乗数
        
        Returns:
            ステップスケジューラー
        """
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        
        logger.info(f"Step scheduler created: step_size={step_size}, gamma={gamma}")
        return scheduler
    
    def create_exponential_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        gamma: float = 0.95
    ) -> ExponentialLR:
        """
        指数スケジューラーを作成
        
        Args:
            optimizer: オプティマイザー
            gamma: 学習率乗数
        
        Returns:
            指数スケジューラー
        """
        scheduler = ExponentialLR(optimizer, gamma=gamma)
        
        logger.info(f"Exponential scheduler created: gamma={gamma}")
        return scheduler
    
    def create_reduce_on_plateau_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        mode: str = 'min',
        factor: float = 0.5,
        patience: int = 10,
        threshold: float = 1e-4
    ) -> ReduceLROnPlateau:
        """
        プラトー減少スケジューラーを作成
        
        Args:
            optimizer: オプティマイザー
            mode: モード ('min' or 'max')
            factor: 学習率乗数
            patience: パティエンス
            threshold: 閾値
        
        Returns:
            プラトー減少スケジューラー
        """
        scheduler = ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience, threshold=threshold
        )
        
        logger.info(f"Reduce on plateau scheduler created: mode={mode}, factor={factor}")
        return scheduler
    
    def create_cosine_warm_restarts_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        T_0: int = 10,
        T_mult: int = 1,
        eta_min: float = 1e-6
    ) -> CosineAnnealingWarmRestarts:
        """
        コサインウォームリスタートスケジューラーを作成
        
        Args:
            optimizer: オプティマイザー
            T_0: 初期周期
            T_mult: 周期乗数
            eta_min: 最小学習率
        
        Returns:
            コサインウォームリスタートスケジューラー
        """
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )
        
        logger.info(f"Cosine warm restarts scheduler created: T_0={T_0}, T_mult={T_mult}")
        return scheduler
    
    def create_one_cycle_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float = 1e-3,
        total_steps: int = 1000,
        pct_start: float = 0.3
    ) -> OneCycleLR:
        """
        ワンサイクルスケジューラーを作成
        
        Args:
            optimizer: オプティマイザー
            max_lr: 最大学習率
            total_steps: 総ステップ数
            pct_start: 上昇期間の割合
        
        Returns:
            ワンサイクルスケジューラー
        """
        scheduler = OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=pct_start
        )
        
        logger.info(f"One cycle scheduler created: max_lr={max_lr}, total_steps={total_steps}")
        return scheduler
    
    def create_cyclic_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float = 1e-6,
        max_lr: float = 1e-3,
        step_size_up: int = 500
    ) -> CyclicLR:
        """
        サイクリックスケジューラーを作成
        
        Args:
            optimizer: オプティマイザー
            base_lr: ベース学習率
            max_lr: 最大学習率
            step_size_up: 上昇ステップ数
        
        Returns:
            サイクリックスケジューラー
        """
        scheduler = CyclicLR(
            optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up
        )
        
        logger.info(f"Cyclic scheduler created: base_lr={base_lr}, max_lr={max_lr}")
        return scheduler
    
    def create_lambda_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        lr_lambda: Callable[[int], float]
    ) -> LambdaLR:
        """
        ラムダスケジューラーを作成
        
        Args:
            optimizer: オプティマイザー
            lr_lambda: 学習率関数
        
        Returns:
            ラムダスケジューラー
        """
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        
        logger.info("Lambda scheduler created")
        return scheduler
    
    def create_multi_step_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        milestones: List[int] = [30, 60, 90],
        gamma: float = 0.1
    ) -> MultiStepLR:
        """
        マルチステップスケジューラーを作成
        
        Args:
            optimizer: オプティマイザー
            milestones: マイルストーン
            gamma: 学習率乗数
        
        Returns:
            マルチステップスケジューラー
        """
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        
        logger.info(f"Multi-step scheduler created: milestones={milestones}")
        return scheduler
    
    def create_linear_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        start_factor: float = 0.1,
        total_iters: int = 100
    ) -> LinearLR:
        """
        リニアスケジューラーを作成
        
        Args:
            optimizer: オプティマイザー
            start_factor: 開始係数
            total_iters: 総イテレーション数
        
        Returns:
            リニアスケジューラー
        """
        scheduler = LinearLR(optimizer, start_factor=start_factor, total_iters=total_iters)
        
        logger.info(f"Linear scheduler created: start_factor={start_factor}")
        return scheduler
    
    def create_polynomial_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        total_iters: int = 100,
        power: float = 2.0
    ) -> PolynomialLR:
        """
        多項式スケジューラーを作成
        
        Args:
            optimizer: オプティマイザー
            total_iters: 総イテレーション数
            power: べき乗
        
        Returns:
            多項式スケジューラー
        """
        scheduler = PolynomialLR(optimizer, total_iters=total_iters, power=power)
        
        logger.info(f"Polynomial scheduler created: total_iters={total_iters}, power={power}")
        return scheduler
    
    def create_custom_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_class: torch.optim.lr_scheduler._LRScheduler,
        **kwargs
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """
        カスタムスケジューラーを作成
        
        Args:
            optimizer: オプティマイザー
            scheduler_class: スケジューラークラス
            **kwargs: 追加引数
        
        Returns:
            カスタムスケジューラー
        """
        scheduler = scheduler_class(optimizer, **kwargs)
        
        logger.info(f"Custom scheduler created: {scheduler_class.__name__}")
        return scheduler
    
    def get_scheduler_info(self, scheduler_type: str) -> Dict[str, Any]:
        """
        スケジューラー情報を取得
        
        Args:
            scheduler_type: スケジューラータイプ
        
        Returns:
            スケジューラー情報
        """
        info = {
            'name': scheduler_type,
            'description': self._get_scheduler_description(scheduler_type),
            'parameters': self.default_params[scheduler_type],
            'use_cases': self._get_scheduler_use_cases(scheduler_type),
            'advantages': self._get_scheduler_advantages(scheduler_type),
            'disadvantages': self._get_scheduler_disadvantages(scheduler_type)
        }
        
        return info
    
    def _get_scheduler_description(self, scheduler_type: str) -> str:
        """スケジューラーの説明を取得"""
        descriptions = {
            'cosine': 'Cosine Annealing - コサインアニーリング',
            'step': 'Step - ステップスケジューラー',
            'exponential': 'Exponential - 指数スケジューラー',
            'reduce_on_plateau': 'Reduce on Plateau - プラトー減少',
            'cosine_warm_restarts': 'Cosine Warm Restarts - コサインウォームリスタート',
            'one_cycle': 'One Cycle - ワンサイクル',
            'cyclic': 'Cyclic - サイクリック',
            'lambda': 'Lambda - ラムダスケジューラー',
            'multi_step': 'Multi Step - マルチステップ',
            'linear': 'Linear - リニアスケジューラー',
            'polynomial': 'Polynomial - 多項式スケジューラー'
        }
        
        return descriptions.get(scheduler_type, 'Unknown scheduler')
    
    def _get_scheduler_use_cases(self, scheduler_type: str) -> List[str]:
        """スケジューラーの使用例を取得"""
        use_cases = {
            'cosine': ['深層学習', '収束が早い', '最終的な性能が良い'],
            'step': ['古典的機械学習', '段階的な学習率減少'],
            'exponential': ['指数関数的な減少', '急速な収束'],
            'reduce_on_plateau': ['プラトー検出', '適応的学習率調整'],
            'cosine_warm_restarts': ['局所最適解回避', '探索と活用のバランス'],
            'one_cycle': ['高速学習', '過学習防止', '最適な学習率探索'],
            'cyclic': ['学習率探索', '最適な学習率発見'],
            'lambda': ['カスタムスケジュール', '柔軟な学習率制御'],
            'multi_step': ['複数のマイルストーン', '段階的な学習率調整'],
            'linear': ['線形な学習率変化', '滑らかな学習率調整'],
            'polynomial': ['多項式的な学習率変化', '柔軟な学習率制御']
        }
        
        return use_cases.get(scheduler_type, [])
    
    def _get_scheduler_advantages(self, scheduler_type: str) -> List[str]:
        """スケジューラーの利点を取得"""
        advantages = {
            'cosine': ['滑らかな学習率変化', '最終的な性能が良い', '収束が早い'],
            'step': ['シンプル', '解釈しやすい', '実装が簡単'],
            'exponential': ['急速な収束', '計算効率が良い'],
            'reduce_on_plateau': ['適応的', 'プラトー検出', '過学習防止'],
            'cosine_warm_restarts': ['局所最適解回避', '探索と活用のバランス'],
            'one_cycle': ['高速学習', '過学習防止', '最適な学習率探索'],
            'cyclic': ['学習率探索', '最適な学習率発見'],
            'lambda': ['柔軟性', 'カスタマイズ可能'],
            'multi_step': ['段階的調整', '複数のマイルストーン'],
            'linear': ['滑らかな変化', '予測可能'],
            'polynomial': ['柔軟な制御', '多様な学習率パターン']
        }
        
        return advantages.get(scheduler_type, [])
    
    def _get_scheduler_disadvantages(self, scheduler_type: str) -> List[str]:
        """スケジューラーの欠点を取得"""
        disadvantages = {
            'cosine': ['パラメータ調整が必要', '複雑'],
            'step': ['急激な変化', '最適でない場合がある'],
            'exponential': ['急速すぎる場合がある', '学習率が小さくなりすぎる'],
            'reduce_on_plateau': ['プラトー検出が困難', 'パラメータ調整が必要'],
            'cosine_warm_restarts': ['複雑', 'パラメータ調整が必要'],
            'one_cycle': ['複雑', 'パラメータ調整が必要'],
            'cyclic': ['複雑', '計算コストが高い'],
            'lambda': ['カスタム実装が必要', 'デバッグが困難'],
            'multi_step': ['マイルストーン設定が必要', '柔軟性が低い'],
            'linear': ['限定的なパターン', '柔軟性が低い'],
            'polynomial': ['複雑', 'パラメータ調整が必要']
        }
        
        return disadvantages.get(scheduler_type, [])
    
    def get_available_schedulers(self) -> List[str]:
        """
        利用可能なスケジューラーを取得
        
        Returns:
            スケジューラーリスト
        """
        return list(self.schedulers.keys())
    
    def create_scheduler_with_warmup(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 1000,
        scheduler_type: str = "cosine",
        **kwargs
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """
        ウォームアップ付きスケジューラーを作成
        
        Args:
            optimizer: オプティマイザー
            warmup_steps: ウォームアップステップ数
            scheduler_type: スケジューラータイプ
            **kwargs: 追加引数
        
        Returns:
            ウォームアップ付きスケジューラー
        """
        # ウォームアップ関数を定義
        def warmup_lr_schedule(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return 1.0
        
        # ラムダスケジューラーを作成
        scheduler = LambdaLR(optimizer, lr_lambda=warmup_lr_schedule)
        
        logger.info(f"Scheduler with warmup created: warmup_steps={warmup_steps}")
        return scheduler
