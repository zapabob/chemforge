"""
Training System Tests

学習・推論システムのユニットテスト
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import tempfile
import shutil
from pathlib import Path

from chemforge.training.trainer import Trainer
from chemforge.training.loss_functions import LossFunctions
from chemforge.training.metrics import Metrics
from chemforge.training.optimizer import OptimizerManager
from chemforge.training.scheduler import SchedulerManager
from chemforge.training.checkpoint import CheckpointManager


class TestModel(nn.Module):
    """テスト用モデル"""
    
    def __init__(self, input_size: int = 10, output_size: int = 1):
        super(TestModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)


class TestTrainer:
    """トレーナーのテストクラス"""
    
    def setup_method(self):
        """テスト前のセットアップ"""
        self.model = TestModel()
        self.device = "cpu"
        self.trainer = Trainer(
            model=self.model,
            device=self.device,
            use_amp=False,
            checkpoint_dir="test_checkpoints",
            log_dir="test_logs"
        )
    
    def test_initialization(self):
        """初期化テスト"""
        assert self.trainer.model == self.model
        assert self.trainer.device == self.device
        assert self.trainer.use_amp == False
        assert self.trainer.checkpoint_dir == Path("test_checkpoints")
        assert self.trainer.log_dir == Path("test_logs")
    
    def test_setup_optimizer(self):
        """オプティマイザー設定テスト"""
        self.trainer.setup_optimizer(
            optimizer_type="adamw",
            learning_rate=1e-4,
            weight_decay=1e-5
        )
        
        assert self.trainer.optimizer is not None
        assert isinstance(self.trainer.optimizer, torch.optim.AdamW)
    
    def test_setup_scheduler(self):
        """スケジューラー設定テスト"""
        self.trainer.setup_optimizer("adamw", 1e-4, 1e-5)
        self.trainer.setup_scheduler("cosine")
        
        assert self.trainer.scheduler is not None
        assert isinstance(self.trainer.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    
    def test_setup_loss_function(self):
        """損失関数設定テスト"""
        self.trainer.setup_loss_function("mse")
        
        assert self.trainer.loss_function is not None
        assert isinstance(self.trainer.loss_function, nn.MSELoss)
    
    def test_train_epoch(self):
        """学習エポックテスト"""
        # オプティマイザーと損失関数を設定
        self.trainer.setup_optimizer("adamw", 1e-4, 1e-5)
        self.trainer.setup_loss_function("mse")
        
        # ダミーデータローダーを作成
        X = torch.randn(100, 10)
        y = torch.randn(100, 1)
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
        
        # 学習エポックを実行
        metrics = self.trainer.train_epoch(dataloader)
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert metrics['loss'] >= 0
    
    def test_validate_epoch(self):
        """検証エポックテスト"""
        # オプティマイザーと損失関数を設定
        self.trainer.setup_optimizer("adamw", 1e-4, 1e-5)
        self.trainer.setup_loss_function("mse")
        
        # ダミーデータローダーを作成
        X = torch.randn(100, 10)
        y = torch.randn(100, 1)
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
        
        # 検証エポックを実行
        metrics = self.trainer.validate_epoch(dataloader)
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert metrics['loss'] >= 0
    
    def test_predict(self):
        """予測テスト"""
        # ダミーデータローダーを作成
        X = torch.randn(100, 10)
        y = torch.randn(100, 1)
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
        
        # 予測を実行
        predictions = self.trainer.predict(dataloader)
        
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape[0] == 100
        assert predictions.shape[1] == 1
    
    def test_evaluate(self):
        """評価テスト"""
        # オプティマイザーと損失関数を設定
        self.trainer.setup_optimizer("adamw", 1e-4, 1e-5)
        self.trainer.setup_loss_function("mse")
        
        # ダミーデータローダーを作成
        X = torch.randn(100, 10)
        y = torch.randn(100, 1)
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
        
        # 評価を実行
        metrics = self.trainer.evaluate(dataloader)
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert metrics['loss'] >= 0
    
    def test_get_training_summary(self):
        """学習要約取得テスト"""
        summary = self.trainer.get_training_summary()
        
        assert isinstance(summary, dict)
        assert 'current_epoch' in summary
        assert 'best_score' in summary
        assert 'patience_counter' in summary
        assert 'model_parameters' in summary
        assert 'trainable_parameters' in summary
        assert 'device' in summary
        assert 'use_amp' in summary


class TestLossFunctions:
    """損失関数のテストクラス"""
    
    def setup_method(self):
        """テスト前のセットアップ"""
        self.loss_functions = LossFunctions()
    
    def test_initialization(self):
        """初期化テスト"""
        assert self.loss_functions.loss_functions is not None
        assert len(self.loss_functions.loss_functions) > 0
    
    def test_get_loss_function(self):
        """損失関数取得テスト"""
        # MSE損失関数
        mse_loss = self.loss_functions.get_loss_function("mse")
        assert isinstance(mse_loss, nn.MSELoss)
        
        # MAE損失関数
        mae_loss = self.loss_functions.get_loss_function("mae")
        assert isinstance(mae_loss, nn.L1Loss)
        
        # Cross Entropy損失関数
        ce_loss = self.loss_functions.get_loss_function("cross_entropy")
        assert isinstance(ce_loss, nn.CrossEntropyLoss)
    
    def test_focal_loss(self):
        """Focal Lossテスト"""
        focal_loss = self.loss_functions.get_loss_function("focal_loss", alpha=1.0, gamma=2.0)
        assert focal_loss is not None
        
        # テストデータ
        inputs = torch.randn(10, 5)
        targets = torch.randint(0, 5, (10,))
        
        # 損失計算
        loss = focal_loss(inputs, targets)
        assert loss.item() >= 0
    
    def test_dice_loss(self):
        """Dice Lossテスト"""
        dice_loss = self.loss_functions.get_loss_function("dice_loss", smooth=1e-5)
        assert dice_loss is not None
        
        # テストデータ
        inputs = torch.randn(10, 1)
        targets = torch.randn(10, 1)
        
        # 損失計算
        loss = dice_loss(inputs, targets)
        assert loss.item() >= 0
    
    def test_multi_task_loss(self):
        """マルチタスク損失テスト"""
        multi_task_loss = self.loss_functions.get_loss_function(
            "multi_task_loss",
            task_weights=[1.0, 0.5],
            loss_types=["mse", "mae"]
        )
        assert multi_task_loss is not None
        
        # テストデータ
        outputs = [torch.randn(10, 1), torch.randn(10, 1)]
        targets = [torch.randn(10, 1), torch.randn(10, 1)]
        
        # 損失計算
        loss = multi_task_loss(outputs, targets)
        assert loss.item() >= 0
    
    def test_get_available_losses(self):
        """利用可能な損失関数取得テスト"""
        losses = self.loss_functions.get_available_losses()
        assert isinstance(losses, list)
        assert len(losses) > 0
        assert "mse" in losses
        assert "mae" in losses
        assert "cross_entropy" in losses
    
    def test_get_loss_info(self):
        """損失関数情報取得テスト"""
        info = self.loss_functions.get_loss_info("mse")
        assert isinstance(info, dict)
        assert 'name' in info
        assert 'description' in info
        assert 'parameters' in info
        assert 'use_cases' in info


class TestMetrics:
    """メトリクスのテストクラス"""
    
    def setup_method(self):
        """テスト前のセットアップ"""
        self.metrics = Metrics()
    
    def test_initialization(self):
        """初期化テスト"""
        assert self.metrics.regression_metrics is not None
        assert self.metrics.classification_metrics is not None
        assert len(self.metrics.regression_metrics) > 0
        assert len(self.metrics.classification_metrics) > 0
    
    def test_calculate_metrics_regression(self):
        """回帰メトリクス計算テスト"""
        predictions = torch.randn(100, 1)
        targets = torch.randn(100, 1)
        
        metrics = self.metrics.calculate_metrics(predictions, targets, "regression")
        
        assert isinstance(metrics, dict)
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'pearson' in metrics
        assert 'spearman' in metrics
    
    def test_calculate_metrics_classification(self):
        """分類メトリクス計算テスト"""
        predictions = torch.randn(100, 5)
        targets = torch.randint(0, 5, (100,))
        
        metrics = self.metrics.calculate_metrics(predictions, targets, "classification")
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'auc' in metrics
        assert 'ap' in metrics
    
    def test_calculate_multi_task_metrics(self):
        """マルチタスクメトリクス計算テスト"""
        predictions = [torch.randn(100, 1), torch.randn(100, 1)]
        targets = [torch.randn(100, 1), torch.randn(100, 1)]
        task_types = ["regression", "regression"]
        
        metrics = self.metrics.calculate_multi_task_metrics(predictions, targets, task_types)
        
        assert isinstance(metrics, dict)
        assert 'task_0' in metrics
        assert 'task_1' in metrics
        assert 'mean_mse' in metrics
        assert 'std_mse' in metrics
    
    def test_get_confusion_matrix(self):
        """混同行列取得テスト"""
        predictions = torch.randn(100, 5)
        targets = torch.randint(0, 5, (100,))
        
        cm = self.metrics.get_confusion_matrix(predictions, targets)
        
        assert isinstance(cm, np.ndarray)
        assert cm.shape[0] == 5
        assert cm.shape[1] == 5
    
    def test_get_classification_report(self):
        """分類レポート取得テスト"""
        predictions = torch.randn(100, 5)
        targets = torch.randint(0, 5, (100,))
        
        report = self.metrics.get_classification_report(predictions, targets)
        
        assert isinstance(report, str)
        assert len(report) > 0
    
    def test_get_available_metrics(self):
        """利用可能なメトリクス取得テスト"""
        regression_metrics = self.metrics.get_available_metrics("regression")
        classification_metrics = self.metrics.get_available_metrics("classification")
        
        assert isinstance(regression_metrics, list)
        assert isinstance(classification_metrics, list)
        assert len(regression_metrics) > 0
        assert len(classification_metrics) > 0
    
    def test_get_metric_info(self):
        """メトリクス情報取得テスト"""
        info = self.metrics.get_metric_info("mse")
        assert isinstance(info, dict)
        assert 'name' in info
        assert 'description' in info
        assert 'range' in info
        assert 'interpretation' in info


class TestOptimizerManager:
    """オプティマイザーマネージャーのテストクラス"""
    
    def setup_method(self):
        """テスト前のセットアップ"""
        self.optimizer_manager = OptimizerManager()
        self.model = TestModel()
    
    def test_initialization(self):
        """初期化テスト"""
        assert self.optimizer_manager.optimizers is not None
        assert self.optimizer_manager.default_params is not None
        assert len(self.optimizer_manager.optimizers) > 0
        assert len(self.optimizer_manager.default_params) > 0
    
    def test_create_optimizer(self):
        """オプティマイザー作成テスト"""
        # AdamWオプティマイザー
        optimizer = self.optimizer_manager.create_optimizer(
            model=self.model,
            optimizer_type="adamw",
            learning_rate=1e-4,
            weight_decay=1e-5
        )
        
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]['lr'] == 1e-4
        assert optimizer.param_groups[0]['weight_decay'] == 1e-5
    
    def test_create_optimizer_with_different_lr(self):
        """異なる学習率でオプティマイザー作成テスト"""
        lr_multipliers = {
            'linear.weight': 2.0,
            'linear.bias': 1.0
        }
        
        optimizer = self.optimizer_manager.create_optimizer_with_different_lr(
            model=self.model,
            optimizer_type="adamw",
            base_lr=1e-4,
            weight_decay=1e-5,
            lr_multipliers=lr_multipliers
        )
        
        assert isinstance(optimizer, torch.optim.AdamW)
        assert len(optimizer.param_groups) == 2
    
    def test_create_optimizer_with_lr_schedule(self):
        """学習率スケジュール付きオプティマイザー作成テスト"""
        optimizer = self.optimizer_manager.create_optimizer_with_lr_schedule(
            model=self.model,
            optimizer_type="adamw",
            learning_rate=1e-4,
            weight_decay=1e-5,
            lr_schedule="cosine"
        )
        
        assert isinstance(optimizer, torch.optim.AdamW)
        assert hasattr(optimizer, 'scheduler')
        assert optimizer.scheduler is not None
    
    def test_create_optimizer_with_warmup(self):
        """ウォームアップ付きオプティマイザー作成テスト"""
        optimizer = self.optimizer_manager.create_optimizer_with_warmup(
            model=self.model,
            optimizer_type="adamw",
            learning_rate=1e-4,
            weight_decay=1e-5,
            warmup_steps=1000
        )
        
        assert isinstance(optimizer, torch.optim.AdamW)
        assert hasattr(optimizer, 'scheduler')
        assert optimizer.scheduler is not None
    
    def test_create_optimizer_with_gradient_clipping(self):
        """勾配クリッピング付きオプティマイザー作成テスト"""
        optimizer = self.optimizer_manager.create_optimizer_with_gradient_clipping(
            model=self.model,
            optimizer_type="adamw",
            learning_rate=1e-4,
            weight_decay=1e-5,
            max_grad_norm=1.0
        )
        
        assert isinstance(optimizer, torch.optim.AdamW)
        assert hasattr(optimizer, 'clip_gradients')
        assert optimizer.clip_gradients is not None
    
    def test_get_optimizer_info(self):
        """オプティマイザー情報取得テスト"""
        info = self.optimizer_manager.get_optimizer_info("adamw")
        assert isinstance(info, dict)
        assert 'name' in info
        assert 'description' in info
        assert 'parameters' in info
        assert 'use_cases' in info
        assert 'advantages' in info
        assert 'disadvantages' in info
    
    def test_get_available_optimizers(self):
        """利用可能なオプティマイザー取得テスト"""
        optimizers = self.optimizer_manager.get_available_optimizers()
        assert isinstance(optimizers, list)
        assert len(optimizers) > 0
        assert "adam" in optimizers
        assert "adamw" in optimizers
        assert "sgd" in optimizers


class TestSchedulerManager:
    """スケジューラーマネージャーのテストクラス"""
    
    def setup_method(self):
        """テスト前のセットアップ"""
        self.scheduler_manager = SchedulerManager()
        self.model = TestModel()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
    
    def test_initialization(self):
        """初期化テスト"""
        assert self.scheduler_manager.schedulers is not None
        assert self.scheduler_manager.default_params is not None
        assert len(self.scheduler_manager.schedulers) > 0
        assert len(self.scheduler_manager.default_params) > 0
    
    def test_create_scheduler(self):
        """スケジューラー作成テスト"""
        # コサインスケジューラー
        scheduler = self.scheduler_manager.create_scheduler(
            optimizer=self.optimizer,
            scheduler_type="cosine"
        )
        
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    
    def test_create_cosine_scheduler(self):
        """コサインスケジューラー作成テスト"""
        scheduler = self.scheduler_manager.create_cosine_scheduler(
            optimizer=self.optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        assert scheduler.T_max == 100
        assert scheduler.eta_min == 1e-6
    
    def test_create_step_scheduler(self):
        """ステップスケジューラー作成テスト"""
        scheduler = self.scheduler_manager.create_step_scheduler(
            optimizer=self.optimizer,
            step_size=30,
            gamma=0.1
        )
        
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
        assert scheduler.step_size == 30
        assert scheduler.gamma == 0.1
    
    def test_create_exponential_scheduler(self):
        """指数スケジューラー作成テスト"""
        scheduler = self.scheduler_manager.create_exponential_scheduler(
            optimizer=self.optimizer,
            gamma=0.95
        )
        
        assert isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR)
        assert scheduler.gamma == 0.95
    
    def test_create_reduce_on_plateau_scheduler(self):
        """プラトー減少スケジューラー作成テスト"""
        scheduler = self.scheduler_manager.create_reduce_on_plateau_scheduler(
            optimizer=self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            threshold=1e-4
        )
        
        assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        assert scheduler.mode == 'min'
        assert scheduler.factor == 0.5
        assert scheduler.patience == 10
        assert scheduler.threshold == 1e-4
    
    def test_get_scheduler_info(self):
        """スケジューラー情報取得テスト"""
        info = self.scheduler_manager.get_scheduler_info("cosine")
        assert isinstance(info, dict)
        assert 'name' in info
        assert 'description' in info
        assert 'parameters' in info
        assert 'use_cases' in info
        assert 'advantages' in info
        assert 'disadvantages' in info
    
    def test_get_available_schedulers(self):
        """利用可能なスケジューラー取得テスト"""
        schedulers = self.scheduler_manager.get_available_schedulers()
        assert isinstance(schedulers, list)
        assert len(schedulers) > 0
        assert "cosine" in schedulers
        assert "step" in schedulers
        assert "exponential" in schedulers


class TestCheckpointManager:
    """チェックポイントマネージャーのテストクラス"""
    
    def setup_method(self):
        """テスト前のセットアップ"""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.temp_dir,
            max_checkpoints=5,
            save_best=True,
            save_frequency=10,
            backup_frequency=100
        )
        self.model = TestModel()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
    
    def teardown_method(self):
        """テスト後のクリーンアップ"""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """初期化テスト"""
        assert self.checkpoint_manager.checkpoint_dir == Path(self.temp_dir)
        assert self.checkpoint_manager.max_checkpoints == 5
        assert self.checkpoint_manager.save_best == True
        assert self.checkpoint_manager.save_frequency == 10
        assert self.checkpoint_manager.backup_frequency == 100
    
    def test_save_checkpoint(self):
        """チェックポイント保存テスト"""
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=1,
            score=0.5,
            is_best=True
        )
        
        assert Path(checkpoint_path).exists()
        assert len(self.checkpoint_manager.checkpoint_history) == 1
        assert self.checkpoint_manager.best_score == 0.5
        assert self.checkpoint_manager.best_checkpoint_path == checkpoint_path
    
    def test_load_checkpoint(self):
        """チェックポイント読み込みテスト"""
        # チェックポイントを保存
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=1,
            score=0.5,
            is_best=True
        )
        
        # チェックポイントを読み込み
        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        
        assert isinstance(checkpoint, dict)
        assert 'epoch' in checkpoint
        assert 'score' in checkpoint
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert checkpoint['epoch'] == 1
        assert checkpoint['score'] == 0.5
    
    def test_load_best_checkpoint(self):
        """最良チェックポイント読み込みテスト"""
        # 複数のチェックポイントを保存
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=1,
            score=0.8,
            is_best=False
        )
        
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=2,
            score=0.3,
            is_best=True
        )
        
        # 最良チェックポイントを読み込み
        checkpoint = self.checkpoint_manager.load_best_checkpoint()
        
        assert isinstance(checkpoint, dict)
        assert checkpoint['score'] == 0.3
        assert checkpoint['is_best'] == True
    
    def test_load_latest_checkpoint(self):
        """最新チェックポイント読み込みテスト"""
        # 複数のチェックポイントを保存
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=1,
            score=0.8,
            is_best=False
        )
        
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=2,
            score=0.3,
            is_best=True
        )
        
        # 最新チェックポイントを読み込み
        checkpoint = self.checkpoint_manager.load_latest_checkpoint()
        
        assert isinstance(checkpoint, dict)
        assert checkpoint['epoch'] == 2
    
    def test_get_checkpoint_list(self):
        """チェックポイントリスト取得テスト"""
        # 複数のチェックポイントを保存
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=1,
            score=0.8,
            is_best=False
        )
        
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=2,
            score=0.3,
            is_best=True
        )
        
        # チェックポイントリストを取得
        checkpoint_list = self.checkpoint_manager.get_checkpoint_list()
        
        assert isinstance(checkpoint_list, list)
        assert len(checkpoint_list) == 2
    
    def test_get_best_checkpoint_info(self):
        """最良チェックポイント情報取得テスト"""
        # チェックポイントを保存
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=1,
            score=0.5,
            is_best=True
        )
        
        # 最良チェックポイント情報を取得
        best_info = self.checkpoint_manager.get_best_checkpoint_info()
        
        assert isinstance(best_info, dict)
        assert 'path' in best_info
        assert 'score' in best_info
        assert 'timestamp' in best_info
        assert best_info['score'] == 0.5
    
    def test_delete_checkpoint(self):
        """チェックポイント削除テスト"""
        # チェックポイントを保存
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=1,
            score=0.5,
            is_best=True
        )
        
        # チェックポイントを削除
        success = self.checkpoint_manager.delete_checkpoint(checkpoint_path)
        
        assert success == True
        assert not Path(checkpoint_path).exists()
        assert len(self.checkpoint_manager.checkpoint_history) == 0
    
    def test_get_checkpoint_summary(self):
        """チェックポイント要約取得テスト"""
        summary = self.checkpoint_manager.get_checkpoint_summary()
        
        assert isinstance(summary, dict)
        assert 'total_checkpoints' in summary
        assert 'best_score' in summary
        assert 'best_checkpoint_path' in summary
        assert 'checkpoint_dir' in summary
        assert 'max_checkpoints' in summary
        assert 'save_best' in summary
        assert 'save_frequency' in summary
        assert 'backup_frequency' in summary
    
    def test_validate_checkpoint(self):
        """チェックポイント検証テスト"""
        # チェックポイントを保存
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=1,
            score=0.5,
            is_best=True
        )
        
        # チェックポイントを検証
        is_valid = self.checkpoint_manager.validate_checkpoint(checkpoint_path)
        
        assert is_valid == True
    
    def test_get_checkpoint_metadata(self):
        """チェックポイントメタデータ取得テスト"""
        # チェックポイントを保存
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=1,
            score=0.5,
            is_best=True
        )
        
        # メタデータを取得
        metadata = self.checkpoint_manager.get_checkpoint_metadata(checkpoint_path)
        
        assert isinstance(metadata, dict)
        assert 'epoch' in metadata
        assert 'score' in metadata
        assert 'timestamp' in metadata
        assert 'is_best' in metadata
        assert 'file_size' in metadata
        assert 'additional_info' in metadata
