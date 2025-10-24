"""
Training Demo

ChemForge学習・推論システムのデモンストレーション
包括的な学習・推論・評価の実装例
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import time
import logging
from pathlib import Path
import tempfile
import shutil

from chemforge.training.trainer import Trainer
from chemforge.training.loss_functions import LossFunctions
from chemforge.training.metrics import Metrics
from chemforge.training.optimizer import OptimizerManager
from chemforge.training.scheduler import SchedulerManager
from chemforge.training.checkpoint import CheckpointManager

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DemoModel(nn.Module):
    """デモ用モデル"""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 64, output_size: int = 1):
        super(DemoModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # ネットワーク層
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
        # 活性化関数
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        # 初期化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """重みを初期化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.linear3(x)
        return x


def create_demo_data(
    num_samples: int = 1000,
    input_size: int = 10,
    output_size: int = 1,
    noise_level: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    デモ用データを作成
    
    Args:
        num_samples: サンプル数
        input_size: 入力サイズ
        output_size: 出力サイズ
        noise_level: ノイズレベル
    
    Returns:
        入力データとターゲットデータ
    """
    # 入力データを生成
    X = torch.randn(num_samples, input_size)
    
    # ターゲットデータを生成（線形関係 + ノイズ）
    true_weights = torch.randn(input_size, output_size)
    y = torch.mm(X, true_weights) + noise_level * torch.randn(num_samples, output_size)
    
    return X, y


def create_data_loaders(
    X: torch.Tensor,
    y: torch.Tensor,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 32
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    データローダーを作成
    
    Args:
        X: 入力データ
        y: ターゲットデータ
        train_ratio: 学習データ比率
        val_ratio: 検証データ比率
        batch_size: バッチサイズ
    
    Returns:
        学習・検証・テストデータローダー
    """
    # データを分割
    num_samples = len(X)
    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * val_ratio)
    
    # インデックスをシャッフル
    indices = torch.randperm(num_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # データセットを作成
    train_dataset = torch.utils.data.TensorDataset(X[train_indices], y[train_indices])
    val_dataset = torch.utils.data.TensorDataset(X[val_indices], y[val_indices])
    test_dataset = torch.utils.data.TensorDataset(X[test_indices], y[test_indices])
    
    # データローダーを作成
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader, test_loader


def demonstrate_training_system():
    """学習システムのデモンストレーション"""
    print("\n" + "="*60)
    print("🚀 Training System Demo")
    print("="*60)
    
    # デモ用データを作成
    print("\n🔹 Creating demo data...")
    X, y = create_demo_data(num_samples=1000, input_size=10, output_size=1)
    print(f"  Data shape: X={X.shape}, y={y.shape}")
    
    # データローダーを作成
    print("\n🔹 Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(X, y)
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # モデルを作成
    print("\n🔹 Creating model...")
    model = DemoModel(input_size=10, hidden_size=64, output_size=1)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # トレーナーを初期化
    print("\n🔹 Initializing trainer...")
    trainer = Trainer(
        model=model,
        device="cpu",
        use_amp=False,
        checkpoint_dir="demo_checkpoints",
        log_dir="demo_logs",
        save_best=True,
        patience=10,
        min_delta=1e-4
    )
    
    # オプティマイザーを設定
    print("\n🔹 Setting up optimizer...")
    trainer.setup_optimizer(
        optimizer_type="adamw",
        learning_rate=1e-3,
        weight_decay=1e-4
    )
    
    # スケジューラーを設定
    print("\n🔹 Setting up scheduler...")
    trainer.setup_scheduler(
        scheduler_type="cosine",
        T_max=50
    )
    
    # 損失関数を設定
    print("\n🔹 Setting up loss function...")
    trainer.setup_loss_function("mse")
    
    # 学習を実行
    print("\n🔹 Starting training...")
    start_time = time.time()
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        save_frequency=10
    )
    
    training_time = time.time() - start_time
    print(f"  Training completed in {training_time:.2f}s")
    
    # 学習履歴を表示
    print("\n🔹 Training history:")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"  Best val loss: {min(history['val_loss']):.4f}")
    
    # 評価を実行
    print("\n🔹 Evaluating model...")
    test_metrics = trainer.evaluate(test_loader)
    print(f"  Test loss: {test_metrics['loss']:.4f}")
    print(f"  Test R²: {test_metrics['r2']:.4f}")
    print(f"  Test MAE: {test_metrics['mae']:.4f}")
    
    # 予測を実行
    print("\n🔹 Making predictions...")
    predictions = trainer.predict(test_loader)
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Predictions mean: {predictions.mean():.4f}")
    print(f"  Predictions std: {predictions.std():.4f}")
    
    # 学習要約を取得
    print("\n🔹 Training summary:")
    summary = trainer.get_training_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    return trainer, history, test_metrics


def demonstrate_loss_functions():
    """損失関数のデモンストレーション"""
    print("\n" + "="*60)
    print("📊 Loss Functions Demo")
    print("="*60)
    
    # 損失関数マネージャーを初期化
    loss_functions = LossFunctions()
    
    print("\n🔹 Available loss functions:")
    available_losses = loss_functions.get_available_losses()
    for loss_name in available_losses:
        print(f"  - {loss_name}")
    
    # テストデータを作成
    print("\n🔹 Creating test data...")
    predictions = torch.randn(100, 1)
    targets = torch.randn(100, 1)
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Targets shape: {targets.shape}")
    
    # 各種損失関数をテスト
    print("\n🔹 Testing loss functions:")
    test_losses = ['mse', 'mae', 'smooth_l1', 'huber']
    
    for loss_name in test_losses:
        loss_fn = loss_functions.get_loss_function(loss_name)
        loss_value = loss_fn(predictions, targets)
        print(f"  {loss_name}: {loss_value.item():.4f}")
    
    # 損失関数情報を取得
    print("\n🔹 Loss function information:")
    for loss_name in test_losses:
        info = loss_functions.get_loss_info(loss_name)
        print(f"  {loss_name}:")
        print(f"    Description: {info['description']}")
        print(f"    Parameters: {info['parameters']}")
        print(f"    Use cases: {info['use_cases']}")


def demonstrate_metrics():
    """メトリクスのデモンストレーション"""
    print("\n" + "="*60)
    print("📈 Metrics Demo")
    print("="*60)
    
    # メトリクスマネージャーを初期化
    metrics = Metrics()
    
    print("\n🔹 Available metrics:")
    regression_metrics = metrics.get_available_metrics("regression")
    classification_metrics = metrics.get_available_metrics("classification")
    print(f"  Regression: {regression_metrics}")
    print(f"  Classification: {classification_metrics}")
    
    # 回帰メトリクスをテスト
    print("\n🔹 Testing regression metrics:")
    predictions = torch.randn(100, 1)
    targets = torch.randn(100, 1)
    
    regression_metrics_result = metrics.calculate_metrics(predictions, targets, "regression")
    for metric_name, metric_value in regression_metrics_result.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    # 分類メトリクスをテスト
    print("\n🔹 Testing classification metrics:")
    predictions_cls = torch.randn(100, 5)
    targets_cls = torch.randint(0, 5, (100,))
    
    classification_metrics_result = metrics.calculate_metrics(predictions_cls, targets_cls, "classification")
    for metric_name, metric_value in classification_metrics_result.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    # メトリクス情報を取得
    print("\n🔹 Metric information:")
    test_metrics = ['mse', 'r2', 'accuracy', 'f1']
    for metric_name in test_metrics:
        info = metrics.get_metric_info(metric_name)
        print(f"  {metric_name}:")
        print(f"    Description: {info['description']}")
        print(f"    Range: {info['range']}")
        print(f"    Interpretation: {info['interpretation']}")


def demonstrate_optimizer():
    """オプティマイザーのデモンストレーション"""
    print("\n" + "="*60)
    print("⚙️ Optimizer Demo")
    print("="*60)
    
    # オプティマイザーマネージャーを初期化
    optimizer_manager = OptimizerManager()
    
    print("\n🔹 Available optimizers:")
    available_optimizers = optimizer_manager.get_available_optimizers()
    for optimizer_name in available_optimizers:
        print(f"  - {optimizer_name}")
    
    # モデルを作成
    model = DemoModel(input_size=10, hidden_size=64, output_size=1)
    
    # 各種オプティマイザーをテスト
    print("\n🔹 Testing optimizers:")
    test_optimizers = ['adam', 'adamw', 'sgd', 'rmsprop']
    
    for optimizer_name in test_optimizers:
        optimizer = optimizer_manager.create_optimizer(
            model=model,
            optimizer_type=optimizer_name,
            learning_rate=1e-3,
            weight_decay=1e-4
        )
        print(f"  {optimizer_name}: {type(optimizer).__name__}")
    
    # オプティマイザー情報を取得
    print("\n🔹 Optimizer information:")
    for optimizer_name in test_optimizers:
        info = optimizer_manager.get_optimizer_info(optimizer_name)
        print(f"  {optimizer_name}:")
        print(f"    Description: {info['description']}")
        print(f"    Use cases: {info['use_cases']}")
        print(f"    Advantages: {info['advantages']}")
        print(f"    Disadvantages: {info['disadvantages']}")


def demonstrate_scheduler():
    """スケジューラーのデモンストレーション"""
    print("\n" + "="*60)
    print("📅 Scheduler Demo")
    print("="*60)
    
    # スケジューラーマネージャーを初期化
    scheduler_manager = SchedulerManager()
    
    print("\n🔹 Available schedulers:")
    available_schedulers = scheduler_manager.get_available_schedulers()
    for scheduler_name in available_schedulers:
        print(f"  - {scheduler_name}")
    
    # モデルとオプティマイザーを作成
    model = DemoModel(input_size=10, hidden_size=64, output_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # 各種スケジューラーをテスト
    print("\n🔹 Testing schedulers:")
    test_schedulers = ['cosine', 'step', 'exponential', 'reduce_on_plateau']
    
    for scheduler_name in test_schedulers:
        scheduler = scheduler_manager.create_scheduler(
            optimizer=optimizer,
            scheduler_type=scheduler_name
        )
        print(f"  {scheduler_name}: {type(scheduler).__name__}")
    
    # スケジューラー情報を取得
    print("\n🔹 Scheduler information:")
    for scheduler_name in test_schedulers:
        info = scheduler_manager.get_scheduler_info(scheduler_name)
        print(f"  {scheduler_name}:")
        print(f"    Description: {info['description']}")
        print(f"    Use cases: {info['use_cases']}")
        print(f"    Advantages: {info['advantages']}")
        print(f"    Disadvantages: {info['disadvantages']}")


def demonstrate_checkpoint():
    """チェックポイントのデモンストレーション"""
    print("\n" + "="*60)
    print("💾 Checkpoint Demo")
    print("="*60)
    
    # チェックポイントマネージャーを初期化
    checkpoint_manager = CheckpointManager(
        checkpoint_dir="demo_checkpoints",
        max_checkpoints=5,
        save_best=True,
        save_frequency=10,
        backup_frequency=100
    )
    
    # モデルとオプティマイザーを作成
    model = DemoModel(input_size=10, hidden_size=64, output_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    print("\n🔹 Testing checkpoint operations:")
    
    # チェックポイントを保存
    print("  Saving checkpoints...")
    for epoch in range(1, 6):
        score = 1.0 - epoch * 0.1 + np.random.normal(0, 0.05)
        is_best = score < checkpoint_manager.best_score
        
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            score=score,
            is_best=is_best
        )
        print(f"    Epoch {epoch}: score={score:.4f}, is_best={is_best}")
    
    # チェックポイントリストを取得
    print("\n🔹 Checkpoint list:")
    checkpoint_list = checkpoint_manager.get_checkpoint_list()
    for cp in checkpoint_list:
        print(f"  Epoch {cp['epoch']}: score={cp['score']:.4f}, is_best={cp['is_best']}")
    
    # 最良チェックポイント情報を取得
    print("\n🔹 Best checkpoint info:")
    best_info = checkpoint_manager.get_best_checkpoint_info()
    if best_info:
        print(f"  Path: {best_info['path']}")
        print(f"  Score: {best_info['score']:.4f}")
        print(f"  Timestamp: {best_info['timestamp']}")
    
    # チェックポイント要約を取得
    print("\n🔹 Checkpoint summary:")
    summary = checkpoint_manager.get_checkpoint_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")


def plot_training_results(history: Dict[str, List[float]]):
    """
    学習結果をプロット
    
    Args:
        history: 学習履歴
    """
    print("\n" + "="*60)
    print("📊 Training Results Visualization")
    print("="*60)
    
    # プロット設定
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # 1. 損失曲線
    ax1 = axes[0]
    ax1.plot(history['train_loss'], label='Train Loss', color='blue')
    ax1.plot(history['val_loss'], label='Val Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 学習率（スケジューラーがある場合）
    ax2 = axes[1]
    if 'learning_rate' in history:
        ax2.plot(history['learning_rate'], color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No learning rate data', ha='center', va='center')
        ax2.set_title('Learning Rate Schedule')
    
    # 3. メトリクス（利用可能な場合）
    ax3 = axes[2]
    if 'train_metrics' in history and history['train_metrics']:
        train_metrics = history['train_metrics']
        if 'r2' in train_metrics[0]:
            r2_scores = [m['r2'] for m in train_metrics]
            ax3.plot(r2_scores, label='Train R²', color='blue')
        if 'val_metrics' in history and history['val_metrics']:
            val_metrics = history['val_metrics']
            if 'r2' in val_metrics[0]:
                val_r2_scores = [m['r2'] for m in val_metrics]
                ax3.plot(val_r2_scores, label='Val R²', color='red')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('R² Score')
        ax3.set_title('R² Score Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No metrics data', ha='center', va='center')
        ax3.set_title('Metrics Over Time')
    
    # 4. 損失分布
    ax4 = axes[3]
    ax4.hist(history['train_loss'], bins=20, alpha=0.7, label='Train Loss', color='blue')
    ax4.hist(history['val_loss'], bins=20, alpha=0.7, label='Val Loss', color='red')
    ax4.set_xlabel('Loss Value')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Loss Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Training Results Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """
    メイン実行関数
    """
    print("🚀 ChemForge Training System Demo")
    print("="*60)
    
    try:
        # 1. 学習システムデモ
        trainer, history, test_metrics = demonstrate_training_system()
        
        # 2. 損失関数デモ
        demonstrate_loss_functions()
        
        # 3. メトリクスデモ
        demonstrate_metrics()
        
        # 4. オプティマイザーデモ
        demonstrate_optimizer()
        
        # 5. スケジューラーデモ
        demonstrate_scheduler()
        
        # 6. チェックポイントデモ
        demonstrate_checkpoint()
        
        # 7. 学習結果可視化
        plot_training_results(history)
        
        print("\n🎉 All training system demonstrations completed successfully!")
        print("ChemForge training system is ready for use!")
        
    except Exception as e:
        logger.error(f"Error in demo: {str(e)}")
        raise
    finally:
        # クリーンアップ
        try:
            shutil.rmtree("demo_checkpoints", ignore_errors=True)
            shutil.rmtree("demo_logs", ignore_errors=True)
        except:
            pass


if __name__ == "__main__":
    main()

