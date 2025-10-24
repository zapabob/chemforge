"""
Deep Learning Models Demo

ChemForge深層学習モデルのデモンストレーション
Transformer, GNN, Ensembleモデルの統合例
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import time
import logging

from chemforge.models.transformer_model import TransformerRegressor
from chemforge.models.gnn_model import GNNRegressor, MolecularGNN, MultiScaleGNN
from chemforge.models.ensemble_model import EnsembleRegressor, HybridEnsemble, AdaptiveEnsemble
from chemforge.models.model_factory import ModelFactory

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_synthetic_data(num_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    合成データを作成（デモ用）
    
    Args:
        num_samples: サンプル数
    
    Returns:
        (特徴量, ターゲット)
    """
    logger.info(f"Creating synthetic data with {num_samples} samples")
    
    # 特徴量（2279次元）
    features = torch.randn(num_samples, 2279)
    
    # ターゲット（13種類のCNSターゲット）
    targets = torch.randn(num_samples, 13) * 1.0 + 8.0
    targets = torch.clamp(targets, 6.0, 10.0)
    
    logger.info(f"Data created: {features.shape} -> {targets.shape}")
    return features, targets


def create_synthetic_graph_data(num_samples: int = 1000) -> List[Dict]:
    """
    合成グラフデータを作成（デモ用）
    
    Args:
        num_samples: サンプル数
    
    Returns:
        グラフデータリスト
    """
    logger.info(f"Creating synthetic graph data with {num_samples} samples")
    
    graph_data = []
    
    for i in range(num_samples):
        # ランダムなグラフサイズ
        num_nodes = np.random.randint(10, 50)
        
        # ノード特徴量
        node_features = torch.randn(num_nodes, 100)
        
        # エッジインデックス
        num_edges = np.random.randint(num_nodes, num_nodes * 3)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        # バッチインデックス
        batch = torch.zeros(num_nodes, dtype=torch.long)
        
        graph_data.append({
            'x': node_features,
            'edge_index': edge_index,
            'batch': batch
        })
    
    logger.info(f"Graph data created: {len(graph_data)} graphs")
    return graph_data


def demonstrate_transformer_models():
    """Transformerモデルのデモンストレーション"""
    print("\n" + "="*60)
    print("🤖 Transformer Models Demo")
    print("="*60)
    
    # データ作成
    features, targets = create_synthetic_data(500)
    
    # データ分割
    train_size = int(0.8 * len(features))
    train_features = features[:train_size]
    train_targets = targets[:train_size]
    test_features = features[train_size:]
    test_targets = targets[train_size:]
    
    # 1. 標準Transformer
    print("\n🔹 Standard Transformer:")
    standard_transformer = TransformerRegressor(
        input_dim=2279,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        num_targets=13,
        use_pwa_pet=False,
        dropout=0.1
    )
    
    # 訓練
    criterion = nn.MSELoss()
    optimizer = optim.Adam(standard_transformer.parameters(), lr=1e-4)
    
    start_time = time.time()
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = standard_transformer(train_features)
        loss = criterion(outputs, train_targets)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: Loss = {loss.item():.4f}")
    
    training_time = time.time() - start_time
    
    # 評価
    with torch.no_grad():
        test_outputs = standard_transformer(test_features)
        test_loss = criterion(test_outputs, test_targets)
    
    print(f"  Training time: {training_time:.2f}s")
    print(f"  Test loss: {test_loss.item():.4f}")
    
    # 2. PWA+PET Transformer
    print("\n🔹 PWA+PET Transformer:")
    pwa_pet_transformer = TransformerRegressor(
        input_dim=2279,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        num_targets=13,
        use_pwa_pet=True,
        buckets={"trivial": 2, "fund": 4, "adj": 2},
        use_rope=True,
        use_pet=True,
        pet_curv_reg=1e-5,
        dropout=0.1
    )
    
    # 訓練
    optimizer = optim.Adam(pwa_pet_transformer.parameters(), lr=1e-4)
    
    start_time = time.time()
    for epoch in range(10):
        optimizer.zero_grad()
        outputs, reg_loss = pwa_pet_transformer(train_features)
        loss = criterion(outputs, train_targets) + reg_loss
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: Loss = {loss.item():.4f}")
    
    training_time = time.time() - start_time
    
    # 評価
    with torch.no_grad():
        test_outputs, _ = pwa_pet_transformer(test_features)
        test_loss = criterion(test_outputs, test_targets)
    
    print(f"  Training time: {training_time:.2f}s")
    print(f"  Test loss: {test_loss.item():.4f}")
    
    return standard_transformer, pwa_pet_transformer


def demonstrate_gnn_models():
    """GNNモデルのデモンストレーション"""
    print("\n" + "="*60)
    print("🕸️ GNN Models Demo")
    print("="*60)
    
    # グラフデータ作成
    graph_data = create_synthetic_graph_data(200)
    
    # データ分割
    train_size = int(0.8 * len(graph_data))
    train_data = graph_data[:train_size]
    test_data = graph_data[train_size:]
    
    # 1. 標準GNN
    print("\n🔹 Standard GNN:")
    standard_gnn = GNNRegressor(
        input_dim=100,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        num_targets=13,
        gnn_type="gcn",
        dropout=0.1
    )
    
    # 訓練
    criterion = nn.MSELoss()
    optimizer = optim.Adam(standard_gnn.parameters(), lr=1e-4)
    
    start_time = time.time()
    for epoch in range(10):
        total_loss = 0
        for graph in train_data:
            optimizer.zero_grad()
            outputs = standard_gnn(graph['x'], graph['edge_index'], graph['batch'])
            targets = torch.randn(1, 13)  # ダミーターゲット
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: Loss = {total_loss/len(train_data):.4f}")
    
    training_time = time.time() - start_time
    print(f"  Training time: {training_time:.2f}s")
    
    # 2. GAT
    print("\n🔹 GAT (Graph Attention Network):")
    gat_gnn = GNNRegressor(
        input_dim=100,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        num_targets=13,
        gnn_type="gat",
        dropout=0.1
    )
    
    # 訓練
    optimizer = optim.Adam(gat_gnn.parameters(), lr=1e-4)
    
    start_time = time.time()
    for epoch in range(10):
        total_loss = 0
        for graph in train_data:
            optimizer.zero_grad()
            outputs = gat_gnn(graph['x'], graph['edge_index'], graph['batch'])
            targets = torch.randn(1, 13)  # ダミーターゲット
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: Loss = {total_loss/len(train_data):.4f}")
    
    training_time = time.time() - start_time
    print(f"  Training time: {training_time:.2f}s")
    
    # 3. マルチスケールGNN
    print("\n🔹 Multi-Scale GNN:")
    multiscale_gnn = MultiScaleGNN(
        input_dim=100,
        hidden_dim=128,
        num_layers=3,
        num_targets=13,
        scales=[1, 2, 3],
        dropout=0.1
    )
    
    # 訓練
    optimizer = optim.Adam(multiscale_gnn.parameters(), lr=1e-4)
    
    start_time = time.time()
    for epoch in range(10):
        total_loss = 0
        for graph in train_data:
            optimizer.zero_grad()
            outputs = multiscale_gnn(graph['x'], graph['edge_index'], graph['batch'])
            targets = torch.randn(1, 13)  # ダミーターゲット
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: Loss = {total_loss/len(train_data):.4f}")
    
    training_time = time.time() - start_time
    print(f"  Training time: {training_time:.2f}s")
    
    return standard_gnn, gat_gnn, multiscale_gnn


def demonstrate_ensemble_models():
    """アンサンブルモデルのデモンストレーション"""
    print("\n" + "="*60)
    print("🎯 Ensemble Models Demo")
    print("="*60)
    
    # データ作成
    features, targets = create_synthetic_data(500)
    
    # データ分割
    train_size = int(0.8 * len(features))
    train_features = features[:train_size]
    train_targets = targets[:train_size]
    test_features = features[train_size:]
    test_targets = targets[train_size:]
    
    # 個別モデルを作成
    transformer = TransformerRegressor(
        input_dim=2279,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        num_targets=13,
        use_pwa_pet=False,
        dropout=0.1
    )
    
    # 1. 重み付き平均アンサンブル
    print("\n🔹 Weighted Average Ensemble:")
    weighted_ensemble = EnsembleRegressor(
        models=[transformer, transformer],  # 同じモデルを2つ使用
        ensemble_method="weighted_average",
        weights=[0.6, 0.4]
    )
    
    # 訓練
    criterion = nn.MSELoss()
    optimizer = optim.Adam(weighted_ensemble.parameters(), lr=1e-4)
    
    start_time = time.time()
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = weighted_ensemble(train_features)
        loss = criterion(outputs, train_targets)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: Loss = {loss.item():.4f}")
    
    training_time = time.time() - start_time
    
    # 評価
    with torch.no_grad():
        test_outputs = weighted_ensemble(test_features)
        test_loss = criterion(test_outputs, test_targets)
    
    print(f"  Training time: {training_time:.2f}s")
    print(f"  Test loss: {test_loss.item():.4f}")
    
    # 2. スタッキングアンサンブル
    print("\n🔹 Stacking Ensemble:")
    stacking_ensemble = EnsembleRegressor(
        models=[transformer, transformer],
        ensemble_method="stacking",
        use_meta_learning=True,
        meta_hidden_dim=128
    )
    
    # 訓練
    optimizer = optim.Adam(stacking_ensemble.parameters(), lr=1e-4)
    
    start_time = time.time()
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = stacking_ensemble(train_features)
        loss = criterion(outputs, train_targets)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: Loss = {loss.item():.4f}")
    
    training_time = time.time() - start_time
    
    # 評価
    with torch.no_grad():
        test_outputs = stacking_ensemble(test_features)
        test_loss = criterion(test_outputs, test_targets)
    
    print(f"  Training time: {training_time:.2f}s")
    print(f"  Test loss: {test_loss.item():.4f}")
    
    # 3. 適応的アンサンブル
    print("\n🔹 Adaptive Ensemble:")
    adaptive_ensemble = AdaptiveEnsemble(
        models=[transformer, transformer],
        input_dim=2279,
        hidden_dim=128,
        num_targets=13
    )
    
    # 訓練
    optimizer = optim.Adam(adaptive_ensemble.parameters(), lr=1e-4)
    
    start_time = time.time()
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = adaptive_ensemble(train_features)
        loss = criterion(outputs, train_targets)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: Loss = {loss.item():.4f}")
    
    training_time = time.time() - start_time
    
    # 評価
    with torch.no_grad():
        test_outputs = adaptive_ensemble(test_features)
        test_loss = criterion(test_outputs, test_targets)
    
    print(f"  Training time: {training_time:.2f}s")
    print(f"  Test loss: {test_loss.item():.4f}")
    
    # 適応的重みを表示
    with torch.no_grad():
        weights = adaptive_ensemble.get_adaptive_weights(test_features[:5])
        print(f"  Adaptive weights (first 5 samples):")
        for i, weight in enumerate(weights):
            print(f"    Sample {i}: {weight.numpy()}")
    
    return weighted_ensemble, stacking_ensemble, adaptive_ensemble


def demonstrate_model_factory():
    """モデルファクトリーのデモンストレーション"""
    print("\n" + "="*60)
    print("🏭 Model Factory Demo")
    print("="*60)
    
    # モデルファクトリーを初期化
    factory = ModelFactory()
    
    # 1. 設定からモデルを作成
    print("\n🔹 Creating models from config:")
    
    # Transformerモデル
    transformer = factory.create_transformer(
        input_dim=2279,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        num_targets=13,
        use_pwa_pet=True
    )
    
    print(f"  Transformer created: {type(transformer).__name__}")
    
    # GNNモデル
    gnn = factory.create_gnn(
        input_dim=100,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        num_targets=13,
        gnn_type="gat"
    )
    
    print(f"  GNN created: {type(gnn).__name__}")
    
    # 2. モデル情報を取得
    print("\n🔹 Model information:")
    
    transformer_info = factory.get_model_info(transformer)
    print(f"  Transformer:")
    print(f"    Parameters: {transformer_info['num_parameters']:,}")
    print(f"    Trainable: {transformer_info['trainable_parameters']:,}")
    print(f"    Size: {transformer_info['model_size_mb']:.2f} MB")
    
    gnn_info = factory.get_model_info(gnn)
    print(f"  GNN:")
    print(f"    Parameters: {gnn_info['num_parameters']:,}")
    print(f"    Trainable: {gnn_info['trainable_parameters']:,}")
    print(f"    Size: {gnn_info['model_size_mb']:.2f} MB")
    
    # 3. アンサンブルモデルを作成
    print("\n🔹 Creating ensemble model:")
    
    ensemble = factory.create_ensemble(
        models=[transformer, transformer],
        ensemble_method="weighted_average",
        weights=[0.6, 0.4]
    )
    
    print(f"  Ensemble created: {type(ensemble).__name__}")
    
    # 4. モデル設定を保存
    print("\n🔹 Saving model config:")
    
    factory.save_model_config(transformer, "transformer_config.yaml")
    factory.save_model_config(gnn, "gnn_config.yaml")
    factory.save_model_config(ensemble, "ensemble_config.yaml")
    
    print("  Model configs saved successfully")
    
    return factory, transformer, gnn, ensemble


def plot_model_comparison(models: Dict[str, torch.nn.Module], test_features: torch.Tensor, test_targets: torch.Tensor):
    """
    モデル比較をプロット
    
    Args:
        models: モデル辞書
        test_features: テスト特徴量
        test_targets: テストターゲット
    """
    print("\n" + "="*60)
    print("📊 Model Comparison Visualization")
    print("="*60)
    
    # 予測を取得
    predictions = {}
    for name, model in models.items():
        with torch.no_grad():
            if hasattr(model, 'forward'):
                pred = model(test_features)
                if isinstance(pred, tuple):
                    pred = pred[0]
                predictions[name] = pred
            else:
                predictions[name] = test_targets  # ダミー
    
    # 予測 vs 実際の値をプロット
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    model_names = list(predictions.keys())
    for i, (name, pred) in enumerate(predictions.items()):
        if i >= 4:
            break
        
        ax = axes[i]
        
        # 最初のターゲットのみをプロット
        true_values = test_targets[:, 0].numpy()
        pred_values = pred[:, 0].numpy()
        
        ax.scatter(true_values, pred_values, alpha=0.6, s=20)
        ax.plot([6, 10], [6, 10], 'r--', alpha=0.8)
        ax.set_xlabel(f"True Target 0")
        ax.set_ylabel(f"Predicted Target 0")
        ax.set_title(f"{name}")
        ax.grid(True, alpha=0.3)
        
        # R²スコアを計算
        r2 = 1 - np.sum((pred_values - true_values) ** 2) / np.sum((true_values - np.mean(true_values)) ** 2)
        ax.text(0.05, 0.95, f"R² = {r2:.3f}", transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 余分なサブプロットを非表示
    for i in range(len(model_names), 4):
        axes[i].set_visible(False)
    
    plt.suptitle("Model Comparison - Prediction vs True Values", fontsize=16)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """
    メイン実行関数
    """
    print("🧬 ChemForge Deep Learning Models Demo")
    print("="*60)
    
    try:
        # 1. Transformerモデルデモ
        standard_transformer, pwa_pet_transformer = demonstrate_transformer_models()
        
        # 2. GNNモデルデモ
        standard_gnn, gat_gnn, multiscale_gnn = demonstrate_gnn_models()
        
        # 3. アンサンブルモデルデモ
        weighted_ensemble, stacking_ensemble, adaptive_ensemble = demonstrate_ensemble_models()
        
        # 4. モデルファクトリーデモ
        factory, transformer, gnn, ensemble = demonstrate_model_factory()
        
        # 5. モデル比較可視化
        test_features, test_targets = create_synthetic_data(100)
        models = {
            "Standard Transformer": standard_transformer,
            "PWA+PET Transformer": pwa_pet_transformer,
            "Weighted Ensemble": weighted_ensemble,
            "Adaptive Ensemble": adaptive_ensemble
        }
        
        plot_model_comparison(models, test_features, test_targets)
        
        print("\n🎉 All demonstrations completed successfully!")
        print("ChemForge deep learning models are ready for use!")
        
    except Exception as e:
        logger.error(f"Error in demo: {str(e)}")
        raise


if __name__ == "__main__":
    main()
