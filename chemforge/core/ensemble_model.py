"""
アンサンブル回帰モデル

TransformerとGNNを組み合わせたアンサンブルモデル。
不確実性推定機能付きで、13個のCNSターゲットのpIC50を予測。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import math

from .transformer_model import TransformerRegressor
from .gnn_model import GNNRegressor


class EnsembleRegressor(nn.Module):
    """アンサンブル回帰モデル"""
    
    def __init__(
        self,
        transformer_model: Optional[TransformerRegressor] = None,
        gnn_model: Optional[GNNRegressor] = None,
        weights: List[float] = [0.6, 0.4],
        num_targets: int = 13,
        use_uncertainty: bool = True,
        uncertainty_threshold: float = 0.3
    ):
        """
        アンサンブル回帰モデルを初期化
        
        Args:
            transformer_model: Transformerモデル
            gnn_model: GNNモデル
            weights: モデル重み [transformer_weight, gnn_weight]
            num_targets: 予測ターゲット数
            use_uncertainty: 不確実性推定を使用するか
            uncertainty_threshold: 不確実性閾値
        """
        super().__init__()
        
        self.transformer_model = transformer_model
        self.gnn_model = gnn_model
        self.weights = weights
        self.num_targets = num_targets
        self.use_uncertainty = use_uncertainty
        self.uncertainty_threshold = uncertainty_threshold
        
        # 重みを正規化
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.weights = F.softmax(self.weights, dim=0)
        
        # 不確実性推定用の重み
        if use_uncertainty:
            self.uncertainty_weights = nn.Parameter(torch.ones(num_targets))
        
        # モデルが提供されない場合はデフォルトを作成
        if transformer_model is None:
            self.transformer_model = TransformerRegressor(
                input_dim=2279,
                hidden_dim=512,
                num_layers=4,
                num_heads=8,
                num_targets=num_targets
            )
        
        if gnn_model is None:
            self.gnn_model = GNNRegressor(
                node_features=78,
                edge_features=12,
                hidden_dim=256,
                num_layers=3,
                num_targets=num_targets,
                use_attention=True,
                use_scaffold_features=True
            )
    
    def forward(
        self,
        x: torch.Tensor,
        node_features: Optional[torch.Tensor] = None,
        edge_features: Optional[torch.Tensor] = None,
        adj_matrix: Optional[torch.Tensor] = None,
        scaffold_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        順伝播
        
        Args:
            x: 統合特徴量 (batch_size, input_dim)
            node_features: ノード特徴量 (batch_size, num_nodes, node_features)
            edge_features: エッジ特徴量 (batch_size, num_nodes, num_nodes, edge_features)
            adj_matrix: 隣接行列 (batch_size, num_nodes, num_nodes)
            scaffold_features: 骨格特徴量 (batch_size, scaffold_dim)
            
        Returns:
            予測値 (batch_size, num_targets)
        """
        predictions = []
        
        # Transformer予測
        if self.transformer_model is not None:
            transformer_pred = self.transformer_model(x)
            predictions.append(transformer_pred)
        
        # GNN予測
        if self.gnn_model is not None and node_features is not None:
            gnn_pred = self.gnn_model(node_features, edge_features, adj_matrix, scaffold_features)
            predictions.append(gnn_pred)
        
        # 予測を結合
        if len(predictions) == 0:
            raise ValueError("No models available for prediction")
        
        # 重み付き平均
        ensemble_pred = torch.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            ensemble_pred += self.weights[i] * pred
        
        return ensemble_pred
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        node_features: Optional[torch.Tensor] = None,
        edge_features: Optional[torch.Tensor] = None,
        adj_matrix: Optional[torch.Tensor] = None,
        scaffold_features: Optional[torch.Tensor] = None,
        num_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        不確実性を考慮した予測
        
        Args:
            x: 統合特徴量
            node_features: ノード特徴量
            edge_features: エッジ特徴量
            adj_matrix: 隣接行列
            scaffold_features: 骨格特徴量
            num_samples: サンプリング回数
            
        Returns:
            予測値、不確実性、信頼度
        """
        if not self.use_uncertainty:
            pred = self.forward(x, node_features, edge_features, adj_matrix, scaffold_features)
            return pred, torch.zeros_like(pred), torch.ones_like(pred)
        
        # 各モデルの不確実性を計算
        transformer_uncertainty = None
        gnn_uncertainty = None
        
        if self.transformer_model is not None:
            transformer_mean, transformer_std = self.transformer_model.predict_with_uncertainty(
                x, num_samples=num_samples
            )
            transformer_uncertainty = transformer_std
        
        if self.gnn_model is not None and node_features is not None:
            gnn_mean, gnn_std = self.gnn_model.predict_with_uncertainty(
                node_features, edge_features, adj_matrix, scaffold_features, num_samples
            )
            gnn_uncertainty = gnn_std
        
        # アンサンブル予測
        ensemble_pred = self.forward(x, node_features, edge_features, adj_matrix, scaffold_features)
        
        # 不確実性を統合
        if transformer_uncertainty is not None and gnn_uncertainty is not None:
            # 両方のモデルが利用可能
            uncertainty = self.weights[0] * transformer_uncertainty + self.weights[1] * gnn_uncertainty
        elif transformer_uncertainty is not None:
            # Transformerのみ
            uncertainty = transformer_uncertainty
        elif gnn_uncertainty is not None:
            # GNNのみ
            uncertainty = gnn_uncertainty
        else:
            # 不確実性が計算できない場合はデフォルト値
            uncertainty = torch.ones_like(ensemble_pred) * 0.5
        
        # 信頼度を計算
        confidence = torch.exp(-uncertainty * self.uncertainty_weights)
        confidence = torch.clamp(confidence, 0.0, 1.0)
        
        return ensemble_pred, uncertainty, confidence
    
    def get_model_contributions(
        self,
        x: torch.Tensor,
        node_features: Optional[torch.Tensor] = None,
        edge_features: Optional[torch.Tensor] = None,
        adj_matrix: Optional[torch.Tensor] = None,
        scaffold_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        各モデルの寄与度を取得
        
        Args:
            x: 統合特徴量
            node_features: ノード特徴量
            edge_features: エッジ特徴量
            adj_matrix: 隣接行列
            scaffold_features: 骨格特徴量
            
        Returns:
            各モデルの寄与度
        """
        contributions = {}
        
        # Transformer寄与度
        if self.transformer_model is not None:
            transformer_pred = self.transformer_model(x)
            contributions['transformer'] = transformer_pred
        
        # GNN寄与度
        if self.gnn_model is not None and node_features is not None:
            gnn_pred = self.gnn_model(node_features, edge_features, adj_matrix, scaffold_features)
            contributions['gnn'] = gnn_pred
        
        return contributions
    
    def get_feature_importance(
        self,
        x: torch.Tensor,
        node_features: Optional[torch.Tensor] = None,
        edge_features: Optional[torch.Tensor] = None,
        adj_matrix: Optional[torch.Tensor] = None,
        scaffold_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        特徴量重要度を取得
        
        Args:
            x: 統合特徴量
            node_features: ノード特徴量
            edge_features: エッジ特徴量
            adj_matrix: 隣接行列
            scaffold_features: 骨格特徴量
            
        Returns:
            各モデルの特徴量重要度
        """
        importance = {}
        
        # Transformer特徴量重要度
        if self.transformer_model is not None:
            transformer_importance = self.transformer_model.get_feature_importance(x)
            importance['transformer'] = transformer_importance
        
        # GNN特徴量重要度（簡易版）
        if self.gnn_model is not None and node_features is not None:
            # ノード特徴量の重要度を計算
            node_importance = torch.abs(node_features).mean(dim=1).mean(dim=0)
            importance['gnn'] = node_importance
        
        return importance
    
    def get_embeddings(
        self,
        x: torch.Tensor,
        node_features: Optional[torch.Tensor] = None,
        edge_features: Optional[torch.Tensor] = None,
        adj_matrix: Optional[torch.Tensor] = None,
        scaffold_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        埋め込み表現を取得
        
        Args:
            x: 統合特徴量
            node_features: ノード特徴量
            edge_features: エッジ特徴量
            adj_matrix: 隣接行列
            scaffold_features: 骨格特徴量
            
        Returns:
            各モデルの埋め込み表現
        """
        embeddings = {}
        
        # Transformer埋め込み
        if self.transformer_model is not None:
            transformer_emb = self.transformer_model.get_embeddings(x)
            embeddings['transformer'] = transformer_emb
        
        # GNN埋め込み
        if self.gnn_model is not None and node_features is not None:
            gnn_emb = self.gnn_model.get_graph_embeddings(
                node_features, edge_features, adj_matrix, scaffold_features
            )
            embeddings['gnn'] = gnn_emb
        
        return embeddings
    
    def update_weights(self, new_weights: List[float]):
        """モデル重みを更新"""
        self.weights = torch.tensor(new_weights, dtype=torch.float32)
        self.weights = F.softmax(self.weights, dim=0)
    
    def get_uncertainty_threshold(self) -> float:
        """不確実性閾値を取得"""
        return self.uncertainty_threshold
    
    def set_uncertainty_threshold(self, threshold: float):
        """不確実性閾値を設定"""
        self.uncertainty_threshold = threshold
    
    def save_model(self, path: str):
        """モデルを保存"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'weights': self.weights.tolist(),
            'num_targets': self.num_targets,
            'use_uncertainty': self.use_uncertainty,
            'uncertainty_threshold': self.uncertainty_threshold,
            'transformer_model': self.transformer_model.state_dict() if self.transformer_model else None,
            'gnn_model': self.gnn_model.state_dict() if self.gnn_model else None
        }, path)
    
    @classmethod
    def load_model(cls, path: str, device: str = 'cpu'):
        """モデルを読み込み"""
        checkpoint = torch.load(path, map_location=device)
        
        # モデルを作成
        model = cls(
            weights=checkpoint['weights'],
            num_targets=checkpoint['num_targets'],
            use_uncertainty=checkpoint['use_uncertainty'],
            uncertainty_threshold=checkpoint['uncertainty_threshold']
        )
        
        # 状態辞書を読み込み
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # サブモデルの状態辞書を読み込み
        if checkpoint['transformer_model'] is not None:
            model.transformer_model.load_state_dict(checkpoint['transformer_model'])
        
        if checkpoint['gnn_model'] is not None:
            model.gnn_model.load_state_dict(checkpoint['gnn_model'])
        
        model.to(device)
        
        return model


class UncertaintyEstimator(nn.Module):
    """不確実性推定器"""
    
    def __init__(self, input_dim: int, num_targets: int, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.num_targets = num_targets
        self.hidden_dim = hidden_dim
        
        self.uncertainty_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_targets),
            nn.Softplus()  # 正の値を保証
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """不確実性を推定"""
        return self.uncertainty_net(x)


# 便利関数
def create_ensemble_model(
    transformer_model: Optional[TransformerRegressor] = None,
    gnn_model: Optional[GNNRegressor] = None,
    weights: List[float] = [0.6, 0.4],
    num_targets: int = 13,
    use_uncertainty: bool = True
) -> EnsembleRegressor:
    """アンサンブルモデルを作成"""
    return EnsembleRegressor(
        transformer_model=transformer_model,
        gnn_model=gnn_model,
        weights=weights,
        num_targets=num_targets,
        use_uncertainty=use_uncertainty
    )


def count_parameters(model: nn.Module) -> int:
    """モデルのパラメータ数を計算"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # テスト実行
    print("🧬 アンサンブル回帰モデルテスト")
    print("=" * 50)
    
    # サブモデルを作成
    transformer_model = TransformerRegressor(
        input_dim=2279,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        num_targets=13
    )
    
    gnn_model = GNNRegressor(
        node_features=78,
        edge_features=12,
        hidden_dim=256,
        num_layers=3,
        num_targets=13,
        use_attention=True,
        use_scaffold_features=True
    )
    
    # アンサンブルモデルを作成
    ensemble_model = create_ensemble_model(
        transformer_model=transformer_model,
        gnn_model=gnn_model,
        weights=[0.6, 0.4],
        num_targets=13,
        use_uncertainty=True
    )
    
    print(f"アンサンブルモデルパラメータ数: {count_parameters(ensemble_model):,}")
    print(f"Transformerパラメータ数: {count_parameters(transformer_model):,}")
    print(f"GNNパラメータ数: {count_parameters(gnn_model):,}")
    
    # テスト用入力データ
    batch_size = 32
    num_nodes = 50
    
    # 統合特徴量
    x = torch.randn(batch_size, 2279)
    
    # グラフ特徴量
    node_features = torch.randn(batch_size, num_nodes, 78)
    edge_features = torch.randn(batch_size, num_nodes, num_nodes, 12)
    adj_matrix = torch.randint(0, 2, (batch_size, num_nodes, num_nodes)).float()
    scaffold_features = torch.randn(batch_size, 20)
    
    print(f"統合特徴量形状: {x.shape}")
    print(f"ノード特徴量形状: {node_features.shape}")
    print(f"エッジ特徴量形状: {edge_features.shape}")
    print(f"隣接行列形状: {adj_matrix.shape}")
    print(f"骨格特徴量形状: {scaffold_features.shape}")
    
    # 順伝播
    ensemble_model.eval()
    with torch.no_grad():
        output = ensemble_model(x, node_features, edge_features, adj_matrix, scaffold_features)
        print(f"出力形状: {output.shape}")
        
        # 各モデルの寄与度を取得
        contributions = ensemble_model.get_model_contributions(
            x, node_features, edge_features, adj_matrix, scaffold_features
        )
        print(f"寄与度: {list(contributions.keys())}")
        
        # 特徴量重要度を取得
        importance = ensemble_model.get_feature_importance(
            x, node_features, edge_features, adj_matrix, scaffold_features
        )
        print(f"特徴量重要度: {list(importance.keys())}")
        
        # 埋め込み表現を取得
        embeddings = ensemble_model.get_embeddings(
            x, node_features, edge_features, adj_matrix, scaffold_features
        )
        print(f"埋め込み表現: {list(embeddings.keys())}")
    
    # 不確実性を考慮した予測
    print("\n不確実性を考慮した予測:")
    mean_pred, uncertainty, confidence = ensemble_model.predict_with_uncertainty(
        x, node_features, edge_features, adj_matrix, scaffold_features, num_samples=10
    )
    print(f"平均予測形状: {mean_pred.shape}")
    print(f"不確実性形状: {uncertainty.shape}")
    print(f"信頼度形状: {confidence.shape}")
    
    # 重みを更新
    print("\n重み更新テスト:")
    ensemble_model.update_weights([0.7, 0.3])
    print(f"新しい重み: {ensemble_model.weights}")
    
    # モデル保存・読み込みテスト
    print("\nモデル保存・読み込みテスト:")
    model_path = "test_ensemble_model.pth"
    ensemble_model.save_model(model_path)
    
    loaded_model = EnsembleRegressor.load_model(model_path)
    print(f"読み込み成功: {type(loaded_model)}")
    
    # テストファイルを削除
    import os
    if os.path.exists(model_path):
        os.remove(model_path)
    
    print("\n✅ アンサンブル回帰モデルテスト完了！")
