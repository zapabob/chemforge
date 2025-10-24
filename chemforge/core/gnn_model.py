"""
GNN回帰モデル

CNS創薬向けのGraph Neural Network回帰モデル。
分子グラフと骨格特徴量を統合して、13個のCNSターゲットのpIC50を予測。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import math


class GraphConvolution(nn.Module):
    """グラフ畳み込み層"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """パラメータを初期化"""
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        順伝播
        
        Args:
            x: ノード特徴量 (batch_size, num_nodes, in_features)
            adj: 隣接行列 (batch_size, num_nodes, num_nodes)
            
        Returns:
            出力特徴量 (batch_size, num_nodes, out_features)
        """
        # 線形変換
        support = torch.matmul(x, self.weight)
        
        # グラフ畳み込み
        output = torch.matmul(adj, support)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class GraphAttentionLayer(nn.Module):
    """グラフアテンション層"""
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1, alpha: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout_layer = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """パラメータを初期化"""
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.weight)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        順伝播
        
        Args:
            x: ノード特徴量 (batch_size, num_nodes, in_features)
            adj: 隣接行列 (batch_size, num_nodes, num_nodes)
            
        Returns:
            出力特徴量 (batch_size, num_nodes, out_features)
        """
        batch_size, num_nodes, _ = x.size()
        
        # 線形変換
        h = self.W(x)  # (batch_size, num_nodes, out_features)
        
        # アテンション計算
        a_input = self._prepare_attentional_mechanism_input(h)
        e = self.leakyrelu(self.a(a_input))  # (batch_size, num_nodes, num_nodes)
        
        # 隣接行列でマスク
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout_layer(attention)
        
        # アテンション重みを適用
        h_prime = torch.matmul(attention, h)
        
        return h_prime
    
    def _prepare_attentional_mechanism_input(self, h: torch.Tensor) -> torch.Tensor:
        """アテンション機構の入力を準備"""
        batch_size, num_nodes, out_features = h.size()
        
        # 全ノードペアの特徴量を結合
        h_repeated_in_chunks = h.repeat_interleave(num_nodes, dim=1)
        h_repeated_alternating = h.repeat(1, num_nodes, 1)
        
        all_combinations_matrix = torch.cat([
            h_repeated_in_chunks, h_repeated_alternating
        ], dim=-1)
        
        return all_combinations_matrix.view(batch_size, num_nodes, num_nodes, 2 * out_features)


class GNNRegressor(nn.Module):
    """GNN回帰モデル"""
    
    def __init__(
        self,
        node_features: int = 78,
        edge_features: int = 12,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_targets: int = 13,
        dropout: float = 0.1,
        use_attention: bool = True,
        use_scaffold_features: bool = True,
        scaffold_dim: int = 20
    ):
        """
        GNN回帰モデルを初期化
        
        Args:
            node_features: ノード特徴量次元
            edge_features: エッジ特徴量次元
            hidden_dim: 隠れ層次元
            num_layers: GNN層数
            num_targets: 予測ターゲット数
            dropout: ドロップアウト率
            use_attention: アテンション機構を使用するか
            use_scaffold_features: 骨格特徴量を使用するか
            scaffold_dim: 骨格特徴量次元
        """
        super().__init__()
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_targets = num_targets
        self.use_attention = use_attention
        self.use_scaffold_features = use_scaffold_features
        self.scaffold_dim = scaffold_dim
        
        # ノード特徴量投影
        self.node_projection = nn.Linear(node_features, hidden_dim)
        
        # エッジ特徴量投影
        self.edge_projection = nn.Linear(edge_features, hidden_dim)
        
        # GNN層
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            if use_attention:
                layer = GraphAttentionLayer(hidden_dim, hidden_dim, dropout)
            else:
                layer = GraphConvolution(hidden_dim, hidden_dim)
            self.gnn_layers.append(layer)
        
        # 骨格特徴量投影
        if use_scaffold_features:
            self.scaffold_projection = nn.Linear(scaffold_dim, hidden_dim)
        
        # グラフプーリング
        self.graph_pooling = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 出力層
        if use_scaffold_features:
            output_dim = hidden_dim // 4 + hidden_dim // 4  # グラフ特徴量 + 骨格特徴量
        else:
            output_dim = hidden_dim // 4
        
        self.output_layer = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_targets)
        )
        
        # ドロップアウト
        self.dropout = nn.Dropout(dropout)
        
        # 重み初期化
        self._init_weights()
    
    def _init_weights(self):
        """重みを初期化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        adj_matrix: torch.Tensor,
        scaffold_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        順伝播
        
        Args:
            node_features: ノード特徴量 (batch_size, num_nodes, node_features)
            edge_features: エッジ特徴量 (batch_size, num_nodes, num_nodes, edge_features)
            adj_matrix: 隣接行列 (batch_size, num_nodes, num_nodes)
            scaffold_features: 骨格特徴量 (batch_size, scaffold_dim)
            
        Returns:
            予測値 (batch_size, num_targets)
        """
        batch_size, num_nodes, _ = node_features.size()
        
        # ノード特徴量を投影
        x = self.node_projection(node_features)  # (batch_size, num_nodes, hidden_dim)
        
        # GNN層を適用
        for layer in self.gnn_layers:
            if self.use_attention:
                x = layer(x, adj_matrix)
            else:
                x = layer(x, adj_matrix)
            x = F.relu(x)
            x = self.dropout(x)
        
        # グラフプーリング（平均プーリング）
        graph_features = x.mean(dim=1)  # (batch_size, hidden_dim)
        graph_features = self.graph_pooling(graph_features)  # (batch_size, hidden_dim // 4)
        
        # 骨格特徴量を統合
        if self.use_scaffold_features and scaffold_features is not None:
            scaffold_proj = self.scaffold_projection(scaffold_features)  # (batch_size, hidden_dim)
            scaffold_pooled = self.graph_pooling(scaffold_proj)  # (batch_size, hidden_dim // 4)
            
            # 特徴量を結合
            combined_features = torch.cat([graph_features, scaffold_pooled], dim=-1)
        else:
            combined_features = graph_features
        
        # 出力層
        output = self.output_layer(combined_features)  # (batch_size, num_targets)
        
        return output
    
    def get_node_embeddings(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        adj_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        ノード埋め込みを取得
        
        Args:
            node_features: ノード特徴量
            edge_features: エッジ特徴量
            adj_matrix: 隣接行列
            
        Returns:
            ノード埋め込み (batch_size, num_nodes, hidden_dim)
        """
        batch_size, num_nodes, _ = node_features.size()
        
        # ノード特徴量を投影
        x = self.node_projection(node_features)
        
        # GNN層を適用
        for layer in self.gnn_layers:
            if self.use_attention:
                x = layer(x, adj_matrix)
            else:
                x = layer(x, adj_matrix)
            x = F.relu(x)
            x = self.dropout(x)
        
        return x
    
    def get_graph_embeddings(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        adj_matrix: torch.Tensor,
        scaffold_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        グラフ埋め込みを取得
        
        Args:
            node_features: ノード特徴量
            edge_features: エッジ特徴量
            adj_matrix: 隣接行列
            scaffold_features: 骨格特徴量
            
        Returns:
            グラフ埋め込み
        """
        batch_size, num_nodes, _ = node_features.size()
        
        # ノード特徴量を投影
        x = self.node_projection(node_features)
        
        # GNN層を適用
        for layer in self.gnn_layers:
            if self.use_attention:
                x = layer(x, adj_matrix)
            else:
                x = layer(x, adj_matrix)
            x = F.relu(x)
            x = self.dropout(x)
        
        # グラフプーリング
        graph_features = x.mean(dim=1)
        graph_features = self.graph_pooling(graph_features)
        
        # 骨格特徴量を統合
        if self.use_scaffold_features and scaffold_features is not None:
            scaffold_proj = self.scaffold_projection(scaffold_features)
            scaffold_pooled = self.graph_pooling(scaffold_proj)
            combined_features = torch.cat([graph_features, scaffold_pooled], dim=-1)
        else:
            combined_features = graph_features
        
        return combined_features
    
    def predict_with_uncertainty(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        adj_matrix: torch.Tensor,
        scaffold_features: Optional[torch.Tensor] = None,
        num_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        不確実性を考慮した予測
        
        Args:
            node_features: ノード特徴量
            edge_features: エッジ特徴量
            adj_matrix: 隣接行列
            scaffold_features: 骨格特徴量
            num_samples: サンプリング回数
            
        Returns:
            予測値と不確実性
        """
        self.eval()
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                # ドロップアウトを有効にして予測
                self.train()
                pred = self.forward(node_features, edge_features, adj_matrix, scaffold_features)
                predictions.append(pred)
                self.eval()
        
        predictions = torch.stack(predictions)  # (num_samples, batch_size, num_targets)
        
        # 平均と標準偏差を計算
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        return mean_pred, std_pred
    
    def save_model(self, path: str):
        """モデルを保存"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'node_features': self.node_features,
            'edge_features': self.edge_features,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_targets': self.num_targets,
            'use_attention': self.use_attention,
            'use_scaffold_features': self.use_scaffold_features,
            'scaffold_dim': self.scaffold_dim
        }, path)
    
    @classmethod
    def load_model(cls, path: str, device: str = 'cpu'):
        """モデルを読み込み"""
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(
            node_features=checkpoint['node_features'],
            edge_features=checkpoint['edge_features'],
            hidden_dim=checkpoint['hidden_dim'],
            num_layers=checkpoint['num_layers'],
            num_targets=checkpoint['num_targets'],
            use_attention=checkpoint['use_attention'],
            use_scaffold_features=checkpoint['use_scaffold_features'],
            scaffold_dim=checkpoint['scaffold_dim']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model


# 便利関数
def create_gnn_model(
    node_features: int = 78,
    edge_features: int = 12,
    hidden_dim: int = 256,
    num_layers: int = 3,
    num_targets: int = 13,
    use_attention: bool = True,
    use_scaffold_features: bool = True
) -> GNNRegressor:
    """GNNモデルを作成"""
    return GNNRegressor(
        node_features=node_features,
        edge_features=edge_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_targets=num_targets,
        use_attention=use_attention,
        use_scaffold_features=use_scaffold_features
    )


def count_parameters(model: nn.Module) -> int:
    """モデルのパラメータ数を計算"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # テスト実行
    print("🧬 GNN回帰モデルテスト")
    print("=" * 50)
    
    # モデルを作成
    model = create_gnn_model(
        node_features=78,
        edge_features=12,
        hidden_dim=256,
        num_layers=3,
        num_targets=13,
        use_attention=True,
        use_scaffold_features=True
    )
    
    print(f"モデルパラメータ数: {count_parameters(model):,}")
    
    # テスト用入力データ
    batch_size = 32
    num_nodes = 50
    node_features = 78
    edge_features = 12
    scaffold_dim = 20
    
    # 入力テンソルを作成
    node_feat = torch.randn(batch_size, num_nodes, node_features)
    edge_feat = torch.randn(batch_size, num_nodes, num_nodes, edge_features)
    adj_matrix = torch.randint(0, 2, (batch_size, num_nodes, num_nodes)).float()
    scaffold_feat = torch.randn(batch_size, scaffold_dim)
    
    print(f"ノード特徴量形状: {node_feat.shape}")
    print(f"エッジ特徴量形状: {edge_feat.shape}")
    print(f"隣接行列形状: {adj_matrix.shape}")
    print(f"骨格特徴量形状: {scaffold_feat.shape}")
    
    # 順伝播
    model.eval()
    with torch.no_grad():
        output = model(node_feat, edge_feat, adj_matrix, scaffold_feat)
        print(f"出力形状: {output.shape}")
        
        # ノード埋め込みを取得
        node_embeddings = model.get_node_embeddings(node_feat, edge_feat, adj_matrix)
        print(f"ノード埋め込み形状: {node_embeddings.shape}")
        
        # グラフ埋め込みを取得
        graph_embeddings = model.get_graph_embeddings(node_feat, edge_feat, adj_matrix, scaffold_feat)
        print(f"グラフ埋め込み形状: {graph_embeddings.shape}")
    
    # 不確実性を考慮した予測
    print("\n不確実性を考慮した予測:")
    mean_pred, std_pred = model.predict_with_uncertainty(
        node_feat, edge_feat, adj_matrix, scaffold_feat, num_samples=10
    )
    print(f"平均予測形状: {mean_pred.shape}")
    print(f"標準偏差形状: {std_pred.shape}")
    
    # モデル保存・読み込みテスト
    print("\nモデル保存・読み込みテスト:")
    model_path = "test_gnn_model.pth"
    model.save_model(model_path)
    
    loaded_model = GNNRegressor.load_model(model_path)
    print(f"読み込み成功: {type(loaded_model)}")
    
    # テストファイルを削除
    import os
    if os.path.exists(model_path):
        os.remove(model_path)
    
    print("\n✅ GNN回帰モデルテスト完了！")
