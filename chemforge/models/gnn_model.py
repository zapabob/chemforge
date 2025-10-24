"""
GNN Model Module

Graph Neural Network (GNN) モデル実装
分子グラフ・原子グラフ・結合グラフの統合処理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class GNNRegressor(nn.Module):
    """
    Graph Neural Network 回帰モデル
    
    分子グラフ・原子グラフ・結合グラフの統合処理
    複数のGNNアーキテクチャをサポート
    """
    
    def __init__(
        self,
        input_dim: int = 2279,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        num_targets: int = 13,
        gnn_type: str = "gcn",
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        use_residual: bool = True,
        use_attention: bool = True,
        use_global_pooling: str = "mean"
    ):
        """
        GNN回帰モデルを初期化
        
        Args:
            input_dim: 入力次元
            hidden_dim: 隠れ次元
            num_layers: レイヤー数
            num_heads: アテンションヘッド数
            num_targets: ターゲット数
            gnn_type: GNNタイプ ("gcn", "gat", "gin", "sage")
            dropout: ドロップアウト率
            use_batch_norm: バッチ正規化を使用するか
            use_residual: 残差接続を使用するか
            use_attention: アテンションを使用するか
            use_global_pooling: グローバルプーリング ("mean", "max", "add")
        """
        super(GNNRegressor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_targets = num_targets
        self.gnn_type = gnn_type
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        self.use_attention = use_attention
        self.use_global_pooling = use_global_pooling
        
        # 入力投影層
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # GNN層
        self.gnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(num_layers):
            if gnn_type == "gcn":
                layer = GCNConv(hidden_dim, hidden_dim)
            elif gnn_type == "gat":
                layer = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
            elif gnn_type == "gin":
                layer = GINConv(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                ))
            elif gnn_type == "sage":
                layer = SAGEConv(hidden_dim, hidden_dim)
            else:
                raise ValueError(f"Unsupported GNN type: {gnn_type}")
            
            self.gnn_layers.append(layer)
            
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
            self.dropouts.append(nn.Dropout(dropout))
        
        # アテンション層
        if use_attention:
            self.attention = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
        
        # グローバルプーリング
        self.global_pooling = use_global_pooling
        
        # 出力層
        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_targets)
        ])
        
        logger.info(f"GNNRegressor initialized: {gnn_type}, {num_layers} layers, {hidden_dim} hidden dim")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        """
        フォワードパス
        
        Args:
            x: ノード特徴量 [num_nodes, input_dim]
            edge_index: エッジインデックス [2, num_edges]
            batch: バッチインデックス [num_nodes]
        
        Returns:
            予測値 [batch_size, num_targets]
        """
        # 入力投影
        x = self.input_projection(x)
        
        # GNN層
        for i, (gnn_layer, dropout) in enumerate(zip(self.gnn_layers, self.dropouts)):
            residual = x
            
            # GNN層
            x = gnn_layer(x, edge_index)
            
            # バッチ正規化
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            # 活性化関数
            x = F.relu(x)
            
            # ドロップアウト
            x = dropout(x)
            
            # 残差接続
            if self.use_residual and x.size() == residual.size():
                x = x + residual
        
        # アテンション
        if self.use_attention and batch is not None:
            # バッチ形式に変換
            x_dense, mask = to_dense_batch(x, batch)
            
            # アテンション適用
            x_attended, _ = self.attention(x_dense, x_dense, x_dense, key_padding_mask=~mask)
            
            # マスクを適用
            x_attended = x_attended * mask.unsqueeze(-1)
            
            # 元の形式に戻す
            x = x_attended[mask]
        
        # グローバルプーリング
        if batch is not None:
            if self.global_pooling == "mean":
                x = global_mean_pool(x, batch)
            elif self.global_pooling == "max":
                x = global_max_pool(x, batch)
            elif self.global_pooling == "add":
                x = global_add_pool(x, batch)
            else:
                raise ValueError(f"Unsupported global pooling: {self.global_pooling}")
        else:
            # バッチがない場合は平均プーリング
            x = x.mean(dim=0, keepdim=True)
        
        # 出力層
        for layer in self.output_layers:
            x = layer(x)
        
        return x
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        """
        埋め込み表現を取得
        
        Args:
            x: ノード特徴量
            edge_index: エッジインデックス
            batch: バッチインデックス
        
        Returns:
            埋め込み表現
        """
        # 入力投影
        x = self.input_projection(x)
        
        # GNN層
        for i, (gnn_layer, dropout) in enumerate(zip(self.gnn_layers, self.dropouts)):
            residual = x
            
            # GNN層
            x = gnn_layer(x, edge_index)
            
            # バッチ正規化
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            # 活性化関数
            x = F.relu(x)
            
            # ドロップアウト
            x = dropout(x)
            
            # 残差接続
            if self.use_residual and x.size() == residual.size():
                x = x + residual
        
        # グローバルプーリング
        if batch is not None:
            if self.global_pooling == "mean":
                x = global_mean_pool(x, batch)
            elif self.global_pooling == "max":
                x = global_max_pool(x, batch)
            elif self.global_pooling == "add":
                x = global_add_pool(x, batch)
        
        return x


class MolecularGNN(nn.Module):
    """
    分子特化GNNモデル
    
    分子グラフの特性を考慮したGNN実装
    原子・結合・環の統合処理
    """
    
    def __init__(
        self,
        atom_dim: int = 100,
        bond_dim: int = 50,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_targets: int = 13,
        gnn_type: str = "gat",
        dropout: float = 0.1
    ):
        """
        分子特化GNNを初期化
        
        Args:
            atom_dim: 原子特徴量次元
            bond_dim: 結合特徴量次元
            hidden_dim: 隠れ次元
            num_layers: レイヤー数
            num_targets: ターゲット数
            gnn_type: GNNタイプ
            dropout: ドロップアウト率
        """
        super(MolecularGNN, self).__init__()
        
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_targets = num_targets
        self.gnn_type = gnn_type
        self.dropout = dropout
        
        # 原子・結合特徴量投影
        self.atom_projection = nn.Linear(atom_dim, hidden_dim)
        self.bond_projection = nn.Linear(bond_dim, hidden_dim)
        
        # GNN層
        self.gnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(num_layers):
            if gnn_type == "gcn":
                layer = GCNConv(hidden_dim, hidden_dim)
            elif gnn_type == "gat":
                layer = GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=dropout)
            elif gnn_type == "gin":
                layer = GINConv(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                ))
            elif gnn_type == "sage":
                layer = SAGEConv(hidden_dim, hidden_dim)
            else:
                raise ValueError(f"Unsupported GNN type: {gnn_type}")
            
            self.gnn_layers.append(layer)
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
        
        # 出力層
        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_targets)
        ])
        
        logger.info(f"MolecularGNN initialized: {gnn_type}, {num_layers} layers")
    
    def forward(
        self,
        atom_features: torch.Tensor,
        bond_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor = None
    ) -> torch.Tensor:
        """
        フォワードパス
        
        Args:
            atom_features: 原子特徴量
            bond_features: 結合特徴量
            edge_index: エッジインデックス
            batch: バッチインデックス
        
        Returns:
            予測値
        """
        # 特徴量投影
        x = self.atom_projection(atom_features)
        
        # GNN層
        for i, (gnn_layer, batch_norm, dropout) in enumerate(zip(self.gnn_layers, self.batch_norms, self.dropouts)):
            # GNN層
            x = gnn_layer(x, edge_index)
            
            # バッチ正規化
            x = batch_norm(x)
            
            # 活性化関数
            x = F.relu(x)
            
            # ドロップアウト
            x = dropout(x)
        
        # グローバルプーリング
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        # 出力層
        for layer in self.output_layers:
            x = layer(x)
        
        return x


class MultiScaleGNN(nn.Module):
    """
    マルチスケールGNNモデル
    
    複数のスケールで分子グラフを処理
    局所・中域・大域の特徴量を統合
    """
    
    def __init__(
        self,
        input_dim: int = 2279,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_targets: int = 13,
        scales: List[int] = [1, 2, 3],
        dropout: float = 0.1
    ):
        """
        マルチスケールGNNを初期化
        
        Args:
            input_dim: 入力次元
            hidden_dim: 隠れ次元
            num_layers: レイヤー数
            num_targets: ターゲット数
            scales: スケールリスト
            dropout: ドロップアウト率
        """
        super(MultiScaleGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_targets = num_targets
        self.scales = scales
        self.dropout = dropout
        
        # 入力投影
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # スケール別GNN
        self.scale_gnns = nn.ModuleDict()
        for scale in scales:
            self.scale_gnns[str(scale)] = nn.ModuleList([
                GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
            ])
        
        # 特徴量統合
        self.feature_fusion = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        
        # 出力層
        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_dim * len(scales), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_targets)
        ])
        
        logger.info(f"MultiScaleGNN initialized: {len(scales)} scales, {num_layers} layers")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        """
        フォワードパス
        
        Args:
            x: ノード特徴量
            edge_index: エッジインデックス
            batch: バッチインデックス
        
        Returns:
            予測値
        """
        # 入力投影
        x = self.input_projection(x)
        
        # スケール別処理
        scale_features = []
        
        for scale in self.scales:
            scale_x = x
            
            # スケール別GNN
            for layer in self.scale_gnns[str(scale)]:
                scale_x = layer(scale_x, edge_index)
                scale_x = F.relu(scale_x)
                scale_x = F.dropout(scale_x, p=self.dropout, training=self.training)
            
            # グローバルプーリング
            if batch is not None:
                scale_x = global_mean_pool(scale_x, batch)
            else:
                scale_x = scale_x.mean(dim=0, keepdim=True)
            
            scale_features.append(scale_x)
        
        # 特徴量統合
        if len(scale_features) > 1:
            # マルチヘッドアテンション
            combined_features = torch.stack(scale_features, dim=1)
            attended_features, _ = self.feature_fusion(combined_features, combined_features, combined_features)
            attended_features = attended_features.mean(dim=1)
        else:
            attended_features = scale_features[0]
        
        # 出力層
        for layer in self.output_layers:
            attended_features = layer(attended_features)
        
        return attended_features
