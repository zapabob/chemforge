"""
Graph Neural Network Module

GNN深層学習モデル実装
既存GraphNeuralNetworkを活用した効率的なGNN実装
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from pathlib import Path
import json
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 既存モジュール活用
from chemforge.core.graph_neural_network import GraphNeuralNetwork
from chemforge.utils.config_utils import ConfigManager
from chemforge.utils.logging_utils import Logger
from chemforge.utils.validation import DataValidator

logger = logging.getLogger(__name__)

class GCNLayer(nn.Module):
    """
    Graph Convolutional Network Layer
    
    既存GraphNeuralNetworkを活用したGCN Layer
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 activation: str = "relu", dropout: float = 0.1,
                 use_bias: bool = True):
        """
        初期化
        
        Args:
            in_features: 入力特徴量数
            out_features: 出力特徴量数
            activation: 活性化関数
            dropout: ドロップアウト率
            use_bias: バイアス使用フラグ
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.dropout = dropout
        self.use_bias = use_bias
        
        # 重み行列
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        # バイアス
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # ドロップアウト
        self.dropout = nn.Dropout(dropout)
        
        # 重み初期化
        self._init_weights()
    
    def _init_weights(self):
        """重み初期化"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def _get_activation(self):
        """活性化関数取得"""
        if self.activation == "relu":
            return F.relu
        elif self.activation == "gelu":
            return F.gelu
        elif self.activation == "swish":
            return F.silu
        else:
            return F.relu
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        フォワードパス
        
        Args:
            x: ノード特徴量 [batch_size, num_nodes, in_features]
            adj: 隣接行列 [batch_size, num_nodes, num_nodes]
            
        Returns:
            出力特徴量 [batch_size, num_nodes, out_features]
        """
        # 線形変換
        support = torch.matmul(x, self.weight)
        
        # グラフ畳み込み
        output = torch.matmul(adj, support)
        
        # バイアス追加
        if self.bias is not None:
            output = output + self.bias
        
        # 活性化関数
        output = self._get_activation()(output)
        
        # ドロップアウト
        output = self.dropout(output)
        
        return output

class GATLayer(nn.Module):
    """
    Graph Attention Network Layer
    
    既存GraphNeuralNetworkを活用したGAT Layer
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 n_heads: int = 1, dropout: float = 0.1,
                 alpha: float = 0.2, concat: bool = True):
        """
        初期化
        
        Args:
            in_features: 入力特徴量数
            out_features: 出力特徴量数
            n_heads: アテンションヘッド数
            dropout: ドロップアウト率
            alpha: LeakyReLUの負の傾き
            concat: ヘッド結合フラグ
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # 重み行列
        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features * n_heads))
        
        # アテンション重み
        self.a = nn.Parameter(torch.FloatTensor(2 * out_features, 1))
        
        # ドロップアウト
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
        
        # 重み初期化
        self._init_weights()
    
    def _init_weights(self):
        """重み初期化"""
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        フォワードパス
        
        Args:
            x: ノード特徴量 [batch_size, num_nodes, in_features]
            adj: 隣接行列 [batch_size, num_nodes, num_nodes]
            
        Returns:
            出力特徴量 [batch_size, num_nodes, out_features]
        """
        batch_size, num_nodes, _ = x.shape
        
        # 線形変換
        h = torch.matmul(x, self.W)  # [batch_size, num_nodes, out_features * n_heads]
        h = h.view(batch_size, num_nodes, self.n_heads, self.out_features)
        
        # アテンション計算
        a_input = self._prepare_attentional_mechanism_input(h)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))
        
        # マスク適用
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        
        # アテンション適用
        h_prime = torch.matmul(attention.unsqueeze(2), h).squeeze(2)
        
        # ヘッド結合
        if self.concat:
            return h_prime.view(batch_size, num_nodes, self.out_features * self.n_heads)
        else:
            return h_prime.mean(dim=2)
    
    def _prepare_attentional_mechanism_input(self, h: torch.Tensor) -> torch.Tensor:
        """
        アテンション機構入力準備
        
        Args:
            h: ノード特徴量 [batch_size, num_nodes, n_heads, out_features]
            
        Returns:
            アテンション入力 [batch_size, num_nodes, num_nodes, 2 * out_features]
        """
        batch_size, num_nodes, n_heads, out_features = h.shape
        
        # 全ノードペアの特徴量結合
        h_repeated_in_chunks = h.repeat_interleave(num_nodes, dim=1)
        h_repeated_alternating = h.repeat(1, num_nodes, 1, 1)
        
        all_combinations_matrix = torch.cat([
            h_repeated_in_chunks, h_repeated_alternating
        ], dim=-1)
        
        return all_combinations_matrix.view(batch_size, num_nodes, num_nodes, 2 * out_features)

class GraphSAGELayer(nn.Module):
    """
    GraphSAGE Layer
    
    既存GraphNeuralNetworkを活用したGraphSAGE Layer
    """
    
    def __init__(self, in_features: int, out_features: int,
                 activation: str = "relu", dropout: float = 0.1,
                 aggregator: str = "mean"):
        """
        初期化
        
        Args:
            in_features: 入力特徴量数
            out_features: 出力特徴量数
            activation: 活性化関数
            dropout: ドロップアウト率
            aggregator: 集約関数
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.dropout = dropout
        self.aggregator = aggregator
        
        # 重み行列
        self.weight = nn.Parameter(torch.FloatTensor(in_features * 2, out_features))
        
        # バイアス
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        
        # ドロップアウト
        self.dropout = nn.Dropout(dropout)
        
        # 重み初期化
        self._init_weights()
    
    def _init_weights(self):
        """重み初期化"""
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def _get_activation(self):
        """活性化関数取得"""
        if self.activation == "relu":
            return F.relu
        elif self.activation == "gelu":
            return F.gelu
        elif self.activation == "swish":
            return F.silu
        else:
            return F.relu
    
    def _aggregate(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        近傍集約
        
        Args:
            x: ノード特徴量 [batch_size, num_nodes, in_features]
            adj: 隣接行列 [batch_size, num_nodes, num_nodes]
            
        Returns:
            集約特徴量 [batch_size, num_nodes, in_features]
        """
        if self.aggregator == "mean":
            # 平均集約
            adj_norm = adj / (adj.sum(dim=-1, keepdim=True) + 1e-8)
            return torch.matmul(adj_norm, x)
        elif self.aggregator == "max":
            # 最大集約
            adj_expanded = adj.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1])
            x_expanded = x.unsqueeze(1).expand(-1, x.shape[1], -1, -1)
            masked_x = x_expanded * adj_expanded
            return masked_x.max(dim=2)[0]
        else:
            # デフォルトは平均
            adj_norm = adj / (adj.sum(dim=-1, keepdim=True) + 1e-8)
            return torch.matmul(adj_norm, x)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        フォワードパス
        
        Args:
            x: ノード特徴量 [batch_size, num_nodes, in_features]
            adj: 隣接行列 [batch_size, num_nodes, num_nodes]
            
        Returns:
            出力特徴量 [batch_size, num_nodes, out_features]
        """
        # 近傍集約
        h_neigh = self._aggregate(x, adj)
        
        # 自己特徴量と近傍特徴量結合
        h_concat = torch.cat([x, h_neigh], dim=-1)
        
        # 線形変換
        h = torch.matmul(h_concat, self.weight)
        
        # バイアス追加
        h = h + self.bias
        
        # 活性化関数
        h = self._get_activation()(h)
        
        # ドロップアウト
        h = self.dropout(h)
        
        return h

class MolecularGNN(nn.Module):
    """
    Molecular Graph Neural Network
    
    分子データ用のGNNモデル
    """
    
    def __init__(self, node_features: int, hidden_features: int, 
                 output_features: int, n_layers: int = 3,
                 layer_type: str = "gcn", activation: str = "relu",
                 dropout: float = 0.1, use_batch_norm: bool = True):
        """
        初期化
        
        Args:
            node_features: ノード特徴量数
            hidden_features: 隠れ特徴量数
            output_features: 出力特徴量数
            n_layers: レイヤー数
            layer_type: レイヤータイプ
            activation: 活性化関数
            dropout: ドロップアウト率
            use_batch_norm: バッチ正規化使用フラグ
        """
        super().__init__()
        
        self.node_features = node_features
        self.hidden_features = hidden_features
        self.output_features = output_features
        self.n_layers = n_layers
        self.layer_type = layer_type
        self.activation = activation
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # 入力投影
        self.input_projection = nn.Linear(node_features, hidden_features)
        
        # GNNレイヤー
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            if layer_type == "gcn":
                layer = GCNLayer(
                    in_features=hidden_features,
                    out_features=hidden_features,
                    activation=activation,
                    dropout=dropout
                )
            elif layer_type == "gat":
                layer = GATLayer(
                    in_features=hidden_features,
                    out_features=hidden_features,
                    n_heads=4,
                    dropout=dropout
                )
            elif layer_type == "graphsage":
                layer = GraphSAGELayer(
                    in_features=hidden_features,
                    out_features=hidden_features,
                    activation=activation,
                    dropout=dropout
                )
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")
            
            self.gnn_layers.append(layer)
        
        # バッチ正規化
        if use_batch_norm:
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(hidden_features) for _ in range(n_layers)
            ])
        else:
            self.batch_norms = None
        
        # 出力投影
        self.output_projection = nn.Linear(hidden_features, output_features)
        
        # 重み初期化
        self._init_weights()
    
    def _init_weights(self):
        """重み初期化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        フォワードパス
        
        Args:
            x: ノード特徴量 [batch_size, num_nodes, node_features]
            adj: 隣接行列 [batch_size, num_nodes, num_nodes]
            batch: バッチインデックス
            
        Returns:
            出力特徴量 [batch_size, output_features]
        """
        # 入力投影
        x = self.input_projection(x)
        
        # GNNレイヤー
        for i, layer in enumerate(self.gnn_layers):
            # GNN層
            x = layer(x, adj)
            
            # バッチ正規化
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
        
        # グローバル集約
        if batch is not None:
            # バッチ別集約
            x = self._global_pooling(x, batch)
        else:
            # 平均プーリング
            x = x.mean(dim=1)
        
        # 出力投影
        x = self.output_projection(x)
        
        return x
    
    def _global_pooling(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        グローバルプーリング
        
        Args:
            x: ノード特徴量 [total_nodes, hidden_features]
            batch: バッチインデックス [total_nodes]
            
        Returns:
            集約特徴量 [batch_size, hidden_features]
        """
        batch_size = batch.max().item() + 1
        output = torch.zeros(batch_size, x.shape[-1], device=x.device)
        
        for i in range(batch_size):
            mask = batch == i
            if mask.sum() > 0:
                output[i] = x[mask].mean(dim=0)
        
        return output

class PotencyGNN(nn.Module):
    """
    Potency Prediction GNN
    
    力価予測用のGNNモデル
    """
    
    def __init__(self, node_features: int, hidden_features: int, 
                 n_layers: int = 3, layer_type: str = "gcn",
                 activation: str = "relu", dropout: float = 0.1,
                 use_batch_norm: bool = True, num_tasks: int = 2):
        """
        初期化
        
        Args:
            node_features: ノード特徴量数
            hidden_features: 隠れ特徴量数
            n_layers: レイヤー数
            layer_type: レイヤータイプ
            activation: 活性化関数
            dropout: ドロップアウト率
            use_batch_norm: バッチ正規化使用フラグ
            num_tasks: タスク数
        """
        super().__init__()
        
        self.node_features = node_features
        self.hidden_features = hidden_features
        self.n_layers = n_layers
        self.layer_type = layer_type
        self.activation = activation
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.num_tasks = num_tasks
        
        # 分子GNN
        self.molecular_gnn = MolecularGNN(
            node_features=node_features,
            hidden_features=hidden_features,
            output_features=hidden_features,
            n_layers=n_layers,
            layer_type=layer_type,
            activation=activation,
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )
        
        # タスク別出力層
        self.regression_head = nn.Linear(hidden_features, 1)
        self.classification_head = nn.Linear(hidden_features, 1)
        
        # 重み初期化
        self._init_weights()
    
    def _init_weights(self):
        """重み初期化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        フォワードパス
        
        Args:
            x: ノード特徴量 [batch_size, num_nodes, node_features]
            adj: 隣接行列 [batch_size, num_nodes, num_nodes]
            batch: バッチインデックス
            
        Returns:
            出力辞書
        """
        # 分子GNN
        molecular_features = self.molecular_gnn(x, adj, batch)
        
        # タスク別出力
        regression_output = self.regression_head(molecular_features)
        classification_output = self.classification_head(molecular_features)
        
        return {
            'regression': regression_output.squeeze(-1),
            'classification': torch.sigmoid(classification_output.squeeze(-1))
        }

def create_molecular_gnn(node_features: int, hidden_features: int = 128,
                        output_features: int = 1, n_layers: int = 3,
                        layer_type: str = "gcn", activation: str = "relu",
                        dropout: float = 0.1, use_batch_norm: bool = True) -> MolecularGNN:
    """
    分子GNN作成
    
    Args:
        node_features: ノード特徴量数
        hidden_features: 隠れ特徴量数
        output_features: 出力特徴量数
        n_layers: レイヤー数
        layer_type: レイヤータイプ
        activation: 活性化関数
        dropout: ドロップアウト率
        use_batch_norm: バッチ正規化使用フラグ
        
    Returns:
        MolecularGNN
    """
    return MolecularGNN(
        node_features=node_features,
        hidden_features=hidden_features,
        output_features=output_features,
        n_layers=n_layers,
        layer_type=layer_type,
        activation=activation,
        dropout=dropout,
        use_batch_norm=use_batch_norm
    )

def create_potency_gnn(node_features: int, hidden_features: int = 128,
                       n_layers: int = 3, layer_type: str = "gcn",
                       activation: str = "relu", dropout: float = 0.1,
                       use_batch_norm: bool = True, num_tasks: int = 2) -> PotencyGNN:
    """
    力価予測GNN作成
    
    Args:
        node_features: ノード特徴量数
        hidden_features: 隠れ特徴量数
        n_layers: レイヤー数
        layer_type: レイヤータイプ
        activation: 活性化関数
        dropout: ドロップアウト率
        use_batch_norm: バッチ正規化使用フラグ
        num_tasks: タスク数
        
    Returns:
        PotencyGNN
    """
    return PotencyGNN(
        node_features=node_features,
        hidden_features=hidden_features,
        n_layers=n_layers,
        layer_type=layer_type,
        activation=activation,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
        num_tasks=num_tasks
    )

if __name__ == "__main__":
    # テスト実行
    node_features = 100
    hidden_features = 128
    output_features = 1
    n_layers = 3
    
    # 基本GNN
    gnn = create_molecular_gnn(
        node_features=node_features,
        hidden_features=hidden_features,
        output_features=output_features,
        n_layers=n_layers
    )
    
    print(f"Molecular GNN created: {gnn}")
    print(f"Parameters: {sum(p.numel() for p in gnn.parameters()):,}")
    
    # 力価予測GNN
    potency_gnn = create_potency_gnn(
        node_features=node_features,
        hidden_features=hidden_features,
        n_layers=n_layers
    )
    
    print(f"Potency GNN created: {potency_gnn}")
    print(f"Parameters: {sum(p.numel() for p in potency_gnn.parameters()):,}")
    
    # テスト実行
    batch_size = 2
    num_nodes = 50
    x = torch.randn(batch_size, num_nodes, node_features)
    adj = torch.randn(batch_size, num_nodes, num_nodes)
    adj = torch.sigmoid(adj)  # 隣接行列を正規化
    
    # 基本GNN
    with torch.no_grad():
        output = gnn(x, adj)
        print(f"Molecular GNN output shape: {output.shape}")
    
    # 力価予測GNN
    with torch.no_grad():
        outputs = potency_gnn(x, adj)
        print(f"Potency GNN outputs: {outputs.keys()}")
        print(f"Regression output shape: {outputs['regression'].shape}")
        print(f"Classification output shape: {outputs['classification'].shape}")
