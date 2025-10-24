"""
GNN Model Tests

GNNモデルのユニットテスト
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from chemforge.models.gnn_model import GNNRegressor, MolecularGNN, MultiScaleGNN


class TestGNNRegressor:
    """GNN回帰モデルのテストクラス"""
    
    def setup_method(self):
        """テスト前のセットアップ"""
        self.input_dim = 100
        self.hidden_dim = 64
        self.num_layers = 3
        self.num_heads = 4
        self.num_targets = 5
        self.batch_size = 8
        self.num_nodes = 20
        
        # テストデータ
        self.x = torch.randn(self.num_nodes, self.input_dim)
        self.edge_index = torch.randint(0, self.num_nodes, (2, 30))
        self.batch = torch.randint(0, self.batch_size, (self.num_nodes,))
    
    def test_initialization(self):
        """初期化テスト"""
        model = GNNRegressor(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_targets=self.num_targets
        )
        
        assert model.input_dim == self.input_dim
        assert model.hidden_dim == self.hidden_dim
        assert model.num_layers == self.num_layers
        assert model.num_targets == self.num_targets
    
    def test_forward_gcn(self):
        """GCNフォワードテスト"""
        model = GNNRegressor(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_targets=self.num_targets,
            gnn_type="gcn"
        )
        
        output = model(self.x, self.edge_index, self.batch)
        
        assert output.shape == (self.batch_size, self.num_targets)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_forward_gat(self):
        """GATフォワードテスト"""
        model = GNNRegressor(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_targets=self.num_targets,
            gnn_type="gat"
        )
        
        output = model(self.x, self.edge_index, self.batch)
        
        assert output.shape == (self.batch_size, self.num_targets)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_forward_gin(self):
        """GINフォワードテスト"""
        model = GNNRegressor(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_targets=self.num_targets,
            gnn_type="gin"
        )
        
        output = model(self.x, self.edge_index, self.batch)
        
        assert output.shape == (self.batch_size, self.num_targets)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_forward_sage(self):
        """SAGEフォワードテスト"""
        model = GNNRegressor(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_targets=self.num_targets,
            gnn_type="sage"
        )
        
        output = model(self.x, self.edge_index, self.batch)
        
        assert output.shape == (self.batch_size, self.num_targets)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_forward_without_batch(self):
        """バッチなしフォワードテスト"""
        model = GNNRegressor(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_targets=self.num_targets
        )
        
        output = model(self.x, self.edge_index)
        
        assert output.shape == (1, self.num_targets)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_attention_enabled(self):
        """アテンション有効テスト"""
        model = GNNRegressor(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_targets=self.num_targets,
            use_attention=True
        )
        
        output = model(self.x, self.edge_index, self.batch)
        
        assert output.shape == (self.batch_size, self.num_targets)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_attention_disabled(self):
        """アテンション無効テスト"""
        model = GNNRegressor(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_targets=self.num_targets,
            use_attention=False
        )
        
        output = model(self.x, self.edge_index, self.batch)
        
        assert output.shape == (self.batch_size, self.num_targets)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_global_pooling_mean(self):
        """平均グローバルプーリングテスト"""
        model = GNNRegressor(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_targets=self.num_targets,
            use_global_pooling="mean"
        )
        
        output = model(self.x, self.edge_index, self.batch)
        
        assert output.shape == (self.batch_size, self.num_targets)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_global_pooling_max(self):
        """最大グローバルプーリングテスト"""
        model = GNNRegressor(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_targets=self.num_targets,
            use_global_pooling="max"
        )
        
        output = model(self.x, self.edge_index, self.batch)
        
        assert output.shape == (self.batch_size, self.num_targets)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_global_pooling_add(self):
        """加算グローバルプーリングテスト"""
        model = GNNRegressor(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_targets=self.num_targets,
            use_global_pooling="add"
        )
        
        output = model(self.x, self.edge_index, self.batch)
        
        assert output.shape == (self.batch_size, self.num_targets)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_get_embeddings(self):
        """埋め込み取得テスト"""
        model = GNNRegressor(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_targets=self.num_targets
        )
        
        embeddings = model.get_embeddings(self.x, self.edge_index, self.batch)
        
        assert embeddings.shape == (self.batch_size, self.hidden_dim)
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()
    
    def test_error_handling(self):
        """エラーハンドリングテスト"""
        # 無効なGNNタイプ
        with pytest.raises(ValueError):
            GNNRegressor(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                num_targets=self.num_targets,
                gnn_type="invalid"
            )
        
        # 無効なグローバルプーリング
        with pytest.raises(ValueError):
            model = GNNRegressor(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                num_targets=self.num_targets,
                use_global_pooling="invalid"
            )
            model(self.x, self.edge_index, self.batch)


class TestMolecularGNN:
    """分子特化GNNのテストクラス"""
    
    def setup_method(self):
        """テスト前のセットアップ"""
        self.atom_dim = 50
        self.bond_dim = 30
        self.hidden_dim = 64
        self.num_layers = 3
        self.num_targets = 5
        self.batch_size = 4
        self.num_atoms = 15
        
        # テストデータ
        self.atom_features = torch.randn(self.num_atoms, self.atom_dim)
        self.bond_features = torch.randn(20, self.bond_dim)
        self.edge_index = torch.randint(0, self.num_atoms, (2, 20))
        self.batch = torch.randint(0, self.batch_size, (self.num_atoms,))
    
    def test_initialization(self):
        """初期化テスト"""
        model = MolecularGNN(
            atom_dim=self.atom_dim,
            bond_dim=self.bond_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_targets=self.num_targets
        )
        
        assert model.atom_dim == self.atom_dim
        assert model.bond_dim == self.bond_dim
        assert model.hidden_dim == self.hidden_dim
        assert model.num_layers == self.num_layers
        assert model.num_targets == self.num_targets
    
    def test_forward(self):
        """フォワードテスト"""
        model = MolecularGNN(
            atom_dim=self.atom_dim,
            bond_dim=self.bond_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_targets=self.num_targets
        )
        
        output = model(self.atom_features, self.bond_features, self.edge_index, self.batch)
        
        assert output.shape == (self.batch_size, self.num_targets)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_forward_without_batch(self):
        """バッチなしフォワードテスト"""
        model = MolecularGNN(
            atom_dim=self.atom_dim,
            bond_dim=self.bond_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_targets=self.num_targets
        )
        
        output = model(self.atom_features, self.bond_features, self.edge_index)
        
        assert output.shape == (1, self.num_targets)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_different_gnn_types(self):
        """異なるGNNタイプテスト"""
        gnn_types = ["gcn", "gat", "gin", "sage"]
        
        for gnn_type in gnn_types:
            model = MolecularGNN(
                atom_dim=self.atom_dim,
                bond_dim=self.bond_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                num_targets=self.num_targets,
                gnn_type=gnn_type
            )
            
            output = model(self.atom_features, self.bond_features, self.edge_index, self.batch)
            
            assert output.shape == (self.batch_size, self.num_targets)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()


class TestMultiScaleGNN:
    """マルチスケールGNNのテストクラス"""
    
    def setup_method(self):
        """テスト前のセットアップ"""
        self.input_dim = 100
        self.hidden_dim = 64
        self.num_layers = 3
        self.num_targets = 5
        self.scales = [1, 2, 3]
        self.batch_size = 4
        self.num_nodes = 20
        
        # テストデータ
        self.x = torch.randn(self.num_nodes, self.input_dim)
        self.edge_index = torch.randint(0, self.num_nodes, (2, 30))
        self.batch = torch.randint(0, self.batch_size, (self.num_nodes,))
    
    def test_initialization(self):
        """初期化テスト"""
        model = MultiScaleGNN(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_targets=self.num_targets,
            scales=self.scales
        )
        
        assert model.input_dim == self.input_dim
        assert model.hidden_dim == self.hidden_dim
        assert model.num_layers == self.num_layers
        assert model.num_targets == self.num_targets
        assert model.scales == self.scales
    
    def test_forward(self):
        """フォワードテスト"""
        model = MultiScaleGNN(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_targets=self.num_targets,
            scales=self.scales
        )
        
        output = model(self.x, self.edge_index, self.batch)
        
        assert output.shape == (self.batch_size, self.num_targets)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_forward_without_batch(self):
        """バッチなしフォワードテスト"""
        model = MultiScaleGNN(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_targets=self.num_targets,
            scales=self.scales
        )
        
        output = model(self.x, self.edge_index)
        
        assert output.shape == (1, self.num_targets)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_different_scales(self):
        """異なるスケールテスト"""
        scales_list = [[1], [1, 2], [1, 2, 3], [1, 3, 5]]
        
        for scales in scales_list:
            model = MultiScaleGNN(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                num_targets=self.num_targets,
                scales=scales
            )
            
            output = model(self.x, self.edge_index, self.batch)
            
            assert output.shape == (self.batch_size, self.num_targets)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
    
    def test_single_scale(self):
        """単一スケールテスト"""
        model = MultiScaleGNN(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_targets=self.num_targets,
            scales=[1]
        )
        
        output = model(self.x, self.edge_index, self.batch)
        
        assert output.shape == (self.batch_size, self.num_targets)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
