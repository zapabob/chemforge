"""
Ensemble Model Tests

アンサンブルモデルのユニットテスト
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from chemforge.models.ensemble_model import EnsembleRegressor, HybridEnsemble, AdaptiveEnsemble


class TestEnsembleRegressor:
    """アンサンブル回帰モデルのテストクラス"""
    
    def setup_method(self):
        """テスト前のセットアップ"""
        self.input_dim = 100
        self.hidden_dim = 64
        self.num_targets = 5
        self.batch_size = 8
        
        # モックモデル
        self.mock_model1 = Mock()
        self.mock_model1.num_targets = self.num_targets
        self.mock_model1.forward.return_value = torch.randn(self.batch_size, self.num_targets)
        
        self.mock_model2 = Mock()
        self.mock_model2.num_targets = self.num_targets
        self.mock_model2.forward.return_value = torch.randn(self.batch_size, self.num_targets)
        
        self.models = [self.mock_model1, self.mock_model2]
        
        # テストデータ
        self.x = torch.randn(self.batch_size, self.input_dim)
    
    def test_initialization(self):
        """初期化テスト"""
        model = EnsembleRegressor(
            models=self.models,
            ensemble_method="weighted_average"
        )
        
        assert len(model.models) == 2
        assert model.ensemble_method == "weighted_average"
        assert model.weights == [0.5, 0.5]
    
    def test_initialization_with_weights(self):
        """重み付き初期化テスト"""
        weights = [0.7, 0.3]
        model = EnsembleRegressor(
            models=self.models,
            ensemble_method="weighted_average",
            weights=weights
        )
        
        assert model.weights == weights
    
    def test_initialization_with_meta_learning(self):
        """メタ学習付き初期化テスト"""
        model = EnsembleRegressor(
            models=self.models,
            ensemble_method="stacking",
            use_meta_learning=True,
            meta_hidden_dim=128
        )
        
        assert model.use_meta_learning == True
        assert model.meta_hidden_dim == 128
    
    def test_forward_weighted_average(self):
        """重み付き平均フォワードテスト"""
        model = EnsembleRegressor(
            models=self.models,
            ensemble_method="weighted_average",
            weights=[0.6, 0.4]
        )
        
        output = model(self.x)
        
        assert output.shape == (self.batch_size, self.num_targets)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_forward_stacking(self):
        """スタッキングフォワードテスト"""
        model = EnsembleRegressor(
            models=self.models,
            ensemble_method="stacking",
            use_meta_learning=False
        )
        
        output = model(self.x)
        
        assert output.shape == (self.batch_size, self.num_targets)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_forward_stacking_with_meta_learning(self):
        """メタ学習付きスタッキングフォワードテスト"""
        model = EnsembleRegressor(
            models=self.models,
            ensemble_method="stacking",
            use_meta_learning=True,
            meta_hidden_dim=128
        )
        
        output = model(self.x)
        
        assert output.shape == (self.batch_size, self.num_targets)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_forward_voting(self):
        """投票フォワードテスト"""
        model = EnsembleRegressor(
            models=self.models,
            ensemble_method="voting"
        )
        
        output = model(self.x)
        
        assert output.shape == (self.batch_size, self.num_targets)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_get_model_weights(self):
        """モデル重み取得テスト"""
        weights = [0.6, 0.4]
        model = EnsembleRegressor(
            models=self.models,
            ensemble_method="weighted_average",
            weights=weights
        )
        
        retrieved_weights = model.get_model_weights()
        assert retrieved_weights == weights
    
    def test_set_model_weights(self):
        """モデル重み設定テスト"""
        model = EnsembleRegressor(
            models=self.models,
            ensemble_method="weighted_average"
        )
        
        new_weights = [0.8, 0.2]
        model.set_model_weights(new_weights)
        
        assert model.weights == new_weights
    
    def test_set_model_weights_invalid_length(self):
        """無効な重み長設定テスト"""
        model = EnsembleRegressor(
            models=self.models,
            ensemble_method="weighted_average"
        )
        
        with pytest.raises(ValueError):
            model.set_model_weights([0.5])  # 長さが一致しない
    
    def test_get_model_predictions(self):
        """モデル予測取得テスト"""
        model = EnsembleRegressor(
            models=self.models,
            ensemble_method="weighted_average"
        )
        
        predictions = model.get_model_predictions(self.x)
        
        assert len(predictions) == 2
        for pred in predictions:
            assert pred.shape == (self.batch_size, self.num_targets)
    
    def test_error_handling(self):
        """エラーハンドリングテスト"""
        # 無効なアンサンブル方法
        with pytest.raises(ValueError):
            model = EnsembleRegressor(
                models=self.models,
                ensemble_method="invalid"
            )
            model(self.x)


class TestHybridEnsemble:
    """ハイブリッドアンサンブルのテストクラス"""
    
    def setup_method(self):
        """テスト前のセットアップ"""
        self.input_dim = 100
        self.hidden_dim = 64
        self.num_targets = 5
        self.batch_size = 8
        
        # 深層学習モデル
        self.deep_model1 = Mock()
        self.deep_model1.num_targets = self.num_targets
        self.deep_model1.forward.return_value = torch.randn(self.batch_size, self.num_targets)
        
        self.deep_model2 = Mock()
        self.deep_model2.num_targets = self.num_targets
        self.deep_model2.forward.return_value = torch.randn(self.batch_size, self.num_targets)
        
        self.deep_models = [self.deep_model1, self.deep_model2]
        
        # 機械学習モデル
        self.ml_model1 = Mock()
        self.ml_model1.predict.return_value = np.random.randn(self.batch_size, self.num_targets)
        
        self.ml_model2 = Mock()
        self.ml_model2.predict.return_value = np.random.randn(self.batch_size, self.num_targets)
        
        self.ml_models = [self.ml_model1, self.ml_model2]
        
        # テストデータ
        self.x = torch.randn(self.batch_size, self.input_dim)
    
    def test_initialization(self):
        """初期化テスト"""
        model = HybridEnsemble(
            deep_models=self.deep_models,
            ml_models=self.ml_models,
            fusion_method="attention"
        )
        
        assert len(model.deep_models) == 2
        assert len(model.ml_models) == 2
        assert model.fusion_method == "attention"
    
    def test_forward_attention(self):
        """アテンション融合フォワードテスト"""
        model = HybridEnsemble(
            deep_models=self.deep_models,
            ml_models=self.ml_models,
            fusion_method="attention"
        )
        
        output = model(self.x)
        
        assert output.shape == (self.batch_size, self.num_targets)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_forward_concat(self):
        """結合融合フォワードテスト"""
        model = HybridEnsemble(
            deep_models=self.deep_models,
            ml_models=self.ml_models,
            fusion_method="concat"
        )
        
        output = model(self.x)
        
        assert output.shape == (self.batch_size, self.num_targets)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_forward_weighted(self):
        """重み付き融合フォワードテスト"""
        model = HybridEnsemble(
            deep_models=self.deep_models,
            ml_models=self.ml_models,
            fusion_method="weighted"
        )
        
        output = model(self.x)
        
        assert output.shape == (self.batch_size, self.num_targets)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestAdaptiveEnsemble:
    """適応的アンサンブルのテストクラス"""
    
    def setup_method(self):
        """テスト前のセットアップ"""
        self.input_dim = 100
        self.hidden_dim = 64
        self.num_targets = 5
        self.batch_size = 8
        
        # モックモデル
        self.mock_model1 = Mock()
        self.mock_model1.num_targets = self.num_targets
        self.mock_model1.forward.return_value = torch.randn(self.batch_size, self.num_targets)
        
        self.mock_model2 = Mock()
        self.mock_model2.num_targets = self.num_targets
        self.mock_model2.forward.return_value = torch.randn(self.batch_size, self.num_targets)
        
        self.models = [self.mock_model1, self.mock_model2]
        
        # テストデータ
        self.x = torch.randn(self.batch_size, self.input_dim)
    
    def test_initialization(self):
        """初期化テスト"""
        model = AdaptiveEnsemble(
            models=self.models,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_targets=self.num_targets
        )
        
        assert len(model.models) == 2
        assert model.input_dim == self.input_dim
        assert model.hidden_dim == self.hidden_dim
        assert model.num_targets == self.num_targets
    
    def test_forward(self):
        """フォワードテスト"""
        model = AdaptiveEnsemble(
            models=self.models,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_targets=self.num_targets
        )
        
        output = model(self.x)
        
        assert output.shape == (self.batch_size, self.num_targets)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_get_adaptive_weights(self):
        """適応的重み取得テスト"""
        model = AdaptiveEnsemble(
            models=self.models,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_targets=self.num_targets
        )
        
        weights = model.get_adaptive_weights(self.x)
        
        assert weights.shape == (self.batch_size, len(self.models))
        assert torch.allclose(weights.sum(dim=1), torch.ones(self.batch_size))
        assert not torch.isnan(weights).any()
        assert not torch.isinf(weights).any()
    
    def test_weights_sum_to_one(self):
        """重みの合計が1になることをテスト"""
        model = AdaptiveEnsemble(
            models=self.models,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_targets=self.num_targets
        )
        
        weights = model.get_adaptive_weights(self.x)
        
        # 各行の重みの合計が1になることを確認
        assert torch.allclose(weights.sum(dim=1), torch.ones(self.batch_size), atol=1e-6)
    
    def test_weights_non_negative(self):
        """重みが非負であることをテスト"""
        model = AdaptiveEnsemble(
            models=self.models,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_targets=self.num_targets
        )
        
        weights = model.get_adaptive_weights(self.x)
        
        # 重みが非負であることを確認
        assert torch.all(weights >= 0)
    
    def test_different_input_sizes(self):
        """異なる入力サイズテスト"""
        model = AdaptiveEnsemble(
            models=self.models,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_targets=self.num_targets
        )
        
        # 異なるバッチサイズ
        x_small = torch.randn(4, self.input_dim)
        x_large = torch.randn(16, self.input_dim)
        
        output_small = model(x_small)
        output_large = model(x_large)
        
        assert output_small.shape == (4, self.num_targets)
        assert output_large.shape == (16, self.num_targets)
    
    def test_consistency(self):
        """一貫性テスト"""
        model = AdaptiveEnsemble(
            models=self.models,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_targets=self.num_targets
        )
        
        # 同じ入力に対して同じ出力が得られることを確認
        output1 = model(self.x)
        output2 = model(self.x)
        
        assert torch.allclose(output1, output2)
    
    def test_gradient_flow(self):
        """勾配フローテスト"""
        model = AdaptiveEnsemble(
            models=self.models,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_targets=self.num_targets
        )
        
        # 勾配を有効にする
        self.x.requires_grad_(True)
        
        output = model(self.x)
        loss = output.sum()
        loss.backward()
        
        # 入力に勾配が流れることを確認
        assert self.x.grad is not None
        assert not torch.isnan(self.x.grad).any()
        assert not torch.isinf(self.x.grad).any()
