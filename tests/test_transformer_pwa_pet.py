"""
PWA+PET Transformer ユニットテスト
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from chemforge.core.transformer_model import TransformerRegressor, PWAPETEncoderLayer


class TestTransformerRegressor:
    """TransformerRegressorのテスト"""
    
    def test_transformer_initialization_vanilla(self):
        """標準Transformer初期化テスト"""
        model = TransformerRegressor(
            input_dim=2279,
            hidden_dim=512,
            num_layers=4,
            num_heads=8,
            num_targets=13,
            use_pwa_pet=False
        )
        
        assert model.input_dim == 2279
        assert model.hidden_dim == 512
        assert model.num_layers == 4
        assert model.num_heads == 8
        assert model.num_targets == 13
        assert model.use_pwa_pet == False
    
    def test_transformer_initialization_pwa_pet(self):
        """PWA+PET Transformer初期化テスト"""
        model = TransformerRegressor(
            input_dim=2279,
            hidden_dim=512,
            num_layers=4,
            num_heads=8,
            num_targets=13,
            use_pwa_pet=True,
            buckets={"trivial": 2, "fund": 4, "adj": 2},
            use_rope=True,
            use_pet=True,
            pet_curv_reg=1e-5
        )
        
        assert model.use_pwa_pet == True
        assert model.buckets == {"trivial": 2, "fund": 4, "adj": 2}
        assert model.use_rope == True
        assert model.use_pet == True
        assert model.pet_curv_reg == 1e-5
    
    def test_transformer_forward_vanilla(self):
        """標準Transformer前向きパステスト"""
        model = TransformerRegressor(
            input_dim=2279,
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            num_targets=13,
            use_pwa_pet=False
        )
        
        batch_size = 2
        x = torch.randn(batch_size, 2279)
        
        output = model(x)
        
        assert output.shape == (batch_size, 13)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_transformer_forward_pwa_pet(self):
        """PWA+PET Transformer前向きパステスト"""
        model = TransformerRegressor(
            input_dim=2279,
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            num_targets=13,
            use_pwa_pet=True,
            buckets={"trivial": 1, "fund": 2, "adj": 1}
        )
        
        batch_size = 2
        x = torch.randn(batch_size, 2279)
        
        output, reg_loss = model(x)
        
        assert output.shape == (batch_size, 13)
        assert isinstance(reg_loss, torch.Tensor)
        assert reg_loss.item() >= 0
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_transformer_gradient_flow(self):
        """勾配フローテスト"""
        model = TransformerRegressor(
            input_dim=2279,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            num_targets=13,
            use_pwa_pet=True
        )
        
        x = torch.randn(2, 2279, requires_grad=True)
        
        if model.use_pwa_pet:
            output, reg_loss = model(x)
            loss = output.sum() + reg_loss
        else:
            output = model(x)
            loss = output.sum()
        
        loss.backward()
        
        # 入力の勾配チェック
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
        
        # パラメータの勾配チェック
        param_grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert all(grad is not None for grad in param_grads)
        assert all(grad.abs().sum() > 0 for grad in param_grads)
    
    def test_transformer_different_batch_sizes(self):
        """異なるバッチサイズでのテスト"""
        model = TransformerRegressor(
            input_dim=2279,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            num_targets=13,
            use_pwa_pet=True
        )
        
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 2279)
            
            if model.use_pwa_pet:
                output, reg_loss = model(x)
                assert output.shape == (batch_size, 13)
                assert reg_loss.item() >= 0
            else:
                output = model(x)
                assert output.shape == (batch_size, 13)
    
    def test_transformer_attention_mask(self):
        """アテンションマスクテスト"""
        model = TransformerRegressor(
            input_dim=2279,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            num_targets=13,
            use_pwa_pet=True
        )
        
        x = torch.randn(2, 2279)
        attention_mask = torch.zeros(2, 1, 1)  # 適切な形状のマスク
        
        output, reg_loss = model(x, attention_mask=attention_mask)
        
        assert output.shape == (2, 13)
        assert reg_loss.item() >= 0
    
    def test_transformer_consistency(self):
        """一貫性テスト"""
        model = TransformerRegressor(
            input_dim=2279,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            num_targets=13,
            use_pwa_pet=True
        )
        
        x = torch.randn(2, 2279)
        
        # 同じ入力で複数回実行
        outputs = []
        reg_losses = []
        
        for _ in range(3):
            model.eval()
            with torch.no_grad():
                output, reg_loss = model(x)
                outputs.append(output)
                reg_losses.append(reg_loss)
        
        # 出力が一貫していることを確認
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i]), "Outputs should be consistent"
            assert torch.allclose(reg_losses[0], reg_losses[i]), "Regularization losses should be consistent"


class TestPWAPETEncoderLayer:
    """PWAPETEncoderLayerのテスト"""
    
    def test_encoder_layer_initialization(self):
        """エンコーダー層初期化テスト"""
        from chemforge.core.pwa_pet_attention import PWA_PET_Attention
        
        d_model = 256
        n_heads = 8
        buckets = {"trivial": 2, "fund": 4, "adj": 2}
        
        attention = PWA_PET_Attention(
            d_model=d_model,
            n_heads=n_heads,
            buckets=buckets
        )
        
        feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        norm1 = nn.LayerNorm(d_model)
        norm2 = nn.LayerNorm(d_model)
        
        layer = PWAPETEncoderLayer(
            attention=attention,
            feedforward=feedforward,
            norm1=norm1,
            norm2=norm2,
            dropout=0.1
        )
        
        assert layer.attention == attention
        assert layer.feedforward == feedforward
        assert layer.norm1 == norm1
        assert layer.norm2 == norm2
    
    def test_encoder_layer_forward(self):
        """エンコーダー層前向きパステスト"""
        from chemforge.core.pwa_pet_attention import PWA_PET_Attention
        
        d_model = 128
        n_heads = 4
        buckets = {"trivial": 1, "fund": 2, "adj": 1}
        
        attention = PWA_PET_Attention(
            d_model=d_model,
            n_heads=n_heads,
            buckets=buckets
        )
        
        feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        norm1 = nn.LayerNorm(d_model)
        norm2 = nn.LayerNorm(d_model)
        
        layer = PWAPETEncoderLayer(
            attention=attention,
            feedforward=feedforward,
            norm1=norm1,
            norm2=norm2,
            dropout=0.0
        )
        
        batch_size = 2
        seq_len = 50
        x = torch.randn(batch_size, seq_len, d_model)
        
        output, reg_loss = layer(x)
        
        assert output.shape == x.shape
        assert isinstance(reg_loss, torch.Tensor)
        assert reg_loss.item() >= 0
    
    def test_encoder_layer_gradient_flow(self):
        """エンコーダー層勾配フローテスト"""
        from chemforge.core.pwa_pet_attention import PWA_PET_Attention
        
        d_model = 64
        n_heads = 2
        buckets = {"trivial": 1, "fund": 1}
        
        attention = PWA_PET_Attention(
            d_model=d_model,
            n_heads=n_heads,
            buckets=buckets
        )
        
        feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        norm1 = nn.LayerNorm(d_model)
        norm2 = nn.LayerNorm(d_model)
        
        layer = PWAPETEncoderLayer(
            attention=attention,
            feedforward=feedforward,
            norm1=norm1,
            norm2=norm2,
            dropout=0.0
        )
        
        x = torch.randn(1, 10, d_model, requires_grad=True)
        output, reg_loss = layer(x)
        
        loss = output.sum() + reg_loss
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
        
        # パラメータの勾配チェック
        param_grads = [p.grad for p in layer.parameters() if p.requires_grad]
        assert all(grad is not None for grad in param_grads)
        assert all(grad.abs().sum() > 0 for grad in param_grads)


class TestTransformerIntegration:
    """Transformer統合テスト"""
    
    def test_vanilla_vs_pwa_pet_comparison(self):
        """標準Transformer vs PWA+PET比較テスト"""
        # 標準Transformer
        model_vanilla = TransformerRegressor(
            input_dim=2279,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            num_targets=13,
            use_pwa_pet=False
        )
        
        # PWA+PET Transformer
        model_pwa_pet = TransformerRegressor(
            input_dim=2279,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            num_targets=13,
            use_pwa_pet=True,
            buckets={"trivial": 1, "fund": 2, "adj": 1}
        )
        
        x = torch.randn(2, 2279)
        
        # 標準Transformer
        output_vanilla = model_vanilla(x)
        assert output_vanilla.shape == (2, 13)
        
        # PWA+PET Transformer
        output_pwa_pet, reg_loss = model_pwa_pet(x)
        assert output_pwa_pet.shape == (2, 13)
        assert reg_loss.item() >= 0
        
        # 出力が異なることを確認（異なるアーキテクチャなので）
        assert not torch.allclose(output_vanilla, output_pwa_pet), "Different architectures should produce different outputs"
    
    def test_transformer_memory_efficiency(self):
        """メモリ効率テスト"""
        model = TransformerRegressor(
            input_dim=2279,
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            num_targets=13,
            use_pwa_pet=True,
            buckets={"trivial": 2, "fund": 4, "adj": 2}
        )
        
        # 大きなバッチサイズでテスト
        batch_size = 16
        x = torch.randn(batch_size, 2279)
        
        output, reg_loss = model(x)
        
        assert output.shape == (batch_size, 13)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert reg_loss.item() >= 0
    
    def test_transformer_different_configurations(self):
        """異なる設定でのテスト"""
        configs = [
            # 標準Transformer
            {"use_pwa_pet": False},
            # PWA+PET (RoPE有効)
            {"use_pwa_pet": True, "use_rope": True, "use_pet": True},
            # PWA+PET (RoPE無効)
            {"use_pwa_pet": True, "use_rope": False, "use_pet": True},
            # PWA+PET (PET無効)
            {"use_pwa_pet": True, "use_rope": True, "use_pet": False},
        ]
        
        for config in configs:
            model = TransformerRegressor(
                input_dim=2279,
                hidden_dim=128,
                num_layers=2,
                num_heads=4,
                num_targets=13,
                **config
            )
            
            x = torch.randn(2, 2279)
            
            if model.use_pwa_pet:
                output, reg_loss = model(x)
                assert output.shape == (2, 13)
                assert reg_loss.item() >= 0
            else:
                output = model(x)
                assert output.shape == (2, 13)
    
    def test_transformer_parameter_count(self):
        """パラメータ数テスト"""
        model_vanilla = TransformerRegressor(
            input_dim=2279,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            num_targets=13,
            use_pwa_pet=False
        )
        
        model_pwa_pet = TransformerRegressor(
            input_dim=2279,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            num_targets=13,
            use_pwa_pet=True
        )
        
        # パラメータ数を計算
        params_vanilla = sum(p.numel() for p in model_vanilla.parameters())
        params_pwa_pet = sum(p.numel() for p in model_pwa_pet.parameters())
        
        # PWA+PETの方がパラメータ数が多いことを確認（SU2Gate等の追加パラメータ）
        assert params_pwa_pet > params_vanilla, "PWA+PET should have more parameters due to additional components"
    
    def test_transformer_training_mode(self):
        """訓練モードテスト"""
        model = TransformerRegressor(
            input_dim=2279,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            num_targets=13,
            use_pwa_pet=True
        )
        
        x = torch.randn(2, 2279)
        
        # 訓練モード
        model.train()
        output_train, reg_loss_train = model(x)
        
        # 評価モード
        model.eval()
        with torch.no_grad():
            output_eval, reg_loss_eval = model(x)
        
        # 出力が異なることを確認（ドロップアウト等の影響）
        assert not torch.allclose(output_train, output_eval), "Training and eval modes should produce different outputs"


if __name__ == "__main__":
    pytest.main([__file__])
