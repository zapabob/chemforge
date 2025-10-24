"""
SU2Gate ユニットテスト
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from chemforge.core.su2_gate import SU2Gate


class TestSU2Gate:
    """SU2Gateのテスト"""
    
    def test_su2_initialization(self):
        """SU2Gate初期化テスト"""
        d_model = 512
        gate_type = 'shared'
        
        gate = SU2Gate(d_model, gate_type)
        
        assert gate.d_model == d_model
        assert gate.gate_type == gate_type
        assert gate.params.shape == (3,)
        assert gate.params.requires_grad == True
    
    def test_su2_initialization_per_head(self):
        """SU2Gate per_head初期化テスト"""
        d_model = 256
        gate_type = 'per_head'
        
        gate = SU2Gate(d_model, gate_type)
        
        assert gate.d_model == d_model
        assert gate.gate_type == gate_type
        assert gate.params.shape == (1, 3)
        assert gate.params.requires_grad == True
    
    def test_su2_forward_basic(self):
        """SU2Gate基本前向きパステスト"""
        d_model = 64
        batch_size = 2
        seq_len = 50
        
        gate = SU2Gate(d_model, 'shared')
        x = torch.randn(batch_size, seq_len, d_model)
        
        y = gate(x)
        
        # 形状チェック
        assert y.shape == x.shape
        assert y.dtype == x.dtype
        assert y.device == x.device
    
    def test_su2_forward_odd_dimension(self):
        """奇数次元でのSU2Gateテスト"""
        d_model = 65  # 奇数次元
        batch_size = 1
        seq_len = 10
        
        gate = SU2Gate(d_model, 'shared')
        x = torch.randn(batch_size, seq_len, d_model)
        
        y = gate(x)
        
        # 形状チェック
        assert y.shape == x.shape
        assert not torch.isnan(y).any(), "Output should not contain NaN"
        assert not torch.isinf(y).any(), "Output should not contain Inf"
    
    def test_su2_forward_4d_tensor(self):
        """4次元テンソルでのSU2Gateテスト"""
        d_model = 64
        batch_size = 2
        seq_len = 10
        n_heads = 4
        head_dim = d_model // n_heads
        
        gate = SU2Gate(d_model, 'shared')
        x = torch.randn(batch_size, seq_len, n_heads, head_dim)
        
        y = gate(x)
        
        # 形状チェック
        assert y.shape == x.shape
    
    def test_su2_spectral_radius(self):
        """スペクトル半径テスト"""
        d_model = 128
        gate = SU2Gate(d_model, 'shared')
        
        spectral_radius = gate.get_spectral_radius()
        
        # SU(2)行列のスペクトル半径は理論的に1.0
        assert abs(spectral_radius - 1.0) < 1e-6, f"Spectral radius should be 1.0, got {spectral_radius}"
    
    def test_su2_norm_preservation(self):
        """ノルム保存テスト"""
        d_model = 64
        batch_size = 2
        seq_len = 20
        
        gate = SU2Gate(d_model, 'shared')
        x = torch.randn(batch_size, seq_len, d_model)
        
        y = gate(x)
        
        # ノルム保存チェック（近似）
        norm_before = torch.norm(x, dim=-1)
        norm_after = torch.norm(y, dim=-1)
        norm_diff = torch.abs(norm_before - norm_after).max()
        
        # SU(2)変換はユニタリなのでノルムを保存するはず
        assert norm_diff < 0.1, f"Norm should be approximately preserved, max diff: {norm_diff}"
    
    def test_su2_gradient_flow(self):
        """勾配フローテスト"""
        d_model = 32
        batch_size = 1
        seq_len = 10
        
        gate = SU2Gate(d_model, 'shared')
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        
        y = gate(x)
        loss = y.sum()
        loss.backward()
        
        # 入力の勾配チェック
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
        
        # パラメータの勾配チェック
        assert gate.params.grad is not None
        assert gate.params.grad.abs().sum() > 0
    
    def test_su2_different_devices(self):
        """異なるデバイスでのテスト"""
        d_model = 64
        gate = SU2Gate(d_model, 'shared')
        
        # CPU
        x_cpu = torch.randn(1, 10, d_model)
        y_cpu = gate(x_cpu)
        assert y_cpu.device.type == 'cpu'
        
        # GPU（利用可能な場合）
        if torch.cuda.is_available():
            x_gpu = x_cpu.cuda()
            gate_gpu = gate.cuda()
            y_gpu = gate_gpu(x_gpu)
            assert y_gpu.device.type == 'cuda'
    
    def test_su2_different_dtypes(self):
        """異なるデータ型でのテスト"""
        d_model = 32
        gate = SU2Gate(d_model, 'shared')
        
        # float32
        x_f32 = torch.randn(1, 5, d_model, dtype=torch.float32)
        y_f32 = gate(x_f32)
        assert y_f32.dtype == torch.float32
        
        # float64
        x_f64 = torch.randn(1, 5, d_model, dtype=torch.float64)
        y_f64 = gate(x_f64)
        assert y_f64.dtype == torch.float64
    
    def test_su2_consistency(self):
        """一貫性テスト"""
        d_model = 64
        gate = SU2Gate(d_model, 'shared')
        
        x = torch.randn(1, 10, d_model)
        
        # 同じ入力で複数回実行
        outputs = []
        for _ in range(3):
            gate.eval()
            with torch.no_grad():
                y = gate(x)
                outputs.append(y)
        
        # 出力が一貫していることを確認
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i]), "Outputs should be consistent"
    
    def test_su2_parameter_update(self):
        """パラメータ更新テスト"""
        d_model = 32
        gate = SU2Gate(d_model, 'shared')
        
        # 初期パラメータ
        initial_params = gate.params.clone()
        
        # 勾配計算
        x = torch.randn(1, 5, d_model, requires_grad=True)
        y = gate(x)
        loss = y.sum()
        loss.backward()
        
        # パラメータ更新
        optimizer = torch.optim.SGD([gate.params], lr=0.01)
        optimizer.step()
        
        # パラメータが更新されたことを確認
        assert not torch.allclose(initial_params, gate.params), "Parameters should be updated"
    
    def test_su2_per_head_vs_shared(self):
        """per_head vs shared比較テスト"""
        d_model = 64
        batch_size = 2
        seq_len = 10
        
        gate_shared = SU2Gate(d_model, 'shared')
        gate_per_head = SU2Gate(d_model, 'per_head')
        
        x = torch.randn(batch_size, seq_len, d_model)
        
        y_shared = gate_shared(x)
        y_per_head = gate_per_head(x)
        
        # 形状は同じ
        assert y_shared.shape == y_per_head.shape
        
        # パラメータ形状が異なる
        assert gate_shared.params.shape == (3,)
        assert gate_per_head.params.shape == (1, 3)
    
    def test_su2_memory_efficiency(self):
        """メモリ効率テスト"""
        d_model = 256
        batch_size = 8
        seq_len = 100
        
        gate = SU2Gate(d_model, 'shared')
        x = torch.randn(batch_size, seq_len, d_model)
        
        # メモリ使用量をチェック（実際のメモリ使用量は環境依存）
        y = gate(x)
        
        assert y.shape == x.shape
        assert not torch.isnan(y).any(), "Output should not contain NaN"
        assert not torch.isinf(y).any(), "Output should not contain Inf"
    
    def test_su2_edge_cases(self):
        """エッジケーステスト"""
        # 最小次元
        gate_min = SU2Gate(2, 'shared')
        x_min = torch.randn(1, 1, 2)
        y_min = gate_min(x_min)
        assert y_min.shape == x_min.shape
        
        # 大きな次元
        gate_large = SU2Gate(1024, 'shared')
        x_large = torch.randn(1, 1, 1024)
        y_large = gate_large(x_large)
        assert y_large.shape == x_large.shape
        
        # ゼロテンソル
        gate_zero = SU2Gate(32, 'shared')
        x_zero = torch.zeros(1, 5, 32)
        y_zero = gate_zero(x_zero)
        assert y_zero.shape == x_zero.shape
        assert torch.allclose(y_zero, x_zero), "Zero input should produce zero output"


class TestSU2GateIntegration:
    """SU2Gate統合テスト"""
    
    def test_su2_with_attention(self):
        """アテンションとの統合テスト"""
        from chemforge.core.pwa_pet_attention import PWA_PET_Attention
        
        d_model = 128
        n_heads = 4
        buckets = {"trivial": 2, "fund": 2}
        
        # PWA+PETアテンション（PET有効）
        attn = PWA_PET_Attention(
            d_model=d_model,
            n_heads=n_heads,
            buckets=buckets,
            use_pet=True
        )
        
        x = torch.randn(1, 10, d_model)
        output, reg_loss = attn(x)
        
        assert output.shape == x.shape
        assert reg_loss.item() > 0  # PET正則化損失が存在
        
        # PWA+PETアテンション（PET無効）
        attn_no_pet = PWA_PET_Attention(
            d_model=d_model,
            n_heads=n_heads,
            buckets=buckets,
            use_pet=False
        )
        
        output_no_pet, reg_loss_no_pet = attn_no_pet(x)
        
        assert output_no_pet.shape == x.shape
        assert reg_loss_no_pet.item() == 0.0  # PET無効なら正則化損失は0
    
    def test_su2_parameter_initialization(self):
        """パラメータ初期化テスト"""
        d_model = 64
        gate = SU2Gate(d_model, 'shared')
        
        # パラメータが適切に初期化されていることを確認
        params = gate.params
        assert params.shape == (3,)
        assert params.requires_grad == True
        
        # パラメータの値が合理的な範囲にあることを確認
        assert torch.all(torch.isfinite(params)), "Parameters should be finite"
        assert torch.all(torch.abs(params) < 1.0), "Parameters should be small initially"
    
    def test_su2_mathematical_properties(self):
        """数学的性質テスト"""
        d_model = 32
        gate = SU2Gate(d_model, 'shared')
        
        # スペクトル半径
        spectral_radius = gate.get_spectral_radius()
        assert abs(spectral_radius - 1.0) < 1e-6
        
        # ユニタリ性の近似チェック
        x = torch.randn(1, 5, d_model)
        y = gate(x)
        
        # ノルム保存
        norm_before = torch.norm(x, dim=-1)
        norm_after = torch.norm(y, dim=-1)
        norm_diff = torch.abs(norm_before - norm_after).max()
        assert norm_diff < 0.1, f"Unitary property should preserve norms, max diff: {norm_diff}"


if __name__ == "__main__":
    pytest.main([__file__])
