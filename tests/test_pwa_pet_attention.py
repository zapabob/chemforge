"""
PWA+PET Attention ユニットテスト
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from chemforge.core.pwa_pet_attention import PWA_PET_Attention, PWAHeadRouter, RoPE


class TestPWAHeadRouter:
    """PWAHeadRouterのテスト"""
    
    def test_router_initialization(self):
        """ルーターの初期化テスト"""
        n_heads = 8
        buckets = {"trivial": 2, "fund": 4, "adj": 2}
        
        router = PWAHeadRouter(n_heads, buckets)
        
        assert router.n_heads == n_heads
        assert router.buckets == buckets
        assert len(router.head_to_bucket) == n_heads
        assert len(router.bucket_to_heads) == len(buckets)
    
    def test_head_assignment(self):
        """ヘッド割り当てテスト"""
        n_heads = 8
        buckets = {"trivial": 2, "fund": 4, "adj": 2}
        
        router = PWAHeadRouter(n_heads, buckets)
        
        # 各バケットのヘッド数を確認
        assert len(router.get_bucket_heads("trivial")) == 2
        assert len(router.get_bucket_heads("fund")) == 4
        assert len(router.get_bucket_heads("adj")) == 2
        
        # 全ヘッドが割り当てられていることを確認
        total_heads = sum(len(heads) for heads in router.bucket_to_heads.values())
        assert total_heads == n_heads
    
    def test_head_bucket_mapping(self):
        """ヘッド-バケットマッピングテスト"""
        n_heads = 6
        buckets = {"trivial": 2, "fund": 3, "adj": 1}
        
        router = PWAHeadRouter(n_heads, buckets)
        
        # 各ヘッドが正しいバケットに割り当てられていることを確認
        for head_idx in range(n_heads):
            bucket = router.get_head_bucket(head_idx)
            assert bucket in buckets.keys()
            assert head_idx in router.get_bucket_heads(bucket)


class TestRoPE:
    """RoPEのテスト"""
    
    def test_rope_initialization(self):
        """RoPE初期化テスト"""
        head_dim = 64
        max_seq_len = 1024
        
        rope = RoPE(head_dim, max_seq_len)
        
        assert rope.head_dim == head_dim
        assert rope.max_seq_len == max_seq_len
        assert rope.inv_freq.shape == (head_dim // 2,)
    
    def test_rope_forward(self):
        """RoPE前向きパステスト"""
        head_dim = 64
        seq_len = 50
        batch_size = 2
        n_heads = 8
        
        rope = RoPE(head_dim)
        
        # テストデータ
        q = torch.randn(batch_size, n_heads, seq_len, head_dim)
        k = torch.randn(batch_size, n_heads, seq_len, head_dim)
        
        # RoPE適用
        q_rotated, k_rotated = rope(q, k, seq_len)
        
        # 形状チェック
        assert q_rotated.shape == q.shape
        assert k_rotated.shape == k.shape
        
        # 回転の性質チェック（ノルム保存）
        q_norm_before = torch.norm(q, dim=-1)
        q_norm_after = torch.norm(q_rotated, dim=-1)
        norm_diff = torch.abs(q_norm_before - q_norm_after).max()
        assert norm_diff < 1e-6, f"RoPE should preserve norms, max diff: {norm_diff}"
    
    def test_rope_caching(self):
        """RoPEキャッシュテスト"""
        head_dim = 32
        rope = RoPE(head_dim)
        
        # 最初の呼び出し
        q1 = torch.randn(1, 1, 10, head_dim)
        k1 = torch.randn(1, 1, 10, head_dim)
        q_rot1, k_rot1 = rope(q1, k1, 10)
        
        # 2回目の呼び出し（キャッシュ使用）
        q2 = torch.randn(1, 1, 10, head_dim)
        k2 = torch.randn(1, 1, 10, head_dim)
        q_rot2, k_rot2 = rope(q2, k2, 10)
        
        # キャッシュが正しく動作していることを確認
        assert rope._cos_cached is not None
        assert rope._sin_cached is not None
        assert rope._seq_len_cached == 10


class TestPWAPETAttention:
    """PWA_PET_Attentionのテスト"""
    
    def test_attention_initialization(self):
        """アテンション初期化テスト"""
        d_model = 512
        n_heads = 8
        buckets = {"trivial": 2, "fund": 4, "adj": 2}
        
        attn = PWA_PET_Attention(
            d_model=d_model,
            n_heads=n_heads,
            buckets=buckets,
            dropout=0.1,
            use_rope=True,
            use_pet=True
        )
        
        assert attn.d_model == d_model
        assert attn.n_heads == n_heads
        assert attn.head_dim == d_model // n_heads
        assert attn.buckets == buckets
        assert attn.use_rope == True
        assert attn.use_pet == True
    
    def test_attention_forward(self):
        """アテンション前向きパステスト"""
        d_model = 256
        n_heads = 8
        buckets = {"trivial": 2, "fund": 4, "adj": 2}
        batch_size = 2
        seq_len = 50
        
        attn = PWA_PET_Attention(
            d_model=d_model,
            n_heads=n_heads,
            buckets=buckets,
            dropout=0.0,
            use_rope=True,
            use_pet=True
        )
        
        # テストデータ
        x = torch.randn(batch_size, seq_len, d_model)
        
        # 前向きパス
        output, reg_loss = attn(x)
        
        # 形状チェック
        assert output.shape == x.shape
        assert isinstance(reg_loss, torch.Tensor)
        assert reg_loss.item() >= 0  # 正則化損失は非負
        
        # 勾配チェック
        x.requires_grad_(True)
        output, reg_loss = attn(x)
        loss = output.sum() + reg_loss
        loss.backward()
        
        assert x.grad is not None, "Gradient should flow through attention"
    
    def test_attention_without_pet(self):
        """PET無しアテンションテスト"""
        d_model = 128
        n_heads = 4
        buckets = {"trivial": 1, "fund": 2, "adj": 1}
        
        attn = PWA_PET_Attention(
            d_model=d_model,
            n_heads=n_heads,
            buckets=buckets,
            use_pet=False
        )
        
        x = torch.randn(1, 10, d_model)
        output, reg_loss = attn(x)
        
        assert output.shape == x.shape
        assert reg_loss.item() == 0.0  # PET無しなら正則化損失は0
    
    def test_attention_without_rope(self):
        """RoPE無しアテンションテスト"""
        d_model = 128
        n_heads = 4
        buckets = {"trivial": 2, "fund": 2}
        
        attn = PWA_PET_Attention(
            d_model=d_model,
            n_heads=n_heads,
            buckets=buckets,
            use_rope=False
        )
        
        x = torch.randn(1, 10, d_model)
        output, reg_loss = attn(x)
        
        assert output.shape == x.shape
        assert attn.rope is None
    
    def test_attention_mask(self):
        """アテンションマスクテスト"""
        d_model = 128
        n_heads = 4
        buckets = {"trivial": 2, "fund": 2}
        seq_len = 10
        
        attn = PWA_PET_Attention(
            d_model=d_model,
            n_heads=n_heads,
            buckets=buckets
        )
        
        x = torch.randn(1, seq_len, d_model)
        
        # マスク作成（後半をマスク）
        mask = torch.zeros(1, seq_len, seq_len)
        mask[:, :, seq_len//2:] = float('-inf')
        
        output, reg_loss = attn(x, attention_mask=mask)
        
        assert output.shape == x.shape
    
    def test_attention_different_batch_sizes(self):
        """異なるバッチサイズでのテスト"""
        d_model = 64
        n_heads = 2
        buckets = {"trivial": 1, "fund": 1}
        
        attn = PWA_PET_Attention(
            d_model=d_model,
            n_heads=n_heads,
            buckets=buckets
        )
        
        # 異なるバッチサイズでテスト
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 20, d_model)
            output, reg_loss = attn(x)
            
            assert output.shape == x.shape
            assert reg_loss.item() >= 0
    
    def test_attention_gradient_flow(self):
        """勾配フローテスト"""
        d_model = 64
        n_heads = 2
        buckets = {"trivial": 1, "fund": 1}
        
        attn = PWA_PET_Attention(
            d_model=d_model,
            n_heads=n_heads,
            buckets=buckets
        )
        
        x = torch.randn(1, 10, d_model, requires_grad=True)
        output, reg_loss = attn(x)
        
        # 損失計算
        loss = output.sum() + reg_loss
        loss.backward()
        
        # 勾配が存在することを確認
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
        
        # パラメータの勾配も確認
        for param in attn.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert param.grad.abs().sum() > 0


class TestPWAPETIntegration:
    """PWA+PET統合テスト"""
    
    def test_full_attention_pipeline(self):
        """完全なアテンションパイプラインテスト"""
        d_model = 256
        n_heads = 8
        buckets = {"trivial": 2, "fund": 4, "adj": 2}
        batch_size = 2
        seq_len = 50
        
        attn = PWA_PET_Attention(
            d_model=d_model,
            n_heads=n_heads,
            buckets=buckets,
            dropout=0.1,
            use_rope=True,
            use_pet=True,
            pet_curv_reg=1e-5
        )
        
        x = torch.randn(batch_size, seq_len, d_model)
        
        # 前向きパス
        output, reg_loss = attn(x)
        
        # 基本チェック
        assert output.shape == x.shape
        assert isinstance(reg_loss, torch.Tensor)
        assert reg_loss.item() >= 0
        
        # 勾配チェック
        x.requires_grad_(True)
        output, reg_loss = attn(x)
        loss = output.sum() + reg_loss
        loss.backward()
        
        assert x.grad is not None
        
        # パラメータの勾配チェック
        param_grads = [p.grad for p in attn.parameters() if p.requires_grad]
        assert all(grad is not None for grad in param_grads)
        assert all(grad.abs().sum() > 0 for grad in param_grads)
    
    def test_attention_consistency(self):
        """アテンション一貫性テスト"""
        d_model = 128
        n_heads = 4
        buckets = {"trivial": 2, "fund": 2}
        
        attn = PWA_PET_Attention(
            d_model=d_model,
            n_heads=n_heads,
            buckets=buckets
        )
        
        x = torch.randn(1, 10, d_model)
        
        # 同じ入力で複数回実行
        outputs = []
        reg_losses = []
        
        for _ in range(3):
            attn.eval()  # 評価モード
            with torch.no_grad():
                output, reg_loss = attn(x)
                outputs.append(output)
                reg_losses.append(reg_loss)
        
        # 出力が一貫していることを確認
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i]), "Outputs should be consistent"
            assert torch.allclose(reg_losses[0], reg_losses[i]), "Regularization losses should be consistent"
    
    def test_attention_memory_efficiency(self):
        """メモリ効率テスト"""
        d_model = 512
        n_heads = 8
        buckets = {"trivial": 2, "fund": 4, "adj": 2}
        
        attn = PWA_PET_Attention(
            d_model=d_model,
            n_heads=n_heads,
            buckets=buckets
        )
        
        # 大きなバッチサイズでテスト
        batch_size = 16
        seq_len = 100
        x = torch.randn(batch_size, seq_len, d_model)
        
        # メモリ使用量をチェック（実際のメモリ使用量は環境依存）
        output, reg_loss = attn(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any(), "Output should not contain NaN"
        assert not torch.isinf(output).any(), "Output should not contain Inf"


if __name__ == "__main__":
    pytest.main([__file__])
