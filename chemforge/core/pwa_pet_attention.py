"""
PWA (Peter-Weyl Attention) + PET (Phase-Enriched Transformer) アテンション実装
バケットルーティング、Q/K共有、SU2Gate、RoPE対応
分子特徴量向けに適応
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple
from .su2_gate import SU2Gate


class PWAHeadRouter:
    """
    PWAヘッドルーター: ヘッドをバケット（trivial/fund/adj）にマッピング
    """
    
    def __init__(self, n_heads: int, buckets: Dict[str, int]):
        self.n_heads = n_heads
        self.buckets = buckets
        self.head_to_bucket = {}
        self.bucket_to_heads = {bucket: [] for bucket in buckets.keys()}
        
        # ヘッドをバケットに割り当て
        head_idx = 0
        for bucket_name, bucket_size in buckets.items():
            for _ in range(bucket_size):
                if head_idx < n_heads:
                    self.head_to_bucket[head_idx] = bucket_name
                    self.bucket_to_heads[bucket_name].append(head_idx)
                    head_idx += 1
    
    def get_bucket_heads(self, bucket_name: str) -> List[int]:
        """指定されたバケットのヘッドインデックスを取得"""
        return self.bucket_to_heads.get(bucket_name, [])
    
    def get_head_bucket(self, head_idx: int) -> str:
        """指定されたヘッドのバケット名を取得"""
        return self.head_to_bucket.get(head_idx, 'trivial')
    
    def groups(self) -> Dict[str, List[int]]:
        """全バケットのヘッドグループを取得"""
        return self.bucket_to_heads.copy()


class RoPE(nn.Module):
    """
    回転位置エンコーディング (Rotary Position Embedding)
    分子グラフノード向けに適応
    """
    
    def __init__(self, head_dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        
        # 事前計算されたsin/cosテーブル
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # キャッシュ
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0
    
    def _get_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """cos/sinテーブルを取得（キャッシュ付き）"""
        if self._cos_cached is None or seq_len > self._seq_len_cached:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)
            self._seq_len_cached = seq_len
        
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        RoPEを適用
        
        Args:
            q, k: [batch, n_heads, seq_len, head_dim]
            seq_len: シーケンス長
        
        Returns:
            回転されたq, k
        """
        cos, sin = self._get_cos_sin(seq_len, q.device, q.dtype)
        
        # 回転適用
        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat((-x2, x1), dim=-1)
        
        q_rotated = q * cos.unsqueeze(0).unsqueeze(0) + rotate_half(q) * sin.unsqueeze(0).unsqueeze(0)
        k_rotated = k * cos.unsqueeze(0).unsqueeze(0) + rotate_half(k) * sin.unsqueeze(0).unsqueeze(0)
        
        return q_rotated, k_rotated


class PWA_PET_Attention(nn.Module):
    """
    PWA + PET アテンション
    分子特徴量向けに適応
    
    - PWA: バケットごとにQ/K投影を共有、Vは各ヘッド独立
    - PET: SU2GateをVチャネルに適用
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        buckets: Dict[str, int],
        dropout: float = 0.0,
        use_rope: bool = True,
        use_pet: bool = True,
        pet_curv_reg: float = 1e-5
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.use_rope = use_rope
        self.use_pet = use_pet
        self.pet_curv_reg = pet_curv_reg
        
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        # ヘッドルーター
        self.router = PWAHeadRouter(n_heads, buckets)
        
        # バケットごとのQ/K投影（共有）
        self.qk_proj = nn.ModuleDict()
        for bucket_name in buckets.keys():
            self.qk_proj[bucket_name] = nn.Linear(d_model, self.head_dim * 2, bias=False)
        
        # ヘッドごとのV投影
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        
        # 出力投影
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # RoPE
        if use_rope:
            self.rope = RoPE(self.head_dim)
        else:
            self.rope = None
        
        # PET: SU2Gate
        if use_pet:
            self.pet_gate = SU2Gate(d_model, gate_type='per_head')
        else:
            self.pet_gate = None
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        アテンション計算
        
        Args:
            x: [batch, seq_len, d_model] - 分子特徴量
            attention_mask: [batch, seq_len, seq_len] or None
        
        Returns:
            (output, regularization_loss)
        """
        batch_size, seq_len, d_model = x.shape
        reg_loss = 0.0
        
        # Q, K, V投影
        qk_outputs = {}
        for bucket_name, heads in self.router.groups().items():
            if heads:  # バケットにヘッドが存在する場合
                qk_out = self.qk_proj[bucket_name](x)  # [batch, seq_len, head_dim * 2]
                qk_out = qk_out.view(batch_size, seq_len, 2, self.head_dim)
                qk_outputs[bucket_name] = {
                    'q': qk_out[:, :, 0],  # [batch, seq_len, head_dim]
                    'k': qk_out[:, :, 1]  # [batch, seq_len, head_dim]
                }
        
        # V投影（全ヘッド独立）
        v = self.v_proj(x)  # [batch, seq_len, d_model]
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # PET: SU2GateをVに適用
        if self.pet_gate is not None:
            # vの形状: [batch, seq_len, n_heads, head_dim]
            # SU2ゲートは最後の次元でペアを作るので、head_dimが偶数である必要がある
            v_reshaped = v.view(batch_size, seq_len, -1)  # [batch, seq_len, d_model]
            v_transformed = self.pet_gate(v_reshaped)
            v = v_transformed.view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # アテンション計算（バケットごと）
        attention_outputs = []
        
        for bucket_name, heads in self.router.groups().items():
            if heads and bucket_name in qk_outputs:
                q = qk_outputs[bucket_name]['q']  # [batch, seq_len, head_dim]
                k = qk_outputs[bucket_name]['k']  # [batch, seq_len, head_dim]
                
                # ヘッド次元を追加
                q = q.unsqueeze(1).expand(-1, len(heads), -1, -1)  # [batch, n_heads_in_bucket, seq_len, head_dim]
                k = k.unsqueeze(1).expand(-1, len(heads), -1, -1)  # [batch, n_heads_in_bucket, seq_len, head_dim]
                
                # RoPE適用
                if self.rope is not None:
                    q, k = self.rope(q, k, seq_len)
                
                # アテンション計算
                v_bucket = v[:, :, heads, :]  # [batch, seq_len, n_heads_in_bucket, head_dim]
                v_bucket = v_bucket.transpose(1, 2)  # [batch, n_heads_in_bucket, seq_len, head_dim]
                
                # Flash Attention確認（1回だけ）
                if not hasattr(self, "_sdp_once"):
                    print(f"Flash SDP: flash={torch.backends.cuda.flash_sdp_enabled()}, "
                          f"mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()}, "
                          f"math={torch.backends.cuda.math_sdp_enabled()}")
                    self._sdp_once = True
                
                # AMPとのdtype衝突を避ける：q/k/vのdtypeを揃える
                dty = torch.promote_types(torch.promote_types(q.dtype, k.dtype), v_bucket.dtype)
                q, k, v_bucket = q.to(dty), k.to(dty), v_bucket.to(dty)
                
                # scaled_dot_product_attention使用
                attn_output = F.scaled_dot_product_attention(
                    q, k, v_bucket,
                    attn_mask=attention_mask,
                    dropout_p=self.dropout if self.training else 0.0
                )  # [batch, n_heads_in_bucket, seq_len, head_dim]
                
                attention_outputs.append((heads, attn_output))
        
        # 出力を結合
        output = torch.zeros(batch_size, seq_len, self.n_heads, self.head_dim, device=x.device, dtype=v.dtype)
        for heads, attn_out in attention_outputs:
            # attn_out: [batch, n_heads_in_bucket, seq_len, head_dim]
            # transposeして[batch, seq_len, n_heads_in_bucket, head_dim]に変換
            attn_out = attn_out.transpose(1, 2)
            output[:, :, heads, :] = attn_out
        
        # 出力投影
        output = output.view(batch_size, seq_len, d_model)
        output = self.out_proj(output)
        
        # 正則化損失（PET curvature regularization）
        if self.pet_gate is not None and self.pet_curv_reg > 0:
            reg_loss = self.pet_curv_reg * torch.norm(self.pet_gate.params)
        
        return output, reg_loss


def test_attention():
    """アテンションモジュールのテスト"""
    print("Testing PWA_PET_Attention...")
    
    # テストパラメータ
    batch_size, seq_len, d_model = 2, 50, 512  # 分子グラフ向けに調整
    n_heads = 8
    buckets = {"trivial": 2, "fund": 4, "adj": 2}
    
    # アテンション作成
    attn = PWA_PET_Attention(
        d_model=d_model,
        n_heads=n_heads,
        buckets=buckets,
        dropout=0.0,
        use_rope=True,
        use_pet=True
    )
    
    # テスト入力
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 前向きパス
    output, reg_loss = attn(x)
    
    # 形状チェック
    assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
    assert isinstance(reg_loss, torch.Tensor), "Regularization loss should be a tensor"
    
    # 勾配チェック
    x.requires_grad_(True)
    output, reg_loss = attn(x)
    loss = output.sum() + reg_loss
    loss.backward()
    
    assert x.grad is not None, "Gradient should flow through attention"
    
    print("✓ PWA_PET_Attention test passed")
    print(f"  Output shape: {output.shape}")
    print(f"  Regularization loss: {reg_loss.item():.6f}")
    print(f"  Router groups: {attn.router.groups()}")


if __name__ == "__main__":
    test_attention()
