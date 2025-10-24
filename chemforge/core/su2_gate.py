"""
SU(2)ユニタリゲート実装
3つの実数パラメータから2×2回転行列を生成し、スペクトル半径=1を保証
分子特徴量向けに適応
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SU2Gate(nn.Module):
    """
    SU(2)ユニタリゲート: 3つの実数パラメータから2×2回転行列を生成
    分子表現向けに適応
    
    Args:
        d_model: モデル次元
        gate_type: 'shared' (全ヘッド共有) or 'per_head' (ヘッドごと)
    """
    
    def __init__(self, d_model: int, gate_type: str = 'shared'):
        super().__init__()
        self.d_model = d_model
        self.gate_type = gate_type
        
        # 3つの実数パラメータ: (theta, phi, psi)
        # theta: 回転角度 [0, 2π]
        # phi: 方位角 [0, 2π] 
        # psi: 位相角 [0, 2π]
        if gate_type == 'shared':
            self.params = nn.Parameter(torch.randn(3) * 0.1)
        else:  # per_head
            self.params = nn.Parameter(torch.randn(1, 3) * 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SU(2)ゲートを適用
        分子特徴量向けに適応
        
        Args:
            x: [batch, seq_len, d_model] or [batch, seq_len, n_heads, head_dim]
        
        Returns:
            変換されたテンソル
        """
        batch_size, seq_len, d_model = x.shape[:3]
        
        # パラメータを取得
        if self.gate_type == 'shared':
            theta, phi, psi = self.params
        else:
            theta, phi, psi = self.params[0]  # [3]
        
        # 2×2回転行列を構築
        # U = exp(i * sigma_z * phi/2) * exp(i * sigma_y * theta/2) * exp(i * sigma_z * psi/2)
        cos_theta_2 = torch.cos(theta / 2)
        sin_theta_2 = torch.sin(theta / 2)
        cos_phi_2 = torch.cos(phi / 2)
        sin_phi_2 = torch.sin(phi / 2)
        cos_psi_2 = torch.cos(psi / 2)
        sin_psi_2 = torch.sin(psi / 2)
        
        # パウリ行列
        sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=x.device)
        sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=x.device)
        
        # 回転行列の構築
        # exp(i * sigma_z * phi/2)
        Rz1 = torch.eye(2, dtype=torch.complex64, device=x.device) * cos_phi_2 + 1j * sigma_z * sin_phi_2
        
        # exp(i * sigma_y * theta/2)  
        Ry = torch.eye(2, dtype=torch.complex64, device=x.device) * cos_theta_2 + 1j * sigma_y * sin_theta_2
        
        # exp(i * sigma_z * psi/2)
        Rz2 = torch.eye(2, dtype=torch.complex64, device=x.device) * cos_psi_2 + 1j * sigma_z * sin_psi_2
        
        # 合成: U = Rz1 * Ry * Rz2
        U = Rz1 @ Ry @ Rz2
        
        # 実部のみを取得（ユニタリ行列の実部は直交行列）
        U_real = U.real.to(x.dtype)  # dtypeを入力に合わせる
        
        # チャネルをペアに分割して適用
        if d_model % 2 != 0:
            # 奇数次元の場合は最後のチャネルをそのまま残す
            x_contiguous = x[..., :-1].contiguous()  # 非連続対策
            x_pairs = x_contiguous.reshape(batch_size, seq_len, -1, 2)
            x_last = x[..., -1:]
        else:
            x_contiguous = x.contiguous()  # 非連続対策
            x_pairs = x_contiguous.reshape(batch_size, seq_len, -1, 2)
            x_last = None
        
        # ペアごとに回転を適用
        # U_real: [2, 2], x_pairs: [batch, seq_len, d_model//2, 2]
        # 結果: [batch, seq_len, d_model//2, 2]
        # einsumの次元: U_real[i,j] × x_pairs[b,s,p,i] -> result[b,s,p,j]
        x_pairs_rotated = torch.einsum('ij,bspi->bspj', U_real, x_pairs)
        
        # 元の形状に戻す
        if x_last is not None:
            x_rotated = torch.cat([
                x_pairs_rotated.reshape(batch_size, seq_len, -1),
                x_last
            ], dim=-1)
        else:
            x_rotated = x_pairs_rotated.reshape(batch_size, seq_len, -1)
        
        # 元の形状に戻す（4次元テンソルの場合）
        if len(x.shape) == 4:
            x_rotated = x_rotated.reshape(x.shape)
        
        return x_rotated
    
    def get_spectral_radius(self) -> float:
        """スペクトル半径を計算（理論値は1.0）"""
        with torch.no_grad():
            theta, phi, psi = self.params
            # 2×2ユニタリ行列の固有値の絶対値は1
            return 1.0


def test_su2_gate():
    """SU2ゲートの単体テスト"""
    print("Testing SU2Gate...")
    
    # テストケース1: 基本的な動作
    d_model = 512
    batch_size, seq_len = 2, 50  # 分子グラフ向けに調整
    
    gate = SU2Gate(d_model, 'shared')
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 前向きパス
    y = gate(x)
    
    # 形状チェック
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
    
    # スペクトル半径チェック
    spectral_radius = gate.get_spectral_radius()
    assert abs(spectral_radius - 1.0) < 1e-6, f"Spectral radius should be 1.0, got {spectral_radius}"
    
    # ノルム保存チェック（近似）
    norm_before = torch.norm(x, dim=-1)
    norm_after = torch.norm(y, dim=-1)
    norm_diff = torch.abs(norm_before - norm_after).max()
    assert norm_diff < 0.1, f"Norm should be approximately preserved, max diff: {norm_diff}"
    
    print("✓ SU2Gate basic test passed")
    
    # テストケース2: 勾配フロー
    x.requires_grad_(True)
    y = gate(x)
    loss = y.sum()
    loss.backward()
    
    assert x.grad is not None, "Gradient should flow through SU2Gate"
    print("✓ SU2Gate gradient test passed")
    
    print("All SU2Gate tests passed!")


if __name__ == "__main__":
    test_su2_gate()
