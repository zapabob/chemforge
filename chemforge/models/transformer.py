"""
Transformer Model Module

Transformer深層学習モデル実装
既存PWA_PET_Attentionを活用した効率的なTransformer実装
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
from chemforge.core.pwa_pet_attention import PWA_PET_Attention
from chemforge.core.su2_gate import SU2Gate
from chemforge.utils.config_utils import ConfigManager
from chemforge.utils.logging_utils import Logger
from chemforge.utils.validation import DataValidator

logger = logging.getLogger(__name__)

class TransformerBlock(nn.Module):
    """
    Transformer Block
    
    既存PWA_PET_Attentionを活用したTransformer Block
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                 dropout: float = 0.1, activation: str = "relu",
                 use_pwa_pet: bool = True, pwa_buckets: Optional[Dict] = None,
                 pet_curv_reg: float = 1e-6):
        """
        初期化
        
        Args:
            d_model: モデル次元
            n_heads: アテンションヘッド数
            d_ff: フィードフォワード次元
            dropout: ドロップアウト率
            activation: 活性化関数
            use_pwa_pet: PWA+PET使用フラグ
            pwa_buckets: PWAバケット設定
            pet_curv_reg: PET曲率正則化
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        self.use_pwa_pet = use_pwa_pet
        self.pwa_buckets = pwa_buckets or {"trivial": 1, "fund": 5, "adj": 2}
        self.pet_curv_reg = pet_curv_reg
        
        # 既存PWA_PET_Attention活用
        if use_pwa_pet:
            self.attention = PWA_PET_Attention(
                d_model=d_model,
                n_heads=n_heads,
                pwa_buckets=pwa_buckets,
                pet_curv_reg=pet_curv_reg
            )
        else:
            self.attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True
            )
        
        # フィードフォワード
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            self._get_activation(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # 正規化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # ドロップアウト
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def _get_activation(self):
        """活性化関数取得"""
        if self.activation == "relu":
            return nn.ReLU()
        elif self.activation == "gelu":
            return nn.GELU()
        elif self.activation == "swish":
            return nn.SiLU()
        else:
            return nn.ReLU()
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        フォワードパス
        
        Args:
            x: 入力テンソル [batch_size, seq_len, d_model]
            mask: アテンションマスク
            
        Returns:
            出力テンソル
        """
        # アテンション
        if self.use_pwa_pet:
            attn_out = self.attention(x, mask)
        else:
            attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        
        # 残差接続 + 正規化
        x = self.norm1(x + self.dropout1(attn_out))
        
        # フィードフォワード
        ff_out = self.ff(x)
        
        # 残差接続 + 正規化
        x = self.norm2(x + self.dropout2(ff_out))
        
        return x

class TransformerEncoder(nn.Module):
    """
    Transformer Encoder
    
    複数のTransformer Blockを組み合わせたEncoder
    """
    
    def __init__(self, d_model: int, n_heads: int, n_layers: int, d_ff: int,
                 dropout: float = 0.1, activation: str = "relu",
                 use_pwa_pet: bool = True, pwa_buckets: Optional[Dict] = None,
                 pet_curv_reg: float = 1e-6):
        """
        初期化
        
        Args:
            d_model: モデル次元
            n_heads: アテンションヘッド数
            n_layers: レイヤー数
            d_ff: フィードフォワード次元
            dropout: ドロップアウト率
            activation: 活性化関数
            use_pwa_pet: PWA+PET使用フラグ
            pwa_buckets: PWAバケット設定
            pet_curv_reg: PET曲率正則化
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        self.use_pwa_pet = use_pwa_pet
        self.pwa_buckets = pwa_buckets
        self.pet_curv_reg = pet_curv_reg
        
        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                use_pwa_pet=use_pwa_pet,
                pwa_buckets=pwa_buckets,
                pet_curv_reg=pet_curv_reg
            )
            for _ in range(n_layers)
        ])
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        フォワードパス
        
        Args:
            x: 入力テンソル [batch_size, seq_len, d_model]
            mask: アテンションマスク
            
        Returns:
            出力テンソル
        """
        for block in self.blocks:
            x = block(x, mask)
        
        return x

class MolecularTransformer(nn.Module):
    """
    Molecular Transformer
    
    分子データ用のTransformerモデル
    """
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, max_len: int = 512, dropout: float = 0.1,
                 activation: str = "relu", use_pwa_pet: bool = True,
                 pwa_buckets: Optional[Dict] = None, pet_curv_reg: float = 1e-6,
                 use_rope: bool = True):
        """
        初期化
        
        Args:
            vocab_size: 語彙サイズ
            d_model: モデル次元
            n_heads: アテンションヘッド数
            n_layers: レイヤー数
            d_ff: フィードフォワード次元
            max_len: 最大シーケンス長
            dropout: ドロップアウト率
            activation: 活性化関数
            use_pwa_pet: PWA+PET使用フラグ
            pwa_buckets: PWAバケット設定
            pet_curv_reg: PET曲率正則化
            use_rope: RoPE使用フラグ
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_len = max_len
        self.dropout = dropout
        self.activation = activation
        self.use_pwa_pet = use_pwa_pet
        self.pwa_buckets = pwa_buckets
        self.pet_curv_reg = pet_curv_reg
        self.use_rope = use_rope
        
        # 埋め込み層
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        # ドロップアウト
        self.dropout = nn.Dropout(dropout)
        
        # Transformer Encoder
        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            use_pwa_pet=use_pwa_pet,
            pwa_buckets=pwa_buckets,
            pet_curv_reg=pet_curv_reg
        )
        
        # 出力層
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # 重み初期化
        self._init_weights()
        
    def _init_weights(self):
        """重み初期化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        フォワードパス
        
        Args:
            input_ids: 入力トークンID [batch_size, seq_len]
            attention_mask: アテンションマスク [batch_size, seq_len]
            
        Returns:
            出力ロジット [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # 位置エンコーディング
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        positions = positions.expand(batch_size, -1)
        
        # 埋め込み
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        
        # 埋め込み結合
        x = token_emb + pos_emb
        x = self.dropout(x)
        
        # アテンションマスク作成
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            mask = mask.expand(batch_size, 1, seq_len, seq_len)
            mask = mask.to(torch.bool)
        else:
            mask = None
        
        # Transformer Encoder
        x = self.encoder(x, mask)
        
        # 出力投影
        logits = self.output_projection(x)
        
        return logits
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 100,
                temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9,
                pad_token_id: int = 0, eos_token_id: int = 1) -> torch.Tensor:
        """
        テキスト生成
        
        Args:
            input_ids: 初期入力トークンID
            max_length: 最大生成長
            temperature: 温度パラメータ
            top_k: Top-kサンプリング
            top_p: Top-pサンプリング
            pad_token_id: パディングトークンID
            eos_token_id: 終了トークンID
            
        Returns:
            生成されたトークンID
        """
        self.eval()
        
        with torch.no_grad():
            generated = input_ids.clone()
            
            for _ in range(max_length):
                # 現在のシーケンスで予測
                logits = self.forward(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-k フィルタリング
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Top-p フィルタリング
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # サンプリング
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 生成シーケンスに追加
                generated = torch.cat([generated, next_token], dim=1)
                
                # 終了条件チェック
                if (next_token == eos_token_id).all():
                    break
            
            return generated

class PotencyTransformer(nn.Module):
    """
    Potency Prediction Transformer
    
    力価予測用のTransformerモデル
    """
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, max_len: int = 512, dropout: float = 0.1,
                 activation: str = "relu", use_pwa_pet: bool = True,
                 pwa_buckets: Optional[Dict] = None, pet_curv_reg: float = 1e-6,
                 use_rope: bool = True, num_tasks: int = 2):
        """
        初期化
        
        Args:
            vocab_size: 語彙サイズ
            d_model: モデル次元
            n_heads: アテンションヘッド数
            n_layers: レイヤー数
            d_ff: フィードフォワード次元
            max_len: 最大シーケンス長
            dropout: ドロップアウト率
            activation: 活性化関数
            use_pwa_pet: PWA+PET使用フラグ
            pwa_buckets: PWAバケット設定
            pet_curv_reg: PET曲率正則化
            use_rope: RoPE使用フラグ
            num_tasks: タスク数
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_len = max_len
        self.dropout = dropout
        self.activation = activation
        self.use_pwa_pet = use_pwa_pet
        self.pwa_buckets = pwa_buckets
        self.pet_curv_reg = pet_curv_reg
        self.use_rope = use_rope
        self.num_tasks = num_tasks
        
        # 埋め込み層
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        # ドロップアウト
        self.dropout = nn.Dropout(dropout)
        
        # Transformer Encoder
        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            use_pwa_pet=use_pwa_pet,
            pwa_buckets=pwa_buckets,
            pet_curv_reg=pet_curv_reg
        )
        
        # タスク別出力層
        self.regression_head = nn.Linear(d_model, 1)
        self.classification_head = nn.Linear(d_model, 1)
        
        # 重み初期化
        self._init_weights()
        
    def _init_weights(self):
        """重み初期化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        フォワードパス
        
        Args:
            input_ids: 入力トークンID [batch_size, seq_len]
            attention_mask: アテンションマスク [batch_size, seq_len]
            
        Returns:
            出力辞書
        """
        batch_size, seq_len = input_ids.shape
        
        # 位置エンコーディング
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        positions = positions.expand(batch_size, -1)
        
        # 埋め込み
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        
        # 埋め込み結合
        x = token_emb + pos_emb
        x = self.dropout(x)
        
        # アテンションマスク作成
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            mask = mask.expand(batch_size, 1, seq_len, seq_len)
            mask = mask.to(torch.bool)
        else:
            mask = None
        
        # Transformer Encoder
        x = self.encoder(x, mask)
        
        # グローバル平均プーリング
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(x)
            x = x * mask_expanded
            pooled = x.sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            pooled = x.mean(dim=1)
        
        # タスク別出力
        regression_output = self.regression_head(pooled)
        classification_output = self.classification_head(pooled)
        
        return {
            'regression': regression_output.squeeze(-1),
            'classification': torch.sigmoid(classification_output.squeeze(-1))
        }

def create_transformer(vocab_size: int, d_model: int = 512, n_heads: int = 8,
                      n_layers: int = 6, d_ff: int = 2048, max_len: int = 512,
                      dropout: float = 0.1, activation: str = "relu",
                      use_pwa_pet: bool = True, pwa_buckets: Optional[Dict] = None,
                      pet_curv_reg: float = 1e-6, use_rope: bool = True) -> MolecularTransformer:
    """
    Transformer作成
    
    Args:
        vocab_size: 語彙サイズ
        d_model: モデル次元
        n_heads: アテンションヘッド数
        n_layers: レイヤー数
        d_ff: フィードフォワード次元
        max_len: 最大シーケンス長
        dropout: ドロップアウト率
        activation: 活性化関数
        use_pwa_pet: PWA+PET使用フラグ
        pwa_buckets: PWAバケット設定
        pet_curv_reg: PET曲率正則化
        use_rope: RoPE使用フラグ
        
    Returns:
        MolecularTransformer
    """
    return MolecularTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout=dropout,
        activation=activation,
        use_pwa_pet=use_pwa_pet,
        pwa_buckets=pwa_buckets,
        pet_curv_reg=pet_curv_reg,
        use_rope=use_rope
    )

def create_potency_transformer(vocab_size: int, d_model: int = 512, n_heads: int = 8,
                              n_layers: int = 6, d_ff: int = 2048, max_len: int = 512,
                              dropout: float = 0.1, activation: str = "relu",
                              use_pwa_pet: bool = True, pwa_buckets: Optional[Dict] = None,
                              pet_curv_reg: float = 1e-6, use_rope: bool = True,
                              num_tasks: int = 2) -> PotencyTransformer:
    """
    力価予測Transformer作成
    
    Args:
        vocab_size: 語彙サイズ
        d_model: モデル次元
        n_heads: アテンションヘッド数
        n_layers: レイヤー数
        d_ff: フィードフォワード次元
        max_len: 最大シーケンス長
        dropout: ドロップアウト率
        activation: 活性化関数
        use_pwa_pet: PWA+PET使用フラグ
        pwa_buckets: PWAバケット設定
        pet_curv_reg: PET曲率正則化
        use_rope: RoPE使用フラグ
        num_tasks: タスク数
        
    Returns:
        PotencyTransformer
    """
    return PotencyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout=dropout,
        activation=activation,
        use_pwa_pet=use_pwa_pet,
        pwa_buckets=pwa_buckets,
        pet_curv_reg=pet_curv_reg,
        use_rope=use_rope,
        num_tasks=num_tasks
    )

if __name__ == "__main__":
    # テスト実行
    vocab_size = 1000
    d_model = 512
    n_heads = 8
    n_layers = 6
    d_ff = 2048
    max_len = 512
    
    # 基本Transformer
    transformer = create_transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_len=max_len
    )
    
    print(f"Transformer created: {transformer}")
    print(f"Parameters: {sum(p.numel() for p in transformer.parameters()):,}")
    
    # 力価予測Transformer
    potency_transformer = create_potency_transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_len=max_len
    )
    
    print(f"Potency Transformer created: {potency_transformer}")
    print(f"Parameters: {sum(p.numel() for p in potency_transformer.parameters()):,}")
    
    # テスト実行
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # 基本Transformer
    with torch.no_grad():
        logits = transformer(input_ids, attention_mask)
        print(f"Basic Transformer output shape: {logits.shape}")
    
    # 力価予測Transformer
    with torch.no_grad():
        outputs = potency_transformer(input_ids, attention_mask)
        print(f"Potency Transformer outputs: {outputs.keys()}")
        print(f"Regression output shape: {outputs['regression'].shape}")
        print(f"Classification output shape: {outputs['classification'].shape}")
