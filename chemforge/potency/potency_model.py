"""
Potency PWA+PET Model

pIC50/pKi力価回帰専用のPWA+PET Transformerモデル
既存のtransformer_model.pyを活用し、マルチタスク出力を実装
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import math

# 既存のPWA+PETモジュールをインポート
from ..core.transformer_model import PWAPETEncoderLayer, PWAPETTransformer
from ..core.pwa_pet_attention import PWA_PET_Attention
from ..core.su2_gate import SU2Gate

class PotencyPWAPETModel(nn.Module):
    """pIC50/pKi力価回帰専用PWA+PET Transformerモデル"""
    
    def __init__(self, config: Dict):
        """
        初期化
        
        Args:
            config: モデル設定辞書
        """
        super().__init__()
        
        # 設定
        self.config = config
        self.d_model = config.get('d_model', 512)
        self.n_layers = config.get('n_layers', 8)
        self.n_heads = config.get('n_heads', 8)
        self.d_ff = config.get('d_ff', 2048)
        self.dropout = config.get('dropout', 0.0)
        self.max_len = config.get('max_len', 256)
        self.vocab_size = config.get('vocab_size', 1000)
        self.num_physchem = config.get('num_physchem', 8)
        self.num_3d_features = config.get('num_3d_features', 10)
        self.use_3d = config.get('use_3d', False)
        
        # PWA+PET設定
        self.rope = config.get('rope', True)
        self.buckets = config.get('buckets', {"trivial": 1, "fund": 5, "adj": 2})
        self.pet_curv_reg = config.get('pet_curv_reg', 1e-6)
        
        # 埋め込み層
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.physchem_embedding = nn.Linear(self.num_physchem, self.d_model // 4)
        if self.use_3d:
            self.features_3d_embedding = nn.Linear(self.num_3d_features, self.d_model // 4)
        
        # 位置エンコーディング
        if self.rope:
            self.rope_emb = self._create_rope_embeddings()
        else:
            self.pos_embedding = nn.Embedding(self.max_len, self.d_model)
        
        # PWA+PET Transformer
        self.transformer = self._create_transformer()
        
        # マルチタスク出力ヘッド
        self.reg_pIC50_head = nn.Linear(self.d_model, 1)
        self.reg_pKi_head = nn.Linear(self.d_model, 1)
        self.cls_pIC50_head = nn.Linear(self.d_model, 1)
        self.cls_pKi_head = nn.Linear(self.d_model, 1)
        
        # 活性化関数
        self.activation = nn.SiLU()
        
        # ドロップアウト
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # 重み初期化
        self._init_weights()
    
    def _create_rope_embeddings(self):
        """RoPE埋め込み作成"""
        # 簡易版RoPE実装
        return None
    
    def _create_transformer(self):
        """PWA+PET Transformer作成"""
        # PWA+PET Attention作成
        attention = PWA_PET_Attention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            buckets=self.buckets,
            dropout=self.dropout
        )
        
        # フィードフォワード
        feedforward = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_ff, self.d_model)
        )
        
        # 正規化層
        norm1 = nn.RMSNorm(self.d_model)
        norm2 = nn.RMSNorm(self.d_model)
        
        # エンコーダー層
        encoder_layer = PWAPETEncoderLayer(
            attention=attention,
            feedforward=feedforward,
            norm1=norm1,
            norm2=norm2,
            dropout=self.dropout
        )
        
        # 複数層スタック
        return nn.ModuleList([encoder_layer for _ in range(self.n_layers)])
    
    def _init_weights(self):
        """重み初期化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向きパス
        
        Args:
            batch: バッチ辞書（tokens, physchem, features_3d, target, mask等）
            
        Returns:
            予測結果辞書
        """
        # 入力取得
        tokens = batch['tokens']  # [batch, seq_len]
        physchem = batch['physchem']  # [batch, num_physchem]
        features_3d = batch.get('features_3d')  # [batch, num_3d_features] or None
        
        batch_size, seq_len = tokens.shape
        
        # トークン埋め込み
        x = self.token_embedding(tokens)  # [batch, seq_len, d_model]
        
        # 物化記述子埋め込み
        physchem_emb = self.physchem_embedding(physchem)  # [batch, d_model//4]
        physchem_emb = physchem_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, d_model//4]
        
        # 3D特徴埋め込み（オプション）
        if self.use_3d and features_3d is not None:
            features_3d_emb = self.features_3d_embedding(features_3d)  # [batch, d_model//4]
            features_3d_emb = features_3d_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, d_model//4]
            
            # 特徴量統合
            x = torch.cat([x, physchem_emb, features_3d_emb], dim=-1)  # [batch, seq_len, d_model + d_model//4 + d_model//4]
            # 次元調整
            if x.shape[-1] != self.d_model:
                x = F.linear(x, torch.randn(self.d_model, x.shape[-1]).to(x.device))
        else:
            # 物化記述子のみ統合
            x = torch.cat([x, physchem_emb], dim=-1)  # [batch, seq_len, d_model + d_model//4]
            # 次元調整
            if x.shape[-1] != self.d_model:
                x = F.linear(x, torch.randn(self.d_model, x.shape[-1]).to(x.device))
        
        # 位置エンコーディング
        if self.rope:
            # RoPE実装（簡易版）
            pass
        else:
            pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
            pos_emb = self.pos_embedding(pos_ids)
            x = x + pos_emb
        
        # ドロップアウト
        x = self.dropout_layer(x)
        
        # PWA+PET Transformer
        total_reg_loss = 0.0
        for layer in self.transformer:
            x, reg_loss = layer(x)
            total_reg_loss += reg_loss
        
        # グローバル平均プーリング
        x_pooled = x.mean(dim=1)  # [batch, d_model]
        
        # マルチタスク出力
        reg_pIC50 = self.reg_pIC50_head(x_pooled)  # [batch, 1]
        reg_pKi = self.reg_pKi_head(x_pooled)  # [batch, 1]
        cls_pIC50 = torch.sigmoid(self.cls_pIC50_head(x_pooled))  # [batch, 1]
        cls_pKi = torch.sigmoid(self.cls_pKi_head(x_pooled))  # [batch, 1]
        
        return {
            'reg_pIC50': reg_pIC50,
            'reg_pKi': reg_pKi,
            'cls_pIC50': cls_pIC50,
            'cls_pKi': cls_pKi,
            'regularization_loss': total_reg_loss
        }
    
    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        推論用前向きパス
        
        Args:
            batch: バッチ辞書
            
        Returns:
            予測結果辞書
        """
        self.eval()
        with torch.no_grad():
            return self.forward(batch)
    
    def get_model_size(self) -> Dict[str, int]:
        """モデルサイズ情報取得"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # float32想定
        }

class PotencyModelConfig:
    """力価回帰モデル設定クラス"""
    
    def __init__(self, config_dict: Optional[Dict] = None):
        """
        初期化
        
        Args:
            config_dict: 設定辞書
        """
        self.config = config_dict or {}
        
        # デフォルト設定
        self.d_model = self.config.get('d_model', 512)
        self.n_layers = self.config.get('n_layers', 8)
        self.n_heads = self.config.get('n_heads', 8)
        self.d_ff = self.config.get('d_ff', 2048)
        self.dropout = self.config.get('dropout', 0.0)
        self.max_len = self.config.get('max_len', 256)
        self.vocab_size = self.config.get('vocab_size', 1000)
        self.num_physchem = self.config.get('num_physchem', 8)
        self.num_3d_features = self.config.get('num_3d_features', 10)
        self.use_3d = self.config.get('use_3d', False)
        
        # PWA+PET設定
        self.rope = self.config.get('rope', True)
        self.buckets = self.config.get('buckets', {"trivial": 1, "fund": 5, "adj": 2})
        self.pet_curv_reg = self.config.get('pet_curv_reg', 1e-6)
    
    def to_dict(self) -> Dict:
        """設定辞書に変換"""
        return {
            'd_model': self.d_model,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'd_ff': self.d_ff,
            'dropout': self.dropout,
            'max_len': self.max_len,
            'vocab_size': self.vocab_size,
            'num_physchem': self.num_physchem,
            'num_3d_features': self.num_3d_features,
            'use_3d': self.use_3d,
            'rope': self.rope,
            'buckets': self.buckets,
            'pet_curv_reg': self.pet_curv_reg
        }

def create_potency_model(config: Dict) -> PotencyPWAPETModel:
    """
    力価回帰モデル作成
    
    Args:
        config: モデル設定
        
    Returns:
        PotencyPWAPETModel
    """
    return PotencyPWAPETModel(config)

def load_potency_model(model_path: str, config: Dict, device: str = 'cuda') -> PotencyPWAPETModel:
    """
    学習済みモデル読み込み
    
    Args:
        model_path: モデルファイルパス
        config: モデル設定
        device: デバイス
        
    Returns:
        読み込み済みモデル
    """
    model = PotencyPWAPETModel(config)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def save_potency_model(model: PotencyPWAPETModel, save_path: str, 
                      optimizer_state: Optional[Dict] = None,
                      epoch: Optional[int] = None,
                      loss: Optional[float] = None):
    """
    モデル保存
    
    Args:
        model: 保存するモデル
        save_path: 保存パス
        optimizer_state: オプティマイザー状態
        epoch: エポック数
        loss: 損失値
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': model.config,
        'model_size': model.get_model_size()
    }
    
    if optimizer_state is not None:
        checkpoint['optimizer_state_dict'] = optimizer_state
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if loss is not None:
        checkpoint['loss'] = loss
    
    torch.save(checkpoint, save_path)
    print(f"[INFO] モデル保存完了: {save_path}")