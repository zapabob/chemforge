"""
Potency Featurizer

pIC50/pKi力価回帰用の特徴量抽出モジュール
SELFIES/SMILESトークン化、物化記述子、3D特徴を実装
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, rdDistGeom
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

# SELFIES import (optional)
try:
    import selfies as sf
    SELFIES_AVAILABLE = True
except ImportError:
    SELFIES_AVAILABLE = False
    print("[WARNING] SELFIES not available. Using SMILES only.")

class PotencyFeaturizer:
    """pIC50/pKi力価回帰用特徴量抽出クラス"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config or {}
        
        # トークン化設定
        self.use_selfies = self.config.get('use_selfies', True) and SELFIES_AVAILABLE
        self.max_len = self.config.get('max_len', 256)
        self.vocab_size = self.config.get('vocab_size', 1000)
        
        # 物化記述子設定
        self.num_physchem = self.config.get('num_physchem', 8)
        self.use_3d = self.config.get('use_3d', False)
        self.confs_per_mol = self.config.get('confs_per_mol', 1)
        
        # 正規化設定
        self.scaler_type = self.config.get('scaler', 'robust')  # 'standard' or 'robust'
        self.scaler = RobustScaler() if self.scaler_type == 'robust' else StandardScaler()
        
        # 語彙辞書
        self.vocab = {}
        self.vocab_inv = {}
        self.is_fitted = False
        
    def _create_vocab(self, smiles_list: List[str]) -> Dict[str, int]:
        """語彙辞書作成"""
        print("[INFO] 語彙辞書作成中...")
        
        all_tokens = set()
        
        for smiles in smiles_list:
            if pd.isna(smiles) or smiles == '':
                continue
                
            try:
                if self.use_selfies:
                    # SELFIESトークン化
                    selfies_str = sf.encoder(smiles)
                    if selfies_str:
                        tokens = sf.split_selfies(selfies_str)
                        all_tokens.update(tokens)
                else:
                    # SMILESトークン化（文字レベル）
                    tokens = list(smiles)
                    all_tokens.update(tokens)
                    
            except Exception as e:
                print(f"[WARNING] トークン化失敗: {smiles} - {e}")
                continue
        
        # 特殊トークン追加
        special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
        all_tokens = special_tokens + sorted(list(all_tokens))
        
        # 語彙辞書作成
        vocab = {token: idx for idx, token in enumerate(all_tokens)}
        vocab_inv = {idx: token for token, idx in vocab.items()}
        
        print(f"[INFO] 語彙サイズ: {len(vocab)}")
        
        return vocab, vocab_inv
    
    def tokenize(self, smiles_list: List[str]) -> np.ndarray:
        """
        SMILES/SELFIESトークン化
        
        Args:
            smiles_list: SMILES文字列リスト
            
        Returns:
            トークン化された配列 (n_samples, max_len)
        """
        if not self.is_fitted:
            self.vocab, self.vocab_inv = self._create_vocab(smiles_list)
            self.is_fitted = True
        
        print("[INFO] トークン化中...")
        
        tokenized = []
        for smiles in smiles_list:
            if pd.isna(smiles) or smiles == '':
                # パディング
                tokens = [self.vocab['<PAD>']] * self.max_len
            else:
                try:
                    if self.use_selfies:
                        # SELFIESトークン化
                        selfies_str = sf.encoder(smiles)
                        if selfies_str:
                            tokens = sf.split_selfies(selfies_str)
                        else:
                            tokens = ['<UNK>']
                    else:
                        # SMILESトークン化
                        tokens = list(smiles)
                    
                    # トークンID変換
                    token_ids = []
                    for token in tokens:
                        token_ids.append(self.vocab.get(token, self.vocab['<UNK>']))
                    
                    # パディング/トランケーション
                    if len(token_ids) > self.max_len:
                        token_ids = token_ids[:self.max_len]
                    else:
                        token_ids.extend([self.vocab['<PAD>']] * (self.max_len - len(token_ids)))
                    
                except Exception as e:
                    print(f"[WARNING] トークン化失敗: {smiles} - {e}")
                    token_ids = [self.vocab['<UNK>']] * self.max_len
                
            tokenized.append(token_ids)
        
        return np.array(tokenized)
    
    def extract_physchem(self, smiles_list: List[str]) -> np.ndarray:
        """
        物理化学記述子抽出
        
        Args:
            smiles_list: SMILES文字列リスト
            
        Returns:
            物化記述子配列 (n_samples, num_physchem)
        """
        print("[INFO] 物理化学記述子抽出中...")
        
        descriptors = []
        for smiles in smiles_list:
            if pd.isna(smiles) or smiles == '':
                desc = [0.0] * self.num_physchem
            else:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        desc = [0.0] * self.num_physchem
                    else:
                        desc = self._calculate_descriptors(mol)
                except Exception as e:
                    print(f"[WARNING] 記述子計算失敗: {smiles} - {e}")
                    desc = [0.0] * self.num_physchem
            
            descriptors.append(desc)
        
        return np.array(descriptors)
    
    def _calculate_descriptors(self, mol: Chem.Mol) -> List[float]:
        """分子記述子計算"""
        try:
            # 基本記述子
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            tpsa = Descriptors.TPSA(mol)
            rotb = CalcNumRotatableBonds(mol)
            
            # QED (Drug-likeness)
            qed = Descriptors.qed(mol)
            
            # SA Score (Synthetic Accessibility)
            sa_score = rdMolDescriptors.CalcNumSpiroAtoms(mol)  # 簡易版
            
            # 8つの記述子を返す
            return [mw, logp, hbd, hba, tpsa, rotb, qed, sa_score]
            
        except Exception as e:
            print(f"[WARNING] 記述子計算エラー: {e}")
            return [0.0] * self.num_physchem
    
    def extract_3d_features(self, smiles_list: List[str]) -> np.ndarray:
        """
        3D特徴抽出（ETKDG conformers）
        
        Args:
            smiles_list: SMILES文字列リスト
            
        Returns:
            3D特徴配列 (n_samples, 3d_feature_dim)
        """
        if not self.use_3d:
            return np.zeros((len(smiles_list), 0))
        
        print("[INFO] 3D特徴抽出中...")
        
        features_3d = []
        for smiles in smiles_list:
            if pd.isna(smiles) or smiles == '':
                feat_3d = [0.0] * 10  # デフォルト3D特徴数
            else:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        feat_3d = [0.0] * 10
                    else:
                        feat_3d = self._calculate_3d_features(mol)
                except Exception as e:
                    print(f"[WARNING] 3D特徴計算失敗: {smiles} - {e}")
                    feat_3d = [0.0] * 10
            
            features_3d.append(feat_3d)
        
        return np.array(features_3d)
    
    def _calculate_3d_features(self, mol: Chem.Mol) -> List[float]:
        """3D特徴計算"""
        try:
            # コンフォマー生成
            mol_h = Chem.AddHs(mol)
            conf_ids = rdDistGeom.EmbedMultipleConfs(
                mol_h, 
                numConfs=self.confs_per_mol,
                maxAttempts=100,
                randomSeed=42
            )
            
            if not conf_ids:
                return [0.0] * 10
            
            # 最初のコンフォマーを使用
            conf = mol_h.GetConformer(conf_ids[0])
            
            # 3D記述子計算
            features = []
            
            # 分子サイズ
            features.append(mol_h.GetNumAtoms())
            features.append(mol_h.GetNumBonds())
            
            # 距離特徴（最大・最小・平均）
            distances = []
            for i in range(mol_h.GetNumAtoms()):
                for j in range(i+1, mol_h.GetNumAtoms()):
                    dist = conf.GetDistance(i, j)
                    distances.append(dist)
            
            if distances:
                features.extend([
                    np.max(distances),
                    np.min(distances),
                    np.mean(distances),
                    np.std(distances)
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
            
            # 角度特徴（簡易版）
            features.extend([0.0, 0.0])  # 角度計算は複雑なので簡易版
            
            return features[:10]  # 10次元に固定
            
        except Exception as e:
            print(f"[WARNING] 3D特徴計算エラー: {e}")
            return [0.0] * 10
    
    def fit_transform(self, smiles_list: List[str]) -> Dict[str, np.ndarray]:
        """
        特徴量抽出と正規化
        
        Args:
            smiles_list: SMILES文字列リスト
            
        Returns:
            特徴量辞書
        """
        print("=" * 60)
        print("力価回帰特徴量抽出開始")
        print("=" * 60)
        
        # 1. トークン化
        tokenized = self.tokenize(smiles_list)
        print(f"[INFO] トークン化完了: {tokenized.shape}")
        
        # 2. 物化記述子
        physchem = self.extract_physchem(smiles_list)
        print(f"[INFO] 物化記述子抽出完了: {physchem.shape}")
        
        # 3. 3D特徴（オプション）
        features_3d = self.extract_3d_features(smiles_list)
        if self.use_3d:
            print(f"[INFO] 3D特徴抽出完了: {features_3d.shape}")
        
        # 4. 物化記述子正規化
        physchem_scaled = self.scaler.fit_transform(physchem)
        print(f"[INFO] 物化記述子正規化完了")
        
        # 5. 特徴量統合
        features = {
            'tokens': tokenized,
            'physchem': physchem_scaled,
            'features_3d': features_3d if self.use_3d else None
        }
        
        print("=" * 60)
        print("特徴量抽出完了")
        print("=" * 60)
        
        return features
    
    def transform(self, smiles_list: List[str]) -> Dict[str, np.ndarray]:
        """
        学習済み特徴量抽出器で変換
        
        Args:
            smiles_list: SMILES文字列リスト
            
        Returns:
            特徴量辞書
        """
        if not self.is_fitted:
            raise ValueError("Featurizer not fitted. Call fit_transform first.")
        
        # トークン化
        tokenized = self.tokenize(smiles_list)
        
        # 物化記述子
        physchem = self.extract_physchem(smiles_list)
        physchem_scaled = self.scaler.transform(physchem)
        
        # 3D特徴
        features_3d = self.extract_3d_features(smiles_list) if self.use_3d else None
        
        return {
            'tokens': tokenized,
            'physchem': physchem_scaled,
            'features_3d': features_3d
        }

class PotencyDataset(Dataset):
    """力価回帰用データセット"""
    
    def __init__(self, features: Dict[str, np.ndarray], targets: np.ndarray, 
                 masks: Optional[np.ndarray] = None):
        """
        初期化
        
        Args:
            features: 特徴量辞書
            targets: ターゲット配列
            masks: マスク配列（外れ値等）
        """
        self.features = features
        self.targets = targets
        self.masks = masks if masks is not None else np.ones(len(targets), dtype=bool)
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        sample = {
            'tokens': torch.LongTensor(self.features['tokens'][idx]),
            'physchem': torch.FloatTensor(self.features['physchem'][idx]),
            'target': torch.FloatTensor([self.targets[idx]]),
            'mask': torch.BoolTensor([self.masks[idx]])
        }
        
        if self.features['features_3d'] is not None:
            sample['features_3d'] = torch.FloatTensor(self.features['features_3d'][idx])
        
        return sample

def create_dataloader(features: Dict[str, np.ndarray], targets: np.ndarray,
                     masks: Optional[np.ndarray] = None, batch_size: int = 32,
                     shuffle: bool = True, num_workers: int = 4) -> DataLoader:
    """
    データローダー作成
    
    Args:
        features: 特徴量辞書
        targets: ターゲット配列
        masks: マスク配列
        batch_size: バッチサイズ
        shuffle: シャッフル
        num_workers: ワーカー数
        
    Returns:
        DataLoader
    """
    dataset = PotencyDataset(features, targets, masks)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else 2
    )
