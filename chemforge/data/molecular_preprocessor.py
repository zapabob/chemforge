"""
分子前処理器（MolecularPreprocessor）

CNS創薬向けの統合分子特徴量抽出器。
RDKit記述子、Morgan Fingerprint、骨格特徴量、ADMET予測値を統合して2279次元の特徴量ベクトルを生成。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski
from rdkit.Chem import QED, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator
import warnings
warnings.filterwarnings('ignore')

from .scaffold_detector import ScaffoldDetector
from .admet_predictor import ADMETPredictor


class MolecularPreprocessor:
    """統合分子前処理器クラス"""
    
    def __init__(
        self, 
        use_scaffold_features: bool = True,
        use_admet: bool = True,
        use_rdkit_descriptors: bool = True,
        use_fingerprints: bool = True,
        verbose: bool = False
    ):
        """
        分子前処理器を初期化
        
        Args:
            use_scaffold_features: 骨格特徴量を使用するか
            use_admet: ADMET予測を使用するか
            use_rdkit_descriptors: RDKit記述子を使用するか
            use_fingerprints: フィンガープリントを使用するか
            verbose: 詳細ログ出力フラグ
        """
        self.use_scaffold_features = use_scaffold_features
        self.use_admet = use_admet
        self.use_rdkit_descriptors = use_rdkit_descriptors
        self.use_fingerprints = use_fingerprints
        self.verbose = verbose
        
        # サブモジュールを初期化
        if self.use_scaffold_features:
            self.scaffold_detector = ScaffoldDetector(verbose=verbose)
        
        if self.use_admet:
            self.admet_predictor = ADMETPredictor(verbose=verbose)
        
        # 特徴量次元を計算
        self.feature_dimensions = self._calculate_feature_dimensions()
        self.total_dimensions = sum(self.feature_dimensions.values())
        
        if self.verbose:
            print(f"MolecularPreprocessor initialized:")
            print(f"  Total dimensions: {self.total_dimensions}")
            print(f"  Feature breakdown: {self.feature_dimensions}")
    
    def _calculate_feature_dimensions(self) -> Dict[str, int]:
        """特徴量次元を計算"""
        dimensions = {}
        
        if self.use_rdkit_descriptors:
            dimensions['rdkit_descriptors'] = 200  # RDKit記述子
        else:
            dimensions['rdkit_descriptors'] = 0
        
        if self.use_fingerprints:
            dimensions['morgan_fingerprints'] = 2048  # Morgan Fingerprint
        else:
            dimensions['morgan_fingerprints'] = 0
        
        if self.use_scaffold_features:
            dimensions['scaffold_features'] = 20  # 骨格特徴量 (4骨格×5特徴)
        else:
            dimensions['scaffold_features'] = 0
        
        if self.use_admet:
            dimensions['admet_features'] = 10  # ADMET特徴量
            dimensions['cns_mpo'] = 1  # CNS-MPO
        else:
            dimensions['admet_features'] = 0
            dimensions['cns_mpo'] = 0
        
        return dimensions
    
    def preprocess(self, smiles: str) -> Dict[str, np.ndarray]:
        """
        分子を前処理して特徴量を抽出
        
        Args:
            smiles: SMILES文字列
            
        Returns:
            特徴量辞書
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            if self.verbose:
                print(f"Invalid SMILES: {smiles}")
            return self._get_default_features()
        
        features = {}
        
        # RDKit記述子
        if self.use_rdkit_descriptors:
            features['rdkit_descriptors'] = self._extract_rdkit_descriptors(mol)
        
        # Morgan Fingerprint
        if self.use_fingerprints:
            features['morgan_fingerprints'] = self._extract_morgan_fingerprints(mol)
        
        # 骨格特徴量
        if self.use_scaffold_features:
            features['scaffold_features'] = self._extract_scaffold_features(mol)
        
        # ADMET特徴量
        if self.use_admet:
            features['admet_features'] = self._extract_admet_features(mol)
            features['cns_mpo'] = self._extract_cns_mpo(mol)
        
        return features
    
    def _extract_rdkit_descriptors(self, mol: Chem.Mol) -> np.ndarray:
        """RDKit記述子を抽出"""
        try:
            descriptors = []
            
            # 基本物性
            descriptors.append(Descriptors.MolWt(mol))  # 分子量
            descriptors.append(Crippen.MolLogP(mol))  # LogP
            descriptors.append(Descriptors.TPSA(mol))  # 極性表面積
            descriptors.append(Descriptors.NumHDonors(mol))  # 水素結合ドナー
            descriptors.append(Descriptors.NumHAcceptors(mol))  # 水素結合アクセプター
            descriptors.append(Descriptors.NumRotatableBonds(mol))  # 回転可能結合
            descriptors.append(Descriptors.NumAromaticRings(mol))  # 芳香環数
            descriptors.append(Descriptors.HeavyAtomCount(mol))  # 重原子数
            descriptors.append(Descriptors.NumSaturatedRings(mol))  # 飽和環数
            descriptors.append(Descriptors.NumAliphaticRings(mol))  # 脂肪族環数
            
            # 電荷関連
            descriptors.append(Descriptors.NumValenceElectrons(mol))  # 価電子数
            descriptors.append(Descriptors.NumRadicalElectrons(mol))  # ラジカル電子数
            descriptors.append(Descriptors.NumHeteroatoms(mol))  # ヘテロ原子数
            descriptors.append(Descriptors.NumSaturatedCarbocycles(mol))  # 飽和炭素環数
            descriptors.append(Descriptors.NumSaturatedHeterocycles(mol))  # 飽和ヘテロ環数
            
            # 立体化学
            descriptors.append(Descriptors.NumSpiroAtoms(mol))  # スピロ原子数
            descriptors.append(Descriptors.NumBridgeheadAtoms(mol))  # ブリッジヘッド原子数
            descriptors.append(Descriptors.NumAliphaticCarbocycles(mol))  # 脂肪族炭素環数
            descriptors.append(Descriptors.NumAliphaticHeterocycles(mol))  # 脂肪族ヘテロ環数
            descriptors.append(Descriptors.NumAromaticCarbocycles(mol))  # 芳香族炭素環数
            
            # 追加記述子
            descriptors.append(Descriptors.BertzCT(mol))  # Bertz CT
            descriptors.append(Descriptors.BalabanJ(mol))  # Balaban J
            descriptors.append(Descriptors.Kappa1(mol))  # Kappa1
            descriptors.append(Descriptors.Kappa2(mol))  # Kappa2
            descriptors.append(Descriptors.Kappa3(mol))  # Kappa3
            
            # 残りを0で埋める
            while len(descriptors) < 200:
                descriptors.append(0.0)
            
            return np.array(descriptors[:200], dtype=np.float32)
            
        except Exception as e:
            if self.verbose:
                print(f"Error extracting RDKit descriptors: {e}")
            return np.zeros(200, dtype=np.float32)
    
    def _extract_morgan_fingerprints(self, mol: Chem.Mol) -> np.ndarray:
        """Morgan Fingerprintを抽出"""
        try:
            # Morgan Fingerprintを生成
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            
            # ビットベクトルを配列に変換
            fp_array = np.array(fp, dtype=np.float32)
            
            return fp_array
            
        except Exception as e:
            if self.verbose:
                print(f"Error extracting Morgan fingerprints: {e}")
            return np.zeros(2048, dtype=np.float32)
    
    def _extract_scaffold_features(self, mol: Chem.Mol) -> np.ndarray:
        """骨格特徴量を抽出"""
        try:
            return self.scaffold_detector.extract_scaffold_features(mol)
        except Exception as e:
            if self.verbose:
                print(f"Error extracting scaffold features: {e}")
            return np.zeros(20, dtype=np.float32)
    
    def _extract_admet_features(self, mol: Chem.Mol) -> np.ndarray:
        """ADMET特徴量を抽出"""
        try:
            return self.admet_predictor.extract_admet_features(mol)
        except Exception as e:
            if self.verbose:
                print(f"Error extracting ADMET features: {e}")
            return np.zeros(10, dtype=np.float32)
    
    def _extract_cns_mpo(self, mol: Chem.Mol) -> np.ndarray:
        """CNS-MPOスコアを抽出"""
        try:
            cns_mpo_score = self.admet_predictor.calculate_cns_mpo(mol)
            return np.array([cns_mpo_score], dtype=np.float32)
        except Exception as e:
            if self.verbose:
                print(f"Error extracting CNS-MPO: {e}")
            return np.array([0.0], dtype=np.float32)
    
    def _get_default_features(self) -> Dict[str, np.ndarray]:
        """デフォルト特徴量を取得"""
        features = {}
        
        if self.use_rdkit_descriptors:
            features['rdkit_descriptors'] = np.zeros(200, dtype=np.float32)
        
        if self.use_fingerprints:
            features['morgan_fingerprints'] = np.zeros(2048, dtype=np.float32)
        
        if self.use_scaffold_features:
            features['scaffold_features'] = np.zeros(20, dtype=np.float32)
        
        if self.use_admet:
            features['admet_features'] = np.zeros(10, dtype=np.float32)
            features['cns_mpo'] = np.array([0.0], dtype=np.float32)
        
        return features
    
    def extract_combined_features(self, smiles: str) -> np.ndarray:
        """
        統合特徴量を抽出
        
        Args:
            smiles: SMILES文字列
            
        Returns:
            統合特徴量ベクトル (2279次元)
        """
        features = self.preprocess(smiles)
        
        # 特徴量を結合
        combined_features = []
        
        if self.use_rdkit_descriptors:
            combined_features.append(features['rdkit_descriptors'])
        
        if self.use_fingerprints:
            combined_features.append(features['morgan_fingerprints'])
        
        if self.use_scaffold_features:
            combined_features.append(features['scaffold_features'])
        
        if self.use_admet:
            combined_features.append(features['admet_features'])
            combined_features.append(features['cns_mpo'])
        
        if combined_features:
            return np.concatenate(combined_features, axis=0)
        else:
            return np.zeros(self.total_dimensions, dtype=np.float32)
    
    def batch_preprocess(self, smiles_list: List[str]) -> List[Dict[str, np.ndarray]]:
        """
        複数分子の前処理をバッチ実行
        
        Args:
            smiles_list: SMILES文字列のリスト
            
        Returns:
            各分子の特徴量辞書のリスト
        """
        results = []
        for smiles in smiles_list:
            features = self.preprocess(smiles)
            results.append(features)
        return results
    
    def batch_extract_combined_features(self, smiles_list: List[str]) -> np.ndarray:
        """
        複数分子の統合特徴量をバッチ抽出
        
        Args:
            smiles_list: SMILES文字列のリスト
            
        Returns:
            統合特徴量行列 (N x 2279)
        """
        features_list = []
        for smiles in smiles_list:
            features = self.extract_combined_features(smiles)
            features_list.append(features)
        return np.array(features_list)
    
    def get_feature_info(self, smiles: str) -> Dict[str, Any]:
        """
        分子の詳細情報を取得
        
        Args:
            smiles: SMILES文字列
            
        Returns:
            分子の詳細情報
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                'valid': False,
                'error': 'Invalid SMILES',
                'features': self._get_default_features()
            }
        
        try:
            # 基本情報
            info = {
                'valid': True,
                'smiles': smiles,
                'mol_weight': Descriptors.MolWt(mol),
                'logp': Crippen.MolLogP(mol),
                'tpsa': Descriptors.TPSA(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'rot_bonds': Descriptors.NumRotatableBonds(mol),
                'aromatic_rings': Descriptors.NumAromaticRings(mol),
                'heavy_atoms': Descriptors.HeavyAtomCount(mol)
            }
            
            # 骨格情報
            if self.use_scaffold_features:
                scaffold_summary = self.scaffold_detector.get_scaffold_summary(mol)
                info['scaffold_info'] = scaffold_summary
            
            # ADMET情報
            if self.use_admet:
                admet_summary = self.admet_predictor.get_admet_summary(mol)
                info['admet_info'] = admet_summary
            
            # 特徴量
            info['features'] = self.preprocess(smiles)
            
            return info
            
        except Exception as e:
            if self.verbose:
                print(f"Error getting feature info: {e}")
            return {
                'valid': False,
                'error': str(e),
                'features': self._get_default_features()
            }
    
    def clean_smiles(self, smiles: str) -> Optional[str]:
        """
        SMILES文字列をクリーニング
        
        Args:
            smiles: 入力SMILES文字列
            
        Returns:
            クリーニング済みSMILES文字列（無効な場合はNone）
        """
        if not smiles or not isinstance(smiles, str):
            return None
        
        # 空白除去
        cleaned = smiles.strip()
        
        # 空文字列チェック
        if not cleaned:
            return None
        
        # SMILES妥当性チェック
        if not self.validate_smiles(cleaned):
            return None
        
        return cleaned

    def validate_smiles(self, smiles: str) -> bool:
        """SMILES文字列の妥当性を検証"""
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """特徴量次元情報を取得"""
        return self.feature_dimensions.copy()
    
    def get_total_dimensions(self) -> int:
        """総特徴量次元を取得"""
        return self.total_dimensions
    
    def update_config(
        self,
        use_scaffold_features: Optional[bool] = None,
        use_admet: Optional[bool] = None,
        use_rdkit_descriptors: Optional[bool] = None,
        use_fingerprints: Optional[bool] = None
    ):
        """設定を更新"""
        if use_scaffold_features is not None:
            self.use_scaffold_features = use_scaffold_features
            if self.use_scaffold_features and not hasattr(self, 'scaffold_detector'):
                self.scaffold_detector = ScaffoldDetector(verbose=self.verbose)
        
        if use_admet is not None:
            self.use_admet = use_admet
            if self.use_admet and not hasattr(self, 'admet_predictor'):
                self.admet_predictor = ADMETPredictor(verbose=self.verbose)
        
        if use_rdkit_descriptors is not None:
            self.use_rdkit_descriptors = use_rdkit_descriptors
        
        if use_fingerprints is not None:
            self.use_fingerprints = use_fingerprints
        
        # 特徴量次元を再計算
        self.feature_dimensions = self._calculate_feature_dimensions()
        self.total_dimensions = sum(self.feature_dimensions.values())
        
        if self.verbose:
            print(f"Configuration updated:")
            print(f"  Total dimensions: {self.total_dimensions}")
            print(f"  Feature breakdown: {self.feature_dimensions}")


# 便利関数
def preprocess_molecule(smiles: str, **kwargs) -> Dict[str, np.ndarray]:
    """分子を前処理"""
    preprocessor = MolecularPreprocessor(**kwargs)
    return preprocessor.preprocess(smiles)


def extract_combined_features(smiles: str, **kwargs) -> np.ndarray:
    """統合特徴量を抽出"""
    preprocessor = MolecularPreprocessor(**kwargs)
    return preprocessor.extract_combined_features(smiles)


def batch_preprocess_molecules(smiles_list: List[str], **kwargs) -> List[Dict[str, np.ndarray]]:
    """複数分子を前処理"""
    preprocessor = MolecularPreprocessor(**kwargs)
    return preprocessor.batch_preprocess(smiles_list)


def batch_extract_combined_features(smiles_list: List[str], **kwargs) -> np.ndarray:
    """複数分子の統合特徴量を抽出"""
    preprocessor = MolecularPreprocessor(**kwargs)
    return preprocessor.batch_extract_combined_features(smiles_list)


if __name__ == "__main__":
    # テスト実行
    print("🧬 分子前処理器テスト")
    print("=" * 50)
    
    # テスト用SMILES
    test_smiles = [
        "CC(CC1=CC=CC=C1)N",  # アンフェタミン
        "CCN(CC)CC1=CC2=C(C=C1)OCO2",  # MDMA
        "CCN(CC)CC1=CNC2=CC=CC=C21",  # DMT
        "CN1CC[C@]23C4=C5C6=C(C=CC=C6)O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5=C4O",  # モルヒネ
        "CCCCCC1=CC(=C(C(=C1)O)C2C=C(CCC2C(C)C)C)C",  # THC
    ]
    
    # 前処理器を初期化
    preprocessor = MolecularPreprocessor(verbose=True)
    
    print(f"総特徴量次元: {preprocessor.get_total_dimensions()}")
    print(f"特徴量内訳: {preprocessor.get_feature_dimensions()}")
    
    for i, smiles in enumerate(test_smiles):
        print(f"\n📋 テスト {i+1}: {smiles}")
        
        # SMILES妥当性チェック
        if not preprocessor.validate_smiles(smiles):
            print("  ❌ Invalid SMILES")
            continue
        
        # 前処理
        features = preprocessor.preprocess(smiles)
        print(f"  特徴量タイプ: {list(features.keys())}")
        
        # 統合特徴量抽出
        combined_features = preprocessor.extract_combined_features(smiles)
        print(f"  統合特徴量次元: {combined_features.shape}")
        print(f"  統合特徴量値: {combined_features[:10]}")  # 最初の10次元のみ表示
        
        # 詳細情報取得
        info = preprocessor.get_feature_info(smiles)
        if info['valid']:
            print(f"  分子量: {info['mol_weight']:.2f}")
            print(f"  LogP: {info['logp']:.2f}")
            print(f"  TPSA: {info['tpsa']:.2f}")
            
            if 'scaffold_info' in info:
                scaffold_counts = info['scaffold_info']['scaffold_counts']
                print(f"  骨格検出: {scaffold_counts}")
            
            if 'admet_info' in info:
                cns_mpo = info['admet_info']['CNS_MPO_score']
                print(f"  CNS-MPO: {cns_mpo:.3f}")
    
    print("\n✅ 分子前処理器テスト完了！")
