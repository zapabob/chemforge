"""
Molecular Features Module

分子特徴量計算・統合・管理
既存MolecularPreprocessorを活用した効率的な特徴量計算
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from pathlib import Path
import json
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, rdMolDescriptors
    from rdkit.Chem import rdMolDescriptors as rdMD
    from rdkit.Chem import rdFreeSASA
    from rdkit.Chem import rdMolAlign
    from rdkit.Chem import rdDistGeom
    from rdkit.Chem import rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. Some features will be disabled.")

# 既存モジュール活用
from chemforge.data.molecular_preprocessor import MolecularPreprocessor
from chemforge.utils.config_utils import ConfigManager
from chemforge.utils.logging_utils import Logger
from chemforge.utils.validation import DataValidator

logger = logging.getLogger(__name__)

class MolecularFeatures:
    """
    分子特徴量計算クラス
    
    既存MolecularPreprocessorを活用した効率的な分子特徴量計算
    """
    
    def __init__(self, config_path: Optional[str] = None, cache_dir: str = "cache"):
        """
        初期化
        
        Args:
            config_path: 設定ファイルパス
            cache_dir: キャッシュディレクトリ
        """
        self.config_path = config_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 既存モジュール活用
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load_config()
        self.logger = Logger("MolecularFeatures")
        self.validator = DataValidator()
        
        # 分子前処理器
        self.preprocessor = MolecularPreprocessor()
        
        # 特徴量計算設定
        self.feature_config = self.config.get('molecular_features', {})
        self.descriptor_types = self.feature_config.get('descriptor_types', [
            'basic', 'lipinski', 'qed', 'sasa', 'conformational'
        ])
        self.normalize_features = self.feature_config.get('normalize_features', True)
        self.cache_features = self.feature_config.get('cache_features', True)
        
        logger.info("MolecularFeatures initialized")
    
    def calculate_features(self, smiles_list: List[str], 
                          feature_types: Optional[List[str]] = None,
                          use_cache: bool = True) -> pd.DataFrame:
        """
        分子特徴量計算
        
        Args:
            smiles_list: SMILESリスト
            feature_types: 特徴量タイプリスト
            use_cache: キャッシュ使用
            
        Returns:
            特徴量データフレーム
        """
        logger.info(f"Calculating features for {len(smiles_list)} molecules")
        
        if feature_types is None:
            feature_types = self.descriptor_types
        
        # キャッシュチェック
        cache_key = f"molecular_features_{len(smiles_list)}_{'_'.join(feature_types)}"
        cache_path = self.cache_dir / f"{cache_key}.csv"
        
        if use_cache and cache_path.exists():
            logger.info(f"Loading from cache: {cache_path}")
            return pd.read_csv(cache_path)
        
        # 特徴量計算
        features_data = []
        
        for i, smiles in enumerate(tqdm(smiles_list, desc="Calculating features")):
            try:
                # 分子前処理
                processed_smiles = self.preprocessor.clean_smiles(smiles)
                if not processed_smiles:
                    logger.warning(f"Failed to process SMILES: {smiles}")
                    continue
                
                # 特徴量計算
                molecule_features = self._calculate_single_molecule_features(
                    processed_smiles, feature_types
                )
                
                if molecule_features:
                    molecule_features['smiles'] = smiles
                    molecule_features['processed_smiles'] = processed_smiles
                    features_data.append(molecule_features)
                
            except Exception as e:
                logger.error(f"Failed to calculate features for {smiles}: {e}")
                continue
        
        if not features_data:
            logger.warning("No features calculated")
            return pd.DataFrame()
        
        # データフレーム作成
        features_df = pd.DataFrame(features_data)
        logger.info(f"Calculated features for {len(features_df)} molecules")
        
        # 特徴量正規化
        if self.normalize_features:
            features_df = self._normalize_features(features_df)
        
        # キャッシュ保存
        if use_cache:
            features_df.to_csv(cache_path, index=False)
            logger.info(f"Features cached to: {cache_path}")
        
        return features_df
    
    def _calculate_single_molecule_features(self, smiles: str, 
                                          feature_types: List[str]) -> Optional[Dict]:
        """
        単一分子特徴量計算
        
        Args:
            smiles: SMILES文字列
            feature_types: 特徴量タイプリスト
            
        Returns:
            特徴量辞書
        """
        try:
            if not RDKIT_AVAILABLE:
                logger.warning("RDKit not available, using basic features only")
                return self._calculate_basic_features(smiles)
            
            # 分子オブジェクト作成
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Failed to create molecule from SMILES: {smiles}")
                return None
            
            features = {}
            
            # 基本特徴量
            if 'basic' in feature_types:
                basic_features = self._calculate_basic_features(smiles, mol)
                features.update(basic_features)
            
            # Lipinski特徴量
            if 'lipinski' in feature_types:
                lipinski_features = self._calculate_lipinski_features(mol)
                features.update(lipinski_features)
            
            # QED特徴量
            if 'qed' in feature_types:
                qed_features = self._calculate_qed_features(mol)
                features.update(qed_features)
            
            # SASA特徴量
            if 'sasa' in feature_types:
                sasa_features = self._calculate_sasa_features(mol)
                features.update(sasa_features)
            
            # 立体構造特徴量
            if 'conformational' in feature_types:
                conformational_features = self._calculate_conformational_features(mol)
                features.update(conformational_features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating features for {smiles}: {e}")
            return None
    
    def _calculate_basic_features(self, smiles: str, mol: Optional[Chem.Mol] = None) -> Dict:
        """
        基本特徴量計算
        
        Args:
            smiles: SMILES文字列
            mol: 分子オブジェクト
            
        Returns:
            基本特徴量辞書
        """
        features = {}
        
        try:
            if mol is None and RDKIT_AVAILABLE:
                mol = Chem.MolFromSmiles(smiles)
            
            if mol is not None and RDKIT_AVAILABLE:
                # 分子量
                features['molecular_weight'] = Descriptors.MolWt(mol)
                
                # LogP
                features['logp'] = Descriptors.MolLogP(mol)
                
                # TPSA
                features['tpsa'] = Descriptors.TPSA(mol)
                
                # HBD/HBA
                features['hbd'] = Descriptors.NumHDonors(mol)
                features['hba'] = Descriptors.NumHAcceptors(mol)
                
                # 回転可能結合
                features['rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
                
                # 芳香族環
                features['aromatic_rings'] = Descriptors.NumAromaticRings(mol)
                
                # 原子数
                features['num_atoms'] = mol.GetNumAtoms()
                features['num_bonds'] = mol.GetNumBonds()
                
                # 環数
                features['num_rings'] = Descriptors.RingCount(mol)
                
                # 立体中心
                features['num_stereocenters'] = Descriptors.NumAliphaticCarbocycles(mol)
                
            else:
                # RDKitが利用できない場合の基本特徴量
                features['smiles_length'] = len(smiles)
                features['num_carbons'] = smiles.count('C')
                features['num_nitrogens'] = smiles.count('N')
                features['num_oxygens'] = smiles.count('O')
                features['num_sulfurs'] = smiles.count('S')
                features['num_rings'] = smiles.count('c') + smiles.count('C')
                
        except Exception as e:
            logger.error(f"Error calculating basic features: {e}")
        
        return features
    
    def _calculate_lipinski_features(self, mol: Chem.Mol) -> Dict:
        """
        Lipinski特徴量計算
        
        Args:
            mol: 分子オブジェクト
            
        Returns:
            Lipinski特徴量辞書
        """
        features = {}
        
        try:
            # Lipinski's Rule of Five
            features['lipinski_mw'] = Descriptors.MolWt(mol)
            features['lipinski_logp'] = Descriptors.MolLogP(mol)
            features['lipinski_hbd'] = Descriptors.NumHDonors(mol)
            features['lipinski_hba'] = Descriptors.NumHAcceptors(mol)
            
            # Lipinski違反数
            violations = 0
            if features['lipinski_mw'] > 500:
                violations += 1
            if features['lipinski_logp'] > 5:
                violations += 1
            if features['lipinski_hbd'] > 5:
                violations += 1
            if features['lipinski_hba'] > 10:
                violations += 1
            
            features['lipinski_violations'] = violations
            features['lipinski_compliant'] = violations == 0
            
        except Exception as e:
            logger.error(f"Error calculating Lipinski features: {e}")
        
        return features
    
    def _calculate_qed_features(self, mol: Chem.Mol) -> Dict:
        """
        QED特徴量計算
        
        Args:
            mol: 分子オブジェクト
            
        Returns:
            QED特徴量辞書
        """
        features = {}
        
        try:
            # QEDスコア
            features['qed_score'] = QED.qed(mol)
            
            # QED成分
            qed_components = QED.properties(mol)
            features['qed_mw'] = qed_components[0]
            features['qed_logp'] = qed_components[1]
            features['qed_hbd'] = qed_components[2]
            features['qed_hba'] = qed_components[3]
            features['qed_psa'] = qed_components[4]
            features['qed_rotb'] = qed_components[5]
            features['qed_aromat'] = qed_components[6]
            features['qed_alert'] = qed_components[7]
            
        except Exception as e:
            logger.error(f"Error calculating QED features: {e}")
        
        return features
    
    def _calculate_sasa_features(self, mol: Chem.Mol) -> Dict:
        """
        SASA特徴量計算
        
        Args:
            mol: 分子オブジェクト
            
        Returns:
            SASA特徴量辞書
        """
        features = {}
        
        try:
            # 3D構造生成
            mol_3d = Chem.AddHs(mol)
            rdDistGeom.EmbedMolecule(mol_3d)
            
            # SASA計算
            sasa = rdFreeSASA.CalcSASA(mol_3d)
            features['sasa'] = sasa
            
            # 原子別SASA
            atom_sasas = rdFreeSASA.CalcSASA(mol_3d, confId=0)
            features['atom_sasa_mean'] = np.mean(atom_sasas) if atom_sasas else 0
            features['atom_sasa_std'] = np.std(atom_sasas) if atom_sasas else 0
            
        except Exception as e:
            logger.error(f"Error calculating SASA features: {e}")
        
        return features
    
    def _calculate_conformational_features(self, mol: Chem.Mol) -> Dict:
        """
        立体構造特徴量計算
        
        Args:
            mol: 分子オブジェクト
            
        Returns:
            立体構造特徴量辞書
        """
        features = {}
        
        try:
            # 3D構造生成
            mol_3d = Chem.AddHs(mol)
            rdDistGeom.EmbedMolecule(mol_3d)
            
            # 立体構造特徴量
            features['num_conformers'] = mol_3d.GetNumConformers()
            
            # 分子形状
            features['molecular_volume'] = rdMolDescriptors.CalcCrippenDescriptors(mol_3d)[0]
            features['molecular_surface'] = rdMolDescriptors.CalcCrippenDescriptors(mol_3d)[1]
            
            # 立体配座
            features['num_stereocenters'] = Descriptors.NumAliphaticCarbocycles(mol_3d)
            features['num_rotatable_bonds'] = Descriptors.NumRotatableBonds(mol_3d)
            
        except Exception as e:
            logger.error(f"Error calculating conformational features: {e}")
        
        return features
    
    def _normalize_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        特徴量正規化
        
        Args:
            features_df: 特徴量データフレーム
            
        Returns:
            正規化済みデータフレーム
        """
        try:
            from sklearn.preprocessing import StandardScaler
            
            # 数値列のみ正規化
            numeric_columns = features_df.select_dtypes(include=[np.number]).columns
            numeric_columns = [col for col in numeric_columns if col not in ['smiles', 'processed_smiles']]
            
            if len(numeric_columns) > 0:
                scaler = StandardScaler()
                features_df[numeric_columns] = scaler.fit_transform(features_df[numeric_columns])
                
                # スケーラー保存
                scaler_path = self.cache_dir / "feature_scaler.pkl"
                import pickle
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
                
                logger.info(f"Features normalized and scaler saved to: {scaler_path}")
            
        except Exception as e:
            logger.error(f"Error normalizing features: {e}")
        
        return features_df
    
    def get_feature_names(self, feature_types: Optional[List[str]] = None) -> List[str]:
        """
        特徴量名取得
        
        Args:
            feature_types: 特徴量タイプリスト
            
        Returns:
            特徴量名リスト
        """
        if feature_types is None:
            feature_types = self.descriptor_types
        
        feature_names = []
        
        if 'basic' in feature_types:
            feature_names.extend([
                'molecular_weight', 'logp', 'tpsa', 'hbd', 'hba',
                'rotatable_bonds', 'aromatic_rings', 'num_atoms', 'num_bonds',
                'num_rings', 'num_stereocenters'
            ])
        
        if 'lipinski' in feature_types:
            feature_names.extend([
                'lipinski_mw', 'lipinski_logp', 'lipinski_hbd', 'lipinski_hba',
                'lipinski_violations', 'lipinski_compliant'
            ])
        
        if 'qed' in feature_types:
            feature_names.extend([
                'qed_score', 'qed_mw', 'qed_logp', 'qed_hbd', 'qed_hba',
                'qed_psa', 'qed_rotb', 'qed_aromat', 'qed_alert'
            ])
        
        if 'sasa' in feature_types:
            feature_names.extend([
                'sasa', 'atom_sasa_mean', 'atom_sasa_std'
            ])
        
        if 'conformational' in feature_types:
            feature_names.extend([
                'num_conformers', 'molecular_volume', 'molecular_surface',
                'num_stereocenters', 'num_rotatable_bonds'
            ])
        
        return feature_names
    
    def export_features(self, features_df: pd.DataFrame, output_path: str, 
                       format: str = "csv") -> bool:
        """
        特徴量エクスポート
        
        Args:
            features_df: 特徴量データフレーム
            output_path: 出力パス
            format: 出力形式
            
        Returns:
            成功フラグ
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == "csv":
                features_df.to_csv(output_path, index=False)
            elif format.lower() == "json":
                features_df.to_json(output_path, orient='records', indent=2)
            elif format.lower() == "parquet":
                features_df.to_parquet(output_path, index=False)
            else:
                logger.error(f"Unsupported format: {format}")
                return False
            
            logger.info(f"Features exported to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting features: {e}")
            return False

def create_molecular_features(config_path: Optional[str] = None, 
                             cache_dir: str = "cache") -> MolecularFeatures:
    """
    分子特徴量計算器作成
    
    Args:
        config_path: 設定ファイルパス
        cache_dir: キャッシュディレクトリ
        
    Returns:
        MolecularFeatures
    """
    return MolecularFeatures(config_path, cache_dir)

def calculate_molecular_features(smiles_list: List[str], 
                                feature_types: Optional[List[str]] = None,
                                config_path: Optional[str] = None,
                                cache_dir: str = "cache") -> pd.DataFrame:
    """
    分子特徴量計算（簡易版）
    
    Args:
        smiles_list: SMILESリスト
        feature_types: 特徴量タイプリスト
        config_path: 設定ファイルパス
        cache_dir: キャッシュディレクトリ
        
    Returns:
        特徴量データフレーム
    """
    features_calculator = create_molecular_features(config_path, cache_dir)
    return features_calculator.calculate_features(smiles_list, feature_types)

if __name__ == "__main__":
    # テスト実行
    features_calculator = MolecularFeatures()
    
    # テストSMILES
    test_smiles = ["CCO", "CCN", "c1ccccc1", "CC(=O)O"]
    
    # 特徴量計算
    features = features_calculator.calculate_features(test_smiles)
    
    print(f"Calculated features for {len(features)} molecules")
    if not features.empty:
        print(f"Feature columns: {list(features.columns)}")
        print(f"Feature shape: {features.shape}")
        print(f"Sample features:")
        print(features.head())