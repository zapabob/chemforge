"""
RDKit Descriptors Module

RDKit記述子計算・統合・管理
既存DataPreprocessorを活用した効率的な記述子計算
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
from chemforge.data.data_preprocessor import DataPreprocessor
from chemforge.utils.config_utils import ConfigManager
from chemforge.utils.logging_utils import Logger
from chemforge.utils.validation import DataValidator

logger = logging.getLogger(__name__)

class RDKitDescriptors:
    """
    RDKit記述子計算クラス
    
    既存DataPreprocessorを活用した効率的なRDKit記述子計算
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
        self.logger = Logger("RDKitDescriptors")
        self.validator = DataValidator()
        
        # データ前処理器
        self.preprocessor = DataPreprocessor()
        
        # 記述子計算設定
        self.descriptor_config = self.config.get('rdkit_descriptors', {})
        self.descriptor_types = self.descriptor_config.get('descriptor_types', [
            'morgan', 'maccs', 'rdkit', 'topological', 'constitutional'
        ])
        self.normalize_descriptors = self.descriptor_config.get('normalize_descriptors', True)
        self.cache_descriptors = self.descriptor_config.get('cache_descriptors', True)
        
        logger.info("RDKitDescriptors initialized")
    
    def calculate_descriptors(self, smiles_list: List[str], 
                             descriptor_types: Optional[List[str]] = None,
                             use_cache: bool = True) -> pd.DataFrame:
        """
        RDKit記述子計算
        
        Args:
            smiles_list: SMILESリスト
            descriptor_types: 記述子タイプリスト
            use_cache: キャッシュ使用
            
        Returns:
            記述子データフレーム
        """
        logger.info(f"Calculating descriptors for {len(smiles_list)} molecules")
        
        if descriptor_types is None:
            descriptor_types = self.descriptor_types
        
        # キャッシュチェック
        cache_key = f"rdkit_descriptors_{len(smiles_list)}_{'_'.join(descriptor_types)}"
        cache_path = self.cache_dir / f"{cache_key}.csv"
        
        if use_cache and cache_path.exists():
            logger.info(f"Loading from cache: {cache_path}")
            return pd.read_csv(cache_path)
        
        # 記述子計算
        descriptors_data = []
        
        for i, smiles in enumerate(tqdm(smiles_list, desc="Calculating descriptors")):
            try:
                # 分子前処理
                processed_smiles = self._clean_smiles(smiles)
                if not processed_smiles:
                    logger.warning(f"Failed to process SMILES: {smiles}")
                    continue
                
                # 記述子計算
                molecule_descriptors = self._calculate_single_molecule_descriptors(
                    processed_smiles, descriptor_types
                )
                
                if molecule_descriptors:
                    molecule_descriptors['smiles'] = smiles
                    molecule_descriptors['processed_smiles'] = processed_smiles
                    descriptors_data.append(molecule_descriptors)
                
            except Exception as e:
                logger.error(f"Failed to calculate descriptors for {smiles}: {e}")
                continue
        
        if not descriptors_data:
            logger.warning("No descriptors calculated")
            return pd.DataFrame()
        
        # データフレーム作成
        descriptors_df = pd.DataFrame(descriptors_data)
        logger.info(f"Calculated descriptors for {len(descriptors_df)} molecules")
        
        # 記述子正規化
        if self.normalize_descriptors:
            descriptors_df = self._normalize_descriptors(descriptors_df)
        
        # キャッシュ保存
        if use_cache:
            descriptors_df.to_csv(cache_path, index=False)
            logger.info(f"Descriptors cached to: {cache_path}")
        
        return descriptors_df
    
    def _clean_smiles(self, smiles: str) -> Optional[str]:
        """
        SMILESクリーニング
        
        Args:
            smiles: SMILES文字列
            
        Returns:
            クリーニング済みSMILES
        """
        try:
            if not RDKIT_AVAILABLE:
                return smiles
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # 標準化
            mol = Chem.RemoveHs(mol)
            mol = Chem.RemoveStereochemistry(mol)
            
            return Chem.MolToSmiles(mol)
            
        except Exception as e:
            logger.error(f"Error cleaning SMILES: {e}")
            return None
    
    def _calculate_single_molecule_descriptors(self, smiles: str, 
                                              descriptor_types: List[str]) -> Optional[Dict]:
        """
        単一分子記述子計算
        
        Args:
            smiles: SMILES文字列
            descriptor_types: 記述子タイプリスト
            
        Returns:
            記述子辞書
        """
        try:
            if not RDKIT_AVAILABLE:
                logger.warning("RDKit not available, using basic descriptors only")
                return self._calculate_basic_descriptors(smiles)
            
            # 分子オブジェクト作成
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Failed to create molecule from SMILES: {smiles}")
                return None
            
            descriptors = {}
            
            # Morgan記述子
            if 'morgan' in descriptor_types:
                morgan_descriptors = self._calculate_morgan_descriptors(mol)
                descriptors.update(morgan_descriptors)
            
            # MACCS記述子
            if 'maccs' in descriptor_types:
                maccs_descriptors = self._calculate_maccs_descriptors(mol)
                descriptors.update(maccs_descriptors)
            
            # RDKit記述子
            if 'rdkit' in descriptor_types:
                rdkit_descriptors = self._calculate_rdkit_descriptors(mol)
                descriptors.update(rdkit_descriptors)
            
            # トポロジー記述子
            if 'topological' in descriptor_types:
                topological_descriptors = self._calculate_topological_descriptors(mol)
                descriptors.update(topological_descriptors)
            
            # 構成記述子
            if 'constitutional' in descriptor_types:
                constitutional_descriptors = self._calculate_constitutional_descriptors(mol)
                descriptors.update(constitutional_descriptors)
            
            return descriptors
            
        except Exception as e:
            logger.error(f"Error calculating descriptors for {smiles}: {e}")
            return None
    
    def _calculate_basic_descriptors(self, smiles: str) -> Dict:
        """
        基本記述子計算
        
        Args:
            smiles: SMILES文字列
            
        Returns:
            基本記述子辞書
        """
        descriptors = {}
        
        try:
            # 基本特徴量
            descriptors['smiles_length'] = len(smiles)
            descriptors['num_carbons'] = smiles.count('C')
            descriptors['num_nitrogens'] = smiles.count('N')
            descriptors['num_oxygens'] = smiles.count('O')
            descriptors['num_sulfurs'] = smiles.count('S')
            descriptors['num_rings'] = smiles.count('c') + smiles.count('C')
            
        except Exception as e:
            logger.error(f"Error calculating basic descriptors: {e}")
        
        return descriptors
    
    def _calculate_morgan_descriptors(self, mol: Chem.Mol) -> Dict:
        """
        Morgan記述子計算
        
        Args:
            mol: 分子オブジェクト
            
        Returns:
            Morgan記述子辞書
        """
        descriptors = {}
        
        try:
            # Morgan fingerprint
            morgan_fp = rdMD.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            descriptors['morgan_fp'] = morgan_fp.ToBitString()
            
            # Morgan fingerprint counts
            morgan_counts = rdMD.GetMorganFingerprint(mol, radius=2)
            descriptors['morgan_counts'] = len(morgan_counts.GetNonzeroElements())
            
        except Exception as e:
            logger.error(f"Error calculating Morgan descriptors: {e}")
        
        return descriptors
    
    def _calculate_maccs_descriptors(self, mol: Chem.Mol) -> Dict:
        """
        MACCS記述子計算
        
        Args:
            mol: 分子オブジェクト
            
        Returns:
            MACCS記述子辞書
        """
        descriptors = {}
        
        try:
            # MACCS fingerprint
            maccs_fp = rdMD.GetMACCSKeysFingerprint(mol)
            descriptors['maccs_fp'] = maccs_fp.ToBitString()
            
            # MACCS counts
            descriptors['maccs_counts'] = len(maccs_fp.GetNonzeroElements())
            
        except Exception as e:
            logger.error(f"Error calculating MACCS descriptors: {e}")
        
        return descriptors
    
    def _calculate_rdkit_descriptors(self, mol: Chem.Mol) -> Dict:
        """
        RDKit記述子計算
        
        Args:
            mol: 分子オブジェクト
            
        Returns:
            RDKit記述子辞書
        """
        descriptors = {}
        
        try:
            # 基本記述子
            descriptors['molecular_weight'] = Descriptors.MolWt(mol)
            descriptors['logp'] = Descriptors.MolLogP(mol)
            descriptors['tpsa'] = Descriptors.TPSA(mol)
            descriptors['hbd'] = Descriptors.NumHDonors(mol)
            descriptors['hba'] = Descriptors.NumHAcceptors(mol)
            descriptors['rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
            descriptors['aromatic_rings'] = Descriptors.NumAromaticRings(mol)
            descriptors['num_atoms'] = mol.GetNumAtoms()
            descriptors['num_bonds'] = mol.GetNumBonds()
            descriptors['num_rings'] = Descriptors.RingCount(mol)
            
        except Exception as e:
            logger.error(f"Error calculating RDKit descriptors: {e}")
        
        return descriptors
    
    def _calculate_topological_descriptors(self, mol: Chem.Mol) -> Dict:
        """
        トポロジー記述子計算
        
        Args:
            mol: 分子オブジェクト
            
        Returns:
            トポロジー記述子辞書
        """
        descriptors = {}
        
        try:
            # トポロジー記述子
            descriptors['balaban_j'] = Descriptors.BalabanJ(mol)
            descriptors['bertz_ct'] = Descriptors.BertzCT(mol)
            descriptors['chi0v'] = Descriptors.Chi0v(mol)
            descriptors['chi1v'] = Descriptors.Chi1v(mol)
            descriptors['chi2v'] = Descriptors.Chi2v(mol)
            descriptors['chi3v'] = Descriptors.Chi3v(mol)
            descriptors['chi4v'] = Descriptors.Chi4v(mol)
            
        except Exception as e:
            logger.error(f"Error calculating topological descriptors: {e}")
        
        return descriptors
    
    def _calculate_constitutional_descriptors(self, mol: Chem.Mol) -> Dict:
        """
        構成記述子計算
        
        Args:
            mol: 分子オブジェクト
            
        Returns:
            構成記述子辞書
        """
        descriptors = {}
        
        try:
            # 構成記述子
            descriptors['num_heavy_atoms'] = Descriptors.HeavyAtomCount(mol)
            descriptors['num_heteroatoms'] = Descriptors.NumHeteroatoms(mol)
            descriptors['num_saturated_rings'] = Descriptors.NumSaturatedRings(mol)
            descriptors['num_aliphatic_rings'] = Descriptors.NumAliphaticRings(mol)
            descriptors['num_aromatic_rings'] = Descriptors.NumAromaticRings(mol)
            descriptors['num_saturated_carbocycles'] = Descriptors.NumSaturatedCarbocycles(mol)
            descriptors['num_aliphatic_carbocycles'] = Descriptors.NumAliphaticCarbocycles(mol)
            descriptors['num_aromatic_carbocycles'] = Descriptors.NumAromaticCarbocycles(mol)
            
        except Exception as e:
            logger.error(f"Error calculating constitutional descriptors: {e}")
        
        return descriptors
    
    def _normalize_descriptors(self, descriptors_df: pd.DataFrame) -> pd.DataFrame:
        """
        記述子正規化
        
        Args:
            descriptors_df: 記述子データフレーム
            
        Returns:
            正規化済みデータフレーム
        """
        try:
            from sklearn.preprocessing import StandardScaler
            
            # 数値列のみ正規化
            numeric_columns = descriptors_df.select_dtypes(include=[np.number]).columns
            numeric_columns = [col for col in numeric_columns if col not in ['smiles', 'processed_smiles']]
            
            if len(numeric_columns) > 0:
                scaler = StandardScaler()
                descriptors_df[numeric_columns] = scaler.fit_transform(descriptors_df[numeric_columns])
                
                # スケーラー保存
                scaler_path = self.cache_dir / "descriptor_scaler.pkl"
                import pickle
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
                
                logger.info(f"Descriptors normalized and scaler saved to: {scaler_path}")
            
        except Exception as e:
            logger.error(f"Error normalizing descriptors: {e}")
        
        return descriptors_df
    
    def get_descriptor_names(self, descriptor_types: Optional[List[str]] = None) -> List[str]:
        """
        記述子名取得
        
        Args:
            descriptor_types: 記述子タイプリスト
            
        Returns:
            記述子名リスト
        """
        if descriptor_types is None:
            descriptor_types = self.descriptor_types
        
        descriptor_names = []
        
        if 'morgan' in descriptor_types:
            descriptor_names.extend(['morgan_fp', 'morgan_counts'])
        
        if 'maccs' in descriptor_types:
            descriptor_names.extend(['maccs_fp', 'maccs_counts'])
        
        if 'rdkit' in descriptor_types:
            descriptor_names.extend([
                'molecular_weight', 'logp', 'tpsa', 'hbd', 'hba',
                'rotatable_bonds', 'aromatic_rings', 'num_atoms', 'num_bonds', 'num_rings'
            ])
        
        if 'topological' in descriptor_types:
            descriptor_names.extend([
                'balaban_j', 'bertz_ct', 'chi0v', 'chi1v', 'chi2v', 'chi3v', 'chi4v'
            ])
        
        if 'constitutional' in descriptor_types:
            descriptor_names.extend([
                'num_heavy_atoms', 'num_heteroatoms', 'num_saturated_rings',
                'num_aliphatic_rings', 'num_aromatic_rings', 'num_saturated_carbocycles',
                'num_aliphatic_carbocycles', 'num_aromatic_carbocycles'
            ])
        
        return descriptor_names
    
    def export_descriptors(self, descriptors_df: pd.DataFrame, output_path: str, 
                          format: str = "csv") -> bool:
        """
        記述子エクスポート
        
        Args:
            descriptors_df: 記述子データフレーム
            output_path: 出力パス
            format: 出力形式
            
        Returns:
            成功フラグ
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == "csv":
                descriptors_df.to_csv(output_path, index=False)
            elif format.lower() == "json":
                descriptors_df.to_json(output_path, orient='records', indent=2)
            elif format.lower() == "parquet":
                descriptors_df.to_parquet(output_path, index=False)
            else:
                logger.error(f"Unsupported format: {format}")
                return False
            
            logger.info(f"Descriptors exported to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting descriptors: {e}")
            return False

def create_rdkit_descriptors(config_path: Optional[str] = None, 
                             cache_dir: str = "cache") -> RDKitDescriptors:
    """
    RDKit記述子計算器作成
    
    Args:
        config_path: 設定ファイルパス
        cache_dir: キャッシュディレクトリ
        
    Returns:
        RDKitDescriptors
    """
    return RDKitDescriptors(config_path, cache_dir)

def calculate_rdkit_descriptors(smiles_list: List[str], 
                               descriptor_types: Optional[List[str]] = None,
                               config_path: Optional[str] = None,
                               cache_dir: str = "cache") -> pd.DataFrame:
    """
    RDKit記述子計算（簡易版）
    
    Args:
        smiles_list: SMILESリスト
        descriptor_types: 記述子タイプリスト
        config_path: 設定ファイルパス
        cache_dir: キャッシュディレクトリ
        
    Returns:
        記述子データフレーム
    """
    descriptors_calculator = create_rdkit_descriptors(config_path, cache_dir)
    return descriptors_calculator.calculate_descriptors(smiles_list, descriptor_types)

if __name__ == "__main__":
    # テスト実行
    descriptors_calculator = RDKitDescriptors()
    
    # テストSMILES
    test_smiles = ["CCO", "CCN", "c1ccccc1", "CC(=O)O"]
    
    # 記述子計算
    descriptors = descriptors_calculator.calculate_descriptors(test_smiles)
    
    print(f"Calculated descriptors for {len(descriptors)} molecules")
    if not descriptors.empty:
        print(f"Descriptor columns: {list(descriptors.columns)}")
        print(f"Descriptor shape: {descriptors.shape}")
        print(f"Sample descriptors:")
        print(descriptors.head())
