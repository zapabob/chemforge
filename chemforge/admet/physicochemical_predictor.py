"""
Physicochemical Properties Predictor Module

物理化学特性予測モジュール
既存MolecularFeaturesを活用した効率的な物理化学特性予測
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
from chemforge.data.molecular_features import MolecularFeatures
from chemforge.utils.config_utils import ConfigManager
from chemforge.utils.logging_utils import Logger
from chemforge.utils.validation import DataValidator

logger = logging.getLogger(__name__)

class PhysicochemicalPredictor:
    """
    物理化学特性予測クラス
    
    既存MolecularFeaturesを活用した効率的な物理化学特性予測
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
        self.logger = Logger("PhysicochemicalPredictor")
        self.validator = DataValidator()
        
        # 分子特徴量計算器
        self.molecular_features = MolecularFeatures(config_path, cache_dir)
        
        # 物理化学特性予測設定
        self.physico_config = self.config.get('physicochemical_prediction', {})
        self.property_types = self.physico_config.get('property_types', [
            'basic', 'lipinski', 'qed', 'sasa', 'conformational'
        ])
        self.use_cache = self.physico_config.get('use_cache', True)
        self.normalize_properties = self.physico_config.get('normalize_properties', True)
        
        logger.info("PhysicochemicalPredictor initialized")
    
    def predict_properties(self, smiles_list: List[str], 
                          property_types: Optional[List[str]] = None,
                          use_cache: bool = True) -> pd.DataFrame:
        """
        物理化学特性予測
        
        Args:
            smiles_list: SMILESリスト
            property_types: 特性タイプリスト
            use_cache: キャッシュ使用
            
        Returns:
            物理化学特性予測結果データフレーム
        """
        logger.info(f"Predicting physicochemical properties for {len(smiles_list)} molecules")
        
        if property_types is None:
            property_types = self.property_types
        
        # キャッシュチェック
        cache_key = f"physicochemical_properties_{len(smiles_list)}_{'_'.join(property_types)}"
        cache_path = self.cache_dir / f"{cache_key}.csv"
        
        if use_cache and cache_path.exists():
            logger.info(f"Loading from cache: {cache_path}")
            return pd.read_csv(cache_path)
        
        # 物理化学特性予測実行
        all_properties = []
        
        for i, smiles in enumerate(tqdm(smiles_list, desc="Predicting physicochemical properties")):
            try:
                # 分子特性予測
                molecule_properties = self._predict_single_molecule_properties(
                    smiles, property_types
                )
                
                if molecule_properties:
                    molecule_properties['smiles'] = smiles
                    all_properties.append(molecule_properties)
                
            except Exception as e:
                logger.error(f"Failed to predict properties for {smiles}: {e}")
                continue
        
        if not all_properties:
            logger.warning("No physicochemical properties predicted")
            return pd.DataFrame()
        
        # データフレーム作成
        properties_df = pd.DataFrame(all_properties)
        logger.info(f"Predicted physicochemical properties for {len(properties_df)} molecules")
        
        # 特性正規化
        if self.normalize_properties:
            properties_df = self._normalize_properties(properties_df)
        
        # キャッシュ保存
        if use_cache:
            properties_df.to_csv(cache_path, index=False)
            logger.info(f"Physicochemical properties cached to: {cache_path}")
        
        return properties_df
    
    def _predict_single_molecule_properties(self, smiles: str, 
                                          property_types: List[str]) -> Optional[Dict]:
        """
        単一分子物理化学特性予測
        
        Args:
            smiles: SMILES文字列
            property_types: 特性タイプリスト
            
        Returns:
            物理化学特性予測結果辞書
        """
        try:
            if not RDKIT_AVAILABLE:
                logger.warning("RDKit not available, using basic properties only")
                return self._predict_basic_properties(smiles)
            
            # 分子オブジェクト作成
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Failed to create molecule from SMILES: {smiles}")
                return None
            
            properties = {}
            
            # 基本特性
            if 'basic' in property_types:
                basic_properties = self._predict_basic_properties(smiles, mol)
                properties.update(basic_properties)
            
            # Lipinski特性
            if 'lipinski' in property_types:
                lipinski_properties = self._predict_lipinski_properties(mol)
                properties.update(lipinski_properties)
            
            # QED特性
            if 'qed' in property_types:
                qed_properties = self._predict_qed_properties(mol)
                properties.update(qed_properties)
            
            # SASA特性
            if 'sasa' in property_types:
                sasa_properties = self._predict_sasa_properties(mol)
                properties.update(sasa_properties)
            
            # 立体構造特性
            if 'conformational' in property_types:
                conformational_properties = self._predict_conformational_properties(mol)
                properties.update(conformational_properties)
            
            return properties
            
        except Exception as e:
            logger.error(f"Error predicting physicochemical properties for {smiles}: {e}")
            return None
    
    def _predict_basic_properties(self, smiles: str, mol: Optional[Chem.Mol] = None) -> Dict:
        """
        基本物理化学特性予測
        
        Args:
            smiles: SMILES文字列
            mol: 分子オブジェクト
            
        Returns:
            基本特性辞書
        """
        properties = {}
        
        try:
            if mol is not None and RDKIT_AVAILABLE:
                # 分子量
                properties['molecular_weight'] = Descriptors.MolWt(mol)
                
                # LogP
                properties['logp'] = Descriptors.MolLogP(mol)
                
                # TPSA
                properties['tpsa'] = Descriptors.TPSA(mol)
                
                # HBD/HBA
                properties['hbd'] = Descriptors.NumHDonors(mol)
                properties['hba'] = Descriptors.NumHAcceptors(mol)
                
                # 回転可能結合
                properties['rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
                
                # 芳香族環
                properties['aromatic_rings'] = Descriptors.NumAromaticRings(mol)
                
                # 原子数
                properties['num_atoms'] = mol.GetNumAtoms()
                properties['num_bonds'] = mol.GetNumBonds()
                
                # 環数
                properties['num_rings'] = Descriptors.RingCount(mol)
                
                # 立体中心
                properties['num_stereocenters'] = Descriptors.NumAliphaticCarbocycles(mol)
                
                # 極性表面積
                properties['polar_surface_area'] = Descriptors.TPSA(mol)
                
                # 分子体積
                properties['molecular_volume'] = Descriptors.MolMR(mol)
                
                # 密度
                properties['density'] = Descriptors.MolWt(mol) / (Descriptors.MolMR(mol) + 1e-8)
                
            else:
                # RDKitが利用できない場合の基本特性
                properties['smiles_length'] = len(smiles)
                properties['num_carbons'] = smiles.count('C')
                properties['num_nitrogens'] = smiles.count('N')
                properties['num_oxygens'] = smiles.count('O')
                properties['num_sulfurs'] = smiles.count('S')
                properties['num_rings'] = smiles.count('c') + smiles.count('C')
                
        except Exception as e:
            logger.error(f"Error predicting basic properties: {e}")
        
        return properties
    
    def _predict_lipinski_properties(self, mol: Chem.Mol) -> Dict:
        """
        Lipinski特性予測
        
        Args:
            mol: 分子オブジェクト
            
        Returns:
            Lipinski特性辞書
        """
        properties = {}
        
        try:
            # Lipinski's Rule of Five
            properties['lipinski_mw'] = Descriptors.MolWt(mol)
            properties['lipinski_logp'] = Descriptors.MolLogP(mol)
            properties['lipinski_hbd'] = Descriptors.NumHDonors(mol)
            properties['lipinski_hba'] = Descriptors.NumHAcceptors(mol)
            
            # Lipinski違反数
            violations = 0
            if properties['lipinski_mw'] > 500:
                violations += 1
            if properties['lipinski_logp'] > 5:
                violations += 1
            if properties['lipinski_hbd'] > 5:
                violations += 1
            if properties['lipinski_hba'] > 10:
                violations += 1
            
            properties['lipinski_violations'] = violations
            properties['lipinski_compliant'] = violations == 0
            
            # Veber's Rule
            properties['veber_tpsa'] = Descriptors.TPSA(mol)
            properties['veber_rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
            
            # Veber違反数
            veber_violations = 0
            if properties['veber_tpsa'] > 140:
                veber_violations += 1
            if properties['veber_rotatable_bonds'] > 10:
                veber_violations += 1
            
            properties['veber_violations'] = veber_violations
            properties['veber_compliant'] = veber_violations == 0
            
        except Exception as e:
            logger.error(f"Error predicting Lipinski properties: {e}")
        
        return properties
    
    def _predict_qed_properties(self, mol: Chem.Mol) -> Dict:
        """
        QED特性予測
        
        Args:
            mol: 分子オブジェクト
            
        Returns:
            QED特性辞書
        """
        properties = {}
        
        try:
            # QEDスコア
            properties['qed_score'] = QED.qed(mol)
            
            # QED成分
            qed_components = QED.properties(mol)
            properties['qed_mw'] = qed_components[0]
            properties['qed_logp'] = qed_components[1]
            properties['qed_hbd'] = qed_components[2]
            properties['qed_hba'] = qed_components[3]
            properties['qed_psa'] = qed_components[4]
            properties['qed_rotb'] = qed_components[5]
            properties['qed_aromat'] = qed_components[6]
            properties['qed_alert'] = qed_components[7]
            
        except Exception as e:
            logger.error(f"Error predicting QED properties: {e}")
        
        return properties
    
    def _predict_sasa_properties(self, mol: Chem.Mol) -> Dict:
        """
        SASA特性予測
        
        Args:
            mol: 分子オブジェクト
            
        Returns:
            SASA特性辞書
        """
        properties = {}
        
        try:
            # 3D構造生成
            mol_3d = Chem.AddHs(mol)
            rdDistGeom.EmbedMolecule(mol_3d)
            
            # SASA計算
            sasa = rdFreeSASA.CalcSASA(mol_3d)
            properties['sasa'] = sasa
            
            # 原子別SASA
            atom_sasas = rdFreeSASA.CalcSASA(mol_3d, confId=0)
            properties['atom_sasa_mean'] = np.mean(atom_sasas) if atom_sasas else 0
            properties['atom_sasa_std'] = np.std(atom_sasas) if atom_sasas else 0
            
            # 分子表面積
            properties['molecular_surface_area'] = sasa
            
        except Exception as e:
            logger.error(f"Error predicting SASA properties: {e}")
        
        return properties
    
    def _predict_conformational_properties(self, mol: Chem.Mol) -> Dict:
        """
        立体構造特性予測
        
        Args:
            mol: 分子オブジェクト
            
        Returns:
            立体構造特性辞書
        """
        properties = {}
        
        try:
            # 3D構造生成
            mol_3d = Chem.AddHs(mol)
            rdDistGeom.EmbedMolecule(mol_3d)
            
            # 立体構造特性
            properties['num_conformers'] = mol_3d.GetNumConformers()
            
            # 分子形状
            properties['molecular_volume'] = rdMolDescriptors.CalcCrippenDescriptors(mol_3d)[0]
            properties['molecular_surface'] = rdMolDescriptors.CalcCrippenDescriptors(mol_3d)[1]
            
            # 立体配座
            properties['num_stereocenters'] = Descriptors.NumAliphaticCarbocycles(mol_3d)
            properties['num_rotatable_bonds'] = Descriptors.NumRotatableBonds(mol_3d)
            
            # 分子形状記述子
            properties['bertz_ct'] = Descriptors.BertzCT(mol_3d)
            properties['balaban_j'] = Descriptors.BalabanJ(mol_3d)
            
        except Exception as e:
            logger.error(f"Error predicting conformational properties: {e}")
        
        return properties
    
    def _normalize_properties(self, properties_df: pd.DataFrame) -> pd.DataFrame:
        """
        特性正規化
        
        Args:
            properties_df: 特性データフレーム
            
        Returns:
            正規化済みデータフレーム
        """
        try:
            from sklearn.preprocessing import StandardScaler
            
            # 数値列のみ正規化
            numeric_columns = properties_df.select_dtypes(include=[np.number]).columns
            numeric_columns = [col for col in numeric_columns if col not in ['smiles']]
            
            if len(numeric_columns) > 0:
                scaler = StandardScaler()
                properties_df[numeric_columns] = scaler.fit_transform(properties_df[numeric_columns])
                
                # スケーラー保存
                scaler_path = self.cache_dir / "physicochemical_scaler.pkl"
                import pickle
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
                
                logger.info(f"Properties normalized and scaler saved to: {scaler_path}")
            
        except Exception as e:
            logger.error(f"Error normalizing properties: {e}")
        
        return properties_df
    
    def get_property_summary(self, properties_df: pd.DataFrame) -> Dict:
        """
        特性サマリー取得
        
        Args:
            properties_df: 特性データフレーム
            
        Returns:
            サマリー辞書
        """
        try:
            summary = {
                'total_molecules': len(properties_df),
                'property_columns': list(properties_df.columns),
                'missing_values': properties_df.isnull().sum().to_dict(),
                'statistics': {}
            }
            
            # 数値列の統計
            numeric_columns = properties_df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                summary['statistics'][col] = {
                    'mean': properties_df[col].mean(),
                    'std': properties_df[col].std(),
                    'min': properties_df[col].min(),
                    'max': properties_df[col].max(),
                    'median': properties_df[col].median()
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating property summary: {e}")
            return {}
    
    def export_properties(self, properties_df: pd.DataFrame, output_path: str, 
                         format: str = "csv") -> bool:
        """
        特性エクスポート
        
        Args:
            properties_df: 特性データフレーム
            output_path: 出力パス
            format: 出力形式
            
        Returns:
            成功フラグ
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == "csv":
                properties_df.to_csv(output_path, index=False)
            elif format.lower() == "json":
                properties_df.to_json(output_path, orient='records', indent=2)
            elif format.lower() == "parquet":
                properties_df.to_parquet(output_path, index=False)
            else:
                logger.error(f"Unsupported format: {format}")
                return False
            
            logger.info(f"Physicochemical properties exported to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting properties: {e}")
            return False

def create_physicochemical_predictor(config_path: Optional[str] = None, 
                                    cache_dir: str = "cache") -> PhysicochemicalPredictor:
    """
    物理化学特性予測器作成
    
    Args:
        config_path: 設定ファイルパス
        cache_dir: キャッシュディレクトリ
        
    Returns:
        PhysicochemicalPredictor
    """
    return PhysicochemicalPredictor(config_path, cache_dir)

def predict_physicochemical_properties(smiles_list: List[str], 
                                      property_types: Optional[List[str]] = None,
                                      config_path: Optional[str] = None,
                                      cache_dir: str = "cache") -> pd.DataFrame:
    """
    物理化学特性予測（簡易版）
    
    Args:
        smiles_list: SMILESリスト
        property_types: 特性タイプリスト
        config_path: 設定ファイルパス
        cache_dir: キャッシュディレクトリ
        
    Returns:
        物理化学特性予測結果データフレーム
    """
    predictor = create_physicochemical_predictor(config_path, cache_dir)
    return predictor.predict_properties(smiles_list, property_types)

if __name__ == "__main__":
    # テスト実行
    predictor = PhysicochemicalPredictor()
    
    # テストSMILES
    test_smiles = ["CCO", "CCN", "c1ccccc1", "CC(=O)O"]
    
    # 物理化学特性予測
    properties = predictor.predict_properties(test_smiles)
    
    print(f"Physicochemical properties for {len(properties)} molecules")
    if not properties.empty:
        print(f"Property columns: {list(properties.columns)}")
        print(f"Property shape: {properties.shape}")
        print(f"Sample properties:")
        print(properties.head())
        
        # サマリー
        summary = predictor.get_property_summary(properties)
        print(f"Property summary: {summary}")
