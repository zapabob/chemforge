"""
Molecular Features Tests

分子特徴量抽出のユニットテスト
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from chemforge.data.molecular_features import MolecularFeatures


class TestMolecularFeatures:
    """分子特徴量抽出のテストクラス"""
    
    def setup_method(self):
        """テスト前のセットアップ"""
        self.features = MolecularFeatures()
        self.test_smiles = "CCO"  # エタノール
        self.invalid_smiles = "INVALID"
    
    def test_initialization(self):
        """初期化テスト"""
        assert self.features.descriptor_names is not None
        assert len(self.features.descriptor_names) > 0
        assert self.features.fingerprint_generator is not None
    
    def test_extract_2d_descriptors_valid_smiles(self):
        """有効なSMILESの2D記述子抽出テスト"""
        descriptors = self.features.extract_2d_descriptors(self.test_smiles)
        
        assert isinstance(descriptors, dict)
        assert len(descriptors) > 0
        assert 'MolWt' in descriptors
        assert 'LogP' in descriptors
        assert 'NumHDonors' in descriptors
        assert 'NumHAcceptors' in descriptors
    
    def test_extract_2d_descriptors_invalid_smiles(self):
        """無効なSMILESの2D記述子抽出テスト"""
        descriptors = self.features.extract_2d_descriptors(self.invalid_smiles)
        
        assert isinstance(descriptors, dict)
        assert len(descriptors) == 0
    
    def test_extract_fingerprints_morgan(self):
        """Morganフィンガープリント抽出テスト"""
        fp = self.features.extract_fingerprints(self.test_smiles, "morgan")
        
        assert isinstance(fp, np.ndarray)
        assert len(fp) == 2048
        assert fp.dtype == bool or fp.dtype == int
    
    def test_extract_fingerprints_rdkit(self):
        """RDKitフィンガープリント抽出テスト"""
        fp = self.features.extract_fingerprints(self.test_smiles, "rdkit")
        
        assert isinstance(fp, np.ndarray)
        assert len(fp) == 2048
    
    def test_extract_fingerprints_maccs(self):
        """MACCSフィンガープリント抽出テスト"""
        fp = self.features.extract_fingerprints(self.test_smiles, "maccs")
        
        assert isinstance(fp, np.ndarray)
        assert len(fp) == 167
    
    def test_extract_3d_coordinates(self):
        """3D座標抽出テスト"""
        coords, atom_types = self.features.extract_3d_coordinates(self.test_smiles)
        
        assert isinstance(coords, np.ndarray)
        assert isinstance(atom_types, np.ndarray)
        assert len(coords) > 0
        assert len(atom_types) > 0
        assert coords.shape[1] == 3  # x, y, z座標
    
    def test_extract_scaffold_features(self):
        """骨格特徴量抽出テスト"""
        features = self.features.extract_scaffold_features(self.test_smiles)
        
        assert isinstance(features, dict)
        assert 'scaffold_smiles' in features
        assert 'scaffold_atoms' in features
        assert 'scaffold_rings' in features
    
    def test_extract_comprehensive_features(self):
        """包括的特徴量抽出テスト"""
        features = self.features.extract_comprehensive_features(
            self.test_smiles,
            include_3d=True,
            include_fingerprints=True,
            include_scaffolds=True
        )
        
        assert isinstance(features, dict)
        assert len(features) > 0
        assert 'MolWt' in features
        assert 'morgan_fp' in features
        assert '3d_coords' in features
        assert 'scaffold_smiles' in features
    
    def test_process_molecule_batch(self):
        """分子バッチ処理テスト"""
        smiles_list = [self.test_smiles, "CCN", "CCC"]
        
        df = self.features.process_molecule_batch(
            smiles_list,
            include_3d=False,
            include_fingerprints=True,
            include_scaffolds=True
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'smiles' in df.columns
        assert 'MolWt' in df.columns
    
    def test_get_feature_vector(self):
        """特徴量ベクトル取得テスト"""
        vector = self.features.get_feature_vector(
            self.test_smiles,
            feature_types=['2d_descriptors', 'morgan_fp']
        )
        
        assert isinstance(vector, np.ndarray)
        assert len(vector) > 0
    
    def test_get_feature_names(self):
        """特徴量名取得テスト"""
        names = self.features.get_feature_names(['2d_descriptors', 'morgan_fp'])
        
        assert isinstance(names, list)
        assert len(names) > 0
        assert 'MolWt' in names
        assert 'morgan_0' in names
    
    def test_error_handling(self):
        """エラーハンドリングテスト"""
        # 無効なSMILES
        result = self.features.extract_2d_descriptors(self.invalid_smiles)
        assert isinstance(result, dict)
        assert len(result) == 0
        
        # 空のSMILES
        result = self.features.extract_2d_descriptors("")
        assert isinstance(result, dict)
        assert len(result) == 0
    
    def test_fingerprint_consistency(self):
        """フィンガープリント一貫性テスト"""
        fp1 = self.features.extract_fingerprints(self.test_smiles, "morgan")
        fp2 = self.features.extract_fingerprints(self.test_smiles, "morgan")
        
        assert np.array_equal(fp1, fp2)
    
    def test_3d_coordinates_consistency(self):
        """3D座標一貫性テスト"""
        coords1, types1 = self.features.extract_3d_coordinates(self.test_smiles)
        coords2, types2 = self.features.extract_3d_coordinates(self.test_smiles)
        
        # 座標は最適化により異なる可能性があるため、原子数とタイプのみ確認
        assert len(coords1) == len(coords2)
        assert len(types1) == len(types2)
        assert np.array_equal(types1, types2)
