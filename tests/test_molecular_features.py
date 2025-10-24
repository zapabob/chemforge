"""
分子特徴量計算のテスト

molecular_features.pyの単体テスト
"""

import unittest
import numpy as np
from chemforge.data.molecular_features import MolecularFeatures

class TestMolecularFeatures(unittest.TestCase):
    """分子特徴量計算のテストクラス"""
    
    def setUp(self):
        """テスト前の準備"""
        self.mf = MolecularFeatures()
        self.test_smiles = "CC(CC1=CC=CC=C1)N"  # アンフェタミン
    
    def test_calculate_features_basic(self):
        """基本特徴量計算のテスト"""
        features = self.mf.calculate_single_features(self.test_smiles)
        
        # 基本チェック
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 0)
        
        # 重要な特徴量の存在チェック
        expected_features = [
            'molecular_weight', 'logp', 'tpsa', 'num_atoms', 'num_bonds',
            'num_rings', 'aromatic_rings'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], (int, float))
    
    def test_calculate_features_stereocenters(self):
        """立体中心計算のテスト"""
        features = self.mf.calculate_single_features(self.test_smiles)
        
        # 立体中心の数が正しく計算されているか
        self.assertIn('num_stereocenters', features)
        self.assertIsInstance(features['num_stereocenters'], int)
        self.assertGreaterEqual(features['num_stereocenters'], 0)
    
    def test_calculate_features_sasa(self):
        """SASA計算のテスト"""
        features = self.mf.calculate_single_features(self.test_smiles)
        
        # SASA関連の特徴量
        sasa_features = ['sasa', 'atom_sasa_mean', 'atom_sasa_std']
        for feature in sasa_features:
            if feature in features:
                self.assertIsInstance(features[feature], (int, float))
                self.assertGreaterEqual(features[feature], 0)
    
    def test_calculate_features_crippen(self):
        """Crippen記述子のテスト"""
        features = self.mf.calculate_single_features(self.test_smiles)
        
        # Crippen記述子の存在チェック
        crippen_features = ['molecular_volume', 'molecular_surface']
        for feature in crippen_features:
            if feature in features:
                self.assertIsInstance(features[feature], (int, float))
    
    def test_invalid_smiles(self):
        """無効なSMILESのテスト"""
        invalid_smiles = "invalid_smiles"
        features = self.mf.calculate_single_features(invalid_smiles)
        
        # 無効なSMILESの場合は空辞書またはデフォルト値
        self.assertIsInstance(features, dict)
    
    def test_empty_smiles(self):
        """空のSMILESのテスト"""
        features = self.mf.calculate_single_features("")
        self.assertIsInstance(features, dict)

if __name__ == '__main__':
    unittest.main()
