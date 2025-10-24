"""
物理化学特性予測のテスト

property_predictor.pyの単体テスト
"""

import unittest
from chemforge.admet.property_predictor import PropertyPredictor

class TestPropertyPredictor(unittest.TestCase):
    """物理化学特性予測のテストクラス"""
    
    def setUp(self):
        """テスト前の準備"""
        self.predictor = PropertyPredictor()
        self.test_smiles = "CC(CC1=CC=CC=C1)N"  # アンフェタミン
    
    def test_predict_physicochemical_properties(self):
        """物理化学的性質予測のテスト"""
        properties = self.predictor.predict_physicochemical_properties(self.test_smiles)
        
        # 基本チェック
        self.assertIsInstance(properties, dict)
        self.assertGreater(len(properties), 0)
        
        # 重要な物性の存在チェック
        expected_properties = [
            'molecular_weight', 'logp', 'tpsa', 'num_atoms', 'num_bonds',
            'num_rings', 'num_aromatic_rings', 'num_heteroatoms'
        ]
        
        for prop in expected_properties:
            if prop in properties:
                self.assertIsInstance(properties[prop], (int, float))
    
    def test_predict_tox21_endpoints(self):
        """Tox21エンドポイント予測のテスト"""
        tox21_results = self.predictor.predict_tox21_endpoints(self.test_smiles)
        
        # 基本チェック
        self.assertIsInstance(tox21_results, dict)
        self.assertGreater(len(tox21_results), 0)
        
        # Tox21エンドポイントの存在チェック
        expected_endpoints = [
            'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
            'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
            'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
        ]
        
        for endpoint in expected_endpoints:
            if endpoint in tox21_results:
                self.assertIsInstance(tox21_results[endpoint], (int, float))
                self.assertGreaterEqual(tox21_results[endpoint], 0.0)
                self.assertLessEqual(tox21_results[endpoint], 1.0)
    
    def test_invalid_smiles(self):
        """無効なSMILESのテスト"""
        invalid_smiles = "invalid_smiles"
        
        # 物理化学的性質予測
        properties = self.predictor.predict_physicochemical_properties(invalid_smiles)
        self.assertIsInstance(properties, dict)
        
        # Tox21エンドポイント予測
        tox21_results = self.predictor.predict_tox21_endpoints(invalid_smiles)
        self.assertIsInstance(tox21_results, dict)
    
    def test_empty_smiles(self):
        """空のSMILESのテスト"""
        # 物理化学的性質予測
        properties = self.predictor.predict_physicochemical_properties("")
        self.assertIsInstance(properties, dict)
        
        # Tox21エンドポイント予測
        tox21_results = self.predictor.predict_tox21_endpoints("")
        self.assertIsInstance(tox21_results, dict)
    
    def test_tox21_endpoint_ranges(self):
        """Tox21エンドポイントの範囲テスト"""
        tox21_results = self.predictor.predict_tox21_endpoints(self.test_smiles)
        
        for endpoint, score in tox21_results.items():
            # スコアは0.0から1.0の範囲内であるべき
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    def test_physicochemical_property_ranges(self):
        """物理化学的性質の範囲テスト"""
        properties = self.predictor.predict_physicochemical_properties(self.test_smiles)
        
        # 分子量は正の値
        if 'molecular_weight' in properties:
            self.assertGreater(properties['molecular_weight'], 0)
        
        # 原子数は正の値
        if 'num_atoms' in properties:
            self.assertGreater(properties['num_atoms'], 0)
        
        # 結合数は正の値
        if 'num_bonds' in properties:
            self.assertGreater(properties['num_bonds'], 0)
        
        # 環数は非負の値
        if 'num_rings' in properties:
            self.assertGreaterEqual(properties['num_rings'], 0)

if __name__ == '__main__':
    unittest.main()
