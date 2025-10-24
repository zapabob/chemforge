"""
CNS骨格ライブラリのテスト

cns_scaffolds.pyの単体テスト
"""

import unittest
from chemforge.data.cns_scaffolds import CNSScaffolds, CNSCompound

class TestCNSScaffolds(unittest.TestCase):
    """CNS骨格ライブラリのテストクラス"""
    
    def setUp(self):
        """テスト前の準備"""
        self.scaffolds = CNSScaffolds()
    
    def test_initialization(self):
        """初期化のテスト"""
        self.assertIsInstance(self.scaffolds, CNSScaffolds)
        self.assertIsInstance(self.scaffolds.scaffolds, dict)
    
    def test_get_all_scaffold_types(self):
        """骨格タイプ取得のテスト"""
        scaffold_types = self.scaffolds.get_all_scaffold_types()
        
        self.assertIsInstance(scaffold_types, list)
        self.assertGreater(len(scaffold_types), 0)
        
        # 期待される骨格タイプ
        expected_types = [
            'phenethylamine', 'tryptamine', 'cannabinoid',
            'opioid', 'anti_nmda', 'gaba_agonist'
        ]
        
        for expected_type in expected_types:
            self.assertIn(expected_type, scaffold_types)
    
    def test_get_scaffold_compounds(self):
        """骨格化合物取得のテスト"""
        scaffold_types = self.scaffolds.get_all_scaffold_types()
        
        for scaffold_type in scaffold_types:
            compounds = self.scaffolds.get_scaffold_compounds(scaffold_type)
            
            self.assertIsInstance(compounds, list)
            self.assertGreater(len(compounds), 0)
            
            # 各化合物の属性チェック
            for compound in compounds:
                self.assertIsInstance(compound, CNSCompound)
                self.assertIsInstance(compound.name, str)
                self.assertIsInstance(compound.smiles, str)
                self.assertIsInstance(compound.smarts, str)
                self.assertIsInstance(compound.mechanism, str)
                self.assertIsInstance(compound.safety_notes, str)
                self.assertIsInstance(compound.therapeutic_use, str)
    
    def test_find_compounds_by_mechanism(self):
        """作用機序検索のテスト"""
        # ドパミン関連の化合物を検索
        dopamine_compounds = self.scaffolds.find_compounds_by_mechanism("dopamine")
        
        self.assertIsInstance(dopamine_compounds, list)
        
        # セロトニン関連の化合物を検索
        serotonin_compounds = self.scaffolds.find_compounds_by_mechanism("serotonin")
        
        self.assertIsInstance(serotonin_compounds, list)
    
    def test_get_safety_warnings(self):
        """安全性情報取得のテスト"""
        # 存在する化合物の安全性情報
        safety_info = self.scaffolds.get_safety_warnings("Amphetamine")
        
        if safety_info:
            self.assertIsInstance(safety_info, str)
            self.assertGreater(len(safety_info), 0)
        
        # 存在しない化合物の安全性情報
        safety_info = self.scaffolds.get_safety_warnings("NonExistentCompound")
        self.assertIsNone(safety_info)
    
    def test_export_scaffolds_to_json(self):
        """JSONエクスポートのテスト"""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            self.scaffolds.export_scaffolds_to_json(temp_path)
            
            # ファイルが作成されたかチェック
            self.assertTrue(os.path.exists(temp_path))
            
            # ファイルサイズが0より大きいかチェック
            self.assertGreater(os.path.getsize(temp_path), 0)
            
        finally:
            # 一時ファイルを削除
            if os.path.exists(temp_path):
                os.unlink(temp_path)

if __name__ == '__main__':
    unittest.main()
