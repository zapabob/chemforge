"""
骨格ベース分子生成のテスト

scaffold_generator.pyの単体テスト
"""

import unittest
from chemforge.generation.scaffold_generator import ScaffoldGenerator, GeneratedMolecule
from chemforge.data.cns_scaffolds import CNSScaffolds

class TestScaffoldGenerator(unittest.TestCase):
    """骨格ベース分子生成のテストクラス"""
    
    def setUp(self):
        """テスト前の準備"""
        self.scaffolds = CNSScaffolds()
        self.generator = ScaffoldGenerator(self.scaffolds)
    
    def test_initialization(self):
        """初期化のテスト"""
        self.assertIsInstance(self.generator, ScaffoldGenerator)
        self.assertIsInstance(self.generator.scaffolds, CNSScaffolds)
        self.assertIsInstance(self.generator.generated_molecules, list)
    
    def test_generate_analogs(self):
        """類似体生成のテスト"""
        scaffold_types = self.scaffolds.get_all_scaffold_types()
        
        for scaffold_type in scaffold_types:
            analogs = self.generator.generate_analogs(scaffold_type, num_analogs=5)
            
            self.assertIsInstance(analogs, list)
            self.assertLessEqual(len(analogs), 5)
            
            # 各類似体の属性チェック
            for analog in analogs:
                self.assertIsInstance(analog, GeneratedMolecule)
                self.assertIsInstance(analog.smiles, str)
                self.assertIsInstance(analog.scaffold_type, str)
                self.assertIsInstance(analog.parent_compound, str)
                self.assertIsInstance(analog.modifications, list)
                self.assertIsInstance(analog.properties, dict)
                self.assertIsInstance(analog.score, (int, float))
                
                # スコアの範囲チェック
                self.assertGreaterEqual(analog.score, 0.0)
                self.assertLessEqual(analog.score, 1.0)
    
    def test_generate_analogs_with_modification_types(self):
        """修飾タイプ指定での類似体生成のテスト"""
        scaffold_types = self.scaffolds.get_all_scaffold_types()
        
        for scaffold_type in scaffold_types:
            modification_types = ['substitution', 'addition', 'removal']
            analogs = self.generator.generate_analogs(
                scaffold_type, 
                num_analogs=3,
                modification_types=modification_types
            )
            
            self.assertIsInstance(analogs, list)
            self.assertLessEqual(len(analogs), 3)
    
    def test_optimize_molecules(self):
        """分子最適化のテスト"""
        # テスト用の分子を生成
        scaffold_types = self.scaffolds.get_all_scaffold_types()
        if scaffold_types:
            scaffold_type = scaffold_types[0]
            analogs = self.generator.generate_analogs(scaffold_type, num_analogs=3)
            
            if analogs:
                # 目標物性を設定
                target_properties = {
                    'molecular_weight': 300.0,
                    'logp': 2.5
                }
                
                optimized = self.generator.optimize_molecules(target_properties)
                
                self.assertIsInstance(optimized, list)
    
    def test_get_best_molecules(self):
        """最高スコア分子取得のテスト"""
        # テスト用の分子を生成
        scaffold_types = self.scaffolds.get_all_scaffold_types()
        if scaffold_types:
            scaffold_type = scaffold_types[0]
            analogs = self.generator.generate_analogs(scaffold_type, num_analogs=5)
            
            if analogs:
                best_molecules = self.generator.get_best_molecules(top_k=3)
                
                self.assertIsInstance(best_molecules, list)
                self.assertLessEqual(len(best_molecules), 3)
                
                # スコアの降順でソートされているかチェック
                if len(best_molecules) > 1:
                    for i in range(len(best_molecules) - 1):
                        self.assertGreaterEqual(
                            best_molecules[i].score, 
                            best_molecules[i + 1].score
                        )
    
    def test_export_results(self):
        """結果エクスポートのテスト"""
        import tempfile
        import os
        
        # テスト用の分子を生成
        scaffold_types = self.scaffolds.get_all_scaffold_types()
        if scaffold_types:
            scaffold_type = scaffold_types[0]
            analogs = self.generator.generate_analogs(scaffold_type, num_analogs=3)
            
            if analogs:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    temp_path = f.name
                
                try:
                    self.generator.export_results(temp_path)
                    
                    # ファイルが作成されたかチェック
                    self.assertTrue(os.path.exists(temp_path))
                    
                    # ファイルサイズが0より大きいかチェック
                    self.assertGreater(os.path.getsize(temp_path), 0)
                    
                finally:
                    # 一時ファイルを削除
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
    
    def test_generated_molecule_properties(self):
        """生成された分子の物性テスト"""
        scaffold_types = self.scaffolds.get_all_scaffold_types()
        
        for scaffold_type in scaffold_types:
            analogs = self.generator.generate_analogs(scaffold_type, num_analogs=1)
            
            if analogs:
                analog = analogs[0]
                
                # 物性の存在チェック
                expected_properties = [
                    'molecular_weight', 'logp', 'tpsa', 'num_atoms', 'num_bonds'
                ]
                
                for prop in expected_properties:
                    if prop in analog.properties:
                        self.assertIsInstance(analog.properties[prop], (int, float))
                        self.assertGreaterEqual(analog.properties[prop], 0)

if __name__ == '__main__':
    unittest.main()
