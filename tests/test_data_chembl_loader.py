"""
ChEMBL Loader Tests

ChEMBLデータローダーのユニットテスト
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

from chemforge.data.chembl_loader import ChEMBLLoader


class TestChEMBLLoader:
    """ChEMBLローダーのテストクラス"""
    
    def setup_method(self):
        """テスト前のセットアップ"""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = ChEMBLLoader(cache_dir=self.temp_dir)
    
    def teardown_method(self):
        """テスト後のクリーンアップ"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """初期化テスト"""
        assert self.loader.cache_dir == self.temp_dir
        assert self.loader.pic50_threshold == 6.0
        assert self.loader.max_outlier_std == 3.0
        assert self.loader.targets is not None
    
    @patch('chemforge.data.chembl_loader.new_client')
    def test_get_target_data_success(self, mock_client):
        """ターゲットデータ取得成功テスト"""
        # モックデータを設定
        mock_activities = [
            {
                'molecule_chembl_id': 'CHEMBL123',
                'canonical_smiles': 'CCO',
                'standard_value': 1000,
                'standard_units': 'nM',
                'assay_type': 'B',
                'assay_description': 'Test assay'
            },
            {
                'molecule_chembl_id': 'CHEMBL456',
                'canonical_smiles': 'CCN',
                'standard_value': 100,
                'standard_units': 'nM',
                'assay_type': 'B',
                'assay_description': 'Test assay'
            }
        ]
        
        mock_client.activity.filter.return_value.only.return_value = mock_activities
        
        # ターゲットデータを取得
        result = self.loader.get_target_data('5HT2A', max_compounds=10)
        
        # 結果を検証
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert 'molecule_chembl_id' in result.columns
        assert 'canonical_smiles' in result.columns
        assert 'pIC50' in result.columns
    
    @patch('chemforge.data.chembl_loader.new_client')
    def test_get_target_data_invalid_target(self, mock_client):
        """無効なターゲットのテスト"""
        # ターゲットデータを取得
        result = self.loader.get_target_data('INVALID_TARGET')
        
        # 結果を検証
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_get_all_cns_data(self):
        """全CNSデータ取得テスト"""
        with patch.object(self.loader, 'get_target_data') as mock_get_data:
            # モックデータを設定
            mock_data = pd.DataFrame({
                'molecule_chembl_id': ['CHEMBL123', 'CHEMBL456'],
                'canonical_smiles': ['CCO', 'CCN'],
                'pIC50': [8.0, 7.5],
                'assay_type': ['B', 'B'],
                'assay_description': ['Test', 'Test']
            })
            mock_get_data.return_value = mock_data
            
            # 全CNSデータを取得
            result = self.loader.get_all_cns_data(max_compounds_per_target=10)
            
            # 結果を検証
            assert isinstance(result, dict)
            assert len(result) > 0
    
    def test_create_multi_target_dataset(self):
        """マルチターゲットデータセット作成テスト"""
        # テストデータを作成
        target_data = {
            '5HT2A': pd.DataFrame({
                'molecule_chembl_id': ['CHEMBL123', 'CHEMBL456'],
                'canonical_smiles': ['CCO', 'CCN'],
                'pIC50': [8.0, 7.5]
            }),
            'D1': pd.DataFrame({
                'molecule_chembl_id': ['CHEMBL123', 'CHEMBL789'],
                'canonical_smiles': ['CCO', 'CCC'],
                'pIC50': [7.0, 6.5]
            })
        }
        
        # マルチターゲットデータセットを作成
        result = self.loader.create_multi_target_dataset(target_data, min_compounds_per_target=1)
        
        # 結果を検証
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert '5HT2A_pIC50' in result.columns
        assert 'D1_pIC50' in result.columns
        assert 'canonical_smiles' in result.columns
    
    def test_filter_outliers_iqr(self):
        """IQR外れ値フィルタリングテスト"""
        # テストデータを作成
        df = pd.DataFrame({
            'target1': [1, 2, 3, 4, 5, 100],  # 100は外れ値
            'target2': [1, 2, 3, 4, 5, 6]
        })
        
        # 外れ値をフィルタリング
        result = self.loader.filter_outliers(df, ['target1'], method='iqr')
        
        # 結果を検証
        assert len(result) < len(df)
        assert 100 not in result['target1'].values
    
    def test_filter_outliers_zscore(self):
        """Z-score外れ値フィルタリングテスト"""
        # テストデータを作成
        df = pd.DataFrame({
            'target1': [1, 2, 3, 4, 5, 100],  # 100は外れ値
            'target2': [1, 2, 3, 4, 5, 6]
        })
        
        # 外れ値をフィルタリング
        result = self.loader.filter_outliers(df, ['target1'], method='zscore')
        
        # 結果を検証
        assert len(result) < len(df)
        assert 100 not in result['target1'].values
    
    def test_save_and_load_dataset(self):
        """データセット保存・読み込みテスト"""
        # テストデータを作成
        df = pd.DataFrame({
            'molecule_chembl_id': ['CHEMBL123', 'CHEMBL456'],
            'canonical_smiles': ['CCO', 'CCN'],
            'pIC50': [8.0, 7.5]
        })
        
        # データセットを保存
        filepath = self.loader.save_dataset(df, 'test_dataset', format='csv')
        assert os.path.exists(filepath)
        
        # データセットを読み込み
        loaded_df = self.loader.load_dataset('test_dataset', format='csv')
        
        # 結果を検証
        assert isinstance(loaded_df, pd.DataFrame)
        assert len(loaded_df) == len(df)
        assert list(loaded_df.columns) == list(df.columns)
    
    def test_get_dataset_summary(self):
        """データセット要約テスト"""
        # テストデータを作成
        df = pd.DataFrame({
            'molecule_chembl_id': ['CHEMBL123', 'CHEMBL456'],
            'canonical_smiles': ['CCO', 'CCN'],
            '5HT2A_pIC50': [8.0, 7.5],
            'D1_pIC50': [7.0, 6.5]
        })
        
        # 要約を取得
        summary = self.loader.get_dataset_summary(df)
        
        # 結果を検証
        assert 'total_compounds' in summary
        assert 'targets' in summary
        assert 'missing_values' in summary
        assert 'target_statistics' in summary
        assert summary['total_compounds'] == 2
        assert len(summary['targets']) == 2
    
    def test_pic50_conversion(self):
        """pIC50変換テスト"""
        # テストデータを作成
        test_cases = [
            (1000, 'nM', 6.0),  # 1000 nM = 6.0 pIC50
            (100, 'nM', 7.0),   # 100 nM = 7.0 pIC50
            (1, 'μM', 6.0),     # 1 μM = 6.0 pIC50
            (1, 'mM', 3.0),     # 1 mM = 3.0 pIC50
        ]
        
        for value, units, expected_pic50 in test_cases:
            if units == 'nM':
                pic50 = -np.log10(value * 1e-9)
            elif units == 'μM':
                pic50 = -np.log10(value * 1e-6)
            elif units == 'mM':
                pic50 = -np.log10(value * 1e-3)
            
            assert abs(pic50 - expected_pic50) < 0.01
    
    def test_error_handling(self):
        """エラーハンドリングテスト"""
        # 無効なターゲット名
        result = self.loader.get_target_data('INVALID_TARGET')
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        
        # 空のデータフレーム
        empty_df = pd.DataFrame()
        result = self.loader.create_multi_target_dataset({'target': empty_df})
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
