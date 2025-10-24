"""
Test Utilities

テストユーティリティ
テスト用ヘルパー関数・クラス
"""

import unittest
import tempfile
import shutil
import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# テスト用設定
TEST_CONFIG = {
    'test_data_dir': 'tests/data',
    'test_cache_dir': 'tests/cache',
    'test_output_dir': 'tests/output',
    'test_log_level': 'DEBUG'
}

class TestBase(unittest.TestCase):
    """
    テストベースクラス
    
    共通のテスト機能を提供
    """
    
    def setUp(self):
        """テスト前処理"""
        # 一時ディレクトリ作成
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = Path(self.temp_dir) / "data"
        self.test_cache_dir = Path(self.temp_dir) / "cache"
        self.test_output_dir = Path(self.temp_dir) / "output"
        
        # ディレクトリ作成
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
        self.test_cache_dir.mkdir(parents=True, exist_ok=True)
        self.test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # テスト用設定ファイル作成
        self.config_path = self.test_data_dir / "test_config.yaml"
        self._create_test_config()
        
        # ログ設定
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def tearDown(self):
        """テスト後処理"""
        # 一時ディレクトリ削除
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_config(self):
        """テスト用設定ファイル作成"""
        config = {
            'test_mode': True,
            'debug': True,
            'log_level': 'DEBUG',
            'database': {
                'type': 'sqlite',
                'path': str(self.test_data_dir / 'test.db')
            },
            'models': {
                'transformer': {
                    'd_model': 128,
                    'n_layers': 2,
                    'n_heads': 4
                },
                'gnn': {
                    'hidden_dim': 64,
                    'n_layers': 2
                }
            },
            'training': {
                'batch_size': 4,
                'epochs': 2,
                'learning_rate': 0.001
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def create_test_dataframe(self, n_rows: int = 10) -> pd.DataFrame:
        """
        テスト用DataFrame作成
        
        Args:
            n_rows: 行数
            
        Returns:
            テスト用DataFrame
        """
        np.random.seed(42)
        
        data = {
            'smiles': [f'C{"C" * i}O' for i in range(n_rows)],
            'target': np.random.choice(['5HT2A', 'D2', 'CB1'], n_rows),
            'pIC50': np.random.normal(6.0, 1.0, n_rows),
            'MW': np.random.normal(300, 50, n_rows),
            'LogP': np.random.normal(2.5, 1.0, n_rows),
            'TPSA': np.random.normal(50, 20, n_rows)
        }
        
        return pd.DataFrame(data)
    
    def create_test_molecules(self, n_molecules: int = 5) -> List[str]:
        """
        テスト用分子SMILES作成
        
        Args:
            n_molecules: 分子数
            
        Returns:
            テスト用分子SMILESリスト
        """
        test_smiles = [
            'CCO',  # エタノール
            'CCN',  # エチルアミン
            'CC(=O)O',  # 酢酸
            'c1ccccc1',  # ベンゼン
            'CC(C)O'  # イソプロパノール
        ]
        
        return test_smiles[:n_molecules]
    
    def create_test_targets(self) -> List[str]:
        """
        テスト用ターゲット作成
        
        Returns:
            テスト用ターゲットリスト
        """
        return ['5HT2A', 'D2', 'CB1', 'MOR', 'DAT']
    
    def assert_dataframe_equal(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                              tolerance: float = 1e-6):
        """
        DataFrame等価性アサーション
        
        Args:
            df1: DataFrame1
            df2: DataFrame2
            tolerance: 数値許容誤差
        """
        self.assertEqual(df1.shape, df2.shape, "DataFrame shapes don't match")
        
        for col in df1.columns:
            if col not in df2.columns:
                self.fail(f"Column {col} not found in df2")
            
            if df1[col].dtype in ['float64', 'float32']:
                np.testing.assert_allclose(
                    df1[col].values, df2[col].values, 
                    rtol=tolerance, atol=tolerance,
                    err_msg=f"Column {col} values don't match"
                )
            else:
                self.assertTrue(
                    df1[col].equals(df2[col]),
                    f"Column {col} values don't match"
                )
    
    def assert_molecular_features(self, features: Dict[str, Any]):
        """
        分子特徴量アサーション
        
        Args:
            features: 分子特徴量辞書
        """
        required_features = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA']
        
        for feature in required_features:
            self.assertIn(feature, features, f"Required feature {feature} not found")
            self.assertIsInstance(features[feature], (int, float), 
                                f"Feature {feature} should be numeric")
            self.assertGreaterEqual(features[feature], 0, 
                                  f"Feature {feature} should be non-negative")
    
    def assert_admet_predictions(self, predictions: Dict[str, Any]):
        """
        ADMET予測アサーション
        
        Args:
            predictions: ADMET予測辞書
        """
        required_predictions = ['MW', 'LogP', 'TPSA', 'ADMET_Score']
        
        for prediction in required_predictions:
            self.assertIn(prediction, predictions, 
                         f"Required prediction {prediction} not found")
            self.assertIsInstance(predictions[prediction], (int, float), 
                                f"Prediction {prediction} should be numeric")
    
    def assert_model_output(self, output: Dict[str, Any], task_type: str = 'regression'):
        """
        モデル出力アサーション
        
        Args:
            output: モデル出力辞書
            task_type: タスクタイプ（regression/classification）
        """
        if task_type == 'regression':
            self.assertIn('predictions', output, "Regression predictions not found")
            self.assertIsInstance(output['predictions'], (list, np.ndarray), 
                                "Predictions should be list or array")
        elif task_type == 'classification':
            self.assertIn('probabilities', output, "Classification probabilities not found")
            self.assertIsInstance(output['probabilities'], (list, np.ndarray), 
                                "Probabilities should be list or array")
        
        if 'uncertainty' in output:
            self.assertIsInstance(output['uncertainty'], (list, np.ndarray), 
                                "Uncertainty should be list or array")
    
    def create_test_model_config(self) -> Dict[str, Any]:
        """
        テスト用モデル設定作成
        
        Returns:
            テスト用モデル設定辞書
        """
        return {
            'transformer': {
                'd_model': 128,
                'n_layers': 2,
                'n_heads': 4,
                'd_ff': 256,
                'dropout': 0.1
            },
            'gnn': {
                'hidden_dim': 64,
                'n_layers': 2,
                'dropout': 0.1
            },
            'training': {
                'batch_size': 4,
                'epochs': 2,
                'learning_rate': 0.001,
                'weight_decay': 0.01
            }
        }
    
    def create_test_training_data(self, n_samples: int = 20) -> Dict[str, Any]:
        """
        テスト用学習データ作成
        
        Args:
            n_samples: サンプル数
            
        Returns:
            テスト用学習データ辞書
        """
        np.random.seed(42)
        
        # 分子SMILES
        smiles = [f'C{"C" * i}O' for i in range(n_samples)]
        
        # ターゲット
        targets = np.random.choice(['5HT2A', 'D2', 'CB1'], n_samples)
        
        # 回帰ターゲット
        regression_targets = np.random.normal(6.0, 1.0, n_samples)
        
        # 分類ターゲット
        classification_targets = (regression_targets > 6.0).astype(int)
        
        return {
            'smiles': smiles,
            'targets': targets,
            'regression_targets': regression_targets,
            'classification_targets': classification_targets
        }

def create_test_config_file(config_path: str, config: Dict[str, Any]):
    """
    テスト用設定ファイル作成
    
    Args:
        config_path: 設定ファイルパス
        config: 設定辞書
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def create_test_dataframe(n_rows: int = 10) -> pd.DataFrame:
    """
    テスト用DataFrame作成
    
    Args:
        n_rows: 行数
        
    Returns:
        テスト用DataFrame
    """
    np.random.seed(42)
    
    data = {
        'smiles': [f'C{"C" * i}O' for i in range(n_rows)],
        'target': np.random.choice(['5HT2A', 'D2', 'CB1'], n_rows),
        'pIC50': np.random.normal(6.0, 1.0, n_rows),
        'MW': np.random.normal(300, 50, n_rows),
        'LogP': np.random.normal(2.5, 1.0, n_rows),
        'TPSA': np.random.normal(50, 20, n_rows)
    }
    
    return pd.DataFrame(data)

def create_test_molecules(n_molecules: int = 5) -> List[str]:
    """
    テスト用分子SMILES作成
    
    Args:
        n_molecules: 分子数
        
    Returns:
        テスト用分子SMILESリスト
    """
    test_smiles = [
        'CCO',  # エタノール
        'CCN',  # エチルアミン
        'CC(=O)O',  # 酢酸
        'c1ccccc1',  # ベンゼン
        'CC(C)O'  # イソプロパノール
    ]
    
    return test_smiles[:n_molecules]

def create_test_targets() -> List[str]:
    """
    テスト用ターゲット作成
    
    Returns:
        テスト用ターゲットリスト
    """
    return ['5HT2A', 'D2', 'CB1', 'MOR', 'DAT']

def assert_dataframe_equal(df1: pd.DataFrame, df2: pd.DataFrame, 
                          tolerance: float = 1e-6):
    """
    DataFrame等価性アサーション
    
    Args:
        df1: DataFrame1
        df2: DataFrame2
        tolerance: 数値許容誤差
    """
    assert df1.shape == df2.shape, "DataFrame shapes don't match"
    
    for col in df1.columns:
        assert col in df2.columns, f"Column {col} not found in df2"
        
        if df1[col].dtype in ['float64', 'float32']:
            np.testing.assert_allclose(
                df1[col].values, df2[col].values, 
                rtol=tolerance, atol=tolerance,
                err_msg=f"Column {col} values don't match"
            )
        else:
            assert df1[col].equals(df2[col]), f"Column {col} values don't match"

def assert_molecular_features(features: Dict[str, Any]):
    """
    分子特徴量アサーション
    
    Args:
        features: 分子特徴量辞書
    """
    required_features = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA']
    
    for feature in required_features:
        assert feature in features, f"Required feature {feature} not found"
        assert isinstance(features[feature], (int, float)), \
            f"Feature {feature} should be numeric"
        assert features[feature] >= 0, f"Feature {feature} should be non-negative"

def assert_admet_predictions(predictions: Dict[str, Any]):
    """
    ADMET予測アサーション
    
    Args:
        predictions: ADMET予測辞書
    """
    required_predictions = ['MW', 'LogP', 'TPSA', 'ADMET_Score']
    
    for prediction in required_predictions:
        assert prediction in predictions, f"Required prediction {prediction} not found"
        assert isinstance(predictions[prediction], (int, float)), \
            f"Prediction {prediction} should be numeric"

def assert_model_output(output: Dict[str, Any], task_type: str = 'regression'):
    """
    モデル出力アサーション
    
    Args:
        output: モデル出力辞書
        task_type: タスクタイプ（regression/classification）
    """
    if task_type == 'regression':
        assert 'predictions' in output, "Regression predictions not found"
        assert isinstance(output['predictions'], (list, np.ndarray)), \
            "Predictions should be list or array"
    elif task_type == 'classification':
        assert 'probabilities' in output, "Classification probabilities not found"
        assert isinstance(output['probabilities'], (list, np.ndarray)), \
            "Probabilities should be list or array"
    
    if 'uncertainty' in output:
        assert isinstance(output['uncertainty'], (list, np.ndarray)), \
            "Uncertainty should be list or array"

if __name__ == "__main__":
    # テスト実行
    unittest.main()
