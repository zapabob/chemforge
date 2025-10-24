"""
Tests Module

テストモジュール
ユニットテスト・統合テスト・パフォーマンステスト
"""

import unittest
import sys
import os
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# テスト設定
TEST_CONFIG = {
    'test_data_dir': 'tests/data',
    'test_cache_dir': 'tests/cache',
    'test_output_dir': 'tests/output',
    'test_log_level': 'DEBUG'
}

# テスト用データ
TEST_SMILES = [
    'CCO',  # エタノール
    'CCN',  # エチルアミン
    'CC(=O)O',  # 酢酸
    'c1ccccc1',  # ベンゼン
    'CC(C)O'  # イソプロパノール
]

TEST_TARGETS = ['5HT2A', 'D2', 'CB1', 'MOR', 'DAT']

# テスト用設定
TEST_CONFIG_YAML = """
# テスト用設定
test_mode: true
debug: true
log_level: DEBUG

# データベース設定
database:
  type: sqlite
  path: tests/data/test.db

# モデル設定
models:
  transformer:
    d_model: 128
    n_layers: 2
    n_heads: 4
  gnn:
    hidden_dim: 64
    n_layers: 2

# 学習設定
training:
  batch_size: 4
  epochs: 2
  learning_rate: 0.001
"""

__all__ = [
    'TEST_CONFIG',
    'TEST_SMILES',
    'TEST_TARGETS',
    'TEST_CONFIG_YAML'
]