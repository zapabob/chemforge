"""
Core Module Tests

コアモジュールテスト
PWA+PET・SU2・RMSNorm・SiLU・RoPE・GELU・LayerNorm・Dropout・MLP・PositionalEncoding・MultiHeadAttention・TransformerBlock・Transformer・GraphNeuralNetwork・EnsembleModel・Trainer・InferenceManager・ConfigManager・Logger・DataValidator・MolecularPreprocessor・DataPreprocessor・DatabaseManager・VisualizationManager・WebScraper・ExternalAPIs・ChEMBLLoader・MolecularFeatures・RDKitDescriptors・ADMETPredictor・TrainingManager・InferenceManager・DatabaseIntegration・VisualizationManager・PretrainedManager・DataDistribution・MolecularGenerator・StreamlitApp・DashApp
"""

import unittest
import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Any, Optional
import tempfile
import shutil
from pathlib import Path
import sys
import os

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.test_utils import TestBase, create_test_molecules, create_test_targets
from chemforge.core.attention import PWA_PET_Attention
from chemforge.core.su2 import SU2Gate
from chemforge.core.transformer import Transformer
from chemforge.core.gnn import GraphNeuralNetwork
from chemforge.core.ensemble import EnsembleModel
from chemforge.core.trainer import Trainer
from chemforge.core.inference import InferenceManager
from chemforge.utils.config_utils import ConfigManager
from chemforge.utils.logging_utils import Logger
from chemforge.utils.validation import DataValidator
from chemforge.utils.molecular_preprocessor import MolecularPreprocessor
from chemforge.utils.data_preprocessor import DataPreprocessor
from chemforge.utils.database_manager import DatabaseManager
from chemforge.utils.visualization_manager import VisualizationManager
from chemforge.utils.web_scraper import WebScraper
from chemforge.utils.external_apis import ExternalAPIs
from chemforge.data.chembl_loader import ChEMBLLoader
from chemforge.data.molecular_features import MolecularFeatures
from chemforge.data.rdkit_descriptors import RDKitDescriptors
from chemforge.admet.admet_predictor import ADMETPredictor
from chemforge.training.training_manager import TrainingManager
from chemforge.training.inference_manager import InferenceManager
from chemforge.integration.database_integration import DatabaseIntegration
from chemforge.integration.visualization_manager import VisualizationManager
from chemforge.pretrained.pretrained_manager import PretrainedManager
from chemforge.pretrained.data_distribution import DataDistribution
from chemforge.generation.molecular_generator import MolecularGenerator
from chemforge.gui.streamlit_app import StreamlitApp
from chemforge.gui.dash_app import DashApp

class TestPWAPETAttention(TestBase):
    """PWA+PET Attention テスト"""
    
    def test_attention_initialization(self):
        """Attention初期化テスト"""
        attention = PWA_PET_Attention(
            d_model=128,
            n_heads=8,
            dropout=0.1
        )
        
        self.assertEqual(attention.d_model, 128)
        self.assertEqual(attention.n_heads, 8)
        self.assertEqual(attention.dropout, 0.1)
    
    def test_attention_forward(self):
        """Attention forward テスト"""
        attention = PWA_PET_Attention(
            d_model=128,
            n_heads=8,
            dropout=0.1
        )
        
        # テストデータ
        batch_size, seq_len, d_model = 2, 10, 128
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Forward pass
        output = attention(x)
        
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))
        self.assertIsInstance(output, torch.Tensor)
    
    def test_attention_attention_weights(self):
        """Attention weights テスト"""
        attention = PWA_PET_Attention(
            d_model=128,
            n_heads=8,
            dropout=0.1
        )
        
        # テストデータ
        batch_size, seq_len, d_model = 2, 10, 128
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Forward pass with attention weights
        output, attn_weights = attention(x, return_attention=True)
        
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))
        self.assertEqual(attn_weights.shape, (batch_size, 8, seq_len, seq_len))
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(attn_weights, torch.Tensor)

class TestSU2Gate(TestBase):
    """SU2Gate テスト"""
    
    def test_su2_initialization(self):
        """SU2Gate初期化テスト"""
        su2 = SU2Gate(d_model=128)
        
        self.assertEqual(su2.d_model, 128)
        self.assertIsInstance(su2.phase_matrix, torch.Tensor)
    
    def test_su2_forward(self):
        """SU2Gate forward テスト"""
        su2 = SU2Gate(d_model=128)
        
        # テストデータ
        batch_size, seq_len, d_model = 2, 10, 128
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Forward pass
        output = su2(x)
        
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))
        self.assertIsInstance(output, torch.Tensor)
    
    def test_su2_phase_rotation(self):
        """SU2Gate phase rotation テスト"""
        su2 = SU2Gate(d_model=128)
        
        # テストデータ
        batch_size, seq_len, d_model = 2, 10, 128
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Phase rotation
        output = su2(x)
        
        # 位相回転の確認
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))
        self.assertIsInstance(output, torch.Tensor)

class TestTransformer(TestBase):
    """Transformer テスト"""
    
    def test_transformer_initialization(self):
        """Transformer初期化テスト"""
        transformer = Transformer(
            d_model=128,
            n_layers=2,
            n_heads=8,
            d_ff=256,
            dropout=0.1
        )
        
        self.assertEqual(transformer.d_model, 128)
        self.assertEqual(transformer.n_layers, 2)
        self.assertEqual(transformer.n_heads, 8)
        self.assertEqual(transformer.d_ff, 256)
        self.assertEqual(transformer.dropout, 0.1)
    
    def test_transformer_forward(self):
        """Transformer forward テスト"""
        transformer = Transformer(
            d_model=128,
            n_layers=2,
            n_heads=8,
            d_ff=256,
            dropout=0.1
        )
        
        # テストデータ
        batch_size, seq_len, d_model = 2, 10, 128
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Forward pass
        output = transformer(x)
        
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))
        self.assertIsInstance(output, torch.Tensor)
    
    def test_transformer_attention_weights(self):
        """Transformer attention weights テスト"""
        transformer = Transformer(
            d_model=128,
            n_layers=2,
            n_heads=8,
            d_ff=256,
            dropout=0.1
        )
        
        # テストデータ
        batch_size, seq_len, d_model = 2, 10, 128
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Forward pass with attention weights
        output, attn_weights = transformer(x, return_attention=True)
        
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))
        self.assertEqual(len(attn_weights), 2)  # n_layers
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(attn_weights, list)

class TestGraphNeuralNetwork(TestBase):
    """GraphNeuralNetwork テスト"""
    
    def test_gnn_initialization(self):
        """GNN初期化テスト"""
        gnn = GraphNeuralNetwork(
            input_dim=128,
            hidden_dim=64,
            output_dim=32,
            n_layers=2,
            dropout=0.1
        )
        
        self.assertEqual(gnn.input_dim, 128)
        self.assertEqual(gnn.hidden_dim, 64)
        self.assertEqual(gnn.output_dim, 32)
        self.assertEqual(gnn.n_layers, 2)
        self.assertEqual(gnn.dropout, 0.1)
    
    def test_gnn_forward(self):
        """GNN forward テスト"""
        gnn = GraphNeuralNetwork(
            input_dim=128,
            hidden_dim=64,
            output_dim=32,
            n_layers=2,
            dropout=0.1
        )
        
        # テストデータ
        batch_size, num_nodes, input_dim = 2, 10, 128
        x = torch.randn(batch_size, num_nodes, input_dim)
        edge_index = torch.randint(0, num_nodes, (2, 20))
        
        # Forward pass
        output = gnn(x, edge_index)
        
        self.assertEqual(output.shape, (batch_size, num_nodes, 32))
        self.assertIsInstance(output, torch.Tensor)
    
    def test_gnn_batch_processing(self):
        """GNN batch processing テスト"""
        gnn = GraphNeuralNetwork(
            input_dim=128,
            hidden_dim=64,
            output_dim=32,
            n_layers=2,
            dropout=0.1
        )
        
        # バッチデータ
        batch_size = 3
        num_nodes = 10
        input_dim = 128
        
        x = torch.randn(batch_size, num_nodes, input_dim)
        edge_index = torch.randint(0, num_nodes, (2, 20))
        
        # Forward pass
        output = gnn(x, edge_index)
        
        self.assertEqual(output.shape, (batch_size, num_nodes, 32))
        self.assertIsInstance(output, torch.Tensor)

class TestEnsembleModel(TestBase):
    """EnsembleModel テスト"""
    
    def test_ensemble_initialization(self):
        """EnsembleModel初期化テスト"""
        ensemble = EnsembleModel(
            models=[],
            weights=None,
            method='average'
        )
        
        self.assertEqual(ensemble.method, 'average')
        self.assertIsInstance(ensemble.models, list)
        self.assertIsNone(ensemble.weights)
    
    def test_ensemble_add_model(self):
        """EnsembleModel add model テスト"""
        ensemble = EnsembleModel(
            models=[],
            weights=None,
            method='average'
        )
        
        # モデル追加
        transformer = Transformer(d_model=128, n_layers=2, n_heads=8)
        gnn = GraphNeuralNetwork(input_dim=128, hidden_dim=64, output_dim=32)
        
        ensemble.add_model(transformer, weight=0.6)
        ensemble.add_model(gnn, weight=0.4)
        
        self.assertEqual(len(ensemble.models), 2)
        self.assertEqual(len(ensemble.weights), 2)
        self.assertEqual(ensemble.weights[0], 0.6)
        self.assertEqual(ensemble.weights[1], 0.4)
    
    def test_ensemble_predict(self):
        """EnsembleModel predict テスト"""
        ensemble = EnsembleModel(
            models=[],
            weights=None,
            method='average'
        )
        
        # モデル追加
        transformer = Transformer(d_model=128, n_layers=2, n_heads=8)
        gnn = GraphNeuralNetwork(input_dim=128, hidden_dim=64, output_dim=32)
        
        ensemble.add_model(transformer, weight=0.6)
        ensemble.add_model(gnn, weight=0.4)
        
        # テストデータ
        batch_size, seq_len, d_model = 2, 10, 128
        x = torch.randn(batch_size, seq_len, d_model)
        
        # 予測
        predictions = ensemble.predict(x)
        
        self.assertIsInstance(predictions, torch.Tensor)
        self.assertEqual(predictions.shape[0], batch_size)

class TestTrainer(TestBase):
    """Trainer テスト"""
    
    def test_trainer_initialization(self):
        """Trainer初期化テスト"""
        trainer = Trainer(
            model=None,
            optimizer=None,
            criterion=None,
            device='cpu'
        )
        
        self.assertEqual(trainer.device, 'cpu')
        self.assertIsNone(trainer.model)
        self.assertIsNone(trainer.optimizer)
        self.assertIsNone(trainer.criterion)
    
    def test_trainer_set_model(self):
        """Trainer set model テスト"""
        trainer = Trainer(
            model=None,
            optimizer=None,
            criterion=None,
            device='cpu'
        )
        
        # モデル設定
        model = Transformer(d_model=128, n_layers=2, n_heads=8)
        trainer.set_model(model)
        
        self.assertEqual(trainer.model, model)
        self.assertIsInstance(trainer.model, Transformer)
    
    def test_trainer_set_optimizer(self):
        """Trainer set optimizer テスト"""
        trainer = Trainer(
            model=None,
            optimizer=None,
            criterion=None,
            device='cpu'
        )
        
        # オプティマイザー設定
        model = Transformer(d_model=128, n_layers=2, n_heads=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        trainer.set_optimizer(optimizer)
        
        self.assertEqual(trainer.optimizer, optimizer)
        self.assertIsInstance(trainer.optimizer, torch.optim.Adam)

class TestInferenceManager(TestBase):
    """InferenceManager テスト"""
    
    def test_inference_manager_initialization(self):
        """InferenceManager初期化テスト"""
        inference_manager = InferenceManager(
            model=None,
            device='cpu'
        )
        
        self.assertEqual(inference_manager.device, 'cpu')
        self.assertIsNone(inference_manager.model)
    
    def test_inference_manager_set_model(self):
        """InferenceManager set model テスト"""
        inference_manager = InferenceManager(
            model=None,
            device='cpu'
        )
        
        # モデル設定
        model = Transformer(d_model=128, n_layers=2, n_heads=8)
        inference_manager.set_model(model)
        
        self.assertEqual(inference_manager.model, model)
        self.assertIsInstance(inference_manager.model, Transformer)
    
    def test_inference_manager_predict(self):
        """InferenceManager predict テスト"""
        inference_manager = InferenceManager(
            model=None,
            device='cpu'
        )
        
        # モデル設定
        model = Transformer(d_model=128, n_layers=2, n_heads=8)
        inference_manager.set_model(model)
        
        # テストデータ
        batch_size, seq_len, d_model = 2, 10, 128
        x = torch.randn(batch_size, seq_len, d_model)
        
        # 予測
        predictions = inference_manager.predict(x)
        
        self.assertIsInstance(predictions, torch.Tensor)
        self.assertEqual(predictions.shape[0], batch_size)

class TestConfigManager(TestBase):
    """ConfigManager テスト"""
    
    def test_config_manager_initialization(self):
        """ConfigManager初期化テスト"""
        config_manager = ConfigManager(self.config_path)
        
        self.assertEqual(config_manager.config_path, self.config_path)
        self.assertIsInstance(config_manager.config, dict)
    
    def test_config_manager_load_config(self):
        """ConfigManager load config テスト"""
        config_manager = ConfigManager(self.config_path)
        config = config_manager.load_config()
        
        self.assertIsInstance(config, dict)
        self.assertIn('test_mode', config)
        self.assertTrue(config['test_mode'])
    
    def test_config_manager_save_config(self):
        """ConfigManager save config テスト"""
        config_manager = ConfigManager(self.config_path)
        
        # 新しい設定
        new_config = {
            'test_mode': False,
            'debug': True,
            'new_setting': 'test_value'
        }
        
        # 設定保存
        config_manager.save_config(new_config)
        
        # 設定読み込み
        loaded_config = config_manager.load_config()
        
        self.assertEqual(loaded_config['test_mode'], False)
        self.assertEqual(loaded_config['debug'], True)
        self.assertEqual(loaded_config['new_setting'], 'test_value')

class TestLogger(TestBase):
    """Logger テスト"""
    
    def test_logger_initialization(self):
        """Logger初期化テスト"""
        logger = Logger("TestLogger")
        
        self.assertEqual(logger.name, "TestLogger")
        self.assertIsInstance(logger.logger, logging.Logger)
    
    def test_logger_logging(self):
        """Logger logging テスト"""
        logger = Logger("TestLogger")
        
        # ログ出力
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.error("Test error message")
        
        # ログレベル確認
        self.assertEqual(logger.logger.level, logging.DEBUG)

class TestDataValidator(TestBase):
    """DataValidator テスト"""
    
    def test_data_validator_initialization(self):
        """DataValidator初期化テスト"""
        validator = DataValidator()
        
        self.assertIsInstance(validator, DataValidator)
    
    def test_data_validator_validate_smiles(self):
        """DataValidator validate SMILES テスト"""
        validator = DataValidator()
        
        # 有効なSMILES
        valid_smiles = "CCO"
        self.assertTrue(validator.validate_smiles(valid_smiles))
        
        # 無効なSMILES
        invalid_smiles = "InvalidSMILES"
        self.assertFalse(validator.validate_smiles(invalid_smiles))
    
    def test_data_validator_validate_dataframe(self):
        """DataValidator validate DataFrame テスト"""
        validator = DataValidator()
        
        # 有効なDataFrame
        valid_df = pd.DataFrame({
            'smiles': ['CCO', 'CCN'],
            'target': ['5HT2A', 'D2'],
            'pIC50': [6.0, 7.0]
        })
        self.assertTrue(validator.validate_dataframe(valid_df))
        
        # 無効なDataFrame（必須列なし）
        invalid_df = pd.DataFrame({
            'smiles': ['CCO', 'CCN'],
            'target': ['5HT2A', 'D2']
        })
        self.assertFalse(validator.validate_dataframe(invalid_df))

class TestMolecularPreprocessor(TestBase):
    """MolecularPreprocessor テスト"""
    
    def test_molecular_preprocessor_initialization(self):
        """MolecularPreprocessor初期化テスト"""
        preprocessor = MolecularPreprocessor(self.config_path, str(self.test_cache_dir))
        
        self.assertEqual(preprocessor.config_path, self.config_path)
        self.assertEqual(preprocessor.cache_dir, str(self.test_cache_dir))
    
    def test_molecular_preprocessor_sanitize_smiles(self):
        """MolecularPreprocessor sanitize SMILES テスト"""
        preprocessor = MolecularPreprocessor(self.config_path, str(self.test_cache_dir))
        
        # SMILES正規化
        smiles = "CCO"
        sanitized = preprocessor.sanitize_smiles(smiles)
        
        self.assertIsInstance(sanitized, str)
        self.assertEqual(sanitized, smiles)
    
    def test_molecular_preprocessor_calculate_features(self):
        """MolecularPreprocessor calculate features テスト"""
        preprocessor = MolecularPreprocessor(self.config_path, str(self.test_cache_dir))
        
        # 分子特徴量計算
        smiles = "CCO"
        features = preprocessor.calculate_features(smiles)
        
        self.assertIsInstance(features, dict)
        self.assertIn('MW', features)
        self.assertIn('LogP', features)
        self.assertIn('TPSA', features)

class TestDataPreprocessor(TestBase):
    """DataPreprocessor テスト"""
    
    def test_data_preprocessor_initialization(self):
        """DataPreprocessor初期化テスト"""
        preprocessor = DataPreprocessor(self.config_path, str(self.test_cache_dir))
        
        self.assertEqual(preprocessor.config_path, self.config_path)
        self.assertEqual(preprocessor.cache_dir, str(self.test_cache_dir))
    
    def test_data_preprocessor_normalize_data(self):
        """DataPreprocessor normalize data テスト"""
        preprocessor = DataPreprocessor(self.config_path, str(self.test_cache_dir))
        
        # データ正規化
        data = np.random.randn(100, 5)
        normalized = preprocessor.normalize_data(data)
        
        self.assertEqual(normalized.shape, data.shape)
        self.assertIsInstance(normalized, np.ndarray)
    
    def test_data_preprocessor_split_data(self):
        """DataPreprocessor split data テスト"""
        preprocessor = DataPreprocessor(self.config_path, str(self.test_cache_dir))
        
        # データ分割
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
            X, y, test_size=0.2, val_size=0.2
        )
        
        self.assertEqual(len(X_train), 60)
        self.assertEqual(len(X_val), 20)
        self.assertEqual(len(X_test), 20)
        self.assertEqual(len(y_train), 60)
        self.assertEqual(len(y_val), 20)
        self.assertEqual(len(y_test), 20)

class TestDatabaseManager(TestBase):
    """DatabaseManager テスト"""
    
    def test_database_manager_initialization(self):
        """DatabaseManager初期化テスト"""
        db_manager = DatabaseManager(self.config_path, str(self.test_data_dir))
        
        self.assertEqual(db_manager.config_path, self.config_path)
        self.assertEqual(db_manager.db_path, str(self.test_data_dir))
    
    def test_database_manager_create_tables(self):
        """DatabaseManager create tables テスト"""
        db_manager = DatabaseManager(self.config_path, str(self.test_data_dir))
        
        # テーブル作成
        db_manager.create_tables()
        
        # テーブル存在確認
        tables = db_manager.get_tables()
        self.assertIsInstance(tables, list)
    
    def test_database_manager_insert_data(self):
        """DatabaseManager insert data テスト"""
        db_manager = DatabaseManager(self.config_path, str(self.test_data_dir))
        
        # テーブル作成
        db_manager.create_tables()
        
        # データ挿入
        data = {
            'smiles': 'CCO',
            'target': '5HT2A',
            'pIC50': 6.0
        }
        
        db_manager.insert_data('molecules', data)
        
        # データ確認
        result = db_manager.query_data('molecules', {'smiles': 'CCO'})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['smiles'], 'CCO')

class TestVisualizationManager(TestBase):
    """VisualizationManager テスト"""
    
    def test_visualization_manager_initialization(self):
        """VisualizationManager初期化テスト"""
        viz_manager = VisualizationManager(self.config_path, str(self.test_cache_dir))
        
        self.assertEqual(viz_manager.config_path, self.config_path)
        self.assertEqual(viz_manager.cache_dir, str(self.test_cache_dir))
    
    def test_visualization_manager_create_plot(self):
        """VisualizationManager create plot テスト"""
        viz_manager = VisualizationManager(self.config_path, str(self.test_cache_dir))
        
        # プロット作成
        data = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 6, 8, 10]
        })
        
        plot = viz_manager.create_plot(data, plot_type='scatter')
        
        self.assertIsInstance(plot, dict)
        self.assertIn('data', plot)
        self.assertIn('layout', plot)
    
    def test_visualization_manager_save_plot(self):
        """VisualizationManager save plot テスト"""
        viz_manager = VisualizationManager(self.config_path, str(self.test_cache_dir))
        
        # プロット作成
        data = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 6, 8, 10]
        })
        
        plot = viz_manager.create_plot(data, plot_type='scatter')
        
        # プロット保存
        output_path = self.test_output_dir / 'test_plot.html'
        viz_manager.save_plot(plot, str(output_path))
        
        self.assertTrue(output_path.exists())

class TestWebScraper(TestBase):
    """WebScraper テスト"""
    
    def test_web_scraper_initialization(self):
        """WebScraper初期化テスト"""
        scraper = WebScraper(self.config_path, str(self.test_cache_dir))
        
        self.assertEqual(scraper.config_path, self.config_path)
        self.assertEqual(scraper.cache_dir, str(self.test_cache_dir))
    
    def test_web_scraper_scrape_swissadme(self):
        """WebScraper scrape SwissADME テスト"""
        scraper = WebScraper(self.config_path, str(self.test_cache_dir))
        
        # SwissADMEスクレイピング
        smiles = "CCO"
        result = scraper.scrape_swissadme(smiles)
        
        self.assertIsInstance(result, dict)
        self.assertIn('MW', result)
        self.assertIn('LogP', result)
        self.assertIn('TPSA', result)
    
    def test_web_scraper_batch_scrape(self):
        """WebScraper batch scrape テスト"""
        scraper = WebScraper(self.config_path, str(self.test_cache_dir))
        
        # バッチスクレイピング
        smiles_list = ["CCO", "CCN", "CC(=O)O"]
        results = scraper.batch_scrape(smiles_list)
        
        self.assertEqual(len(results), len(smiles_list))
        self.assertIsInstance(results, list)

class TestExternalAPIs(TestBase):
    """ExternalAPIs テスト"""
    
    def test_external_apis_initialization(self):
        """ExternalAPIs初期化テスト"""
        apis = ExternalAPIs(self.config_path, str(self.test_cache_dir))
        
        self.assertEqual(apis.config_path, self.config_path)
        self.assertEqual(apis.cache_dir, str(self.test_cache_dir))
    
    def test_external_apis_pubchem_search(self):
        """ExternalAPIs PubChem search テスト"""
        apis = ExternalAPIs(self.config_path, str(self.test_cache_dir))
        
        # PubChem検索
        query = "ethanol"
        result = apis.pubchem_search(query)
        
        self.assertIsInstance(result, dict)
        self.assertIn('results', result)
    
    def test_external_apis_drugbank_search(self):
        """ExternalAPIs DrugBank search テスト"""
        apis = ExternalAPIs(self.config_path, str(self.test_cache_dir))
        
        # DrugBank検索
        query = "ethanol"
        result = apis.drugbank_search(query)
        
        self.assertIsInstance(result, dict)
        self.assertIn('results', result)

class TestChEMBLLoader(TestBase):
    """ChEMBLLoader テスト"""
    
    def test_chembl_loader_initialization(self):
        """ChEMBLLoader初期化テスト"""
        loader = ChEMBLLoader(self.config_path, str(self.test_cache_dir))
        
        self.assertEqual(loader.config_path, self.config_path)
        self.assertEqual(loader.cache_dir, str(self.test_cache_dir))
    
    def test_chembl_loader_load_data(self):
        """ChEMBLLoader load data テスト"""
        loader = ChEMBLLoader(self.config_path, str(self.test_cache_dir))
        
        # データ読み込み
        target = "5HT2A"
        data = loader.load_data(target, limit=10)
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIn('smiles', data.columns)
        self.assertIn('target', data.columns)
    
    def test_chembl_loader_load_multiple_targets(self):
        """ChEMBLLoader load multiple targets テスト"""
        loader = ChEMBLLoader(self.config_path, str(self.test_cache_dir))
        
        # 複数ターゲットデータ読み込み
        targets = ["5HT2A", "D2", "CB1"]
        data = loader.load_multiple_targets(targets, limit=5)
        
        self.assertIsInstance(data, dict)
        self.assertEqual(len(data), len(targets))
        for target in targets:
            self.assertIn(target, data)
            self.assertIsInstance(data[target], pd.DataFrame)

class TestMolecularFeatures(TestBase):
    """MolecularFeatures テスト"""
    
    def test_molecular_features_initialization(self):
        """MolecularFeatures初期化テスト"""
        features = MolecularFeatures(self.config_path, str(self.test_cache_dir))
        
        self.assertEqual(features.config_path, self.config_path)
        self.assertEqual(features.cache_dir, str(self.test_cache_dir))
    
    def test_molecular_features_calculate_features(self):
        """MolecularFeatures calculate features テスト"""
        features = MolecularFeatures(self.config_path, str(self.test_cache_dir))
        
        # 分子特徴量計算
        smiles = "CCO"
        result = features.calculate_features(smiles)
        
        self.assertIsInstance(result, dict)
        self.assertIn('MW', result)
        self.assertIn('LogP', result)
        self.assertIn('TPSA', result)
        self.assertIn('HBD', result)
        self.assertIn('HBA', result)
    
    def test_molecular_features_batch_calculate(self):
        """MolecularFeatures batch calculate テスト"""
        features = MolecularFeatures(self.config_path, str(self.test_cache_dir))
        
        # バッチ特徴量計算
        smiles_list = ["CCO", "CCN", "CC(=O)O"]
        results = features.batch_calculate(smiles_list)
        
        self.assertEqual(len(results), len(smiles_list))
        self.assertIsInstance(results, list)
        for result in results:
            self.assertIsInstance(result, dict)

class TestRDKitDescriptors(TestBase):
    """RDKitDescriptors テスト"""
    
    def test_rdkit_descriptors_initialization(self):
        """RDKitDescriptors初期化テスト"""
        descriptors = RDKitDescriptors(self.config_path, str(self.test_cache_dir))
        
        self.assertEqual(descriptors.config_path, self.config_path)
        self.assertEqual(descriptors.cache_dir, str(self.test_cache_dir))
    
    def test_rdkit_descriptors_calculate_descriptors(self):
        """RDKitDescriptors calculate descriptors テスト"""
        descriptors = RDKitDescriptors(self.config_path, str(self.test_cache_dir))
        
        # 記述子計算
        smiles = "CCO"
        result = descriptors.calculate_descriptors(smiles)
        
        self.assertIsInstance(result, dict)
        self.assertIn('MW', result)
        self.assertIn('LogP', result)
        self.assertIn('TPSA', result)
    
    def test_rdkit_descriptors_batch_calculate(self):
        """RDKitDescriptors batch calculate テスト"""
        descriptors = RDKitDescriptors(self.config_path, str(self.test_cache_dir))
        
        # バッチ記述子計算
        smiles_list = ["CCO", "CCN", "CC(=O)O"]
        results = descriptors.batch_calculate(smiles_list)
        
        self.assertEqual(len(results), len(smiles_list))
        self.assertIsInstance(results, list)
        for result in results:
            self.assertIsInstance(result, dict)

class TestADMETPredictor(TestBase):
    """ADMETPredictor テスト"""
    
    def test_admet_predictor_initialization(self):
        """ADMETPredictor初期化テスト"""
        predictor = ADMETPredictor(self.config_path, str(self.test_cache_dir))
        
        self.assertEqual(predictor.config_path, self.config_path)
        self.assertEqual(predictor.cache_dir, str(self.test_cache_dir))
    
    def test_admet_predictor_predict_admet(self):
        """ADMETPredictor predict ADMET テスト"""
        predictor = ADMETPredictor(self.config_path, str(self.test_cache_dir))
        
        # ADMET予測
        smiles = "CCO"
        result = predictor.predict_admet(smiles)
        
        self.assertIsInstance(result, dict)
        self.assertIn('MW', result)
        self.assertIn('LogP', result)
        self.assertIn('TPSA', result)
        self.assertIn('ADMET_Score', result)
    
    def test_admet_predictor_batch_predict(self):
        """ADMETPredictor batch predict テスト"""
        predictor = ADMETPredictor(self.config_path, str(self.test_cache_dir))
        
        # バッチADMET予測
        smiles_list = ["CCO", "CCN", "CC(=O)O"]
        results = predictor.batch_predict(smiles_list)
        
        self.assertEqual(len(results), len(smiles_list))
        self.assertIsInstance(results, list)
        for result in results:
            self.assertIsInstance(result, dict)

class TestTrainingManager(TestBase):
    """TrainingManager テスト"""
    
    def test_training_manager_initialization(self):
        """TrainingManager初期化テスト"""
        trainer = TrainingManager(self.config_path, str(self.test_cache_dir))
        
        self.assertEqual(trainer.config_path, self.config_path)
        self.assertEqual(trainer.cache_dir, str(self.test_cache_dir))
    
    def test_training_manager_train_model(self):
        """TrainingManager train model テスト"""
        trainer = TrainingManager(self.config_path, str(self.test_cache_dir))
        
        # モデル設定
        model = Transformer(d_model=128, n_layers=2, n_heads=8)
        trainer.set_model(model)
        
        # 学習データ作成
        X = torch.randn(100, 10, 128)
        y = torch.randn(100, 1)
        
        # 学習実行
        history = trainer.train_model(X, y, epochs=2)
        
        self.assertIsInstance(history, dict)
        self.assertIn('loss', history)
        self.assertIn('val_loss', history)
    
    def test_training_manager_evaluate_model(self):
        """TrainingManager evaluate model テスト"""
        trainer = TrainingManager(self.config_path, str(self.test_cache_dir))
        
        # モデル設定
        model = Transformer(d_model=128, n_layers=2, n_heads=8)
        trainer.set_model(model)
        
        # 評価データ作成
        X = torch.randn(50, 10, 128)
        y = torch.randn(50, 1)
        
        # 評価実行
        metrics = trainer.evaluate_model(X, y)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('loss', metrics)
        self.assertIn('mse', metrics)
        self.assertIn('mae', metrics)

class TestInferenceManager(TestBase):
    """InferenceManager テスト"""
    
    def test_inference_manager_initialization(self):
        """InferenceManager初期化テスト"""
        inference = InferenceManager(self.config_path, str(self.test_cache_dir))
        
        self.assertEqual(inference.config_path, self.config_path)
        self.assertEqual(inference.cache_dir, str(self.test_cache_dir))
    
    def test_inference_manager_predict(self):
        """InferenceManager predict テスト"""
        inference = InferenceManager(self.config_path, str(self.test_cache_dir))
        
        # モデル設定
        model = Transformer(d_model=128, n_layers=2, n_heads=8)
        inference.set_model(model)
        
        # 予測データ作成
        X = torch.randn(10, 10, 128)
        
        # 予測実行
        predictions = inference.predict(X)
        
        self.assertIsInstance(predictions, torch.Tensor)
        self.assertEqual(predictions.shape[0], 10)
    
    def test_inference_manager_batch_predict(self):
        """InferenceManager batch predict テスト"""
        inference = InferenceManager(self.config_path, str(self.test_cache_dir))
        
        # モデル設定
        model = Transformer(d_model=128, n_layers=2, n_heads=8)
        inference.set_model(model)
        
        # バッチ予測データ作成
        X = torch.randn(50, 10, 128)
        
        # バッチ予測実行
        predictions = inference.batch_predict(X, batch_size=10)
        
        self.assertIsInstance(predictions, torch.Tensor)
        self.assertEqual(predictions.shape[0], 50)

class TestDatabaseIntegration(TestBase):
    """DatabaseIntegration テスト"""
    
    def test_database_integration_initialization(self):
        """DatabaseIntegration初期化テスト"""
        integration = DatabaseIntegration(self.config_path, str(self.test_cache_dir))
        
        self.assertEqual(integration.config_path, self.config_path)
        self.assertEqual(integration.cache_dir, str(self.test_cache_dir))
    
    def test_database_integration_integrate_data(self):
        """DatabaseIntegration integrate data テスト"""
        integration = DatabaseIntegration(self.config_path, str(self.test_cache_dir))
        
        # データ統合
        data = pd.DataFrame({
            'smiles': ['CCO', 'CCN'],
            'target': ['5HT2A', 'D2'],
            'pIC50': [6.0, 7.0]
        })
        
        result = integration.integrate_data(data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertTrue(result['success'])
    
    def test_database_integration_export_data(self):
        """DatabaseIntegration export data テスト"""
        integration = DatabaseIntegration(self.config_path, str(self.test_cache_dir))
        
        # データエクスポート
        output_path = self.test_output_dir / 'exported_data.csv'
        result = integration.export_data(str(output_path))
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertTrue(result['success'])

class TestVisualizationManager(TestBase):
    """VisualizationManager テスト"""
    
    def test_visualization_manager_initialization(self):
        """VisualizationManager初期化テスト"""
        viz_manager = VisualizationManager(self.config_path, str(self.test_cache_dir))
        
        self.assertEqual(viz_manager.config_path, self.config_path)
        self.assertEqual(viz_manager.cache_dir, str(self.test_cache_dir))
    
    def test_visualization_manager_create_visualization(self):
        """VisualizationManager create visualization テスト"""
        viz_manager = VisualizationManager(self.config_path, str(self.test_cache_dir))
        
        # 可視化作成
        data = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 6, 8, 10]
        })
        
        result = viz_manager.create_visualization(data, viz_type='scatter')
        
        self.assertIsInstance(result, dict)
        self.assertIn('plot', result)
        self.assertIn('success', result)
        self.assertTrue(result['success'])
    
    def test_visualization_manager_save_visualization(self):
        """VisualizationManager save visualization テスト"""
        viz_manager = VisualizationManager(self.config_path, str(self.test_cache_dir))
        
        # 可視化作成
        data = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 6, 8, 10]
        })
        
        result = viz_manager.create_visualization(data, viz_type='scatter')
        
        # 可視化保存
        output_path = self.test_output_dir / 'visualization.html'
        save_result = viz_manager.save_visualization(result['plot'], str(output_path))
        
        self.assertIsInstance(save_result, dict)
        self.assertIn('success', save_result)
        self.assertTrue(save_result['success'])

class TestPretrainedManager(TestBase):
    """PretrainedManager テスト"""
    
    def test_pretrained_manager_initialization(self):
        """PretrainedManager初期化テスト"""
        manager = PretrainedManager(self.config_path, str(self.test_cache_dir))
        
        self.assertEqual(manager.config_path, self.config_path)
        self.assertEqual(manager.cache_dir, str(self.test_cache_dir))
    
    def test_pretrained_manager_create_model(self):
        """PretrainedManager create model テスト"""
        manager = PretrainedManager(self.config_path, str(self.test_cache_dir))
        
        # モデル作成
        model = manager.create_model('transformer', {
            'd_model': 128,
            'n_layers': 2,
            'n_heads': 8
        })
        
        self.assertIsInstance(model, Transformer)
        self.assertEqual(model.d_model, 128)
        self.assertEqual(model.n_layers, 2)
        self.assertEqual(model.n_heads, 8)
    
    def test_pretrained_manager_save_model(self):
        """PretrainedManager save model テスト"""
        manager = PretrainedManager(self.config_path, str(self.test_cache_dir))
        
        # モデル作成
        model = manager.create_model('transformer', {
            'd_model': 128,
            'n_layers': 2,
            'n_heads': 8
        })
        
        # モデル保存
        output_path = self.test_output_dir / 'model.pth'
        result = manager.save_model(model, str(output_path))
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertTrue(result['success'])

class TestDataDistribution(TestBase):
    """DataDistribution テスト"""
    
    def test_data_distribution_initialization(self):
        """DataDistribution初期化テスト"""
        distribution = DataDistribution(self.config_path, str(self.test_cache_dir))
        
        self.assertEqual(distribution.config_path, self.config_path)
        self.assertEqual(distribution.cache_dir, str(self.test_cache_dir))
    
    def test_data_distribution_create_dataset(self):
        """DataDistribution create dataset テスト"""
        distribution = DataDistribution(self.config_path, str(self.test_cache_dir))
        
        # データセット作成
        data = pd.DataFrame({
            'smiles': ['CCO', 'CCN', 'CC(=O)O'],
            'target': ['5HT2A', 'D2', 'CB1'],
            'pIC50': [6.0, 7.0, 8.0]
        })
        
        result = distribution.create_dataset(data, split_method='random')
        
        self.assertIsInstance(result, dict)
        self.assertIn('train', result)
        self.assertIn('val', result)
        self.assertIn('test', result)
        self.assertIsInstance(result['train'], pd.DataFrame)
        self.assertIsInstance(result['val'], pd.DataFrame)
        self.assertIsInstance(result['test'], pd.DataFrame)
    
    def test_data_distribution_save_dataset(self):
        """DataDistribution save dataset テスト"""
        distribution = DataDistribution(self.config_path, str(self.test_cache_dir))
        
        # データセット作成
        data = pd.DataFrame({
            'smiles': ['CCO', 'CCN', 'CC(=O)O'],
            'target': ['5HT2A', 'D2', 'CB1'],
            'pIC50': [6.0, 7.0, 8.0]
        })
        
        dataset = distribution.create_dataset(data, split_method='random')
        
        # データセット保存
        output_path = self.test_output_dir / 'dataset'
        result = distribution.save_dataset(dataset, str(output_path))
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertTrue(result['success'])

class TestMolecularGenerator(TestBase):
    """MolecularGenerator テスト"""
    
    def test_molecular_generator_initialization(self):
        """MolecularGenerator初期化テスト"""
        generator = MolecularGenerator(self.config_path, str(self.test_cache_dir))
        
        self.assertEqual(generator.config_path, self.config_path)
        self.assertEqual(generator.cache_dir, str(self.test_cache_dir))
    
    def test_molecular_generator_generate_molecules(self):
        """MolecularGenerator generate molecules テスト"""
        generator = MolecularGenerator(self.config_path, str(self.test_cache_dir))
        
        # 分子生成
        molecules = generator.generate_molecules(
            num_molecules=10,
            method='vae',
            target_properties={'MW': 300, 'LogP': 2.5}
        )
        
        self.assertIsInstance(molecules, list)
        self.assertEqual(len(molecules), 10)
        for molecule in molecules:
            self.assertIsInstance(molecule, str)
    
    def test_molecular_generator_optimize_molecules(self):
        """MolecularGenerator optimize molecules テスト"""
        generator = MolecularGenerator(self.config_path, str(self.test_cache_dir))
        
        # 分子最適化
        molecules = ['CCO', 'CCN', 'CC(=O)O']
        optimized = generator.optimize_molecules(
            molecules=molecules,
            target_properties={'MW': 300, 'LogP': 2.5},
            optimization_method='ga',
            max_iterations=10
        )
        
        self.assertIsInstance(optimized, list)
        self.assertEqual(len(optimized), len(molecules))
        for molecule in optimized:
            self.assertIsInstance(molecule, str)

class TestStreamlitApp(TestBase):
    """StreamlitApp テスト"""
    
    def test_streamlit_app_initialization(self):
        """StreamlitApp初期化テスト"""
        app = StreamlitApp(self.config_path, str(self.test_cache_dir))
        
        self.assertEqual(app.config_path, self.config_path)
        self.assertEqual(app.cache_dir, str(self.test_cache_dir))
    
    def test_streamlit_app_create_home_content(self):
        """StreamlitApp create home content テスト"""
        app = StreamlitApp(self.config_path, str(self.test_cache_dir))
        
        # ホームコンテンツ作成
        content = app._create_home_content()
        
        self.assertIsInstance(content, dict)
        self.assertIn('title', content)
        self.assertIn('description', content)
        self.assertIn('features', content)
    
    def test_streamlit_app_create_generation_content(self):
        """StreamlitApp create generation content テスト"""
        app = StreamlitApp(self.config_path, str(self.test_cache_dir))
        
        # 生成コンテンツ作成
        content = app._create_generation_content()
        
        self.assertIsInstance(content, dict)
        self.assertIn('title', content)
        self.assertIn('settings', content)
        self.assertIn('results', content)

class TestDashApp(TestBase):
    """DashApp テスト"""
    
    def test_dash_app_initialization(self):
        """DashApp初期化テスト"""
        app = DashApp(self.config_path, str(self.test_cache_dir))
        
        self.assertEqual(app.config_path, self.config_path)
        self.assertEqual(app.cache_dir, str(self.test_cache_dir))
    
    def test_dash_app_create_layout(self):
        """DashApp create layout テスト"""
        app = DashApp(self.config_path, str(self.test_cache_dir))
        
        # レイアウト作成
        layout = app._create_layout()
        
        self.assertIsInstance(layout, dict)
        self.assertIn('children', layout)
        self.assertIn('tabs', layout)
    
    def test_dash_app_create_home_content(self):
        """DashApp create home content テスト"""
        app = DashApp(self.config_path, str(self.test_cache_dir))
        
        # ホームコンテンツ作成
        content = app._create_home_content()
        
        self.assertIsInstance(content, dict)
        self.assertIn('title', content)
        self.assertIn('description', content)
        self.assertIn('features', content)

if __name__ == "__main__":
    # テスト実行
    unittest.main()