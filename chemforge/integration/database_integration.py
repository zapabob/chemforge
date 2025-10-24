"""
Database Integration Module

データベース統合モジュール
既存DatabaseManagerを活用した効率的なデータベース統合
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

# 既存モジュール活用
from chemforge.utilities.database_manager import DatabaseManager
from chemforge.data.chembl_loader import ChEMBLLoader
from chemforge.data.molecular_features import MolecularFeatures
from chemforge.data.rdkit_descriptors import RDKitDescriptors
from chemforge.utils.config_utils import ConfigManager
from chemforge.utils.logging_utils import Logger
from chemforge.utils.validation import DataValidator

logger = logging.getLogger(__name__)

class DatabaseIntegration:
    """
    データベース統合クラス
    
    既存DatabaseManagerを活用した効率的なデータベース統合
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
        self.logger = Logger("DatabaseIntegration")
        self.validator = DataValidator()
        
        # データベース管理
        self.db_manager = DatabaseManager()
        
        # データローダー
        self.chembl_loader = ChEMBLLoader(config_path, cache_dir)
        self.molecular_features = MolecularFeatures(config_path, cache_dir)
        self.rdkit_descriptors = RDKitDescriptors(config_path, cache_dir)
        
        # データベース統合設定
        self.integration_config = self.config.get('database_integration', {})
        self.auto_sync = self.integration_config.get('auto_sync', True)
        self.batch_size = self.integration_config.get('batch_size', 1000)
        self.use_cache = self.integration_config.get('use_cache', True)
        
        logger.info("DatabaseIntegration initialized")
    
    def integrate_chembl_data(self, target_chembl_ids: List[str], 
                             include_features: bool = True,
                             include_descriptors: bool = True) -> pd.DataFrame:
        """
        ChEMBLデータ統合
        
        Args:
            target_chembl_ids: ターゲットChEMBL IDリスト
            include_features: 分子特徴量含むフラグ
            include_descriptors: RDKit記述子含むフラグ
            
        Returns:
            統合データフレーム
        """
        logger.info(f"Integrating ChEMBL data for {len(target_chembl_ids)} targets")
        
        # ChEMBLデータロード
        chembl_data = self.chembl_loader.load_data(target_chembl_ids)
        
        if chembl_data.empty:
            logger.warning("No ChEMBL data loaded")
            return pd.DataFrame()
        
        logger.info(f"Loaded {len(chembl_data)} ChEMBL entries")
        
        # 分子特徴量追加
        if include_features:
            logger.info("Adding molecular features...")
            chembl_data = self.molecular_features.featurize_dataframe(
                chembl_data, smiles_col='smiles', include_3d=True
            )
            logger.info(f"Added molecular features. Shape: {chembl_data.shape}")
        
        # RDKit記述子追加
        if include_descriptors:
            logger.info("Adding RDKit descriptors...")
            chembl_data = self.rdkit_descriptors.featurize_dataframe(
                chembl_data, smiles_col='smiles',
                include_morgan=True, include_maccs=True, include_2d_descriptors=True
            )
            logger.info(f"Added RDKit descriptors. Shape: {chembl_data.shape}")
        
        # データベース保存
        if self.auto_sync:
            self._save_to_database(chembl_data, "chembl_integrated")
        
        logger.info(f"ChEMBL data integration completed. Final shape: {chembl_data.shape}")
        return chembl_data
    
    def integrate_admet_data(self, smiles_list: List[str], 
                            prediction_types: Optional[List[str]] = None) -> pd.DataFrame:
        """
        ADMETデータ統合
        
        Args:
            smiles_list: SMILESリスト
            prediction_types: 予測タイプリスト
            
        Returns:
            統合ADMETデータフレーム
        """
        logger.info(f"Integrating ADMET data for {len(smiles_list)} molecules")
        
        # ADMET予測実行
        from chemforge.admet.admet_predictor import ADMETPredictor
        admet_predictor = ADMETPredictor(self.config_path, self.cache_dir)
        
        admet_data = admet_predictor.predict_admet(smiles_list, prediction_types)
        
        if admet_data.empty:
            logger.warning("No ADMET data generated")
            return pd.DataFrame()
        
        logger.info(f"Generated ADMET data. Shape: {admet_data.shape}")
        
        # データベース保存
        if self.auto_sync:
            self._save_to_database(admet_data, "admet_predictions")
        
        return admet_data
    
    def integrate_training_data(self, model_name: str, 
                               train_data: pd.DataFrame,
                               val_data: pd.DataFrame,
                               test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        学習データ統合
        
        Args:
            model_name: モデル名
            train_data: 学習データ
            val_data: 検証データ
            test_data: テストデータ
            
        Returns:
            統合学習データ辞書
        """
        logger.info(f"Integrating training data for model: {model_name}")
        
        # データ統合
        integrated_data = {
            'model_name': model_name,
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'train_size': len(train_data),
            'val_size': len(val_data),
            'test_size': len(test_data),
            'total_size': len(train_data) + len(val_data) + len(test_data),
            'timestamp': time.time()
        }
        
        # データベース保存
        if self.auto_sync:
            self._save_training_data_to_database(integrated_data)
        
        logger.info(f"Training data integration completed for {model_name}")
        return integrated_data
    
    def integrate_prediction_data(self, model_name: str, 
                                 predictions: Dict[str, np.ndarray],
                                 metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        予測データ統合
        
        Args:
            model_name: モデル名
            predictions: 予測結果
            metadata: メタデータ
            
        Returns:
            統合予測データ辞書
        """
        logger.info(f"Integrating prediction data for model: {model_name}")
        
        # 予測データ統合
        integrated_predictions = {
            'model_name': model_name,
            'predictions': predictions,
            'metadata': metadata or {},
            'timestamp': time.time(),
            'prediction_count': len(next(iter(predictions.values()))) if predictions else 0
        }
        
        # データベース保存
        if self.auto_sync:
            self._save_prediction_data_to_database(integrated_predictions)
        
        logger.info(f"Prediction data integration completed for {model_name}")
        return integrated_predictions
    
    def _save_to_database(self, data: pd.DataFrame, table_name: str):
        """
        データベース保存
        
        Args:
            data: 保存データ
            table_name: テーブル名
        """
        try:
            # データベース接続
            conn = self.db_manager.get_connection()
            
            # データ保存
            data.to_sql(table_name, conn, if_exists='replace', index=False)
            
            # 接続閉じる
            conn.close()
            
            logger.info(f"Data saved to database table: {table_name}")
            
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
    
    def _save_training_data_to_database(self, training_data: Dict[str, Any]):
        """
        学習データベース保存
        
        Args:
            training_data: 学習データ
        """
        try:
            # データベース接続
            conn = self.db_manager.get_connection()
            
            # 学習データ保存
            training_data['train_data'].to_sql(
                f"{training_data['model_name']}_train", conn, if_exists='replace', index=False
            )
            training_data['val_data'].to_sql(
                f"{training_data['model_name']}_val", conn, if_exists='replace', index=False
            )
            training_data['test_data'].to_sql(
                f"{training_data['model_name']}_test", conn, if_exists='replace', index=False
            )
            
            # メタデータ保存
            metadata_df = pd.DataFrame([{
                'model_name': training_data['model_name'],
                'train_size': training_data['train_size'],
                'val_size': training_data['val_size'],
                'test_size': training_data['test_size'],
                'total_size': training_data['total_size'],
                'timestamp': training_data['timestamp']
            }])
            metadata_df.to_sql(
                f"{training_data['model_name']}_metadata", conn, if_exists='replace', index=False
            )
            
            # 接続閉じる
            conn.close()
            
            logger.info(f"Training data saved to database for model: {training_data['model_name']}")
            
        except Exception as e:
            logger.error(f"Error saving training data to database: {e}")
    
    def _save_prediction_data_to_database(self, prediction_data: Dict[str, Any]):
        """
        予測データベース保存
        
        Args:
            prediction_data: 予測データ
        """
        try:
            # データベース接続
            conn = self.db_manager.get_connection()
            
            # 予測結果保存
            for key, values in prediction_data['predictions'].items():
                if isinstance(values, np.ndarray):
                    pred_df = pd.DataFrame({key: values})
                    pred_df.to_sql(
                        f"{prediction_data['model_name']}_{key}_predictions", 
                        conn, if_exists='replace', index=False
                    )
            
            # メタデータ保存
            metadata_df = pd.DataFrame([{
                'model_name': prediction_data['model_name'],
                'prediction_count': prediction_data['prediction_count'],
                'timestamp': prediction_data['timestamp'],
                'metadata': json.dumps(prediction_data['metadata'])
            }])
            metadata_df.to_sql(
                f"{prediction_data['model_name']}_prediction_metadata", 
                conn, if_exists='replace', index=False
            )
            
            # 接続閉じる
            conn.close()
            
            logger.info(f"Prediction data saved to database for model: {prediction_data['model_name']}")
            
        except Exception as e:
            logger.error(f"Error saving prediction data to database: {e}")
    
    def get_integrated_data(self, data_type: str, 
                           filters: Optional[Dict] = None) -> pd.DataFrame:
        """
        統合データ取得
        
        Args:
            data_type: データタイプ
            filters: フィルター条件
            
        Returns:
            統合データフレーム
        """
        try:
            # データベース接続
            conn = self.db_manager.get_connection()
            
            # データ取得
            if filters:
                where_clause = " AND ".join([f"{k} = '{v}'" for k, v in filters.items()])
                query = f"SELECT * FROM {data_type} WHERE {where_clause}"
            else:
                query = f"SELECT * FROM {data_type}"
            
            data = pd.read_sql_query(query, conn)
            
            # 接続閉じる
            conn.close()
            
            logger.info(f"Retrieved {len(data)} records from {data_type}")
            return data
            
        except Exception as e:
            logger.error(f"Error retrieving integrated data: {e}")
            return pd.DataFrame()
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        データサマリー取得
        
        Returns:
            データサマリー辞書
        """
        try:
            # データベース接続
            conn = self.db_manager.get_connection()
            
            # テーブル一覧取得
            tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
            tables = pd.read_sql_query(tables_query, conn)
            
            summary = {
                'total_tables': len(tables),
                'tables': [],
                'total_records': 0
            }
            
            # 各テーブルの情報取得
            for table_name in tables['name']:
                try:
                    count_query = f"SELECT COUNT(*) as count FROM {table_name}"
                    count_result = pd.read_sql_query(count_query, conn)
                    record_count = count_result['count'].iloc[0]
                    
                    summary['tables'].append({
                        'name': table_name,
                        'record_count': record_count
                    })
                    summary['total_records'] += record_count
                    
                except Exception as e:
                    logger.warning(f"Error getting info for table {table_name}: {e}")
            
            # 接続閉じる
            conn.close()
            
            logger.info(f"Data summary: {summary['total_tables']} tables, {summary['total_records']} records")
            return summary
            
        except Exception as e:
            logger.error(f"Error getting data summary: {e}")
            return {}
    
    def export_integrated_data(self, data_type: str, output_path: str, 
                              format: str = "csv", filters: Optional[Dict] = None) -> bool:
        """
        統合データエクスポート
        
        Args:
            data_type: データタイプ
            output_path: 出力パス
            format: 出力形式
            filters: フィルター条件
            
        Returns:
            成功フラグ
        """
        try:
            # データ取得
            data = self.get_integrated_data(data_type, filters)
            
            if data.empty:
                logger.warning(f"No data found for {data_type}")
                return False
            
            # 出力パス準備
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 形式別保存
            if format.lower() == "csv":
                data.to_csv(output_path, index=False)
            elif format.lower() == "json":
                data.to_json(output_path, orient='records', indent=2)
            elif format.lower() == "parquet":
                data.to_parquet(output_path, index=False)
            else:
                logger.error(f"Unsupported format: {format}")
                return False
            
            logger.info(f"Integrated data exported to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting integrated data: {e}")
            return False

def create_database_integration(config_path: Optional[str] = None, 
                              cache_dir: str = "cache") -> DatabaseIntegration:
    """
    データベース統合作成
    
    Args:
        config_path: 設定ファイルパス
        cache_dir: キャッシュディレクトリ
        
    Returns:
        DatabaseIntegration
    """
    return DatabaseIntegration(config_path, cache_dir)

if __name__ == "__main__":
    # テスト実行
    db_integration = DatabaseIntegration()
    
    print(f"DatabaseIntegration created: {db_integration}")
    print(f"Auto sync: {db_integration.auto_sync}")
    print(f"Batch size: {db_integration.batch_size}")
    print(f"Use cache: {db_integration.use_cache}")
