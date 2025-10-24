"""
ADMET Predictor Module

ADMET予測統合モジュール
既存SwissADME・外部APIを活用した効率的なADMET予測
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
from chemforge.utils.web_scraper import SwissADMEScraper
from chemforge.utils.external_apis import ExternalAPIManager
from chemforge.utils.config_utils import ConfigManager
from chemforge.utils.logging_utils import Logger
from chemforge.utils.validation import DataValidator

logger = logging.getLogger(__name__)

class ADMETPredictor:
    """
    ADMET予測統合クラス
    
    既存SwissADME・外部APIを活用した効率的なADMET予測
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
        self.logger = Logger("ADMETPredictor")
        self.validator = DataValidator()
        
        # 外部サービス統合
        self.swiss_adme = SwissADMEScraper()
        self.external_apis = ExternalAPIManager()
        
        # ADMET予測設定
        self.admet_config = self.config.get('admet_prediction', {})
        self.prediction_types = self.admet_config.get('prediction_types', [
            'physicochemical', 'pharmacokinetic', 'toxicity', 'druglikeness'
        ])
        self.use_cache = self.admet_config.get('use_cache', True)
        self.batch_size = self.admet_config.get('batch_size', 10)
        
        logger.info("ADMETPredictor initialized")
    
    def predict_admet(self, smiles_list: List[str], 
                      prediction_types: Optional[List[str]] = None,
                      use_cache: bool = True) -> pd.DataFrame:
        """
        ADMET予測実行
        
        Args:
            smiles_list: SMILESリスト
            prediction_types: 予測タイプリスト
            use_cache: キャッシュ使用
            
        Returns:
            ADMET予測結果データフレーム
        """
        logger.info(f"Predicting ADMET for {len(smiles_list)} molecules")
        
        if prediction_types is None:
            prediction_types = self.prediction_types
        
        # キャッシュチェック
        cache_key = f"admet_predictions_{len(smiles_list)}_{'_'.join(prediction_types)}"
        cache_path = self.cache_dir / f"{cache_key}.csv"
        
        if use_cache and cache_path.exists():
            logger.info(f"Loading from cache: {cache_path}")
            return pd.read_csv(cache_path)
        
        # ADMET予測実行
        all_predictions = []
        
        for i, smiles in enumerate(tqdm(smiles_list, desc="Predicting ADMET")):
            try:
                # 分子予測
                molecule_predictions = self._predict_single_molecule(
                    smiles, prediction_types
                )
                
                if molecule_predictions:
                    molecule_predictions['smiles'] = smiles
                    all_predictions.append(molecule_predictions)
                
            except Exception as e:
                logger.error(f"Failed to predict ADMET for {smiles}: {e}")
                continue
        
        if not all_predictions:
            logger.warning("No ADMET predictions generated")
            return pd.DataFrame()
        
        # データフレーム作成
        predictions_df = pd.DataFrame(all_predictions)
        logger.info(f"Generated ADMET predictions for {len(predictions_df)} molecules")
        
        # キャッシュ保存
        if use_cache:
            predictions_df.to_csv(cache_path, index=False)
            logger.info(f"ADMET predictions cached to: {cache_path}")
        
        return predictions_df
    
    def _predict_single_molecule(self, smiles: str, 
                               prediction_types: List[str]) -> Optional[Dict]:
        """
        単一分子ADMET予測
        
        Args:
            smiles: SMILES文字列
            prediction_types: 予測タイプリスト
            
        Returns:
            予測結果辞書
        """
        try:
            predictions = {}
            
            # 物理化学特性予測
            if 'physicochemical' in prediction_types:
                physico_predictions = self._predict_physicochemical(smiles)
                predictions.update(physico_predictions)
            
            # 薬物動態予測
            if 'pharmacokinetic' in prediction_types:
                pk_predictions = self._predict_pharmacokinetic(smiles)
                predictions.update(pk_predictions)
            
            # 毒性予測
            if 'toxicity' in prediction_types:
                toxicity_predictions = self._predict_toxicity(smiles)
                predictions.update(toxicity_predictions)
            
            # 薬物らしさ予測
            if 'druglikeness' in prediction_types:
                druglikeness_predictions = self._predict_druglikeness(smiles)
                predictions.update(druglikeness_predictions)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting ADMET for {smiles}: {e}")
            return None
    
    def _predict_physicochemical(self, smiles: str) -> Dict:
        """
        物理化学特性予測
        
        Args:
            smiles: SMILES文字列
            
        Returns:
            物理化学特性予測結果
        """
        try:
            # SwissADME予測
            swiss_results = self.swiss_adme.predict_single_molecule(smiles)
            
            # 物理化学特性抽出
            physicochemical = {
                'molecular_weight': swiss_results.get('molecular_weight', 0.0),
                'logp': swiss_results.get('logp', 0.0),
                'tpsa': swiss_results.get('tpsa', 0.0),
                'hbd': swiss_results.get('hbd', 0),
                'hba': swiss_results.get('hba', 0),
                'rotatable_bonds': swiss_results.get('rotatable_bonds', 0),
                'aromatic_rings': swiss_results.get('aromatic_rings', 0),
                'molecular_volume': swiss_results.get('molecular_volume', 0.0),
                'polar_surface_area': swiss_results.get('polar_surface_area', 0.0)
            }
            
            return physicochemical
            
        except Exception as e:
            logger.error(f"Error predicting physicochemical properties: {e}")
            return {}
    
    def _predict_pharmacokinetic(self, smiles: str) -> Dict:
        """
        薬物動態予測
        
        Args:
            smiles: SMILES文字列
            
        Returns:
            薬物動態予測結果
        """
        try:
            # SwissADME予測
            swiss_results = self.swiss_adme.predict_single_molecule(smiles)
            
            # 薬物動態特性抽出
            pharmacokinetic = {
                'caco2_permeability': swiss_results.get('caco2_permeability', 0.0),
                'mdck_permeability': swiss_results.get('mdck_permeability', 0.0),
                'p_gp_substrate': swiss_results.get('p_gp_substrate', False),
                'p_gp_inhibitor': swiss_results.get('p_gp_inhibitor', False),
                'cyp1a2_inhibitor': swiss_results.get('cyp1a2_inhibitor', False),
                'cyp2c19_inhibitor': swiss_results.get('cyp2c19_inhibitor', False),
                'cyp2c9_inhibitor': swiss_results.get('cyp2c9_inhibitor', False),
                'cyp2d6_inhibitor': swiss_results.get('cyp2d6_inhibitor', False),
                'cyp3a4_inhibitor': swiss_results.get('cyp3a4_inhibitor', False),
                'cyp1a2_substrate': swiss_results.get('cyp1a2_substrate', False),
                'cyp2c19_substrate': swiss_results.get('cyp2c19_substrate', False),
                'cyp2c9_substrate': swiss_results.get('cyp2c9_substrate', False),
                'cyp2d6_substrate': swiss_results.get('cyp2d6_substrate', False),
                'cyp3a4_substrate': swiss_results.get('cyp3a4_substrate', False)
            }
            
            return pharmacokinetic
            
        except Exception as e:
            logger.error(f"Error predicting pharmacokinetic properties: {e}")
            return {}
    
    def _predict_toxicity(self, smiles: str) -> Dict:
        """
        毒性予測
        
        Args:
            smiles: SMILES文字列
            
        Returns:
            毒性予測結果
        """
        try:
            # SwissADME予測
            swiss_results = self.swiss_adme.predict_single_molecule(smiles)
            
            # 毒性特性抽出
            toxicity = {
                'ames_toxicity': swiss_results.get('ames_toxicity', False),
                'carcinogenicity': swiss_results.get('carcinogenicity', False),
                'hepatotoxicity': swiss_results.get('hepatotoxicity', False),
                'cardiotoxicity': swiss_results.get('cardiotoxicity', False),
                'nephrotoxicity': swiss_results.get('nephrotoxicity', False),
                'neurotoxicity': swiss_results.get('neurotoxicity', False),
                'mutagenicity': swiss_results.get('mutagenicity', False),
                'teratogenicity': swiss_results.get('teratogenicity', False),
                'ld50_oral': swiss_results.get('ld50_oral', 0.0),
                'ld50_intravenous': swiss_results.get('ld50_intravenous', 0.0)
            }
            
            return toxicity
            
        except Exception as e:
            logger.error(f"Error predicting toxicity: {e}")
            return {}
    
    def _predict_druglikeness(self, smiles: str) -> Dict:
        """
        薬物らしさ予測
        
        Args:
            smiles: SMILES文字列
            
        Returns:
            薬物らしさ予測結果
        """
        try:
            # SwissADME予測
            swiss_results = self.swiss_adme.predict_single_molecule(smiles)
            
            # 薬物らしさ特性抽出
            druglikeness = {
                'lipinski_violations': swiss_results.get('lipinski_violations', 0),
                'veber_violations': swiss_results.get('veber_violations', 0),
                'qed_score': swiss_results.get('qed_score', 0.0),
                'sa_score': swiss_results.get('sa_score', 0.0),
                'druglikeness_score': swiss_results.get('druglikeness_score', 0.0),
                'leadlikeness_score': swiss_results.get('leadlikeness_score', 0.0),
                'synthetic_accessibility': swiss_results.get('synthetic_accessibility', 0.0),
                'bioavailability_score': swiss_results.get('bioavailability_score', 0.0)
            }
            
            return druglikeness
            
        except Exception as e:
            logger.error(f"Error predicting druglikeness: {e}")
            return {}
    
    def predict_batch(self, smiles_list: List[str], 
                     prediction_types: Optional[List[str]] = None,
                     batch_size: Optional[int] = None) -> pd.DataFrame:
        """
        バッチADMET予測
        
        Args:
            smiles_list: SMILESリスト
            prediction_types: 予測タイプリスト
            batch_size: バッチサイズ
            
        Returns:
            ADMET予測結果データフレーム
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        logger.info(f"Batch ADMET prediction for {len(smiles_list)} molecules")
        
        all_predictions = []
        
        # バッチ処理
        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(smiles_list) + batch_size - 1)//batch_size}")
            
            batch_predictions = self.predict_admet(
                batch_smiles, prediction_types, use_cache=True
            )
            
            if not batch_predictions.empty:
                all_predictions.append(batch_predictions)
        
        if not all_predictions:
            logger.warning("No batch predictions generated")
            return pd.DataFrame()
        
        # バッチ結果統合
        combined_predictions = pd.concat(all_predictions, ignore_index=True)
        logger.info(f"Batch ADMET prediction completed: {len(combined_predictions)} molecules")
        
        return combined_predictions
    
    def get_prediction_summary(self, predictions_df: pd.DataFrame) -> Dict:
        """
        予測結果サマリー取得
        
        Args:
            predictions_df: 予測結果データフレーム
            
        Returns:
            サマリー辞書
        """
        try:
            summary = {
                'total_molecules': len(predictions_df),
                'prediction_columns': list(predictions_df.columns),
                'missing_values': predictions_df.isnull().sum().to_dict(),
                'statistics': {}
            }
            
            # 数値列の統計
            numeric_columns = predictions_df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                summary['statistics'][col] = {
                    'mean': predictions_df[col].mean(),
                    'std': predictions_df[col].std(),
                    'min': predictions_df[col].min(),
                    'max': predictions_df[col].max(),
                    'median': predictions_df[col].median()
                }
            
            # カテゴリ列の統計
            categorical_columns = predictions_df.select_dtypes(include=['object', 'bool']).columns
            for col in categorical_columns:
                summary['statistics'][col] = {
                    'unique_values': predictions_df[col].nunique(),
                    'most_common': predictions_df[col].mode().iloc[0] if not predictions_df[col].mode().empty else None
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating prediction summary: {e}")
            return {}
    
    def export_predictions(self, predictions_df: pd.DataFrame, output_path: str, 
                          format: str = "csv") -> bool:
        """
        予測結果エクスポート
        
        Args:
            predictions_df: 予測結果データフレーム
            output_path: 出力パス
            format: 出力形式
            
        Returns:
            成功フラグ
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == "csv":
                predictions_df.to_csv(output_path, index=False)
            elif format.lower() == "json":
                predictions_df.to_json(output_path, orient='records', indent=2)
            elif format.lower() == "parquet":
                predictions_df.to_parquet(output_path, index=False)
            else:
                logger.error(f"Unsupported format: {format}")
                return False
            
            logger.info(f"ADMET predictions exported to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting predictions: {e}")
            return False

def create_admet_predictor(config_path: Optional[str] = None, 
                          cache_dir: str = "cache") -> ADMETPredictor:
    """
    ADMET予測器作成
    
    Args:
        config_path: 設定ファイルパス
        cache_dir: キャッシュディレクトリ
        
    Returns:
        ADMETPredictor
    """
    return ADMETPredictor(config_path, cache_dir)

def predict_admet(smiles_list: List[str], 
                  prediction_types: Optional[List[str]] = None,
                  config_path: Optional[str] = None,
                  cache_dir: str = "cache") -> pd.DataFrame:
    """
    ADMET予測（簡易版）
    
    Args:
        smiles_list: SMILESリスト
        prediction_types: 予測タイプリスト
        config_path: 設定ファイルパス
        cache_dir: キャッシュディレクトリ
        
    Returns:
        ADMET予測結果データフレーム
    """
    predictor = create_admet_predictor(config_path, cache_dir)
    return predictor.predict_admet(smiles_list, prediction_types)

if __name__ == "__main__":
    # テスト実行
    predictor = ADMETPredictor()
    
    # テストSMILES
    test_smiles = ["CCO", "CCN", "c1ccccc1", "CC(=O)O"]
    
    # ADMET予測
    predictions = predictor.predict_admet(test_smiles)
    
    print(f"ADMET predictions for {len(predictions)} molecules")
    if not predictions.empty:
        print(f"Prediction columns: {list(predictions.columns)}")
        print(f"Prediction shape: {predictions.shape}")
        print(f"Sample predictions:")
        print(predictions.head())
        
        # サマリー
        summary = predictor.get_prediction_summary(predictions)
        print(f"Prediction summary: {summary}")