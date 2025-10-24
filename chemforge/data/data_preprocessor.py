"""
Data Preprocessor Module

データ前処理・正規化・特徴量選択の統合処理
ChEMBLデータと分子特徴量の統合前処理
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    データ前処理器
    
    包括的なデータ前処理・正規化・特徴量選択
    """
    
    def __init__(self, random_state: int = 42):
        """
        データ前処理器を初期化
        
        Args:
            random_state: ランダムシード
        """
        self.random_state = random_state
        self.scalers = {}
        self.imputers = {}
        self.feature_selectors = {}
        self.feature_names = []
        
        logger.info("DataPreprocessor initialized")
    
    def clean_data(
        self,
        df: pd.DataFrame,
        target_columns: List[str],
        min_pic50: float = 6.0,
        max_pic50: float = 12.0
    ) -> pd.DataFrame:
        """
        データクリーニング
        
        Args:
            df: データフレーム
            target_columns: ターゲット列名
            min_pic50: 最小pIC50値
            max_pic50: 最大pIC50値
        
        Returns:
            クリーニング後のデータフレーム
        """
        logger.info("Starting data cleaning")
        
        # 元のサイズ
        original_size = len(df)
        
        # 1. 欠損値の多い行を削除
        df = df.dropna(subset=target_columns, how='all')
        
        # 2. ターゲット値の範囲フィルタリング
        for col in target_columns:
            if col in df.columns:
                df = df[(df[col] >= min_pic50) & (df[col] <= max_pic50)]
        
        # 3. 重複するSMILESを削除
        df = df.drop_duplicates(subset=['canonical_smiles'], keep='first')
        
        # 4. 無効なSMILESを削除
        df = df.dropna(subset=['canonical_smiles'])
        df = df[df['canonical_smiles'].str.len() > 0]
        
        cleaned_size = len(df)
        logger.info(f"Data cleaning: {original_size} -> {cleaned_size} molecules")
        
        return df
    
    def handle_outliers(
        self,
        df: pd.DataFrame,
        target_columns: List[str],
        method: str = "iqr"
    ) -> pd.DataFrame:
        """
        外れ値処理
        
        Args:
            df: データフレーム
            target_columns: ターゲット列名
            method: 外れ値検出方法 ("iqr", "zscore", "isolation")
        
        Returns:
            外れ値処理後のデータフレーム
        """
        logger.info(f"Handling outliers using {method} method")
        
        original_size = len(df)
        filtered_df = df.copy()
        
        for col in target_columns:
            if col not in df.columns:
                continue
            
            values = df[col].dropna()
            
            if method == "iqr":
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
                
            elif method == "zscore":
                mean = values.mean()
                std = values.std()
                z_scores = np.abs((df[col] - mean) / std)
                mask = z_scores <= 3.0
            
            elif method == "isolation":
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination=0.1, random_state=self.random_state)
                outlier_mask = iso_forest.fit_predict(values.values.reshape(-1, 1))
                mask = outlier_mask == 1
            
            else:
                logger.warning(f"Unknown outlier detection method: {method}")
                continue
            
            filtered_df = filtered_df[mask]
            logger.info(f"Filtered {col}: {len(df)} -> {len(filtered_df)} molecules")
        
        filtered_size = len(filtered_df)
        logger.info(f"Outlier handling: {original_size} -> {filtered_size} molecules")
        
        return filtered_df
    
    def impute_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = "median"
    ) -> pd.DataFrame:
        """
        欠損値補完
        
        Args:
            df: データフレーム
            strategy: 補完戦略 ("mean", "median", "mode", "constant")
        
        Returns:
            欠損値補完後のデータフレーム
        """
        logger.info(f"Imputing missing values using {strategy} strategy")
        
        # 数値列のみを処理
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                if strategy == "mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif strategy == "median":
                    df[col].fillna(df[col].median(), inplace=True)
                elif strategy == "mode":
                    df[col].fillna(df[col].mode()[0], inplace=True)
                elif strategy == "constant":
                    df[col].fillna(0, inplace=True)
                
                logger.info(f"Imputed {col}: {df[col].isnull().sum()} missing values")
        
        return df
    
    def normalize_features(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        method: str = "standard"
    ) -> pd.DataFrame:
        """
        特徴量正規化
        
        Args:
            df: データフレーム
            feature_columns: 特徴量列名
            method: 正規化方法 ("standard", "minmax", "robust")
        
        Returns:
            正規化後のデータフレーム
        """
        logger.info(f"Normalizing features using {method} method")
        
        # 正規化器を選択
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # 特徴量を正規化
        for col in feature_columns:
            if col in df.columns:
                # 欠損値を一時的に補完
                temp_values = df[col].fillna(df[col].median())
                
                # 正規化
                normalized_values = scaler.fit_transform(temp_values.values.reshape(-1, 1))
                df[col] = normalized_values.flatten()
                
                # スケーラーを保存
                self.scalers[col] = scaler
        
        logger.info(f"Normalized {len(feature_columns)} features")
        return df
    
    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        k: int = 1000,
        method: str = "mutual_info"
    ) -> Tuple[np.ndarray, List[str]]:
        """
        特徴量選択
        
        Args:
            X: 特徴量配列
            y: ターゲット配列
            feature_names: 特徴量名リスト
            k: 選択する特徴量数
            method: 選択方法 ("mutual_info", "f_regression", "variance")
        
        Returns:
            (選択された特徴量, 選択された特徴量名)
        """
        logger.info(f"Selecting {k} features using {method} method")
        
        # 特徴量選択器を選択
        if method == "mutual_info":
            selector = SelectKBest(score_func=mutual_info_regression, k=k)
        elif method == "f_regression":
            selector = SelectKBest(score_func=f_regression, k=k)
        elif method == "variance":
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=0.01)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        # 特徴量選択
        X_selected = selector.fit_transform(X, y)
        
        # 選択された特徴量名を取得
        if hasattr(selector, 'get_support'):
            selected_indices = selector.get_support(indices=True)
            selected_features = [feature_names[i] for i in selected_indices]
        else:
            selected_features = feature_names[:X_selected.shape[1]]
        
        # セレクターを保存
        self.feature_selectors[method] = selector
        
        logger.info(f"Selected {len(selected_features)} features")
        return X_selected, selected_features
    
    def create_train_test_split(
        self,
        df: pd.DataFrame,
        target_columns: List[str],
        test_size: float = 0.2,
        stratify_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        訓練・テスト分割
        
        Args:
            df: データフレーム
            target_columns: ターゲット列名
            test_size: テストサイズ
            stratify_column: 層化列名
        
        Returns:
            (訓練データ, テストデータ)
        """
        logger.info(f"Creating train-test split (test_size={test_size})")
        
        if stratify_column and stratify_column in df.columns:
            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=self.random_state,
                stratify=df[stratify_column]
            )
        else:
            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=self.random_state
            )
        
        logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
        return train_df, test_df
    
    def preprocess_dataset(
        self,
        df: pd.DataFrame,
        target_columns: List[str],
        feature_columns: List[str],
        preprocessing_config: Dict
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        データセットの包括的前処理
        
        Args:
            df: データフレーム
            target_columns: ターゲット列名
            feature_columns: 特徴量列名
            preprocessing_config: 前処理設定
        
        Returns:
            (前処理後のデータフレーム, 前処理情報)
        """
        logger.info("Starting comprehensive preprocessing")
        
        preprocessing_info = {
            "original_size": len(df),
            "steps": []
        }
        
        # 1. データクリーニング
        if preprocessing_config.get("clean_data", True):
            df = self.clean_data(
                df,
                target_columns,
                min_pic50=preprocessing_config.get("min_pic50", 6.0),
                max_pic50=preprocessing_config.get("max_pic50", 12.0)
            )
            preprocessing_info["steps"].append("data_cleaning")
        
        # 2. 外れ値処理
        if preprocessing_config.get("handle_outliers", True):
            df = self.handle_outliers(
                df,
                target_columns,
                method=preprocessing_config.get("outlier_method", "iqr")
            )
            preprocessing_info["steps"].append("outlier_handling")
        
        # 3. 欠損値補完
        if preprocessing_config.get("impute_missing", True):
            df = self.impute_missing_values(
                df,
                strategy=preprocessing_config.get("impute_strategy", "median")
            )
            preprocessing_info["steps"].append("missing_value_imputation")
        
        # 4. 特徴量正規化
        if preprocessing_config.get("normalize_features", True):
            df = self.normalize_features(
                df,
                feature_columns,
                method=preprocessing_config.get("normalization_method", "standard")
            )
            preprocessing_info["steps"].append("feature_normalization")
        
        preprocessing_info["final_size"] = len(df)
        preprocessing_info["feature_columns"] = feature_columns
        preprocessing_info["target_columns"] = target_columns
        
        logger.info(f"Preprocessing completed: {preprocessing_info['original_size']} -> {preprocessing_info['final_size']}")
        
        return df, preprocessing_info
    
    def get_preprocessing_summary(self, preprocessing_info: Dict) -> Dict:
        """
        前処理要約を取得
        
        Args:
            preprocessing_info: 前処理情報
        
        Returns:
            前処理要約
        """
        summary = {
            "original_size": preprocessing_info["original_size"],
            "final_size": preprocessing_info["final_size"],
            "reduction_rate": (preprocessing_info["original_size"] - preprocessing_info["final_size"]) / preprocessing_info["original_size"],
            "preprocessing_steps": preprocessing_info["steps"],
            "feature_count": len(preprocessing_info["feature_columns"]),
            "target_count": len(preprocessing_info["target_columns"])
        }
        
        return summary