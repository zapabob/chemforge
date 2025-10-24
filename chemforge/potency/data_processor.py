"""
Potency Data Processor

pIC50/pKi力価回帰用のデータ前処理モジュール
メタプロンプト仕様に従った前処理を実装
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

class PotencyDataProcessor:
    """pIC50/pKi力価回帰用データ前処理クラス"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config or {}
        
        # デフォルト設定
        self.pIC50_cutoff = self.config.get('pIC50_low_activity_ic50_uM', 1000.0)  # IC50≥1000μM → pIC50≤3
        self.pKi_cutoff = self.config.get('pKi_low_activity_ki_uM', 10.0)  # Ki≥10μM → pKi≤5
        self.strict_outlier_drop = self.config.get('strict_outlier_drop', True)
        self.outlier_multiplier = self.config.get('outlier_multiplier', 3.0)  # IQR±3×IQR
        
        # スプリット設定
        self.split_method = self.config.get('split', 'scaffold')  # 'scaffold' or 'time'
        self.train_ratio = self.config.get('train_ratio', 0.8)
        self.val_ratio = self.config.get('val_ratio', 0.1)
        self.test_ratio = self.config.get('test_ratio', 0.1)
        
    def clean_and_convert(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        SMILES正規化と単位変換→pIC50/pKi計算
        
        Args:
            df: 入力DataFrame（smiles, assay_id, unit, value, type, target列が必要）
            
        Returns:
            処理済みDataFrame
        """
        print("[INFO] SMILES正規化と単位変換開始...")
        
        # 必要な列の確認
        required_cols = ['smiles', 'assay_id', 'unit', 'value', 'type', 'target']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # コピー作成
        df_clean = df.copy()
        
        # SMILES正規化
        print("[INFO] SMILES正規化中...")
        df_clean['smiles_clean'] = df_clean['smiles'].apply(self._normalize_smiles)
        
        # 無効なSMILESを除外
        valid_mask = df_clean['smiles_clean'].notna()
        df_clean = df_clean[valid_mask].copy()
        print(f"[INFO] 有効なSMILES: {len(df_clean)}/{len(df)} ({len(df_clean)/len(df)*100:.1f}%)")
        
        # 単位変換→p値計算
        print("[INFO] 単位変換→p値計算中...")
        df_clean['p_value'] = df_clean.apply(self._convert_to_p_value, axis=1)
        
        # 無効なp値を除外
        valid_p_mask = df_clean['p_value'].notna() & (df_clean['p_value'] > 0)
        df_clean = df_clean[valid_p_mask].copy()
        print(f"[INFO] 有効なp値: {len(df_clean)}/{len(df)} ({len(df_clean)/len(df)*100:.1f}%)")
        
        return df_clean
    
    def _normalize_smiles(self, smiles: str) -> Optional[str]:
        """SMILES正規化（塩離脱、標準化、カノニカル化）"""
        try:
            if pd.isna(smiles) or smiles == '':
                return None
            
            # RDKitで分子オブジェクト作成
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # 塩離脱（最大フラグメントを選択）
            mol = Chem.GetMolFrags(mol, asMols=True)
            if not mol:
                return None
            mol = max(mol, key=lambda x: x.GetNumAtoms())
            
            # 標準化（水素追加、芳香性修正）
            mol = Chem.AddHs(mol)
            Chem.SanitizeMol(mol)
            mol = Chem.RemoveHs(mol)
            
            # カノニカルSMILES生成
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            return canonical_smiles
            
        except Exception as e:
            print(f"[WARNING] SMILES正規化失敗: {smiles} - {e}")
            return None
    
    def _convert_to_p_value(self, row: pd.Series) -> Optional[float]:
        """単位変換→p値計算"""
        try:
            value = float(row['value'])
            unit = str(row['unit']).lower()
            value_type = str(row['type']).upper()
            
            # 単位をMolarに変換
            if 'nm' in unit or 'nmol' in unit:
                value_molar = value * 1e-9
            elif 'μm' in unit or 'umol' in unit:
                value_molar = value * 1e-6
            elif 'mm' in unit or 'mmol' in unit:
                value_molar = value * 1e-3
            elif 'm' in unit or 'mol' in unit:
                value_molar = value
            else:
                print(f"[WARNING] 未知の単位: {unit}")
                return None
            
            # p値計算
            if value_molar > 0:
                p_value = -np.log10(value_molar)
                return p_value
            else:
                return None
                
        except Exception as e:
            print(f"[WARNING] 単位変換失敗: {row['value']} {row['unit']} - {e}")
            return None
    
    def filter_low_activity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        低活性カット（pIC50≤3, pKi≤5）
        
        Args:
            df: 入力DataFrame
            
        Returns:
            フィルタ済みDataFrame
        """
        print("[INFO] 低活性カット開始...")
        
        df_filtered = df.copy()
        original_count = len(df_filtered)
        
        # pIC50の低活性カット
        ic50_mask = (df_filtered['type'] == 'IC50') & (df_filtered['p_value'] <= 3.0)
        ic50_count = ic50_mask.sum()
        if ic50_count > 0:
            print(f"[INFO] pIC50≤3.0 (IC50≥1000μM) 除外: {ic50_count}件")
            df_filtered = df_filtered[~ic50_mask]
        
        # pKiの低活性カット
        ki_mask = (df_filtered['type'] == 'Ki') & (df_filtered['p_value'] <= 5.0)
        ki_count = ki_mask.sum()
        if ki_count > 0:
            print(f"[INFO] pKi≤5.0 (Ki≥10μM) 除外: {ki_count}件")
            df_filtered = df_filtered[~ki_mask]
        
        filtered_count = len(df_filtered)
        print(f"[INFO] 低活性カット完了: {original_count} → {filtered_count} ({filtered_count/original_count*100:.1f}%)")
        
        return df_filtered
    
    def detect_outliers_iqr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        IQR外れ値検出（assay_id×typeごと）
        
        Args:
            df: 入力DataFrame
            
        Returns:
            外れ値マーク済みDataFrame
        """
        print("[INFO] IQR外れ値検出開始...")
        
        df_outlier = df.copy()
        df_outlier['is_outlier'] = False
        
        # assay_id×typeごとにIQR計算
        for (assay_id, value_type), group in df_outlier.groupby(['assay_id', 'type']):
            if len(group) < 4:  # サンプル数が少なすぎる場合はスキップ
                continue
            
            p_values = group['p_value'].values
            Q1 = np.percentile(p_values, 25)
            Q3 = np.percentile(p_values, 75)
            IQR = Q3 - Q1
            
            # Tukey fence: IQR±3×IQR
            lower_bound = Q1 - self.outlier_multiplier * IQR
            upper_bound = Q3 + self.outlier_multiplier * IQR
            
            # 外れ値マーク
            outlier_mask = (p_values < lower_bound) | (p_values > upper_bound)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                print(f"[INFO] {assay_id}×{value_type}: {outlier_count}/{len(group)} 外れ値検出")
                df_outlier.loc[group.index[outlier_mask], 'is_outlier'] = True
        
        total_outliers = df_outlier['is_outlier'].sum()
        print(f"[INFO] 外れ値検出完了: {total_outliers}/{len(df_outlier)} ({total_outliers/len(df_outlier)*100:.1f}%)")
        
        return df_outlier
    
    def scaffold_split(self, df: pd.DataFrame, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Bemis-Murcko scaffold split
        
        Args:
            df: 入力DataFrame
            seed: ランダムシード
            
        Returns:
            (train_df, val_df, test_df)
        """
        print("[INFO] Scaffold split開始...")
        
        # Bemis-Murcko scaffold計算
        scaffolds = []
        for smiles in df['smiles_clean']:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    scaffolds.append('invalid')
                else:
                    scaffold = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)
                    scaffold_smiles = Chem.MolToSmiles(scaffold) if scaffold else 'no_scaffold'
                    scaffolds.append(scaffold_smiles)
            except:
                scaffolds.append('invalid')
        
        df_scaffold = df.copy()
        df_scaffold['scaffold'] = scaffolds
        
        # 外れ値を除外（strict_outlier_drop=Trueの場合）
        if self.strict_outlier_drop:
            df_scaffold = df_scaffold[~df_scaffold['is_outlier']].copy()
            print(f"[INFO] 外れ値除外後: {len(df_scaffold)}件")
        
        # Scaffold別に分割
        unique_scaffolds = df_scaffold['scaffold'].unique()
        np.random.seed(seed)
        np.random.shuffle(unique_scaffolds)
        
        n_scaffolds = len(unique_scaffolds)
        train_end = int(n_scaffolds * self.train_ratio)
        val_end = int(n_scaffolds * (self.train_ratio + self.val_ratio))
        
        train_scaffolds = unique_scaffolds[:train_end]
        val_scaffolds = unique_scaffolds[train_end:val_end]
        test_scaffolds = unique_scaffolds[val_end:]
        
        train_df = df_scaffold[df_scaffold['scaffold'].isin(train_scaffolds)]
        val_df = df_scaffold[df_scaffold['scaffold'].isin(val_scaffolds)]
        test_df = df_scaffold[df_scaffold['scaffold'].isin(test_scaffolds)]
        
        print(f"[INFO] Scaffold split完了:")
        print(f"  Train: {len(train_df)} ({len(train_df)/len(df_scaffold)*100:.1f}%)")
        print(f"  Val:   {len(val_df)} ({len(val_df)/len(df_scaffold)*100:.1f}%)")
        print(f"  Test:  {len(test_df)} ({len(test_df)/len(df_scaffold)*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def time_split(self, df: pd.DataFrame, date_col: str = 'date') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Time split（日付順）
        
        Args:
            df: 入力DataFrame
            date_col: 日付列名
            
        Returns:
            (train_df, val_df, test_df)
        """
        print("[INFO] Time split開始...")
        
        if date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not found")
        
        # 日付でソート
        df_time = df.copy()
        df_time[date_col] = pd.to_datetime(df_time[date_col])
        df_time = df_time.sort_values(date_col)
        
        # 時系列分割
        n_samples = len(df_time)
        train_end = int(n_samples * self.train_ratio)
        val_end = int(n_samples * (self.train_ratio + self.val_ratio))
        
        train_df = df_time.iloc[:train_end]
        val_df = df_time.iloc[train_end:val_end]
        test_df = df_time.iloc[val_end:]
        
        print(f"[INFO] Time split完了:")
        print(f"  Train: {len(train_df)} ({len(train_df)/n_samples*100:.1f}%)")
        print(f"  Val:   {len(val_df)} ({len(val_df)/n_samples*100:.1f}%)")
        print(f"  Test:  {len(test_df)} ({len(test_df)/n_samples*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def process_pipeline(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        完全な前処理パイプライン
        
        Args:
            df: 入力DataFrame
            
        Returns:
            (train_df, val_df, test_df)
        """
        print("=" * 60)
        print("pIC50/pKi力価回帰データ前処理パイプライン開始")
        print("=" * 60)
        
        # 1. SMILES正規化と単位変換
        df_clean = self.clean_and_convert(df)
        
        # 2. 低活性カット
        df_filtered = self.filter_low_activity(df_clean)
        
        # 3. IQR外れ値検出
        df_outlier = self.detect_outliers_iqr(df_filtered)
        
        # 4. スプリット
        if self.split_method == 'scaffold':
            train_df, val_df, test_df = self.scaffold_split(df_outlier)
        elif self.split_method == 'time':
            train_df, val_df, test_df = self.time_split(df_outlier)
        else:
            raise ValueError(f"Unknown split method: {self.split_method}")
        
        print("=" * 60)
        print("前処理パイプライン完了")
        print("=" * 60)
        
        return train_df, val_df, test_df
