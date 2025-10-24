"""
カスタムデータローダー

CSVやSDF形式のカスタムデータセットを読み込み、
骨格特徴量を付与して前処理済みデータセットを構築。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from .molecular_preprocessor import MolecularPreprocessor


class CustomDataLoader:
    """カスタムデータローダークラス"""
    
    def __init__(
        self,
        use_scaffold_features: bool = True,
        use_admet: bool = True,
        verbose: bool = False
    ):
        """
        カスタムデータローダーを初期化
        
        Args:
            use_scaffold_features: 骨格特徴量を使用するか
            use_admet: ADMET予測を使用するか
            verbose: 詳細ログ出力フラグ
        """
        self.verbose = verbose
        self.preprocessor = MolecularPreprocessor(
            use_scaffold_features=use_scaffold_features,
            use_admet=use_admet,
            verbose=verbose
        )
        
        if self.verbose:
            print("CustomDataLoader initialized")
            print(f"  Use scaffold features: {use_scaffold_features}")
            print(f"  Use ADMET: {use_admet}")
    
    def load_csv(
        self,
        path: str,
        smiles_col: str = 'smiles',
        target_col: str = 'pIC50',
        pIC50_threshold: float = 6.0,
        remove_outliers: bool = True,
        fill_missing: bool = True
    ) -> pd.DataFrame:
        """
        CSVファイルを読み込み・前処理
        
        Args:
            path: CSVファイルパス
            smiles_col: SMILES列名
            target_col: ターゲット列名
            pIC50_threshold: pIC50閾値
            remove_outliers: アウトライアを除外するか
            fill_missing: 欠損値を補完するか
            
        Returns:
            前処理済みデータフレーム
        """
        if self.verbose:
            print(f"Loading CSV from: {path}")
        
        try:
            # CSVファイルを読み込み
            df = pd.read_csv(path)
            
            if self.verbose:
                print(f"Loaded {len(df)} records from CSV")
                print(f"Columns: {list(df.columns)}")
            
            # データを前処理
            processed_df = self._preprocess_data(
                df, smiles_col, target_col, pIC50_threshold, remove_outliers, fill_missing
            )
            
            if self.verbose:
                print(f"Processed {len(processed_df)} records")
            
            return processed_df
            
        except Exception as e:
            if self.verbose:
                print(f"Error loading CSV: {e}")
            return pd.DataFrame()
    
    def load_sdf(
        self,
        path: str,
        target_col: str = 'pIC50',
        pIC50_threshold: float = 6.0,
        remove_outliers: bool = True,
        fill_missing: bool = True
    ) -> pd.DataFrame:
        """
        SDFファイルを読み込み・前処理
        
        Args:
            path: SDFファイルパス
            target_col: ターゲット列名
            pIC50_threshold: pIC50閾値
            remove_outliers: アウトライアを除外するか
            fill_missing: 欠損値を補完するか
            
        Returns:
            前処理済みデータフレーム
        """
        if self.verbose:
            print(f"Loading SDF from: {path}")
        
        try:
            # SDFファイルを読み込み
            from rdkit import Chem
            from rdkit.Chem import PandasTools
            
            # SDFファイルを読み込み
            supplier = Chem.SDMolSupplier(path)
            mols = [mol for mol in supplier if mol is not None]
            
            if self.verbose:
                print(f"Loaded {len(mols)} molecules from SDF")
            
            # データフレームに変換
            df = pd.DataFrame()
            smiles_list = []
            target_values = []
            
            for mol in mols:
                try:
                    # SMILESを取得
                    smiles = Chem.MolToSmiles(mol)
                    smiles_list.append(smiles)
                    
                    # ターゲット値を取得（SDFのプロパティから）
                    if target_col in mol.GetPropNames():
                        target_value = float(mol.GetProp(target_col))
                        target_values.append(target_value)
                    else:
                        target_values.append(np.nan)
                        
                except Exception as e:
                    if self.verbose:
                        print(f"Error processing molecule: {e}")
                    continue
            
            # データフレームを作成
            df['smiles'] = smiles_list
            df[target_col] = target_values
            
            if self.verbose:
                print(f"Created dataframe with {len(df)} records")
            
            # データを前処理
            processed_df = self._preprocess_data(
                df, 'smiles', target_col, pIC50_threshold, remove_outliers, fill_missing
            )
            
            if self.verbose:
                print(f"Processed {len(processed_df)} records")
            
            return processed_df
            
        except Exception as e:
            if self.verbose:
                print(f"Error loading SDF: {e}")
            return pd.DataFrame()
    
    def load_excel(
        self,
        path: str,
        sheet_name: str = 0,
        smiles_col: str = 'smiles',
        target_col: str = 'pIC50',
        pIC50_threshold: float = 6.0,
        remove_outliers: bool = True,
        fill_missing: bool = True
    ) -> pd.DataFrame:
        """
        Excelファイルを読み込み・前処理
        
        Args:
            path: Excelファイルパス
            sheet_name: シート名
            smiles_col: SMILES列名
            target_col: ターゲット列名
            pIC50_threshold: pIC50閾値
            remove_outliers: アウトライアを除外するか
            fill_missing: 欠損値を補完するか
            
        Returns:
            前処理済みデータフレーム
        """
        if self.verbose:
            print(f"Loading Excel from: {path}")
        
        try:
            # Excelファイルを読み込み
            df = pd.read_excel(path, sheet_name=sheet_name)
            
            if self.verbose:
                print(f"Loaded {len(df)} records from Excel")
                print(f"Columns: {list(df.columns)}")
            
            # データを前処理
            processed_df = self._preprocess_data(
                df, smiles_col, target_col, pIC50_threshold, remove_outliers, fill_missing
            )
            
            if self.verbose:
                print(f"Processed {len(processed_df)} records")
            
            return processed_df
            
        except Exception as e:
            if self.verbose:
                print(f"Error loading Excel: {e}")
            return pd.DataFrame()
    
    def _preprocess_data(
        self,
        df: pd.DataFrame,
        smiles_col: str,
        target_col: str,
        pIC50_threshold: float,
        remove_outliers: bool,
        fill_missing: bool
    ) -> pd.DataFrame:
        """データを前処理"""
        if df.empty:
            return df
        
        # コピーを作成
        processed_df = df.copy()
        
        # 1. 必要な列が存在するかチェック
        if smiles_col not in processed_df.columns:
            if self.verbose:
                print(f"SMILES column '{smiles_col}' not found")
            return pd.DataFrame()
        
        if target_col not in processed_df.columns:
            if self.verbose:
                print(f"Target column '{target_col}' not found")
            return pd.DataFrame()
        
        # 2. 無効なSMILESを除外
        valid_smiles = []
        valid_indices = []
        
        for i, smiles in enumerate(processed_df[smiles_col]):
            if pd.isna(smiles):
                continue
            
            if self.preprocessor.validate_smiles(str(smiles)):
                valid_smiles.append(str(smiles))
                valid_indices.append(i)
            else:
                if self.verbose:
                    print(f"Invalid SMILES at index {i}: {smiles}")
        
        processed_df = processed_df.iloc[valid_indices].reset_index(drop=True)
        
        if self.verbose:
            print(f"Valid SMILES: {len(valid_smiles)}/{len(df)}")
        
        # 3. ターゲット値を数値に変換
        try:
            processed_df[target_col] = pd.to_numeric(processed_df[target_col], errors='coerce')
        except Exception as e:
            if self.verbose:
                print(f"Error converting target values: {e}")
            return pd.DataFrame()
        
        # 4. pIC50閾値でフィルタ
        if target_col in processed_df.columns:
            before_count = len(processed_df)
            processed_df = processed_df[processed_df[target_col] >= pIC50_threshold]
            after_count = len(processed_df)
            
            if self.verbose:
                print(f"Filtered by pIC50 threshold: {before_count} -> {after_count}")
        
        # 5. アウトライアを除外
        if remove_outliers and target_col in processed_df.columns:
            processed_df = self._remove_outliers(processed_df, target_col)
        
        # 6. 欠損値を補完
        if fill_missing:
            processed_df = self._fill_missing_values(processed_df)
        
        # 7. 骨格特徴量を追加
        if smiles_col in processed_df.columns:
            processed_df = self._add_scaffold_features(processed_df, smiles_col)
        
        # 8. ADMET特徴量を追加
        if smiles_col in processed_df.columns:
            processed_df = self._add_admet_features(processed_df, smiles_col)
        
        # 9. 統合特徴量を追加
        if smiles_col in processed_df.columns:
            processed_df = self._add_combined_features(processed_df, smiles_col)
        
        return processed_df
    
    def _remove_outliers(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """IQR法でアウトライアを除外"""
        if column not in df.columns:
            return df
        
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        
        if self.verbose:
            removed_count = len(df) - len(filtered_df)
            print(f"Removed {removed_count} outliers using IQR method")
        
        return filtered_df
    
    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """欠損値をモードで補完"""
        for column in df.columns:
            if df[column].dtype == 'object':
                # カテゴリカル変数はモードで補完
                mode_value = df[column].mode()
                if not mode_value.empty:
                    df[column] = df[column].fillna(mode_value[0])
            else:
                # 数値変数は中央値で補完
                median_value = df[column].median()
                df[column] = df[column].fillna(median_value)
        
        return df
    
    def _add_scaffold_features(self, df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
        """骨格特徴量を追加"""
        if smiles_col not in df.columns:
            return df
        
        scaffold_features = []
        
        for smiles in df[smiles_col]:
            try:
                features = self.preprocessor.preprocess(smiles)
                scaffold_feature = features.get('scaffold_features', np.zeros(20))
                scaffold_features.append(scaffold_feature)
            except Exception as e:
                if self.verbose:
                    print(f"Error processing SMILES {smiles}: {e}")
                scaffold_features.append(np.zeros(20))
        
        # 骨格特徴量をデータフレームに追加
        scaffold_df = pd.DataFrame(scaffold_features)
        scaffold_df.columns = [f'scaffold_{i}' for i in range(20)]
        
        return pd.concat([df, scaffold_df], axis=1)
    
    def _add_admet_features(self, df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
        """ADMET特徴量を追加"""
        if smiles_col not in df.columns:
            return df
        
        admet_features = []
        cns_mpo_scores = []
        
        for smiles in df[smiles_col]:
            try:
                features = self.preprocessor.preprocess(smiles)
                admet_feature = features.get('admet_features', np.zeros(10))
                cns_mpo = features.get('cns_mpo', np.array([0.0]))[0]
                
                admet_features.append(admet_feature)
                cns_mpo_scores.append(cns_mpo)
            except Exception as e:
                if self.verbose:
                    print(f"Error processing SMILES {smiles}: {e}")
                admet_features.append(np.zeros(10))
                cns_mpo_scores.append(0.0)
        
        # ADMET特徴量をデータフレームに追加
        admet_df = pd.DataFrame(admet_features)
        admet_df.columns = [f'admet_{i}' for i in range(10)]
        admet_df['cns_mpo'] = cns_mpo_scores
        
        return pd.concat([df, admet_df], axis=1)
    
    def _add_combined_features(self, df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
        """統合特徴量を追加"""
        if smiles_col not in df.columns:
            return df
        
        combined_features = []
        
        for smiles in df[smiles_col]:
            try:
                features = self.preprocessor.extract_combined_features(smiles)
                combined_features.append(features)
            except Exception as e:
                if self.verbose:
                    print(f"Error processing SMILES {smiles}: {e}")
                combined_features.append(np.zeros(self.preprocessor.get_total_dimensions()))
        
        # 統合特徴量をデータフレームに追加
        combined_df = pd.DataFrame(combined_features)
        combined_df.columns = [f'feature_{i}' for i in range(len(combined_features[0]))]
        
        return pd.concat([df, combined_df], axis=1)
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """前処理済みデータを保存"""
        try:
            df.to_csv(output_path, index=False)
            if self.verbose:
                print(f"Processed data saved to: {output_path}")
        except Exception as e:
            if self.verbose:
                print(f"Error saving data: {e}")
    
    def load_processed_data(self, input_path: str) -> pd.DataFrame:
        """前処理済みデータを読み込み"""
        try:
            df = pd.read_csv(input_path)
            if self.verbose:
                print(f"Processed data loaded from: {input_path}")
            return df
        except Exception as e:
            if self.verbose:
                print(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """データサマリーを取得"""
        if df.empty:
            return {'total_records': 0}
        
        summary = {
            'total_records': len(df),
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict()
        }
        
        # 数値列の統計情報を追加
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            summary[f'{col}_stats'] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median()
            }
        
        return summary
    
    def validate_data(self, df: pd.DataFrame, smiles_col: str, target_col: str) -> Dict[str, Any]:
        """データの妥当性を検証"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # 1. 必要な列が存在するかチェック
        if smiles_col not in df.columns:
            validation_results['valid'] = False
            validation_results['errors'].append(f"SMILES column '{smiles_col}' not found")
        
        if target_col not in df.columns:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Target column '{target_col}' not found")
        
        # 2. SMILESの妥当性をチェック
        if smiles_col in df.columns:
            invalid_smiles = 0
            for i, smiles in enumerate(df[smiles_col]):
                if pd.isna(smiles):
                    continue
                if not self.preprocessor.validate_smiles(str(smiles)):
                    invalid_smiles += 1
            
            if invalid_smiles > 0:
                validation_results['warnings'].append(f"{invalid_smiles} invalid SMILES found")
        
        # 3. ターゲット値の妥当性をチェック
        if target_col in df.columns:
            try:
                target_values = pd.to_numeric(df[target_col], errors='coerce')
                nan_count = target_values.isna().sum()
                if nan_count > 0:
                    validation_results['warnings'].append(f"{nan_count} non-numeric target values found")
            except Exception as e:
                validation_results['errors'].append(f"Error validating target values: {e}")
        
        return validation_results


# 便利関数
def load_csv_data(
    path: str,
    smiles_col: str = 'smiles',
    target_col: str = 'pIC50',
    **kwargs
) -> pd.DataFrame:
    """CSVデータを読み込み"""
    loader = CustomDataLoader(**kwargs)
    return loader.load_csv(path, smiles_col, target_col)


def load_sdf_data(
    path: str,
    target_col: str = 'pIC50',
    **kwargs
) -> pd.DataFrame:
    """SDFデータを読み込み"""
    loader = CustomDataLoader(**kwargs)
    return loader.load_sdf(path, target_col)


def load_excel_data(
    path: str,
    smiles_col: str = 'smiles',
    target_col: str = 'pIC50',
    **kwargs
) -> pd.DataFrame:
    """Excelデータを読み込み"""
    loader = CustomDataLoader(**kwargs)
    return loader.load_excel(path, smiles_col=smiles_col, target_col=target_col)


if __name__ == "__main__":
    # テスト実行
    print("🧬 カスタムデータローダーテスト")
    print("=" * 50)
    
    # テスト用データを作成
    test_data = pd.DataFrame({
        'smiles': [
            'CC(CC1=CC=CC=C1)N',  # アンフェタミン
            'CCN(CC)CC1=CC2=C(C=C1)OCO2',  # MDMA
            'CCN(CC)CC1=CNC2=CC=CC=C21',  # DMT
            'CN1CC[C@]23C4=C5C6=C(C=CC=C6)O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5=C4O',  # モルヒネ
            'CCCCCC1=CC(=C(C(=C1)O)C2C=C(CCC2C(C)C)C)C',  # THC
        ],
        'pIC50': [7.2, 6.8, 8.1, 7.5, 6.3],
        'target': ['DAT', 'DAT', '5HT2A', 'MOR', 'CB1']
    })
    
    # テスト用CSVファイルを作成
    test_csv_path = 'test_data.csv'
    test_data.to_csv(test_csv_path, index=False)
    
    # データローダーを初期化
    loader = CustomDataLoader(verbose=True)
    
    # CSVデータを読み込み
    print("\n📋 CSVデータ読み込みテスト:")
    data = loader.load_csv(test_csv_path, smiles_col='smiles', target_col='pIC50')
    
    if not data.empty:
        print(f"  Loaded {len(data)} records")
        print(f"  Columns: {list(data.columns)}")
        
        if 'pIC50' in data.columns:
            print(f"  pIC50 range: {data['pIC50'].min():.2f} - {data['pIC50'].max():.2f}")
        
        if 'cns_mpo' in data.columns:
            print(f"  CNS-MPO range: {data['cns_mpo'].min():.3f} - {data['cns_mpo'].max():.3f}")
        
        # データサマリーを取得
        summary = loader.get_data_summary(data)
        print(f"  Data summary: {summary}")
        
        # データの妥当性を検証
        validation = loader.validate_data(data, 'smiles', 'pIC50')
        print(f"  Validation: {validation}")
    else:
        print("  No data loaded")
    
    # テスト用CSVファイルを削除
    import os
    if os.path.exists(test_csv_path):
        os.remove(test_csv_path)
    
    print("\n✅ カスタムデータローダーテスト完了！")
