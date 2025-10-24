"""
Data Distribution Module

データ配布モジュール
ChEMBLデータを活用した効率的なデータ配布システム
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
from chemforge.data.chembl_loader import ChEMBLLoader
from chemforge.data.molecular_features import MolecularFeatures
from chemforge.data.rdkit_descriptors import RDKitDescriptors
from chemforge.integration.database_integration import DatabaseIntegration
from chemforge.utils.config_utils import ConfigManager
from chemforge.utils.logging_utils import Logger
from chemforge.utils.validation import DataValidator

logger = logging.getLogger(__name__)

class DataDistribution:
    """
    データ配布クラス
    
    ChEMBLデータを活用した効率的なデータ配布システム
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
        self.logger = Logger("DataDistribution")
        self.validator = DataValidator()
        
        # データローダー
        self.chembl_loader = ChEMBLLoader(config_path, cache_dir)
        self.molecular_features = MolecularFeatures(config_path, cache_dir)
        self.rdkit_descriptors = RDKitDescriptors(config_path, cache_dir)
        self.db_integration = DatabaseIntegration(config_path, cache_dir)
        
        # データ配布設定
        self.distribution_config = self.config.get('data_distribution', {})
        self.target_chembl_ids = self.distribution_config.get('target_chembl_ids', [])
        self.data_splits = self.distribution_config.get('data_splits', {'train': 0.8, 'val': 0.1, 'test': 0.1})
        self.include_features = self.distribution_config.get('include_features', True)
        self.include_descriptors = self.distribution_config.get('include_descriptors', True)
        
        logger.info("DataDistribution initialized")
    
    def create_distribution_dataset(self, target_chembl_ids: Optional[List[str]] = None,
                                   include_features: bool = True,
                                   include_descriptors: bool = True) -> pd.DataFrame:
        """
        配布データセット作成
        
        Args:
            target_chembl_ids: ターゲットChEMBL IDリスト
            include_features: 分子特徴量含むフラグ
            include_descriptors: RDKit記述子含むフラグ
            
        Returns:
            配布データセット
        """
        if target_chembl_ids is None:
            target_chembl_ids = self.target_chembl_ids
        
        logger.info(f"Creating distribution dataset for {len(target_chembl_ids)} targets")
        
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
        
        # データセット保存
        dataset_path = self.cache_dir / "distribution_dataset.csv"
        chembl_data.to_csv(dataset_path, index=False)
        logger.info(f"Distribution dataset saved to: {dataset_path}")
        
        return chembl_data
    
    def create_data_splits(self, data: pd.DataFrame,
                          split_method: str = 'scaffold',
                          random_seed: int = 42) -> Dict[str, pd.DataFrame]:
        """
        データ分割作成
        
        Args:
            data: データフレーム
            split_method: 分割方法
            random_seed: 乱数シード
            
        Returns:
            分割データ辞書
        """
        logger.info(f"Creating data splits using {split_method} method")
        
        if split_method == 'scaffold':
            splits = self._create_scaffold_splits(data, random_seed)
        elif split_method == 'random':
            splits = self._create_random_splits(data, random_seed)
        elif split_method == 'time':
            splits = self._create_time_splits(data)
        else:
            raise ValueError(f"Unsupported split method: {split_method}")
        
        # 分割データ保存
        for split_name, split_data in splits.items():
            split_path = self.cache_dir / f"{split_name}_split.csv"
            split_data.to_csv(split_path, index=False)
            logger.info(f"{split_name} split saved to: {split_path} ({len(split_data)} samples)")
        
        return splits
    
    def _create_scaffold_splits(self, data: pd.DataFrame, random_seed: int) -> Dict[str, pd.DataFrame]:
        """
        スキャフォールド分割作成
        
        Args:
            data: データフレーム
            random_seed: 乱数シード
            
        Returns:
            分割データ辞書
        """
        from chemforge.data.scaffold_detector import ScaffoldDetector
        
        scaffold_detector = ScaffoldDetector()
        
        # スキャフォールド計算
        scaffolds = []
        for smiles in tqdm(data['smiles'], desc="Calculating scaffolds"):
            scaffold = scaffold_detector.get_scaffold(smiles)
            scaffolds.append(scaffold)
        
        data['scaffold'] = scaffolds
        
        # スキャフォールドごとにグループ化
        scaffold_groups = data.groupby('scaffold')
        scaffold_sizes = scaffold_groups.size()
        
        # スキャフォールドをサイズ順にソート
        sorted_scaffolds = scaffold_sizes.sort_values(ascending=False)
        
        # 分割
        train_scaffolds = []
        val_scaffolds = []
        test_scaffolds = []
        
        train_size = 0
        val_size = 0
        test_size = 0
        
        total_samples = len(data)
        target_train_size = int(total_samples * self.data_splits['train'])
        target_val_size = int(total_samples * self.data_splits['val'])
        target_test_size = int(total_samples * self.data_splits['test'])
        
        for scaffold, size in sorted_scaffolds.items():
            if train_size < target_train_size:
                train_scaffolds.append(scaffold)
                train_size += size
            elif val_size < target_val_size:
                val_scaffolds.append(scaffold)
                val_size += size
            else:
                test_scaffolds.append(scaffold)
                test_size += size
        
        # 分割データ作成
        train_data = data[data['scaffold'].isin(train_scaffolds)]
        val_data = data[data['scaffold'].isin(val_scaffolds)]
        test_data = data[data['scaffold'].isin(test_scaffolds)]
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
    
    def _create_random_splits(self, data: pd.DataFrame, random_seed: int) -> Dict[str, pd.DataFrame]:
        """
        ランダム分割作成
        
        Args:
            data: データフレーム
            random_seed: 乱数シード
            
        Returns:
            分割データ辞書
        """
        np.random.seed(random_seed)
        
        # ランダムインデックス生成
        indices = np.random.permutation(len(data))
        
        # 分割サイズ計算
        train_size = int(len(data) * self.data_splits['train'])
        val_size = int(len(data) * self.data_splits['val'])
        
        # インデックス分割
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # 分割データ作成
        train_data = data.iloc[train_indices]
        val_data = data.iloc[val_indices]
        test_data = data.iloc[test_indices]
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
    
    def _create_time_splits(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        時間分割作成
        
        Args:
            data: データフレーム
            
        Returns:
            分割データ辞書
        """
        if 'date' not in data.columns:
            logger.warning("Date column not found, using random splits instead")
            return self._create_random_splits(data, 42)
        
        # 日付でソート
        data_sorted = data.sort_values('date')
        
        # 分割サイズ計算
        train_size = int(len(data_sorted) * self.data_splits['train'])
        val_size = int(len(data_sorted) * self.data_splits['val'])
        
        # 時間順分割
        train_data = data_sorted.iloc[:train_size]
        val_data = data_sorted.iloc[train_size:train_size + val_size]
        test_data = data_sorted.iloc[train_size + val_size:]
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
    
    def create_metadata(self, data: pd.DataFrame,
                       splits: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        メタデータ作成
        
        Args:
            data: データフレーム
            splits: 分割データ辞書
            
        Returns:
            メタデータ辞書
        """
        logger.info("Creating metadata")
        
        metadata = {
            'dataset_info': {
                'total_samples': len(data),
                'total_features': len(data.columns),
                'target_chembl_ids': self.target_chembl_ids,
                'include_features': self.include_features,
                'include_descriptors': self.include_descriptors,
                'created_at': time.time()
            },
            'data_splits': {
                'method': 'scaffold',
                'splits': {
                    'train': len(splits['train']),
                    'val': len(splits['val']),
                    'test': len(splits['test'])
                },
                'ratios': {
                    'train': len(splits['train']) / len(data),
                    'val': len(splits['val']) / len(data),
                    'test': len(splits['test']) / len(data)
                }
            },
            'feature_info': {
                'numeric_features': len(data.select_dtypes(include=[np.number]).columns),
                'categorical_features': len(data.select_dtypes(include=['object']).columns),
                'missing_values': data.isnull().sum().to_dict()
            },
            'target_info': {
                'unique_targets': data['target_id'].nunique() if 'target_id' in data.columns else 0,
                'activity_types': data['type'].value_counts().to_dict() if 'type' in data.columns else {},
                'p_value_range': {
                    'min': data['p_value'].min() if 'p_value' in data.columns else None,
                    'max': data['p_value'].max() if 'p_value' in data.columns else None,
                    'mean': data['p_value'].mean() if 'p_value' in data.columns else None
                }
            }
        }
        
        # メタデータ保存
        metadata_path = self.cache_dir / "distribution_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to: {metadata_path}")
        return metadata
    
    def export_distribution_package(self, output_dir: str,
                                   include_splits: bool = True,
                                   include_metadata: bool = True) -> bool:
        """
        配布パッケージエクスポート
        
        Args:
            output_dir: 出力ディレクトリ
            include_splits: 分割データ含むフラグ
            include_metadata: メタデータ含むフラグ
            
        Returns:
            成功フラグ
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # メインデータセットコピー
            main_dataset_path = self.cache_dir / "distribution_dataset.csv"
            if main_dataset_path.exists():
                import shutil
                shutil.copy2(main_dataset_path, output_dir / "dataset.csv")
            
            # 分割データコピー
            if include_splits:
                for split_name in ['train', 'val', 'test']:
                    split_path = self.cache_dir / f"{split_name}_split.csv"
                    if split_path.exists():
                        shutil.copy2(split_path, output_dir / f"{split_name}.csv")
            
            # メタデータコピー
            if include_metadata:
                metadata_path = self.cache_dir / "distribution_metadata.json"
                if metadata_path.exists():
                    shutil.copy2(metadata_path, output_dir / "metadata.json")
            
            # README作成
            readme_path = output_dir / "README.md"
            self._create_readme(readme_path)
            
            logger.info(f"Distribution package exported to: {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting distribution package: {e}")
            return False
    
    def _create_readme(self, readme_path: Path):
        """
        README作成
        
        Args:
            readme_path: READMEパス
        """
        readme_content = """# ChemForge Distribution Dataset

## Overview
This dataset contains molecular data from ChEMBL with molecular features and RDKit descriptors for CNS drug discovery.

## Files
- `dataset.csv`: Complete dataset
- `train.csv`: Training split
- `val.csv`: Validation split  
- `test.csv`: Test split
- `metadata.json`: Dataset metadata

## Usage
```python
import pandas as pd

# Load dataset
data = pd.read_csv('dataset.csv')

# Load splits
train_data = pd.read_csv('train.csv')
val_data = pd.read_csv('val.csv')
test_data = pd.read_csv('test.csv')
```

## Features
- Molecular SMILES
- ChEMBL activity data
- Molecular features (MW, LogP, TPSA, etc.)
- RDKit descriptors (Morgan fingerprints, MACCS keys, 2D descriptors)
- 3D molecular features

## Targets
- AMPA receptor (CHEMBL4205)
- NMDA receptor (CHEMBL240)
- GABA-A receptor (CHEMBL2093872)
- GABA-B receptor (CHEMBL2093873)

## Citation
Please cite ChemForge when using this dataset.
"""
        
        with open(readme_path, 'w') as f:
            f.write(readme_content)

def create_data_distribution(config_path: Optional[str] = None, 
                           cache_dir: str = "cache") -> DataDistribution:
    """
    データ配布作成
    
    Args:
        config_path: 設定ファイルパス
        cache_dir: キャッシュディレクトリ
        
    Returns:
        DataDistribution
    """
    return DataDistribution(config_path, cache_dir)

if __name__ == "__main__":
    # テスト実行
    data_distribution = DataDistribution()
    
    print(f"DataDistribution created: {data_distribution}")
    print(f"Cache directory: {data_distribution.cache_dir}")
    print(f"Target ChEMBL IDs: {data_distribution.target_chembl_ids}")
    print(f"Data splits: {data_distribution.data_splits}")
    print(f"Include features: {data_distribution.include_features}")
    print(f"Include descriptors: {data_distribution.include_descriptors}")
