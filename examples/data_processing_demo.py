"""
Data Processing Demo

ChemForgeデータ処理モジュールのデモンストレーション
ChEMBLデータ取得・分子特徴量抽出・前処理の統合例
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import time
import logging

from chemforge.data.chembl_loader import ChEMBLLoader
from chemforge.data.molecular_features import MolecularFeatures
from chemforge.data.rdkit_descriptors import RDKitDescriptors
from chemforge.data.data_preprocessor import DataPreprocessor

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_synthetic_data(num_compounds: int = 1000) -> pd.DataFrame:
    """
    合成データを作成（デモ用）
    
    Args:
        num_compounds: 化合物数
    
    Returns:
        合成データフレーム
    """
    logger.info(f"Creating synthetic data with {num_compounds} compounds")
    
    # 合成SMILES
    smiles_list = [
        "CCO", "CCN", "CCC", "CCCC", "CCCCC",
        "C1=CC=CC=C1", "C1=CC=CC=C1O", "C1=CC=CC=C1N",
        "C1=CC=CC=C1C", "C1=CC=CC=C1CC",
        "C1=CC=CC=C1CCC", "C1=CC=CC=C1CCCC",
        "C1=CC=CC=C1CCCCC", "C1=CC=CC=C1CCCCCC",
        "C1=CC=CC=C1CCCCCCC", "C1=CC=CC=C1CCCCCCCC"
    ] * (num_compounds // 16 + 1)
    smiles_list = smiles_list[:num_compounds]
    
    # 合成pIC50値
    np.random.seed(42)
    data = {
        'molecule_chembl_id': [f'CHEMBL{i:06d}' for i in range(num_compounds)],
        'canonical_smiles': smiles_list,
        '5HT2A_pIC50': np.random.normal(7.5, 1.0, num_compounds),
        'D1_pIC50': np.random.normal(7.0, 1.0, num_compounds),
        'CB1_pIC50': np.random.normal(6.5, 1.0, num_compounds),
        'MOR_pIC50': np.random.normal(7.2, 1.0, num_compounds)
    }
    
    df = pd.DataFrame(data)
    
    # pIC50値を6-10の範囲にクランプ
    for col in ['5HT2A_pIC50', 'D1_pIC50', 'CB1_pIC50', 'MOR_pIC50']:
        df[col] = np.clip(df[col], 6.0, 10.0)
    
    logger.info(f"Created synthetic dataset with {len(df)} compounds")
    return df


def demonstrate_chembl_loader():
    """ChEMBLローダーのデモンストレーション"""
    print("\n" + "="*60)
    print("🔬 ChEMBL Data Loader Demo")
    print("="*60)
    
    # ChEMBLローダーを初期化
    loader = ChEMBLLoader(cache_dir="./data/cache")
    
    # 合成データを作成（実際のChEMBL APIの代わり）
    synthetic_data = create_synthetic_data(500)
    
    # データセット要約を表示
    summary = loader.get_dataset_summary(synthetic_data)
    print(f"\n📊 Dataset Summary:")
    print(f"  Total compounds: {summary['total_compounds']}")
    print(f"  Targets: {summary['targets']}")
    print(f"  Missing values: {sum(summary['missing_values'].values())}")
    
    # ターゲット別統計
    print(f"\n🎯 Target Statistics:")
    for target, stats in summary['target_statistics'].items():
        print(f"  {target}:")
        print(f"    Count: {stats['count']}")
        print(f"    Mean: {stats['mean']:.3f}")
        print(f"    Std: {stats['std']:.3f}")
        print(f"    Range: {stats['min']:.3f} - {stats['max']:.3f}")
    
    return synthetic_data


def demonstrate_molecular_features():
    """分子特徴量抽出のデモンストレーション"""
    print("\n" + "="*60)
    print("🧬 Molecular Features Demo")
    print("="*60)
    
    # 分子特徴量抽出器を初期化
    features = MolecularFeatures()
    
    # テスト分子
    test_smiles = [
        "CCO",  # エタノール
        "C1=CC=CC=C1",  # ベンゼン
        "C1=CC=CC=C1O",  # フェノール
        "C1=CC=CC=C1N",  # アニリン
        "C1=CC=CC=C1C(=O)O"  # 安息香酸
    ]
    
    print(f"\n🔍 Processing {len(test_smiles)} test molecules...")
    
    # 2D記述子を抽出
    print("\n📋 2D Descriptors:")
    for smiles in test_smiles:
        descriptors = features.extract_2d_descriptors(smiles)
        print(f"  {smiles}:")
        print(f"    MolWt: {descriptors.get('MolWt', 0):.2f}")
        print(f"    LogP: {descriptors.get('LogP', 0):.2f}")
        print(f"    TPSA: {descriptors.get('TPSA', 0):.2f}")
        print(f"    NumRings: {descriptors.get('NumRings', 0)}")
    
    # フィンガープリントを抽出
    print("\n🔢 Fingerprints:")
    for smiles in test_smiles:
        morgan_fp = features.extract_fingerprints(smiles, "morgan")
        rdkit_fp = features.extract_fingerprints(smiles, "rdkit")
        print(f"  {smiles}:")
        print(f"    Morgan FP: {morgan_fp.sum()} bits set")
        print(f"    RDKit FP: {rdkit_fp.sum()} bits set")
    
    # 骨格特徴量を抽出
    print("\n🏗️ Scaffold Features:")
    for smiles in test_smiles:
        scaffold_features = features.extract_scaffold_features(smiles)
        print(f"  {smiles}:")
        print(f"    Scaffold: {scaffold_features.get('scaffold_smiles', 'N/A')}")
        print(f"    Scaffold atoms: {scaffold_features.get('scaffold_atoms', 0)}")
        print(f"    Scaffold rings: {scaffold_features.get('scaffold_rings', 0)}")
    
    return features


def demonstrate_rdkit_descriptors():
    """RDKit記述子のデモンストレーション"""
    print("\n" + "="*60)
    print("⚗️ RDKit Descriptors Demo")
    print("="*60)
    
    # RDKit記述子抽出器を初期化
    descriptors = RDKitDescriptors()
    
    # テスト分子
    test_smiles = [
        "CCO",  # エタノール
        "C1=CC=CC=C1",  # ベンゼン
        "C1=CC=CC=C1O",  # フェノール
        "C1=CC=CC=C1N",  # アニリン
        "C1=CC=CC=C1C(=O)O"  # 安息香酸
    ]
    
    print(f"\n🔍 Processing {len(test_smiles)} test molecules...")
    
    # 基本記述子を計算
    print("\n📊 Basic Descriptors:")
    for smiles in test_smiles:
        basic_desc = descriptors.calculate_basic_descriptors(
            descriptors._get_descriptor_functions()['MolWt'].__self__.MolFromSmiles(smiles)
        )
        print(f"  {smiles}:")
        print(f"    MolWt: {basic_desc.get('MolWt', 0):.2f}")
        print(f"    LogP: {basic_desc.get('LogP', 0):.2f}")
        print(f"    TPSA: {basic_desc.get('TPSA', 0):.2f}")
        print(f"    NumRings: {basic_desc.get('NumRings', 0)}")
    
    # Lipinski's Ruleを計算
    print("\n📏 Lipinski's Rule of Five:")
    for smiles in test_smiles:
        lipinski = descriptors.calculate_lipinski_rule(
            descriptors._get_descriptor_functions()['MolWt'].__self__.MolFromSmiles(smiles)
        )
        print(f"  {smiles}:")
        print(f"    MolWt: {lipinski.get('MolWt', 0):.2f}")
        print(f"    LogP: {lipinski.get('LogP', 0):.2f}")
        print(f"    HBD: {lipinski.get('NumHDonors', 0)}")
        print(f"    HBA: {lipinski.get('NumHAcceptors', 0)}")
        print(f"    Violations: {lipinski.get('LipinskiViolations', 0)}")
        print(f"    Compliant: {lipinski.get('LipinskiCompliant', False)}")
    
    return descriptors


def demonstrate_data_preprocessing():
    """データ前処理のデモンストレーション"""
    print("\n" + "="*60)
    print("🔧 Data Preprocessing Demo")
    print("="*60)
    
    # データ前処理器を初期化
    preprocessor = DataPreprocessor()
    
    # 合成データを作成
    synthetic_data = create_synthetic_data(1000)
    
    # 前処理設定
    preprocessing_config = {
        "clean_data": True,
        "handle_outliers": True,
        "impute_missing": True,
        "normalize_features": True,
        "min_pic50": 6.0,
        "max_pic50": 10.0,
        "outlier_method": "iqr",
        "impute_strategy": "median",
        "normalization_method": "standard"
    }
    
    # 特徴量列とターゲット列を定義
    feature_columns = ['MolWt', 'LogP', 'TPSA', 'NumRings']
    target_columns = ['5HT2A_pIC50', 'D1_pIC50', 'CB1_pIC50', 'MOR_pIC50']
    
    # 合成特徴量を追加
    np.random.seed(42)
    for col in feature_columns:
        synthetic_data[col] = np.random.normal(100, 20, len(synthetic_data))
    
    print(f"\n📊 Original Dataset:")
    print(f"  Size: {len(synthetic_data)}")
    print(f"  Features: {len(feature_columns)}")
    print(f"  Targets: {len(target_columns)}")
    
    # データ前処理を実行
    processed_data, preprocessing_info = preprocessor.preprocess_dataset(
        synthetic_data,
        target_columns,
        feature_columns,
        preprocessing_config
    )
    
    print(f"\n🔧 Preprocessing Results:")
    print(f"  Original size: {preprocessing_info['original_size']}")
    print(f"  Final size: {preprocessing_info['final_size']}")
    print(f"  Reduction rate: {preprocessing_info['reduction_rate']:.2%}")
    print(f"  Steps: {preprocessing_info['steps']}")
    
    # 前処理要約を取得
    summary = preprocessor.get_preprocessing_summary(preprocessing_info)
    print(f"\n📈 Preprocessing Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    return processed_data, preprocessing_info


def demonstrate_integrated_pipeline():
    """統合パイプラインのデモンストレーション"""
    print("\n" + "="*60)
    print("🚀 Integrated Data Processing Pipeline")
    print("="*60)
    
    # 1. データ取得
    print("\n1️⃣ Data Acquisition:")
    synthetic_data = create_synthetic_data(500)
    print(f"   Retrieved {len(synthetic_data)} compounds")
    
    # 2. 分子特徴量抽出
    print("\n2️⃣ Molecular Feature Extraction:")
    features = MolecularFeatures()
    
    # サンプル分子の特徴量を抽出
    sample_smiles = synthetic_data['canonical_smiles'].head(10).tolist()
    feature_df = features.process_molecule_batch(
        sample_smiles,
        include_3d=False,
        include_fingerprints=True,
        include_scaffolds=True
    )
    
    print(f"   Extracted features for {len(feature_df)} molecules")
    print(f"   Feature columns: {len(feature_df.columns)}")
    
    # 3. データ前処理
    print("\n3️⃣ Data Preprocessing:")
    preprocessor = DataPreprocessor()
    
    # 特徴量列を定義
    feature_columns = [col for col in feature_df.columns if col != 'smiles']
    target_columns = ['5HT2A_pIC50', 'D1_pIC50', 'CB1_pIC50', 'MOR_pIC50']
    
    # データを統合
    integrated_data = synthetic_data.merge(
        feature_df, 
        left_on='canonical_smiles', 
        right_on='smiles', 
        how='inner'
    )
    
    # 前処理設定
    preprocessing_config = {
        "clean_data": True,
        "handle_outliers": True,
        "impute_missing": True,
        "normalize_features": True,
        "min_pic50": 6.0,
        "max_pic50": 10.0,
        "outlier_method": "iqr",
        "impute_strategy": "median",
        "normalization_method": "standard"
    }
    
    # 前処理を実行
    processed_data, preprocessing_info = preprocessor.preprocess_dataset(
        integrated_data,
        target_columns,
        feature_columns,
        preprocessing_config
    )
    
    print(f"   Processed {len(processed_data)} molecules")
    print(f"   Final features: {len(feature_columns)}")
    
    # 4. 結果可視化
    print("\n4️⃣ Results Visualization:")
    
    # ターゲット分布をプロット
    plt.figure(figsize=(15, 10))
    
    # ターゲット別分布
    for i, target in enumerate(target_columns, 1):
        plt.subplot(2, 2, i)
        plt.hist(processed_data[target], bins=20, alpha=0.7, edgecolor='black')
        plt.title(f'{target} Distribution')
        plt.xlabel('pIC50')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('target_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 特徴量相関をプロット
    plt.figure(figsize=(12, 8))
    correlation_matrix = processed_data[target_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Target Correlation Matrix')
    plt.tight_layout()
    plt.savefig('target_correlations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Integrated pipeline completed successfully!")
    print(f"   Final dataset: {len(processed_data)} compounds")
    print(f"   Features: {len(feature_columns)}")
    print(f"   Targets: {len(target_columns)}")
    
    return processed_data


def main():
    """
    メイン実行関数
    """
    print("🧬 ChemForge Data Processing Demo")
    print("="*60)
    
    try:
        # 1. ChEMBLローダーデモ
        synthetic_data = demonstrate_chembl_loader()
        
        # 2. 分子特徴量抽出デモ
        features = demonstrate_molecular_features()
        
        # 3. RDKit記述子デモ
        descriptors = demonstrate_rdkit_descriptors()
        
        # 4. データ前処理デモ
        processed_data, preprocessing_info = demonstrate_data_preprocessing()
        
        # 5. 統合パイプラインデモ
        final_data = demonstrate_integrated_pipeline()
        
        print("\n🎉 All demonstrations completed successfully!")
        print("ChemForge data processing modules are ready for use!")
        
    except Exception as e:
        logger.error(f"Error in demo: {str(e)}")
        raise


if __name__ == "__main__":
    main()
