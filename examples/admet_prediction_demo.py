"""
ADMET Prediction Demo

ChemForge ADMET予測システムのデモンストレーション
包括的なADMET評価・最適化指標の実装例
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import time
import logging

from chemforge.admet.admet_predictor import ADMETPredictor
from chemforge.admet.property_predictor import PropertyPredictor
from chemforge.admet.toxicity_predictor import ToxicityPredictor
from chemforge.admet.drug_likeness import DrugLikenessPredictor
from chemforge.admet.cns_mpo import CNSMPOCalculator

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_molecules() -> List[str]:
    """
    テスト用分子リストを作成
    
    Returns:
        SMILES文字列リスト
    """
    test_molecules = [
        "CCO",  # エタノール
        "CCCO",  # プロパノール
        "CCCCCO",  # ペンタノール
        "c1ccccc1",  # ベンゼン
        "c1ccccc1O",  # フェノール
        "CC(=O)O",  # 酢酸
        "CCN(CC)CC",  # トリエチルアミン
        "c1ccc2c(c1)cccc2",  # ナフタレン
        "CC(C)CO",  # イソブタノール
        "CC(C)(C)CO"  # ネオペンチルアルコール
    ]
    
    logger.info(f"Created {len(test_molecules)} test molecules")
    return test_molecules


def demonstrate_admet_prediction():
    """ADMET予測のデモンストレーション"""
    print("\n" + "="*60)
    print("🧬 ADMET Prediction Demo")
    print("="*60)
    
    # テスト分子
    test_molecules = create_test_molecules()
    
    # ADMET予測器を初期化
    admet_predictor = ADMETPredictor()
    
    print("\n🔹 Comprehensive ADMET Prediction:")
    
    # 包括的ADMET予測
    for i, smiles in enumerate(test_molecules[:3]):  # 最初の3分子のみ
        print(f"\n  Molecule {i+1}: {smiles}")
        
        admet_results = admet_predictor.predict_comprehensive_admet(smiles)
        
        for category, properties in admet_results.items():
            if properties:
                print(f"    {category.upper()}:")
                for prop_name, prop_value in properties.items():
                    if isinstance(prop_value, float):
                        print(f"      {prop_name}: {prop_value:.3f}")
                    else:
                        print(f"      {prop_name}: {prop_value}")
    
    # バッチ処理
    print(f"\n🔹 Batch Processing {len(test_molecules)} molecules:")
    start_time = time.time()
    
    df = admet_predictor.process_molecule_batch(test_molecules)
    
    processing_time = time.time() - start_time
    print(f"  Processing time: {processing_time:.2f}s")
    print(f"  Processed {len(df)} molecules successfully")
    print(f"  Columns: {len(df.columns)}")
    
    # ADMET要約
    admet_summary = admet_predictor.get_admet_summary(admet_results)
    print(f"\n  ADMET Summary:")
    for key, value in admet_summary.items():
        print(f"    {key}: {value}")
    
    return df


def demonstrate_property_prediction():
    """物性予測のデモンストレーション"""
    print("\n" + "="*60)
    print("📊 Property Prediction Demo")
    print("="*60)
    
    # テスト分子
    test_molecules = create_test_molecules()
    
    # 物性予測器を初期化
    property_predictor = PropertyPredictor()
    
    print("\n🔹 Comprehensive Property Prediction:")
    
    # 包括的物性予測
    for i, smiles in enumerate(test_molecules[:3]):  # 最初の3分子のみ
        print(f"\n  Molecule {i+1}: {smiles}")
        
        properties = property_predictor.predict_comprehensive_properties(smiles)
        
        # 主要な物性を表示
        key_properties = ['molecular_weight', 'logp', 'tpsa', 'violations', 'compliant', 'qed', 'sa_score']
        for prop in key_properties:
            if prop in properties:
                value = properties[prop]
                if isinstance(value, float):
                    print(f"    {prop}: {value:.3f}")
                else:
                    print(f"    {prop}: {value}")
    
    # バッチ処理
    print(f"\n🔹 Batch Processing {len(test_molecules)} molecules:")
    start_time = time.time()
    
    df = property_predictor.process_molecule_batch(test_molecules)
    
    processing_time = time.time() - start_time
    print(f"  Processing time: {processing_time:.2f}s")
    print(f"  Processed {len(df)} molecules successfully")
    print(f"  Columns: {len(df.columns)}")
    
    # 物性要約
    property_summary = property_predictor.get_property_summary(df)
    print(f"\n  Property Summary:")
    print(f"    Total molecules: {property_summary['total_molecules']}")
    print(f"    Property columns: {len(property_summary['property_columns'])}")
    
    return df


def demonstrate_toxicity_prediction():
    """毒性予測のデモンストレーション"""
    print("\n" + "="*60)
    print("☠️ Toxicity Prediction Demo")
    print("="*60)
    
    # テスト分子
    test_molecules = create_test_molecules()
    
    # 毒性予測器を初期化
    toxicity_predictor = ToxicityPredictor()
    
    print("\n🔹 Comprehensive Toxicity Prediction:")
    
    # 包括的毒性予測
    for i, smiles in enumerate(test_molecules[:3]):  # 最初の3分子のみ
        print(f"\n  Molecule {i+1}: {smiles}")
        
        toxicity_results = toxicity_predictor.predict_comprehensive_toxicity(smiles)
        
        for category, properties in toxicity_results.items():
            if properties:
                print(f"    {category.upper()}:")
                for prop_name, prop_value in properties.items():
                    if isinstance(prop_value, float):
                        print(f"      {prop_name}: {prop_value:.3f}")
                    else:
                        print(f"      {prop_name}: {prop_value}")
    
    # バッチ処理
    print(f"\n🔹 Batch Processing {len(test_molecules)} molecules:")
    start_time = time.time()
    
    df = toxicity_predictor.process_molecule_batch(test_molecules)
    
    processing_time = time.time() - start_time
    print(f"  Processing time: {processing_time:.2f}s")
    print(f"  Processed {len(df)} molecules successfully")
    print(f"  Columns: {len(df.columns)}")
    
    # 毒性要約
    toxicity_summary = toxicity_predictor.get_toxicity_summary(df)
    print(f"\n  Toxicity Summary:")
    print(f"    Total molecules: {toxicity_summary['total_molecules']}")
    print(f"    High risk molecules: {toxicity_summary['high_risk_molecules']}")
    print(f"    Medium risk molecules: {toxicity_summary['medium_risk_molecules']}")
    print(f"    Low risk molecules: {toxicity_summary['low_risk_molecules']}")
    
    return df


def demonstrate_drug_likeness_prediction():
    """薬物らしさ予測のデモンストレーション"""
    print("\n" + "="*60)
    print("💊 Drug Likeness Prediction Demo")
    print("="*60)
    
    # テスト分子
    test_molecules = create_test_molecules()
    
    # 薬物らしさ予測器を初期化
    drug_likeness_predictor = DrugLikenessPredictor()
    
    print("\n🔹 Comprehensive Drug Likeness Prediction:")
    
    # 包括的薬物らしさ予測
    for i, smiles in enumerate(test_molecules[:3]):  # 最初の3分子のみ
        print(f"\n  Molecule {i+1}: {smiles}")
        
        drug_likeness = drug_likeness_predictor.predict_comprehensive_drug_likeness(smiles)
        
        # 主要な薬物らしさ指標を表示
        key_metrics = ['molecular_weight', 'logp', 'tpsa', 'violations', 'compliant', 'qed', 'sa_score', 'drug_likeness_score']
        for metric in key_metrics:
            if metric in drug_likeness:
                value = drug_likeness[metric]
                if isinstance(value, float):
                    print(f"    {metric}: {value:.3f}")
                else:
                    print(f"    {metric}: {value}")
    
    # バッチ処理
    print(f"\n🔹 Batch Processing {len(test_molecules)} molecules:")
    start_time = time.time()
    
    df = drug_likeness_predictor.process_molecule_batch(test_molecules)
    
    processing_time = time.time() - start_time
    print(f"  Processing time: {processing_time:.2f}s")
    print(f"  Processed {len(df)} molecules successfully")
    print(f"  Columns: {len(df.columns)}")
    
    # 薬物らしさ要約
    drug_likeness_summary = drug_likeness_predictor.get_drug_likeness_summary(df)
    print(f"\n  Drug Likeness Summary:")
    print(f"    Total molecules: {drug_likeness_summary['total_molecules']}")
    print(f"    Drug-like molecules: {drug_likeness_summary['drug_like_molecules']}")
    print(f"    Lipinski compliant: {drug_likeness_summary['lipinski_compliant']}")
    print(f"    Veber compliant: {drug_likeness_summary['veber_compliant']}")
    print(f"    Lead-like molecules: {drug_likeness_summary['lead_like_molecules']}")
    print(f"    Fragment-like molecules: {drug_likeness_summary['fragment_like_molecules']}")
    
    return df


def demonstrate_cns_mpo_calculation():
    """CNS-MPO計算のデモンストレーション"""
    print("\n" + "="*60)
    print("🧠 CNS-MPO Calculation Demo")
    print("="*60)
    
    # テスト分子
    test_molecules = create_test_molecules()
    
    # CNS-MPO計算器を初期化
    cns_mpo_calculator = CNSMPOCalculator()
    
    print("\n🔹 CNS-MPO Calculation:")
    
    # CNS-MPO計算
    for i, smiles in enumerate(test_molecules[:3]):  # 最初の3分子のみ
        print(f"\n  Molecule {i+1}: {smiles}")
        
        cns_mpo = cns_mpo_calculator.calculate_cns_mpo_optimized(smiles)
        
        # 主要なCNS-MPO指標を表示
        key_metrics = ['molecular_weight', 'logp', 'tpsa', 'hbd', 'hba', 'rotatable_bonds', 'cns_mpo_optimized_score', 'cns_mpo_optimized_interpretation']
        for metric in key_metrics:
            if metric in cns_mpo:
                value = cns_mpo[metric]
                if isinstance(value, float):
                    print(f"    {metric}: {value:.3f}")
                else:
                    print(f"    {metric}: {value}")
        
        # 最適化推奨事項
        if 'optimization_recommendations' in cns_mpo:
            recommendations = cns_mpo['optimization_recommendations']
            if recommendations:
                print(f"    Optimization recommendations:")
                for rec in recommendations:
                    print(f"      - {rec}")
    
    # バッチ処理
    print(f"\n🔹 Batch Processing {len(test_molecules)} molecules:")
    start_time = time.time()
    
    df = cns_mpo_calculator.process_molecule_batch(test_molecules)
    
    processing_time = time.time() - start_time
    print(f"  Processing time: {processing_time:.2f}s")
    print(f"  Processed {len(df)} molecules successfully")
    print(f"  Columns: {len(df.columns)}")
    
    # CNS-MPO要約
    cns_mpo_summary = cns_mpo_calculator.get_cns_mpo_summary(df)
    print(f"\n  CNS-MPO Summary:")
    print(f"    Total molecules: {cns_mpo_summary['total_molecules']}")
    if 'cns_mpo_statistics' in cns_mpo_summary:
        stats = cns_mpo_summary['cns_mpo_statistics']
        print(f"    Mean score: {stats['mean_score']:.3f}")
        print(f"    Std score: {stats['std_score']:.3f}")
        print(f"    Min score: {stats['min_score']:.3f}")
        print(f"    Max score: {stats['max_score']:.3f}")
        print(f"    Median score: {stats['median_score']:.3f}")
    
    return df


def plot_admet_comparison(admet_df: pd.DataFrame, property_df: pd.DataFrame, 
                         toxicity_df: pd.DataFrame, drug_likeness_df: pd.DataFrame, 
                         cns_mpo_df: pd.DataFrame):
    """
    ADMET比較をプロット
    
    Args:
        admet_df: ADMETデータフレーム
        property_df: 物性データフレーム
        toxicity_df: 毒性データフレーム
        drug_likeness_df: 薬物らしさデータフレーム
        cns_mpo_df: CNS-MPOデータフレーム
    """
    print("\n" + "="*60)
    print("📊 ADMET Comparison Visualization")
    print("="*60)
    
    # プロット設定
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # 1. 分子量分布
    ax1 = axes[0]
    ax1.hist(property_df['molecular_weight'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Molecular Weight')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Molecular Weight Distribution')
    ax1.grid(True, alpha=0.3)
    
    # 2. LogP分布
    ax2 = axes[1]
    ax2.hist(property_df['logp'], bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_xlabel('LogP')
    ax2.set_ylabel('Frequency')
    ax2.set_title('LogP Distribution')
    ax2.grid(True, alpha=0.3)
    
    # 3. TPSA分布
    ax3 = axes[2]
    ax3.hist(property_df['tpsa'], bins=10, alpha=0.7, color='lightcoral', edgecolor='black')
    ax3.set_xlabel('TPSA')
    ax3.set_ylabel('Frequency')
    ax3.set_title('TPSA Distribution')
    ax3.grid(True, alpha=0.3)
    
    # 4. 薬物らしさスコア分布
    ax4 = axes[3]
    if 'drug_likeness_score' in drug_likeness_df.columns:
        ax4.hist(drug_likeness_df['drug_likeness_score'], bins=10, alpha=0.7, color='gold', edgecolor='black')
        ax4.set_xlabel('Drug Likeness Score')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Drug Likeness Score Distribution')
        ax4.grid(True, alpha=0.3)
    
    # 5. CNS-MPOスコア分布
    ax5 = axes[4]
    if 'cns_mpo_optimized_score' in cns_mpo_df.columns:
        ax5.hist(cns_mpo_df['cns_mpo_optimized_score'], bins=10, alpha=0.7, color='plum', edgecolor='black')
        ax5.set_xlabel('CNS-MPO Score')
        ax5.set_ylabel('Frequency')
        ax5.set_title('CNS-MPO Score Distribution')
        ax5.grid(True, alpha=0.3)
    
    # 6. 毒性スコア分布
    ax6 = axes[5]
    if 'ames_ames_score' in toxicity_df.columns:
        ax6.hist(toxicity_df['ames_ames_score'], bins=10, alpha=0.7, color='salmon', edgecolor='black')
        ax6.set_xlabel('Ames Toxicity Score')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Ames Toxicity Score Distribution')
        ax6.grid(True, alpha=0.3)
    
    plt.suptitle('ADMET Prediction Results Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig('admet_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """
    メイン実行関数
    """
    print("🧬 ChemForge ADMET Prediction Demo")
    print("="*60)
    
    try:
        # 1. ADMET予測デモ
        admet_df = demonstrate_admet_prediction()
        
        # 2. 物性予測デモ
        property_df = demonstrate_property_prediction()
        
        # 3. 毒性予測デモ
        toxicity_df = demonstrate_toxicity_prediction()
        
        # 4. 薬物らしさ予測デモ
        drug_likeness_df = demonstrate_drug_likeness_prediction()
        
        # 5. CNS-MPO計算デモ
        cns_mpo_df = demonstrate_cns_mpo_calculation()
        
        # 6. 比較可視化
        plot_admet_comparison(admet_df, property_df, toxicity_df, drug_likeness_df, cns_mpo_df)
        
        print("\n🎉 All ADMET demonstrations completed successfully!")
        print("ChemForge ADMET prediction system is ready for use!")
        
    except Exception as e:
        logger.error(f"Error in demo: {str(e)}")
        raise


if __name__ == "__main__":
    main()
