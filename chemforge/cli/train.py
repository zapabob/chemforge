"""
ChemForge CLI - Train Command

学習コマンド実装
力価予測モデル学習、ADMET予測モデル学習
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import warnings
warnings.filterwarnings('ignore')

from chemforge.potency.data_processor import DataProcessor
from chemforge.potency.featurizer import MolecularFeaturizer
from chemforge.potency.potency_model import PotencyPredictor
from chemforge.potency.trainer import PotencyTrainer
from chemforge.potency.metrics import PotencyMetrics
from chemforge.utils.external_apis import ExternalAPIManager

def load_config(config_path: str) -> Dict[str, Any]:
    """
    設定ファイル読み込み
    
    Args:
        config_path: 設定ファイルパス
        
    Returns:
        設定辞書
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"[INFO] 設定読み込み完了: {config_path}")
        return config
    except Exception as e:
        print(f"[ERROR] 設定読み込み失敗: {e}")
        return {}

def setup_data_processor(config: Dict[str, Any]) -> DataProcessor:
    """
    データプロセッサー設定
    
    Args:
        config: 設定辞書
        
    Returns:
        DataProcessor
    """
    data_config = config.get('data', {})
    
    processor = DataProcessor(
        low_activity_cutoff=data_config.get('low_activity_cutoff', True),
        strict_outlier_drop=data_config.get('strict_outlier_drop', False),
        scaffold_split=data_config.get('scaffold_split', True),
        time_split=data_config.get('time_split', False),
        train_ratio=data_config.get('train_ratio', 0.8),
        val_ratio=data_config.get('val_ratio', 0.1),
        test_ratio=data_config.get('test_ratio', 0.1)
    )
    
    print("[INFO] データプロセッサー設定完了")
    return processor

def setup_featurizer(config: Dict[str, Any]) -> MolecularFeaturizer:
    """
    フィーチャライザー設定
    
    Args:
        config: 設定辞書
        
    Returns:
        MolecularFeaturizer
    """
    featurizer_config = config.get('featurizer', {})
    
    featurizer = MolecularFeaturizer(
        tokenizer_type=featurizer_config.get('tokenizer_type', 'selfies'),
        max_length=featurizer_config.get('max_length', 128),
        include_3d_features=featurizer_config.get('include_3d_features', False),
        include_descriptors=featurizer_config.get('include_descriptors', True),
        descriptor_types=featurizer_config.get('descriptor_types', ['morgan', 'rdkit', 'maccs']),
        normalize_descriptors=featurizer_config.get('normalize_descriptors', True)
    )
    
    print("[INFO] フィーチャライザー設定完了")
    return featurizer

def setup_model(config: Dict[str, Any]) -> PotencyPredictor:
    """
    モデル設定
    
    Args:
        config: 設定辞書
        
    Returns:
        PotencyPredictor
    """
    model_config = config.get('model', {})
    
    model = PotencyPredictor(
        d_model=model_config.get('d_model', 512),
        n_layers=model_config.get('n_layers', 8),
        n_heads=model_config.get('n_heads', 8),
        d_ff=model_config.get('d_ff', 2048),
        dropout=model_config.get('dropout', 0.1),
        max_length=model_config.get('max_length', 128),
        pwa_buckets=model_config.get('pwa_buckets', {"trivial": 1, "fund": 5, "adj": 2}),
        pet_curv_reg=model_config.get('pet_curv_reg', 1e-6),
        rope=model_config.get('rope', True),
        descriptor_dim=model_config.get('descriptor_dim', 2048)
    )
    
    print("[INFO] モデル設定完了")
    return model

def setup_trainer(config: Dict[str, Any], model: PotencyPredictor) -> PotencyTrainer:
    """
    トレーナー設定
    
    Args:
        config: 設定辞書
        model: モデル
        
    Returns:
        PotencyTrainer
    """
    training_config = config.get('training', {})
    
    trainer = PotencyTrainer(
        model=model,
        learning_rate=training_config.get('learning_rate', 3e-4),
        weight_decay=training_config.get('weight_decay', 0.05),
        warmup_steps=training_config.get('warmup_steps', 2000),
        max_epochs=training_config.get('max_epochs', 100),
        early_stopping_patience=training_config.get('early_stopping_patience', 10),
        grad_clip=training_config.get('grad_clip', 1.0),
        use_amp=training_config.get('use_amp', True),
        device=training_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    print("[INFO] トレーナー設定完了")
    return trainer

def train_potency_model(config_path: str, data_path: str, output_dir: str):
    """
    力価予測モデル学習
    
    Args:
        config_path: 設定ファイルパス
        data_path: データファイルパス
        output_dir: 出力ディレクトリ
    """
    print("=" * 60)
    print("ChemForge 力価予測モデル学習開始")
    print("=" * 60)
    
    # 設定読み込み
    config = load_config(config_path)
    if not config:
        print("[ERROR] 設定読み込み失敗")
        return
    
    # 出力ディレクトリ作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # データプロセッサー設定
        processor = setup_data_processor(config)
        
        # データ読み込み・前処理
        print(f"[INFO] データ読み込み: {data_path}")
        data = processor.load_data(data_path)
        print(f"[INFO] データ数: {len(data)}")
        
        # データ前処理
        print("[INFO] データ前処理開始")
        processed_data = processor.process_data(data)
        print(f"[INFO] 前処理後データ数: {len(processed_data)}")
        
        # データ分割
        print("[INFO] データ分割開始")
        train_data, val_data, test_data = processor.split_data(processed_data)
        print(f"[INFO] 学習: {len(train_data)}, 検証: {len(val_data)}, テスト: {len(test_data)}")
        
        # フィーチャライザー設定
        featurizer = setup_featurizer(config)
        
        # 特徴量生成
        print("[INFO] 特徴量生成開始")
        train_features = featurizer.fit_transform(train_data)
        val_features = featurizer.transform(val_data)
        test_features = featurizer.transform(test_data)
        
        print(f"[INFO] 特徴量次元: {train_features['tokens'].shape}")
        
        # モデル設定
        model = setup_model(config)
        
        # トレーナー設定
        trainer = setup_trainer(config, model)
        
        # 学習実行
        print("[INFO] 学習開始")
        trainer.fit(
            train_features, val_features,
            save_path=str(output_path / "model.pt"),
            log_path=str(output_path / "training.log")
        )
    
        # テスト評価
        print("[INFO] テスト評価開始")
        test_metrics = trainer.evaluate(test_features)
        
        # 結果保存
        results = {
            'test_metrics': test_metrics,
            'config': config,
            'data_info': {
                'total_samples': len(data),
                'train_samples': len(train_data),
                'val_samples': len(val_data),
                'test_samples': len(test_data)
            }
        }
        
        # 結果保存
        import json
        with open(output_path / "results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] 学習完了: {output_dir}")
        print(f"[INFO] テストRMSE: {test_metrics.get('rmse', 'N/A'):.4f}")
        print(f"[INFO] テストR²: {test_metrics.get('r2', 'N/A'):.4f}")
        
    except Exception as e:
        print(f"[ERROR] 学習失敗: {e}")
        import traceback
        traceback.print_exc()

def train_admet_model(config_path: str, data_path: str, output_dir: str):
    """
    ADMET予測モデル学習
    
    Args:
        config_path: 設定ファイルパス
        data_path: データファイルパス
        output_dir: 出力ディレクトリ
    """
    print("=" * 60)
    print("ChemForge ADMET予測モデル学習開始")
    print("=" * 60)
    
    # 設定読み込み
    config = load_config(config_path)
    if not config:
        print("[ERROR] 設定読み込み失敗")
        return
    
    # 出力ディレクトリ作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # データ読み込み
        print(f"[INFO] データ読み込み: {data_path}")
        import pandas as pd
        data = pd.read_csv(data_path)
        print(f"[INFO] データ数: {len(data)}")
        
        # 外部API統合
        print("[INFO] 外部API統合開始")
        api_manager = ExternalAPIManager(cache_dir=str(output_path / "cache"))
        
        # 分子情報取得
        smiles_list = data['smiles'].tolist()
        molecule_info = api_manager.batch_molecule_info(smiles_list)
        
        # 結果保存
        import json
        with open(output_path / "molecule_info.json", 'w', encoding='utf-8') as f:
            json.dump(molecule_info, f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] ADMET学習完了: {output_dir}")
        
    except Exception as e:
        print(f"[ERROR] ADMET学習失敗: {e}")
        import traceback
        traceback.print_exc()

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="ChemForge Train Command")
    parser.add_argument("--config", required=True, help="設定ファイルパス")
    parser.add_argument("--data", required=True, help="データファイルパス")
    parser.add_argument("--output", required=True, help="出力ディレクトリ")
    parser.add_argument("--model-type", choices=["potency", "admet"], default="potency", help="モデルタイプ")
    
    args = parser.parse_args()
    
    if args.model_type == "potency":
        train_potency_model(args.config, args.data, args.output)
    elif args.model_type == "admet":
        train_admet_model(args.config, args.data, args.output)
    else:
        print(f"[ERROR] 未知のモデルタイプ: {args.model_type}")

if __name__ == "__main__":
    main()