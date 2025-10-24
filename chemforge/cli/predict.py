"""
ChemForge CLI - Predict Command

予測コマンド実装
力価予測、ADMET予測、分子生成
"""

import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import warnings
warnings.filterwarnings('ignore')

from chemforge.potency.data_processor import DataProcessor
from chemforge.potency.featurizer import MolecularFeaturizer
from chemforge.potency.potency_model import PotencyPredictor
from chemforge.potency.trainer import PotencyTrainer
from chemforge.potency.metrics import PotencyMetrics
from chemforge.utils.external_apis import ExternalAPIManager
from chemforge.utils.web_scraper import SwissADMEScraper

def load_model(model_path: str, config_path: str) -> tuple:
    """
    モデル読み込み
    
    Args:
        model_path: モデルファイルパス
        config_path: 設定ファイルパス
        
    Returns:
        (model, config, featurizer)
    """
    try:
        # 設定読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # モデル設定
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
        
        # モデル読み込み
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # フィーチャライザー設定
        featurizer_config = config.get('featurizer', {})
        featurizer = MolecularFeaturizer(
            tokenizer_type=featurizer_config.get('tokenizer_type', 'selfies'),
            max_length=featurizer_config.get('max_length', 128),
            include_3d_features=featurizer_config.get('include_3d_features', False),
            include_descriptors=featurizer_config.get('include_descriptors', True),
            descriptor_types=featurizer_config.get('descriptor_types', ['morgan', 'rdkit', 'maccs']),
            normalize_descriptors=featurizer_config.get('normalize_descriptors', True)
        )
        
        print(f"[INFO] モデル読み込み完了: {model_path}")
        return model, config, featurizer
        
    except Exception as e:
        print(f"[ERROR] モデル読み込み失敗: {e}")
        return None, None, None

def predict_potency(model_path: str, config_path: str, smiles_list: List[str], output_path: str):
    """
    力価予測
    
    Args:
        model_path: モデルファイルパス
        config_path: 設定ファイルパス
        smiles_list: SMILES文字列リスト
        output_path: 出力パス
    """
    print("=" * 60)
    print("ChemForge 力価予測開始")
    print("=" * 60)
    
    # モデル読み込み
    model, config, featurizer = load_model(model_path, config_path)
    if model is None:
        return
    
    try:
        # デバイス設定
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # データ準備
        data = [{'smiles': smiles} for smiles in smiles_list]
        
        # 特徴量生成
        print("[INFO] 特徴量生成開始")
        features = featurizer.transform(data)
        
        # 予測実行
        print("[INFO] 予測実行開始")
        with torch.no_grad():
            # バッチ予測
            batch_size = 32
            predictions = []
            
            for i in range(0, len(features['tokens']), batch_size):
                batch_tokens = features['tokens'][i:i+batch_size].to(device)
                batch_descriptors = features['descriptors'][i:i+batch_size].to(device)
                
                # 予測
                outputs = model(batch_tokens, batch_descriptors)
                
                # 回帰予測
                regression_pred = outputs['regression'].cpu().numpy()
                predictions.extend(regression_pred)
        
        # 結果整理
        results = []
        for i, smiles in enumerate(smiles_list):
            result = {
                'smiles': smiles,
                'predicted_pic50': float(predictions[i][0]) if len(predictions[i]) > 0 else None,
                'predicted_pki': float(predictions[i][1]) if len(predictions[i]) > 1 else None
            }
            results.append(result)
        
        # 結果保存
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        
        print(f"[INFO] 予測完了: {output_path}")
        print(f"[INFO] 予測数: {len(results)}")
        
        # 統計表示
        if 'predicted_pic50' in df.columns:
            pic50_mean = df['predicted_pic50'].mean()
            pic50_std = df['predicted_pic50'].std()
            print(f"[INFO] pIC50予測: {pic50_mean:.2f} ± {pic50_std:.2f}")
        
        if 'predicted_pki' in df.columns:
            pki_mean = df['predicted_pki'].mean()
            pki_std = df['predicted_pki'].std()
            print(f"[INFO] pKi予測: {pki_mean:.2f} ± {pki_std:.2f}")
        
    except Exception as e:
        print(f"[ERROR] 予測失敗: {e}")
        import traceback
        traceback.print_exc()

def predict_admet(smiles_list: List[str], output_path: str):
    """
    ADMET予測（SwissADME）
    
    Args:
        smiles_list: SMILES文字列リスト
        output_path: 出力パス
    """
    print("=" * 60)
    print("ChemForge ADMET予測開始（SwissADME）")
    print("=" * 60)
    
    try:
        # SwissADMEスクレイパー
        scraper = SwissADMEScraper(rate_limit=1.0)
        
        # バッチ予測
        print(f"[INFO] ADMET予測開始: {len(smiles_list)} molecules")
        results = scraper.predict_batch(smiles_list)
        
        # 結果整理
        processed_results = []
        for result in results:
            processed_result = {
                'smiles': result['smiles'],
                'status': result['status']
            }
            
            if result['status'] == 'success':
                # 分子特性
                mol_props = result.get('molecular_properties', {})
                processed_result.update({
                    'molecular_weight': mol_props.get('molecular_weight'),
                    'logp': mol_props.get('logp'),
                    'tpsa': mol_props.get('tpsa'),
                    'hbd': mol_props.get('hbd'),
                    'hba': mol_props.get('hba')
                })
                
                # ADMET特性
                admet_props = result.get('admet_properties', {})
                processed_result.update({
                    'gi_absorption': admet_props.get('gi_absorption'),
                    'bbb_permeant': admet_props.get('bbb_permeant'),
                    'pgp_substrate': admet_props.get('pgp_substrate'),
                    'cyp1a2_inhibitor': admet_props.get('cyp1a2_inhibitor'),
                    'cyp2c19_inhibitor': admet_props.get('cyp2c19_inhibitor'),
                    'cyp2c9_inhibitor': admet_props.get('cyp2c9_inhibitor'),
                    'cyp2d6_inhibitor': admet_props.get('cyp2d6_inhibitor'),
                    'cyp3a4_inhibitor': admet_props.get('cyp3a4_inhibitor')
                })
                
                # ドラッグライクネス
                drug_likeness = result.get('drug_likeness', {})
                processed_result.update({
                    'lipinski': drug_likeness.get('lipinski'),
                    'veber': drug_likeness.get('veber'),
                    'egan': drug_likeness.get('egan'),
                    'muegge': drug_likeness.get('muegge')
                })
            else:
                processed_result[processed_result['error'] = result.get('error', 'Unknown error')
            
            processed_results.append(processed_result)
        
        # 結果保存
        df = pd.DataFrame(processed_results)
        df.to_csv(output_path, index=False)
        
        print(f"[INFO] ADMET予測完了: {output_path}")
        print(f"[INFO] 予測数: {len(processed_results)}")
        
        # 統計表示
        success_count = len([r for r in processed_results if r['status'] == 'success'])
        print(f"[INFO] 成功数: {success_count}/{len(processed_results)}")
        
    except Exception as e:
        print(f"[ERROR] ADMET予測失敗: {e}")
        import traceback
        traceback.print_exc()

def predict_external_apis(smiles_list: List[str], output_path: str):
    """
    外部API予測
    
    Args:
        smiles_list: SMILES文字列リスト
        output_path: 出力パス
    """
    print("=" * 60)
    print("ChemForge 外部API予測開始")
    print("=" * 60)
    
    try:
        # 外部APIマネージャー
        api_manager = ExternalAPIManager(cache_dir="cache")
        
        # バッチ予測
        print(f"[INFO] 外部API予測開始: {len(smiles_list)} molecules")
        results = api_manager.batch_molecule_info(smiles_list)
        
        # 結果整理
        processed_results = []
        for result in results:
            processed_result = {
                'smiles': result['smiles'],
                'status': result['status']
            }
            
            if result['status'] == 'success':
                # PubChem情報
                pubchem = result.get('pubchem', {})
                if pubchem.get('status') == 'success':
                    processed_result.update({
                        'pubchem_cid': pubchem.get('cid'),
                        'pubchem_molecular_weight': pubchem.get('properties', {}).get('molecular_weight'),
                        'pubchem_logp': pubchem.get('properties', {}).get('logp')
                    })
                
                # ChEMBL情報
                chembl = result.get('chembl', {})
                if chembl.get('status') == 'success':
                    processed_result.update({
                        'chembl_id': chembl.get('chembl_id'),
                        'chembl_activities_count': chembl.get('activities', {}).get('count', 0)
                    })
            
            processed_results.append(processed_result)
        
        # 結果保存
        df = pd.DataFrame(processed_results)
        df.to_csv(output_path, index=False)
        
        print(f"[INFO] 外部API予測完了: {output_path}")
        print(f"[INFO] 予測数: {len(processed_results)}")
        
    except Exception as e:
        print(f"[ERROR] 外部API予測失敗: {e}")
        import traceback
        traceback.print_exc()

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="ChemForge Predict Command")
    parser.add_argument("--model", help="モデルファイルパス（力価予測用）")
    parser.add_argument("--config", help="設定ファイルパス（力価予測用）")
    parser.add_argument("--smiles", nargs="+", help="SMILES文字列リスト")
    parser.add_argument("--input", help="入力ファイルパス（CSV）")
    parser.add_argument("--output", required=True, help="出力ファイルパス")
    parser.add_argument("--prediction-type", choices=["potency", "admet", "external"], required=True, help="予測タイプ")
    
    args = parser.parse_args()
    
    # SMILES取得
    smiles_list = []
    if args.smiles:
        smiles_list = args.smiles
    elif args.input:
        try:
            df = pd.read_csv(args.input)
            if 'smiles' in df.columns:
                smiles_list = df['smiles'].tolist()
            else:
                print("[ERROR] 入力ファイルに'smiles'列が見つかりません")
                return
        except Exception as e:
            print(f"[ERROR] 入力ファイル読み込み失敗: {e}")
            return
    else:
        print("[ERROR] --smiles または --input を指定してください")
        return
    
    if not smiles_list:
        print("[ERROR] SMILESが指定されていません")
        return

    print(f"[INFO] 予測対象: {len(smiles_list)} molecules")
    
    # 予測実行
    if args.prediction_type == "potency":
        if not args.model or not args.config:
            print("[ERROR] 力価予測には --model と --config が必要です")
            return
        predict_potency(args.model, args.config, smiles_list, args.output)
    
    elif args.prediction_type == "admet":
        predict_admet(smiles_list, args.output)
    
    elif args.prediction_type == "external":
        predict_external_apis(smiles_list, args.output)

if __name__ == "__main__":
    main()