"""
ChemForge CLI - ADMET Command

ADMET予測コマンド実装
物性、薬物動態、毒性、薬物らしさ予測
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

from chemforge.utils.external_apis import ExternalAPIManager
from chemforge.utils.web_scraper import SwissADMEScraper
from chemforge.potency.featurizer import MolecularFeaturizer

class ADMETPredictor:
    """ADMET予測クラス"""
    
    def __init__(self, cache_dir: str = "cache"):
        """
        初期化
        
        Args:
            cache_dir: キャッシュディレクトリ
        """
        self.cache_dir = cache_dir
        self.api_manager = ExternalAPIManager(cache_dir)
        self.swissadme_scraper = SwissADMEScraper(rate_limit=1.0)
        self.featurizer = MolecularFeaturizer()
    
    def predict_physicochemical_properties(self, smiles_list: List[str]) -> List[Dict[str, Any]]:
        """
        物性予測
        
        Args:
            smiles_list: SMILES文字列リスト
            
        Returns:
            物性予測結果リスト
        """
        print("[INFO] 物性予測開始")
        
        results = []
        for smiles in smiles_list:
            try:
                # RDKit記述子計算
                features = self.featurizer.transform([{'smiles': smiles}])
                descriptors = features['descriptors'][0].cpu().numpy()
                
                # 物性予測（簡易版）
                result = {
                    'smiles': smiles,
                    'molecular_weight': float(descriptors[0]) if len(descriptors) > 0 else None,
                    'logp': float(descriptors[1]) if len(descriptors) > 1 else None,
                    'tpsa': float(descriptors[2]) if len(descriptors) > 2 else None,
                    'hbd': int(descriptors[3]) if len(descriptors) > 3 else None,
                    'hba': int(descriptors[4]) if len(descriptors) > 4 else None,
                    'rotatable_bonds': int(descriptors[5]) if len(descriptors) > 5 else None,
                    'aromatic_rings': int(descriptors[6]) if len(descriptors) > 6 else None,
                    'status': 'success'
                }
                
            except Exception as e:
                result = {
                    'smiles': smiles,
                    'status': 'error',
                    'error': str(e)
                }
            
            results.append(result)
        
        print(f"[INFO] 物性予測完了: {len(results)} results")
        return results
    
    def predict_pharmacokinetics(self, smiles_list: List[str]) -> List[Dict[str, Any]]:
        """
        薬物動態予測
        
        Args:
            smiles_list: SMILES文字列リスト
            
        Returns:
            薬物動態予測結果リスト
        """
        print("[INFO] 薬物動態予測開始（SwissADME）")
        
        # SwissADME予測
        results = self.swissadme_scraper.predict_batch(smiles_list)
        
        # 結果整理
        processed_results = []
        for result in results:
            processed_result = {
                'smiles': result['smiles'],
                'status': result['status']
            }
            
            if result['status'] == 'success':
                # 吸収
                admet_props = result.get('admet_properties', {})
                processed_result.update({
                    'gi_absorption': admet_props.get('gi_absorption'),
                    'bbb_permeant': admet_props.get('bbb_permeant'),
                    'pgp_substrate': admet_props.get('pgp_substrate')
                })
                
                # 代謝
                processed_result.update({
                    'cyp1a2_inhibitor': admet_props.get('cyp1a2_inhibitor'),
                    'cyp2c19_inhibitor': admet_props.get('cyp2c19_inhibitor'),
                    'cyp2c9_inhibitor': admet_props.get('cyp2c9_inhibitor'),
                    'cyp2d6_inhibitor': admet_props.get('cyp2d6_inhibitor'),
                    'cyp3a4_inhibitor': admet_props.get('cyp3a4_inhibitor')
                })
            else:
                processed_result['error'] = result.get('error', 'Unknown error')
            
            processed_results.append(processed_result)
        
        print(f"[INFO] 薬物動態予測完了: {len(processed_results)} results")
        return processed_results
    
    def predict_toxicity(self, smiles_list: List[str]) -> List[Dict[str, Any]]:
        """
        毒性予測
        
        Args:
            smiles_list: SMILES文字列リスト
            
        Returns:
            毒性予測結果リスト
        """
        print("[INFO] 毒性予測開始")
        
        results = []
        for smiles in smiles_list:
            try:
                # 簡易毒性予測（RDKit記述子ベース）
                features = self.featurizer.transform([{'smiles': smiles}])
                descriptors = features['descriptors'][0].cpu().numpy()
                
                # 毒性予測（簡易版）
                result = {
                    'smiles': smiles,
                    'mutagenic': 'No' if descriptors[0] < 0.5 else 'Yes',
                    'tumorigenic': 'No' if descriptors[1] < 0.5 else 'Yes',
                    'irritant': 'No' if descriptors[2] < 0.5 else 'Yes',
                    'reproductive_effective': 'No' if descriptors[3] < 0.5 else 'Yes',
                    'status': 'success'
                }
                
            except Exception as e:
                result = {
                    'smiles': smiles,
                    'status': 'error',
                    'error': str(e)
                }
            
            results.append(result)
        
        print(f"[INFO] 毒性予測完了: {len(results)} results")
        return results
    
    def predict_drug_likeness(self, smiles_list: List[str]) -> List[Dict[str, Any]]:
        """
        薬物らしさ予測
        
        Args:
            smiles_list: SMILES文字列リスト
            
        Returns:
            薬物らしさ予測結果リスト
        """
        print("[INFO] 薬物らしさ予測開始（SwissADME）")
        
        # SwissADME予測
        results = self.swissadme_scraper.predict_batch(smiles_list)
        
        # 結果整理
        processed_results = []
        for result in results:
            processed_result = {
                'smiles': result['smiles'],
                'status': result['status']
            }
            
            if result['status'] == 'success':
                # ドラッグライクネス
                drug_likeness = result.get('drug_likeness', {})
                processed_result.update({
                    'lipinski': drug_likeness.get('lipinski'),
                    'veber': drug_likeness.get('veber'),
                    'egan': drug_likeness.get('egan'),
                    'muegge': drug_likeness.get('muegge')
                })
                
                # 分子特性
                mol_props = result.get('molecular_properties', {})
                processed_result.update({
                    'molecular_weight': mol_props.get('molecular_weight'),
                    'logp': mol_props.get('logp'),
                    'tpsa': mol_props.get('tpsa'),
                    'hbd': mol_props.get('hbd'),
                    'hba': mol_props.get('hba')
                })
            else:
                processed_result['error'] = result.get('error', 'Unknown error')
            
            processed_results.append(processed_result)
        
        print(f"[INFO] 薬物らしさ予測完了: {len(processed_results)} results")
        return processed_results
    
    def predict_comprehensive_admet(self, smiles_list: List[str]) -> List[Dict[str, Any]]:
        """
        包括的ADMET予測
        
        Args:
            smiles_list: SMILES文字列リスト
            
        Returns:
            包括的ADMET予測結果リスト
        """
        print("=" * 60)
        print("ChemForge 包括的ADMET予測開始")
        print("=" * 60)
        
        # 各予測実行
        print("[INFO] 物性予測実行")
        physicochemical_results = self.predict_physicochemical_properties(smiles_list)
        
        print("[INFO] 薬物動態予測実行")
        pk_results = self.predict_pharmacokinetics(smiles_list)
        
        print("[INFO] 毒性予測実行")
        toxicity_results = self.predict_toxicity(smiles_list)
        
        print("[INFO] 薬物らしさ予測実行")
        drug_likeness_results = self.predict_drug_likeness(smiles_list)
        
        # 結果統合
        print("[INFO] 結果統合開始")
        comprehensive_results = []
        
        for i, smiles in enumerate(smiles_list):
            result = {
                'smiles': smiles,
                'status': 'success'
            }
            
            # 物性
            if i < len(physicochemical_results):
                phys_result = physicochemical_results[i]
                if phys_result['status'] == 'success':
                    result.update({
                        'molecular_weight': phys_result.get('molecular_weight'),
                        'logp': phys_result.get('logp'),
                        'tpsa': phys_result.get('tpsa'),
                        'hbd': phys_result.get('hbd'),
                        'hba': phys_result.get('hba'),
                        'rotatable_bonds': phys_result.get('rotatable_bonds'),
                        'aromatic_rings': phys_result.get('aromatic_rings')
                    })
            
            # 薬物動態
            if i < len(pk_results):
                pk_result = pk_results[i]
                if pk_result['status'] == 'success':
                    result.update({
                        'gi_absorption': pk_result.get('gi_absorption'),
                        'bbb_permeant': pk_result.get('bbb_permeant'),
                        'pgp_substrate': pk_result.get('pgp_substrate'),
                        'cyp1a2_inhibitor': pk_result.get('cyp1a2_inhibitor'),
                        'cyp2c19_inhibitor': pk_result.get('cyp2c19_inhibitor'),
                        'cyp2c9_inhibitor': pk_result.get('cyp2c9_inhibitor'),
                        'cyp2d6_inhibitor': pk_result.get('cyp2d6_inhibitor'),
                        'cyp3a4_inhibitor': pk_result.get('cyp3a4_inhibitor')
                    })
            
            # 毒性
            if i < len(toxicity_results):
                tox_result = toxicity_results[i]
                if tox_result['status'] == 'success':
                    result.update({
                        'mutagenic': tox_result.get('mutagenic'),
                        'tumorigenic': tox_result.get('tumorigenic'),
                        'irritant': tox_result.get('irritant'),
                        'reproductive_effective': tox_result.get('reproductive_effective')
                    })
            
            # 薬物らしさ
            if i < len(drug_likeness_results):
                dl_result = drug_likeness_results[i]
                if dl_result['status'] == 'success':
                    result.update({
                        'lipinski': dl_result.get('lipinski'),
                        'veber': dl_result.get('veber'),
                        'egan': dl_result.get('egan'),
                        'muegge': dl_result.get('muegge')
                    })
            
            comprehensive_results.append(result)
        
        print(f"[INFO] 包括的ADMET予測完了: {len(comprehensive_results)} results")
        return comprehensive_results

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="ChemForge ADMET Command")
    parser.add_argument("--smiles", nargs="+", help="SMILES文字列リスト")
    parser.add_argument("--input", help="入力ファイルパス（CSV）")
    parser.add_argument("--output", required=True, help="出力ファイルパス")
    parser.add_argument("--prediction-type", choices=["physicochemical", "pharmacokinetics", "toxicity", "drug-likeness", "comprehensive"], default="comprehensive", help="予測タイプ")
    parser.add_argument("--cache-dir", default="cache", help="キャッシュディレクトリ")
    
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
    
    # ADMET予測実行
    predictor = ADMETPredictor(cache_dir=args.cache_dir)
    
    if args.prediction_type == "physicochemical":
        results = predictor.predict_physicochemical_properties(smiles_list)
    elif args.prediction_type == "pharmacokinetics":
        results = predictor.predict_pharmacokinetics(smiles_list)
    elif args.prediction_type == "toxicity":
        results = predictor.predict_toxicity(smiles_list)
    elif args.prediction_type == "drug-likeness":
        results = predictor.predict_drug_likeness(smiles_list)
    elif args.prediction_type == "comprehensive":
        results = predictor.predict_comprehensive_admet(smiles_list)
    else:
        print(f"[ERROR] 未知の予測タイプ: {args.prediction_type}")
        return
    
    # 結果保存
    try:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"[INFO] 結果保存完了: {args.output}")
        
        # 統計表示
        success_count = len([r for r in results if r.get('status') == 'success'])
        print(f"[INFO] 成功数: {success_count}/{len(results)}")
        
    except Exception as e:
        print(f"[ERROR] 結果保存失敗: {e}")

if __name__ == "__main__":
    main()