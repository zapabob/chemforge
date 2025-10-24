"""
UniProt ID検証スクリプト

UniProtの公式データベースから正確なUniProt IDを検証・取得する。
"""

import requests
import json
import time
from typing import Dict, List, Optional

def verify_uniprot_ids():
    """UniProt APIから正確なUniProt IDを検証"""
    
    # 現在使用中のUniProt ID
    current_uniprot_ids = {
        '5HT2A': 'P28223',
        '5HT1A': 'P08908', 
        'D1': 'P21728',
        'D2': 'P14416',
        'CB1': 'P21554',
        'CB2': 'P34972',
        'MOR': 'P35372',
        'DOR': 'P41143',
        'KOR': 'P41145',
        'NOP': 'P41146',
        'SERT': 'P31645',
        'DAT': 'Q01959',
        'NET': 'P23975'
    }
    
    # UniProt API エンドポイント
    uniprot_api_base = "https://rest.uniprot.org"
    
    results = {}
    
    print("UniProt ID検証開始...")
    print("=" * 60)
    
    for receptor_key, uniprot_id in current_uniprot_ids.items():
        try:
            # UniProt REST APIでIDを検証
            url = f"{uniprot_api_base}/uniprotkb/{uniprot_id}"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # 基本情報を抽出
                entry_name = data.get('entryName', '')
                protein_name = data.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', '')
                organism = data.get('organism', {}).get('scientificName', '')
                gene_names = data.get('genes', [{}])[0].get('geneName', {}).get('value', '')
                
                # データベースクロスリファレンスを取得
                cross_references = data.get('uniProtKBCrossReferences', [])
                chembl_refs = [ref for ref in cross_references if ref.get('database') == 'ChEMBL']
                
                results[receptor_key] = {
                    'uniprot_id': uniprot_id,
                    'entry_name': entry_name,
                    'protein_name': protein_name,
                    'organism': organism,
                    'gene_names': gene_names,
                    'chembl_refs': [ref.get('id') for ref in chembl_refs],
                    'status': 'VERIFIED'
                }
                
                print(f"[OK] {receptor_key:6s} -> {uniprot_id:8s} ({protein_name[:50]}...)")
                
            elif response.status_code == 404:
                print(f"[!!] {receptor_key:6s} -> {uniprot_id:8s} NOT FOUND")
                results[receptor_key] = {
                    'uniprot_id': uniprot_id,
                    'status': 'NOT_FOUND'
                }
            else:
                print(f"[!!] {receptor_key:6s} -> {uniprot_id:8s} ERROR: {response.status_code}")
                results[receptor_key] = {
                    'uniprot_id': uniprot_id,
                    'status': 'ERROR',
                    'error_code': response.status_code
                }
                
        except Exception as e:
            print(f"[!!] {receptor_key:6s} -> {uniprot_id:8s} EXCEPTION: {e}")
            results[receptor_key] = {
                'uniprot_id': uniprot_id,
                'status': 'EXCEPTION',
                'error': str(e)
            }
        
        # API制限を避けるため少し待機
        time.sleep(0.5)
    
    print("=" * 60)
    verified_count = sum(1 for r in results.values() if r.get('status') == 'VERIFIED')
    print(f"[OK] 検証完了: {verified_count}/{len(current_uniprot_ids)} targets")
    
    # 結果をJSONで保存
    output_file = 'uniprot_verification_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] 結果を保存: {output_file}")
    
    return results


def search_alternative_uniprot_ids():
    """代替UniProt IDを検索"""
    
    # 検索する受容体名のリスト
    search_terms = {
        '5HT2A': ['5-hydroxytryptamine receptor 2A', '5HT2A', 'serotonin receptor 2A'],
        '5HT1A': ['5-hydroxytryptamine receptor 1A', '5HT1A', 'serotonin receptor 1A'],
        'D1': ['D(1A) dopamine receptor', 'D1 dopamine receptor', 'DRD1'],
        'D2': ['D(2) dopamine receptor', 'D2 dopamine receptor', 'DRD2'],
        'CB1': ['Cannabinoid receptor 1', 'CB1', 'CNR1'],
        'CB2': ['Cannabinoid receptor 2', 'CB2', 'CNR2'],
        'MOR': ['Mu-type opioid receptor', 'MOR', 'OPRM1'],
        'DOR': ['Delta-type opioid receptor', 'DOR', 'OPRD1'],
        'KOR': ['Kappa-type opioid receptor', 'KOR', 'OPRK1'],
        'NOP': ['Nociceptin receptor', 'NOP', 'OPRL1'],
        'SERT': ['Sodium-dependent serotonin transporter', 'SERT', 'SLC6A4'],
        'DAT': ['Sodium-dependent dopamine transporter', 'DAT', 'SLC6A3'],
        'NET': ['Sodium-dependent noradrenaline transporter', 'NET', 'SLC6A2']
    }
    
    uniprot_api_base = "https://rest.uniprot.org"
    results = {}
    
    print("\n代替UniProt ID検索開始...")
    print("=" * 60)
    
    for receptor_key, search_terms_list in search_terms.items():
        try:
            # UniProt検索APIを使用
            search_query = f"organism_id:9606 AND ({' OR '.join([f'name:{term}' for term in search_terms_list])})"
            url = f"{uniprot_api_base}/uniprotkb/search"
            params = {
                'query': search_query,
                'format': 'json',
                'size': 5
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                results[receptor_key] = data.get('results', [])
                
                if results[receptor_key]:
                    best_match = results[receptor_key][0]
                    uniprot_id = best_match.get('primaryAccession', '')
                    protein_name = best_match.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', '')
                    print(f"[OK] {receptor_key:6s} -> {uniprot_id:8s} ({protein_name[:50]}...)")
                else:
                    print(f"[!!] {receptor_key:6s} -> NO RESULTS")
            else:
                print(f"[!!] {receptor_key:6s} -> SEARCH ERROR: {response.status_code}")
                
        except Exception as e:
            print(f"[!!] {receptor_key:6s} -> SEARCH EXCEPTION: {e}")
        
        # API制限を避けるため少し待機
        time.sleep(0.5)
    
    print("=" * 60)
    print("[OK] 代替ID検索完了")
    
    return results


if __name__ == "__main__":
    # 現在のUniProt IDを検証
    verification_results = verify_uniprot_ids()
    
    # 代替UniProt IDを検索
    alternative_results = search_alternative_uniprot_ids()
    
    print("\n[OK] UniProt ID検証・検索完了")
    print("=" * 60)
    
    # 検証結果のサマリー
    verified_count = sum(1 for r in verification_results.values() if r.get('status') == 'VERIFIED')
    print(f"検証済み: {verified_count}/{len(verification_results)} targets")
    
    # 問題のあるIDを表示
    print("\n問題のあるUniProt ID:")
    for key, result in verification_results.items():
        if result.get('status') != 'VERIFIED':
            print(f"  {key}: {result.get('uniprot_id')} ({result.get('status')})")
