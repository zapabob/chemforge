"""
ChEMBL正規IDフェッチャー

ChEMBL Web Resource Clientを使用して、CNS受容体の正確なChEMBL IDを取得する。
"""

from chembl_webresource_client.new_client import new_client
import json

def fetch_chembl_ids():
    """ChEMBL APIから正確な受容体IDを取得"""
    
    targets_api = new_client.target
    
    # 検索する受容体のリスト（UniProt IDと名前）
    receptors = {
        '5HT2A': {'uniprot': 'P28223', 'name': '5-hydroxytryptamine receptor 2A'},
        '5HT1A': {'uniprot': 'P08908', 'name': '5-hydroxytryptamine receptor 1A'},
        'D1': {'uniprot': 'P21728', 'name': 'D(1A) dopamine receptor'},
        'D2': {'uniprot': 'P14416', 'name': 'D(2) dopamine receptor'},
        'CB1': {'uniprot': 'P21554', 'name': 'Cannabinoid receptor 1'},
        'CB2': {'uniprot': 'P34972', 'name': 'Cannabinoid receptor 2'},
        'MOR': {'uniprot': 'P35372', 'name': 'Mu-type opioid receptor'},
        'DOR': {'uniprot': 'P41143', 'name': 'Delta-type opioid receptor'},
        'KOR': {'uniprot': 'P41145', 'name': 'Kappa-type opioid receptor'},
        'NOP': {'uniprot': 'P41146', 'name': 'Nociceptin receptor'},
        'SERT': {'uniprot': 'P31645', 'name': 'Sodium-dependent serotonin transporter'},
        'DAT': {'uniprot': 'Q01959', 'name': 'Sodium-dependent dopamine transporter'},
        'NET': {'uniprot': 'P23975', 'name': 'Sodium-dependent noradrenaline transporter'}
    }
    
    results = {}
    
    print("ChEMBL正規ID取得開始...")
    print("=" * 60)
    
    for receptor_key, receptor_info in receptors.items():
        uniprot_id = receptor_info['uniprot']
        expected_name = receptor_info['name']
        
        try:
            # UniProt IDでターゲットを検索
            target = targets_api.filter(
                target_components__accession=uniprot_id,
                target_organism='Homo sapiens'
            ).only([
                'target_chembl_id', 
                'pref_name', 
                'organism', 
                'target_type'
            ])
            
            target_list = list(target)
            
            if target_list:
                target_data = target_list[0]
                chembl_id = target_data['target_chembl_id']
                pref_name = target_data['pref_name']
                
                results[receptor_key] = {
                    'chembl_id': chembl_id,
                    'name': pref_name,
                    'uniprot_id': uniprot_id,
                    'organism': target_data['organism'],
                    'target_type': target_data['target_type']
                }
                
                print(f"[OK] {receptor_key:6s} -> {chembl_id:12s} ({pref_name})")
            else:
                print(f"[!!] {receptor_key:6s} -> NOT FOUND (UniProt: {uniprot_id})")
                
        except Exception as e:
            print(f"[!!] {receptor_key:6s} -> ERROR: {e}")
    
    print("=" * 60)
    print(f"[OK] 取得完了: {len(results)}/{len(receptors)} targets")
    
    # 結果をJSONで保存
    output_file = 'correct_chembl_ids.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] 結果を保存: {output_file}")
    
    return results


if __name__ == "__main__":
    results = fetch_chembl_ids()
    
    print("\n[OK] 取得したChEMBL ID一覧:")
    print("=" * 60)
    for key, data in results.items():
        print(f"{key:6s}: {data['chembl_id']:12s} (UniProt: {data['uniprot_id']})")

