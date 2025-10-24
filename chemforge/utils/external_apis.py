"""
External APIs Integration

外部API統合（PubChem、DrugBank、ChEMBL、UniProt）
レート制限対応、エラーハンドリング、キャッシュ機能
"""

import requests
import time
import json
import hashlib
import os
from typing import Dict, List, Optional, Union, Any, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class ExternalAPIClient:
    """外部APIクライアント基底クラス"""
    
    def __init__(self, cache_dir: str = "cache", rate_limit: float = 1.0):
        """
        初期化
        
        Args:
            cache_dir: キャッシュディレクトリ
            rate_limit: レート制限（秒）
        """
        self.cache_dir = cache_dir
        self.rate_limit = rate_limit
        self.last_request_time = 0
        
        # キャッシュディレクトリ作成
        os.makedirs(cache_dir, exist_ok=True)
        
        # セッション作成
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ChemForge/1.0 (https://github.com/chemforge/chemforge)',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
    
    def _wait_for_rate_limit(self):
        """レート制限待機"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit:
            sleep_time = self.rate_limit - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _get_cache_path(self, key: str) -> str:
        """キャッシュパス取得"""
        cache_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def _load_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """キャッシュから読み込み"""
        cache_path = self._get_cache_path(key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARNING] Cache load failed: {e}")
        return None
    
    def _save_to_cache(self, key: str, data: Dict[str, Any]):
        """キャッシュに保存"""
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[WARNING] Cache save failed: {e}")

class PubChemClient(ExternalAPIClient):
    """PubChem APIクライアント"""
    
    def __init__(self, cache_dir: str = "cache", rate_limit: float = 0.1):
        super().__init__(cache_dir, rate_limit)
        self.base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    
    def get_compound_by_smiles(self, smiles: str) -> Dict[str, Any]:
        """
        SMILESから化合物情報取得
        
        Args:
            smiles: SMILES文字列
            
        Returns:
            化合物情報辞書
        """
        cache_key = f"pubchem_smiles_{smiles}"
        cached_data = self._load_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        try:
            self._wait_for_rate_limit()
            
            # CID取得
            cid_url = f"{self.base_url}/compound/smiles/{smiles}/cids/JSON"
            cid_response = self.session.get(cid_url, timeout=30)
            cid_response.raise_for_status()
            
            cid_data = cid_response.json()
            if not cid_data.get('IdentifierList', {}).get('CID'):
                return {'smiles': smiles, 'status': 'not_found'}
            
            cid = cid_data['IdentifierList']['CID'][0]
            
            # 化合物詳細情報取得
            compound_url = f"{self.base_url}/compound/cid/{cid}/property/MolecularFormula,MolecularWeight,CanonicalSMILES,IsomericSMILES,InChI,InChIKey,LogP,XLogP/JSON"
            compound_response = self.session.get(compound_url, timeout=30)
            compound_response.raise_for_status()
            
            compound_data = compound_response.json()
            
            result = {
                'smiles': smiles,
                'cid': cid,
                'status': 'success',
                'properties': compound_data.get('PC_Compounds', [{}])[0].get('props', [])
            }
            
            # キャッシュ保存
            self._save_to_cache(cache_key, result)
            return result
            
        except Exception as e:
            print(f"[ERROR] PubChem API failed for {smiles}: {e}")
            return {'smiles': smiles, 'status': 'error', 'error': str(e)}
    
    def get_compound_by_name(self, name: str) -> Dict[str, Any]:
        """
        化合物名から情報取得
        
        Args:
            name: 化合物名
            
        Returns:
            化合物情報辞書
        """
        cache_key = f"pubchem_name_{name}"
        cached_data = self._load_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        try:
            self._wait_for_rate_limit()
            
            # CID取得
            cid_url = f"{self.base_url}/compound/name/{name}/cids/JSON"
            cid_response = self.session.get(cid_url, timeout=30)
            cid_response.raise_for_status()
            
            cid_data = cid_response.json()
            if not cid_data.get('IdentifierList', {}).get('CID'):
                return {'name': name, 'status': 'not_found'}
            
            cid = cid_data['IdentifierList']['CID'][0]
            
            # 化合物詳細情報取得
            compound_url = f"{self.base_url}/compound/cid/{cid}/property/MolecularFormula,MolecularWeight,CanonicalSMILES,IsomericSMILES,InChI,InChIKey,LogP,XLogP/JSON"
            compound_response = self.session.get(compound_url, timeout=30)
            compound_response.raise_for_status()
            
            compound_data = compound_response.json()
            
            result = {
                'name': name,
                'cid': cid,
                'status': 'success',
                'properties': compound_data.get('PC_Compounds', [{}])[0].get('props', [])
            }
            
            # キャッシュ保存
            self._save_to_cache(cache_key, result)
            return result
            
        except Exception as e:
            print(f"[ERROR] PubChem API failed for {name}: {e}")
            return {'name': name, 'status': 'error', 'error': str(e)}

class DrugBankClient(ExternalAPIClient):
    """DrugBank APIクライアント"""
    
    def __init__(self, cache_dir: str = "cache", rate_limit: float = 1.0):
        super().__init__(cache_dir, rate_limit)
        self.base_url = "https://go.drugbank.com"
    
    def search_drug(self, query: str) -> Dict[str, Any]:
        """
        ドラッグ検索
        
        Args:
            query: 検索クエリ
            
        Returns:
            検索結果辞書
        """
        cache_key = f"drugbank_search_{query}"
        cached_data = self._load_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        try:
            self._wait_for_rate_limit()
            
            # 検索URL
            search_url = f"{self.base_url}/search"
            params = {'q': query}
            
            response = self.session.get(search_url, params=params, timeout=30)
            response.raise_for_status()
            
            # 結果解析（簡易版）
            result = {
                'query': query,
                'status': 'success',
                'results': []  # 実際の解析は複雑なため簡易版
            }
            
            # キャッシュ保存
            self._save_to_cache(cache_key, result)
            return result
            
        except Exception as e:
            print(f"[ERROR] DrugBank API failed for {query}: {e}")
            return {'query': query, 'status': 'error', 'error': str(e)}

class ChEMBLClient(ExternalAPIClient):
    """ChEMBL APIクライアント"""
    
    def __init__(self, cache_dir: str = "cache", rate_limit: float = 0.5):
        super().__init__(cache_dir, rate_limit)
        self.base_url = "https://www.ebi.ac.uk/chembl/api/data"
    
    def get_compound_by_smiles(self, smiles: str) -> Dict[str, Any]:
        """
        SMILESから化合物情報取得
        
        Args:
            smiles: SMILES文字列
            
        Returns:
            化合物情報辞書
        """
        cache_key = f"chembl_smiles_{smiles}"
        cached_data = self._load_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        try:
            self._wait_for_rate_limit()
            
            # 化合物検索
            search_url = f"{self.base_url}/molecule"
            params = {
                'molecule_structures__canonical_smiles__exact': smiles,
                'format': 'json',
                'limit': 1
            }
            
            response = self.session.get(search_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            molecules = data.get('molecules', [])
            
            if not molecules:
                return {'smiles': smiles, 'status': 'not_found'}
            
            molecule = molecules[0]
            
            result = {
                'smiles': smiles,
                'chembl_id': molecule.get('molecule_chembl_id'),
                'status': 'success',
                'molecule': molecule
            }
            
            # キャッシュ保存
            self._save_to_cache(cache_key, result)
            return result
            
        except Exception as e:
            print(f"[ERROR] ChEMBL API failed for {smiles}: {e}")
            return {'smiles': smiles, 'status': 'error', 'error': str(e)}
    
    def get_activities_by_chembl_id(self, chembl_id: str) -> Dict[str, Any]:
        """
        ChEMBL IDから活性データ取得
        
        Args:
            chembl_id: ChEMBL ID
            
        Returns:
            活性データ辞書
        """
        cache_key = f"chembl_activities_{chembl_id}"
        cached_data = self._load_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        try:
            self._wait_for_rate_limit()
            
            # 活性データ取得
            activities_url = f"{self.base_url}/activity"
            params = {
                'molecule_chembl_id': chembl_id,
                'format': 'json',
                'limit': 100
            }
            
            response = self.session.get(activities_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            activities = data.get('activities', [])
            
            result = {
                'chembl_id': chembl_id,
                'status': 'success',
                'activities': activities,
                'count': len(activities)
            }
            
            # キャッシュ保存
            self._save_to_cache(cache_key, result)
            return result
            
        except Exception as e:
            print(f"[ERROR] ChEMBL activities API failed for {chembl_id}: {e}")
            return {'chembl_id': chembl_id, 'status': 'error', 'error': str(e)}

class UniProtClient(ExternalAPIClient):
    """UniProt APIクライアント"""
    
    def __init__(self, cache_dir: str = "cache", rate_limit: float = 0.5):
        super().__init__(cache_dir, rate_limit)
        self.base_url = "https://www.uniprot.org/uniprot"
    
    def get_protein_by_id(self, uniprot_id: str) -> Dict[str, Any]:
        """
        UniProt IDからタンパク質情報取得
        
        Args:
            uniprot_id: UniProt ID
            
        Returns:
            タンパク質情報辞書
        """
        cache_key = f"uniprot_id_{uniprot_id}"
        cached_data = self._load_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        try:
            self._wait_for_rate_limit()
            
            # タンパク質情報取得
            protein_url = f"{self.base_url}/{uniprot_id}.json"
            
            response = self.session.get(protein_url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            result = {
                'uniprot_id': uniprot_id,
                'status': 'success',
                'protein': data
            }
            
            # キャッシュ保存
            self._save_to_cache(cache_key, result)
            return result
            
        except Exception as e:
            print(f"[ERROR] UniProt API failed for {uniprot_id}: {e}")
            return {'uniprot_id': uniprot_id, 'status': 'error', 'error': str(e)}
    
    def search_protein_by_name(self, name: str) -> Dict[str, Any]:
        """
        タンパク質名から検索
        
        Args:
            name: タンパク質名
            
        Returns:
            検索結果辞書
        """
        cache_key = f"uniprot_search_{name}"
        cached_data = self._load_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        try:
            self._wait_for_rate_limit()
            
            # 検索URL
            search_url = f"{self.base_url}/?query={name}&format=json"
            
            response = self.session.get(search_url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            result = {
                'name': name,
                'status': 'success',
                'results': data.get('results', [])
            }
            
            # キャッシュ保存
            self._save_to_cache(cache_key, result)
            return result
            
        except Exception as e:
            print(f"[ERROR] UniProt search API failed for {name}: {e}")
            return {'name': name, 'status': 'error', 'error': str(e)}

class ExternalAPIManager:
    """外部API統合マネージャー"""
    
    def __init__(self, cache_dir: str = "cache"):
        """
        初期化
        
        Args:
            cache_dir: キャッシュディレクトリ
        """
        self.cache_dir = cache_dir
        
        # 各APIクライアント初期化
        self.pubchem = PubChemClient(cache_dir, rate_limit=0.1)
        self.drugbank = DrugBankClient(cache_dir, rate_limit=1.0)
        self.chembl = ChEMBLClient(cache_dir, rate_limit=0.5)
        self.uniprot = UniProtClient(cache_dir, rate_limit=0.5)
    
    def get_molecule_info(self, smiles: str) -> Dict[str, Any]:
        """
        分子情報統合取得
        
        Args:
            smiles: SMILES文字列
            
        Returns:
            統合分子情報辞書
        """
        result = {
            'smiles': smiles,
            'pubchem': {},
            'chembl': {},
            'status': 'success'
        }
        
        try:
            # PubChem情報取得
            print(f"[INFO] PubChem情報取得中: {smiles}")
            pubchem_result = self.pubchem.get_compound_by_smiles(smiles)
            result['pubchem'] = pubchem_result
            
            # ChEMBL情報取得
            print(f"[INFO] ChEMBL情報取得中: {smiles}")
            chembl_result = self.chembl.get_compound_by_smiles(smiles)
            result['chembl'] = chembl_result
            
            # ChEMBL IDがある場合、活性データ取得
            if chembl_result.get('status') == 'success' and chembl_result.get('chembl_id'):
                chembl_id = chembl_result['chembl_id']
                print(f"[INFO] ChEMBL活性データ取得中: {chembl_id}")
                activities_result = self.chembl.get_activities_by_chembl_id(chembl_id)
                result['chembl']['activities'] = activities_result
            
        except Exception as e:
            print(f"[ERROR] 分子情報取得失敗: {e}")
            result['status'] = 'error'
            result['error'] = str(e)
        
        return result
    
    def get_protein_info(self, uniprot_id: str) -> Dict[str, Any]:
        """
        タンパク質情報取得
        
        Args:
            uniprot_id: UniProt ID
            
        Returns:
            タンパク質情報辞書
        """
        return self.uniprot.get_protein_by_id(uniprot_id)
    
    def search_protein(self, name: str) -> Dict[str, Any]:
        """
        タンパク質検索
        
        Args:
            name: タンパク質名
            
        Returns:
            検索結果辞書
        """
        return self.uniprot.search_protein_by_name(name)
    
    def batch_molecule_info(self, smiles_list: List[str]) -> List[Dict[str, Any]]:
        """
        バッチ分子情報取得
        
        Args:
            smiles_list: SMILES文字列リスト
            
        Returns:
            分子情報リスト
        """
        print(f"[INFO] バッチ分子情報取得開始: {len(smiles_list)} molecules")
        
        results = []
        for i, smiles in enumerate(tqdm(smiles_list, desc="External API batch")):
            try:
                result = self.get_molecule_info(smiles)
                results.append(result)
                
                # 進捗表示
                if (i + 1) % 10 == 0:
                    print(f"[INFO] 進捗: {i + 1}/{len(smiles_list)}")
                
            except Exception as e:
                print(f"[ERROR] バッチ取得失敗 for {smiles}: {e}")
                results.append({
                    'smiles': smiles,
                    'status': 'error',
                    'error': str(e)
                })
        
        print(f"[INFO] バッチ分子情報取得完了: {len(results)} results")
        return results

def create_external_api_manager(cache_dir: str = "cache") -> ExternalAPIManager:
    """
    外部APIマネージャー作成
    
    Args:
        cache_dir: キャッシュディレクトリ
        
    Returns:
        ExternalAPIManager
    """
    return ExternalAPIManager(cache_dir)

def get_molecule_info(smiles: str, cache_dir: str = "cache") -> Dict[str, Any]:
    """
    分子情報取得（簡易版）
    
    Args:
        smiles: SMILES文字列
        cache_dir: キャッシュディレクトリ
        
    Returns:
        分子情報辞書
    """
    manager = create_external_api_manager(cache_dir)
    return manager.get_molecule_info(smiles)

if __name__ == "__main__":
    # テスト実行
    test_smiles = ["CCO", "CCN", "c1ccccc1"]
    
    manager = ExternalAPIManager()
    results = manager.batch_molecule_info(test_smiles)
    
    for result in results:
        print(f"SMILES: {result['smiles']}")
        print(f"Status: {result['status']}")
        if 'pubchem' in result and result['pubchem'].get('status') == 'success':
            print(f"PubChem CID: {result['pubchem'].get('cid', 'N/A')}")
        if 'chembl' in result and result['chembl'].get('status') == 'success':
            print(f"ChEMBL ID: {result['chembl'].get('chembl_id', 'N/A')}")
        print("-" * 40)
