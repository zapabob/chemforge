"""
Web Scraper for SwissADME

SwissADME (http://www.swissadme.ch/) のWebスクレイピング実装
SMILES入力→ADMET予測取得、レート制限対応、エラーハンドリング
"""

import requests
import time
import json
import re
from typing import Dict, List, Optional, Union, Any
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class SwissADMEScraper:
    """SwissADME Webスクレイピングクラス"""
    
    def __init__(self, rate_limit: float = 1.0, timeout: int = 30, max_retries: int = 3):
        """
        初期化
        
        Args:
            rate_limit: リクエスト間隔（秒）
            timeout: タイムアウト（秒）
            max_retries: 最大リトライ回数
        """
        self.base_url = "http://www.swissadme.ch"
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.max_retries = max_retries
        
        # セッション作成
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # 最後のリクエスト時間
        self.last_request_time = 0
    
    def _wait_for_rate_limit(self):
        """レート制限待機"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit:
            sleep_time = self.rate_limit - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def predict_admet(self, smiles: str) -> Dict[str, Any]:
        """
        SMILESからADMET予測取得
        
        Args:
            smiles: SMILES文字列
            
        Returns:
            ADMET予測結果辞書
        """
        try:
            # レート制限待機
            self._wait_for_rate_limit()
            
            # リトライループ
            for attempt in range(self.max_retries):
                try:
                    # 1. メインページ取得
                    main_response = self.session.get(
                        self.base_url,
                        timeout=self.timeout
                    )
                    main_response.raise_for_status()
                    
                    # 2. SMILES送信
                    prediction_result = self._submit_smiles(smiles)
                    
                    if prediction_result:
                        return prediction_result
                    
                except requests.exceptions.RequestException as e:
                    print(f"[WARNING] Attempt {attempt + 1} failed: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        raise
            
            return {}
            
        except Exception as e:
            print(f"[ERROR] SwissADME prediction failed for {smiles}: {e}")
            return {}
    
    def _submit_smiles(self, smiles: str) -> Optional[Dict[str, Any]]:
        """SMILES送信と結果取得"""
        try:
            # SMILES送信用データ
            data = {
                'smiles': smiles,
                'submit': 'Submit'
            }
            
            # リクエスト送信
            response = self.session.post(
                f"{self.base_url}/index.php",
                data=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # 結果解析
            result = self._parse_results(response.text, smiles)
            return result
            
        except Exception as e:
            print(f"[ERROR] SMILES submission failed: {e}")
            return None
    
    def _parse_results(self, html_content: str, smiles: str) -> Dict[str, Any]:
        """HTML結果解析"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            result = {
                'smiles': smiles,
                'status': 'success',
                'molecular_properties': {},
                'admet_properties': {},
                'drug_likeness': {},
                'errors': []
            }
            
            # 分子特性解析
            self._parse_molecular_properties(soup, result)
            
            # ADMET特性解析
            self._parse_admet_properties(soup, result)
            
            # ドラッグライクネス解析
            self._parse_drug_likeness(soup, result)
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Result parsing failed: {e}")
            return {
                'smiles': smiles,
                'status': 'error',
                'error': str(e)
            }
    
    def _parse_molecular_properties(self, soup: BeautifulSoup, result: Dict[str, Any]):
        """分子特性解析"""
        try:
            # 分子量
            mw_elem = soup.find('td', string=re.compile(r'Molecular weight'))
            if mw_elem and mw_elem.find_next_sibling('td'):
                mw_text = mw_elem.find_next_sibling('td').get_text().strip()
                mw_value = re.search(r'(\d+\.?\d*)', mw_text)
                if mw_value:
                    result['molecular_properties']['molecular_weight'] = float(mw_value.group(1))
            
            # LogP
            logp_elem = soup.find('td', string=re.compile(r'LogP'))
            if logp_elem and logp_elem.find_next_sibling('td'):
                logp_text = logp_elem.find_next_sibling('td').get_text().strip()
                logp_value = re.search(r'(-?\d+\.?\d*)', logp_text)
                if logp_value:
                    result['molecular_properties']['logp'] = float(logp_value.group(1))
            
            # TPSA
            tpsa_elem = soup.find('td', string=re.compile(r'TPSA'))
            if tpsa_elem and tpsa_elem.find_next_sibling('td'):
                tpsa_text = tpsa_elem.find_next_sibling('td').get_text().strip()
                tpsa_value = re.search(r'(\d+\.?\d*)', tpsa_text)
                if tpsa_value:
                    result['molecular_properties']['tpsa'] = float(tpsa_value.group(1))
            
            # HBD/HBA
            hbd_elem = soup.find('td', string=re.compile(r'HBD'))
            if hbd_elem and hbd_elem.find_next_sibling('td'):
                hbd_text = hbd_elem.find_next_sibling('td').get_text().strip()
                hbd_value = re.search(r'(\d+)', hbd_text)
                if hbd_value:
                    result['molecular_properties']['hbd'] = int(hbd_value.group(1))
            
            hba_elem = soup.find('td', string=re.compile(r'HBA'))
            if hba_elem and hba_elem.find_next_sibling('td'):
                hba_text = hba_elem.find_next_sibling('td').get_text().strip()
                hba_value = re.search(r'(\d+)', hba_text)
                if hba_value:
                    result['molecular_properties']['hba'] = int(hba_value.group(1))
            
        except Exception as e:
            result['errors'].append(f"Molecular properties parsing error: {e}")
    
    def _parse_admet_properties(self, soup: BeautifulSoup, result: Dict[str, Any]):
        """ADMET特性解析"""
        try:
            # GI absorption
            gi_elem = soup.find('td', string=re.compile(r'GI absorption'))
            if gi_elem and gi_elem.find_next_sibling('td'):
                gi_text = gi_elem.find_next_sibling('td').get_text().strip()
                result['admet_properties']['gi_absorption'] = gi_text
            
            # BBB permeant
            bbb_elem = soup.find('td', string=re.compile(r'BBB permeant'))
            if bbb_elem and bbb_elem.find_next_sibling('td'):
                bbb_text = bbb_elem.find_next_sibling('td').get_text().strip()
                result['admet_properties']['bbb_permeant'] = bbb_text
            
            # P-gp substrate
            pgp_elem = soup.find('td', string=re.compile(r'P-gp substrate'))
            if pgp_elem and pgp_elem.find_next_sibling('td'):
                pgp_text = pgp_elem.find_next_sibling('td').get_text().strip()
                result['admet_properties']['pgp_substrate'] = pgp_text
            
            # CYP1A2 inhibitor
            cyp1a2_elem = soup.find('td', string=re.compile(r'CYP1A2 inhibitor'))
            if cyp1a2_elem and cyp1a2_elem.find_next_sibling('td'):
                cyp1a2_text = cyp1a2_elem.find_next_sibling('td').get_text().strip()
                result['admet_properties']['cyp1a2_inhibitor'] = cyp1a2_text
            
            # CYP2C19 inhibitor
            cyp2c19_elem = soup.find('td', string=re.compile(r'CYP2C19 inhibitor'))
            if cyp2c19_elem and cyp2c19_elem.find_next_sibling('td'):
                cyp2c19_text = cyp2c19_elem.find_next_sibling('td').get_text().strip()
                result['admet_properties']['cyp2c19_inhibitor'] = cyp2c19_text
            
            # CYP2C9 inhibitor
            cyp2c9_elem = soup.find('td', string=re.compile(r'CYP2C9 inhibitor'))
            if cyp2c9_elem and cyp2c9_elem.find_next_sibling('td'):
                cyp2c9_text = cyp2c9_elem.find_next_sibling('td').get_text().strip()
                result['admet_properties']['cyp2c9_inhibitor'] = cyp2c9_text
            
            # CYP2D6 inhibitor
            cyp2d6_elem = soup.find('td', string=re.compile(r'CYP2D6 inhibitor'))
            if cyp2d6_elem and cyp2d6_elem.find_next_sibling('td'):
                cyp2d6_text = cyp2d6_elem.find_next_sibling('td').get_text().strip()
                result['admet_properties']['cyp2d6_inhibitor'] = cyp2d6_text
            
            # CYP3A4 inhibitor
            cyp3a4_elem = soup.find('td', string=re.compile(r'CYP3A4 inhibitor'))
            if cyp3a4_elem and cyp3a4_elem.find_next_sibling('td'):
                cyp3a4_text = cyp3a4_elem.find_next_sibling('td').get_text().strip()
                result['admet_properties']['cyp3a4_inhibitor'] = cyp3a4_text
            
        except Exception as e:
            result['errors'].append(f"ADMET properties parsing error: {e}")
    
    def _parse_drug_likeness(self, soup: BeautifulSoup, result: Dict[str, Any]):
        """ドラッグライクネス解析"""
        try:
            # Lipinski's Rule of Five
            lipinski_elem = soup.find('td', string=re.compile(r'Lipinski'))
            if lipinski_elem and lipinski_elem.find_next_sibling('td'):
                lipinski_text = lipinski_elem.find_next_sibling('td').get_text().strip()
                result['drug_likeness']['lipinski'] = lipinski_text
            
            # Veber's Rule
            veber_elem = soup.find('td', string=re.compile(r'Veber'))
            if veber_elem and veber_elem.find_next_sibling('td'):
                veber_text = veber_elem.find_next_sibling('td').get_text().strip()
                result['drug_likeness']['veber'] = veber_text
            
            # Egan's Rule
            egan_elem = soup.find('td', string=re.compile(r'Egan'))
            if egan_elem and egan_elem.find_next_sibling('td'):
                egan_text = egan_elem.find_next_sibling('td').get_text().strip()
                result['drug_likeness']['egan'] = egan_text
            
            # Muegge's Rule
            muegge_elem = soup.find('td', string=re.compile(r'Muegge'))
            if muegge_elem and muegge_elem.find_next_sibling('td'):
                muegge_text = muegge_elem.find_next_sibling('td').get_text().strip()
                result['drug_likeness']['muegge'] = muegge_text
            
        except Exception as e:
            result['errors'].append(f"Drug likeness parsing error: {e}")
    
    def predict_batch(self, smiles_list: List[str]) -> List[Dict[str, Any]]:
        """
        バッチ予測
        
        Args:
            smiles_list: SMILES文字列リスト
            
        Returns:
            予測結果リスト
        """
        print(f"[INFO] SwissADME batch prediction開始: {len(smiles_list)} molecules")
        
        results = []
        for i, smiles in enumerate(tqdm(smiles_list, desc="SwissADME prediction")):
            try:
                result = self.predict_admet(smiles)
                results.append(result)
                
                # 進捗表示
                if (i + 1) % 10 == 0:
                    print(f"[INFO] 進捗: {i + 1}/{len(smiles_list)}")
                
            except Exception as e:
                print(f"[ERROR] Batch prediction failed for {smiles}: {e}")
                results.append({
                    'smiles': smiles,
                    'status': 'error',
                    'error': str(e)
                })
        
        print(f"[INFO] SwissADME batch prediction完了: {len(results)} results")
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """
        結果保存
        
        Args:
            results: 予測結果リスト
            output_path: 出力パス
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"[INFO] 結果保存完了: {output_path}")
        except Exception as e:
            print(f"[ERROR] 結果保存失敗: {e}")
    
    def load_results(self, input_path: str) -> List[Dict[str, Any]]:
        """
        結果読み込み
        
        Args:
            input_path: 入力パス
            
        Returns:
            予測結果リスト
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            print(f"[INFO] 結果読み込み完了: {len(results)} results")
            return results
        except Exception as e:
            print(f"[ERROR] 結果読み込み失敗: {e}")
            return []

def create_swissadme_scraper(rate_limit: float = 1.0) -> SwissADMEScraper:
    """
    SwissADMEスクレイパー作成
    
    Args:
        rate_limit: レート制限（秒）
        
    Returns:
        SwissADMEScraper
    """
    return SwissADMEScraper(rate_limit=rate_limit)

def predict_admet_batch(smiles_list: List[str], output_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    バッチADMET予測
    
    Args:
        smiles_list: SMILES文字列リスト
        output_path: 出力パス（オプション）
        
    Returns:
        予測結果リスト
    """
    scraper = create_swissadme_scraper()
    results = scraper.predict_batch(smiles_list)
    
    if output_path:
        scraper.save_results(results, output_path)
    
    return results

if __name__ == "__main__":
    # テスト実行
    test_smiles = ["CCO", "CCN", "c1ccccc1"]
    
    scraper = SwissADMEScraper()
    results = scraper.predict_batch(test_smiles)
    
    for result in results:
        print(f"SMILES: {result['smiles']}")
        print(f"Status: {result['status']}")
        if 'molecular_properties' in result:
            print(f"Molecular Weight: {result['molecular_properties'].get('molecular_weight', 'N/A')}")
        print("-" * 40)
