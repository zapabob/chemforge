#!/usr/bin/env python3
"""
ChEMBL ID検証スクリプト v2
chembl-webresource-clientライブラリを使って正しいCHEMBL IDを取得するで〜！
"""

import json
import time
from typing import Dict, List, Optional
from tqdm import tqdm
import logging

# chembl-webresource-clientをインポート
try:
    from chembl_webresource_client.new_client import new_client
    from chembl_webresource_client.unichem import unichem_client
except ImportError:
    print("❌ chembl-webresource-client がインストールされていません")
    print("💡 インストール方法: pip install chembl-webresource-client")
    exit(1)

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChEMBLIDVerifierV2:
    """ChEMBL ID検証クラス v2 - chembl-webresource-client使用"""
    
    def __init__(self):
        # ChEMBLクライアント初期化
        self.target_client = new_client.target
        self.molecule_client = new_client.molecule
        self.assay_client = new_client.assay
        
        # 現在のターゲットマッピング
        self.current_targets = {
            "DAT": {"chembl_id": "CHEMBL238", "name": "Dopamine Transporter", "type": "Transporter"},
            "5HT2A": {"chembl_id": "CHEMBL224", "name": "5-HT2A receptor", "type": "GPCR"},
            "5HT1A": {"chembl_id": "CHEMBL214", "name": "5-HT1A receptor", "type": "GPCR"},
            "D1": {"chembl_id": "CHEMBL2056", "name": "D1 receptor", "type": "GPCR"},
            "D2": {"chembl_id": "CHEMBL217", "name": "D2 receptor", "type": "GPCR"},
            "CB1": {"chembl_id": "CHEMBL218", "name": "CB1 receptor", "type": "GPCR"},
            "CB2": {"chembl_id": "CHEMBL253", "name": "CB2 receptor", "type": "GPCR"},
            "MOR": {"chembl_id": "CHEMBL233", "name": "Mu opioid receptor", "type": "GPCR"},
            "DOR": {"chembl_id": "CHEMBL236", "name": "Delta opioid receptor", "type": "GPCR"},
            "KOR": {"chembl_id": "CHEMBL237", "name": "Kappa opioid receptor", "type": "GPCR"},
            "NOP": {"chembl_id": "CHEMBL2014", "name": "Nociceptin receptor", "type": "GPCR"},
            "SERT": {"chembl_id": "CHEMBL228", "name": "Serotonin Transporter", "type": "Transporter"},
            "NET": {"chembl_id": "CHEMBL222", "name": "Norepinephrine Transporter", "type": "Transporter"},
            "AMPA": {"chembl_id": "CHEMBL1908", "name": "AMPA receptor", "type": "Ion Channel"},
            "NMDA": {"chembl_id": "CHEMBL1910", "name": "NMDA receptor", "type": "Ion Channel"},
            "GABA-A": {"chembl_id": "CHEMBL1909", "name": "GABA-A receptor", "type": "Ion Channel"},
            "GABA-B": {"chembl_id": "CHEMBL1911", "name": "GABA-B receptor", "type": "GPCR"}
        }
        
        self.verified_targets = {}
        self.errors = []
    
    def verify_target_id(self, target_name: str, chembl_id: str) -> Optional[Dict]:
        """
        単一ターゲットのCHEMBL IDを検証
        
        Args:
            target_name: ターゲット名
            chembl_id: 現在のCHEMBL ID
            
        Returns:
            検証結果辞書
        """
        try:
            logger.info(f"🔍 {target_name} ({chembl_id}) を検証中...")
            
            # ChEMBLクライアントでターゲット情報を取得
            target_info = self.target_client.get(chembl_id)
            
            if target_info:
                verified_info = {
                    'target_name': target_name,
                    'chembl_id': chembl_id,
                    'pref_name': target_info.get('pref_name', ''),
                    'organism': target_info.get('organism', ''),
                    'target_type': target_info.get('target_type', ''),
                    'status': 'verified',
                    'confidence': 'high'
                }
                
                logger.info(f"✅ {target_name}: {target_info.get('pref_name', 'N/A')}")
                return verified_info
            else:
                logger.warning(f"❌ {target_name} ({chembl_id}): ターゲットが見つかりません")
                return self._search_alternative_id(target_name)
                
        except Exception as e:
            logger.error(f"❌ {target_name} ({chembl_id}): エラー - {str(e)}")
            self.errors.append(f"{target_name}: {str(e)}")
            return None
    
    def _search_alternative_id(self, target_name: str) -> Optional[Dict]:
        """
        代替CHEMBL IDを検索
        
        Args:
            target_name: ターゲット名
            
        Returns:
            代替ID情報
        """
        try:
            # ターゲット名で検索
            search_terms = [
                target_name,
                self.current_targets[target_name]['name'],
                target_name.replace('-', ' '),
                target_name.replace('_', ' ')
            ]
            
            for term in search_terms:
                try:
                    # ChEMBLクライアントで検索
                    results = self.target_client.search(term)
                    
                    if results and len(results) > 0:
                        best_match = results[0]
                        logger.info(f"🔍 {target_name} の代替ID発見: {best_match.get('target_chembl_id')}")
                        
                        return {
                            'target_name': target_name,
                            'chembl_id': best_match.get('target_chembl_id'),
                            'pref_name': best_match.get('pref_name', ''),
                            'organism': best_match.get('organism', ''),
                            'target_type': best_match.get('target_type', ''),
                            'status': 'alternative_found',
                            'confidence': 'medium'
                        }
                except Exception as e:
                    logger.debug(f"検索語 '{term}' でエラー: {str(e)}")
                    continue
            
            logger.warning(f"❌ {target_name}: 代替IDも見つかりませんでした")
            return None
            
        except Exception as e:
            logger.error(f"❌ {target_name}: 代替検索エラー - {str(e)}")
            return None
    
    def verify_all_targets(self) -> Dict[str, Dict]:
        """
        全ターゲットのCHEMBL IDを検証
        
        Returns:
            検証結果辞書
        """
        logger.info("🚀 ChEMBL ID検証開始！chembl-webresource-client使用版")
        
        with tqdm(total=len(self.current_targets), desc="CHEMBL ID検証中") as pbar:
            for target_name, target_info in self.current_targets.items():
                chembl_id = target_info['chembl_id']
                
                verified_info = self.verify_target_id(target_name, chembl_id)
                
                if verified_info:
                    self.verified_targets[target_name] = verified_info
                else:
                    # 検証失敗時は元の情報を保持
                    self.verified_targets[target_name] = {
                        'target_name': target_name,
                        'chembl_id': chembl_id,
                        'pref_name': target_info['name'],
                        'organism': 'Unknown',
                        'target_type': target_info['type'],
                        'status': 'verification_failed',
                        'confidence': 'low'
                    }
                
                pbar.update(1)
                time.sleep(0.5)  # API制限対策
        
        logger.info(f"✅ 検証完了: {len(self.verified_targets)} ターゲット")
        return self.verified_targets
    
    def generate_updated_mapping(self) -> Dict:
        """
        更新されたターゲットマッピングを生成
        
        Returns:
            更新されたマッピング辞書
        """
        updated_mapping = {}
        
        for target_name, verified_info in self.verified_targets.items():
            updated_mapping[target_name] = {
                'chembl_id': verified_info['chembl_id'],
                'name': verified_info['pref_name'],
                'type': verified_info['target_type']
            }
        
        return updated_mapping
    
    def save_results(self, output_file: str = "correct_chembl_ids_v2.json"):
        """
        検証結果をJSONファイルに保存
        
        Args:
            output_file: 出力ファイル名
        """
        results = {
            'verification_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_targets': len(self.verified_targets),
            'verified_targets': self.verified_targets,
            'errors': self.errors,
            'updated_mapping': self.generate_updated_mapping()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 検証結果を保存: {output_file}")
    
    def print_summary(self):
        """検証結果サマリーを表示"""
        print("\n" + "="*60)
        print("🔍 ChEMBL ID検証結果サマリー (chembl-webresource-client版)")
        print("="*60)
        
        verified_count = sum(1 for info in self.verified_targets.values() 
                           if info['status'] == 'verified')
        alternative_count = sum(1 for info in self.verified_targets.values() 
                              if info['status'] == 'alternative_found')
        failed_count = sum(1 for info in self.verified_targets.values() 
                         if info['status'] == 'verification_failed')
        
        print(f"✅ 検証成功: {verified_count} ターゲット")
        print(f"🔄 代替ID発見: {alternative_count} ターゲット")
        print(f"❌ 検証失敗: {failed_count} ターゲット")
        print(f"📊 総数: {len(self.verified_targets)} ターゲット")
        
        if self.errors:
            print(f"\n⚠️ エラー: {len(self.errors)} 件")
            for error in self.errors[:5]:  # 最初の5件のみ表示
                print(f"  - {error}")
        
        print("\n📋 詳細結果:")
        for target_name, info in self.verified_targets.items():
            status_emoji = {
                'verified': '✅',
                'alternative_found': '🔄',
                'verification_failed': '❌'
            }.get(info['status'], '❓')
            
            print(f"  {status_emoji} {target_name}: {info['chembl_id']} - {info['pref_name']}")


def main():
    """メイン実行関数"""
    print("🚀 ChEMBL ID検証スクリプト v2開始！")
    print("なんｊ風にしゃべるで〜！Don't hold back. Give it your all!!")
    print("💡 chembl-webresource-clientライブラリ使用版")
    
    verifier = ChEMBLIDVerifierV2()
    
    try:
        # 全ターゲット検証
        verifier.verify_all_targets()
        
        # 結果表示
        verifier.print_summary()
        
        # 結果保存
        verifier.save_results()
        
        print("\n🎉 検証完了！correct_chembl_ids_v2.json に結果を保存したで〜！")
        
    except KeyboardInterrupt:
        print("\n⚠️ ユーザーによって中断されました")
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {str(e)}")
        logger.exception("予期しないエラー")


if __name__ == "__main__":
    main()
