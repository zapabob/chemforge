#!/usr/bin/env python3
"""
正しいイオンチャンネル・受容体のCHEMBL IDを検索するスクリプト
AMPA、NMDA、GABA-A、GABA-Bの正しいIDを見つけるで〜！
"""

import json
import time
from typing import Dict, List, Optional
from tqdm import tqdm
import logging

# chembl-webresource-clientをインポート
try:
    from chembl_webresource_client.new_client import new_client
except ImportError:
    print("❌ chembl-webresource-client がインストールされていません")
    exit(1)

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IonChannelFinder:
    """イオンチャンネル・受容体の正しいCHEMBL IDを検索するクラス"""
    
    def __init__(self):
        self.target_client = new_client.target
        
        # 検索対象のターゲット
        self.target_searches = {
            "AMPA": [
                "AMPA receptor",
                "glutamate receptor AMPA",
                "GRIA1",
                "GRIA2", 
                "GRIA3",
                "GRIA4",
                "ionotropic glutamate receptor AMPA"
            ],
            "NMDA": [
                "NMDA receptor",
                "glutamate receptor NMDA",
                "GRIN1",
                "GRIN2A",
                "GRIN2B",
                "GRIN2C",
                "GRIN2D",
                "ionotropic glutamate receptor NMDA"
            ],
            "GABA-A": [
                "GABA-A receptor",
                "GABRA1",
                "GABRA2",
                "GABRA3",
                "GABRA4",
                "GABRA5",
                "GABRA6",
                "gamma-aminobutyric acid receptor A",
                "GABAA receptor"
            ],
            "GABA-B": [
                "GABA-B receptor",
                "GABBR1",
                "GABBR2",
                "gamma-aminobutyric acid receptor B",
                "GABAB receptor"
            ]
        }
        
        self.found_targets = {}
        self.errors = []
    
    def search_target(self, target_name: str, search_terms: List[str]) -> Optional[Dict]:
        """
        ターゲットを検索
        
        Args:
            target_name: ターゲット名
            search_terms: 検索語リスト
            
        Returns:
            見つかったターゲット情報
        """
        logger.info(f"🔍 {target_name} を検索中...")
        
        best_matches = []
        
        for term in search_terms:
            try:
                logger.info(f"  📝 検索語: {term}")
                
                # ChEMBLクライアントで検索
                results = self.target_client.search(term)
                
                if results:
                    for result in results[:5]:  # 上位5件をチェック
                        target_info = {
                            'target_name': target_name,
                            'chembl_id': result.get('target_chembl_id'),
                            'pref_name': result.get('pref_name', ''),
                            'organism': result.get('organism', ''),
                            'target_type': result.get('target_type', ''),
                            'search_term': term,
                            'confidence': self._calculate_confidence(target_name, result)
                        }
                        
                        # ヒトのタンパク質で、信頼度が高いものを優先
                        if (target_info['organism'] == 'Homo sapiens' and 
                            target_info['confidence'] > 0.7):
                            best_matches.append(target_info)
                
                time.sleep(0.5)  # API制限対策
                
            except Exception as e:
                logger.error(f"  ❌ 検索語 '{term}' でエラー: {str(e)}")
                continue
        
        if best_matches:
            # 信頼度順にソート
            best_matches.sort(key=lambda x: x['confidence'], reverse=True)
            best_match = best_matches[0]
            
            logger.info(f"  ✅ 最適なマッチ: {best_match['pref_name']} ({best_match['chembl_id']})")
            logger.info(f"  📊 信頼度: {best_match['confidence']:.2f}")
            
            return best_match
        else:
            logger.warning(f"  ❌ {target_name}: 適切なターゲットが見つかりませんでした")
            return None
    
    def _calculate_confidence(self, target_name: str, result: Dict) -> float:
        """
        信頼度を計算
        
        Args:
            target_name: ターゲット名
            result: 検索結果
            
        Returns:
            信頼度 (0.0-1.0)
        """
        confidence = 0.0
        pref_name = result.get('pref_name', '').lower()
        target_type = result.get('target_type', '').lower()
        organism = result.get('organism', '').lower()
        
        # ヒトのタンパク質であることを確認
        if organism == 'homo sapiens':
            confidence += 0.3
        
        # ターゲットタイプが適切であることを確認
        if 'single protein' in target_type:
            confidence += 0.2
        
        # 名前の一致度をチェック
        if target_name.lower() in pref_name:
            confidence += 0.3
        elif any(keyword in pref_name for keyword in ['receptor', 'channel', 'transporter']):
            confidence += 0.2
        
        # 特定のキーワードマッチング
        keyword_matches = {
            'AMPA': ['ampa', 'glutamate', 'gria'],
            'NMDA': ['nmda', 'glutamate', 'grin'],
            'GABA-A': ['gaba', 'gamma-aminobutyric', 'gabra'],
            'GABA-B': ['gaba', 'gamma-aminobutyric', 'gabbr']
        }
        
        if target_name in keyword_matches:
            for keyword in keyword_matches[target_name]:
                if keyword in pref_name:
                    confidence += 0.1
        
        return min(confidence, 1.0)
    
    def find_all_targets(self) -> Dict[str, Dict]:
        """
        全ターゲットを検索
        
        Returns:
            検索結果辞書
        """
        logger.info("🚀 イオンチャンネル・受容体検索開始！")
        
        with tqdm(total=len(self.target_searches), desc="ターゲット検索中") as pbar:
            for target_name, search_terms in self.target_searches.items():
                found_target = self.search_target(target_name, search_terms)
                
                if found_target:
                    self.found_targets[target_name] = found_target
                else:
                    # 検索失敗時はエラー情報を記録
                    self.found_targets[target_name] = {
                        'target_name': target_name,
                        'chembl_id': None,
                        'pref_name': 'Not found',
                        'organism': 'Unknown',
                        'target_type': 'Unknown',
                        'search_term': 'N/A',
                        'confidence': 0.0,
                        'status': 'not_found'
                    }
                    self.errors.append(f"{target_name}: 適切なターゲットが見つかりませんでした")
                
                pbar.update(1)
        
        logger.info(f"✅ 検索完了: {len(self.found_targets)} ターゲット")
        return self.found_targets
    
    def save_results(self, output_file: str = "correct_ion_channels.json"):
        """
        検索結果をJSONファイルに保存
        
        Args:
            output_file: 出力ファイル名
        """
        results = {
            'search_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_targets': len(self.found_targets),
            'found_targets': self.found_targets,
            'errors': self.errors
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 検索結果を保存: {output_file}")
    
    def print_summary(self):
        """検索結果サマリーを表示"""
        print("\n" + "="*60)
        print("🔍 イオンチャンネル・受容体検索結果サマリー")
        print("="*60)
        
        found_count = sum(1 for info in self.found_targets.values() 
                         if info.get('chembl_id') is not None)
        not_found_count = sum(1 for info in self.found_targets.values() 
                             if info.get('chembl_id') is None)
        
        print(f"✅ 発見: {found_count} ターゲット")
        print(f"❌ 未発見: {not_found_count} ターゲット")
        print(f"📊 総数: {len(self.found_targets)} ターゲット")
        
        if self.errors:
            print(f"\n⚠️ エラー: {len(self.errors)} 件")
            for error in self.errors[:5]:  # 最初の5件のみ表示
                print(f"  - {error}")
        
        print("\n📋 詳細結果:")
        for target_name, info in self.found_targets.items():
            if info.get('chembl_id'):
                confidence = info.get('confidence', 0.0)
                print(f"  ✅ {target_name}: {info['chembl_id']} - {info['pref_name']} (信頼度: {confidence:.2f})")
            else:
                print(f"  ❌ {target_name}: 見つかりませんでした")


def main():
    """メイン実行関数"""
    print("🚀 イオンチャンネル・受容体検索スクリプト開始！")
    print("なんｊ風にしゃべるで〜！Don't hold back. Give it your all!!")
    print("💡 AMPA、NMDA、GABA-A、GABA-Bの正しいCHEMBL IDを探すで〜！")
    
    finder = IonChannelFinder()
    
    try:
        # 全ターゲット検索
        finder.find_all_targets()
        
        # 結果表示
        finder.print_summary()
        
        # 結果保存
        finder.save_results()
        
        print("\n🎉 検索完了！correct_ion_channels.json に結果を保存したで〜！")
        
    except KeyboardInterrupt:
        print("\n⚠️ ユーザーによって中断されました")
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {str(e)}")
        logger.exception("予期しないエラー")


if __name__ == "__main__":
    main()
