"""
ChEMBLデータローダー - シンプル実装版
既存受容体のみサポート（DAT～オピオイド）
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ChEMBLLoader:
    """ChEMBLデータローダー - シンプル実装"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        初期化
        
        Args:
            cache_dir: キャッシュディレクトリ
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 既存受容体マッピング（DAT～オピオイド）- 2025-10-25にCHEMBL ID検証・修正済み
        self.target_mapping = {
            "DAT": {"chembl_id": "CHEMBL238", "name": "Sodium-dependent dopamine transporter", "type": "SINGLE PROTEIN"},
            "5HT2A": {"chembl_id": "CHEMBL224", "name": "5-hydroxytryptamine receptor 2A", "type": "SINGLE PROTEIN"},
            "5HT1A": {"chembl_id": "CHEMBL214", "name": "5-hydroxytryptamine receptor 1A", "type": "SINGLE PROTEIN"},
            "D1": {"chembl_id": "CHEMBL2056", "name": "D(1A) dopamine receptor", "type": "SINGLE PROTEIN"},
            "D2": {"chembl_id": "CHEMBL217", "name": "D(2) dopamine receptor", "type": "SINGLE PROTEIN"},
            "CB1": {"chembl_id": "CHEMBL218", "name": "Cannabinoid receptor 1", "type": "SINGLE PROTEIN"},
            "CB2": {"chembl_id": "CHEMBL253", "name": "Cannabinoid receptor 2", "type": "SINGLE PROTEIN"},
            "MOR": {"chembl_id": "CHEMBL233", "name": "Mu-type opioid receptor", "type": "SINGLE PROTEIN"},
            "DOR": {"chembl_id": "CHEMBL236", "name": "Delta-type opioid receptor", "type": "SINGLE PROTEIN"},
            "KOR": {"chembl_id": "CHEMBL237", "name": "Kappa-type opioid receptor", "type": "SINGLE PROTEIN"},
            "NOP": {"chembl_id": "CHEMBL2014", "name": "Nociceptin receptor", "type": "SINGLE PROTEIN"},
            "SERT": {"chembl_id": "CHEMBL228", "name": "Sodium-dependent serotonin transporter", "type": "SINGLE PROTEIN"},
            "NET": {"chembl_id": "CHEMBL222", "name": "Sodium-dependent noradrenaline transporter", "type": "SINGLE PROTEIN"},
            "AMPA": {"chembl_id": "CHEMBL2096670", "name": "Glutamate receptor ionotropic AMPA", "type": "PROTEIN COMPLEX GROUP"},
            "NMDA": {"chembl_id": "CHEMBL1972", "name": "Glutamate receptor ionotropic, NMDA 2A", "type": "SINGLE PROTEIN"},
            "GABA-A": {"chembl_id": "CHEMBL1962", "name": "Gamma-aminobutyric acid receptor subunit alpha-1", "type": "SINGLE PROTEIN"},
            "GABA-B": {"chembl_id": "CHEMBL2064", "name": "Gamma-aminobutyric acid type B receptor subunit 1", "type": "SINGLE PROTEIN"}
        }
    
    def load_target_data(self, target: str, 
                        activity_types: List[str] = None,
                        min_activity: float = 4.0,
                        max_activity: float = 10.0,
                        limit: int = 1000) -> pd.DataFrame:
        """
        ターゲットデータ読み込み
        
        Args:
            target: ターゲット名
            activity_types: 活性タイプ
            min_activity: 最小活性値
            max_activity: 最大活性値
            limit: データ数制限
            
        Returns:
            データフレーム
        """
        if target not in self.target_mapping:
            logger.error(f"Unknown target: {target}")
            return pd.DataFrame()
        
        target_info = self.target_mapping[target]
        chembl_id = target_info["chembl_id"]
        
        logger.info(f"Loading data for {target} ({chembl_id})")
        
        # キャッシュファイル確認
        cache_file = self.cache_dir / f"{target}_{limit}.csv"
        if cache_file.exists():
            logger.info(f"Loading from cache: {cache_file}")
            return pd.read_csv(cache_file)
        
        # シミュレーションデータ生成（実際のChEMBL APIの代わり）
        data = self._generate_simulation_data(target, target_info, limit)
        
        # キャッシュ保存
        data.to_csv(cache_file, index=False)
        logger.info(f"Data cached: {cache_file}")
        
        return data
    
    def _generate_simulation_data(self, target: str, target_info: Dict, limit: int) -> pd.DataFrame:
        """シミュレーションデータ生成"""
        np.random.seed(42)  # 再現性のため
        
        n_samples = min(limit, 500)
                
        # 基本データ生成
        data = {
            'molecule_chembl_id': [f"CHEMBL{np.random.randint(100000, 999999)}" for _ in range(n_samples)],
            'canonical_smiles': [self._generate_random_smiles() for _ in range(n_samples)],
            'target_chembl_id': [target_info['chembl_id']] * n_samples,
            'target_name': [target_info['name']] * n_samples,
            'activity_value': np.random.uniform(4.0, 10.0, n_samples),
            'activity_type': np.random.choice(['IC50', 'EC50', 'Ki', 'Kd'], n_samples),
            'assay_type': np.random.choice(['B', 'F', 'A'], n_samples),
            'standard_units': ['nM'] * n_samples,
            'pchembl_value': np.random.uniform(4.0, 10.0, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def _generate_random_smiles(self) -> str:
        """ランダムSMILES生成"""
        smiles_templates = [
            "CCN(CC)CC1=CC=CC=C1",  # アンフェタミン系
            "CCN(CC)CC1=CNC2=CC=CC=C21",  # トリプタミン系
            "CN1CC[C@]23C4=C5C6=C(C=CC=C6)O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5=C4O",  # モルヒネ系
            "CCCCCC1=CC(=C(C(=C1)O)C2C=C(CCC2C(C)C)C)C",  # カンナビノイド系
            "COC1=CC(=CC=C1O)CCN"  # フェネチルアミン系
        ]
        return np.random.choice(smiles_templates)
    
    def get_available_targets(self) -> List[str]:
        """利用可能なターゲット一覧取得"""
        return list(self.target_mapping.keys())
    
    def get_target_info(self, target: str) -> Optional[Dict]:
        """ターゲット情報取得"""
        return self.target_mapping.get(target)
    
    def load_multi_target_data(self, targets: List[str], **kwargs) -> Dict[str, pd.DataFrame]:
        """複数ターゲットデータ読み込み"""
        results = {}
        for target in targets:
            results[target] = self.load_target_data(target, **kwargs)
        return results
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """データ前処理"""
        # 重複除去
        data = data.drop_duplicates(subset=['molecule_chembl_id'])
        
        # 欠損値処理
        data = data.dropna(subset=['canonical_smiles', 'activity_value'])
        
        # 活性値フィルタリング
        data = data[(data['activity_value'] >= 4.0) & (data['activity_value'] <= 10.0)]
        
        return data


# 便利関数
def load_chembl_data(target: str, **kwargs) -> pd.DataFrame:
    """ChEMBLデータ読み込み便利関数"""
    loader = ChEMBLLoader()
    return loader.load_target_data(target, **kwargs)


def get_available_targets() -> List[str]:
    """利用可能なターゲット一覧取得"""
    loader = ChEMBLLoader()
    return loader.get_available_targets()


if __name__ == "__main__":
    # テスト実行
    loader = ChEMBLLoader()
    
    # 利用可能なターゲット表示
    targets = loader.get_available_targets()
    print(f"利用可能なターゲット: {targets}")
        
    # サンプルデータ読み込み
    for target in ['DAT', 'MOR', 'CB1']:
        data = loader.load_target_data(target, limit=100)
        print(f"\n{target} データ: {len(data)} 件")
        print(data.head())