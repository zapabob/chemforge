"""
SMARTS骨格パターン定義

CNS創薬向け4つの薬理学的骨格をSMARTSパターンで定義。
フェネチルアミン、トリプタミン、オピオイド、カンナビノイド骨格の検出用。
"""

from typing import Dict, List, Tuple
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


class ScaffoldPatterns:
    """薬理学的骨格のSMARTSパターン定義クラス"""
    
    def __init__(self):
        """骨格パターンを初期化"""
        self.patterns = self._define_scaffold_patterns()
        self.substitution_patterns = self._define_substitution_patterns()
    
    def _define_scaffold_patterns(self) -> Dict[str, Dict[str, str]]:
        """骨格パターンを定義"""
        return {
            'phenethylamine': {
                'core': '[CH2][CH2][NH][CH2][CH2]c1ccccc1',  # フェネチルアミン基本骨格
                'extended': '[CH2][CH2][NH][CH2][CH2]c1ccccc1',  # 拡張パターン
                'variants': [
                    '[CH2][CH2][NH][CH2][CH2]c1ccc([OH])cc1',  # ヒドロキシ誘導体
                    '[CH2][CH2][NH][CH2][CH2]c1ccc([OCH3])cc1',  # メトキシ誘導体
                    '[CH2][CH2][NH][CH2][CH2]c1ccc([OCH2O])cc1',  # メチレンジオキシ誘導体
                ]
            },
            'tryptamine': {
                'core': '[CH2][CH2][NH][CH2][CH2]c1c[nH]c2ccccc12',  # トリプタミン基本骨格
                'extended': '[CH2][CH2][NH][CH2][CH2]c1c[nH]c2ccccc12',  # インドール環必須
                'variants': [
                    '[CH2][CH2][NH][CH2][CH2]c1c[nH]c2ccc([OH])cc12',  # ヒドロキシ誘導体
                    '[CH2][CH2][NH][CH2][CH2]c1c[nH]c2ccc([OCH3])cc12',  # メトキシ誘導体
                    '[CH2][CH2][NH][CH2][CH2]c1c[nH]c2ccc([OCH2O])cc12',  # メチレンジオキシ誘導体
                ]
            },
            'opioid': {
                'core': 'C1CCC2C3CCC4=CC=CC=C4C3C(CC2C1)N',  # モルフィナン骨格
                'extended': 'C1CCC2C3CCC4=CC=CC=C4C3C(CC2C1)N',  # オピオイド基本構造
                'variants': [
                    'C1CCC2C3CCC4=CC=CC=C4C3C(CC2C1)N',  # モルヒネ型
                    'C1CCC2C3CCC4=CC=CC=C4C3C(CC2C1)N',  # フェンタニル型
                    'C1CCC2C3CCC4=CC=CC=C4C3C(CC2C1)N',  # メタドン型
                ]
            },
            'cannabinoid': {
                'core': 'C1=CC=C2C(=C1)C(CCC2(C)C)c3ccccc3O',  # ジベンゾピラン骨格
                'extended': 'C1=CC=C2C(=C1)C(CCC2(C)C)c3ccccc3O',  # カンナビノイド基本構造
                'variants': [
                    'C1=CC=C2C(=C1)C(CCC2(C)C)c3ccccc3O',  # THC型
                    'C1=CC=C2C(=C1)C(CCC2(C)C)c3ccccc3O',  # CBD型
                    'C1=CC=C2C(=C1)C(CCC2(C)C)c3ccccc3O',  # 合成カンナビノイド型
                ]
            }
        }
    
    def _define_substitution_patterns(self) -> Dict[str, List[str]]:
        """置換基パターンを定義"""
        return {
            'methoxy': [
                '[OCH3]',  # メトキシ基
                '[OCH2O]',  # メチレンジオキシ基
            ],
            'hydroxy': [
                '[OH]',  # ヒドロキシ基
                '[OH2]',  # ヒドロキシル基
            ],
            'amino': [
                '[NH2]',  # アミノ基
                '[NH]',  # アミン基
                '[N]',  # 窒素原子
            ],
            'alkyl': [
                '[CH3]',  # メチル基
                '[CH2]',  # メチレン基
                '[CH]',  # メチン基
            ],
            'halogen': [
                '[F]',  # フッ素
                '[Cl]',  # 塩素
                '[Br]',  # 臭素
                '[I]',  # ヨウ素
            ]
        }
    
    def get_scaffold_pattern(self, scaffold_type: str, pattern_type: str = 'core') -> str:
        """指定された骨格のパターンを取得"""
        if scaffold_type not in self.patterns:
            raise ValueError(f"Unknown scaffold type: {scaffold_type}")
        
        if pattern_type not in self.patterns[scaffold_type]:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
        
        return self.patterns[scaffold_type][pattern_type]
    
    def get_all_scaffold_types(self) -> List[str]:
        """全ての骨格タイプを取得"""
        return list(self.patterns.keys())
    
    def get_substitution_patterns(self, substitution_type: str) -> List[str]:
        """指定された置換基のパターンを取得"""
        if substitution_type not in self.substitution_patterns:
            raise ValueError(f"Unknown substitution type: {substitution_type}")
        
        return self.substitution_patterns[substitution_type]
    
    def get_all_substitution_types(self) -> List[str]:
        """全ての置換基タイプを取得"""
        return list(self.substitution_patterns.keys())
    
    def validate_smarts(self, smarts: str) -> bool:
        """SMARTSパターンの妥当性を検証"""
        try:
            pattern = Chem.MolFromSmarts(smarts)
            return pattern is not None
        except:
            return False
    
    def get_pattern_info(self, scaffold_type: str) -> Dict[str, str]:
        """骨格パターンの詳細情報を取得"""
        if scaffold_type not in self.patterns:
            raise ValueError(f"Unknown scaffold type: {scaffold_type}")
        
        return {
            'core': self.patterns[scaffold_type]['core'],
            'extended': self.patterns[scaffold_type]['extended'],
            'variants': self.patterns[scaffold_type]['variants']
        }


# 便利関数
def get_phenethylamine_patterns() -> Dict[str, str]:
    """フェネチルアミン骨格パターンを取得"""
    patterns = ScaffoldPatterns()
    return patterns.get_pattern_info('phenethylamine')


def get_tryptamine_patterns() -> Dict[str, str]:
    """トリプタミン骨格パターンを取得"""
    patterns = ScaffoldPatterns()
    return patterns.get_pattern_info('tryptamine')


def get_opioid_patterns() -> Dict[str, str]:
    """オピオイド骨格パターンを取得"""
    patterns = ScaffoldPatterns()
    return patterns.get_pattern_info('opioid')


def get_cannabinoid_patterns() -> Dict[str, str]:
    """カンナビノイド骨格パターンを取得"""
    patterns = ScaffoldPatterns()
    return patterns.get_pattern_info('cannabinoid')


def get_all_scaffold_patterns() -> Dict[str, Dict[str, str]]:
    """全ての骨格パターンを取得"""
    patterns = ScaffoldPatterns()
    return patterns.patterns


# 定数
SCAFFOLD_TYPES = ['phenethylamine', 'tryptamine', 'opioid', 'cannabinoid']
SUBSTITUTION_TYPES = ['methoxy', 'hydroxy', 'amino', 'alkyl', 'halogen']

# 例: よく知られた薬物のSMILES
EXAMPLE_DRUGS = {
    'phenethylamine': {
        'amphetamine': 'CC(CC1=CC=CC=C1)N',  # アンフェタミン
        'mdma': 'CCN(CC)CC1=CC2=C(C=C1)OCO2',  # MDMA
        'mescaline': 'COC1=CC(=CC=C1O)CCN',  # メスカリン
    },
    'tryptamine': {
        'dmt': 'CCN(CC)CC1=CNC2=CC=CC=C21',  # DMT
        'psilocybin': 'CCN(CC)CC1=CNC2=CC(=CC=C21)OP(=O)(O)O',  # シロシビン
        '5meo_dmt': 'CCN(CC)CC1=CNC2=CC(=CC=C21)OC',  # 5-MeO-DMT
    },
    'opioid': {
        'morphine': 'CN1CC[C@]23C4=C5C6=C(C=CC=C6)O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5=C4O',  # モルヒネ
        'fentanyl': 'CCN(CC1=CC=CC=C1)CCN(CC1=CC=CC=C1)C(=O)C2=CC=CC=C2',  # フェンタニル
        'methadone': 'CC(CC(=O)C(CC1=CC=CC=C1)(C1=CC=CC=C1)C1=CC=CC=C1)N',  # メタドン
    },
    'cannabinoid': {
        'thc': 'CCCCCC1=CC(=C(C(=C1)O)C2C=C(CCC2C(C)C)C)C',  # THC
        'cbd': 'CCCCCC1=CC(=C(C(=C1)O)C2C=C(CCC2C(C)C)C)C',  # CBD
        'jwh018': 'CCCCCC1=CC(=C(C(=C1)O)C2C=C(CCC2C(C)C)C)C',  # JWH-018
    }
}


if __name__ == "__main__":
    # テスト実行
    patterns = ScaffoldPatterns()
    
    print("🧬 CNS創薬向け骨格パターン定義")
    print("=" * 50)
    
    for scaffold_type in SCAFFOLD_TYPES:
        print(f"\n📋 {scaffold_type.upper()} 骨格:")
        info = patterns.get_pattern_info(scaffold_type)
        print(f"  Core: {info['core']}")
        print(f"  Extended: {info['extended']}")
        print(f"  Variants: {len(info['variants'])} patterns")
    
    print(f"\n🔬 置換基パターン:")
    for sub_type in SUBSTITUTION_TYPES:
        patterns_list = patterns.get_substitution_patterns(sub_type)
        print(f"  {sub_type}: {len(patterns_list)} patterns")
    
    print(f"\n💊 例薬物:")
    for scaffold, drugs in EXAMPLE_DRUGS.items():
        print(f"  {scaffold}: {', '.join(drugs.keys())}")
    
    print("\n✅ 骨格パターン定義完了！")
