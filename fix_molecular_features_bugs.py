#!/usr/bin/env python3
"""
molecular_features.pyのバグを一括修正するスクリプト
"""

import re

# ファイルを読み込む
with open('chemforge/data/molecular_features.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Bug 1: 存在しないモジュールのインポートをコメントアウト
content = re.sub(
    r'from chemforge\.data\.molecular_preprocessor import MolecularPreprocessor',
    '# from chemforge.data.molecular_preprocessor import MolecularPreprocessor',
    content
)
content = re.sub(
    r'from chemforge\.utils\.config_utils import ConfigManager',
    '# from chemforge.utils.config_utils import ConfigManager',
    content
)
content = re.sub(
    r'from chemforge\.utils\.logging_utils import Logger',
    '# from chemforge.utils.logging_utils import Logger',
    content
)
content = re.sub(
    r'from chemforge\.utils\.validation import DataValidator',
    '# from chemforge.utils.validation import DataValidator',
    content
)

# Bug 2: clean_smilesメソッドの呼び出しを修正
content = re.sub(
    r'self\.preprocessor\.clean_smiles\(smiles\)',
    'self._clean_smiles(smiles)',
    content
)

# Bug 3: NumAliphaticCarbocyclesをNumStereocentersに修正
content = re.sub(
    r'Descriptors\.NumAliphaticCarbocycles\(mol\)',
    'Descriptors.NumStereocenters(mol)',
    content
)
content = re.sub(
    r'Descriptors\.NumAliphaticCarbocycles\(mol_3d\)',
    'Descriptors.NumStereocenters(mol_3d)',
    content
)

# Bug 4: SASA計算の修正
content = re.sub(
    r'# 原子別SASA\n\s+atom_sasas = rdFreeSASA\.CalcSASA\(mol_3d, confId=0\)\n\s+features\[\'atom_sasa_mean\'\] = np\.mean\(atom_sasas\) if atom_sasas else 0\n\s+features\[\'atom_sasa_std\'\] = np\.std\(atom_sasas\) if atom_sasas else 0',
    '# 原子別SASA（CalcSASAは単一のfloat値を返す）\n            features[\'atom_sasa_mean\'] = sasa\n            features[\'atom_sasa_std\'] = 0  # 単一値のため標準偏差は0',
    content
)

# Bug 5: CrippenDescriptorsの修正
content = re.sub(
    r'# 分子形状\n\s+features\[\'molecular_volume\'\] = rdMolDescriptors\.CalcCrippenDescriptors\(mol_3d\)\[0\]\n\s+features\[\'molecular_surface\'\] = rdMolDescriptors\.CalcCrippenDescriptors\(mol_3d\)\[1\]',
    '# 分子形状（CrippenDescriptorsは(logP, MR)を返す）\n            crippen_descriptors = rdMolDescriptors.CalcCrippenDescriptors(mol_3d)\n            features[\'logp\'] = crippen_descriptors[0]  # 分配係数\n            features[\'molar_refractivity\'] = crippen_descriptors[1]  # モル屈折率',
    content
)

# Bug 7: ヒドロキシル基検出の修正
content = re.sub(
    r'neighbor\.GetSymbol\(\) == \'H\'',
    'neighbor.GetAtomicNum() == 1',
    content
)

# Bug 8: MACCS fingerprintの修正
content = re.sub(
    r'maccs_fp\.GetNonzeroElements\(\)',
    'list(maccs_fp.GetOnBits())',
    content
)

# Bug 9: RemoveStereochemistryの修正
content = re.sub(
    r'mol = Chem\.RemoveStereochemistry\(mol\)',
    'Chem.RemoveStereochemistry(mol)',
    content
)

# Bug 10: Morgan descriptorsの修正
content = re.sub(
    r'len\(morgan_counts\.GetNonzeroElements\(\)\)',
    'sum(morgan_counts.GetNonzeroElements().values())',
    content
)

# ファイルに書き戻す
with open('chemforge/data/molecular_features.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ molecular_features.pyのバグを修正しました！")
