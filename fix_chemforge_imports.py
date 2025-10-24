#!/usr/bin/env python3
"""
ChemForgeのインポートエラーを一括修正するスクリプト
"""

import re

# ファイルを読み込む
with open('chemforge/potency/potency_model.py', 'r', encoding='utf-8') as f:
    content = f.read()

# PWAPETEncoderLayerとPWAPETTransformerを正しいクラス名に修正
content = re.sub(
    r'from \.\.core\.transformer_model import PWAPETEncoderLayer, PWAPETTransformer',
    'from ..core.transformer_model import PhaseEnrichedTransformerLayer, TransformerRegressor',
    content
)

content = re.sub(
    r'encoder_layer = PWAPETEncoderLayer\(',
    'encoder_layer = PhaseEnrichedTransformerLayer(',
    content
)

# ファイルに書き戻す
with open('chemforge/potency/potency_model.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ potency_model.pyのインポートエラーを修正しました！")
