#!/usr/bin/env python3
"""
ChemForgeのインポートエラーを一括修正するスクリプト
"""

import re

# ファイルを読み込む
with open('chemforge/training/metrics.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 重複したelse文を修正
content = re.sub(
    r'        else:\n            # 自動判定\n            if self\._is_classification_task\(predictions, targets\):\n                metrics\.update\(self\._calculate_classification_metrics\(predictions, targets\)\)\n        else:\n                metrics\.update\(self\._calculate_regression_metrics\(predictions, targets\)\)',
    '        else:\n            # 自動判定\n            if self._is_classification_task(predictions, targets):\n                metrics.update(self._calculate_classification_metrics(predictions, targets))\n            else:\n                metrics.update(self._calculate_regression_metrics(predictions, targets))',
    content
)

# ファイルに書き戻す
with open('chemforge/training/metrics.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ metrics.pyのシンタックスエラーを修正しました！")
