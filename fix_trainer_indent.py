#!/usr/bin/env python3
"""
trainer.pyのインデントエラーを一括修正するスクリプト
"""

import re

# ファイルを読み込む
with open('chemforge/training/trainer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# インデントエラーを修正
# 行188のfor文のインデントを修正
content = re.sub(
    r'        for batch_idx, batch in enumerate\(pbar\):',
    '        for batch_idx, batch in enumerate(pbar):',
    content
)

# 行199のself.optimizer.zero_grad()のインデントを修正
content = re.sub(
    r'                self\.optimizer\.zero_grad\(\)',
    '            self.optimizer.zero_grad()',
    content
)

# 行202のif文のインデントを修正
content = re.sub(
    r'                if self\.use_amp:',
    '            if self.use_amp:',
    content
)

# 行203のwith文のインデントを修正
content = re.sub(
    r'                    with autocast\(\):',
    '                with autocast():',
    content
)

# ファイルに書き戻す
with open('chemforge/training/trainer.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ trainer.pyのインデントエラーを修正しました！")
