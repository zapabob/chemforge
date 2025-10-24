"""
骨格検出器クラス

SMARTSパターンマッチングを使用して、分子内の薬理学的骨格を検出。
フェネチルアミン、トリプタミン、オピオイド、カンナビノイド骨格の検出と置換基情報の抽出。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from .scaffold_patterns import ScaffoldPatterns, SCAFFOLD_TYPES, SUBSTITUTION_TYPES


class ScaffoldDetector:
    """薬理学的骨格検出器クラス"""
    
    def __init__(self, verbose: bool = False):
        """
        骨格検出器を初期化
        
        Args:
            verbose: 詳細ログ出力フラグ
        """
        self.verbose = verbose
        self.patterns = ScaffoldPatterns()
        self._compiled_patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[str, Dict[str, Chem.Mol]]:
        """SMARTSパターンをコンパイル"""
        compiled = {}
        
        for scaffold_type in SCAFFOLD_TYPES:
            compiled[scaffold_type] = {}
            pattern_info = self.patterns.get_pattern_info(scaffold_type)
            
            # コアパターンをコンパイル
            core_pattern = Chem.MolFromSmarts(pattern_info['core'])
            if core_pattern is not None:
                compiled[scaffold_type]['core'] = core_pattern
            else:
                if self.verbose:
                    print(f"Warning: Failed to compile core pattern for {scaffold_type}")
            
            # 拡張パターンをコンパイル
            extended_pattern = Chem.MolFromSmarts(pattern_info['extended'])
            if extended_pattern is not None:
                compiled[scaffold_type]['extended'] = extended_pattern
            else:
                if self.verbose:
                    print(f"Warning: Failed to compile extended pattern for {scaffold_type}")
            
            # バリエーションパターンをコンパイル
            compiled[scaffold_type]['variants'] = []
            for variant in pattern_info['variants']:
                variant_pattern = Chem.MolFromSmarts(variant)
                if variant_pattern is not None:
                    compiled[scaffold_type]['variants'].append(variant_pattern)
                else:
                    if self.verbose:
                        print(f"Warning: Failed to compile variant pattern for {scaffold_type}")
        
        return compiled
    
    def detect_scaffolds(self, mol: Chem.Mol) -> Dict[str, int]:
        """
        分子内の骨格を検出
        
        Args:
            mol: RDKit分子オブジェクト
            
        Returns:
            骨格タイプごとの検出数
        """
        if mol is None:
            return {scaffold_type: 0 for scaffold_type in SCAFFOLD_TYPES}
        
        results = {}
        
        for scaffold_type in SCAFFOLD_TYPES:
            count = 0
            
            # コアパターンで検出
            if 'core' in self._compiled_patterns[scaffold_type]:
                core_matches = mol.GetSubstructMatches(self._compiled_patterns[scaffold_type]['core'])
                count += len(core_matches)
            
            # 拡張パターンで検出
            if 'extended' in self._compiled_patterns[scaffold_type]:
                extended_matches = mol.GetSubstructMatches(self._compiled_patterns[scaffold_type]['extended'])
                count += len(extended_matches)
            
            # バリエーションパターンで検出
            for variant_pattern in self._compiled_patterns[scaffold_type]['variants']:
                variant_matches = mol.GetSubstructMatches(variant_pattern)
                count += len(variant_matches)
            
            results[scaffold_type] = count
            
            if self.verbose and count > 0:
                print(f"Detected {count} {scaffold_type} scaffold(s)")
        
        return results
    
    def get_substitution_info(self, mol: Chem.Mol, scaffold_type: str) -> Dict[str, List[Dict]]:
        """
        指定された骨格の置換基情報を取得
        
        Args:
            mol: RDKit分子オブジェクト
            scaffold_type: 骨格タイプ
            
        Returns:
            置換基情報の辞書
        """
        if scaffold_type not in SCAFFOLD_TYPES:
            raise ValueError(f"Unknown scaffold type: {scaffold_type}")
        
        if mol is None:
            return {sub_type: [] for sub_type in SUBSTITUTION_TYPES}
        
        substitution_info = {}
        
        for sub_type in SUBSTITUTION_TYPES:
            substitution_info[sub_type] = []
            patterns = self.patterns.get_substitution_patterns(sub_type)
            
            for pattern in patterns:
                try:
                    substruct_pattern = Chem.MolFromSmarts(pattern)
                    if substruct_pattern is not None:
                        matches = mol.GetSubstructMatches(substruct_pattern)
                        for match in matches:
                            substitution_info[sub_type].append({
                                'pattern': pattern,
                                'atoms': list(match),
                                'count': len(match)
                            })
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Failed to process substitution pattern {pattern}: {e}")
        
        return substitution_info
    
    def extract_scaffold_features(self, mol: Chem.Mol) -> np.ndarray:
        """
        骨格特徴量を抽出
        
        Args:
            mol: RDKit分子オブジェクト
            
        Returns:
            骨格特徴量ベクトル (20次元)
        """
        if mol is None:
            return np.zeros(20, dtype=np.float32)
        
        # 骨格検出
        scaffold_counts = self.detect_scaffolds(mol)
        
        # 特徴量ベクトルを初期化
        features = np.zeros(20, dtype=np.float32)
        
        # 各骨格の有無フラグ (4次元)
        for i, scaffold_type in enumerate(SCAFFOLD_TYPES):
            features[i] = 1.0 if scaffold_counts[scaffold_type] > 0 else 0.0
        
        # 各骨格のカウント (4次元)
        for i, scaffold_type in enumerate(SCAFFOLD_TYPES):
            features[4 + i] = float(scaffold_counts[scaffold_type])
        
        # 主要置換基情報 (12次元: メトキシ、メチレンジオキシ、ヒドロキシ)
        substitution_features = self._extract_substitution_features(mol)
        features[8:20] = substitution_features
        
        return features
    
    def _extract_substitution_features(self, mol: Chem.Mol) -> np.ndarray:
        """置換基特徴量を抽出"""
        features = np.zeros(12, dtype=np.float32)
        
        # メトキシ基 (3次元)
        methoxy_info = self.get_substitution_info(mol, 'phenethylamine')['methoxy']
        features[0] = len(methoxy_info)  # メトキシ基数
        features[1] = 1.0 if len(methoxy_info) > 0 else 0.0  # メトキシ基有無
        features[2] = sum(info['count'] for info in methoxy_info)  # メトキシ基原子数
        
        # メチレンジオキシ基 (3次元)
        methylenedioxy_info = self.get_substitution_info(mol, 'phenethylamine')['methoxy']
        features[3] = len(methylenedioxy_info)  # メチレンジオキシ基数
        features[4] = 1.0 if len(methylenedioxy_info) > 0 else 0.0  # メチレンジオキシ基有無
        features[5] = sum(info['count'] for info in methylenedioxy_info)  # メチレンジオキシ基原子数
        
        # ヒドロキシ基 (3次元)
        hydroxy_info = self.get_substitution_info(mol, 'phenethylamine')['hydroxy']
        features[6] = len(hydroxy_info)  # ヒドロキシ基数
        features[7] = 1.0 if len(hydroxy_info) > 0 else 0.0  # ヒドロキシ基有無
        features[8] = sum(info['count'] for info in hydroxy_info)  # ヒドロキシ基原子数
        
        # アミノ基 (3次元)
        amino_info = self.get_substitution_info(mol, 'phenethylamine')['amino']
        features[9] = len(amino_info)  # アミノ基数
        features[10] = 1.0 if len(amino_info) > 0 else 0.0  # アミノ基有無
        features[11] = sum(info['count'] for info in amino_info)  # アミノ基原子数
        
        return features
    
    def get_scaffold_summary(self, mol: Chem.Mol) -> Dict[str, Union[int, Dict]]:
        """
        骨格検出のサマリーを取得
        
        Args:
            mol: RDKit分子オブジェクト
            
        Returns:
            骨格検出サマリー
        """
        if mol is None:
            return {
                'total_scaffolds': 0,
                'scaffold_counts': {scaffold_type: 0 for scaffold_type in SCAFFOLD_TYPES},
                'substitution_info': {scaffold_type: {} for scaffold_type in SCAFFOLD_TYPES}
            }
        
        scaffold_counts = self.detect_scaffolds(mol)
        total_scaffolds = sum(scaffold_counts.values())
        
        substitution_info = {}
        for scaffold_type in SCAFFOLD_TYPES:
            if scaffold_counts[scaffold_type] > 0:
                substitution_info[scaffold_type] = self.get_substitution_info(mol, scaffold_type)
            else:
                substitution_info[scaffold_type] = {}
        
        return {
            'total_scaffolds': total_scaffolds,
            'scaffold_counts': scaffold_counts,
            'substitution_info': substitution_info
        }
    
    def visualize_scaffolds(self, mol: Chem.Mol, output_path: Optional[str] = None) -> str:
        """
        骨格をハイライトして可視化
        
        Args:
            mol: RDKit分子オブジェクト
            output_path: 出力ファイルパス
            
        Returns:
            可視化結果のパス
        """
        if mol is None:
            return ""
        
        # 骨格検出
        scaffold_counts = self.detect_scaffolds(mol)
        detected_scaffolds = [scaffold for scaffold, count in scaffold_counts.items() if count > 0]
        
        if not detected_scaffolds:
            if self.verbose:
                print("No scaffolds detected for visualization")
            return ""
        
        # ハイライト色を設定
        highlight_colors = {
            'phenethylamine': (1.0, 0.0, 0.0),  # 赤
            'tryptamine': (0.0, 1.0, 0.0),      # 緑
            'opioid': (0.0, 0.0, 1.0),           # 青
            'cannabinoid': (1.0, 1.0, 0.0)       # 黄
        }
        
        # ハイライト原子を収集
        highlight_atoms = []
        highlight_bonds = []
        colors = []
        
        for scaffold_type in detected_scaffolds:
            if scaffold_type in self._compiled_patterns:
                # コアパターンのマッチを取得
                if 'core' in self._compiled_patterns[scaffold_type]:
                    matches = mol.GetSubstructMatches(self._compiled_patterns[scaffold_type]['core'])
                    for match in matches:
                        highlight_atoms.extend(match)
                        colors.extend([highlight_colors[scaffold_type]] * len(match))
        
        # 分子描画
        rdDepictor.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)
        drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms, highlightBonds=highlight_bonds)
        drawer.FinishDrawing()
        
        if output_path is None:
            output_path = "scaffold_visualization.png"
        
        with open(output_path, 'wb') as f:
            f.write(drawer.GetDrawingText())
        
        if self.verbose:
            print(f"Scaffold visualization saved to: {output_path}")
        
        return output_path
    
    def batch_detect(self, mols: List[Chem.Mol]) -> List[Dict[str, int]]:
        """
        複数分子の骨格検出をバッチ処理
        
        Args:
            mols: RDKit分子オブジェクトのリスト
            
        Returns:
            各分子の骨格検出結果
        """
        results = []
        for mol in mols:
            results.append(self.detect_scaffolds(mol))
        return results
    
    def batch_extract_features(self, mols: List[Chem.Mol]) -> np.ndarray:
        """
        複数分子の骨格特徴量をバッチ抽出
        
        Args:
            mols: RDKit分子オブジェクトのリスト
            
        Returns:
            骨格特徴量行列 (N x 20)
        """
        features_list = []
        for mol in mols:
            features = self.extract_scaffold_features(mol)
            features_list.append(features)
        return np.array(features_list)


# 便利関数
def detect_scaffolds_from_smiles(smiles: str, verbose: bool = False) -> Dict[str, int]:
    """SMILES文字列から骨格を検出"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        if verbose:
            print(f"Invalid SMILES: {smiles}")
        return {scaffold_type: 0 for scaffold_type in SCAFFOLD_TYPES}
    
    detector = ScaffoldDetector(verbose=verbose)
    return detector.detect_scaffolds(mol)


def extract_scaffold_features_from_smiles(smiles: str, verbose: bool = False) -> np.ndarray:
    """SMILES文字列から骨格特徴量を抽出"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        if verbose:
            print(f"Invalid SMILES: {smiles}")
        return np.zeros(20, dtype=np.float32)
    
    detector = ScaffoldDetector(verbose=verbose)
    return detector.extract_scaffold_features(mol)


def get_scaffold_summary_from_smiles(smiles: str, verbose: bool = False) -> Dict[str, Union[int, Dict]]:
    """SMILES文字列から骨格サマリーを取得"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        if verbose:
            print(f"Invalid SMILES: {smiles}")
        return {
            'total_scaffolds': 0,
            'scaffold_counts': {scaffold_type: 0 for scaffold_type in SCAFFOLD_TYPES},
            'substitution_info': {scaffold_type: {} for scaffold_type in SCAFFOLD_TYPES}
        }
    
    detector = ScaffoldDetector(verbose=verbose)
    return detector.get_scaffold_summary(mol)


if __name__ == "__main__":
    # テスト実行
    print("🧬 骨格検出器テスト")
    print("=" * 50)
    
    # テスト用SMILES
    test_smiles = [
        "CC(CC1=CC=CC=C1)N",  # アンフェタミン (フェネチルアミン)
        "CCN(CC)CC1=CC2=C(C=C1)OCO2",  # MDMA (フェネチルアミン)
        "CCN(CC)CC1=CNC2=CC=CC=C21",  # DMT (トリプタミン)
        "CCN(CC)CC1=CNC2=CC(=CC=C21)OC",  # 5-MeO-DMT (トリプタミン)
        "CN1CC[C@]23C4=C5C6=C(C=CC=C6)O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5=C4O",  # モルヒネ (オピオイド)
        "CCCCCC1=CC(=C(C(=C1)O)C2C=C(CCC2C(C)C)C)C",  # THC (カンナビノイド)
    ]
    
    detector = ScaffoldDetector(verbose=True)
    
    for i, smiles in enumerate(test_smiles):
        print(f"\n📋 テスト {i+1}: {smiles}")
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("  ❌ Invalid SMILES")
            continue
        
        # 骨格検出
        scaffold_counts = detector.detect_scaffolds(mol)
        print(f"  骨格検出結果: {scaffold_counts}")
        
        # 特徴量抽出
        features = detector.extract_scaffold_features(mol)
        print(f"  特徴量次元: {features.shape}")
        print(f"  特徴量値: {features[:8]}")  # 最初の8次元のみ表示
        
        # サマリー取得
        summary = detector.get_scaffold_summary(mol)
        print(f"  総骨格数: {summary['total_scaffolds']}")
    
    print("\n✅ 骨格検出器テスト完了！")
