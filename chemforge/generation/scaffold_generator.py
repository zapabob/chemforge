"""
骨格ベース分子生成モジュール

CNS作動薬骨格を活用した分子生成・最適化
"""

import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import json
from dataclasses import dataclass

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Lipinski, QED
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem import rdMolDescriptors as rdMD
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from chemforge.data.cns_scaffolds import CNSScaffolds, CNSCompound
from chemforge.utils.logging_utils import Logger

logger = logging.getLogger(__name__)

@dataclass
class GeneratedMolecule:
    """生成された分子情報"""
    smiles: str
    scaffold_type: str
    parent_compound: str
    modifications: List[str]
    properties: Dict[str, float]
    score: float

class ScaffoldGenerator:
    """
    骨格ベース分子生成器
    
    CNS作動薬骨格を活用した分子生成・最適化
    """
    
    def __init__(self, scaffolds: Optional[CNSScaffolds] = None):
        """
        初期化
        
        Args:
            scaffolds: CNS骨格ライブラリ
        """
        self.scaffolds = scaffolds or CNSScaffolds()
        self.generated_molecules = []
        logger.info("ScaffoldGenerator initialized")
    
    def generate_analogs(self, 
                        scaffold_type: str, 
                        num_analogs: int = 10,
                        modification_types: List[str] = None) -> List[GeneratedMolecule]:
        """
        骨格ベースの類似体生成
        
        Args:
            scaffold_type: 骨格タイプ
            num_analogs: 生成する類似体数
            modification_types: 修飾タイプ
            
        Returns:
            生成された分子リスト
        """
        if modification_types is None:
            modification_types = ['substitution', 'addition', 'removal']
        
        compounds = self.scaffolds.get_scaffold_compounds(scaffold_type)
        if not compounds:
            logger.warning(f"No compounds found for scaffold type: {scaffold_type}")
            return []
        
        generated = []
        for _ in range(num_analogs):
            # ランダムに親化合物を選択
            parent = random.choice(compounds)
            
            # 修飾タイプをランダムに選択
            mod_type = random.choice(modification_types)
            
            # 分子生成
            analog = self._generate_single_analog(parent, mod_type)
            if analog:
                generated.append(analog)
        
        self.generated_molecules.extend(generated)
        return generated
    
    def _generate_single_analog(self, parent: CNSCompound, mod_type: str) -> Optional[GeneratedMolecule]:
        """
        単一類似体生成
        
        Args:
            parent: 親化合物
            mod_type: 修飾タイプ
            
        Returns:
            生成された分子
        """
        try:
            mol = Chem.MolFromSmiles(parent.smiles)
            if mol is None:
                return None
            
            modifications = []
            
            if mod_type == 'substitution':
                modified_mol = self._substitute_atoms(mol)
                modifications.append("原子置換")
            elif mod_type == 'addition':
                modified_mol = self._add_functional_groups(mol)
                modifications.append("官能基追加")
            elif mod_type == 'removal':
                modified_mol = self._remove_functional_groups(mol)
                modifications.append("官能基除去")
            else:
                modified_mol = mol
            
            if modified_mol is None:
                return None
            
            # SMILESに変換
            smiles = Chem.MolToSmiles(modified_mol)
            if not smiles or smiles == parent.smiles:
                return None
            
            # 物性計算
            properties = self._calculate_properties(modified_mol)
            
            # スコア計算
            score = self._calculate_score(modified_mol, parent)
            
            return GeneratedMolecule(
                smiles=smiles,
                scaffold_type=parent.name,
                parent_compound=parent.name,
                modifications=modifications,
                properties=properties,
                score=score
            )
            
        except Exception as e:
            logger.error(f"Error generating analog: {e}")
            return None
    
    def _substitute_atoms(self, mol) -> Optional[Chem.Mol]:
        """原子置換"""
        try:
            # 簡単な原子置換（H → F, Cl, Br）
            substitutions = [
                ('H', 'F'),
                ('H', 'Cl'),
                ('H', 'Br'),
                ('C', 'N'),
                ('N', 'C')
            ]
            
            for old_atom, new_atom in substitutions:
                # ランダムに1つの原子を置換
                atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == old_atom]
                if atoms:
                    atom_to_replace = random.choice(atoms)
                    # 実際の置換は複雑なので、簡易的にSMILES文字列操作
                    smiles = Chem.MolToSmiles(mol)
                    if old_atom == 'H' and new_atom in ['F', 'Cl', 'Br']:
                        # HをF, Cl, Brに置換（簡易的）
                        modified_smiles = smiles.replace('H', new_atom, 1)
                        return Chem.MolFromSmiles(modified_smiles)
            
            return mol
        except Exception as e:
            logger.error(f"Error in atom substitution: {e}")
            return None
    
    def _add_functional_groups(self, mol) -> Optional[Chem.Mol]:
        """官能基追加"""
        try:
            # 簡単な官能基追加
            functional_groups = ['OH', 'NH2', 'COOH', 'CH3', 'F', 'Cl']
            
            # ランダムに官能基を選択
            fg = random.choice(functional_groups)
            
            # SMILES文字列に官能基を追加（簡易的）
            smiles = Chem.MolToSmiles(mol)
            modified_smiles = f"{smiles}{fg}"
            
            return Chem.MolFromSmiles(modified_smiles)
        except Exception as e:
            logger.error(f"Error adding functional groups: {e}")
            return None
    
    def _remove_functional_groups(self, mol) -> Optional[Chem.Mol]:
        """官能基除去"""
        try:
            # 簡単な官能基除去
            smiles = Chem.MolToSmiles(mol)
            
            # 一般的な官能基を除去
            functional_groups = ['OH', 'NH2', 'COOH', 'CH3']
            for fg in functional_groups:
                if fg in smiles:
                    modified_smiles = smiles.replace(fg, '', 1)
                    modified_mol = Chem.MolFromSmiles(modified_smiles)
                    if modified_mol is not None:
                        return modified_mol
            
            return mol
        except Exception as e:
            logger.error(f"Error removing functional groups: {e}")
            return None
    
    def _calculate_properties(self, mol) -> Dict[str, float]:
        """物性計算"""
        try:
            return {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Crippen.MolLogP(mol),
                'tpsa': Descriptors.TPSA(mol),
                'num_atoms': mol.GetNumAtoms(),
                'num_bonds': mol.GetNumBonds(),
                'num_rings': Descriptors.RingCount(mol),
                'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
                'num_heteroatoms': Descriptors.NumHeteroatoms(mol),
                'fraction_csp3': Descriptors.FractionCSP3(mol),
                'heavy_atom_count': Descriptors.HeavyAtomCount(mol),
                'nhoh_count': Descriptors.NHOHCount(mol),
                'no_count': Descriptors.NOCount(mol)
            }
        except Exception as e:
            logger.error(f"Error calculating properties: {e}")
            return {}
    
    def _calculate_score(self, mol, parent: CNSCompound) -> float:
        """スコア計算"""
        try:
            # Lipinski's Rule of Five
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            
            # Lipinskiスコア
            lipinski_score = 0
            if mw <= 500: lipinski_score += 1
            if logp <= 5: lipinski_score += 1
            if hbd <= 5: lipinski_score += 1
            if hba <= 10: lipinski_score += 1
            
            # QEDスコア
            qed_score = QED.qed(mol)
            
            # 総合スコア
            total_score = (lipinski_score / 4.0) * 0.6 + qed_score * 0.4
            
            return min(1.0, max(0.0, total_score))
        except Exception as e:
            logger.error(f"Error calculating score: {e}")
            return 0.0
    
    def optimize_molecules(self, 
                          target_properties: Dict[str, float],
                          max_iterations: int = 100) -> List[GeneratedMolecule]:
        """
        分子最適化
        
        Args:
            target_properties: 目標物性
            max_iterations: 最大反復回数
            
        Returns:
            最適化された分子リスト
        """
        optimized = []
        
        for molecule in self.generated_molecules:
            if self._is_optimized(molecule, target_properties):
                optimized.append(molecule)
        
        return optimized
    
    def _is_optimized(self, molecule: GeneratedMolecule, target_properties: Dict[str, float]) -> bool:
        """最適化判定"""
        try:
            for prop, target_value in target_properties.items():
                if prop in molecule.properties:
                    current_value = molecule.properties[prop]
                    # 許容範囲内かチェック
                    tolerance = target_value * 0.1  # 10%の許容範囲
                    if abs(current_value - target_value) > tolerance:
                        return False
            return True
        except Exception as e:
            logger.error(f"Error in optimization check: {e}")
            return False
    
    def get_best_molecules(self, top_k: int = 10) -> List[GeneratedMolecule]:
        """最高スコアの分子を取得"""
        sorted_molecules = sorted(self.generated_molecules, 
                                key=lambda x: x.score, reverse=True)
        return sorted_molecules[:top_k]
    
    def export_results(self, output_path: str) -> None:
        """結果をエクスポート"""
        try:
            export_data = []
            for molecule in self.generated_molecules:
                export_data.append({
                    'smiles': molecule.smiles,
                    'scaffold_type': molecule.scaffold_type,
                    'parent_compound': molecule.parent_compound,
                    'modifications': molecule.modifications,
                    'properties': molecule.properties,
                    'score': molecule.score
                })
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results exported to {output_path}")
        except Exception as e:
            logger.error(f"Error exporting results: {e}")

# 便利な関数
def generate_scaffold_analogs(scaffold_type: str, 
                            num_analogs: int = 10) -> List[GeneratedMolecule]:
    """
    骨格類似体生成の便利関数
    
    Args:
        scaffold_type: 骨格タイプ
        num_analogs: 生成する類似体数
        
    Returns:
        生成された分子リスト
    """
    generator = ScaffoldGenerator()
    return generator.generate_analogs(scaffold_type, num_analogs)

def optimize_for_cns_targets(molecules: List[GeneratedMolecule]) -> List[GeneratedMolecule]:
    """
    CNSターゲット向け最適化
    
    Args:
        molecules: 分子リスト
        
    Returns:
        最適化された分子リスト
    """
    # CNS向け目標物性
    target_properties = {
        'molecular_weight': 300.0,
        'logp': 2.5,
        'tpsa': 60.0,
        'num_aromatic_rings': 2.0
    }
    
    generator = ScaffoldGenerator()
    generator.generated_molecules = molecules
    return generator.optimize_molecules(target_properties)
