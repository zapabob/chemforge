"""
CNS-MPO Calculator Module

CNS-MPO (Central Nervous System Multiparameter Optimization) 計算
CNS創薬最適化指標の包括的評価
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from rdkit.Chem import rdMolDescriptors
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class CNSMPOCalculator:
    """
    CNS-MPO計算器
    
    CNS-MPO (Central Nervous System Multiparameter Optimization) 計算
    CNS創薬最適化指標の包括的評価
    """
    
    def __init__(self):
        """CNS-MPO計算器を初期化"""
        logger.info("CNSMPOCalculator initialized")
    
    def calculate_cns_mpo(self, smiles: str) -> Dict[str, Union[float, int, bool]]:
        """
        CNS-MPOを計算
        
        Args:
            smiles: SMILES文字列
        
        Returns:
            CNS-MPO指標辞書
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return {}
            
            # 基本物性
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            rot_bonds = Descriptors.NumRotatableBonds(mol)
            
            # CNS-MPO指標計算
            cns_mpo = {
                'molecular_weight': mw,
                'logp': logp,
                'tpsa': tpsa,
                'hbd': hbd,
                'hba': hba,
                'rotatable_bonds': rot_bonds,
                'cns_mpo_score': self._calculate_cns_mpo_score(mw, logp, tpsa, hbd, hba, rot_bonds),
                'cns_mpo_interpretation': self._interpret_cns_mpo_score(self._calculate_cns_mpo_score(mw, logp, tpsa, hbd, hba, rot_bonds))
            }
            
            return cns_mpo
            
        except Exception as e:
            logger.error(f"Error calculating CNS-MPO for {smiles}: {str(e)}")
            return {}
    
    def _calculate_cns_mpo_score(self, mw: float, logp: float, tpsa: float, 
                                hbd: int, hba: int, rot_bonds: int) -> float:
        """
        CNS-MPOスコアを計算
        
        Args:
            mw: 分子量
            logp: LogP
            tpsa: TPSA
            hbd: 水素結合ドナー数
            hba: 水素結合アクセプター数
            rot_bonds: 回転可能結合数
        
        Returns:
            CNS-MPOスコア
        """
        score = 0.0
        
        # 分子量 (150-350)
        if 150 <= mw <= 350:
            score += 1.0
        elif 100 <= mw < 150 or 350 < mw <= 400:
            score += 0.5
        
        # LogP (2-4)
        if 2.0 <= logp <= 4.0:
            score += 1.0
        elif 1.5 <= logp < 2.0 or 4.0 < logp <= 4.5:
            score += 0.5
        
        # TPSA (20-60)
        if 20 <= tpsa <= 60:
            score += 1.0
        elif 10 <= tpsa < 20 or 60 < tpsa <= 80:
            score += 0.5
        
        # 水素結合ドナー数 (0-2)
        if 0 <= hbd <= 2:
            score += 1.0
        elif hbd == 3:
            score += 0.5
        
        # 水素結合アクセプター数 (0-6)
        if 0 <= hba <= 6:
            score += 1.0
        elif 7 <= hba <= 8:
            score += 0.5
        
        # 回転可能結合数 (0-6)
        if 0 <= rot_bonds <= 6:
            score += 1.0
        elif 7 <= rot_bonds <= 8:
            score += 0.5
        
        return score
    
    def _interpret_cns_mpo_score(self, score: float) -> str:
        """
        CNS-MPOスコアを解釈
        
        Args:
            score: CNS-MPOスコア
        
        Returns:
            解釈文字列
        """
        if score >= 5.0:
            return "Excellent"
        elif score >= 4.0:
            return "Good"
        elif score >= 3.0:
            return "Moderate"
        elif score >= 2.0:
            return "Poor"
        else:
            return "Very Poor"
    
    def calculate_cns_mpo_plus(self, smiles: str) -> Dict[str, Union[float, int, bool]]:
        """
        CNS-MPO+を計算（拡張版）
        
        Args:
            smiles: SMILES文字列
        
        Returns:
            CNS-MPO+指標辞書
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return {}
            
            # 基本物性
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            rot_bonds = Descriptors.NumRotatableBonds(mol)
            
            # 追加指標
            num_rings = Descriptors.RingCount(mol)
            num_aromatic_rings = Descriptors.NumAromaticRings(mol)
            num_heteroatoms = Descriptors.NumHeteroatoms(mol)
            fraction_csp3 = Descriptors.FractionCsp3(mol)
            
            # CNS-MPO+スコア計算
            cns_mpo_plus = {
                'molecular_weight': mw,
                'logp': logp,
                'tpsa': tpsa,
                'hbd': hbd,
                'hba': hba,
                'rotatable_bonds': rot_bonds,
                'num_rings': num_rings,
                'num_aromatic_rings': num_aromatic_rings,
                'num_heteroatoms': num_heteroatoms,
                'fraction_csp3': fraction_csp3,
                'cns_mpo_plus_score': self._calculate_cns_mpo_plus_score(
                    mw, logp, tpsa, hbd, hba, rot_bonds, 
                    num_rings, num_aromatic_rings, num_heteroatoms, fraction_csp3
                ),
                'cns_mpo_plus_interpretation': self._interpret_cns_mpo_plus_score(
                    self._calculate_cns_mpo_plus_score(
                        mw, logp, tpsa, hbd, hba, rot_bonds, 
                        num_rings, num_aromatic_rings, num_heteroatoms, fraction_csp3
                    )
                )
            }
            
            return cns_mpo_plus
            
        except Exception as e:
            logger.error(f"Error calculating CNS-MPO+ for {smiles}: {str(e)}")
            return {}
    
    def _calculate_cns_mpo_plus_score(self, mw: float, logp: float, tpsa: float, 
                                     hbd: int, hba: int, rot_bonds: int,
                                     num_rings: int, num_aromatic_rings: int, 
                                     num_heteroatoms: int, fraction_csp3: float) -> float:
        """
        CNS-MPO+スコアを計算
        
        Args:
            mw: 分子量
            logp: LogP
            tpsa: TPSA
            hbd: 水素結合ドナー数
            hba: 水素結合アクセプター数
            rot_bonds: 回転可能結合数
            num_rings: 環数
            num_aromatic_rings: 芳香族環数
            num_heteroatoms: ヘテロ原子数
            fraction_csp3: Csp3割合
        
        Returns:
            CNS-MPO+スコア
        """
        score = 0.0
        
        # 基本CNS-MPO指標
        score += self._calculate_cns_mpo_score(mw, logp, tpsa, hbd, hba, rot_bonds)
        
        # 環数 (1-3)
        if 1 <= num_rings <= 3:
            score += 1.0
        elif num_rings == 4:
            score += 0.5
        
        # 芳香族環数 (1-2)
        if 1 <= num_aromatic_rings <= 2:
            score += 1.0
        elif num_aromatic_rings == 3:
            score += 0.5
        
        # ヘテロ原子数 (1-4)
        if 1 <= num_heteroatoms <= 4:
            score += 1.0
        elif 5 <= num_heteroatoms <= 6:
            score += 0.5
        
        # Csp3割合 (0.2-0.8)
        if 0.2 <= fraction_csp3 <= 0.8:
            score += 1.0
        elif 0.1 <= fraction_csp3 < 0.2 or 0.8 < fraction_csp3 <= 0.9:
            score += 0.5
        
        return score
    
    def _interpret_cns_mpo_plus_score(self, score: float) -> str:
        """
        CNS-MPO+スコアを解釈
        
        Args:
            score: CNS-MPO+スコア
        
        Returns:
            解釈文字列
        """
        if score >= 8.0:
            return "Excellent"
        elif score >= 6.0:
            return "Good"
        elif score >= 4.0:
            return "Moderate"
        elif score >= 2.0:
            return "Poor"
        else:
            return "Very Poor"
    
    def calculate_cns_mpo_optimized(self, smiles: str) -> Dict[str, Union[float, int, bool]]:
        """
        最適化されたCNS-MPOを計算
        
        Args:
            smiles: SMILES文字列
        
        Returns:
            最適化されたCNS-MPO指標辞書
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return {}
            
            # 基本物性
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            rot_bonds = Descriptors.NumRotatableBonds(mol)
            
            # 最適化されたCNS-MPOスコア計算
            cns_mpo_optimized = {
                'molecular_weight': mw,
                'logp': logp,
                'tpsa': tpsa,
                'hbd': hbd,
                'hba': hba,
                'rotatable_bonds': rot_bonds,
                'cns_mpo_optimized_score': self._calculate_cns_mpo_optimized_score(
                    mw, logp, tpsa, hbd, hba, rot_bonds
                ),
                'cns_mpo_optimized_interpretation': self._interpret_cns_mpo_optimized_score(
                    self._calculate_cns_mpo_optimized_score(mw, logp, tpsa, hbd, hba, rot_bonds)
                ),
                'optimization_recommendations': self._get_optimization_recommendations(
                    mw, logp, tpsa, hbd, hba, rot_bonds
                )
            }
            
            return cns_mpo_optimized
            
        except Exception as e:
            logger.error(f"Error calculating optimized CNS-MPO for {smiles}: {str(e)}")
            return {}
    
    def _calculate_cns_mpo_optimized_score(self, mw: float, logp: float, tpsa: float, 
                                         hbd: int, hba: int, rot_bonds: int) -> float:
        """
        最適化されたCNS-MPOスコアを計算
        
        Args:
            mw: 分子量
            logp: LogP
            tpsa: TPSA
            hbd: 水素結合ドナー数
            hba: 水素結合アクセプター数
            rot_bonds: 回転可能結合数
        
        Returns:
            最適化されたCNS-MPOスコア
        """
        score = 0.0
        
        # 分子量 (150-350) - 重み付き
        if 150 <= mw <= 350:
            score += 1.0
        elif 100 <= mw < 150 or 350 < mw <= 400:
            score += 0.5
        
        # LogP (2-4) - 重み付き
        if 2.0 <= logp <= 4.0:
            score += 1.0
        elif 1.5 <= logp < 2.0 or 4.0 < logp <= 4.5:
            score += 0.5
        
        # TPSA (20-60) - 重み付き
        if 20 <= tpsa <= 60:
            score += 1.0
        elif 10 <= tpsa < 20 or 60 < tpsa <= 80:
            score += 0.5
        
        # 水素結合ドナー数 (0-2) - 重み付き
        if 0 <= hbd <= 2:
            score += 1.0
        elif hbd == 3:
            score += 0.5
        
        # 水素結合アクセプター数 (0-6) - 重み付き
        if 0 <= hba <= 6:
            score += 1.0
        elif 7 <= hba <= 8:
            score += 0.5
        
        # 回転可能結合数 (0-6) - 重み付き
        if 0 <= rot_bonds <= 6:
            score += 1.0
        elif 7 <= rot_bonds <= 8:
            score += 0.5
        
        return score
    
    def _interpret_cns_mpo_optimized_score(self, score: float) -> str:
        """
        最適化されたCNS-MPOスコアを解釈
        
        Args:
            score: 最適化されたCNS-MPOスコア
        
        Returns:
            解釈文字列
        """
        if score >= 5.0:
            return "Excellent"
        elif score >= 4.0:
            return "Good"
        elif score >= 3.0:
            return "Moderate"
        elif score >= 2.0:
            return "Poor"
        else:
            return "Very Poor"
    
    def _get_optimization_recommendations(self, mw: float, logp: float, tpsa: float, 
                                         hbd: int, hba: int, rot_bonds: int) -> List[str]:
        """
        最適化推奨事項を取得
        
        Args:
            mw: 分子量
            logp: LogP
            tpsa: TPSA
            hbd: 水素結合ドナー数
            hba: 水素結合アクセプター数
            rot_bonds: 回転可能結合数
        
        Returns:
            最適化推奨事項リスト
        """
        recommendations = []
        
        # 分子量最適化
        if mw < 150:
            recommendations.append("Consider increasing molecular weight to 150-350 range")
        elif mw > 350:
            recommendations.append("Consider reducing molecular weight to 150-350 range")
        
        # LogP最適化
        if logp < 2.0:
            recommendations.append("Consider increasing LogP to 2-4 range")
        elif logp > 4.0:
            recommendations.append("Consider reducing LogP to 2-4 range")
        
        # TPSA最適化
        if tpsa < 20:
            recommendations.append("Consider increasing TPSA to 20-60 range")
        elif tpsa > 60:
            recommendations.append("Consider reducing TPSA to 20-60 range")
        
        # 水素結合ドナー最適化
        if hbd > 2:
            recommendations.append("Consider reducing hydrogen bond donors to 0-2 range")
        
        # 水素結合アクセプター最適化
        if hba > 6:
            recommendations.append("Consider reducing hydrogen bond acceptors to 0-6 range")
        
        # 回転可能結合最適化
        if rot_bonds > 6:
            recommendations.append("Consider reducing rotatable bonds to 0-6 range")
        
        return recommendations
    
    def process_molecule_batch(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        分子バッチを処理
        
        Args:
            smiles_list: SMILES文字列リスト
        
        Returns:
            CNS-MPO予測結果データフレーム
        """
        logger.info(f"Processing {len(smiles_list)} molecules for CNS-MPO calculation")
        
        results = []
        
        for i, smiles in enumerate(smiles_list):
            if i % 100 == 0:
                logger.info(f"Processing molecule {i}/{len(smiles_list)}")
            
            cns_mpo_results = self.calculate_cns_mpo_optimized(smiles)
            cns_mpo_results['smiles'] = smiles
            results.append(cns_mpo_results)
        
        # データフレームに変換
        df = pd.DataFrame(results)
        
        # 数値列の欠損値を処理
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        logger.info(f"Processed {len(df)} molecules successfully")
        return df
    
    def get_cns_mpo_summary(self, df: pd.DataFrame) -> Dict:
        """
        CNS-MPO要約を取得
        
        Args:
            df: CNS-MPOデータフレーム
        
        Returns:
            CNS-MPO要約辞書
        """
        summary = {
            "total_molecules": len(df),
            "cns_mpo_statistics": {},
            "interpretation_distribution": {},
            "optimization_recommendations": {}
        }
        
        # CNS-MPO統計
        if 'cns_mpo_optimized_score' in df.columns:
            summary["cns_mpo_statistics"] = {
                "mean_score": df['cns_mpo_optimized_score'].mean(),
                "std_score": df['cns_mpo_optimized_score'].std(),
                "min_score": df['cns_mpo_optimized_score'].min(),
                "max_score": df['cns_mpo_optimized_score'].max(),
                "median_score": df['cns_mpo_optimized_score'].median()
            }
        
        # 解釈分布
        if 'cns_mpo_optimized_interpretation' in df.columns:
            interpretation_counts = df['cns_mpo_optimized_interpretation'].value_counts()
            summary["interpretation_distribution"] = interpretation_counts.to_dict()
        
        return summary
