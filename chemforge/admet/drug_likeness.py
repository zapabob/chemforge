"""
Drug Likeness Module

薬物らしさ予測システム
包括的な薬物らしさ評価・最適化指標
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED
from rdkit.Chem import rdMolDescriptors
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class DrugLikenessPredictor:
    """
    薬物らしさ予測器
    
    包括的な薬物らしさ評価・最適化指標
    """
    
    def __init__(self):
        """薬物らしさ予測器を初期化"""
        logger.info("DrugLikenessPredictor initialized")
    
    def predict_lipinski_rule(self, smiles: str) -> Dict[str, Union[float, int, bool]]:
        """
        Lipinski's Rule of Fiveを予測
        
        Args:
            smiles: SMILES文字列
        
        Returns:
            Lipinski指標辞書
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return {}
            
            # Lipinski's Rule of Five
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            
            # 違反数計算
            violations = 0
            if mw > 500:
                violations += 1
            if logp > 5:
                violations += 1
            if hbd > 5:
                violations += 1
            if hba > 10:
                violations += 1
            
            lipinski = {
                'molecular_weight': mw,
                'logp': logp,
                'hbd': hbd,
                'hba': hba,
                'violations': violations,
                'compliant': violations <= 1
            }
            
            return lipinski
            
        except Exception as e:
            logger.error(f"Error predicting Lipinski rule for {smiles}: {str(e)}")
            return {}
    
    def predict_veber_rule(self, smiles: str) -> Dict[str, Union[float, int, bool]]:
        """
        Veber's Ruleを予測
        
        Args:
            smiles: SMILES文字列
        
        Returns:
            Veber指標辞書
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return {}
            
            # Veber's Rule
            tpsa = Descriptors.TPSA(mol)
            rot_bonds = Descriptors.NumRotatableBonds(mol)
            
            # 適合性判定
            veber_compliant = tpsa <= 140 and rot_bonds <= 10
            
            veber = {
                'tpsa': tpsa,
                'rotatable_bonds': rot_bonds,
                'compliant': veber_compliant
            }
            
            return veber
            
        except Exception as e:
            logger.error(f"Error predicting Veber rule for {smiles}: {str(e)}")
            return {}
    
    def predict_qed(self, smiles: str) -> Dict[str, float]:
        """
        QED (Quantitative Estimate of Drug-likeness)を予測
        
        Args:
            smiles: SMILES文字列
        
        Returns:
            QED指標辞書
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return {}
            
            # QED計算
            qed = QED.qed(mol)
            
            # QED解釈
            if qed >= 0.8:
                qed_interpretation = "Excellent"
            elif qed >= 0.6:
                qed_interpretation = "Good"
            elif qed >= 0.4:
                qed_interpretation = "Moderate"
            else:
                qed_interpretation = "Poor"
            
            qed_results = {
                'qed': qed,
                'qed_interpretation': qed_interpretation,
                'drug_like': qed >= 0.5
            }
            
            return qed_results
            
        except Exception as e:
            logger.error(f"Error predicting QED for {smiles}: {str(e)}")
            return {}
    
    def predict_sa_score(self, smiles: str) -> Dict[str, float]:
        """
        SA Score (Synthetic Accessibility)を予測
        
        Args:
            smiles: SMILES文字列
        
        Returns:
            SA Score指標辞書
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return {}
            
            # SA Score計算
            sa_score = rdMolDescriptors.CalcSAscore(mol)
            
            # SA Score解釈
            if sa_score <= 3.0:
                sa_interpretation = "Easy"
            elif sa_score <= 6.0:
                sa_interpretation = "Moderate"
            else:
                sa_interpretation = "Difficult"
            
            sa_results = {
                'sa_score': sa_score,
                'sa_interpretation': sa_interpretation,
                'synthesizable': sa_score <= 6.0
            }
            
            return sa_results
            
        except Exception as e:
            logger.error(f"Error predicting SA score for {smiles}: {str(e)}")
            return {}
    
    def predict_lead_likeness(self, smiles: str) -> Dict[str, Union[float, bool]]:
        """
        Lead-likenessを予測
        
        Args:
            smiles: SMILES文字列
        
        Returns:
            Lead-likeness指標辞書
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return {}
            
            # Lead-likeness指標
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            rot_bonds = Descriptors.NumRotatableBonds(mol)
            
            # Lead-likeness判定
            lead_like = (
                150 <= mw <= 350 and
                1.0 <= logp <= 3.5 and
                20 <= tpsa <= 130 and
                rot_bonds <= 7
            )
            
            lead_likeness = {
                'molecular_weight': mw,
                'logp': logp,
                'tpsa': tpsa,
                'rotatable_bonds': rot_bonds,
                'lead_like': lead_like
            }
            
            return lead_likeness
            
        except Exception as e:
            logger.error(f"Error predicting lead-likeness for {smiles}: {str(e)}")
            return {}
    
    def predict_fragment_likeness(self, smiles: str) -> Dict[str, Union[float, bool]]:
        """
        Fragment-likenessを予測
        
        Args:
            smiles: SMILES文字列
        
        Returns:
            Fragment-likeness指標辞書
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return {}
            
            # Fragment-likeness指標
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            rot_bonds = Descriptors.NumRotatableBonds(mol)
            
            # Fragment-likeness判定
            fragment_like = (
                mw <= 300 and
                logp <= 3.0 and
                tpsa <= 60 and
                rot_bonds <= 3
            )
            
            fragment_likeness = {
                'molecular_weight': mw,
                'logp': logp,
                'tpsa': tpsa,
                'rotatable_bonds': rot_bonds,
                'fragment_like': fragment_like
            }
            
            return fragment_likeness
            
        except Exception as e:
            logger.error(f"Error predicting fragment-likeness for {smiles}: {str(e)}")
            return {}
    
    def predict_drug_likeness_score(self, smiles: str) -> Dict[str, float]:
        """
        包括的薬物らしさスコアを予測
        
        Args:
            smiles: SMILES文字列
        
        Returns:
            包括的薬物らしさスコア辞書
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return {}
            
            # 各種指標を取得
            lipinski = self.predict_lipinski_rule(smiles)
            veber = self.predict_veber_rule(smiles)
            qed = self.predict_qed(smiles)
            sa_score = self.predict_sa_score(smiles)
            lead_likeness = self.predict_lead_likeness(smiles)
            fragment_likeness = self.predict_fragment_likeness(smiles)
            
            # 包括的スコア計算
            drug_likeness_score = 0.0
            total_weight = 0.0
            
            # Lipinski適合性
            if lipinski.get('compliant', False):
                drug_likeness_score += 0.3
            total_weight += 0.3
            
            # Veber適合性
            if veber.get('compliant', False):
                drug_likeness_score += 0.2
            total_weight += 0.2
            
            # QED
            qed_value = qed.get('qed', 0.0)
            drug_likeness_score += qed_value * 0.3
            total_weight += 0.3
            
            # SA Score
            sa_value = sa_score.get('sa_score', 10.0)
            sa_normalized = max(0.0, 1.0 - (sa_value - 1.0) / 9.0)
            drug_likeness_score += sa_normalized * 0.2
            total_weight += 0.2
            
            # 正規化
            if total_weight > 0:
                drug_likeness_score /= total_weight
            
            # 解釈
            if drug_likeness_score >= 0.8:
                interpretation = "Excellent"
            elif drug_likeness_score >= 0.6:
                interpretation = "Good"
            elif drug_likeness_score >= 0.4:
                interpretation = "Moderate"
            else:
                interpretation = "Poor"
            
            comprehensive_score = {
                'drug_likeness_score': drug_likeness_score,
                'interpretation': interpretation,
                'drug_like': drug_likeness_score >= 0.5,
                'lipinski_compliant': lipinski.get('compliant', False),
                'veber_compliant': veber.get('compliant', False),
                'qed': qed_value,
                'sa_score': sa_value,
                'lead_like': lead_likeness.get('lead_like', False),
                'fragment_like': fragment_likeness.get('fragment_like', False)
            }
            
            return comprehensive_score
            
        except Exception as e:
            logger.error(f"Error predicting drug likeness score for {smiles}: {str(e)}")
            return {}
    
    def predict_comprehensive_drug_likeness(self, smiles: str) -> Dict[str, Union[float, int, bool, str]]:
        """
        包括的薬物らしさ予測
        
        Args:
            smiles: SMILES文字列
        
        Returns:
            包括的薬物らしさ予測値
        """
        logger.info(f"Predicting comprehensive drug likeness for {smiles}")
        
        drug_likeness_results = {}
        
        # 各種指標を予測
        drug_likeness_results.update(self.predict_lipinski_rule(smiles))
        drug_likeness_results.update(self.predict_veber_rule(smiles))
        drug_likeness_results.update(self.predict_qed(smiles))
        drug_likeness_results.update(self.predict_sa_score(smiles))
        drug_likeness_results.update(self.predict_lead_likeness(smiles))
        drug_likeness_results.update(self.predict_fragment_likeness(smiles))
        drug_likeness_results.update(self.predict_drug_likeness_score(smiles))
        
        return drug_likeness_results
    
    def process_molecule_batch(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        分子バッチを処理
        
        Args:
            smiles_list: SMILES文字列リスト
        
        Returns:
            薬物らしさ予測結果データフレーム
        """
        logger.info(f"Processing {len(smiles_list)} molecules for drug likeness prediction")
        
        results = []
        
        for i, smiles in enumerate(smiles_list):
            if i % 100 == 0:
                logger.info(f"Processing molecule {i}/{len(smiles_list)}")
            
            drug_likeness_results = self.predict_comprehensive_drug_likeness(smiles)
            drug_likeness_results['smiles'] = smiles
            results.append(drug_likeness_results)
        
        # データフレームに変換
        df = pd.DataFrame(results)
        
        # 数値列の欠損値を処理
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        logger.info(f"Processed {len(df)} molecules successfully")
        return df
    
    def get_drug_likeness_summary(self, df: pd.DataFrame) -> Dict:
        """
        薬物らしさ要約を取得
        
        Args:
            df: 薬物らしさデータフレーム
        
        Returns:
            薬物らしさ要約辞書
        """
        summary = {
            "total_molecules": len(df),
            "drug_like_molecules": len(df[df.get('drug_like', False)]),
            "lipinski_compliant": len(df[df.get('compliant', False)]),
            "veber_compliant": len(df[df.get('compliant', False)]),
            "lead_like_molecules": len(df[df.get('lead_like', False)]),
            "fragment_like_molecules": len(df[df.get('fragment_like', False)]),
            "drug_likeness_statistics": {}
        }
        
        # 薬物らしさ統計
        if 'drug_likeness_score' in df.columns:
            summary["drug_likeness_statistics"] = {
                "mean_score": df['drug_likeness_score'].mean(),
                "std_score": df['drug_likeness_score'].std(),
                "min_score": df['drug_likeness_score'].min(),
                "max_score": df['drug_likeness_score'].max(),
                "median_score": df['drug_likeness_score'].median()
            }
        
        return summary
