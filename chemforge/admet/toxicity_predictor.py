"""
Toxicity Predictor Module

毒性予測システム
包括的な毒性評価・リスクアセスメント
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


class ToxicityPredictor:
    """
    毒性予測器
    
    包括的な毒性評価・リスクアセスメント
    """
    
    def __init__(self):
        """毒性予測器を初期化"""
        logger.info("ToxicityPredictor initialized")
    
    def predict_ames_toxicity(self, smiles: str) -> Dict[str, float]:
        """
        Ames試験毒性を予測
        
        Args:
            smiles: SMILES文字列
        
        Returns:
            Ames試験毒性予測値
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return {}
            
            # 簡易的なAmes試験予測（実際の実装ではより複雑なモデルを使用）
            # 構造アラートベースの予測
            ames_score = self._calculate_structural_alerts(mol)
            
            ames = {
                'ames_score': ames_score,
                'ames_positive': ames_score > 0.5,
                'structural_alerts': self._get_structural_alerts(mol)
            }
            
            return ames
            
        except Exception as e:
            logger.error(f"Error predicting Ames toxicity for {smiles}: {str(e)}")
            return {}
    
    def predict_herg_toxicity(self, smiles: str) -> Dict[str, float]:
        """
        hERG阻害毒性を予測
        
        Args:
            smiles: SMILES文字列
        
        Returns:
            hERG阻害毒性予測値
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return {}
            
            # hERG阻害予測
            herg_score = self._predict_herg_inhibition(mol)
            
            herg = {
                'herg_score': herg_score,
                'herg_positive': herg_score > 0.5,
                'herg_risk': self._assess_herg_risk(herg_score)
            }
            
            return herg
            
        except Exception as e:
            logger.error(f"Error predicting hERG toxicity for {smiles}: {str(e)}")
            return {}
    
    def predict_hepatotoxicity(self, smiles: str) -> Dict[str, float]:
        """
        肝毒性を予測
        
        Args:
            smiles: SMILES文字列
        
        Returns:
            肝毒性予測値
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return {}
            
            # 肝毒性予測
            hepatotoxicity_score = self._predict_hepatotoxicity(mol)
            
            hepatotoxicity = {
                'hepatotoxicity_score': hepatotoxicity_score,
                'hepatotoxicity_positive': hepatotoxicity_score > 0.5,
                'liver_risk': self._assess_liver_risk(hepatotoxicity_score)
            }
            
            return hepatotoxicity
            
        except Exception as e:
            logger.error(f"Error predicting hepatotoxicity for {smiles}: {str(e)}")
            return {}
    
    def predict_cardiovascular_toxicity(self, smiles: str) -> Dict[str, float]:
        """
        心血管毒性を予測
        
        Args:
            smiles: SMILES文字列
        
        Returns:
            心血管毒性予測値
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return {}
            
            # 心血管毒性予測
            cardio_score = self._predict_cardiovascular_toxicity(mol)
            
            cardiovascular = {
                'cardiovascular_score': cardio_score,
                'cardiovascular_positive': cardio_score > 0.5,
                'cardio_risk': self._assess_cardio_risk(cardio_score)
            }
            
            return cardiovascular
            
        except Exception as e:
            logger.error(f"Error predicting cardiovascular toxicity for {smiles}: {str(e)}")
            return {}
    
    def predict_skin_toxicity(self, smiles: str) -> Dict[str, float]:
        """
        皮膚毒性を予測
        
        Args:
            smiles: SMILES文字列
        
        Returns:
            皮膚毒性予測値
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return {}
            
            # 皮膚毒性予測
            skin_score = self._predict_skin_toxicity(mol)
            
            skin = {
                'skin_score': skin_score,
                'skin_positive': skin_score > 0.5,
                'skin_risk': self._assess_skin_risk(skin_score)
            }
            
            return skin
            
        except Exception as e:
            logger.error(f"Error predicting skin toxicity for {smiles}: {str(e)}")
            return {}
    
    def predict_respiratory_toxicity(self, smiles: str) -> Dict[str, float]:
        """
        呼吸器毒性を予測
        
        Args:
            smiles: SMILES文字列
        
        Returns:
            呼吸器毒性予測値
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return {}
            
            # 呼吸器毒性予測
            respiratory_score = self._predict_respiratory_toxicity(mol)
            
            respiratory = {
                'respiratory_score': respiratory_score,
                'respiratory_positive': respiratory_score > 0.5,
                'respiratory_risk': self._assess_respiratory_risk(respiratory_score)
            }
            
            return respiratory
            
        except Exception as e:
            logger.error(f"Error predicting respiratory toxicity for {smiles}: {str(e)}")
            return {}
    
    def predict_comprehensive_toxicity(self, smiles: str) -> Dict[str, Dict[str, float]]:
        """
        包括的毒性予測
        
        Args:
            smiles: SMILES文字列
        
        Returns:
            包括的毒性予測値
        """
        logger.info(f"Predicting comprehensive toxicity for {smiles}")
        
        toxicity_results = {
            'ames': self.predict_ames_toxicity(smiles),
            'herg': self.predict_herg_toxicity(smiles),
            'hepatotoxicity': self.predict_hepatotoxicity(smiles),
            'cardiovascular': self.predict_cardiovascular_toxicity(smiles),
            'skin': self.predict_skin_toxicity(smiles),
            'respiratory': self.predict_respiratory_toxicity(smiles)
        }
        
        return toxicity_results
    
    def _calculate_structural_alerts(self, mol: Chem.Mol) -> float:
        """構造アラートを計算"""
        try:
            # 簡易的な構造アラート計算
            alerts = 0
            
            # 芳香族アミン
            if self._has_aromatic_amine(mol):
                alerts += 1
            
            # ニトロ基
            if self._has_nitro_group(mol):
                alerts += 1
            
            # アゾ基
            if self._has_azo_group(mol):
                alerts += 1
            
            # エポキシド
            if self._has_epoxide(mol):
                alerts += 1
            
            # アルデヒド
            if self._has_aldehyde(mol):
                alerts += 1
            
            return min(1.0, alerts / 5.0)
            
        except:
            return 0.0
    
    def _get_structural_alerts(self, mol: Chem.Mol) -> List[str]:
        """構造アラートを取得"""
        alerts = []
        
        if self._has_aromatic_amine(mol):
            alerts.append("aromatic_amine")
        if self._has_nitro_group(mol):
            alerts.append("nitro_group")
        if self._has_azo_group(mol):
            alerts.append("azo_group")
        if self._has_epoxide(mol):
            alerts.append("epoxide")
        if self._has_aldehyde(mol):
            alerts.append("aldehyde")
        
        return alerts
    
    def _has_aromatic_amine(self, mol: Chem.Mol) -> bool:
        """芳香族アミンの存在確認"""
        try:
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'N' and atom.GetIsAromatic():
                    return True
            return False
        except:
            return False
    
    def _has_nitro_group(self, mol: Chem.Mol) -> bool:
        """ニトロ基の存在確認"""
        try:
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'N':
                    for bond in atom.GetBonds():
                        if bond.GetOtherAtom(atom).GetSymbol() == 'O':
                            return True
            return False
        except:
            return False
    
    def _has_azo_group(self, mol: Chem.Mol) -> bool:
        """アゾ基の存在確認"""
        try:
            for bond in mol.GetBonds():
                if bond.GetBondType() == Chem.BondType.DOUBLE:
                    atom1 = bond.GetBeginAtom()
                    atom2 = bond.GetEndAtom()
                    if atom1.GetSymbol() == 'N' and atom2.GetSymbol() == 'N':
                        return True
            return False
        except:
            return False
    
    def _has_epoxide(self, mol: Chem.Mol) -> bool:
        """エポキシドの存在確認"""
        try:
            for ring in mol.GetRingInfo().AtomRings():
                if len(ring) == 3:
                    atoms = [mol.GetAtomWithIdx(i) for i in ring]
                    if all(atom.GetSymbol() == 'C' for atom in atoms):
                        return True
            return False
        except:
            return False
    
    def _has_aldehyde(self, mol: Chem.Mol) -> bool:
        """アルデヒドの存在確認"""
        try:
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'C':
                    for bond in atom.GetBonds():
                        if bond.GetBondType() == Chem.BondType.DOUBLE:
                            other_atom = bond.GetOtherAtom(atom)
                            if other_atom.GetSymbol() == 'O':
                                return True
            return False
        except:
            return False
    
    def _predict_herg_inhibition(self, mol: Chem.Mol) -> float:
        """hERG阻害予測"""
        try:
            # 簡易的なhERG阻害予測
            logp = Crippen.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            tpsa = Descriptors.TPSA(mol)
            
            # 簡易的な予測式
            herg_score = 0.3 * logp - 0.01 * mw + 0.05 * tpsa + 0.2
            return max(0.0, min(1.0, herg_score))
        except:
            return 0.0
    
    def _predict_hepatotoxicity(self, mol: Chem.Mol) -> float:
        """肝毒性予測"""
        try:
            # 簡易的な肝毒性予測
            logp = Crippen.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            
            # 簡易的な予測式
            hepatotoxicity_score = 0.2 * logp - 0.005 * mw + 0.3
            return max(0.0, min(1.0, hepatotoxicity_score))
        except:
            return 0.0
    
    def _predict_cardiovascular_toxicity(self, mol: Chem.Mol) -> float:
        """心血管毒性予測"""
        try:
            # 簡易的な心血管毒性予測
            logp = Crippen.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            
            # 簡易的な予測式
            cardio_score = 0.25 * logp - 0.008 * mw + 0.25
            return max(0.0, min(1.0, cardio_score))
        except:
            return 0.0
    
    def _predict_skin_toxicity(self, mol: Chem.Mol) -> float:
        """皮膚毒性予測"""
        try:
            # 簡易的な皮膚毒性予測
            logp = Crippen.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            
            # 簡易的な予測式
            skin_score = 0.15 * logp - 0.003 * mw + 0.2
            return max(0.0, min(1.0, skin_score))
        except:
            return 0.0
    
    def _predict_respiratory_toxicity(self, mol: Chem.Mol) -> float:
        """呼吸器毒性予測"""
        try:
            # 簡易的な呼吸器毒性予測
            logp = Crippen.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            
            # 簡易的な予測式
            respiratory_score = 0.2 * logp - 0.004 * mw + 0.15
            return max(0.0, min(1.0, respiratory_score))
        except:
            return 0.0
    
    def _assess_herg_risk(self, herg_score: float) -> str:
        """hERGリスク評価"""
        if herg_score < 0.3:
            return "Low"
        elif herg_score < 0.7:
            return "Medium"
        else:
            return "High"
    
    def _assess_liver_risk(self, hepatotoxicity_score: float) -> str:
        """肝リスク評価"""
        if hepatotoxicity_score < 0.3:
            return "Low"
        elif hepatotoxicity_score < 0.7:
            return "Medium"
        else:
            return "High"
    
    def _assess_cardio_risk(self, cardio_score: float) -> str:
        """心血管リスク評価"""
        if cardio_score < 0.3:
            return "Low"
        elif cardio_score < 0.7:
            return "Medium"
        else:
            return "High"
    
    def _assess_skin_risk(self, skin_score: float) -> str:
        """皮膚リスク評価"""
        if skin_score < 0.3:
            return "Low"
        elif skin_score < 0.7:
            return "Medium"
        else:
            return "High"
    
    def _assess_respiratory_risk(self, respiratory_score: float) -> str:
        """呼吸器リスク評価"""
        if respiratory_score < 0.3:
            return "Low"
        elif respiratory_score < 0.7:
            return "Medium"
        else:
            return "High"
    
    def process_molecule_batch(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        分子バッチを処理
        
        Args:
            smiles_list: SMILES文字列リスト
        
        Returns:
            毒性予測結果データフレーム
        """
        logger.info(f"Processing {len(smiles_list)} molecules for toxicity prediction")
        
        results = []
        
        for i, smiles in enumerate(smiles_list):
            if i % 100 == 0:
                logger.info(f"Processing molecule {i}/{len(smiles_list)}")
            
            toxicity_results = self.predict_comprehensive_toxicity(smiles)
            
            # 結果をフラット化
            flat_results = {'smiles': smiles}
            for category, properties in toxicity_results.items():
                for prop_name, prop_value in properties.items():
                    flat_results[f"{category}_{prop_name}"] = prop_value
            
            results.append(flat_results)
        
        # データフレームに変換
        df = pd.DataFrame(results)
        
        # 数値列の欠損値を処理
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        logger.info(f"Processed {len(df)} molecules successfully")
        return df
    
    def get_toxicity_summary(self, df: pd.DataFrame) -> Dict:
        """
        毒性要約を取得
        
        Args:
            df: 毒性データフレーム
        
        Returns:
            毒性要約辞書
        """
        summary = {
            "total_molecules": len(df),
            "toxicity_categories": ["ames", "herg", "hepatotoxicity", "cardiovascular", "skin", "respiratory"],
            "high_risk_molecules": 0,
            "medium_risk_molecules": 0,
            "low_risk_molecules": 0,
            "toxicity_statistics": {}
        }
        
        # リスクレベル別カウント
        for category in summary["toxicity_categories"]:
            score_col = f"{category}_score"
            if score_col in df.columns:
                high_risk = len(df[df[score_col] > 0.7])
                medium_risk = len(df[(df[score_col] > 0.3) & (df[score_col] <= 0.7)])
                low_risk = len(df[df[score_col] <= 0.3])
                
                summary["toxicity_statistics"][category] = {
                    "high_risk": high_risk,
                    "medium_risk": medium_risk,
                    "low_risk": low_risk,
                    "mean_score": df[score_col].mean(),
                    "std_score": df[score_col].std()
                }
        
        return summary
