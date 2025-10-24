"""
Property Predictor Module

分子物性予測システム
物理化学的性質・薬物らしさ指標の予測
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


class PropertyPredictor:
    """
    分子物性予測器
    
    物理化学的性質・薬物らしさ指標の予測
    """
    
    def __init__(self):
        """分子物性予測器を初期化"""
        logger.info("PropertyPredictor initialized")
    
    def predict_tox21_endpoints(self, smiles: str) -> Dict[str, float]:
        """
        Tox21エンドポイント予測
        
        Args:
            smiles: SMILES文字列
        
        Returns:
            Tox21エンドポイント予測結果
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return {}
            
            # Tox21エンドポイント（簡易的なRDKit記述子ベース予測）
            tox21_predictions = {
                # Nuclear Receptor (NR) assays
                'NR-AR': self._predict_nr_ar(mol),
                'NR-AR-LBD': self._predict_nr_ar_lbd(mol),
                'NR-AhR': self._predict_nr_ahr(mol),
                'NR-Aromatase': self._predict_nr_aromatase(mol),
                'NR-ER': self._predict_nr_er(mol),
                'NR-ER-LBD': self._predict_nr_er_lbd(mol),
                'NR-PPAR-gamma': self._predict_nr_ppar_gamma(mol),
                
                # Stress Response (SR) assays
                'SR-ARE': self._predict_sr_are(mol),
                'SR-ATAD5': self._predict_sr_atad5(mol),
                'SR-HSE': self._predict_sr_hse(mol),
                'SR-MMP': self._predict_sr_mmp(mol),
                'SR-p53': self._predict_sr_p53(mol)
            }
            
            return tox21_predictions
            
        except Exception as e:
            logger.error(f"Error predicting Tox21 endpoints: {e}")
            return {}
    
    def _predict_nr_ar(self, mol) -> float:
        """Androgen Receptor予測"""
        # 簡易的なRDKit記述子ベース予測
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        
        # ステロイド様構造の簡易判定
        steroid_score = 0.0
        if mw > 200 and mw < 400 and logp > 2.0:
            steroid_score = 0.3
        
        return min(1.0, max(0.0, steroid_score))
    
    def _predict_nr_ar_lbd(self, mol) -> float:
        """Androgen Receptor LBD予測"""
        return self._predict_nr_ar(mol) * 0.8
    
    def _predict_nr_ahr(self, mol) -> float:
        """Aryl Hydrocarbon Receptor予測"""
        # 芳香族環の数に基づく簡易予測
        aromatic_rings = Descriptors.NumAromaticRings(mol)
        return min(1.0, aromatic_rings * 0.2)
    
    def _predict_nr_aromatase(self, mol) -> float:
        """Aromatase予測"""
        # ステロイド様構造の簡易判定
        return self._predict_nr_ar(mol) * 0.6
    
    def _predict_nr_er(self, mol) -> float:
        """Estrogen Receptor予測"""
        # フェノール基の存在チェック
        phenol_score = 0.0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'O' and atom.GetDegree() == 1:
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetSymbol() == 'C' and neighbor.GetIsAromatic():
                        phenol_score = 0.4
                        break
        
        return min(1.0, phenol_score)
    
    def _predict_nr_er_lbd(self, mol) -> float:
        """Estrogen Receptor LBD予測"""
        return self._predict_nr_er(mol) * 0.9
    
    def _predict_nr_ppar_gamma(self, mol) -> float:
        """PPAR-gamma予測"""
        # カルボン酸基の存在チェック
        carboxyl_score = 0.0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C':
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetSymbol() == 'O' and neighbor.GetDegree() == 1:
                        carboxyl_score = 0.3
                        break
        
        return min(1.0, carboxyl_score)
    
    def _predict_sr_are(self, mol) -> float:
        """Antioxidant Response Element予測"""
        # チオール基の存在チェック
        thiol_score = 0.0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'S' and atom.GetDegree() == 1:
                thiol_score = 0.5
                break
        
        return min(1.0, thiol_score)
    
    def _predict_sr_atad5(self, mol) -> float:
        """ATAD5予測"""
        # DNA損傷関連の簡易予測
        mw = Descriptors.MolWt(mol)
        return min(1.0, max(0.0, (mw - 100) / 500))
    
    def _predict_sr_hse(self, mol) -> float:
        """Heat Shock Element予測"""
        # 熱ショック応答関連の簡易予測
        return self._predict_sr_atad5(mol) * 0.7
    
    def _predict_sr_mmp(self, mol) -> float:
        """Mitochondrial Membrane Potential予測"""
        # ミトコンドリア毒性の簡易予測
        logp = Crippen.MolLogP(mol)
        return min(1.0, max(0.0, (logp - 2.0) / 3.0))
    
    def _predict_sr_p53(self, mol) -> float:
        """p53予測"""
        # p53経路活性化の簡易予測
        return self._predict_sr_atad5(mol) * 0.8

    def predict_physicochemical_properties(self, smiles: str) -> Dict[str, float]:
        """
        物理化学的性質を予測
        
        Args:
            smiles: SMILES文字列
        
        Returns:
            物理化学的性質辞書
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return {}
            
            # 基本物性
            properties = {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Crippen.MolLogP(mol),
                'logd': Crippen.MolLogD(mol),
                'tpsa': Descriptors.TPSA(mol),
                'num_atoms': mol.GetNumAtoms(),
                'num_bonds': mol.GetNumBonds(),
                'num_rings': Descriptors.RingCount(mol),
                'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
                'num_saturated_rings': Descriptors.NumSaturatedRings(mol),
                'num_aliphatic_rings': Descriptors.NumAliphaticRings(mol),
                'num_heteroatoms': Descriptors.NumHeteroatoms(mol),
                'fraction_csp3': Descriptors.FractionCsp3(mol),
                'heavy_atom_count': Descriptors.HeavyAtomCount(mol),
                'nhoh_count': Descriptors.NHOHCount(mol),
                'no_count': Descriptors.NOCount(mol)
            }
            
            return properties
            
        except Exception as e:
            logger.error(f"Error predicting physicochemical properties for {smiles}: {str(e)}")
            return {}
    
    def predict_lipinski_rule(self, smiles: str) -> Dict[str, Union[float, bool]]:
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
    
    def predict_veber_rule(self, smiles: str) -> Dict[str, Union[float, bool]]:
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
    
    def predict_drug_likeness(self, smiles: str) -> Dict[str, float]:
        """
        薬物らしさ指標を予測
        
        Args:
            smiles: SMILES文字列
        
        Returns:
            薬物らしさ指標辞書
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return {}
            
            # QED (Quantitative Estimate of Drug-likeness)
            qed = QED.qed(mol)
            
            # SA Score (Synthetic Accessibility)
            sa_score = rdMolDescriptors.CalcSAscore(mol)
            
            # SlogP
            slogp = rdMolDescriptors.CalcSlogP(mol)
            
            # 薬物らしさ判定
            drug_like = qed > 0.5 and sa_score < 6.0
            
            drug_likeness = {
                'qed': qed,
                'sa_score': sa_score,
                'slogp': slogp,
                'drug_like': drug_like
            }
            
            return drug_likeness
            
        except Exception as e:
            logger.error(f"Error predicting drug likeness for {smiles}: {str(e)}")
            return {}
    
    def predict_pharmacophore_features(self, smiles: str) -> Dict[str, int]:
        """
        ファーマコフォア特徴量を予測
        
        Args:
            smiles: SMILES文字列
        
        Returns:
            ファーマコフォア特徴量辞書
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return {}
            
            # 官能基の数
            features = {
                'carboxylic_acids': len([a for a in mol.GetAtoms() if a.GetSymbol() == 'C' and 
                                       any(b.GetOtherAtom(a).GetSymbol() == 'O' for b in a.GetBonds())]),
                'amines': len([a for a in mol.GetAtoms() if a.GetSymbol() == 'N' and 
                             a.GetFormalCharge() > 0]),
                'hydroxyls': len([a for a in mol.GetAtoms() if a.GetSymbol() == 'O' and 
                                a.GetHybridization() == Chem.HybridizationType.SP3]),
                'carbonyls': len([a for a in mol.GetAtoms() if a.GetSymbol() == 'C' and 
                                any(b.GetBondType() == Chem.BondType.DOUBLE for b in a.GetBonds())]),
                'aromatics': Descriptors.NumAromaticAtoms(mol),
                'halogens': len([a for a in mol.GetAtoms() if a.GetSymbol() in ['F', 'Cl', 'Br', 'I']])
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error predicting pharmacophore features for {smiles}: {str(e)}")
            return {}
    
    def predict_molecular_descriptors(self, smiles: str) -> Dict[str, float]:
        """
        分子記述子を予測
        
        Args:
            smiles: SMILES文字列
        
        Returns:
            分子記述子辞書
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return {}
            
            # 包括的な分子記述子
            descriptors = {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Crippen.MolLogP(mol),
                'logd': Crippen.MolLogD(mol),
                'tpsa': Descriptors.TPSA(mol),
                'num_atoms': mol.GetNumAtoms(),
                'num_bonds': mol.GetNumBonds(),
                'num_rings': Descriptors.RingCount(mol),
                'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
                'num_saturated_rings': Descriptors.NumSaturatedRings(mol),
                'num_aliphatic_rings': Descriptors.NumAliphaticRings(mol),
                'num_heteroatoms': Descriptors.NumHeteroatoms(mol),
                'fraction_csp3': Descriptors.FractionCsp3(mol),
                'heavy_atom_count': Descriptors.HeavyAtomCount(mol),
                'nhoh_count': Descriptors.NHOHCount(mol),
                'no_count': Descriptors.NOCount(mol),
                'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'num_h_donors': Lipinski.NumHDonors(mol),
                'num_h_acceptors': Lipinski.NumHAcceptors(mol),
                'num_aromatic_atoms': Descriptors.NumAromaticAtoms(mol),
                'num_aromatic_carbocycles': Descriptors.NumAromaticCarbocycles(mol),
                'num_aromatic_heterocycles': Descriptors.NumAromaticHeterocycles(mol),
                'num_saturated_carbocycles': Descriptors.NumSaturatedCarbocycles(mol),
                'num_saturated_heterocycles': Descriptors.NumSaturatedHeterocycles(mol),
                'num_aliphatic_carbocycles': Descriptors.NumAliphaticCarbocycles(mol),
                'num_aliphatic_heterocycles': Descriptors.NumAliphaticHeterocycles(mol),
                'num_spiro_atoms': Descriptors.NumSpiroAtoms(mol),
                'num_bridgehead_atoms': Descriptors.NumBridgeheadAtoms(mol),
                'num_stereocenters': Descriptors.NumStereocenters(mol),
                'num_unspecified_stereocenters': Descriptors.NumUnspecifiedStereocenters(mol)
            }
            
            return descriptors
            
        except Exception as e:
            logger.error(f"Error predicting molecular descriptors for {smiles}: {str(e)}")
            return {}
    
    def predict_comprehensive_properties(self, smiles: str) -> Dict[str, Union[float, int, bool]]:
        """
        包括的物性を予測
        
        Args:
            smiles: SMILES文字列
        
        Returns:
            包括的物性辞書
        """
        logger.info(f"Predicting comprehensive properties for {smiles}")
        
        properties = {}
        
        # 物理化学的性質
        properties.update(self.predict_physicochemical_properties(smiles))
        
        # Lipinski's Rule
        properties.update(self.predict_lipinski_rule(smiles))
        
        # Veber's Rule
        properties.update(self.predict_veber_rule(smiles))
        
        # 薬物らしさ
        properties.update(self.predict_drug_likeness(smiles))
        
        # ファーマコフォア特徴量
        properties.update(self.predict_pharmacophore_features(smiles))
        
        return properties
    
    def process_molecule_batch(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        分子バッチを処理
        
        Args:
            smiles_list: SMILES文字列リスト
        
        Returns:
            物性予測結果データフレーム
        """
        logger.info(f"Processing {len(smiles_list)} molecules for property prediction")
        
        results = []
        
        for i, smiles in enumerate(smiles_list):
            if i % 100 == 0:
                logger.info(f"Processing molecule {i}/{len(smiles_list)}")
            
            properties = self.predict_comprehensive_properties(smiles)
            properties['smiles'] = smiles
            results.append(properties)
        
        # データフレームに変換
        df = pd.DataFrame(results)
        
        # 数値列の欠損値を処理
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        logger.info(f"Processed {len(df)} molecules successfully")
        return df
    
    def get_property_summary(self, df: pd.DataFrame) -> Dict:
        """
        物性要約を取得
        
        Args:
            df: 物性データフレーム
        
        Returns:
            物性要約辞書
        """
        summary = {
            "total_molecules": len(df),
            "property_columns": [col for col in df.columns if col != 'smiles'],
            "missing_values": df.isnull().sum().to_dict(),
            "property_statistics": {}
        }
        
        # 物性別統計
        for col in summary["property_columns"]:
            if col in df.columns:
                values = df[col].dropna()
                if len(values) > 0:
                    summary["property_statistics"][col] = {
                        "count": len(values),
                        "mean": values.mean(),
                        "std": values.std(),
                        "min": values.min(),
                        "max": values.max(),
                        "median": values.median()
                    }
        
        return summary
