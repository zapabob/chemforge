"""
ADMET予測とCNS-MPO計算モジュール

CNS創薬向けのADMET（Absorption, Distribution, Metabolism, Excretion, Toxicity）予測と
CNS-MPO（Central Nervous System - Multiparameter Optimization）スコア計算を実装。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski
from rdkit.Chem import QED, rdMolDescriptors
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import warnings
warnings.filterwarnings('ignore')


class ADMETPredictor:
    """ADMET予測とCNS-MPO計算クラス"""
    
    def __init__(self, verbose: bool = False):
        """
        ADMET予測器を初期化
        
        Args:
            verbose: 詳細ログ出力フラグ
        """
        self.verbose = verbose
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """ADMET予測パラメータを初期化"""
        # CNS-MPO計算用パラメータ
        self.cns_mpo_weights = {
            'MW': 0.25,      # 分子量
            'LogP': 0.25,    # 脂溶性
            'HBD': 0.20,     # 水素結合ドナー
            'TPSA': 0.15,    # 極性表面積
            'pKa': 0.15      # 解離定数
        }
        
        # CNS-MPO閾値
        self.cns_mpo_thresholds = {
            'MW': (150, 500),      # 分子量範囲
            'LogP': (1, 3),        # LogP範囲
            'HBD': (0, 3),         # 水素結合ドナー数
            'TPSA': (20, 90),      # 極性表面積
            'pKa': (6, 8)          # pKa範囲
        }
        
        # ADMET予測用パラメータ
        self.admet_parameters = {
            'BBB_permeability': {
                'high': 0.8,    # 高透過性閾値
                'medium': 0.5,  # 中透過性閾値
                'low': 0.2      # 低透過性閾値
            },
            'CYP_inhibition': {
                'CYP1A2': 0.5,
                'CYP2C9': 0.5,
                'CYP2C19': 0.5,
                'CYP2D6': 0.5,
                'CYP3A4': 0.5
            },
            'hERG_toxicity': {
                'high': 0.7,     # 高毒性閾値
                'medium': 0.4,   # 中毒性閾値
                'low': 0.1       # 低毒性閾値
            }
        }
    
    def predict_admet(self, smiles: str) -> Dict[str, float]:
        """
        ADMET予測を実行
        
        Args:
            smiles: SMILES文字列
            
        Returns:
            ADMET予測結果
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            if self.verbose:
                print(f"Invalid SMILES: {smiles}")
            return self._get_default_admet_values()
        
        admet_results = {}
        
        # 基本物性
        admet_results.update(self._calculate_basic_properties(mol))
        
        # 吸収（Absorption）
        admet_results.update(self._predict_absorption(mol))
        
        # 分布（Distribution）
        admet_results.update(self._predict_distribution(mol))
        
        # 代謝（Metabolism）
        admet_results.update(self._predict_metabolism(mol))
        
        # 排泄（Excretion）
        admet_results.update(self._predict_excretion(mol))
        
        # 毒性（Toxicity）
        admet_results.update(self._predict_toxicity(mol))
        
        return admet_results
    
    def _calculate_basic_properties(self, mol: Chem.Mol) -> Dict[str, float]:
        """基本物性を計算"""
        properties = {}
        
        try:
            # 分子量
            properties['MW'] = Descriptors.MolWt(mol)
            
            # LogP
            properties['LogP'] = Crippen.MolLogP(mol)
            
            # 水素結合ドナー・アクセプター
            properties['HBD'] = Descriptors.NumHDonors(mol)
            properties['HBA'] = Descriptors.NumHAcceptors(mol)
            
            # 極性表面積
            properties['TPSA'] = Descriptors.TPSA(mol)
            
            # 回転可能結合数
            properties['RotBonds'] = Descriptors.NumRotatableBonds(mol)
            
            # 芳香環数
            properties['AromaticRings'] = Descriptors.NumAromaticRings(mol)
            
            # 重原子数
            properties['HeavyAtoms'] = Descriptors.HeavyAtomCount(mol)
            
        except Exception as e:
            if self.verbose:
                print(f"Error calculating basic properties: {e}")
            properties = self._get_default_basic_properties()
        
        return properties
    
    def _predict_absorption(self, mol: Chem.Mol) -> Dict[str, float]:
        """吸収予測"""
        absorption = {}
        
        try:
            # Caco-2透過性予測
            caco2_score = self._predict_caco2_permeability(mol)
            absorption['Caco2_permeability'] = caco2_score
            
            # 溶解度予測
            solubility_score = self._predict_solubility(mol)
            absorption['Solubility'] = solubility_score
            
            # 吸収率予測
            absorption_rate = self._predict_absorption_rate(mol)
            absorption['Absorption_rate'] = absorption_rate
            
        except Exception as e:
            if self.verbose:
                print(f"Error predicting absorption: {e}")
            absorption = {
                'Caco2_permeability': 0.5,
                'Solubility': 0.5,
                'Absorption_rate': 0.5
            }
        
        return absorption
    
    def _predict_distribution(self, mol: Chem.Mol) -> Dict[str, float]:
        """分布予測"""
        distribution = {}
        
        try:
            # BBB透過性予測
            bbb_score = self._predict_bbb_permeability(mol)
            distribution['BBB_permeability'] = bbb_score
            
            # 血漿タンパク結合率予測
            protein_binding = self._predict_protein_binding(mol)
            distribution['Protein_binding'] = protein_binding
            
            # 分布容積予測
            vd_score = self._predict_volume_distribution(mol)
            distribution['Volume_distribution'] = vd_score
            
        except Exception as e:
            if self.verbose:
                print(f"Error predicting distribution: {e}")
            distribution = {
                'BBB_permeability': 0.5,
                'Protein_binding': 0.5,
                'Volume_distribution': 0.5
            }
        
        return distribution
    
    def _predict_metabolism(self, mol: Chem.Mol) -> Dict[str, float]:
        """代謝予測"""
        metabolism = {}
        
        try:
            # CYP阻害予測
            cyp_inhibition = self._predict_cyp_inhibition(mol)
            metabolism.update(cyp_inhibition)
            
            # 代謝安定性予測
            metabolic_stability = self._predict_metabolic_stability(mol)
            metabolism['Metabolic_stability'] = metabolic_stability
            
            # クリアランス予測
            clearance = self._predict_clearance(mol)
            metabolism['Clearance'] = clearance
            
        except Exception as e:
            if self.verbose:
                print(f"Error predicting metabolism: {e}")
            metabolism = {
                'CYP1A2_inhibition': 0.5,
                'CYP2C9_inhibition': 0.5,
                'CYP2C19_inhibition': 0.5,
                'CYP2D6_inhibition': 0.5,
                'CYP3A4_inhibition': 0.5,
                'Metabolic_stability': 0.5,
                'Clearance': 0.5
            }
        
        return metabolism
    
    def _predict_excretion(self, mol: Chem.Mol) -> Dict[str, float]:
        """排泄予測"""
        excretion = {}
        
        try:
            # 半減期予測
            half_life = self._predict_half_life(mol)
            excretion['Half_life'] = half_life
            
            # 腎排泄予測
            renal_excretion = self._predict_renal_excretion(mol)
            excretion['Renal_excretion'] = renal_excretion
            
            # 胆汁排泄予測
            biliary_excretion = self._predict_biliary_excretion(mol)
            excretion['Biliary_excretion'] = biliary_excretion
            
        except Exception as e:
            if self.verbose:
                print(f"Error predicting excretion: {e}")
            excretion = {
                'Half_life': 0.5,
                'Renal_excretion': 0.5,
                'Biliary_excretion': 0.5
            }
        
        return excretion
    
    def _predict_toxicity(self, mol: Chem.Mol) -> Dict[str, float]:
        """毒性予測"""
        toxicity = {}
        
        try:
            # hERG毒性予測
            herg_toxicity = self._predict_herg_toxicity(mol)
            toxicity['hERG_toxicity'] = herg_toxicity
            
            # 肝毒性予測
            hepatotoxicity = self._predict_hepatotoxicity(mol)
            toxicity['Hepatotoxicity'] = hepatotoxicity
            
            # 遺伝毒性予測
            genotoxicity = self._predict_genotoxicity(mol)
            toxicity['Genotoxicity'] = genotoxicity
            
            # 皮膚感作性予測
            skin_sensitization = self._predict_skin_sensitization(mol)
            toxicity['Skin_sensitization'] = skin_sensitization
            
        except Exception as e:
            if self.verbose:
                print(f"Error predicting toxicity: {e}")
            toxicity = {
                'hERG_toxicity': 0.5,
                'Hepatotoxicity': 0.5,
                'Genotoxicity': 0.5,
                'Skin_sensitization': 0.5
            }
        
        return toxicity
    
    def calculate_cns_mpo(self, mol: Chem.Mol) -> float:
        """
        CNS-MPOスコアを計算
        
        Args:
            mol: RDKit分子オブジェクト
            
        Returns:
            CNS-MPOスコア (0-1)
        """
        if mol is None:
            return 0.0
        
        try:
            # 基本物性を取得
            properties = self._calculate_basic_properties(mol)
            
            # 各パラメータのスコアを計算
            scores = {}
            
            # 分子量スコア
            mw = properties['MW']
            if self.cns_mpo_thresholds['MW'][0] <= mw <= self.cns_mpo_thresholds['MW'][1]:
                scores['MW'] = 1.0
            else:
                scores['MW'] = 0.0
            
            # LogPスコア
            logp = properties['LogP']
            if self.cns_mpo_thresholds['LogP'][0] <= logp <= self.cns_mpo_thresholds['LogP'][1]:
                scores['LogP'] = 1.0
            else:
                scores['LogP'] = 0.0
            
            # 水素結合ドナースコア
            hbd = properties['HBD']
            if self.cns_mpo_thresholds['HBD'][0] <= hbd <= self.cns_mpo_thresholds['HBD'][1]:
                scores['HBD'] = 1.0
            else:
                scores['HBD'] = 0.0
            
            # 極性表面積スコア
            tpsa = properties['TPSA']
            if self.cns_mpo_thresholds['TPSA'][0] <= tpsa <= self.cns_mpo_thresholds['TPSA'][1]:
                scores['TPSA'] = 1.0
            else:
                scores['TPSA'] = 0.0
            
            # pKaスコア（簡易計算）
            pka_score = self._calculate_pka_score(mol)
            scores['pKa'] = pka_score
            
            # 重み付きスコアを計算
            cns_mpo_score = sum(
                self.cns_mpo_weights[param] * scores[param]
                for param in self.cns_mpo_weights.keys()
            )
            
            return float(cns_mpo_score)
            
        except Exception as e:
            if self.verbose:
                print(f"Error calculating CNS-MPO: {e}")
            return 0.0
    
    def _calculate_pka_score(self, mol: Chem.Mol) -> float:
        """pKaスコアを計算（簡易版）"""
        try:
            # 簡易的なpKa推定
            # 実際の実装では、より高度なpKa予測アルゴリズムを使用
            
            # 酸性基の数
            acidic_groups = 0
            # 塩基性基の数
            basic_groups = 0
            
            # 簡易的な官能基検出
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'N' and atom.GetFormalCharge() == 0:
                    basic_groups += 1
                elif atom.GetSymbol() == 'O' and atom.GetFormalCharge() == 0:
                    # ヒドロキシ基の検出
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetSymbol() == 'H':
                            acidic_groups += 1
            
            # pKaスコアの計算
            if basic_groups > 0 and acidic_groups > 0:
                return 0.8  # 両性イオン
            elif basic_groups > 0:
                return 0.6  # 塩基性
            elif acidic_groups > 0:
                return 0.4  # 酸性
            else:
                return 0.5  # 中性
            
        except Exception as e:
            if self.verbose:
                print(f"Error calculating pKa score: {e}")
            return 0.5
    
    def _predict_caco2_permeability(self, mol: Chem.Mol) -> float:
        """Caco-2透過性予測"""
        try:
            # 簡易的なCaco-2透過性予測
            # 実際の実装では、より高度な予測モデルを使用
            
            logp = Crippen.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            mw = Descriptors.MolWt(mol)
            
            # 簡易スコア計算
            if logp > 2.0 and tpsa < 90 and mw < 500:
                return 0.8  # 高透過性
            elif logp > 1.0 and tpsa < 120 and mw < 600:
                return 0.6  # 中透過性
            else:
                return 0.3  # 低透過性
                
        except Exception as e:
            if self.verbose:
                print(f"Error predicting Caco-2 permeability: {e}")
            return 0.5
    
    def _predict_solubility(self, mol: Chem.Mol) -> float:
        """溶解度予測"""
        try:
            # 簡易的な溶解度予測
            logp = Crippen.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            
            if logp < 2.0 and tpsa > 60:
                return 0.8  # 高溶解度
            elif logp < 3.0 and tpsa > 40:
                return 0.6  # 中溶解度
            else:
                return 0.3  # 低溶解度
                
        except Exception as e:
            if self.verbose:
                print(f"Error predicting solubility: {e}")
            return 0.5
    
    def _predict_absorption_rate(self, mol: Chem.Mol) -> float:
        """吸収率予測"""
        try:
            # 簡易的な吸収率予測
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            
            if mw < 400 and 1.0 < logp < 3.0:
                return 0.8  # 高吸収率
            elif mw < 500 and 0.0 < logp < 4.0:
                return 0.6  # 中吸収率
            else:
                return 0.4  # 低吸収率
                
        except Exception as e:
            if self.verbose:
                print(f"Error predicting absorption rate: {e}")
            return 0.5
    
    def _predict_bbb_permeability(self, mol: Chem.Mol) -> float:
        """BBB透過性予測"""
        try:
            # 簡易的なBBB透過性予測
            logp = Crippen.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            mw = Descriptors.MolWt(mol)
            hbd = Descriptors.NumHDonors(mol)
            
            # CNS-MPO基準に基づく予測
            if (1.0 <= logp <= 3.0 and 
                tpsa <= 90 and 
                mw <= 500 and 
                hbd <= 3):
                return 0.8  # 高BBB透過性
            elif (0.0 <= logp <= 4.0 and 
                  tpsa <= 120 and 
                  mw <= 600 and 
                  hbd <= 5):
                return 0.6  # 中BBB透過性
            else:
                return 0.3  # 低BBB透過性
                
        except Exception as e:
            if self.verbose:
                print(f"Error predicting BBB permeability: {e}")
            return 0.5
    
    def _predict_protein_binding(self, mol: Chem.Mol) -> float:
        """血漿タンパク結合率予測"""
        try:
            # 簡易的なタンパク結合率予測
            logp = Crippen.MolLogP(mol)
            aromatic_rings = Descriptors.NumAromaticRings(mol)
            
            if logp > 3.0 and aromatic_rings > 2:
                return 0.8  # 高結合率
            elif logp > 2.0 and aromatic_rings > 1:
                return 0.6  # 中結合率
            else:
                return 0.4  # 低結合率
                
        except Exception as e:
            if self.verbose:
                print(f"Error predicting protein binding: {e}")
            return 0.5
    
    def _predict_volume_distribution(self, mol: Chem.Mol) -> float:
        """分布容積予測"""
        try:
            # 簡易的な分布容積予測
            logp = Crippen.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            
            if logp > 2.0 and mw < 400:
                return 0.8  # 高分布容積
            elif logp > 1.0 and mw < 500:
                return 0.6  # 中分布容積
            else:
                return 0.4  # 低分布容積
                
        except Exception as e:
            if self.verbose:
                print(f"Error predicting volume distribution: {e}")
            return 0.5
    
    def _predict_cyp_inhibition(self, mol: Chem.Mol) -> Dict[str, float]:
        """CYP阻害予測"""
        try:
            # 簡易的なCYP阻害予測
            # 実際の実装では、より高度な予測モデルを使用
            
            cyp_inhibition = {}
            for cyp in ['CYP1A2', 'CYP2C9', 'CYP2C19', 'CYP2D6', 'CYP3A4']:
                # 簡易スコア計算
                logp = Crippen.MolLogP(mol)
                aromatic_rings = Descriptors.NumAromaticRings(mol)
                
                if logp > 2.0 and aromatic_rings > 1:
                    cyp_inhibition[f'{cyp}_inhibition'] = 0.7  # 高阻害
                elif logp > 1.0 and aromatic_rings > 0:
                    cyp_inhibition[f'{cyp}_inhibition'] = 0.5  # 中阻害
                else:
                    cyp_inhibition[f'{cyp}_inhibition'] = 0.3  # 低阻害
            
            return cyp_inhibition
            
        except Exception as e:
            if self.verbose:
                print(f"Error predicting CYP inhibition: {e}")
            return {f'{cyp}_inhibition': 0.5 for cyp in ['CYP1A2', 'CYP2C9', 'CYP2C19', 'CYP2D6', 'CYP3A4']}
    
    def _predict_metabolic_stability(self, mol: Chem.Mol) -> float:
        """代謝安定性予測"""
        try:
            # 簡易的な代謝安定性予測
            logp = Crippen.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            aromatic_rings = Descriptors.NumAromaticRings(mol)
            
            if logp > 2.0 and tpsa < 80 and aromatic_rings > 1:
                return 0.8  # 高安定性
            elif logp > 1.0 and tpsa < 100:
                return 0.6  # 中安定性
            else:
                return 0.4  # 低安定性
                
        except Exception as e:
            if self.verbose:
                print(f"Error predicting metabolic stability: {e}")
            return 0.5
    
    def _predict_clearance(self, mol: Chem.Mol) -> float:
        """クリアランス予測"""
        try:
            # 簡易的なクリアランス予測
            logp = Crippen.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            
            if logp < 2.0 and mw < 400:
                return 0.8  # 高クリアランス
            elif logp < 3.0 and mw < 500:
                return 0.6  # 中クリアランス
            else:
                return 0.4  # 低クリアランス
                
        except Exception as e:
            if self.verbose:
                print(f"Error predicting clearance: {e}")
            return 0.5
    
    def _predict_half_life(self, mol: Chem.Mol) -> float:
        """半減期予測"""
        try:
            # 簡易的な半減期予測
            logp = Crippen.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            
            if logp > 3.0 and mw > 400:
                return 0.8  # 長半減期
            elif logp > 2.0 and mw > 300:
                return 0.6  # 中半減期
            else:
                return 0.4  # 短半減期
                
        except Exception as e:
            if self.verbose:
                print(f"Error predicting half-life: {e}")
            return 0.5
    
    def _predict_renal_excretion(self, mol: Chem.Mol) -> float:
        """腎排泄予測"""
        try:
            # 簡易的な腎排泄予測
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            
            if mw < 300 and logp < 2.0:
                return 0.8  # 高腎排泄
            elif mw < 500 and logp < 3.0:
                return 0.6  # 中腎排泄
            else:
                return 0.4  # 低腎排泄
                
        except Exception as e:
            if self.verbose:
                print(f"Error predicting renal excretion: {e}")
            return 0.5
    
    def _predict_biliary_excretion(self, mol: Chem.Mol) -> float:
        """胆汁排泄予測"""
        try:
            # 簡易的な胆汁排泄予測
            logp = Crippen.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            
            if logp > 2.0 and mw > 400:
                return 0.8  # 高胆汁排泄
            elif logp > 1.0 and mw > 300:
                return 0.6  # 中胆汁排泄
            else:
                return 0.4  # 低胆汁排泄
                
        except Exception as e:
            if self.verbose:
                print(f"Error predicting biliary excretion: {e}")
            return 0.5
    
    def _predict_herg_toxicity(self, mol: Chem.Mol) -> float:
        """hERG毒性予測"""
        try:
            # 簡易的なhERG毒性予測
            # 実際の実装では、より高度な毒性予測モデルを使用
            
            logp = Crippen.MolLogP(mol)
            aromatic_rings = Descriptors.NumAromaticRings(mol)
            rot_bonds = Descriptors.NumRotatableBonds(mol)
            
            if logp > 3.0 and aromatic_rings > 2 and rot_bonds > 5:
                return 0.8  # 高毒性
            elif logp > 2.0 and aromatic_rings > 1 and rot_bonds > 3:
                return 0.6  # 中毒性
            else:
                return 0.3  # 低毒性
                
        except Exception as e:
            if self.verbose:
                print(f"Error predicting hERG toxicity: {e}")
            return 0.5
    
    def _predict_hepatotoxicity(self, mol: Chem.Mol) -> float:
        """肝毒性予測"""
        try:
            # 簡易的な肝毒性予測
            logp = Crippen.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            aromatic_rings = Descriptors.NumAromaticRings(mol)
            
            if logp > 3.0 and mw > 400 and aromatic_rings > 2:
                return 0.7  # 高肝毒性
            elif logp > 2.0 and mw > 300 and aromatic_rings > 1:
                return 0.5  # 中肝毒性
            else:
                return 0.3  # 低肝毒性
                
        except Exception as e:
            if self.verbose:
                print(f"Error predicting hepatotoxicity: {e}")
            return 0.5
    
    def _predict_genotoxicity(self, mol: Chem.Mol) -> float:
        """遺伝毒性予測"""
        try:
            # 簡易的な遺伝毒性予測
            aromatic_rings = Descriptors.NumAromaticRings(mol)
            mw = Descriptors.MolWt(mol)
            
            if aromatic_rings > 3 and mw > 300:
                return 0.7  # 高遺伝毒性
            elif aromatic_rings > 1 and mw > 200:
                return 0.5  # 中遺伝毒性
            else:
                return 0.3  # 低遺伝毒性
                
        except Exception as e:
            if self.verbose:
                print(f"Error predicting genotoxicity: {e}")
            return 0.5
    
    def _predict_skin_sensitization(self, mol: Chem.Mol) -> float:
        """皮膚感作性予測"""
        try:
            # 簡易的な皮膚感作性予測
            logp = Crippen.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            
            if logp > 2.0 and mw < 500:
                return 0.6  # 中感作性
            else:
                return 0.3  # 低感作性
                
        except Exception as e:
            if self.verbose:
                print(f"Error predicting skin sensitization: {e}")
            return 0.5
    
    def _get_default_admet_values(self) -> Dict[str, float]:
        """デフォルトADMET値を取得"""
        return {
            'MW': 0.0, 'LogP': 0.0, 'HBD': 0.0, 'HBA': 0.0, 'TPSA': 0.0,
            'RotBonds': 0.0, 'AromaticRings': 0.0, 'HeavyAtoms': 0.0,
            'Caco2_permeability': 0.5, 'Solubility': 0.5, 'Absorption_rate': 0.5,
            'BBB_permeability': 0.5, 'Protein_binding': 0.5, 'Volume_distribution': 0.5,
            'CYP1A2_inhibition': 0.5, 'CYP2C9_inhibition': 0.5, 'CYP2C19_inhibition': 0.5,
            'CYP2D6_inhibition': 0.5, 'CYP3A4_inhibition': 0.5,
            'Metabolic_stability': 0.5, 'Clearance': 0.5,
            'Half_life': 0.5, 'Renal_excretion': 0.5, 'Biliary_excretion': 0.5,
            'hERG_toxicity': 0.5, 'Hepatotoxicity': 0.5, 'Genotoxicity': 0.5,
            'Skin_sensitization': 0.5
        }
    
    def _get_default_basic_properties(self) -> Dict[str, float]:
        """デフォルト基本物性値を取得"""
        return {
            'MW': 0.0, 'LogP': 0.0, 'HBD': 0.0, 'HBA': 0.0, 'TPSA': 0.0,
            'RotBonds': 0.0, 'AromaticRings': 0.0, 'HeavyAtoms': 0.0
        }
    
    def extract_admet_features(self, mol: Chem.Mol) -> np.ndarray:
        """
        ADMET特徴量を抽出
        
        Args:
            mol: RDKit分子オブジェクト
            
        Returns:
            ADMET特徴量ベクトル (10次元)
        """
        if mol is None:
            return np.zeros(10, dtype=np.float32)
        
        try:
            # ADMET予測を実行
            admet_results = self.predict_admet(Chem.MolToSmiles(mol))
            
            # 特徴量ベクトルを構築
            features = np.zeros(10, dtype=np.float32)
            
            # 主要ADMET特徴量を選択
            features[0] = admet_results.get('BBB_permeability', 0.5)
            features[1] = admet_results.get('CYP3A4_inhibition', 0.5)
            features[2] = admet_results.get('hERG_toxicity', 0.5)
            features[3] = admet_results.get('Caco2_permeability', 0.5)
            features[4] = admet_results.get('LogP', 0.0)
            features[5] = admet_results.get('Solubility', 0.5)
            features[6] = admet_results.get('Protein_binding', 0.5)
            features[7] = admet_results.get('Clearance', 0.5)
            features[8] = admet_results.get('Half_life', 0.5)
            features[9] = admet_results.get('Hepatotoxicity', 0.5)
            
            return features
            
        except Exception as e:
            if self.verbose:
                print(f"Error extracting ADMET features: {e}")
            return np.zeros(10, dtype=np.float32)
    
    def get_admet_summary(self, mol: Chem.Mol) -> Dict[str, Union[float, Dict]]:
        """
        ADMET予測サマリーを取得
        
        Args:
            mol: RDKit分子オブジェクト
            
        Returns:
            ADMET予測サマリー
        """
        if mol is None:
            return {
                'CNS_MPO_score': 0.0,
                'ADMET_results': self._get_default_admet_values()
            }
        
        try:
            # CNS-MPOスコアを計算
            cns_mpo_score = self.calculate_cns_mpo(mol)
            
            # ADMET予測を実行
            admet_results = self.predict_admet(Chem.MolToSmiles(mol))
            
            return {
                'CNS_MPO_score': cns_mpo_score,
                'ADMET_results': admet_results
            }
            
        except Exception as e:
            if self.verbose:
                print(f"Error getting ADMET summary: {e}")
            return {
                'CNS_MPO_score': 0.0,
                'ADMET_results': self._get_default_admet_values()
            }


# 便利関数
def predict_admet_from_smiles(smiles: str, verbose: bool = False) -> Dict[str, float]:
    """SMILES文字列からADMET予測を実行"""
    predictor = ADMETPredictor(verbose=verbose)
    return predictor.predict_admet(smiles)


def calculate_cns_mpo_from_smiles(smiles: str, verbose: bool = False) -> float:
    """SMILES文字列からCNS-MPOスコアを計算"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        if verbose:
            print(f"Invalid SMILES: {smiles}")
        return 0.0
    
    predictor = ADMETPredictor(verbose=verbose)
    return predictor.calculate_cns_mpo(mol)


def extract_admet_features_from_smiles(smiles: str, verbose: bool = False) -> np.ndarray:
    """SMILES文字列からADMET特徴量を抽出"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        if verbose:
            print(f"Invalid SMILES: {smiles}")
        return np.zeros(10, dtype=np.float32)
    
    predictor = ADMETPredictor(verbose=verbose)
    return predictor.extract_admet_features(mol)


if __name__ == "__main__":
    # テスト実行
    print("🧬 ADMET予測器テスト")
    print("=" * 50)
    
    # テスト用SMILES
    test_smiles = [
        "CC(CC1=CC=CC=C1)N",  # アンフェタミン
        "CCN(CC)CC1=CC2=C(C=C1)OCO2",  # MDMA
        "CCN(CC)CC1=CNC2=CC=CC=C21",  # DMT
        "CN1CC[C@]23C4=C5C6=C(C=CC=C6)O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5=C4O",  # モルヒネ
        "CCCCCC1=CC(=C(C(=C1)O)C2C=C(CCC2C(C)C)C)C",  # THC
    ]
    
    predictor = ADMETPredictor(verbose=True)
    
    for i, smiles in enumerate(test_smiles):
        print(f"\n📋 テスト {i+1}: {smiles}")
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("  ❌ Invalid SMILES")
            continue
        
        # ADMET予測
        admet_results = predictor.predict_admet(smiles)
        print(f"  BBB透過性: {admet_results.get('BBB_permeability', 0.0):.3f}")
        print(f"  hERG毒性: {admet_results.get('hERG_toxicity', 0.0):.3f}")
        print(f"  LogP: {admet_results.get('LogP', 0.0):.3f}")
        
        # CNS-MPOスコア
        cns_mpo_score = predictor.calculate_cns_mpo(mol)
        print(f"  CNS-MPO: {cns_mpo_score:.3f}")
        
        # 特徴量抽出
        features = predictor.extract_admet_features(mol)
        print(f"  特徴量次元: {features.shape}")
        print(f"  特徴量値: {features[:5]}")  # 最初の5次元のみ表示
    
    print("\n✅ ADMET予測器テスト完了！")
