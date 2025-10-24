"""
CNS作動薬骨格ライブラリ

代表的なCNS作動薬の骨格SMARTSパターンとサンプル分子
フェネチルアミン、トリプタミン、カンナビノイド、オピオイド、抗NMDA、GABA作動薬
"""

from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class CNSCompound:
    """CNS化合物情報"""
    name: str
    smiles: str
    smarts: str
    mechanism: str
    safety_notes: str
    therapeutic_use: str

class CNSScaffolds:
    """
    CNS作動薬骨格ライブラリ
    
    代表的なCNS作動薬の骨格SMARTSパターンとサンプル分子
    """
    
    def __init__(self):
        """初期化"""
        self.scaffolds = self._initialize_scaffolds()
    
    def _initialize_scaffolds(self) -> Dict[str, List[CNSCompound]]:
        """骨格データの初期化"""
        return {
            "phenethylamine": [
                CNSCompound(
                    name="Amphetamine",
                    smiles="CC(CC1=CC=CC=C1)N",
                    smarts="[CH3][CH]([NH2])[CH2][CH2][c]1[cH][cH][cH][cH][cH]1",
                    mechanism="Dopamine/Norepinephrine reuptake inhibitor",
                    safety_notes="High abuse potential, controlled substance",
                    therapeutic_use="ADHD treatment, narcolepsy"
                ),
                CNSCompound(
                    name="MDMA",
                    smiles="CCN(CC)CC1=CC2=C(C=C1)OCO2",
                    smarts="[CH3][CH2][N]([CH2][CH3])[CH2][CH2][c]1[cH][cH][c]2[cH][cH][cH][cH]1[O][CH2][O]2",
                    mechanism="Serotonin/Dopamine/Norepinephrine releaser",
                    safety_notes="Schedule I controlled substance, neurotoxicity risk",
                    therapeutic_use="Research only (PTSD therapy under investigation)"
                ),
                CNSCompound(
                    name="Mescaline",
                    smiles="COC1=CC=C(C=C1O)CCN",
                    smarts="[CH3][O][c]1[cH][cH][c]([OH])[cH][cH]1[CH2][CH2][NH2]",
                    mechanism="5-HT2A agonist",
                    safety_notes="Hallucinogen, controlled substance",
                    therapeutic_use="Research only (psychedelic therapy)"
                )
            ],
            
            "tryptamine": [
                CNSCompound(
                    name="Serotonin",
                    smiles="C1=CC2=C(C=C1O)C(=CN2)CCN",
                    smarts="[c]1[cH][cH][c]2[cH][cH][cH][cH]1[O][c]2[CH]=[CH][CH2][CH2][NH2]",
                    mechanism="Endogenous neurotransmitter",
                    safety_notes="Endogenous compound",
                    therapeutic_use="Natural neurotransmitter"
                ),
                CNSCompound(
                    name="DMT",
                    smiles="CN(C)CCC1=CNC2=CC=CC=C21",
                    smarts="[CH3][N]([CH3])[CH2][CH2][CH2][c]1[cH][nH][c]2[cH][cH][cH][cH][cH]1",
                    mechanism="5-HT2A agonist",
                    safety_notes="Hallucinogen, controlled substance",
                    therapeutic_use="Research only (psychedelic therapy)"
                ),
                CNSCompound(
                    name="Psilocybin",
                    smiles="CN(C)CC1=CNC2=CC=C(O)C=C2OP(=O)(O)O1",
                    smarts="[CH3][N]([CH3])[CH2][CH2][c]1[cH][nH][c]2[cH][cH][c]([OH])[cH][cH]2[O][P](=[O])([OH])[O]1",
                    mechanism="5-HT2A agonist (prodrug of psilocin)",
                    safety_notes="Hallucinogen, controlled substance",
                    therapeutic_use="Research only (depression, anxiety therapy)"
                )
            ],
            
            "cannabinoid": [
                CNSCompound(
                    name="THC",
                    smiles="CCCCCC1=CC(=C(C=C1)C2C=C(CCC2C(=O)O)C)C",
                    smarts="[CH3][CH2][CH2][CH2][CH2][CH2][c]1[cH][cH][c]([CH3])[cH][cH]1[CH2][CH]=[CH][CH2][CH2][CH2][C](=[O])[OH]",
                    mechanism="CB1/CB2 receptor agonist",
                    safety_notes="Psychoactive, controlled substance",
                    therapeutic_use="Medical cannabis (pain, nausea, appetite)"
                ),
                CNSCompound(
                    name="CBD",
                    smiles="CCCCCC1=CC(=C(C=C1)O)C2C=C(CCC2C(=O)O)C",
                    smarts="[CH3][CH2][CH2][CH2][CH2][CH2][c]1[cH][cH][c]([OH])[cH][cH]1[CH2][CH]=[CH][CH2][CH2][CH2][C](=[O])[OH]",
                    mechanism="CB1 antagonist, CB2 partial agonist",
                    safety_notes="Non-psychoactive, generally safe",
                    therapeutic_use="Epilepsy, anxiety, pain management"
                ),
                CNSCompound(
                    name="Anandamide",
                    smiles="CCCCCCCCCCCCCCCCCC(=O)NCC1=CC=C(C=C1)O",
                    smarts="[CH3][CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH2][C](=[O])[NH][CH2][CH2][c]1[cH][cH][c]([OH])[cH][cH]1",
                    mechanism="Endogenous CB1/CB2 agonist",
                    safety_notes="Endogenous compound",
                    therapeutic_use="Natural endocannabinoid"
                )
            ],
            
            "opioid": [
                CNSCompound(
                    name="Morphine",
                    smiles="CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",
                    smarts="[CH3][N]1[CH2][CH2][C@]23[C]4=[C]5[C]=[CH][C]([OH])=[C]4[O][C@H]2[C@@H]([OH])[CH]=[CH][C@H]3[C@H]1[C]5",
                    mechanism="μ-opioid receptor agonist",
                    safety_notes="High abuse potential, respiratory depression risk",
                    therapeutic_use="Severe pain management"
                ),
                CNSCompound(
                    name="Codeine",
                    smiles="CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",
                    smarts="[CH3][N]1[CH2][CH2][C@]23[C]4=[C]5[C]=[CH][C]([OH])=[C]4[O][C@H]2[C@@H]([OH])[CH]=[CH][C@H]3[C@H]1[C]5",
                    mechanism="μ-opioid receptor agonist (prodrug)",
                    safety_notes="Moderate abuse potential, controlled substance",
                    therapeutic_use="Cough suppression, mild pain"
                ),
                CNSCompound(
                    name="Fentanyl",
                    smiles="CCN(CC1=CC=CC=C1)C(=O)C2=CC3=C(C=C2)N(C(=O)C4=CC=CC=C4)C5=CC=CC=C5",
                    smarts="[CH3][CH2][N]([CH2][CH2][c]1[cH][cH][cH][cH][cH]1)[C](=[O])[c]2[cH][cH][c]3[cH][cH][cH][cH]2[N]([C](=[O])[c]4[cH][cH][cH][cH][cH]4)[c]5[cH][cH][cH][cH][cH]5",
                    mechanism="μ-opioid receptor agonist",
                    safety_notes="Extremely potent, high overdose risk",
                    therapeutic_use="Severe pain, anesthesia"
                )
            ],
            
            "anti_nmda": [
                CNSCompound(
                    name="Ketamine",
                    smiles="CN1CCCC1=O",
                    smarts="[CH3][N]1[CH2][CH2][CH2][CH]1=[O]",
                    mechanism="NMDA receptor antagonist",
                    safety_notes="Dissociative anesthetic, abuse potential",
                    therapeutic_use="Anesthesia, treatment-resistant depression"
                ),
                CNSCompound(
                    name="PCP",
                    smiles="CN1CCCC1=O",
                    smarts="[CH3][N]1[CH2][CH2][CH2][CH]1=[O]",
                    mechanism="NMDA receptor antagonist",
                    safety_notes="High abuse potential, Schedule II controlled",
                    therapeutic_use="Veterinary anesthetic (limited human use)"
                ),
                CNSCompound(
                    name="Memantine",
                    smiles="CN(C)CC1=CC=C(C=C1)C(C)(C)C",
                    smarts="[CH3][N]([CH3])[CH2][CH2][c]1[cH][cH][c]([CH2][C]([CH3])([CH3])[CH3])[cH][cH]1",
                    mechanism="NMDA receptor antagonist",
                    safety_notes="Generally well-tolerated",
                    therapeutic_use="Alzheimer's disease treatment"
                ),
                CNSCompound(
                    name="Dextromethorphan",
                    smiles="CN1CC[C@H]2C3=C4C=CC(O)=C3O[C@H]2[C@@H](O)C=C[C@H]1C4",
                    smarts="[CH3][N]1[CH2][CH2][C@H]2[C]3=[C]4[C]=[CH][C]([OH])=[C]3[O][C@H]2[C@@H]([OH])[CH]=[CH][C@H]1[C]4",
                    mechanism="NMDA receptor antagonist, σ1 agonist",
                    safety_notes="Generally safe at therapeutic doses",
                    therapeutic_use="Cough suppression, neuropathic pain"
                )
            ],
            
            "gaba_agonist": [
                CNSCompound(
                    name="Diazepam",
                    smiles="CN1C(=O)C2=CC=CC=C2N=C1C3=CC=CC=C3",
                    smarts="[CH3][N]1[C](=[O])[c]2[cH][cH][cH][cH][cH]2[N]=[C]1[c]3[cH][cH][cH][cH][cH]3",
                    mechanism="GABA-A receptor positive allosteric modulator",
                    safety_notes="Dependence risk, withdrawal symptoms",
                    therapeutic_use="Anxiety, muscle spasms, seizures"
                ),
                CNSCompound(
                    name="Alprazolam",
                    smiles="CN1C(=O)C2=CC=CC=C2N=C1C3=CC=CC=C3",
                    smarts="[CH3][N]1[C](=[O])[c]2[cH][cH][cH][cH][cH]2[N]=[C]1[c]3[cH][cH][cH][cH][cH]3",
                    mechanism="GABA-A receptor positive allosteric modulator",
                    safety_notes="High dependence risk, controlled substance",
                    therapeutic_use="Anxiety disorders, panic disorder"
                ),
                CNSCompound(
                    name="Phenobarbital",
                    smiles="CC1(C)C(=O)NC(=O)NC1=O",
                    smarts="[CH3][C]1([CH3])[C](=[O])[NH][C](=[O])[NH][C]1=[O]",
                    mechanism="GABA-A receptor positive allosteric modulator",
                    safety_notes="Dependence risk, respiratory depression",
                    therapeutic_use="Seizures, sedation"
                )
            ]
        }
    
    def get_scaffold_compounds(self, scaffold_type: str) -> List[CNSCompound]:
        """
        指定された骨格タイプの化合物リストを取得
        
        Args:
            scaffold_type: 骨格タイプ名
            
        Returns:
            化合物リスト
        """
        return self.scaffolds.get(scaffold_type, [])
    
    def get_all_scaffold_types(self) -> List[str]:
        """すべての骨格タイプを取得"""
        return list(self.scaffolds.keys())
    
    def find_compounds_by_mechanism(self, mechanism_keyword: str) -> List[CNSCompound]:
        """
        作用機序キーワードで化合物を検索
        
        Args:
            mechanism_keyword: 作用機序キーワード
            
        Returns:
            該当する化合物リスト
        """
        results = []
        for scaffold_type, compounds in self.scaffolds.items():
            for compound in compounds:
                if mechanism_keyword.lower() in compound.mechanism.lower():
                    results.append(compound)
        return results
    
    def get_safety_warnings(self, compound_name: str) -> Optional[str]:
        """
        化合物の安全性情報を取得
        
        Args:
            compound_name: 化合物名
            
        Returns:
            安全性情報
        """
        for scaffold_type, compounds in self.scaffolds.items():
            for compound in compounds:
                if compound.name.lower() == compound_name.lower():
                    return compound.safety_notes
        return None
    
    def export_scaffolds_to_json(self, output_path: str) -> None:
        """
        骨格データをJSONファイルにエクスポート
        
        Args:
            output_path: 出力ファイルパス
        """
        export_data = {}
        for scaffold_type, compounds in self.scaffolds.items():
            export_data[scaffold_type] = [
                {
                    "name": compound.name,
                    "smiles": compound.smiles,
                    "smarts": compound.smarts,
                    "mechanism": compound.mechanism,
                    "safety_notes": compound.safety_notes,
                    "therapeutic_use": compound.therapeutic_use
                }
                for compound in compounds
            ]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Scaffolds exported to {output_path}")

# 便利な関数
def get_cns_scaffolds() -> CNSScaffolds:
    """CNS骨格ライブラリのインスタンスを取得"""
    return CNSScaffolds()

def match_scaffold_pattern(smiles: str, scaffold_type: str) -> bool:
    """
    SMILESが指定された骨格パターンにマッチするかチェック
    
    Args:
        smiles: 分子のSMILES
        scaffold_type: 骨格タイプ
        
    Returns:
        マッチするかどうか
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        
        scaffolds = get_cns_scaffolds()
        compounds = scaffolds.get_scaffold_compounds(scaffold_type)
        
        for compound in compounds:
            pattern = Chem.MolFromSmarts(compound.smarts)
            if pattern is not None and mol.HasSubstructMatch(pattern):
                return True
        
        return False
    except ImportError:
        logger.warning("RDKit not available for scaffold matching")
        return False
    except Exception as e:
        logger.error(f"Error in scaffold matching: {e}")
        return False
