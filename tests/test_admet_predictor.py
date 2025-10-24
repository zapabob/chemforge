"""
ADMET Predictor Tests

ADMET予測器のユニットテスト
"""

import pytest
import torch
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from chemforge.admet.admet_predictor import ADMETPredictor
from chemforge.admet.property_predictor import PropertyPredictor
from chemforge.admet.toxicity_predictor import ToxicityPredictor
from chemforge.admet.drug_likeness import DrugLikenessPredictor
from chemforge.admet.cns_mpo import CNSMPOCalculator


class TestADMETPredictor:
    """ADMET予測器のテストクラス"""
    
    def setup_method(self):
        """テスト前のセットアップ"""
        self.predictor = ADMETPredictor()
        self.test_smiles = "CCO"  # エタノール
    
    def test_initialization(self):
        """初期化テスト"""
        assert self.predictor.use_external_apis == True
        assert self.predictor.swissadme_url == "http://www.swissadme.ch/api"
        assert self.predictor.pkcsm_url == "http://biosig.unimelb.edu.au/pkcsm/api"
    
    def test_predict_absorption(self):
        """吸収予測テスト"""
        absorption = self.predictor.predict_absorption(self.test_smiles)
        
        assert isinstance(absorption, dict)
        assert 'molecular_weight' in absorption
        assert 'logp' in absorption
        assert 'tpsa' in absorption
        assert 'hbd' in absorption
        assert 'hba' in absorption
        assert 'caco2' in absorption
        assert 'hia' in absorption
        assert 'bioavailability' in absorption
    
    def test_predict_distribution(self):
        """分布予測テスト"""
        distribution = self.predictor.predict_distribution(self.test_smiles)
        
        assert isinstance(distribution, dict)
        assert 'vd' in distribution
        assert 'ppb' in distribution
        assert 'bbb' in distribution
        assert 'cns_penetration' in distribution
    
    def test_predict_metabolism(self):
        """代謝予測テスト"""
        metabolism = self.predictor.predict_metabolism(self.test_smiles)
        
        assert isinstance(metabolism, dict)
        assert 'cyp450_2d6' in metabolism
        assert 'cyp450_3a4' in metabolism
        assert 'cyp450_1a2' in metabolism
        assert 'cyp450_2c19' in metabolism
        assert 'cyp450_2c9' in metabolism
        assert 'cyp450_inhibition' in metabolism
    
    def test_predict_excretion(self):
        """排泄予測テスト"""
        excretion = self.predictor.predict_excretion(self.test_smiles)
        
        assert isinstance(excretion, dict)
        assert 'cl' in excretion
        assert 't1_2' in excretion
        assert 'renal_clearance' in excretion
    
    def test_predict_toxicity(self):
        """毒性予測テスト"""
        toxicity = self.predictor.predict_toxicity(self.test_smiles)
        
        assert isinstance(toxicity, dict)
        assert 'ames' in toxicity
        assert 'herg' in toxicity
        assert 'dili' in toxicity
        assert 'skin_sens' in toxicity
        assert 'respiratory' in toxicity
        assert 'hepatotoxicity' in toxicity
    
    def test_predict_comprehensive_admet(self):
        """包括的ADMET予測テスト"""
        admet_results = self.predictor.predict_comprehensive_admet(self.test_smiles)
        
        assert isinstance(admet_results, dict)
        assert 'absorption' in admet_results
        assert 'distribution' in admet_results
        assert 'metabolism' in admet_results
        assert 'excretion' in admet_results
        assert 'toxicity' in admet_results
    
    def test_get_admet_summary(self):
        """ADMET要約取得テスト"""
        admet_results = self.predictor.predict_comprehensive_admet(self.test_smiles)
        summary = self.predictor.get_admet_summary(admet_results)
        
        assert isinstance(summary, dict)
        assert 'absorption_count' in summary
        assert 'distribution_count' in summary
        assert 'metabolism_count' in summary
        assert 'excretion_count' in summary
        assert 'toxicity_count' in summary
    
    def test_process_molecule_batch(self):
        """分子バッチ処理テスト"""
        smiles_list = ["CCO", "CCCO", "CCCCCO"]
        df = self.predictor.process_molecule_batch(smiles_list)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'smiles' in df.columns
        assert not df.empty
    
    def test_invalid_smiles(self):
        """無効なSMILESテスト"""
        invalid_smiles = "invalid_smiles"
        absorption = self.predictor.predict_absorption(invalid_smiles)
        
        assert absorption == {}


class TestPropertyPredictor:
    """物性予測器のテストクラス"""
    
    def setup_method(self):
        """テスト前のセットアップ"""
        self.predictor = PropertyPredictor()
        self.test_smiles = "CCO"  # エタノール
    
    def test_initialization(self):
        """初期化テスト"""
        assert self.predictor is not None
    
    def test_predict_physicochemical_properties(self):
        """物理化学的性質予測テスト"""
        properties = self.predictor.predict_physicochemical_properties(self.test_smiles)
        
        assert isinstance(properties, dict)
        assert 'molecular_weight' in properties
        assert 'logp' in properties
        assert 'logd' in properties
        assert 'tpsa' in properties
        assert 'num_atoms' in properties
        assert 'num_bonds' in properties
    
    def test_predict_lipinski_rule(self):
        """Lipinski's Rule予測テスト"""
        lipinski = self.predictor.predict_lipinski_rule(self.test_smiles)
        
        assert isinstance(lipinski, dict)
        assert 'molecular_weight' in lipinski
        assert 'logp' in lipinski
        assert 'hbd' in lipinski
        assert 'hba' in lipinski
        assert 'violations' in lipinski
        assert 'compliant' in lipinski
    
    def test_predict_veber_rule(self):
        """Veber's Rule予測テスト"""
        veber = self.predictor.predict_veber_rule(self.test_smiles)
        
        assert isinstance(veber, dict)
        assert 'tpsa' in veber
        assert 'rotatable_bonds' in veber
        assert 'compliant' in veber
    
    def test_predict_drug_likeness(self):
        """薬物らしさ予測テスト"""
        drug_likeness = self.predictor.predict_drug_likeness(self.test_smiles)
        
        assert isinstance(drug_likeness, dict)
        assert 'qed' in drug_likeness
        assert 'sa_score' in drug_likeness
        assert 'slogp' in drug_likeness
        assert 'drug_like' in drug_likeness
    
    def test_predict_pharmacophore_features(self):
        """ファーマコフォア特徴量予測テスト"""
        features = self.predictor.predict_pharmacophore_features(self.test_smiles)
        
        assert isinstance(features, dict)
        assert 'carboxylic_acids' in features
        assert 'amines' in features
        assert 'hydroxyls' in features
        assert 'carbonyls' in features
        assert 'aromatics' in features
        assert 'halogens' in features
    
    def test_predict_molecular_descriptors(self):
        """分子記述子予測テスト"""
        descriptors = self.predictor.predict_molecular_descriptors(self.test_smiles)
        
        assert isinstance(descriptors, dict)
        assert 'molecular_weight' in descriptors
        assert 'logp' in descriptors
        assert 'tpsa' in descriptors
        assert 'num_atoms' in descriptors
        assert 'num_bonds' in descriptors
    
    def test_predict_comprehensive_properties(self):
        """包括的物性予測テスト"""
        properties = self.predictor.predict_comprehensive_properties(self.test_smiles)
        
        assert isinstance(properties, dict)
        assert 'molecular_weight' in properties
        assert 'logp' in properties
        assert 'tpsa' in properties
        assert 'violations' in properties
        assert 'compliant' in properties
        assert 'qed' in properties
        assert 'sa_score' in properties
    
    def test_process_molecule_batch(self):
        """分子バッチ処理テスト"""
        smiles_list = ["CCO", "CCCO", "CCCCCO"]
        df = self.predictor.process_molecule_batch(smiles_list)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'smiles' in df.columns
        assert not df.empty
    
    def test_get_property_summary(self):
        """物性要約取得テスト"""
        smiles_list = ["CCO", "CCCO", "CCCCCO"]
        df = self.predictor.process_molecule_batch(smiles_list)
        summary = self.predictor.get_property_summary(df)
        
        assert isinstance(summary, dict)
        assert 'total_molecules' in summary
        assert 'property_columns' in summary
        assert 'missing_values' in summary
        assert 'property_statistics' in summary


class TestToxicityPredictor:
    """毒性予測器のテストクラス"""
    
    def setup_method(self):
        """テスト前のセットアップ"""
        self.predictor = ToxicityPredictor()
        self.test_smiles = "CCO"  # エタノール
    
    def test_initialization(self):
        """初期化テスト"""
        assert self.predictor is not None
    
    def test_predict_ames_toxicity(self):
        """Ames試験毒性予測テスト"""
        ames = self.predictor.predict_ames_toxicity(self.test_smiles)
        
        assert isinstance(ames, dict)
        assert 'ames_score' in ames
        assert 'ames_positive' in ames
        assert 'structural_alerts' in ames
    
    def test_predict_herg_toxicity(self):
        """hERG毒性予測テスト"""
        herg = self.predictor.predict_herg_toxicity(self.test_smiles)
        
        assert isinstance(herg, dict)
        assert 'herg_score' in herg
        assert 'herg_positive' in herg
        assert 'herg_risk' in herg
    
    def test_predict_hepatotoxicity(self):
        """肝毒性予測テスト"""
        hepatotoxicity = self.predictor.predict_hepatotoxicity(self.test_smiles)
        
        assert isinstance(hepatotoxicity, dict)
        assert 'hepatotoxicity_score' in hepatotoxicity
        assert 'hepatotoxicity_positive' in hepatotoxicity
        assert 'liver_risk' in hepatotoxicity
    
    def test_predict_cardiovascular_toxicity(self):
        """心血管毒性予測テスト"""
        cardiovascular = self.predictor.predict_cardiovascular_toxicity(self.test_smiles)
        
        assert isinstance(cardiovascular, dict)
        assert 'cardiovascular_score' in cardiovascular
        assert 'cardiovascular_positive' in cardiovascular
        assert 'cardio_risk' in cardiovascular
    
    def test_predict_skin_toxicity(self):
        """皮膚毒性予測テスト"""
        skin = self.predictor.predict_skin_toxicity(self.test_smiles)
        
        assert isinstance(skin, dict)
        assert 'skin_score' in skin
        assert 'skin_positive' in skin
        assert 'skin_risk' in skin
    
    def test_predict_respiratory_toxicity(self):
        """呼吸器毒性予測テスト"""
        respiratory = self.predictor.predict_respiratory_toxicity(self.test_smiles)
        
        assert isinstance(respiratory, dict)
        assert 'respiratory_score' in respiratory
        assert 'respiratory_positive' in respiratory
        assert 'respiratory_risk' in respiratory
    
    def test_predict_comprehensive_toxicity(self):
        """包括的毒性予測テスト"""
        toxicity = self.predictor.predict_comprehensive_toxicity(self.test_smiles)
        
        assert isinstance(toxicity, dict)
        assert 'ames' in toxicity
        assert 'herg' in toxicity
        assert 'hepatotoxicity' in toxicity
        assert 'cardiovascular' in toxicity
        assert 'skin' in toxicity
        assert 'respiratory' in toxicity
    
    def test_process_molecule_batch(self):
        """分子バッチ処理テスト"""
        smiles_list = ["CCO", "CCCO", "CCCCCO"]
        df = self.predictor.process_molecule_batch(smiles_list)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'smiles' in df.columns
        assert not df.empty
    
    def test_get_toxicity_summary(self):
        """毒性要約取得テスト"""
        smiles_list = ["CCO", "CCCO", "CCCCCO"]
        df = self.predictor.process_molecule_batch(smiles_list)
        summary = self.predictor.get_toxicity_summary(df)
        
        assert isinstance(summary, dict)
        assert 'total_molecules' in summary
        assert 'toxicity_categories' in summary
        assert 'high_risk_molecules' in summary
        assert 'medium_risk_molecules' in summary
        assert 'low_risk_molecules' in summary
        assert 'toxicity_statistics' in summary


class TestDrugLikenessPredictor:
    """薬物らしさ予測器のテストクラス"""
    
    def setup_method(self):
        """テスト前のセットアップ"""
        self.predictor = DrugLikenessPredictor()
        self.test_smiles = "CCO"  # エタノール
    
    def test_initialization(self):
        """初期化テスト"""
        assert self.predictor is not None
    
    def test_predict_lipinski_rule(self):
        """Lipinski's Rule予測テスト"""
        lipinski = self.predictor.predict_lipinski_rule(self.test_smiles)
        
        assert isinstance(lipinski, dict)
        assert 'molecular_weight' in lipinski
        assert 'logp' in lipinski
        assert 'hbd' in lipinski
        assert 'hba' in lipinski
        assert 'violations' in lipinski
        assert 'compliant' in lipinski
    
    def test_predict_veber_rule(self):
        """Veber's Rule予測テスト"""
        veber = self.predictor.predict_veber_rule(self.test_smiles)
        
        assert isinstance(veber, dict)
        assert 'tpsa' in veber
        assert 'rotatable_bonds' in veber
        assert 'compliant' in veber
    
    def test_predict_qed(self):
        """QED予測テスト"""
        qed = self.predictor.predict_qed(self.test_smiles)
        
        assert isinstance(qed, dict)
        assert 'qed' in qed
        assert 'qed_interpretation' in qed
        assert 'drug_like' in qed
    
    def test_predict_sa_score(self):
        """SA Score予測テスト"""
        sa_score = self.predictor.predict_sa_score(self.test_smiles)
        
        assert isinstance(sa_score, dict)
        assert 'sa_score' in sa_score
        assert 'sa_interpretation' in sa_score
        assert 'synthesizable' in sa_score
    
    def test_predict_lead_likeness(self):
        """Lead-likeness予測テスト"""
        lead_likeness = self.predictor.predict_lead_likeness(self.test_smiles)
        
        assert isinstance(lead_likeness, dict)
        assert 'molecular_weight' in lead_likeness
        assert 'logp' in lead_likeness
        assert 'tpsa' in lead_likeness
        assert 'rotatable_bonds' in lead_likeness
        assert 'lead_like' in lead_likeness
    
    def test_predict_fragment_likeness(self):
        """Fragment-likeness予測テスト"""
        fragment_likeness = self.predictor.predict_fragment_likeness(self.test_smiles)
        
        assert isinstance(fragment_likeness, dict)
        assert 'molecular_weight' in fragment_likeness
        assert 'logp' in fragment_likeness
        assert 'tpsa' in fragment_likeness
        assert 'rotatable_bonds' in fragment_likeness
        assert 'fragment_like' in fragment_likeness
    
    def test_predict_drug_likeness_score(self):
        """薬物らしさスコア予測テスト"""
        drug_likeness_score = self.predictor.predict_drug_likeness_score(self.test_smiles)
        
        assert isinstance(drug_likeness_score, dict)
        assert 'drug_likeness_score' in drug_likeness_score
        assert 'interpretation' in drug_likeness_score
        assert 'drug_like' in drug_likeness_score
        assert 'lipinski_compliant' in drug_likeness_score
        assert 'veber_compliant' in drug_likeness_score
        assert 'qed' in drug_likeness_score
        assert 'sa_score' in drug_likeness_score
        assert 'lead_like' in drug_likeness_score
        assert 'fragment_like' in drug_likeness_score
    
    def test_predict_comprehensive_drug_likeness(self):
        """包括的薬物らしさ予測テスト"""
        drug_likeness = self.predictor.predict_comprehensive_drug_likeness(self.test_smiles)
        
        assert isinstance(drug_likeness, dict)
        assert 'molecular_weight' in drug_likeness
        assert 'logp' in drug_likeness
        assert 'tpsa' in drug_likeness
        assert 'violations' in drug_likeness
        assert 'compliant' in drug_likeness
        assert 'qed' in drug_likeness
        assert 'sa_score' in drug_likeness
        assert 'drug_likeness_score' in drug_likeness
    
    def test_process_molecule_batch(self):
        """分子バッチ処理テスト"""
        smiles_list = ["CCO", "CCCO", "CCCCCO"]
        df = self.predictor.process_molecule_batch(smiles_list)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'smiles' in df.columns
        assert not df.empty
    
    def test_get_drug_likeness_summary(self):
        """薬物らしさ要約取得テスト"""
        smiles_list = ["CCO", "CCCO", "CCCCCO"]
        df = self.predictor.process_molecule_batch(smiles_list)
        summary = self.predictor.get_drug_likeness_summary(df)
        
        assert isinstance(summary, dict)
        assert 'total_molecules' in summary
        assert 'drug_like_molecules' in summary
        assert 'lipinski_compliant' in summary
        assert 'veber_compliant' in summary
        assert 'lead_like_molecules' in summary
        assert 'fragment_like_molecules' in summary
        assert 'drug_likeness_statistics' in summary


class TestCNSMPOCalculator:
    """CNS-MPO計算器のテストクラス"""
    
    def setup_method(self):
        """テスト前のセットアップ"""
        self.calculator = CNSMPOCalculator()
        self.test_smiles = "CCO"  # エタノール
    
    def test_initialization(self):
        """初期化テスト"""
        assert self.calculator is not None
    
    def test_calculate_cns_mpo(self):
        """CNS-MPO計算テスト"""
        cns_mpo = self.calculator.calculate_cns_mpo(self.test_smiles)
        
        assert isinstance(cns_mpo, dict)
        assert 'molecular_weight' in cns_mpo
        assert 'logp' in cns_mpo
        assert 'tpsa' in cns_mpo
        assert 'hbd' in cns_mpo
        assert 'hba' in cns_mpo
        assert 'rotatable_bonds' in cns_mpo
        assert 'cns_mpo_score' in cns_mpo
        assert 'cns_mpo_interpretation' in cns_mpo
    
    def test_calculate_cns_mpo_plus(self):
        """CNS-MPO+計算テスト"""
        cns_mpo_plus = self.calculator.calculate_cns_mpo_plus(self.test_smiles)
        
        assert isinstance(cns_mpo_plus, dict)
        assert 'molecular_weight' in cns_mpo_plus
        assert 'logp' in cns_mpo_plus
        assert 'tpsa' in cns_mpo_plus
        assert 'hbd' in cns_mpo_plus
        assert 'hba' in cns_mpo_plus
        assert 'rotatable_bonds' in cns_mpo_plus
        assert 'num_rings' in cns_mpo_plus
        assert 'num_aromatic_rings' in cns_mpo_plus
        assert 'num_heteroatoms' in cns_mpo_plus
        assert 'fraction_csp3' in cns_mpo_plus
        assert 'cns_mpo_plus_score' in cns_mpo_plus
        assert 'cns_mpo_plus_interpretation' in cns_mpo_plus
    
    def test_calculate_cns_mpo_optimized(self):
        """最適化されたCNS-MPO計算テスト"""
        cns_mpo_optimized = self.calculator.calculate_cns_mpo_optimized(self.test_smiles)
        
        assert isinstance(cns_mpo_optimized, dict)
        assert 'molecular_weight' in cns_mpo_optimized
        assert 'logp' in cns_mpo_optimized
        assert 'tpsa' in cns_mpo_optimized
        assert 'hbd' in cns_mpo_optimized
        assert 'hba' in cns_mpo_optimized
        assert 'rotatable_bonds' in cns_mpo_optimized
        assert 'cns_mpo_optimized_score' in cns_mpo_optimized
        assert 'cns_mpo_optimized_interpretation' in cns_mpo_optimized
        assert 'optimization_recommendations' in cns_mpo_optimized
    
    def test_process_molecule_batch(self):
        """分子バッチ処理テスト"""
        smiles_list = ["CCO", "CCCO", "CCCCCO"]
        df = self.calculator.process_molecule_batch(smiles_list)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'smiles' in df.columns
        assert not df.empty
    
    def test_get_cns_mpo_summary(self):
        """CNS-MPO要約取得テスト"""
        smiles_list = ["CCO", "CCCO", "CCCCCO"]
        df = self.calculator.process_molecule_batch(smiles_list)
        summary = self.calculator.get_cns_mpo_summary(df)
        
        assert isinstance(summary, dict)
        assert 'total_molecules' in summary
        assert 'cns_mpo_statistics' in summary
        assert 'interpretation_distribution' in summary
        assert 'optimization_recommendations' in summary

