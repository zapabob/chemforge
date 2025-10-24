"""
分子活性予測ウィジェット

CNSターゲット予測とADMET予測のStreamlitウィジェット
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import plotly.express as px
import plotly.graph_objects as go

from chemforge.admet.property_predictor import PropertyPredictor
from chemforge.data.molecular_features import MolecularFeatures
from chemforge.utils.logging_utils import Logger

logger = logging.getLogger(__name__)

class PredictionWidget:
    """
    分子活性予測ウィジェット
    
    CNSターゲット予測とADMET予測のStreamlitウィジェット
    """
    
    def __init__(self):
        """初期化"""
        self.property_predictor = PropertyPredictor()
        self.molecular_features = MolecularFeatures()
        logger.info("PredictionWidget initialized")
    
    def render_prediction_interface(self, smiles: str) -> Dict[str, Any]:
        """
        予測インターフェース表示
        
        Args:
            smiles: SMILES文字列
            
        Returns:
            予測結果辞書
        """
        st.subheader("🔬 分子活性予測")
        
        # 予測タイプ選択
        prediction_types = st.multiselect(
            "予測タイプを選択",
            ["CNSターゲット", "ADMET特性", "Tox21エンドポイント", "物理化学的性質"],
            default=["CNSターゲット", "ADMET特性"]
        )
        
        results = {}
        
        if "CNSターゲット" in prediction_types:
            results["cns_targets"] = self.predict_cns_targets(smiles)
        
        if "ADMET特性" in prediction_types:
            results["admet"] = self.predict_admet_properties(smiles)
        
        if "Tox21エンドポイント" in prediction_types:
            results["tox21"] = self.predict_tox21_endpoints(smiles)
        
        if "物理化学的性質" in prediction_types:
            results["physicochemical"] = self.predict_physicochemical_properties(smiles)
        
        return results
    
    def predict_cns_targets(self, smiles: str) -> Dict[str, float]:
        """CNSターゲット予測"""
        st.subheader("🧠 CNSターゲット活性予測")
        
        # 簡易的なCNSターゲット予測（実際の実装では学習済みモデルを使用）
        cns_targets = {
            "Dopamine D1": self._predict_dopamine_d1(smiles),
            "Dopamine D2": self._predict_dopamine_d2(smiles),
            "Serotonin 5-HT1A": self._predict_serotonin_5ht1a(smiles),
            "Serotonin 5-HT2A": self._predict_serotonin_5ht2a(smiles),
            "GABA-A": self._predict_gaba_a(smiles),
            "NMDA": self._predict_nmda(smiles),
            "Cannabinoid CB1": self._predict_cb1(smiles),
            "Opioid μ": self._predict_opioid_mu(smiles)
        }
        
        # 結果表示
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**ドパミン系**")
            st.metric("D1受容体", f"{cns_targets['Dopamine D1']:.3f}")
            st.metric("D2受容体", f"{cns_targets['Dopamine D2']:.3f}")
            
            st.write("**セロトニン系**")
            st.metric("5-HT1A", f"{cns_targets['Serotonin 5-HT1A']:.3f}")
            st.metric("5-HT2A", f"{cns_targets['Serotonin 5-HT2A']:.3f}")
        
        with col2:
            st.write("**その他**")
            st.metric("GABA-A", f"{cns_targets['GABA-A']:.3f}")
            st.metric("NMDA", f"{cns_targets['NMDA']:.3f}")
            st.metric("CB1", f"{cns_targets['Cannabinoid CB1']:.3f}")
            st.metric("μ-Opioid", f"{cns_targets['Opioid μ']:.3f}")
        
        # 可視化
        self._plot_cns_targets(cns_targets)
        
        return cns_targets
    
    def predict_admet_properties(self, smiles: str) -> Dict[str, float]:
        """ADMET特性予測"""
        st.subheader("💊 ADMET特性予測")
        
        # 物理化学的性質予測
        properties = self.property_predictor.predict_physicochemical_properties(smiles)
        
        # ADMET指標計算
        admet_properties = {
            "Absorption": self._calculate_absorption_score(properties),
            "Distribution": self._calculate_distribution_score(properties),
            "Metabolism": self._calculate_metabolism_score(properties),
            "Excretion": self._calculate_excretion_score(properties),
            "Toxicity": self._calculate_toxicity_score(properties)
        }
        
        # 結果表示
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("吸収 (Absorption)", f"{admet_properties['Absorption']:.3f}")
            st.metric("分布 (Distribution)", f"{admet_properties['Distribution']:.3f}")
        
        with col2:
            st.metric("代謝 (Metabolism)", f"{admet_properties['Metabolism']:.3f}")
            st.metric("排泄 (Excretion)", f"{admet_properties['Excretion']:.3f}")
        
        with col3:
            st.metric("毒性 (Toxicity)", f"{admet_properties['Toxicity']:.3f}")
        
        # 可視化
        self._plot_admet_radar(admet_properties)
        
        return admet_properties
    
    def predict_tox21_endpoints(self, smiles: str) -> Dict[str, float]:
        """Tox21エンドポイント予測"""
        st.subheader("🧪 Tox21エンドポイント予測")
        
        # Tox21エンドポイント予測
        tox21_results = self.property_predictor.predict_tox21_endpoints(smiles)
        
        # 結果表示
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**核内受容体 (NR)**")
            for endpoint in ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase']:
                if endpoint in tox21_results:
                    st.metric(endpoint, f"{tox21_results[endpoint]:.3f}")
        
        with col2:
            st.write("**ストレス応答 (SR)**")
            for endpoint in ['SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']:
                if endpoint in tox21_results:
                    st.metric(endpoint, f"{tox21_results[endpoint]:.3f}")
        
        # 可視化
        self._plot_tox21_heatmap(tox21_results)
        
        return tox21_results
    
    def predict_physicochemical_properties(self, smiles: str) -> Dict[str, float]:
        """物理化学的性質予測"""
        st.subheader("⚗️ 物理化学的性質")
        
        # 物理化学的性質予測
        properties = self.property_predictor.predict_physicochemical_properties(smiles)
        
        # 結果表示
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("分子量", f"{properties.get('molecular_weight', 0):.1f}")
            st.metric("LogP", f"{properties.get('logp', 0):.2f}")
            st.metric("TPSA", f"{properties.get('tpsa', 0):.1f}")
        
        with col2:
            st.metric("原子数", f"{properties.get('num_atoms', 0)}")
            st.metric("結合数", f"{properties.get('num_bonds', 0)}")
            st.metric("環数", f"{properties.get('num_rings', 0)}")
        
        with col3:
            st.metric("芳香族環数", f"{properties.get('num_aromatic_rings', 0)}")
            st.metric("ヘテロ原子数", f"{properties.get('num_heteroatoms', 0)}")
            st.metric("Csp3割合", f"{properties.get('fraction_csp3', 0):.2f}")
        
        return properties
    
    def _predict_dopamine_d1(self, smiles: str) -> float:
        """ドパミンD1受容体予測"""
        # 簡易的な予測（実際の実装では学習済みモデルを使用）
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0
            
            # フェネチルアミン骨格の簡易判定
            if "CC(CC1=CC=CC=C1)N" in smiles or "amphetamine" in smiles.lower():
                return 0.8
            return 0.2
        except:
            return 0.0
    
    def _predict_dopamine_d2(self, smiles: str) -> float:
        """ドパミンD2受容体予測"""
        return self._predict_dopamine_d1(smiles) * 0.9
    
    def _predict_serotonin_5ht1a(self, smiles: str) -> float:
        """セロトニン5-HT1A予測"""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0
            
            # トリプタミン骨格の簡易判定
            if "tryptamine" in smiles.lower() or "serotonin" in smiles.lower():
                return 0.7
            return 0.1
        except:
            return 0.0
    
    def _predict_serotonin_5ht2a(self, smiles: str) -> float:
        """セロトニン5-HT2A予測"""
        return self._predict_serotonin_5ht1a(smiles) * 0.8
    
    def _predict_gaba_a(self, smiles: str) -> float:
        """GABA-A予測"""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0
            
            # ベンゾジアゼピン骨格の簡易判定
            if "diazepam" in smiles.lower() or "benzodiazepine" in smiles.lower():
                return 0.9
            return 0.1
        except:
            return 0.0
    
    def _predict_nmda(self, smiles: str) -> float:
        """NMDA予測"""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0
            
            # ケタミン様構造の簡易判定
            if "ketamine" in smiles.lower() or "nmda" in smiles.lower():
                return 0.8
            return 0.1
        except:
            return 0.0
    
    def _predict_cb1(self, smiles: str) -> float:
        """カンナビノイドCB1予測"""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0
            
            # カンナビノイド骨格の簡易判定
            if "thc" in smiles.lower() or "cannabinoid" in smiles.lower():
                return 0.9
            return 0.1
        except:
            return 0.0
    
    def _predict_opioid_mu(self, smiles: str) -> float:
        """μ-オピオイド予測"""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0
            
            # オピオイド骨格の簡易判定
            if "morphine" in smiles.lower() or "opioid" in smiles.lower():
                return 0.9
            return 0.1
        except:
            return 0.0
    
    def _calculate_absorption_score(self, properties: Dict[str, float]) -> float:
        """吸収スコア計算"""
        mw = properties.get('molecular_weight', 0)
        logp = properties.get('logp', 0)
        tpsa = properties.get('tpsa', 0)
        
        # Lipinski's Rule of Five
        score = 0.0
        if mw <= 500: score += 0.25
        if logp <= 5: score += 0.25
        if tpsa <= 140: score += 0.25
        if properties.get('num_heteroatoms', 0) <= 10: score += 0.25
        
        return score
    
    def _calculate_distribution_score(self, properties: Dict[str, float]) -> float:
        """分布スコア計算"""
        logp = properties.get('logp', 0)
        mw = properties.get('molecular_weight', 0)
        
        # 適切なLogPと分子量
        score = 1.0
        if logp < 0 or logp > 5: score -= 0.3
        if mw < 200 or mw > 600: score -= 0.3
        
        return max(0.0, score)
    
    def _calculate_metabolism_score(self, properties: Dict[str, float]) -> float:
        """代謝スコア計算"""
        # 代謝しやすさの簡易判定
        mw = properties.get('molecular_weight', 0)
        logp = properties.get('logp', 0)
        
        score = 1.0
        if mw > 500: score -= 0.2
        if logp > 4: score -= 0.2
        
        return max(0.0, score)
    
    def _calculate_excretion_score(self, properties: Dict[str, float]) -> float:
        """排泄スコア計算"""
        mw = properties.get('molecular_weight', 0)
        tpsa = properties.get('tpsa', 0)
        
        score = 1.0
        if mw > 500: score -= 0.3
        if tpsa < 60: score -= 0.2
        
        return max(0.0, score)
    
    def _calculate_toxicity_score(self, properties: Dict[str, float]) -> float:
        """毒性スコア計算"""
        # 毒性の簡易判定
        mw = properties.get('molecular_weight', 0)
        logp = properties.get('logp', 0)
        
        score = 0.5  # デフォルト
        if mw > 600: score += 0.2
        if logp > 5: score += 0.2
        
        return min(1.0, score)
    
    def _plot_cns_targets(self, targets: Dict[str, float]):
        """CNSターゲット可視化"""
        fig = px.bar(
            x=list(targets.keys()),
            y=list(targets.values()),
            title="CNSターゲット活性",
            labels={'x': 'ターゲット', 'y': '活性スコア'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def _plot_admet_radar(self, admet: Dict[str, float]):
        """ADMETレーダーチャート"""
        categories = list(admet.keys())
        values = list(admet.values())
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='ADMET'
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="ADMET特性"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _plot_tox21_heatmap(self, tox21: Dict[str, float]):
        """Tox21ヒートマップ"""
        # NRとSRに分けて表示
        nr_data = {k: v for k, v in tox21.items() if k.startswith('NR-')}
        sr_data = {k: v for k, v in tox21.items() if k.startswith('SR-')}
        
        if nr_data:
            st.write("**核内受容体 (NR)**")
            fig = px.imshow(
                [list(nr_data.values())],
                x=list(nr_data.keys()),
                title="NRエンドポイント",
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if sr_data:
            st.write("**ストレス応答 (SR)**")
            fig = px.imshow(
                [list(sr_data.values())],
                x=list(sr_data.keys()),
                title="SRエンドポイント",
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig, use_container_width=True)
