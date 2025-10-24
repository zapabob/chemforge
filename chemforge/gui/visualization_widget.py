"""
分子可視化ウィジェット

2D/3D分子構造表示と特徴量可視化のStreamlitウィジェット
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Draw
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# py3Dmol imports
try:
    import py3Dmol
    PY3DMOL_AVAILABLE = True
except ImportError:
    PY3DMOL_AVAILABLE = False

from chemforge.data.molecular_features import MolecularFeatures
from chemforge.utils.logging_utils import Logger

logger = logging.getLogger(__name__)

class VisualizationWidget:
    """
    分子可視化ウィジェット
    
    2D/3D分子構造表示と特徴量可視化のStreamlitウィジェット
    """
    
    def __init__(self):
        """初期化"""
        self.molecular_features = MolecularFeatures()
        logger.info("VisualizationWidget initialized")
    
    def render_2d_structure(self, smiles: str):
        """2D分子構造表示"""
        st.subheader("🧬 2D分子構造")
        
        if not RDKIT_AVAILABLE:
            st.error("RDKitが利用できません")
            return
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                st.error("無効なSMILES文字列です")
                return
            
            # 2D構造画像生成
            img = Draw.MolToImage(mol, size=(400, 400))
            st.image(img, caption=f"SMILES: {smiles}")
            
            # 分子情報
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**基本情報**")
                st.write(f"原子数: {mol.GetNumAtoms()}")
                st.write(f"結合数: {mol.GetNumBonds()}")
                st.write(f"分子量: {Descriptors.MolWt(mol):.2f}")
            
            with col2:
                st.write("**環情報**")
                st.write(f"環数: {Descriptors.RingCount(mol)}")
                st.write(f"芳香族環数: {Descriptors.NumAromaticRings(mol)}")
                st.write(f"飽和環数: {Descriptors.NumSaturatedRings(mol)}")
            
        except Exception as e:
            st.error(f"2D構造表示エラー: {e}")
    
    def render_3d_structure(self, smiles: str):
        """3D分子構造表示"""
        st.subheader("🌐 3D分子構造")
        
        if not PY3DMOL_AVAILABLE:
            st.warning("py3Dmolが利用できません。2D表示に切り替えます。")
            self.render_2d_structure(smiles)
            return
        
        if not RDKIT_AVAILABLE:
            st.error("RDKitが利用できません")
            return
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                st.error("無効なSMILES文字列です")
                return
            
            # 3D座標生成
            from rdkit.Chem import rdDistGeom
            mol_3d = Chem.AddHs(mol)
            rdDistGeom.EmbedMolecule(mol_3d)
            
            # py3Dmol表示
            view = py3Dmol.view(width=600, height=400)
            view.addModel(Chem.MolToMolBlock(mol_3d), 'mol')
            view.setStyle({'stick': {}})
            view.setBackgroundColor('white')
            view.zoomTo()
            
            # HTMLとして表示
            view_html = view._make_html()
            st.components.v1.html(view_html, height=400)
            
        except Exception as e:
            st.error(f"3D構造表示エラー: {e}")
            # フォールバック: 2D表示
            self.render_2d_structure(smiles)
    
    def render_property_heatmap(self, smiles: str):
        """特徴量ヒートマップ表示"""
        st.subheader("📊 分子特徴量ヒートマップ")
        
        try:
            # 分子特徴量計算
            features = self.molecular_features.calculate_features(smiles)
            
            if not features:
                st.error("特徴量の計算に失敗しました")
                return
            
            # 特徴量をカテゴリ別に分類
            feature_categories = {
                "基本物性": ["molecular_weight", "logp", "tpsa", "num_atoms", "num_bonds"],
                "環構造": ["num_rings", "num_aromatic_rings", "num_saturated_rings", "num_aliphatic_rings"],
                "原子情報": ["num_heteroatoms", "num_carbons", "num_nitrogens", "num_oxygens"],
                "立体化学": ["num_stereocenters", "num_rotatable_bonds", "fraction_csp3"],
                "表面積": ["sasa", "atom_sasa_mean", "atom_sasa_std"],
                "分子形状": ["molecular_volume", "molecular_surface"]
            }
            
            # ヒートマップデータ準備
            heatmap_data = []
            category_labels = []
            
            for category, feature_names in feature_categories.items():
                category_values = []
                for feature_name in feature_names:
                    if feature_name in features:
                        category_values.append(features[feature_name])
                    else:
                        category_values.append(0.0)
                
                heatmap_data.append(category_values)
                category_labels.append(category)
            
            # ヒートマップ表示
            fig = px.imshow(
                heatmap_data,
                x=list(feature_categories.keys()),
                y=category_labels,
                title="分子特徴量ヒートマップ",
                color_continuous_scale="Viridis",
                aspect="auto"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # 詳細な特徴量表
            st.subheader("📋 詳細特徴量")
            feature_df = pd.DataFrame([
                {"特徴量": k, "値": v} for k, v in features.items()
            ])
            st.dataframe(feature_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"特徴量ヒートマップ表示エラー: {e}")
    
    def render_pca_plot(self, molecules: List[str]):
        """PCAプロット表示"""
        st.subheader("📈 分子PCAプロット")
        
        if len(molecules) < 2:
            st.warning("PCAプロットには2つ以上の分子が必要です")
            return
        
        try:
            # 全分子の特徴量計算
            all_features = []
            valid_molecules = []
            
            for smiles in molecules:
                features = self.molecular_features.calculate_features(smiles)
                if features:
                    all_features.append(features)
                    valid_molecules.append(smiles)
            
            if len(all_features) < 2:
                st.error("有効な特徴量が不足しています")
                return
            
            # 特徴量をDataFrameに変換
            feature_df = pd.DataFrame(all_features)
            
            # 欠損値を0で埋める
            feature_df = feature_df.fillna(0)
            
            # 標準化
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_df)
            
            # PCA実行
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_features)
            
            # プロット作成
            fig = px.scatter(
                x=pca_result[:, 0],
                y=pca_result[:, 1],
                text=valid_molecules,
                title="分子PCAプロット",
                labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
                       'y': f'PC2 ({pca.explained_variance_ratio_[1]:.2%})'}
            )
            fig.update_traces(textposition="top center")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # 寄与率表示
            st.write(f"**寄与率**: PC1: {pca.explained_variance_ratio_[0]:.2%}, PC2: {pca.explained_variance_ratio_[1]:.2%}")
            
        except Exception as e:
            st.error(f"PCAプロット表示エラー: {e}")
    
    def render_molecular_diversity(self, molecules: List[str]):
        """分子多様性表示"""
        st.subheader("🌈 分子多様性分析")
        
        if len(molecules) < 2:
            st.warning("多様性分析には2つ以上の分子が必要です")
            return
        
        try:
            # 分子多様性指標計算
            diversity_metrics = self._calculate_diversity_metrics(molecules)
            
            # 指標表示
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("平均分子量", f"{diversity_metrics['avg_mw']:.1f}")
                st.metric("平均LogP", f"{diversity_metrics['avg_logp']:.2f}")
            
            with col2:
                st.metric("分子量範囲", f"{diversity_metrics['mw_range']:.1f}")
                st.metric("LogP範囲", f"{diversity_metrics['logp_range']:.2f}")
            
            with col3:
                st.metric("多様性スコア", f"{diversity_metrics['diversity_score']:.3f}")
                st.metric("類似度", f"{diversity_metrics['similarity']:.3f}")
            
            # 散布図表示
            self._plot_diversity_scatter(molecules)
            
        except Exception as e:
            st.error(f"多様性分析エラー: {e}")
    
    def _calculate_diversity_metrics(self, molecules: List[str]) -> Dict[str, float]:
        """分子多様性指標計算"""
        try:
            features_list = []
            for smiles in molecules:
                features = self.molecular_features.calculate_features(smiles)
                if features:
                    features_list.append(features)
            
            if not features_list:
                return {}
            
            # 基本統計
            mw_values = [f.get('molecular_weight', 0) for f in features_list]
            logp_values = [f.get('logp', 0) for f in features_list]
            
            return {
                'avg_mw': np.mean(mw_values),
                'avg_logp': np.mean(logp_values),
                'mw_range': np.max(mw_values) - np.min(mw_values),
                'logp_range': np.max(logp_values) - np.min(logp_values),
                'diversity_score': np.std(mw_values) + np.std(logp_values),
                'similarity': 1.0 - (np.std(mw_values) / np.mean(mw_values))
            }
        except Exception as e:
            logger.error(f"多様性指標計算エラー: {e}")
            return {}
    
    def _plot_diversity_scatter(self, molecules: List[str]):
        """多様性散布図表示"""
        try:
            features_list = []
            valid_molecules = []
            
            for smiles in molecules:
                features = self.molecular_features.calculate_features(smiles)
                if features:
                    features_list.append(features)
                    valid_molecules.append(smiles)
            
            if len(features_list) < 2:
                return
            
            # 散布図データ準備
            mw_values = [f.get('molecular_weight', 0) for f in features_list]
            logp_values = [f.get('logp', 0) for f in features_list]
            
            fig = px.scatter(
                x=mw_values,
                y=logp_values,
                text=valid_molecules,
                title="分子多様性散布図 (分子量 vs LogP)",
                labels={'x': '分子量', 'y': 'LogP'}
            )
            fig.update_traces(textposition="top center")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"多様性散布図表示エラー: {e}")
    
    def render_molecular_comparison(self, molecules: List[str]):
        """分子比較表示"""
        st.subheader("⚖️ 分子比較")
        
        if len(molecules) < 2:
            st.warning("比較には2つ以上の分子が必要です")
            return
        
        try:
            # 比較データ準備
            comparison_data = []
            
            for i, smiles in enumerate(molecules):
                features = self.molecular_features.calculate_features(smiles)
                if features:
                    comparison_data.append({
                        '分子': f"分子{i+1}",
                        'SMILES': smiles,
                        '分子量': features.get('molecular_weight', 0),
                        'LogP': features.get('logp', 0),
                        'TPSA': features.get('tpsa', 0),
                        '環数': features.get('num_rings', 0),
                        '芳香族環数': features.get('num_aromatic_rings', 0)
                    })
            
            if not comparison_data:
                st.error("比較データの取得に失敗しました")
                return
            
            # 比較表表示
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # 比較グラフ
            numeric_columns = ['分子量', 'LogP', 'TPSA', '環数', '芳香族環数']
            fig = px.bar(
                comparison_df,
                x='分子',
                y=numeric_columns,
                title="分子比較グラフ",
                barmode='group'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"分子比較表示エラー: {e}")
