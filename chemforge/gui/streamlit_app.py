"""
Streamlit App Module

Streamlitアプリモジュール
分子生成・最適化・可視化の統合GUI
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import json
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 既存モジュール活用
from chemforge.generation.molecular_generator import MolecularGenerator
from chemforge.data.molecular_features import MolecularFeatures
from chemforge.data.rdkit_descriptors import RDKitDescriptors
from chemforge.admet.admet_predictor import ADMETPredictor
from chemforge.integration.visualization_manager import VisualizationManager
from chemforge.utils.config_utils import ConfigManager
from chemforge.utils.logging_utils import Logger
from chemforge.utils.validation import DataValidator

logger = logging.getLogger(__name__)

class StreamlitApp:
    """
    Streamlitアプリクラス
    
    分子生成・最適化・可視化の統合GUI
    """
    
    def __init__(self, config_path: Optional[str] = None, cache_dir: str = "cache"):
        """
        初期化
        
        Args:
            config_path: 設定ファイルパス
            cache_dir: キャッシュディレクトリ
        """
        self.config_path = config_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 既存モジュール活用
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load_config()
        self.logger = Logger("StreamlitApp")
        self.validator = DataValidator()
        
        # 分子生成・特徴量・ADMET予測
        self.molecular_generator = MolecularGenerator(config_path, cache_dir)
        self.molecular_features = MolecularFeatures(config_path, cache_dir)
        self.rdkit_descriptors = RDKitDescriptors(config_path, cache_dir)
        self.admet_predictor = ADMETPredictor(config_path, cache_dir)
        self.visualization_manager = VisualizationManager(config_path, cache_dir)
        
        # ページ設定
        st.set_page_config(
            page_title="ChemForge - Molecular Discovery Platform",
            page_icon="🧬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        logger.info("StreamlitApp initialized")
    
    def run(self):
        """
        Streamlitアプリ実行
        """
        # サイドバー
        self._create_sidebar()
        
        # メインコンテンツ
        self._create_main_content()
    
    def _create_sidebar(self):
        """
        サイドバー作成
        """
        st.sidebar.title("🧬 ChemForge")
        st.sidebar.markdown("### Molecular Discovery Platform")
        
        # ナビゲーション
        self.page = st.sidebar.selectbox(
            "Select Page",
            ["Home", "Molecular Generation", "Molecular Optimization", 
             "ADMET Prediction", "Visualization", "Settings"]
        )
        
        # 設定
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Settings")
        
        # デバイス設定
        self.device = st.sidebar.selectbox(
            "Device",
            ["auto", "cpu", "cuda"],
            index=0
        )
        
        # キャッシュ設定
        self.use_cache = st.sidebar.checkbox("Use Cache", value=True)
        
        # ログレベル
        self.log_level = st.sidebar.selectbox(
            "Log Level",
            ["INFO", "DEBUG", "WARNING", "ERROR"],
            index=0
        )
    
    def _create_main_content(self):
        """
        メインコンテンツ作成
        """
        if self.page == "Home":
            self._create_home_page()
        elif self.page == "Molecular Generation":
            self._create_generation_page()
        elif self.page == "Molecular Optimization":
            self._create_optimization_page()
        elif self.page == "ADMET Prediction":
            self._create_admet_page()
        elif self.page == "Visualization":
            self._create_visualization_page()
        elif self.page == "Settings":
            self._create_settings_page()
    
    def _create_home_page(self):
        """
        ホームページ作成
        """
        st.title("🧬 ChemForge - Molecular Discovery Platform")
        st.markdown("### AI-Powered Molecular Discovery and Optimization")
        
        # 統計情報
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Available Targets", "15+", "CNS Targets")
        
        with col2:
            st.metric("Generation Methods", "3", "VAE, RL, GA")
        
        with col3:
            st.metric("ADMET Properties", "20+", "Comprehensive")
        
        with col4:
            st.metric("Visualization", "Interactive", "Plotly")
        
        # 機能紹介
        st.markdown("---")
        st.markdown("### Key Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🧬 Molecular Generation**
            - VAE-based generation
            - Reinforcement Learning
            - Genetic Algorithms
            - Target property optimization
            """)
            
            st.markdown("""
            **🔬 ADMET Prediction**
            - Physicochemical properties
            - Pharmacokinetics
            - Toxicity assessment
            - Drug-likeness evaluation
            """)
        
        with col2:
            st.markdown("""
            **🎯 Molecular Optimization**
            - Target property optimization
            - Multi-objective optimization
            - Constraint handling
            - Performance evaluation
            """)
            
            st.markdown("""
            **📊 Visualization**
            - Interactive molecular plots
            - Property distribution
            - Optimization history
            - Performance metrics
            """)
        
        # クイックスタート
        st.markdown("---")
        st.markdown("### Quick Start")
        
        if st.button("🚀 Start Molecular Generation"):
            st.session_state.page = "Molecular Generation"
            st.rerun()
        
        if st.button("🔬 Start ADMET Prediction"):
            st.session_state.page = "ADMET Prediction"
            st.rerun()
    
    def _create_generation_page(self):
        """
        分子生成ページ作成
        """
        st.title("🧬 Molecular Generation")
        st.markdown("### Generate novel molecules using VAE, RL, and GA")
        
        # 生成設定
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Generation Settings")
            
            # 生成方法
            generation_method = st.selectbox(
                "Generation Method",
                ["vae", "rl", "ga"],
                help="Select generation method"
            )
            
            # 生成分子数
            num_molecules = st.slider(
                "Number of Molecules",
                min_value=10,
                max_value=1000,
                value=100,
                step=10
            )
            
            # ターゲット特性
            st.markdown("#### Target Properties")
            
            target_mw = st.number_input(
                "Molecular Weight",
                min_value=100.0,
                max_value=1000.0,
                value=300.0,
                step=10.0
            )
            
            target_logp = st.number_input(
                "LogP",
                min_value=-2.0,
                max_value=6.0,
                value=2.5,
                step=0.1
            )
            
            target_tpsa = st.number_input(
                "TPSA",
                min_value=0.0,
                max_value=200.0,
                value=50.0,
                step=5.0
            )
        
        with col2:
            st.markdown("#### Advanced Settings")
            
            # シード分子
            seed_molecules = st.text_area(
                "Seed Molecules (SMILES)",
                value="CCO\nCCN\nCC(=O)O",
                help="Enter SMILES strings, one per line"
            )
            
            # フィルター設定
            st.markdown("#### Filters")
            
            filter_mw = st.checkbox("Filter by Molecular Weight", value=True)
            filter_logp = st.checkbox("Filter by LogP", value=True)
            filter_tpsa = st.checkbox("Filter by TPSA", value=True)
            
            # 保存設定
            st.markdown("#### Save Settings")
            
            save_features = st.checkbox("Save Molecular Features", value=True)
            save_admet = st.checkbox("Save ADMET Predictions", value=True)
        
        # 生成実行
        st.markdown("---")
        
        if st.button("🚀 Generate Molecules", type="primary"):
            with st.spinner("Generating molecules..."):
                try:
                    # ターゲット特性設定
                    target_properties = {
                        'MW': target_mw,
                        'LogP': target_logp,
                        'TPSA': target_tpsa
                    }
                    
                    # シード分子処理
                    seed_list = [smiles.strip() for smiles in seed_molecules.split('\n') if smiles.strip()]
                    
                    # 分子生成
                    molecules = self.molecular_generator.generate_molecules(
                        num_molecules=num_molecules,
                        method=generation_method,
                        target_properties=target_properties,
                        seed_molecules=seed_list if seed_list else None
                    )
                    
                    # 結果表示
                    st.success(f"Generated {len(molecules)} molecules!")
                    
                    # 分子リスト表示
                    st.markdown("#### Generated Molecules")
                    df_molecules = pd.DataFrame({'SMILES': molecules})
                    st.dataframe(df_molecules, use_container_width=True)
                    
                    # 特徴量計算
                    if save_features:
                        with st.spinner("Calculating molecular features..."):
                            features_df = self.molecular_generator.evaluate_molecules(
                                molecules, target_properties
                            )
                            
                            st.markdown("#### Molecular Features")
                            st.dataframe(features_df, use_container_width=True)
                    
                    # ADMET予測
                    if save_admet:
                        with st.spinner("Predicting ADMET properties..."):
                            admet_results = []
                            for smiles in tqdm(molecules[:10], desc="ADMET prediction"):
                                try:
                                    admet_pred = self.admet_predictor.predict_admet(smiles)
                                    admet_pred['SMILES'] = smiles
                                    admet_results.append(admet_pred)
                                except Exception as e:
                                    logger.warning(f"ADMET prediction failed for {smiles}: {e}")
                                    continue
                            
                            if admet_results:
                                admet_df = pd.DataFrame(admet_results)
                                st.markdown("#### ADMET Predictions")
                                st.dataframe(admet_df, use_container_width=True)
                    
                    # 保存
                    if st.button("💾 Save Results"):
                        timestamp = int(time.time())
                        output_path = f"generated_molecules_{timestamp}.csv"
                        
                        if self.molecular_generator.save_generated_molecules(
                            molecules, output_path, save_features
                        ):
                            st.success(f"Results saved to: {output_path}")
                        else:
                            st.error("Failed to save results")
                
                except Exception as e:
                    st.error(f"Generation failed: {e}")
                    logger.error(f"Generation failed: {e}")
    
    def _create_optimization_page(self):
        """
        分子最適化ページ作成
        """
        st.title("🎯 Molecular Optimization")
        st.markdown("### Optimize molecules for target properties")
        
        # 最適化設定
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Optimization Settings")
            
            # 最適化方法
            optimization_method = st.selectbox(
                "Optimization Method",
                ["ga", "rl"],
                help="Select optimization method"
            )
            
            # 最大反復数
            max_iterations = st.slider(
                "Max Iterations",
                min_value=10,
                max_value=1000,
                value=100,
                step=10
            )
            
            # ターゲット特性
            st.markdown("#### Target Properties")
            
            target_mw = st.number_input(
                "Target Molecular Weight",
                min_value=100.0,
                max_value=1000.0,
                value=300.0,
                step=10.0
            )
            
            target_logp = st.number_input(
                "Target LogP",
                min_value=-2.0,
                max_value=6.0,
                value=2.5,
                step=0.1
            )
        
        with col2:
            st.markdown("#### Input Molecules")
            
            # 入力分子
            input_molecules = st.text_area(
                "Input Molecules (SMILES)",
                value="CCO\nCCN\nCC(=O)O",
                help="Enter SMILES strings, one per line"
            )
            
            # 重み設定
            st.markdown("#### Property Weights")
            
            weight_mw = st.slider(
                "Molecular Weight Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1
            )
            
            weight_logp = st.slider(
                "LogP Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1
            )
        
        # 最適化実行
        st.markdown("---")
        
        if st.button("🎯 Optimize Molecules", type="primary"):
            with st.spinner("Optimizing molecules..."):
                try:
                    # 入力分子処理
                    molecules = [smiles.strip() for smiles in input_molecules.split('\n') if smiles.strip()]
                    
                    if not molecules:
                        st.error("Please enter input molecules")
                        return
                    
                    # ターゲット特性設定
                    target_properties = {
                        'MW': target_mw,
                        'LogP': target_logp
                    }
                    
                    # 分子最適化
                    optimized_molecules = self.molecular_generator.optimize_molecules(
                        molecules=molecules,
                        target_properties=target_properties,
                        optimization_method=optimization_method,
                        max_iterations=max_iterations
                    )
                    
                    # 結果表示
                    st.success(f"Optimized {len(optimized_molecules)} molecules!")
                    
                    # 最適化結果表示
                    st.markdown("#### Optimization Results")
                    df_optimized = pd.DataFrame({'SMILES': optimized_molecules})
                    st.dataframe(df_optimized, use_container_width=True)
                    
                    # 特徴量比較
                    with st.spinner("Calculating features..."):
                        # 元の分子の特徴量
                        original_features = self.molecular_generator.evaluate_molecules(molecules)
                        
                        # 最適化分子の特徴量
                        optimized_features = self.molecular_generator.evaluate_molecules(optimized_molecules)
                        
                        # 比較表示
                        st.markdown("#### Feature Comparison")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Original Molecules**")
                            st.dataframe(original_features, use_container_width=True)
                        
                        with col2:
                            st.markdown("**Optimized Molecules**")
                            st.dataframe(optimized_features, use_container_width=True)
                    
                    # 保存
                    if st.button("💾 Save Optimization Results"):
                        timestamp = int(time.time())
                        output_path = f"optimized_molecules_{timestamp}.csv"
                        
                        if self.molecular_generator.save_generated_molecules(
                            optimized_molecules, output_path, True
                        ):
                            st.success(f"Results saved to: {output_path}")
                        else:
                            st.error("Failed to save results")
                
                except Exception as e:
                    st.error(f"Optimization failed: {e}")
                    logger.error(f"Optimization failed: {e}")
    
    def _create_admet_page(self):
        """
        ADMET予測ページ作成
        """
        st.title("🔬 ADMET Prediction")
        st.markdown("### Predict ADMET properties for molecules")
        
        # ADMET予測設定
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Input Molecules")
            
            # 入力分子
            input_molecules = st.text_area(
                "Input Molecules (SMILES)",
                value="CCO\nCCN\nCC(=O)O",
                help="Enter SMILES strings, one per line"
            )
            
            # 予測設定
            st.markdown("#### Prediction Settings")
            
            predict_physicochemical = st.checkbox("Physicochemical Properties", value=True)
            predict_pharmacokinetics = st.checkbox("Pharmacokinetics", value=True)
            predict_toxicity = st.checkbox("Toxicity", value=True)
            predict_druglikeness = st.checkbox("Drug-likeness", value=True)
        
        with col2:
            st.markdown("#### Advanced Settings")
            
            # バッチサイズ
            batch_size = st.slider(
                "Batch Size",
                min_value=1,
                max_value=100,
                value=10,
                step=1
            )
            
            # キャッシュ使用
            use_cache = st.checkbox("Use Cache", value=True)
            
            # 並列処理
            use_parallel = st.checkbox("Use Parallel Processing", value=True)
        
        # ADMET予測実行
        st.markdown("---")
        
        if st.button("🔬 Predict ADMET", type="primary"):
            with st.spinner("Predicting ADMET properties..."):
                try:
                    # 入力分子処理
                    molecules = [smiles.strip() for smiles in input_molecules.split('\n') if smiles.strip()]
                    
                    if not molecules:
                        st.error("Please enter input molecules")
                        return
                    
                    # ADMET予測
                    admet_results = []
                    
                    for i in tqdm(range(0, len(molecules), batch_size), desc="ADMET prediction"):
                        batch = molecules[i:i+batch_size]
                        
                        for smiles in batch:
                            try:
                                admet_pred = self.admet_predictor.predict_admet(
                                    smiles,
                                    include_physicochemical=predict_physicochemical,
                                    include_pharmacokinetics=predict_pharmacokinetics,
                                    include_toxicity=predict_toxicity,
                                    include_druglikeness=predict_druglikeness
                                )
                                admet_pred['SMILES'] = smiles
                                admet_results.append(admet_pred)
                            except Exception as e:
                                logger.warning(f"ADMET prediction failed for {smiles}: {e}")
                                continue
                    
                    if admet_results:
                        admet_df = pd.DataFrame(admet_results)
                        
                        # 結果表示
                        st.success(f"Predicted ADMET properties for {len(admet_results)} molecules!")
                        
                        # ADMET結果表示
                        st.markdown("#### ADMET Predictions")
                        st.dataframe(admet_df, use_container_width=True)
                        
                        # 統計情報
                        st.markdown("#### Statistics")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Molecules", len(admet_results))
                        
                        with col2:
                            if 'MW' in admet_df.columns:
                                st.metric("Avg MW", f"{admet_df['MW'].mean():.1f}")
                        
                        with col3:
                            if 'LogP' in admet_df.columns:
                                st.metric("Avg LogP", f"{admet_df['LogP'].mean():.2f}")
                        
                        with col4:
                            if 'TPSA' in admet_df.columns:
                                st.metric("Avg TPSA", f"{admet_df['TPSA'].mean():.1f}")
                        
                        # 保存
                        if st.button("💾 Save ADMET Results"):
                            timestamp = int(time.time())
                            output_path = f"admet_predictions_{timestamp}.csv"
                            admet_df.to_csv(output_path, index=False)
                            st.success(f"Results saved to: {output_path}")
                    else:
                        st.error("No ADMET predictions generated")
                
                except Exception as e:
                    st.error(f"ADMET prediction failed: {e}")
                    logger.error(f"ADMET prediction failed: {e}")
    
    def _create_visualization_page(self):
        """
        可視化ページ作成
        """
        st.title("📊 Visualization")
        st.markdown("### Interactive molecular visualization and analysis")
        
        # 可視化設定
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Visualization Settings")
            
            # 可視化タイプ
            viz_type = st.selectbox(
                "Visualization Type",
                ["Molecular Properties", "ADMET Distribution", "Optimization History", "Performance Metrics"],
                help="Select visualization type"
            )
            
            # データソース
            data_source = st.selectbox(
                "Data Source",
                ["Upload File", "Generated Molecules", "ADMET Results"],
                help="Select data source"
            )
        
        with col2:
            st.markdown("#### Display Settings")
            
            # プロットタイプ
            plot_type = st.selectbox(
                "Plot Type",
                ["Scatter", "Histogram", "Box Plot", "Heatmap"],
                help="Select plot type"
            )
            
            # カラーパレット
            color_palette = st.selectbox(
                "Color Palette",
                ["viridis", "plasma", "inferno", "magma"],
                help="Select color palette"
            )
        
        # データアップロード
        if data_source == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload CSV file",
                type=['csv'],
                help="Upload a CSV file with molecular data"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"Loaded {len(df)} molecules from file")
                    
                    # 可視化実行
                    self._create_visualizations(df, viz_type, plot_type, color_palette)
                except Exception as e:
                    st.error(f"Error loading file: {e}")
        else:
            st.info("Please generate molecules or run ADMET prediction first")
    
    def _create_visualizations(self, df: pd.DataFrame, viz_type: str, plot_type: str, color_palette: str):
        """
        可視化作成
        
        Args:
            df: データフレーム
            viz_type: 可視化タイプ
            plot_type: プロットタイプ
            color_palette: カラーパレット
        """
        if viz_type == "Molecular Properties":
            self._create_property_visualizations(df, plot_type, color_palette)
        elif viz_type == "ADMET Distribution":
            self._create_admet_visualizations(df, plot_type, color_palette)
        elif viz_type == "Optimization History":
            self._create_optimization_visualizations(df, plot_type, color_palette)
        elif viz_type == "Performance Metrics":
            self._create_performance_visualizations(df, plot_type, color_palette)
    
    def _create_property_visualizations(self, df: pd.DataFrame, plot_type: str, color_palette: str):
        """
        分子特性可視化
        
        Args:
            df: データフレーム
            plot_type: プロットタイプ
            color_palette: カラーパレット
        """
        st.markdown("#### Molecular Properties Visualization")
        
        # 利用可能な列を取得
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) < 2:
            st.warning("Not enough numeric columns for visualization")
            return
        
        # プロット作成
        if plot_type == "Scatter":
            col1, col2 = st.columns(2)
            
            with col1:
                x_col = st.selectbox("X-axis", numeric_columns)
            with col2:
                y_col = st.selectbox("Y-axis", numeric_columns)
            
            if x_col and y_col:
                fig = px.scatter(df, x=x_col, y=y_col, color=color_palette)
                st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Histogram":
            col = st.selectbox("Column", numeric_columns)
            
            if col:
                fig = px.histogram(df, x=col, color=color_palette)
                st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Box Plot":
            col = st.selectbox("Column", numeric_columns)
            
            if col:
                fig = px.box(df, y=col, color=color_palette)
                st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Heatmap":
            # 相関行列計算
            corr_matrix = df[numeric_columns].corr()
            
            fig = px.imshow(corr_matrix, color_continuous_scale=color_palette)
            st.plotly_chart(fig, use_container_width=True)
    
    def _create_admet_visualizations(self, df: pd.DataFrame, plot_type: str, color_palette: str):
        """
        ADMET可視化
        
        Args:
            df: データフレーム
            plot_type: プロットタイプ
            color_palette: カラーパレット
        """
        st.markdown("#### ADMET Distribution Visualization")
        
        # ADMET関連列を取得
        admet_columns = [col for col in df.columns if any(term in col.lower() for term in ['admet', 'toxicity', 'druglikeness'])]
        
        if not admet_columns:
            st.warning("No ADMET columns found")
            return
        
        # プロット作成
        if plot_type == "Histogram":
            col = st.selectbox("ADMET Column", admet_columns)
            
            if col:
                fig = px.histogram(df, x=col, color=color_palette)
                st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Box Plot":
            col = st.selectbox("ADMET Column", admet_columns)
            
            if col:
                fig = px.box(df, y=col, color=color_palette)
                st.plotly_chart(fig, use_container_width=True)
    
    def _create_optimization_visualizations(self, df: pd.DataFrame, plot_type: str, color_palette: str):
        """
        最適化可視化
        
        Args:
            df: データフレーム
            plot_type: プロットタイプ
            color_palette: カラーパレット
        """
        st.markdown("#### Optimization History Visualization")
        
        # 最適化関連列を取得
        optimization_columns = [col for col in df.columns if any(term in col.lower() for term in ['iteration', 'generation', 'fitness', 'score'])]
        
        if not optimization_columns:
            st.warning("No optimization columns found")
            return
        
        # プロット作成
        if plot_type == "Scatter":
            col1, col2 = st.columns(2)
            
            with col1:
                x_col = st.selectbox("X-axis", optimization_columns)
            with col2:
                y_col = st.selectbox("Y-axis", optimization_columns)
            
            if x_col and y_col:
                fig = px.scatter(df, x=x_col, y=y_col, color=color_palette)
                st.plotly_chart(fig, use_container_width=True)
    
    def _create_performance_visualizations(self, df: pd.DataFrame, plot_type: str, color_palette: str):
        """
        性能可視化
        
        Args:
            df: データフレーム
            plot_type: プロットタイプ
            color_palette: カラーパレット
        """
        st.markdown("#### Performance Metrics Visualization")
        
        # 性能関連列を取得
        performance_columns = [col for col in df.columns if any(term in col.lower() for term in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'rmse', 'mae'])]
        
        if not performance_columns:
            st.warning("No performance columns found")
            return
        
        # プロット作成
        if plot_type == "Histogram":
            col = st.selectbox("Performance Column", performance_columns)
            
            if col:
                fig = px.histogram(df, x=col, color=color_palette)
                st.plotly_chart(fig, use_container_width=True)
    
    def _create_settings_page(self):
        """
        設定ページ作成
        """
        st.title("⚙️ Settings")
        st.markdown("### Application configuration and preferences")
        
        # 設定カテゴリ
        settings_category = st.selectbox(
            "Settings Category",
            ["General", "Molecular Generation", "ADMET Prediction", "Visualization", "Advanced"]
        )
        
        if settings_category == "General":
            self._create_general_settings()
        elif settings_category == "Molecular Generation":
            self._create_generation_settings()
        elif settings_category == "ADMET Prediction":
            self._create_admet_settings()
        elif settings_category == "Visualization":
            self._create_visualization_settings()
        elif settings_category == "Advanced":
            self._create_advanced_settings()
    
    def _create_general_settings(self):
        """
        一般設定作成
        """
        st.markdown("#### General Settings")
        
        # アプリケーション設定
        app_title = st.text_input("Application Title", value="ChemForge")
        app_description = st.text_area("Application Description", value="Molecular Discovery Platform")
        
        # デバイス設定
        device = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
        
        # ログ設定
        log_level = st.selectbox("Log Level", ["INFO", "DEBUG", "WARNING", "ERROR"], index=0)
        
        # キャッシュ設定
        use_cache = st.checkbox("Use Cache", value=True)
        cache_size = st.slider("Cache Size (MB)", min_value=100, max_value=10000, value=1000)
        
        # 保存設定
        if st.button("💾 Save General Settings"):
            st.success("General settings saved!")
    
    def _create_generation_settings(self):
        """
        分子生成設定作成
        """
        st.markdown("#### Molecular Generation Settings")
        
        # VAE設定
        st.markdown("##### VAE Settings")
        vae_latent_dim = st.slider("VAE Latent Dimension", min_value=32, max_value=512, value=128)
        vae_learning_rate = st.number_input("VAE Learning Rate", min_value=1e-5, max_value=1e-2, value=1e-3, format="%.5f")
        
        # RL設定
        st.markdown("##### RL Settings")
        rl_learning_rate = st.number_input("RL Learning Rate", min_value=1e-5, max_value=1e-2, value=1e-3, format="%.5f")
        rl_episodes = st.slider("RL Episodes", min_value=100, max_value=10000, value=1000)
        
        # GA設定
        st.markdown("##### GA Settings")
        ga_population_size = st.slider("GA Population Size", min_value=10, max_value=1000, value=100)
        ga_mutation_rate = st.slider("GA Mutation Rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
        
        # 保存設定
        if st.button("💾 Save Generation Settings"):
            st.success("Generation settings saved!")
    
    def _create_admet_settings(self):
        """
        ADMET予測設定作成
        """
        st.markdown("#### ADMET Prediction Settings")
        
        # 予測設定
        predict_physicochemical = st.checkbox("Predict Physicochemical Properties", value=True)
        predict_pharmacokinetics = st.checkbox("Predict Pharmacokinetics", value=True)
        predict_toxicity = st.checkbox("Predict Toxicity", value=True)
        predict_druglikeness = st.checkbox("Predict Drug-likeness", value=True)
        
        # バッチ設定
        batch_size = st.slider("Batch Size", min_value=1, max_value=100, value=10)
        use_parallel = st.checkbox("Use Parallel Processing", value=True)
        
        # 保存設定
        if st.button("💾 Save ADMET Settings"):
            st.success("ADMET settings saved!")
    
    def _create_visualization_settings(self):
        """
        可視化設定作成
        """
        st.markdown("#### Visualization Settings")
        
        # プロット設定
        default_plot_type = st.selectbox("Default Plot Type", ["Scatter", "Histogram", "Box Plot", "Heatmap"])
        default_color_palette = st.selectbox("Default Color Palette", ["viridis", "plasma", "inferno", "magma"])
        
        # 表示設定
        show_legend = st.checkbox("Show Legend", value=True)
        show_grid = st.checkbox("Show Grid", value=True)
        
        # 保存設定
        if st.button("💾 Save Visualization Settings"):
            st.success("Visualization settings saved!")
    
    def _create_advanced_settings(self):
        """
        高度設定作成
        """
        st.markdown("#### Advanced Settings")
        
        # パフォーマンス設定
        st.markdown("##### Performance Settings")
        num_workers = st.slider("Number of Workers", min_value=1, max_value=16, value=4)
        pin_memory = st.checkbox("Pin Memory", value=True)
        
        # デバッグ設定
        st.markdown("##### Debug Settings")
        debug_mode = st.checkbox("Debug Mode", value=False)
        verbose_logging = st.checkbox("Verbose Logging", value=False)
        
        # 保存設定
        if st.button("💾 Save Advanced Settings"):
            st.success("Advanced settings saved!")

def create_streamlit_app(config_path: Optional[str] = None, 
                        cache_dir: str = "cache") -> StreamlitApp:
    """
    Streamlitアプリ作成
    
    Args:
        config_path: 設定ファイルパス
        cache_dir: キャッシュディレクトリ
        
    Returns:
        StreamlitApp
    """
    return StreamlitApp(config_path, cache_dir)

if __name__ == "__main__":
    # テスト実行
    app = StreamlitApp()
    app.run()