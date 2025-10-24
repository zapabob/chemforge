"""
Streamlitベースのメインウィンドウ

ChemForgeの統合GUIインターフェース
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import json
import time

# ChemForge imports
from chemforge.gui.streamlit_app import StreamlitApp
from chemforge.gui.prediction_widget import PredictionWidget
from chemforge.gui.visualization_widget import VisualizationWidget
from chemforge.gui.chat_widget import ChatWidget
from chemforge.data.cns_scaffolds import CNSScaffolds
from chemforge.generation.scaffold_generator import ScaffoldGenerator
from chemforge.admet.property_predictor import PropertyPredictor
from chemforge.utils.logging_utils import Logger

logger = logging.getLogger(__name__)

class MainWindow:
    """
    ChemForgeメインウィンドウ
    
    Streamlitベースの統合GUIインターフェース
    """
    
    def __init__(self):
        """初期化"""
        self.setup_page_config()
        self.initialize_session_state()
        self.app = StreamlitApp()
        self.prediction_widget = PredictionWidget()
        self.visualization_widget = VisualizationWidget()
        self.chat_widget = ChatWidget()
        self.scaffolds = CNSScaffolds()
        self.generator = ScaffoldGenerator(self.scaffolds)
        self.property_predictor = PropertyPredictor()
        
        logger.info("MainWindow initialized")
    
    def setup_page_config(self):
        """ページ設定"""
        st.set_page_config(
            page_title="ChemForge - CNS Drug Discovery Platform",
            page_icon="🧬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def initialize_session_state(self):
        """セッション状態初期化"""
        if 'molecules' not in st.session_state:
            st.session_state.molecules = []
        if 'selected_molecule' not in st.session_state:
            st.session_state.selected_molecule = None
        if 'generated_analogs' not in st.session_state:
            st.session_state.generated_analogs = []
        if 'prediction_results' not in st.session_state:
            st.session_state.prediction_results = {}
    
    def run(self):
        """メインアプリケーション実行"""
        # ヘッダー
        self.render_header()
        
        # サイドバー
        self.render_sidebar()
        
        # メインコンテンツ
        self.render_main_content()
        
        # フッター
        self.render_footer()
    
    def render_header(self):
        """ヘッダー表示"""
        st.title("🧬 ChemForge - CNS Drug Discovery Platform")
        st.markdown("**統合分子設計・予測・最適化プラットフォーム**")
        
        # タブ
        self.tabs = st.tabs([
            "🏠 ホーム",
            "🔬 分子予測",
            "📊 可視化",
            "💬 チャット",
            "🧪 骨格生成",
            "📈 ADMET分析"
        ])
    
    def render_sidebar(self):
        """サイドバー表示"""
        with st.sidebar:
            st.header("🔧 ナビゲーション")
            
            # 分子入力
            st.subheader("分子入力")
            smiles_input = st.text_input(
                "SMILES入力",
                placeholder="例: CC(CC1=CC=CC=C1)N",
                help="分子のSMILES文字列を入力してください"
            )
            
            if st.button("分子を追加", type="primary"):
                if smiles_input:
                    self.add_molecule(smiles_input)
                    st.success("分子が追加されました！")
                else:
                    st.error("SMILES文字列を入力してください")
            
            # 骨格選択
            st.subheader("骨格選択")
            scaffold_types = self.scaffolds.get_all_scaffold_types()
            selected_scaffold = st.selectbox(
                "骨格タイプ",
                scaffold_types,
                help="CNS作動薬の骨格を選択してください"
            )
            
            if st.button("骨格類似体生成"):
                self.generate_scaffold_analogs(selected_scaffold)
                st.success("骨格類似体を生成しました！")
            
            # 設定
            st.subheader("⚙️ 設定")
            st.slider("生成数", 1, 50, 10)
            st.checkbox("詳細ログ表示", value=False)
    
    def render_main_content(self):
        """メインコンテンツ表示"""
        # ホームタブ
        with self.tabs[0]:
            self.render_home_tab()
        
        # 分子予測タブ
        with self.tabs[1]:
            self.render_prediction_tab()
        
        # 可視化タブ
        with self.tabs[2]:
            self.render_visualization_tab()
        
        # チャットタブ
        with self.tabs[3]:
            self.render_chat_tab()
        
        # 骨格生成タブ
        with self.tabs[4]:
            self.render_scaffold_tab()
        
        # ADMET分析タブ
        with self.tabs[5]:
            self.render_admet_tab()
    
    def render_home_tab(self):
        """ホームタブ表示"""
        st.header("🏠 ホーム")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 統計情報")
            st.metric("登録分子数", len(st.session_state.molecules))
            st.metric("生成類似体数", len(st.session_state.generated_analogs))
            st.metric("予測実行回数", len(st.session_state.prediction_results))
        
        with col2:
            st.subheader("🔬 最近の活動")
            if st.session_state.molecules:
                st.write("最近追加された分子:")
                for i, mol in enumerate(st.session_state.molecules[-3:]):
                    st.write(f"{i+1}. {mol}")
            else:
                st.write("まだ分子が追加されていません")
        
        st.subheader("🚀 クイックスタート")
        st.markdown("""
        1. **分子入力**: サイドバーでSMILES文字列を入力
        2. **予測実行**: 分子予測タブでCNSターゲット予測
        3. **可視化**: 可視化タブで分子構造を表示
        4. **骨格生成**: 骨格生成タブで類似体を生成
        5. **ADMET分析**: ADMET分析タブで毒性予測
        """)
    
    def render_prediction_tab(self):
        """分子予測タブ表示"""
        st.header("🔬 分子予測")
        
        if st.session_state.molecules:
            selected_mol = st.selectbox(
                "予測する分子を選択",
                st.session_state.molecules
            )
            
            if st.button("予測実行", type="primary"):
                with st.spinner("予測を実行中..."):
                    results = self.run_predictions(selected_mol)
                    st.session_state.prediction_results[selected_mol] = results
                    st.success("予測が完了しました！")
            
            # 予測結果表示
            if selected_mol in st.session_state.prediction_results:
                self.display_prediction_results(selected_mol)
        else:
            st.warning("まず分子を追加してください")
    
    def render_visualization_tab(self):
        """可視化タブ表示"""
        st.header("📊 分子可視化")
        
        if st.session_state.molecules:
            selected_mol = st.selectbox(
                "可視化する分子を選択",
                st.session_state.molecules,
                key="viz_mol_select"
            )
            
            # 2D/3D表示切り替え
            viz_type = st.radio(
                "表示タイプ",
                ["2D構造", "3D構造", "特徴量ヒートマップ"]
            )
            
            if viz_type == "2D構造":
                self.visualization_widget.render_2d_structure(selected_mol)
            elif viz_type == "3D構造":
                self.visualization_widget.render_3d_structure(selected_mol)
            else:
                self.visualization_widget.render_property_heatmap(selected_mol)
        else:
            st.warning("まず分子を追加してください")
    
    def render_chat_tab(self):
        """チャットタブ表示"""
        st.header("💬 分子設計アシスタント")
        self.chat_widget.render_chat_interface()
    
    def render_scaffold_tab(self):
        """骨格生成タブ表示"""
        st.header("🧪 骨格ベース分子生成")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("骨格選択")
            scaffold_types = self.scaffolds.get_all_scaffold_types()
            selected_scaffold = st.selectbox(
                "骨格タイプ",
                scaffold_types,
                key="scaffold_select"
            )
            
            num_analogs = st.slider("生成数", 1, 50, 10)
            
            if st.button("類似体生成", type="primary"):
                with st.spinner("類似体を生成中..."):
                    analogs = self.generator.generate_analogs(
                        selected_scaffold, num_analogs
                    )
                    st.session_state.generated_analogs = analogs
                    st.success(f"{len(analogs)}個の類似体を生成しました！")
        
        with col2:
            st.subheader("生成結果")
            if st.session_state.generated_analogs:
                for i, analog in enumerate(st.session_state.generated_analogs[:5]):
                    with st.expander(f"類似体 {i+1} (スコア: {analog.score:.3f})"):
                        st.write(f"**SMILES**: {analog.smiles}")
                        st.write(f"**親化合物**: {analog.parent_compound}")
                        st.write(f"**修飾**: {', '.join(analog.modifications)}")
                        st.write(f"**物性**: {analog.properties}")
            else:
                st.info("類似体を生成してください")
    
    def render_admet_tab(self):
        """ADMET分析タブ表示"""
        st.header("📈 ADMET分析")
        
        if st.session_state.molecules:
            selected_mol = st.selectbox(
                "分析する分子を選択",
                st.session_state.molecules,
                key="admet_mol_select"
            )
            
            if st.button("ADMET分析実行", type="primary"):
                with st.spinner("ADMET分析を実行中..."):
                    # 物理化学的性質予測
                    phys_props = self.property_predictor.predict_physicochemical_properties(selected_mol)
                    
                    # Tox21エンドポイント予測
                    tox21_results = self.property_predictor.predict_tox21_endpoints(selected_mol)
                    
                    # 結果表示
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("物理化学的性質")
                        for prop, value in phys_props.items():
                            st.metric(prop, f"{value:.3f}")
                    
                    with col2:
                        st.subheader("Tox21エンドポイント")
                        for endpoint, score in tox21_results.items():
                            st.metric(endpoint, f"{score:.3f}")
        else:
            st.warning("まず分子を追加してください")
    
    def render_footer(self):
        """フッター表示"""
        st.markdown("---")
        st.markdown(
            "**ChemForge** - CNS Drug Discovery Platform | "
            "Powered by Streamlit & RDKit"
        )
    
    def add_molecule(self, smiles: str):
        """分子を追加"""
        st.session_state.molecules.append(smiles)
    
    def generate_scaffold_analogs(self, scaffold_type: str):
        """骨格類似体生成"""
        analogs = self.generator.generate_analogs(scaffold_type, 10)
        st.session_state.generated_analogs = analogs
    
    def run_predictions(self, smiles: str) -> Dict[str, Any]:
        """予測実行"""
        # 簡易的な予測（実際の実装ではより複雑な予測を行う）
        return {
            "CNS_targets": {"Dopamine": 0.8, "Serotonin": 0.6, "GABA": 0.4},
            "ADMET": {"LogP": 2.5, "MW": 300, "TPSA": 60},
            "Tox21": {"NR-AR": 0.3, "SR-ARE": 0.2}
        }
    
    def display_prediction_results(self, smiles: str):
        """予測結果表示"""
        results = st.session_state.prediction_results[smiles]
        
        st.subheader("予測結果")
        
        # CNSターゲット
        st.write("**CNSターゲット活性**")
        for target, score in results["CNS_targets"].items():
            st.metric(target, f"{score:.3f}")
        
        # ADMET
        st.write("**ADMET特性**")
        for prop, value in results["ADMET"].items():
            st.metric(prop, f"{value:.3f}")

def main():
    """メイン関数"""
    app = MainWindow()
    app.run()

if __name__ == "__main__":
    main()
