"""
チャットインターフェースウィジェット

分子設計アシスタントのStreamlitウィジェット
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import json
import time
from datetime import datetime

from chemforge.data.cns_scaffolds import CNSScaffolds, CNSCompound
from chemforge.generation.scaffold_generator import ScaffoldGenerator
from chemforge.admet.property_predictor import PropertyPredictor
from chemforge.utils.logging_utils import Logger

logger = logging.getLogger(__name__)

class ChatWidget:
    """
    チャットインターフェースウィジェット
    
    分子設計アシスタントのStreamlitウィジェット
    """
    
    def __init__(self):
        """初期化"""
        self.scaffolds = CNSScaffolds()
        self.generator = ScaffoldGenerator(self.scaffolds)
        self.property_predictor = PropertyPredictor()
        self.initialize_chat_history()
        logger.info("ChatWidget initialized")
    
    def initialize_chat_history(self):
        """チャット履歴初期化"""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'current_conversation' not in st.session_state:
            st.session_state.current_conversation = []
    
    def render_chat_interface(self):
        """チャットインターフェース表示"""
        st.subheader("💬 分子設計アシスタント")
        
        # チャット履歴表示
        self.render_chat_history()
        
        # 入力フォーム
        self.render_input_form()
        
        # サイドバー機能
        with st.sidebar:
            self.render_sidebar_features()
    
    def render_chat_history(self):
        """チャット履歴表示"""
        if st.session_state.chat_history:
            st.subheader("📝 チャット履歴")
            
            for i, message in enumerate(st.session_state.chat_history):
                with st.expander(f"会話 {i+1} - {message['timestamp']}"):
                    st.write(f"**ユーザー**: {message['user_input']}")
                    st.write(f"**アシスタント**: {message['assistant_response']}")
                    
                    if 'suggestions' in message:
                        st.write("**提案**:")
                        for suggestion in message['suggestions']:
                            st.write(f"- {suggestion}")
        
        # 現在の会話
        if st.session_state.current_conversation:
            st.subheader("💭 現在の会話")
            for message in st.session_state.current_conversation:
                if message['role'] == 'user':
                    st.write(f"**あなた**: {message['content']}")
                else:
                    st.write(f"**アシスタント**: {message['content']}")
    
    def render_input_form(self):
        """入力フォーム表示"""
        st.subheader("💭 メッセージ入力")
        
        # 入力タイプ選択
        input_type = st.radio(
            "入力タイプ",
            ["テキスト", "SMILES", "骨格検索", "分子生成"]
        )
        
        if input_type == "テキスト":
            user_input = st.text_area(
                "メッセージを入力してください",
                placeholder="例: ドパミン受容体に結合する分子を設計したい",
                height=100
            )
        elif input_type == "SMILES":
            user_input = st.text_input(
                "SMILES文字列を入力してください",
                placeholder="例: CC(CC1=CC=CC=C1)N"
            )
        elif input_type == "骨格検索":
            scaffold_types = self.scaffolds.get_all_scaffold_types()
            selected_scaffold = st.selectbox("骨格タイプを選択", scaffold_types)
            user_input = f"骨格検索: {selected_scaffold}"
        else:  # 分子生成
            scaffold_types = self.scaffolds.get_all_scaffold_types()
            selected_scaffold = st.selectbox("生成する骨格タイプ", scaffold_types)
            num_analogs = st.slider("生成数", 1, 20, 5)
            user_input = f"分子生成: {selected_scaffold} x{num_analogs}"
        
        # 送信ボタン
        if st.button("送信", type="primary"):
            if user_input:
                self.process_user_input(user_input, input_type)
            else:
                st.warning("メッセージを入力してください")
    
    def render_sidebar_features(self):
        """サイドバー機能表示"""
        st.subheader("🔧 便利機能")
        
        # 骨格情報
        if st.button("骨格情報表示"):
            self.show_scaffold_info()
        
        # 分子変換
        if st.button("SMILES変換"):
            self.show_smiles_conversion()
        
        # 履歴クリア
        if st.button("履歴クリア"):
            st.session_state.chat_history = []
            st.session_state.current_conversation = []
            st.success("履歴をクリアしました")
        
        # エクスポート
        if st.button("会話エクスポート"):
            self.export_conversation()
    
    def process_user_input(self, user_input: str, input_type: str):
        """ユーザー入力処理"""
        try:
            # 現在の会話に追加
            st.session_state.current_conversation.append({
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
            
            # アシスタント応答生成
            response = self.generate_assistant_response(user_input, input_type)
            
            # 現在の会話に追加
            st.session_state.current_conversation.append({
                'role': 'assistant',
                'content': response['response'],
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
            
            # 提案があれば表示
            if 'suggestions' in response:
                st.write("**💡 提案**:")
                for suggestion in response['suggestions']:
                    st.write(f"- {suggestion}")
            
            # 結果があれば表示
            if 'results' in response:
                self.display_results(response['results'])
            
        except Exception as e:
            st.error(f"入力処理エラー: {e}")
            logger.error(f"Input processing error: {e}")
    
    def generate_assistant_response(self, user_input: str, input_type: str) -> Dict[str, Any]:
        """アシスタント応答生成"""
        response = {
            'response': '',
            'suggestions': [],
            'results': {}
        }
        
        try:
            if input_type == "テキスト":
                response = self._handle_text_input(user_input)
            elif input_type == "SMILES":
                response = self._handle_smiles_input(user_input)
            elif input_type == "骨格検索":
                response = self._handle_scaffold_search(user_input)
            elif input_type == "分子生成":
                response = self._handle_molecule_generation(user_input)
            
        except Exception as e:
            response['response'] = f"申し訳ございません。エラーが発生しました: {e}"
            logger.error(f"Response generation error: {e}")
        
        return response
    
    def _handle_text_input(self, user_input: str) -> Dict[str, Any]:
        """テキスト入力処理"""
        response = {
            'response': '',
            'suggestions': [],
            'results': {}
        }
        
        # キーワードベースの応答
        if "ドパミン" in user_input or "dopamine" in user_input.lower():
            response['response'] = "ドパミン受容体に結合する分子についてお答えします。"
            response['suggestions'] = [
                "フェネチルアミン骨格の分子を検索",
                "ドパミンD1/D2受容体予測を実行",
                "類似体を生成"
            ]
        elif "セロトニン" in user_input or "serotonin" in user_input.lower():
            response['response'] = "セロトニン受容体に結合する分子についてお答えします。"
            response['suggestions'] = [
                "トリプタミン骨格の分子を検索",
                "セロトニン5-HT1A/5-HT2A受容体予測を実行",
                "類似体を生成"
            ]
        elif "GABA" in user_input or "gaba" in user_input.lower():
            response['response'] = "GABA受容体に結合する分子についてお答えします。"
            response['suggestions'] = [
                "GABA作動薬骨格の分子を検索",
                "GABA-A受容体予測を実行",
                "類似体を生成"
            ]
        else:
            response['response'] = "ご質問を理解しました。どのような分子設計をお手伝いしましょうか？"
            response['suggestions'] = [
                "CNSターゲットを指定してください",
                "SMILES文字列を入力してください",
                "骨格タイプを選択してください"
            ]
        
        return response
    
    def _handle_smiles_input(self, smiles: str) -> Dict[str, Any]:
        """SMILES入力処理"""
        response = {
            'response': '',
            'suggestions': [],
            'results': {}
        }
        
        try:
            # 分子特徴量計算
            features = self.property_predictor.predict_physicochemical_properties(smiles)
            
            if features:
                response['response'] = f"SMILES: {smiles} の分析結果です。"
                response['results'] = {
                    'features': features,
                    'smiles': smiles
                }
                response['suggestions'] = [
                    "CNSターゲット予測を実行",
                    "ADMET特性を分析",
                    "類似体を生成"
                ]
            else:
                response['response'] = "無効なSMILES文字列です。正しい形式で入力してください。"
                response['suggestions'] = [
                    "SMILES形式を確認してください",
                    "例: CC(CC1=CC=CC=C1)N"
                ]
        
        except Exception as e:
            response['response'] = f"SMILES解析エラー: {e}"
        
        return response
    
    def _handle_scaffold_search(self, user_input: str) -> Dict[str, Any]:
        """骨格検索処理"""
        response = {
            'response': '',
            'suggestions': [],
            'results': {}
        }
        
        try:
            # 骨格タイプ抽出
            scaffold_type = user_input.replace("骨格検索: ", "")
            compounds = self.scaffolds.get_scaffold_compounds(scaffold_type)
            
            if compounds:
                response['response'] = f"{scaffold_type}骨格の化合物を{len(compounds)}個見つけました。"
                response['results'] = {
                    'scaffold_type': scaffold_type,
                    'compounds': compounds
                }
                response['suggestions'] = [
                    "代表的な化合物を表示",
                    "類似体を生成",
                    "物性を予測"
                ]
            else:
                response['response'] = f"{scaffold_type}骨格の化合物が見つかりませんでした。"
                response['suggestions'] = [
                    "利用可能な骨格タイプを確認",
                    "別の骨格タイプを試す"
                ]
        
        except Exception as e:
            response['response'] = f"骨格検索エラー: {e}"
        
        return response
    
    def _handle_molecule_generation(self, user_input: str) -> Dict[str, Any]:
        """分子生成処理"""
        response = {
            'response': '',
            'suggestions': [],
            'results': {}
        }
        
        try:
            # 生成パラメータ抽出
            parts = user_input.replace("分子生成: ", "").split(" x")
            scaffold_type = parts[0]
            num_analogs = int(parts[1]) if len(parts) > 1 else 5
            
            # 類似体生成
            analogs = self.generator.generate_analogs(scaffold_type, num_analogs)
            
            if analogs:
                response['response'] = f"{scaffold_type}骨格から{len(analogs)}個の類似体を生成しました。"
                response['results'] = {
                    'scaffold_type': scaffold_type,
                    'analogs': analogs
                }
                response['suggestions'] = [
                    "最高スコアの分子を表示",
                    "物性を分析",
                    "さらに生成"
                ]
            else:
                response['response'] = "類似体の生成に失敗しました。"
                response['suggestions'] = [
                    "別の骨格タイプを試す",
                    "生成数を減らす"
                ]
        
        except Exception as e:
            response['response'] = f"分子生成エラー: {e}"
        
        return response
    
    def display_results(self, results: Dict[str, Any]):
        """結果表示"""
        if 'features' in results:
            st.subheader("📊 分子特徴量")
            features_df = pd.DataFrame([results['features']])
            st.dataframe(features_df, use_container_width=True)
        
        if 'compounds' in results:
            st.subheader("🧪 骨格化合物")
            for compound in results['compounds'][:3]:  # 最初の3つを表示
                with st.expander(f"{compound.name}"):
                    st.write(f"**SMILES**: {compound.smiles}")
                    st.write(f"**作用機序**: {compound.mechanism}")
                    st.write(f"**治療用途**: {compound.therapeutic_use}")
        
        if 'analogs' in results:
            st.subheader("🔬 生成された類似体")
            for i, analog in enumerate(results['analogs'][:5]):  # 最初の5つを表示
                with st.expander(f"類似体 {i+1} (スコア: {analog.score:.3f})"):
                    st.write(f"**SMILES**: {analog.smiles}")
                    st.write(f"**親化合物**: {analog.parent_compound}")
                    st.write(f"**修飾**: {', '.join(analog.modifications)}")
    
    def show_scaffold_info(self):
        """骨格情報表示"""
        st.subheader("🧬 利用可能な骨格")
        
        scaffold_types = self.scaffolds.get_all_scaffold_types()
        
        for scaffold_type in scaffold_types:
            compounds = self.scaffolds.get_scaffold_compounds(scaffold_type)
            st.write(f"**{scaffold_type}**: {len(compounds)}個の化合物")
    
    def show_smiles_conversion(self):
        """SMILES変換表示"""
        st.subheader("🔄 SMILES変換")
        
        smiles_input = st.text_input("SMILES文字列を入力してください")
        
        if smiles_input:
            try:
                from rdkit import Chem
                mol = Chem.MolFromSmiles(smiles_input)
                if mol:
                    st.success("有効なSMILES文字列です")
                    st.write(f"**分子式**: {Chem.MolToSmiles(mol)}")
                else:
                    st.error("無効なSMILES文字列です")
            except Exception as e:
                st.error(f"変換エラー: {e}")
    
    def export_conversation(self):
        """会話エクスポート"""
        if st.session_state.current_conversation:
            # JSON形式でエクスポート
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'conversation': st.session_state.current_conversation
            }
            
            st.download_button(
                label="会話をダウンロード",
                data=json.dumps(export_data, ensure_ascii=False, indent=2),
                file_name=f"chemforge_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        else:
            st.warning("エクスポートする会話がありません")
