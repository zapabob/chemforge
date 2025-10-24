"""
GUIウィジェットのテスト

GUIコンポーネントの単体テスト
"""

import unittest
from unittest.mock import Mock, patch
from chemforge.gui.prediction_widget import PredictionWidget
from chemforge.gui.visualization_widget import VisualizationWidget
from chemforge.gui.chat_widget import ChatWidget

class TestPredictionWidget(unittest.TestCase):
    """予測ウィジェットのテストクラス"""
    
    def setUp(self):
        """テスト前の準備"""
        self.widget = PredictionWidget()
        self.test_smiles = "CC(CC1=CC=CC=C1)N"  # アンフェタミン
    
    def test_initialization(self):
        """初期化のテスト"""
        self.assertIsInstance(self.widget, PredictionWidget)
        self.assertIsNotNone(self.widget.property_predictor)
        self.assertIsNotNone(self.widget.molecular_features)
    
    @patch('chemforge.gui.prediction_widget.st')
    def test_predict_cns_targets(self, mock_st):
        """CNSターゲット予測のテスト"""
        # Streamlitのモック設定
        mock_col1 = Mock()
        mock_col2 = Mock()
        mock_st.columns.return_value = [mock_col1, mock_col2]
        mock_st.write.return_value = None
        mock_st.metric.return_value = None
        mock_st.plotly_chart.return_value = None
        
        # コンテキストマネージャーのモック
        mock_col1.__enter__ = Mock(return_value=mock_col1)
        mock_col1.__exit__ = Mock(return_value=None)
        mock_col2.__enter__ = Mock(return_value=mock_col2)
        mock_col2.__exit__ = Mock(return_value=None)
        
        results = self.widget.predict_cns_targets(self.test_smiles)
        
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)
        
        # CNSターゲットの存在チェック
        expected_targets = [
            'Dopamine D1', 'Dopamine D2', 'Serotonin 5-HT1A', 'Serotonin 5-HT2A',
            'GABA-A', 'NMDA', 'Cannabinoid CB1', 'Opioid μ'
        ]
        
        for target in expected_targets:
            if target in results:
                self.assertIsInstance(results[target], (int, float))
                self.assertGreaterEqual(results[target], 0.0)
                self.assertLessEqual(results[target], 1.0)
    
    @patch('chemforge.gui.prediction_widget.st')
    def test_predict_admet_properties(self, mock_st):
        """ADMET特性予測のテスト"""
        # Streamlitのモック設定
        mock_col1 = Mock()
        mock_col2 = Mock()
        mock_col3 = Mock()
        mock_st.columns.return_value = [mock_col1, mock_col2, mock_col3]
        mock_st.write.return_value = None
        mock_st.metric.return_value = None
        mock_st.plotly_chart.return_value = None
        
        # コンテキストマネージャーのモック
        mock_col1.__enter__ = Mock(return_value=mock_col1)
        mock_col1.__exit__ = Mock(return_value=None)
        mock_col2.__enter__ = Mock(return_value=mock_col2)
        mock_col2.__exit__ = Mock(return_value=None)
        mock_col3.__enter__ = Mock(return_value=mock_col3)
        mock_col3.__exit__ = Mock(return_value=None)
        
        results = self.widget.predict_admet_properties(self.test_smiles)
        
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)
        
        # ADMET特性の存在チェック
        expected_properties = [
            'Absorption', 'Distribution', 'Metabolism', 'Excretion', 'Toxicity'
        ]
        
        for prop in expected_properties:
            if prop in results:
                self.assertIsInstance(results[prop], (int, float))
                self.assertGreaterEqual(results[prop], 0.0)
                self.assertLessEqual(results[prop], 1.0)

class TestVisualizationWidget(unittest.TestCase):
    """可視化ウィジェットのテストクラス"""
    
    def setUp(self):
        """テスト前の準備"""
        self.widget = VisualizationWidget()
        self.test_smiles = "CC(CC1=CC=CC=C1)N"  # アンフェタミン
    
    def test_initialization(self):
        """初期化のテスト"""
        self.assertIsInstance(self.widget, VisualizationWidget)
        self.assertIsNotNone(self.widget.molecular_features)
    
    @patch('chemforge.gui.visualization_widget.st')
    def test_render_2d_structure(self, mock_st):
        """2D構造表示のテスト"""
        # Streamlitのモック設定
        mock_st.subheader.return_value = None
        mock_st.error.return_value = None
        mock_st.image.return_value = None
        mock_st.write.return_value = None
        
        # エラーが発生しないことを確認
        try:
            self.widget.render_2d_structure(self.test_smiles)
        except Exception as e:
            self.fail(f"render_2d_structure raised an exception: {e}")
    
    @patch('chemforge.gui.visualization_widget.st')
    def test_render_property_heatmap(self, mock_st):
        """物性ヒートマップ表示のテスト"""
        # Streamlitのモック設定
        mock_st.subheader.return_value = None
        mock_st.error.return_value = None
        mock_st.plotly_chart.return_value = None
        mock_st.dataframe.return_value = None
        
        # エラーが発生しないことを確認
        try:
            self.widget.render_property_heatmap(self.test_smiles)
        except Exception as e:
            self.fail(f"render_property_heatmap raised an exception: {e}")

class TestChatWidget(unittest.TestCase):
    """チャットウィジェットのテストクラス"""
    
    def setUp(self):
        """テスト前の準備"""
        self.widget = ChatWidget()
    
    def test_initialization(self):
        """初期化のテスト"""
        self.assertIsInstance(self.widget, ChatWidget)
        self.assertIsNotNone(self.widget.scaffolds)
        self.assertIsNotNone(self.widget.generator)
        self.assertIsNotNone(self.widget.property_predictor)
    
    @patch('chemforge.gui.chat_widget.st')
    def test_initialize_chat_history(self, mock_st):
        """チャット履歴初期化のテスト"""
        # セッション状態のモック
        mock_st.session_state = type('MockSessionState', (), {})()
        
        self.widget.initialize_chat_history()
        
        # セッション状態が初期化されることを確認
        self.assertTrue(hasattr(mock_st.session_state, 'chat_history'))
        self.assertTrue(hasattr(mock_st.session_state, 'current_conversation'))
    
    def test_generate_assistant_response_text(self):
        """テキスト入力でのアシスタント応答生成のテスト"""
        user_input = "ドパミン受容体に結合する分子を設計したい"
        
        response = self.widget.generate_assistant_response(user_input, "テキスト")
        
        self.assertIsInstance(response, dict)
        self.assertIn('response', response)
        self.assertIn('suggestions', response)
        self.assertIn('results', response)
        
        self.assertIsInstance(response['response'], str)
        self.assertIsInstance(response['suggestions'], list)
        self.assertIsInstance(response['results'], dict)
    
    def test_generate_assistant_response_smiles(self):
        """SMILES入力でのアシスタント応答生成のテスト"""
        user_input = "CC(CC1=CC=CC=C1)N"
        
        response = self.widget.generate_assistant_response(user_input, "SMILES")
        
        self.assertIsInstance(response, dict)
        self.assertIn('response', response)
        self.assertIn('suggestions', response)
        self.assertIn('results', response)
    
    def test_generate_assistant_response_scaffold_search(self):
        """骨格検索でのアシスタント応答生成のテスト"""
        user_input = "骨格検索: phenethylamine"
        
        response = self.widget.generate_assistant_response(user_input, "骨格検索")
        
        self.assertIsInstance(response, dict)
        self.assertIn('response', response)
        self.assertIn('suggestions', response)
        self.assertIn('results', response)
    
    def test_generate_assistant_response_molecule_generation(self):
        """分子生成でのアシスタント応答生成のテスト"""
        user_input = "分子生成: phenethylamine x5"
        
        response = self.widget.generate_assistant_response(user_input, "分子生成")
        
        self.assertIsInstance(response, dict)
        self.assertIn('response', response)
        self.assertIn('suggestions', response)
        self.assertIn('results', response)

if __name__ == '__main__':
    unittest.main()
