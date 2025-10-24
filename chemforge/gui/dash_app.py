"""
Dash App Module

Dashアプリモジュール
高度な可視化・インタラクティブ機能の統合GUI
"""

import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
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

class DashApp:
    """
    Dashアプリクラス
    
    高度な可視化・インタラクティブ機能の統合GUI
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
        self.logger = Logger("DashApp")
        self.validator = DataValidator()
        
        # 分子生成・特徴量・ADMET予測
        self.molecular_generator = MolecularGenerator(config_path, cache_dir)
        self.molecular_features = MolecularFeatures(config_path, cache_dir)
        self.rdkit_descriptors = RDKitDescriptors(config_path, cache_dir)
        self.admet_predictor = ADMETPredictor(config_path, cache_dir)
        self.visualization_manager = VisualizationManager(config_path, cache_dir)
        
        # Dashアプリ初期化
        self.app = dash.Dash(__name__)
        self.app.title = "ChemForge - Molecular Discovery Platform"
        
        # レイアウト設定
        self.app.layout = self._create_layout()
        
        # コールバック設定
        self._setup_callbacks()
        
        logger.info("DashApp initialized")
    
    def _create_layout(self):
        """
        レイアウト作成
        
        Returns:
            Dashレイアウト
        """
        return html.Div([
            # ヘッダー
            html.Div([
                html.H1("🧬 ChemForge - Molecular Discovery Platform", 
                       style={'textAlign': 'center', 'color': '#2E86AB', 'marginBottom': '30px'}),
                html.P("AI-Powered Molecular Discovery and Optimization", 
                      style={'textAlign': 'center', 'color': '#666', 'fontSize': '18px'})
            ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'marginBottom': '20px'}),
            
            # タブ
            dcc.Tabs(id="main-tabs", value="home", children=[
                dcc.Tab(label="🏠 Home", value="home"),
                dcc.Tab(label="🧬 Molecular Generation", value="generation"),
                dcc.Tab(label="🎯 Molecular Optimization", value="optimization"),
                dcc.Tab(label="🔬 ADMET Prediction", value="admet"),
                dcc.Tab(label="📊 Visualization", value="visualization"),
                dcc.Tab(label="⚙️ Settings", value="settings")
            ]),
            
            # メインコンテンツ
            html.Div(id="main-content", style={'marginTop': '20px'}),
            
            # フッター
            html.Div([
                html.P("© 2025 ChemForge - Molecular Discovery Platform", 
                      style={'textAlign': 'center', 'color': '#666', 'fontSize': '14px'})
            ], style={'marginTop': '50px', 'padding': '20px'})
        ])
    
    def _setup_callbacks(self):
        """
        コールバック設定
        """
        @self.app.callback(
            Output("main-content", "children"),
            Input("main-tabs", "value")
        )
        def update_main_content(active_tab):
            if active_tab == "home":
                return self._create_home_content()
            elif active_tab == "generation":
                return self._create_generation_content()
            elif active_tab == "optimization":
                return self._create_optimization_content()
            elif active_tab == "admet":
                return self._create_admet_content()
            elif active_tab == "visualization":
                return self._create_visualization_content()
            elif active_tab == "settings":
                return self._create_settings_content()
            else:
                return html.Div("Page not found")
        
        # 分子生成コールバック
        @self.app.callback(
            [Output("generation-results", "children"),
             Output("generation-status", "children")],
            [Input("generate-button", "n_clicks")],
            [State("generation-method", "value"),
             State("num-molecules", "value"),
             State("target-mw", "value"),
             State("target-logp", "value"),
             State("target-tpsa", "value"),
             State("seed-molecules", "value")]
        )
        def generate_molecules(n_clicks, method, num_molecules, target_mw, target_logp, target_tpsa, seed_molecules):
            if n_clicks is None:
                return "", ""
            
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
                    method=method,
                    target_properties=target_properties,
                    seed_molecules=seed_list if seed_list else None
                )
                
                # 結果表示
                df_molecules = pd.DataFrame({'SMILES': molecules})
                
                results_table = dash_table.DataTable(
                    data=df_molecules.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in df_molecules.columns],
                    style_cell={'textAlign': 'left'},
                    style_header={'backgroundColor': '#2E86AB', 'color': 'white'},
                    style_data={'backgroundColor': '#f8f9fa'},
                    page_size=10,
                    sort_action="native",
                    filter_action="native"
                )
                
                status = f"✅ Generated {len(molecules)} molecules successfully!"
                
                return results_table, status
                
            except Exception as e:
                error_msg = f"❌ Generation failed: {str(e)}"
                return "", error_msg
        
        # 分子最適化コールバック
        @self.app.callback(
            [Output("optimization-results", "children"),
             Output("optimization-status", "children")],
            [Input("optimize-button", "n_clicks")],
            [State("optimization-method", "value"),
             State("max-iterations", "value"),
             State("opt-target-mw", "value"),
             State("opt-target-logp", "value"),
             State("input-molecules", "value")]
        )
        def optimize_molecules(n_clicks, method, max_iterations, target_mw, target_logp, input_molecules):
            if n_clicks is None:
                return "", ""
            
            try:
                # 入力分子処理
                molecules = [smiles.strip() for smiles in input_molecules.split('\n') if smiles.strip()]
                
                if not molecules:
                    return "", "❌ Please enter input molecules"
                
                # ターゲット特性設定
                target_properties = {
                    'MW': target_mw,
                    'LogP': target_logp
                }
                
                # 分子最適化
                optimized_molecules = self.molecular_generator.optimize_molecules(
                    molecules=molecules,
                    target_properties=target_properties,
                    optimization_method=method,
                    max_iterations=max_iterations
                )
                
                # 結果表示
                df_optimized = pd.DataFrame({'SMILES': optimized_molecules})
                
                results_table = dash_table.DataTable(
                    data=df_optimized.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in df_optimized.columns],
                    style_cell={'textAlign': 'left'},
                    style_header={'backgroundColor': '#2E86AB', 'color': 'white'},
                    style_data={'backgroundColor': '#f8f9fa'},
                    page_size=10,
                    sort_action="native",
                    filter_action="native"
                )
                
                status = f"✅ Optimized {len(optimized_molecules)} molecules successfully!"
                
                return results_table, status
                
            except Exception as e:
                error_msg = f"❌ Optimization failed: {str(e)}"
                return "", error_msg
        
        # ADMET予測コールバック
        @self.app.callback(
            [Output("admet-results", "children"),
             Output("admet-status", "children")],
            [Input("predict-button", "n_clicks")],
            [State("admet-molecules", "value"),
             State("predict-physicochemical", "value"),
             State("predict-pharmacokinetics", "value"),
             State("predict-toxicity", "value"),
             State("predict-druglikeness", "value")]
        )
        def predict_admet(n_clicks, molecules, physicochemical, pharmacokinetics, toxicity, druglikeness):
            if n_clicks is None:
                return "", ""
            
            try:
                # 入力分子処理
                molecule_list = [smiles.strip() for smiles in molecules.split('\n') if smiles.strip()]
                
                if not molecule_list:
                    return "", "❌ Please enter input molecules"
                
                # ADMET予測
                admet_results = []
                
                for smiles in tqdm(molecule_list[:10], desc="ADMET prediction"):
                    try:
                        admet_pred = self.admet_predictor.predict_admet(
                            smiles,
                            include_physicochemical=physicochemical,
                            include_pharmacokinetics=pharmacokinetics,
                            include_toxicity=toxicity,
                            include_druglikeness=druglikeness
                        )
                        admet_pred['SMILES'] = smiles
                        admet_results.append(admet_pred)
                    except Exception as e:
                        logger.warning(f"ADMET prediction failed for {smiles}: {e}")
                        continue
                
                if admet_results:
                    admet_df = pd.DataFrame(admet_results)
                    
                    results_table = dash_table.DataTable(
                        data=admet_df.to_dict('records'),
                        columns=[{"name": i, "id": i} for i in admet_df.columns],
                        style_cell={'textAlign': 'left'},
                        style_header={'backgroundColor': '#2E86AB', 'color': 'white'},
                        style_data={'backgroundColor': '#f8f9fa'},
                        page_size=10,
                        sort_action="native",
                        filter_action="native"
                    )
                    
                    status = f"✅ Predicted ADMET properties for {len(admet_results)} molecules!"
                    
                    return results_table, status
                else:
                    return "", "❌ No ADMET predictions generated"
                
            except Exception as e:
                error_msg = f"❌ ADMET prediction failed: {str(e)}"
                return "", error_msg
        
        # 可視化コールバック
        @self.app.callback(
            Output("visualization-plot", "figure"),
            [Input("viz-type", "value"),
             Input("plot-type", "value"),
             Input("color-palette", "value")]
        )
        def update_visualization(viz_type, plot_type, color_palette):
            # サンプルデータ生成
            np.random.seed(42)
            n_samples = 100
            
            if viz_type == "Molecular Properties":
                df = pd.DataFrame({
                    'MW': np.random.normal(300, 50, n_samples),
                    'LogP': np.random.normal(2.5, 1.0, n_samples),
                    'TPSA': np.random.normal(50, 20, n_samples),
                    'HBD': np.random.poisson(2, n_samples),
                    'HBA': np.random.poisson(4, n_samples)
                })
            elif viz_type == "ADMET Distribution":
                df = pd.DataFrame({
                    'ADMET_Score': np.random.normal(0.7, 0.2, n_samples),
                    'Toxicity_Risk': np.random.normal(0.3, 0.1, n_samples),
                    'Druglikeness': np.random.normal(0.8, 0.15, n_samples)
                })
            else:
                df = pd.DataFrame({
                    'X': np.random.normal(0, 1, n_samples),
                    'Y': np.random.normal(0, 1, n_samples)
                })
            
            # プロット作成
            if plot_type == "Scatter":
                if len(df.columns) >= 2:
                    fig = px.scatter(df, x=df.columns[0], y=df.columns[1], 
                                   color_discrete_sequence=[color_palette])
                else:
                    fig = go.Figure()
            elif plot_type == "Histogram":
                fig = px.histogram(df, x=df.columns[0], 
                                 color_discrete_sequence=[color_palette])
            elif plot_type == "Box Plot":
                fig = px.box(df, y=df.columns[0], 
                           color_discrete_sequence=[color_palette])
            else:
                fig = go.Figure()
            
            fig.update_layout(
                title=f"{viz_type} - {plot_type}",
                xaxis_title=df.columns[0] if len(df.columns) > 0 else "X",
                yaxis_title=df.columns[1] if len(df.columns) > 1 else "Y",
                template="plotly_white"
            )
            
            return fig
    
    def _create_home_content(self):
        """
        ホームコンテンツ作成
        
        Returns:
            ホームコンテンツ
        """
        return html.Div([
            # 統計情報
            html.Div([
                html.Div([
                    html.H3("15+", style={'color': '#2E86AB', 'fontSize': '36px'}),
                    html.P("Available Targets", style={'color': '#666'})
                ], className="col-md-3", style={'textAlign': 'center', 'padding': '20px'}),
                
                html.Div([
                    html.H3("3", style={'color': '#2E86AB', 'fontSize': '36px'}),
                    html.P("Generation Methods", style={'color': '#666'})
                ], className="col-md-3", style={'textAlign': 'center', 'padding': '20px'}),
                
                html.Div([
                    html.H3("20+", style={'color': '#2E86AB', 'fontSize': '36px'}),
                    html.P("ADMET Properties", style={'color': '#666'})
                ], className="col-md-3", style={'textAlign': 'center', 'padding': '20px'}),
                
                html.Div([
                    html.H3("Interactive", style={'color': '#2E86AB', 'fontSize': '36px'}),
                    html.P("Visualization", style={'color': '#666'})
                ], className="col-md-3", style={'textAlign': 'center', 'padding': '20px'})
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '40px'}),
            
            # 機能紹介
            html.Div([
                html.H2("Key Features", style={'textAlign': 'center', 'marginBottom': '30px'}),
                
                html.Div([
                    html.Div([
                        html.H4("🧬 Molecular Generation", style={'color': '#2E86AB'}),
                        html.Ul([
                            html.Li("VAE-based generation"),
                            html.Li("Reinforcement Learning"),
                            html.Li("Genetic Algorithms"),
                            html.Li("Target property optimization")
                        ])
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '4%'}),
                    
                    html.Div([
                        html.H4("🎯 Molecular Optimization", style={'color': '#2E86AB'}),
                        html.Ul([
                            html.Li("Target property optimization"),
                            html.Li("Multi-objective optimization"),
                            html.Li("Constraint handling"),
                            html.Li("Performance evaluation")
                        ])
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
                ]),
                
                html.Div([
                    html.Div([
                        html.H4("🔬 ADMET Prediction", style={'color': '#2E86AB'}),
                        html.Ul([
                            html.Li("Physicochemical properties"),
                            html.Li("Pharmacokinetics"),
                            html.Li("Toxicity assessment"),
                            html.Li("Drug-likeness evaluation")
                        ])
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '4%'}),
                    
                    html.Div([
                        html.H4("📊 Visualization", style={'color': '#2E86AB'}),
                        html.Ul([
                            html.Li("Interactive molecular plots"),
                            html.Li("Property distribution"),
                            html.Li("Optimization history"),
                            html.Li("Performance metrics")
                        ])
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
                ], style={'marginTop': '20px'})
            ])
        ])
    
    def _create_generation_content(self):
        """
        分子生成コンテンツ作成
        
        Returns:
            分子生成コンテンツ
        """
        return html.Div([
            html.H2("🧬 Molecular Generation", style={'textAlign': 'center', 'marginBottom': '30px'}),
            
            # 生成設定
            html.Div([
                html.Div([
                    html.H4("Generation Settings"),
                    
                    html.Label("Generation Method:"),
                    dcc.Dropdown(
                        id="generation-method",
                        options=[
                            {'label': 'VAE', 'value': 'vae'},
                            {'label': 'Reinforcement Learning', 'value': 'rl'},
                            {'label': 'Genetic Algorithm', 'value': 'ga'}
                        ],
                        value='vae',
                        style={'marginBottom': '20px'}
                    ),
                    
                    html.Label("Number of Molecules:"),
                    dcc.Slider(
                        id="num-molecules",
                        min=10,
                        max=1000,
                        step=10,
                        value=100,
                        marks={i: str(i) for i in range(10, 1001, 100)},
                        style={'marginBottom': '20px'}
                    ),
                    
                    html.Label("Target Properties:"),
                    html.Div([
                        html.Div([
                            html.Label("Molecular Weight:"),
                            dcc.Input(id="target-mw", type="number", value=300, min=100, max=1000)
                        ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%'}),
                        
                        html.Div([
                            html.Label("LogP:"),
                            dcc.Input(id="target-logp", type="number", value=2.5, min=-2, max=6, step=0.1)
                        ], style={'width': '48%', 'display': 'inline-block'})
                    ]),
                    
                    html.Label("TPSA:"),
                    dcc.Input(id="target-tpsa", type="number", value=50, min=0, max=200, style={'marginBottom': '20px'}),
                    
                    html.Label("Seed Molecules (SMILES):"),
                    dcc.Textarea(
                        id="seed-molecules",
                        value="CCO\nCCN\nCC(=O)O",
                        style={'width': '100%', 'height': '100px', 'marginBottom': '20px'}
                    )
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '4%'}),
                
                html.Div([
                    html.H4("Advanced Settings"),
                    
                    html.Label("Filters:"),
                    dcc.Checklist(
                        options=[
                            {'label': 'Filter by Molecular Weight', 'value': 'mw'},
                            {'label': 'Filter by LogP', 'value': 'logp'},
                            {'label': 'Filter by TPSA', 'value': 'tpsa'}
                        ],
                        value=['mw', 'logp', 'tpsa'],
                        style={'marginBottom': '20px'}
                    ),
                    
                    html.Label("Save Settings:"),
                    dcc.Checklist(
                        options=[
                            {'label': 'Save Molecular Features', 'value': 'features'},
                            {'label': 'Save ADMET Predictions', 'value': 'admet'}
                        ],
                        value=['features', 'admet'],
                        style={'marginBottom': '20px'}
                    )
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ]),
            
            # 生成ボタン
            html.Div([
                html.Button("🚀 Generate Molecules", id="generate-button", 
                           style={'backgroundColor': '#2E86AB', 'color': 'white', 'padding': '10px 20px', 
                                 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'})
            ], style={'textAlign': 'center', 'margin': '30px 0'}),
            
            # ステータス
            html.Div(id="generation-status", style={'textAlign': 'center', 'margin': '20px 0'}),
            
            # 結果表示
            html.Div([
                html.H4("Generated Molecules"),
                html.Div(id="generation-results")
            ])
        ])
    
    def _create_optimization_content(self):
        """
        分子最適化コンテンツ作成
        
        Returns:
            分子最適化コンテンツ
        """
        return html.Div([
            html.H2("🎯 Molecular Optimization", style={'textAlign': 'center', 'marginBottom': '30px'}),
            
            # 最適化設定
            html.Div([
                html.Div([
                    html.H4("Optimization Settings"),
                    
                    html.Label("Optimization Method:"),
                    dcc.Dropdown(
                        id="optimization-method",
                        options=[
                            {'label': 'Genetic Algorithm', 'value': 'ga'},
                            {'label': 'Reinforcement Learning', 'value': 'rl'}
                        ],
                        value='ga',
                        style={'marginBottom': '20px'}
                    ),
                    
                    html.Label("Max Iterations:"),
                    dcc.Slider(
                        id="max-iterations",
                        min=10,
                        max=1000,
                        step=10,
                        value=100,
                        marks={i: str(i) for i in range(10, 1001, 100)},
                        style={'marginBottom': '20px'}
                    ),
                    
                    html.Label("Target Properties:"),
                    html.Div([
                        html.Div([
                            html.Label("Target Molecular Weight:"),
                            dcc.Input(id="opt-target-mw", type="number", value=300, min=100, max=1000)
                        ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%'}),
                        
                        html.Div([
                            html.Label("Target LogP:"),
                            dcc.Input(id="opt-target-logp", type="number", value=2.5, min=-2, max=6, step=0.1)
                        ], style={'width': '48%', 'display': 'inline-block'})
                    ])
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '4%'}),
                
                html.Div([
                    html.H4("Input Molecules"),
                    
                    html.Label("Input Molecules (SMILES):"),
                    dcc.Textarea(
                        id="input-molecules",
                        value="CCO\nCCN\nCC(=O)O",
                        style={'width': '100%', 'height': '200px', 'marginBottom': '20px'}
                    ),
                    
                    html.Label("Property Weights:"),
                    html.Div([
                        html.Div([
                            html.Label("Molecular Weight Weight:"),
                            dcc.Slider(id="weight-mw", min=0, max=1, step=0.1, value=0.5)
                        ], style={'marginBottom': '20px'}),
                        
                        html.Div([
                            html.Label("LogP Weight:"),
                            dcc.Slider(id="weight-logp", min=0, max=1, step=0.1, value=0.5)
                        ])
                    ])
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ]),
            
            # 最適化ボタン
            html.Div([
                html.Button("🎯 Optimize Molecules", id="optimize-button", 
                           style={'backgroundColor': '#2E86AB', 'color': 'white', 'padding': '10px 20px', 
                                 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'})
            ], style={'textAlign': 'center', 'margin': '30px 0'}),
            
            # ステータス
            html.Div(id="optimization-status", style={'textAlign': 'center', 'margin': '20px 0'}),
            
            # 結果表示
            html.Div([
                html.H4("Optimization Results"),
                html.Div(id="optimization-results")
            ])
        ])
    
    def _create_admet_content(self):
        """
        ADMET予測コンテンツ作成
        
        Returns:
            ADMET予測コンテンツ
        """
        return html.Div([
            html.H2("🔬 ADMET Prediction", style={'textAlign': 'center', 'marginBottom': '30px'}),
            
            # ADMET予測設定
            html.Div([
                html.Div([
                    html.H4("Input Molecules"),
                    
                    html.Label("Input Molecules (SMILES):"),
                    dcc.Textarea(
                        id="admet-molecules",
                        value="CCO\nCCN\nCC(=O)O",
                        style={'width': '100%', 'height': '200px', 'marginBottom': '20px'}
                    )
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '4%'}),
                
                html.Div([
                    html.H4("Prediction Settings"),
                    
                    html.Label("Prediction Options:"),
                    dcc.Checklist(
                        id="predict-physicochemical",
                        options=[{'label': 'Physicochemical Properties', 'value': True}],
                        value=[True],
                        style={'marginBottom': '20px'}
                    ),
                    
                    dcc.Checklist(
                        id="predict-pharmacokinetics",
                        options=[{'label': 'Pharmacokinetics', 'value': True}],
                        value=[True],
                        style={'marginBottom': '20px'}
                    ),
                    
                    dcc.Checklist(
                        id="predict-toxicity",
                        options=[{'label': 'Toxicity', 'value': True}],
                        value=[True],
                        style={'marginBottom': '20px'}
                    ),
                    
                    dcc.Checklist(
                        id="predict-druglikeness",
                        options=[{'label': 'Drug-likeness', 'value': True}],
                        value=[True],
                        style={'marginBottom': '20px'}
                    ),
                    
                    html.Label("Batch Size:"),
                    dcc.Slider(
                        id="batch-size",
                        min=1,
                        max=100,
                        step=1,
                        value=10,
                        marks={i: str(i) for i in range(1, 101, 10)},
                        style={'marginBottom': '20px'}
                    ),
                    
                    dcc.Checkbox(
                        id="use-cache",
                        label="Use Cache",
                        checked=True,
                        style={'marginBottom': '20px'}
                    ),
                    
                    dcc.Checkbox(
                        id="use-parallel",
                        label="Use Parallel Processing",
                        checked=True
                    )
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ]),
            
            # 予測ボタン
            html.Div([
                html.Button("🔬 Predict ADMET", id="predict-button", 
                           style={'backgroundColor': '#2E86AB', 'color': 'white', 'padding': '10px 20px', 
                                 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'})
            ], style={'textAlign': 'center', 'margin': '30px 0'}),
            
            # ステータス
            html.Div(id="admet-status", style={'textAlign': 'center', 'margin': '20px 0'}),
            
            # 結果表示
            html.Div([
                html.H4("ADMET Predictions"),
                html.Div(id="admet-results")
            ])
        ])
    
    def _create_visualization_content(self):
        """
        可視化コンテンツ作成
        
        Returns:
            可視化コンテンツ
        """
        return html.Div([
            html.H2("📊 Visualization", style={'textAlign': 'center', 'marginBottom': '30px'}),
            
            # 可視化設定
            html.Div([
                html.Div([
                    html.H4("Visualization Settings"),
                    
                    html.Label("Visualization Type:"),
                    dcc.Dropdown(
                        id="viz-type",
                        options=[
                            {'label': 'Molecular Properties', 'value': 'Molecular Properties'},
                            {'label': 'ADMET Distribution', 'value': 'ADMET Distribution'},
                            {'label': 'Optimization History', 'value': 'Optimization History'},
                            {'label': 'Performance Metrics', 'value': 'Performance Metrics'}
                        ],
                        value='Molecular Properties',
                        style={'marginBottom': '20px'}
                    ),
                    
                    html.Label("Plot Type:"),
                    dcc.Dropdown(
                        id="plot-type",
                        options=[
                            {'label': 'Scatter', 'value': 'Scatter'},
                            {'label': 'Histogram', 'value': 'Histogram'},
                            {'label': 'Box Plot', 'value': 'Box Plot'},
                            {'label': 'Heatmap', 'value': 'Heatmap'}
                        ],
                        value='Scatter',
                        style={'marginBottom': '20px'}
                    ),
                    
                    html.Label("Color Palette:"),
                    dcc.Dropdown(
                        id="color-palette",
                        options=[
                            {'label': 'Viridis', 'value': 'viridis'},
                            {'label': 'Plasma', 'value': 'plasma'},
                            {'label': 'Inferno', 'value': 'inferno'},
                            {'label': 'Magma', 'value': 'magma'}
                        ],
                        value='viridis',
                        style={'marginBottom': '20px'}
                    )
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '4%'}),
                
                html.Div([
                    html.H4("Display Settings"),
                    
                    dcc.Checkbox(
                        id="show-legend",
                        label="Show Legend",
                        checked=True,
                        style={'marginBottom': '20px'}
                    ),
                    
                    dcc.Checkbox(
                        id="show-grid",
                        label="Show Grid",
                        checked=True,
                        style={'marginBottom': '20px'}
                    ),
                    
                    html.Label("Figure Size:"),
                    html.Div([
                        html.Div([
                            html.Label("Width:"),
                            dcc.Input(id="fig-width", type="number", value=800, min=400, max=1200)
                        ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%'}),
                        
                        html.Div([
                            html.Label("Height:"),
                            dcc.Input(id="fig-height", type="number", value=600, min=300, max=900)
                        ], style={'width': '48%', 'display': 'inline-block'})
                    ])
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ]),
            
            # 可視化プロット
            html.Div([
                dcc.Graph(id="visualization-plot", style={'height': '600px'})
            ], style={'marginTop': '30px'})
        ])
    
    def _create_settings_content(self):
        """
        設定コンテンツ作成
        
        Returns:
            設定コンテンツ
        """
        return html.Div([
            html.H2("⚙️ Settings", style={'textAlign': 'center', 'marginBottom': '30px'}),
            
            # 設定カテゴリ
            html.Div([
                html.Label("Settings Category:"),
                dcc.Dropdown(
                    id="settings-category",
                    options=[
                        {'label': 'General', 'value': 'general'},
                        {'label': 'Molecular Generation', 'value': 'generation'},
                        {'label': 'ADMET Prediction', 'value': 'admet'},
                        {'label': 'Visualization', 'value': 'visualization'},
                        {'label': 'Advanced', 'value': 'advanced'}
                    ],
                    value='general',
                    style={'marginBottom': '20px'}
                )
            ]),
            
            # 一般設定
            html.Div(id="settings-content", style={'marginTop': '20px'})
        ])
    
    def run(self, debug: bool = False, port: int = 8050):
        """
        Dashアプリ実行
        
        Args:
            debug: デバッグモード
            port: ポート番号
        """
        self.app.run_server(debug=debug, port=port)

def create_dash_app(config_path: Optional[str] = None, 
                    cache_dir: str = "cache") -> DashApp:
    """
    Dashアプリ作成
    
    Args:
        config_path: 設定ファイルパス
        cache_dir: キャッシュディレクトリ
        
    Returns:
        DashApp
    """
    return DashApp(config_path, cache_dir)

if __name__ == "__main__":
    # テスト実行
    app = DashApp()
    app.run(debug=True, port=8050)