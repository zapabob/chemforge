"""
Visualization Manager Module

可視化管理モジュール
既存VisualizationManagerを活用した効率的な可視化統合
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from pathlib import Path
import json
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 既存モジュール活用
from chemforge.utils.config_utils import ConfigManager
from chemforge.utils.logging_utils import Logger
from chemforge.utils.validation import DataValidator

logger = logging.getLogger(__name__)

class VisualizationManager:
    """
    可視化管理クラス
    
    既存VisualizationManagerを活用した効率的な可視化統合
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
        self.logger = Logger("VisualizationManager")
        self.validator = DataValidator()
        
        # 可視化設定
        self.viz_config = self.config.get('visualization', {})
        self.default_style = self.viz_config.get('default_style', 'seaborn')
        self.color_palette = self.viz_config.get('color_palette', 'viridis')
        self.figure_size = self.viz_config.get('figure_size', (12, 8))
        self.dpi = self.viz_config.get('dpi', 300)
        
        # スタイル設定
        self._setup_plotting_style()
        
        logger.info("VisualizationManager initialized")
    
    def _setup_plotting_style(self):
        """プロットスタイル設定"""
        try:
            # Matplotlibスタイル設定
            plt.style.use(self.default_style)
            plt.rcParams['figure.figsize'] = self.figure_size
            plt.rcParams['figure.dpi'] = self.dpi
            plt.rcParams['font.size'] = 12
            plt.rcParams['axes.labelsize'] = 14
            plt.rcParams['axes.titlesize'] = 16
            plt.rcParams['xtick.labelsize'] = 12
            plt.rcParams['ytick.labelsize'] = 12
            plt.rcParams['legend.fontsize'] = 12
            
            # Seabornスタイル設定
            sns.set_palette(self.color_palette)
            sns.set_style("whitegrid")
            
        except Exception as e:
            logger.warning(f"Error setting up plotting style: {e}")
    
    def plot_molecular_properties(self, data: pd.DataFrame, 
                                 properties: List[str],
                                 plot_type: str = "scatter",
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        分子特性可視化
        
        Args:
            data: データフレーム
            properties: 特性リスト
            plot_type: プロットタイプ
            save_path: 保存パス
            
        Returns:
            Matplotlib図
        """
        logger.info(f"Plotting molecular properties: {properties}")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, prop in enumerate(properties[:4]):
            if prop in data.columns:
                if plot_type == "scatter":
                    axes[i].scatter(data.index, data[prop], alpha=0.6)
                elif plot_type == "histogram":
                    axes[i].hist(data[prop], bins=30, alpha=0.7)
                elif plot_type == "box":
                    axes[i].boxplot(data[prop])
                
                axes[i].set_title(f'{prop} Distribution')
                axes[i].set_xlabel('Index' if plot_type == "scatter" else prop)
                axes[i].set_ylabel(prop)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Molecular properties plot saved to: {save_path}")
        
        return fig
    
    def plot_admet_predictions(self, data: pd.DataFrame,
                             prediction_types: List[str],
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        ADMET予測可視化
        
        Args:
            data: データフレーム
            prediction_types: 予測タイプリスト
            save_path: 保存パス
            
        Returns:
            Matplotlib図
        """
        logger.info(f"Plotting ADMET predictions: {prediction_types}")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, pred_type in enumerate(prediction_types[:4]):
            if pred_type in data.columns:
                # ヒストグラム
                axes[i].hist(data[pred_type], bins=30, alpha=0.7, color=f'C{i}')
                axes[i].set_title(f'{pred_type} Distribution')
                axes[i].set_xlabel(pred_type)
                axes[i].set_ylabel('Frequency')
                
                # 統計情報追加
                mean_val = data[pred_type].mean()
                std_val = data[pred_type].std()
                axes[i].axvline(mean_val, color='red', linestyle='--', 
                              label=f'Mean: {mean_val:.2f}')
                axes[i].axvline(mean_val + std_val, color='orange', linestyle='--', 
                              label=f'±1σ: {std_val:.2f}')
                axes[i].axvline(mean_val - std_val, color='orange', linestyle='--')
                axes[i].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"ADMET predictions plot saved to: {save_path}")
        
        return fig
    
    def plot_training_history(self, history: Dict[str, List],
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        学習履歴可視化
        
        Args:
            history: 学習履歴
            save_path: 保存パス
            
        Returns:
            Matplotlib図
        """
        logger.info("Plotting training history")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 損失プロット
        if 'train_loss' in history and 'val_loss' in history:
            axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
            axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # 学習率プロット
        if 'learning_rates' in history:
            axes[0, 1].plot(history['learning_rates'], color='green')
            axes[0, 1].set_title('Learning Rate Schedule')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].grid(True)
        
        # 評価指標プロット
        if 'train_metrics' in history and 'val_metrics' in history:
            train_metrics = history['train_metrics']
            val_metrics = history['val_metrics']
            
            if train_metrics and val_metrics:
                # 最初のメトリクスをプロット
                first_metric = list(train_metrics[0].keys())[0] if train_metrics[0] else None
                if first_metric:
                    train_values = [m[first_metric] for m in train_metrics]
                    val_values = [m[first_metric] for m in val_metrics]
                    
                    axes[1, 0].plot(train_values, label=f'Train {first_metric}', color='blue')
                    axes[1, 0].plot(val_values, label=f'Val {first_metric}', color='red')
                    axes[1, 0].set_title(f'{first_metric} Over Time')
                    axes[1, 0].set_xlabel('Epoch')
                    axes[1, 0].set_ylabel(first_metric)
                    axes[1, 0].legend()
                    axes[1, 0].grid(True)
        
        # 損失分布
        if 'train_loss' in history:
            axes[1, 1].hist(history['train_loss'], bins=20, alpha=0.7, color='blue', label='Train Loss')
            if 'val_loss' in history:
                axes[1, 1].hist(history['val_loss'], bins=20, alpha=0.7, color='red', label='Val Loss')
            axes[1, 1].set_title('Loss Distribution')
            axes[1, 1].set_xlabel('Loss Value')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Training history plot saved to: {save_path}")
        
        return fig
    
    def plot_prediction_results(self, predictions: Dict[str, np.ndarray],
                              targets: Dict[str, np.ndarray],
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        予測結果可視化
        
        Args:
            predictions: 予測結果
            targets: 正解ラベル
            save_path: 保存パス
            
        Returns:
            Matplotlib図
        """
        logger.info("Plotting prediction results")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, (key, pred) in enumerate(predictions.items()):
            if i >= 4:
                break
                
            if key in targets:
                target = targets[key]
                
                # 散布図
                axes[i].scatter(target, pred, alpha=0.6)
                axes[i].plot([target.min(), target.max()], [target.min(), target.max()], 
                           'r--', label='Perfect Prediction')
                axes[i].set_xlabel(f'True {key}')
                axes[i].set_ylabel(f'Predicted {key}')
                axes[i].set_title(f'{key} Prediction vs Truth')
                axes[i].legend()
                axes[i].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Prediction results plot saved to: {save_path}")
        
        return fig
    
    def create_interactive_plot(self, data: pd.DataFrame,
                              x_col: str, y_col: str,
                              color_col: Optional[str] = None,
                              plot_type: str = "scatter") -> go.Figure:
        """
        インタラクティブプロット作成
        
        Args:
            data: データフレーム
            x_col: X軸カラム
            y_col: Y軸カラム
            color_col: 色分けカラム
            plot_type: プロットタイプ
            
        Returns:
            Plotly図
        """
        logger.info(f"Creating interactive plot: {plot_type}")
        
        if plot_type == "scatter":
            fig = px.scatter(data, x=x_col, y=y_col, color=color_col,
                           title=f'{y_col} vs {x_col}',
                           hover_data=data.columns.tolist())
        elif plot_type == "line":
            fig = px.line(data, x=x_col, y=y_col, color=color_col,
                         title=f'{y_col} vs {x_col}')
        elif plot_type == "bar":
            fig = px.bar(data, x=x_col, y=y_col, color=color_col,
                        title=f'{y_col} vs {x_col}')
        elif plot_type == "histogram":
            fig = px.histogram(data, x=x_col, color=color_col,
                             title=f'{x_col} Distribution')
        else:
            fig = px.scatter(data, x=x_col, y=y_col, color=color_col,
                           title=f'{y_col} vs {x_col}')
        
        fig.update_layout(
            width=800,
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_dashboard(self, data: pd.DataFrame,
                        dashboard_config: Dict[str, Any],
                        save_path: Optional[str] = None) -> go.Figure:
        """
        ダッシュボード作成
        
        Args:
            data: データフレーム
            dashboard_config: ダッシュボード設定
            save_path: 保存パス
            
        Returns:
            Plotly図
        """
        logger.info("Creating dashboard")
        
        # サブプロット作成
        rows = dashboard_config.get('rows', 2)
        cols = dashboard_config.get('cols', 2)
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=dashboard_config.get('titles', []),
            specs=dashboard_config.get('specs', [])
        )
        
        # 各プロット追加
        plots = dashboard_config.get('plots', [])
        for i, plot_config in enumerate(plots):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            if plot_config['type'] == 'scatter':
                fig.add_trace(
                    go.Scatter(
                        x=data[plot_config['x']],
                        y=data[plot_config['y']],
                        mode='markers',
                        name=plot_config.get('name', f'Plot {i+1}')
                    ),
                    row=row, col=col
                )
            elif plot_config['type'] == 'histogram':
                fig.add_trace(
                    go.Histogram(
                        x=data[plot_config['x']],
                        name=plot_config.get('name', f'Plot {i+1}')
                    ),
                    row=row, col=col
                )
        
        # レイアウト更新
        fig.update_layout(
            title_text=dashboard_config.get('title', 'Dashboard'),
            showlegend=True,
            height=dashboard_config.get('height', 600),
            width=dashboard_config.get('width', 1000)
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Dashboard saved to: {save_path}")
        
        return fig
    
    def save_plot(self, fig: Union[plt.Figure, go.Figure], 
                 save_path: str, format: str = "png") -> bool:
        """
        プロット保存
        
        Args:
            fig: 図
            save_path: 保存パス
            format: 保存形式
            
        Returns:
            成功フラグ
        """
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            if isinstance(fig, plt.Figure):
                # Matplotlib図
                fig.savefig(save_path, format=format, dpi=self.dpi, bbox_inches='tight')
            elif isinstance(fig, go.Figure):
                # Plotly図
                if format.lower() == "html":
                    fig.write_html(save_path)
                elif format.lower() == "png":
                    fig.write_image(save_path)
                elif format.lower() == "pdf":
                    fig.write_image(save_path)
                else:
                    fig.write_html(save_path)
            
            logger.info(f"Plot saved to: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving plot: {e}")
            return False
    
    def get_plot_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        プロットサマリー取得
        
        Args:
            data: データフレーム
            
        Returns:
            サマリー辞書
        """
        try:
            summary = {
                'total_records': len(data),
                'total_columns': len(data.columns),
                'numeric_columns': len(data.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(data.select_dtypes(include=['object']).columns),
                'missing_values': data.isnull().sum().to_dict(),
                'data_types': data.dtypes.to_dict()
            }
            
            # 数値列の統計
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                summary['numeric_stats'] = data[numeric_cols].describe().to_dict()
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting plot summary: {e}")
            return {}

def create_visualization_manager(config_path: Optional[str] = None, 
                               cache_dir: str = "cache") -> VisualizationManager:
    """
    可視化管理器作成
    
    Args:
        config_path: 設定ファイルパス
        cache_dir: キャッシュディレクトリ
        
    Returns:
        VisualizationManager
    """
    return VisualizationManager(config_path, cache_dir)

if __name__ == "__main__":
    # テスト実行
    viz_manager = VisualizationManager()
    
    print(f"VisualizationManager created: {viz_manager}")
    print(f"Default style: {viz_manager.default_style}")
    print(f"Color palette: {viz_manager.color_palette}")
    print(f"Figure size: {viz_manager.figure_size}")
    print(f"DPI: {viz_manager.dpi}")
