"""
Visualization utilities for ChemForge platform.

This module provides visualization functionality for molecular data, ADMET properties,
CNS-MPO scores, scaffold analysis, and training progress.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from pathlib import Path
import json

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class MolecularVisualizer:
    """Molecular visualization utilities."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize molecular visualizer.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
    def plot_molecular_properties(self, data: pd.DataFrame, 
                                properties: List[str] = None,
                                figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot molecular properties distribution.
        
        Args:
            data: DataFrame containing molecular data
            properties: List of properties to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if properties is None:
            properties = ['mol_weight', 'logp', 'hbd', 'hba', 'tpsa']
        
        # Filter available properties
        available_props = [prop for prop in properties if prop in data.columns]
        
        if not available_props:
            raise ValueError("No valid properties found in data")
        
        n_props = len(available_props)
        n_cols = min(3, n_props)
        n_rows = (n_props + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_props == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, prop in enumerate(available_props):
            row = i // n_cols
            col = i % n_cols
            
            if n_rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]
                
            # Plot histogram
            ax.hist(data[prop].dropna(), bins=30, alpha=0.7, edgecolor='black')
            ax.set_title(f'Distribution of {prop}')
            ax.set_xlabel(prop)
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
                
        # Hide empty subplots
        for i in range(n_props, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows == 1:
                axes[col].set_visible(False)
            else:
                axes[row, col].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_molecular_scatter(self, data: pd.DataFrame,
                              x_prop: str, y_prop: str,
                              color_prop: Optional[str] = None,
                              figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot molecular properties scatter plot.
        
        Args:
            data: DataFrame containing molecular data
            x_prop: X-axis property
            y_prop: Y-axis property
            color_prop: Color property
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if color_prop and color_prop in data.columns:
            scatter = ax.scatter(data[x_prop], data[y_prop], 
                               c=data[color_prop], cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, label=color_prop)
        else:
            ax.scatter(data[x_prop], data[y_prop], alpha=0.6)
        
        ax.set_xlabel(x_prop)
        ax.set_ylabel(y_prop)
        ax.set_title(f'{x_prop} vs {y_prop}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_molecular_correlation(self, data: pd.DataFrame,
                                  properties: List[str] = None,
                                  figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot molecular properties correlation matrix.
        
        Args:
            data: DataFrame containing molecular data
            properties: List of properties to include
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if properties is None:
            properties = ['mol_weight', 'logp', 'hbd', 'hba', 'tpsa', 'rotatable_bonds']
        
        # Filter available properties
        available_props = [prop for prop in properties if prop in data.columns]
        
        if not available_props:
            raise ValueError("No valid properties found in data")
        
        # Calculate correlation matrix
        corr_matrix = data[available_props].corr()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(len(available_props)))
        ax.set_yticks(range(len(available_props)))
        ax.set_xticklabels(available_props, rotation=45, ha='right')
        ax.set_yticklabels(available_props)
        
        # Add correlation values
        for i in range(len(available_props)):
            for j in range(len(available_props)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black")
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Correlation')
        
        ax.set_title('Molecular Properties Correlation Matrix')
        plt.tight_layout()
        return fig
    

class ADMETVisualizer:
    """ADMET properties visualization utilities."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize ADMET visualizer.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
    def plot_admet_radar(self, data: pd.DataFrame,
                        molecule_id: str,
                        admet_props: List[str] = None,
                        figsize: Tuple[int, int] = (8, 8)) -> plt.Figure:
        """
        Plot ADMET properties radar chart.
        
        Args:
            data: DataFrame containing ADMET data
            molecule_id: Molecule ID to plot
            admet_props: List of ADMET properties
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if admet_props is None:
            admet_props = ['absorption', 'distribution', 'metabolism', 
                          'excretion', 'toxicity']
        
        # Filter data for specific molecule
        molecule_data = data[data['molecule_id'] == molecule_id]
        
        if molecule_data.empty:
            raise ValueError(f"No data found for molecule {molecule_id}")
        
        # Get ADMET values
        values = []
        labels = []
        
        for prop in admet_props:
            if prop in molecule_data.columns:
                values.append(molecule_data[prop].iloc[0])
                labels.append(prop.title())
        
        if not values:
            raise ValueError("No ADMET properties found in data")
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        # Calculate angles
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        # Plot radar chart
        ax.plot(angles, values, 'o-', linewidth=2, label=f'Molecule {molecule_id}')
        ax.fill(angles, values, alpha=0.25)
        
        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        
        ax.set_title(f'ADMET Properties - Molecule {molecule_id}')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_admet_distribution(self, data: pd.DataFrame,
                               admet_props: List[str] = None,
                               figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot ADMET properties distribution.
        
        Args:
            data: DataFrame containing ADMET data
            admet_props: List of ADMET properties
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if admet_props is None:
            admet_props = ['absorption', 'distribution', 'metabolism', 
                          'excretion', 'toxicity']
        
        # Filter available properties
        available_props = [prop for prop in admet_props if prop in data.columns]
        
        if not available_props:
            raise ValueError("No valid ADMET properties found in data")
        
        n_props = len(available_props)
        n_cols = min(3, n_props)
        n_rows = (n_props + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_props == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, prop in enumerate(available_props):
            row = i // n_cols
            col = i % n_cols
            
            if n_rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]
            
            # Plot histogram
            ax.hist(data[prop].dropna(), bins=30, alpha=0.7, edgecolor='black')
            ax.set_title(f'Distribution of {prop.title()}')
            ax.set_xlabel(prop.title())
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_props, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows == 1:
                axes[col].set_visible(False)
            else:
                axes[row, col].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_admet_heatmap(self, data: pd.DataFrame,
                          admet_props: List[str] = None,
                          figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot ADMET properties heatmap.
        
        Args:
            data: DataFrame containing ADMET data
            admet_props: List of ADMET properties
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if admet_props is None:
            admet_props = ['absorption', 'distribution', 'metabolism', 
                          'excretion', 'toxicity']
        
        # Filter available properties
        available_props = [prop for prop in admet_props if prop in data.columns]
        
        if not available_props:
            raise ValueError("No valid ADMET properties found in data")
        
        # Create heatmap data
        heatmap_data = data[available_props].T
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(data)))
        ax.set_yticks(range(len(available_props)))
        ax.set_yticklabels([prop.title() for prop in available_props])
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='ADMET Score')
        
        ax.set_title('ADMET Properties Heatmap')
        ax.set_xlabel('Molecule Index')
        
        plt.tight_layout()
        return fig


class CNSMPOVisualizer:
    """CNS-MPO visualization utilities."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize CNS-MPO visualizer.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
    def plot_cns_mpo_scores(self, data: pd.DataFrame,
                           figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot CNS-MPO scores distribution.
        
        Args:
            data: DataFrame containing CNS-MPO data
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if 'cns_mpo_score' not in data.columns:
            raise ValueError("CNS-MPO score column not found in data")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        ax1.hist(data['cns_mpo_score'].dropna(), bins=30, alpha=0.7, edgecolor='black')
        ax1.set_title('CNS-MPO Score Distribution')
        ax1.set_xlabel('CNS-MPO Score')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(data['cns_mpo_score'].dropna())
        ax2.set_title('CNS-MPO Score Box Plot')
        ax2.set_ylabel('CNS-MPO Score')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_cns_mpo_vs_properties(self, data: pd.DataFrame,
                                  properties: List[str] = None,
                                  figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot CNS-MPO scores vs molecular properties.
        
        Args:
            data: DataFrame containing CNS-MPO and molecular data
            properties: List of properties to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if 'cns_mpo_score' not in data.columns:
            raise ValueError("CNS-MPO score column not found in data")
        
        if properties is None:
            properties = ['mol_weight', 'logp', 'hbd', 'hba', 'tpsa']
        
        # Filter available properties
        available_props = [prop for prop in properties if prop in data.columns]
        
        if not available_props:
            raise ValueError("No valid properties found in data")
        
        n_props = len(available_props)
        n_cols = min(3, n_props)
        n_rows = (n_props + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_props == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, prop in enumerate(available_props):
            row = i // n_cols
            col = i % n_cols
            
            if n_rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]
            
            # Scatter plot
            ax.scatter(data[prop], data['cns_mpo_score'], alpha=0.6)
            ax.set_xlabel(prop)
            ax.set_ylabel('CNS-MPO Score')
            ax.set_title(f'CNS-MPO Score vs {prop}')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_props, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows == 1:
                axes[col].set_visible(False)
        else:
                axes[row, col].set_visible(False)
        
        plt.tight_layout()
        return fig


class ScaffoldVisualizer:
    """Scaffold analysis visualization utilities."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize scaffold visualizer.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
    def plot_scaffold_distribution(self, data: pd.DataFrame,
                                  figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot scaffold distribution.
        
        Args:
            data: DataFrame containing scaffold data
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if 'scaffold_type' not in data.columns:
            raise ValueError("Scaffold type column not found in data")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Count scaffold types
        scaffold_counts = data['scaffold_type'].value_counts()
        
        # Bar plot
        scaffold_counts.plot(kind='bar', ax=ax1)
        ax1.set_title('Scaffold Type Distribution')
        ax1.set_xlabel('Scaffold Type')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Pie chart
        scaffold_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%')
        ax2.set_title('Scaffold Type Distribution')
        ax2.set_ylabel('')
        
        plt.tight_layout()
        return fig
    
    def plot_scaffold_properties(self, data: pd.DataFrame,
                                properties: List[str] = None,
                                figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot scaffold properties.
        
        Args:
            data: DataFrame containing scaffold data
            properties: List of properties to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if 'scaffold_type' not in data.columns:
            raise ValueError("Scaffold type column not found in data")
        
        if properties is None:
            properties = ['mol_weight', 'logp', 'hbd', 'hba', 'tpsa']
        
        # Filter available properties
        available_props = [prop for prop in properties if prop in data.columns]
        
        if not available_props:
            raise ValueError("No valid properties found in data")
        
        n_props = len(available_props)
        n_cols = min(3, n_props)
        n_rows = (n_props + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_props == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, prop in enumerate(available_props):
            row = i // n_cols
            col = i % n_cols
            
            if n_rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]
            
            # Box plot by scaffold type
            data.boxplot(column=prop, by='scaffold_type', ax=ax)
            ax.set_title(f'{prop} by Scaffold Type')
            ax.set_xlabel('Scaffold Type')
            ax.set_ylabel(prop)
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_props, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows == 1:
                axes[col].set_visible(False)
            else:
                axes[row, col].set_visible(False)
        
        plt.tight_layout()
        return fig
    

class TrainingVisualizer:
    """Training progress visualization utilities."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize training visualizer.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
    def plot_training_curves(self, training_data: Dict[str, List[float]],
                           figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot training curves.
        
        Args:
            training_data: Dictionary containing training metrics
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Training loss
        if 'train_loss' in training_data:
            axes[0, 0].plot(training_data['train_loss'], label='Training Loss')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Validation loss
        if 'val_loss' in training_data:
            axes[0, 1].plot(training_data['val_loss'], label='Validation Loss')
            axes[0, 1].set_title('Validation Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        # Training accuracy
        if 'train_acc' in training_data:
            axes[1, 0].plot(training_data['train_acc'], label='Training Accuracy')
            axes[1, 0].set_title('Training Accuracy')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Validation accuracy
        if 'val_acc' in training_data:
            axes[1, 1].plot(training_data['val_acc'], label='Validation Accuracy')
            axes[1, 1].set_title('Validation Accuracy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_learning_rate_schedule(self, lr_schedule: List[float],
                                   figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot learning rate schedule.
        
        Args:
            lr_schedule: List of learning rates
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(lr_schedule)
        ax.set_title('Learning Rate Schedule')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def plot_model_comparison(self, model_results: Dict[str, Dict[str, float]],
                             figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
    Plot model comparison.
    
    Args:
            model_results: Dictionary containing model results
            figsize: Figure size
        
    Returns:
            Matplotlib figure
        """
        models = list(model_results.keys())
        metrics = list(model_results[models[0]].keys())
        
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, metric in enumerate(metrics):
            row = i // n_cols
            col = i % n_cols
            
            if n_rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]
            
            # Get metric values for all models
            values = [model_results[model][metric] for model in models]
            
            # Bar plot
            ax.bar(models, values)
            ax.set_title(f'{metric.title()} Comparison')
            ax.set_ylabel(metric.title())
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
    
        # Hide empty subplots
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows == 1:
                axes[col].set_visible(False)
            else:
                axes[row, col].set_visible(False)
        plt.tight_layout()
        return fig
