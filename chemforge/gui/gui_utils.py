"""
GUI utilities for ChemForge.

This module provides utility functions for GUI applications
including data processing, visualization, and file handling.
"""

import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any, Tuple
import base64
import io
from pathlib import Path

from chemforge.utils.logging_utils import Logger


class GUIUtils:
    """
    Utility class for GUI applications.
    
    This class provides utility functions for GUI applications
    including data processing, visualization, and file handling.
    """
    
    def __init__(self):
        """Initialize GUI utilities."""
        self.logger = Logger('gui_utils')
    
    def process_uploaded_file(self, contents: str, filename: str) -> List[str]:
        """
        Process uploaded file and extract molecules.
        
        Args:
            contents: File contents as base64 string
            filename: Name of the uploaded file
            
        Returns:
            List of molecules (SMILES strings)
        """
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            
            if filename.endswith('.csv'):
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                if 'smiles' in df.columns:
                    molecules = df['smiles'].tolist()
                else:
                    molecules = df.iloc[:, 0].tolist()
            elif filename.endswith('.txt'):
                molecules = decoded.decode('utf-8').split('\n')
                molecules = [mol.strip() for mol in molecules if mol.strip()]
            elif filename.endswith('.json'):
                data = json.loads(decoded.decode('utf-8'))
                if isinstance(data, list):
                    molecules = data
                elif isinstance(data, dict) and 'molecules' in data:
                    molecules = data['molecules']
                else:
                    molecules = []
            else:
                molecules = []
            
            self.logger.info(f"Processed {len(molecules)} molecules from {filename}")
            return molecules
            
        except Exception as e:
            self.logger.error(f"Error processing file {filename}: {str(e)}")
            return []
    
    def create_molecular_properties_plot(self, df: pd.DataFrame) -> go.Figure:
        """
        Create molecular properties visualization.
        
        Args:
            df: DataFrame with molecular properties
            
        Returns:
            Plotly figure
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=["MW vs LogP", "HBD vs HBA", "TPSA Distribution", "MW Distribution"]
            )
            
            # MW vs LogP scatter plot
            fig.add_trace(
                go.Scatter(
                    x=df['LogP'], y=df['MW'],
                    mode='markers',
                    name='MW vs LogP',
                    text=df['SMILES'],
                    hovertemplate='<b>%{text}</b><br>LogP: %{x}<br>MW: %{y}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # HBD vs HBA scatter plot
            fig.add_trace(
                go.Scatter(
                    x=df['HBA'], y=df['HBD'],
                    mode='markers',
                    name='HBD vs HBA',
                    text=df['SMILES'],
                    hovertemplate='<b>%{text}</b><br>HBA: %{x}<br>HBD: %{y}<extra></extra>'
                ),
                row=1, col=2
            )
            
            # TPSA distribution
            fig.add_trace(
                go.Histogram(
                    x=df['TPSA'],
                    name='TPSA Distribution',
                    nbinsx=20
                ),
                row=2, col=1
            )
            
            # MW distribution
            fig.add_trace(
                go.Histogram(
                    x=df['MW'],
                    name='MW Distribution',
                    nbinsx=20
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                height=600,
                showlegend=False,
                title_text="Molecular Properties Analysis"
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating properties plot: {str(e)}")
            return go.Figure()
    
    def create_similarity_heatmap(self, similarity_matrix: np.ndarray, 
                                molecule_labels: List[str]) -> go.Figure:
        """
        Create molecular similarity heatmap.
        
        Args:
            similarity_matrix: Similarity matrix
            molecule_labels: List of molecule labels
            
        Returns:
            Plotly figure
        """
        try:
            fig = px.imshow(
                similarity_matrix,
                labels=dict(x="Molecule", y="Molecule", color="Similarity"),
                x=molecule_labels,
                y=molecule_labels,
                title="Molecular Similarity Matrix",
                color_continuous_scale="Blues"
            )
            
            fig.update_layout(
                width=800,
                height=600
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating similarity heatmap: {str(e)}")
            return go.Figure()
    
    def create_prediction_results_plot(self, df: pd.DataFrame) -> go.Figure:
        """
        Create prediction results visualization.
        
        Args:
            df: DataFrame with prediction results
            
        Returns:
            Plotly figure
        """
        try:
            # Box plot of predictions by target
            fig = px.box(
                df, 
                x="Target", 
                y="pIC50",
                title="Prediction Results by Target",
                color="Target"
            )
            
            fig.update_layout(
                xaxis_title="Target",
                yaxis_title="pIC50",
                height=500
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating prediction plot: {str(e)}")
            return go.Figure()
    
    def create_admet_radar_chart(self, admet_data: Dict[str, float]) -> go.Figure:
        """
        Create ADMET radar chart.
        
        Args:
            admet_data: Dictionary with ADMET properties
            
        Returns:
            Plotly figure
        """
        try:
            categories = list(admet_data.keys())
            values = list(admet_data.values())
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='ADMET Properties'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(values) * 1.1]
                    )),
                showlegend=True,
                title="ADMET Properties Radar Chart"
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating ADMET radar chart: {str(e)}")
            return go.Figure()
    
    def create_generation_history_plot(self, history: Dict[str, List[float]]) -> go.Figure:
        """
        Create generation history plot.
        
        Args:
            history: Dictionary with generation history
            
        Returns:
            Plotly figure
        """
        try:
            fig = go.Figure()
            
            for key, values in history.items():
                fig.add_trace(go.Scatter(
                    y=values,
                    mode='lines+markers',
                    name=key
                ))
            
            fig.update_layout(
                title="Generation History",
                xaxis_title="Generation",
                yaxis_title="Value",
                height=500
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating generation history plot: {str(e)}")
            return go.Figure()
    
    def create_descriptors_correlation_plot(self, df: pd.DataFrame) -> go.Figure:
        """
        Create molecular descriptors correlation plot.
        
        Args:
            df: DataFrame with molecular descriptors
            
        Returns:
            Plotly figure
        """
        try:
            # Select numeric columns only
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                return go.Figure()
            
            # Calculate correlation matrix
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                title="Molecular Descriptors Correlation Matrix",
                color_continuous_scale="RdBu",
                aspect="auto"
            )
            
            fig.update_layout(
                width=800,
                height=600
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating descriptors correlation plot: {str(e)}")
            return go.Figure()
    
    def create_ensemble_prediction_plot(self, predictions: Dict[str, Dict[str, float]]) -> go.Figure:
        """
        Create ensemble prediction visualization.
        
        Args:
            predictions: Dictionary with ensemble predictions
            
        Returns:
            Plotly figure
        """
        try:
            # Prepare data for plotting
            models = list(predictions.keys())
            targets = list(next(iter(predictions.values())).keys())
            
            fig = go.Figure()
            
            for target in targets:
                values = [predictions[model][target] for model in models]
                fig.add_trace(go.Bar(
                    x=models,
                    y=values,
                    name=target
                ))
            
            fig.update_layout(
                title="Ensemble Predictions by Model",
                xaxis_title="Model",
                yaxis_title="pIC50",
                barmode='group',
                height=500
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating ensemble prediction plot: {str(e)}")
            return go.Figure()
    
    def create_cns_mpo_plot(self, cns_mpo_scores: List[float]) -> go.Figure:
        """
        Create CNS-MPO scores visualization.
        
        Args:
            cns_mpo_scores: List of CNS-MPO scores
            
        Returns:
            Plotly figure
        """
        try:
            fig = go.Figure()
            
            # Histogram of CNS-MPO scores
            fig.add_trace(go.Histogram(
                x=cns_mpo_scores,
                nbinsx=20,
                name='CNS-MPO Scores'
            ))
            
            # Add vertical line for mean
            mean_score = np.mean(cns_mpo_scores)
            fig.add_vline(
                x=mean_score,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean_score:.2f}"
            )
            
            fig.update_layout(
                title="CNS-MPO Scores Distribution",
                xaxis_title="CNS-MPO Score",
                yaxis_title="Count",
                height=500
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating CNS-MPO plot: {str(e)}")
            return go.Figure()
    
    def create_optimization_progress_plot(self, progress_data: Dict[str, List[float]]) -> go.Figure:
        """
        Create optimization progress visualization.
        
        Args:
            progress_data: Dictionary with optimization progress data
            
        Returns:
            Plotly figure
        """
        try:
            fig = go.Figure()
            
            for key, values in progress_data.items():
                fig.add_trace(go.Scatter(
                    y=values,
                    mode='lines+markers',
                    name=key
                ))
            
            fig.update_layout(
                title="Optimization Progress",
                xaxis_title="Iteration",
                yaxis_title="Value",
                height=500
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating optimization progress plot: {str(e)}")
            return go.Figure()
    
    def export_results_to_csv(self, data: Dict[str, Any], filename: str) -> str:
        """
        Export results to CSV file.
        
        Args:
            data: Data to export
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        try:
            # Convert data to DataFrame
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame(data)
            
            # Export to CSV
            output_path = Path(filename)
            df.to_csv(output_path, index=False)
            
            self.logger.info(f"Exported results to {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error exporting results: {str(e)}")
            return ""
    
    def export_results_to_json(self, data: Dict[str, Any], filename: str) -> str:
        """
        Export results to JSON file.
        
        Args:
            data: Data to export
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        try:
            output_path = Path(filename)
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.logger.info(f"Exported results to {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error exporting results: {str(e)}")
            return ""
    
    def create_summary_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create summary statistics from data.
        
        Args:
            data: List of data dictionaries
            
        Returns:
            Dictionary with summary statistics
        """
        try:
            if not data:
                return {}
            
            df = pd.DataFrame(data)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            summary = {}
            for col in numeric_cols:
                summary[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'median': df[col].median()
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error creating summary statistics: {str(e)}")
            return {}
    
    def validate_smiles(self, smiles: str) -> bool:
        """
        Validate SMILES string.
        
        Args:
            smiles: SMILES string to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Basic SMILES validation
            if not smiles or len(smiles) < 1:
                return False
            
            # Check for basic SMILES characters
            valid_chars = set('CNOSFClBrI()[]=#-+\\/')
            if not all(c in valid_chars or c.isdigit() for c in smiles):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating SMILES: {str(e)}")
            return False
    
    def format_molecular_data(self, molecules: List[str]) -> pd.DataFrame:
        """
        Format molecular data for display.
        
        Args:
            molecules: List of SMILES strings
            
        Returns:
            Formatted DataFrame
        """
        try:
            data = []
            for i, smiles in enumerate(molecules):
                data.append({
                    'ID': f"Molecule_{i+1}",
                    'SMILES': smiles,
                    'Length': len(smiles),
                    'Valid': self.validate_smiles(smiles)
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            self.logger.error(f"Error formatting molecular data: {str(e)}")
            return pd.DataFrame()
    
    def create_download_link(self, data: Any, filename: str, file_type: str = 'csv') -> str:
        """
        Create download link for data.
        
        Args:
            data: Data to download
            filename: Filename for download
            file_type: Type of file ('csv' or 'json')
            
        Returns:
            Download link
        """
        try:
            if file_type == 'csv':
                if isinstance(data, dict):
                    df = pd.DataFrame([data])
                elif isinstance(data, list):
                    df = pd.DataFrame(data)
                else:
                    df = pd.DataFrame(data)
                
                csv_string = df.to_csv(index=False)
                b64 = base64.b64encode(csv_string.encode()).decode()
                
            elif file_type == 'json':
                json_string = json.dumps(data, indent=2, default=str)
                b64 = base64.b64encode(json_string.encode()).decode()
            
            else:
                return ""
            
            href = f'<a href="data:file/{file_type};base64,{b64}" download="{filename}">Download {filename}</a>'
            return href
            
        except Exception as e:
            self.logger.error(f"Error creating download link: {str(e)}")
            return ""
