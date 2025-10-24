"""
Model evaluator for pre-trained models.

This module provides functionality for evaluating pre-trained models
and comparing their performance across different metrics.
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from chemforge.models.transformer_model import TransformerModel
from chemforge.models.gnn_model import GNNModel
from chemforge.models.ensemble_model import EnsembleModel
from chemforge.training.metrics import MultiTargetMetrics
from chemforge.utils.logging_utils import Logger


class ModelEvaluator:
    """
    Model evaluator for pre-trained models.
    
    This class provides comprehensive evaluation of pre-trained models
    across multiple metrics and visualization capabilities.
    """
    
    def __init__(
        self,
        output_dir: str = "./evaluation_results",
        log_dir: str = "./logs"
    ):
        """
        Initialize the model evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
            log_dir: Directory for logs
        """
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.logger = Logger('model_evaluator', log_dir=str(self.log_dir))
        
        # Evaluation results storage
        self.evaluation_results = {}
        self.comparison_results = {}
        
        self.logger.info("ModelEvaluator initialized")
    
    def evaluate_model(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        model_name: str,
        targets: List[str],
        device: torch.device,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a single model on test data.
        
        Args:
            model: The model to evaluate
            test_loader: DataLoader for test data
            model_name: Name of the model
            targets: List of target names
            device: Device to use for evaluation
            save_results: Whether to save evaluation results
            
        Returns:
            Dictionary containing evaluation results
        """
        self.logger.info(f"Evaluating model: {model_name}")
        
        model.eval()
        all_predictions = []
        all_targets = []
        all_losses = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Extract features and targets
                features = batch['features'].to(device)
                targets_batch = batch['targets'].to(device)
                
                # Get predictions
                predictions = model(features)
                
                # Calculate loss
                loss = nn.MSELoss()(predictions, targets_batch)
                all_losses.append(loss.item())
                
                # Store predictions and targets
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets_batch.cpu().numpy())
        
        # Concatenate all predictions and targets
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_predictions, all_targets, targets)
        
        # Calculate target-specific metrics
        target_metrics = self._calculate_target_metrics(all_predictions, all_targets, targets)
        
        # Prepare evaluation results
        evaluation_results = {
            'model_name': model_name,
            'overall_metrics': metrics,
            'target_metrics': target_metrics,
            'predictions': all_predictions,
            'targets': all_targets,
            'losses': all_losses,
            'evaluation_metadata': {
                'num_samples': len(all_predictions),
                'num_targets': len(targets),
                'targets': targets
            }
        }
        
        # Store results
        self.evaluation_results[model_name] = evaluation_results
        
        # Save results if requested
        if save_results:
            self._save_evaluation_results(evaluation_results, model_name)
        
        self.logger.info(f"Evaluation completed for {model_name}")
        self.logger.info(f"Overall R²: {metrics['r2_score']:.4f}")
        self.logger.info(f"Overall RMSE: {metrics['rmse']:.4f}")
        self.logger.info(f"Overall MAE: {metrics['mae']:.4f}")
        
        return evaluation_results
    
    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        target_names: List[str]
    ) -> Dict[str, float]:
        """Calculate overall metrics for the model."""
        # Flatten predictions and targets
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        # Calculate regression metrics
        mse = mean_squared_error(target_flat, pred_flat)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(target_flat, pred_flat)
        r2 = r2_score(target_flat, pred_flat)
        
        # Calculate correlation
        correlation = np.corrcoef(pred_flat, target_flat)[0, 1]
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'correlation': correlation
        }
    
    def _calculate_target_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        target_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for each target individually."""
        target_metrics = {}
        
        for i, target_name in enumerate(target_names):
            pred_target = predictions[:, i]
            target_target = targets[:, i]
            
            # Calculate metrics for this target
            mse = mean_squared_error(target_target, pred_target)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(target_target, pred_target)
            r2 = r2_score(target_target, pred_target)
            correlation = np.corrcoef(pred_target, target_target)[0, 1]
            
            target_metrics[target_name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2_score': r2,
                'correlation': correlation
            }
        
        return target_metrics
    
    def compare_models(
        self,
        model_results: Dict[str, Dict[str, Any]],
        save_comparison: bool = True
    ) -> Dict[str, Any]:
        """
        Compare multiple models across different metrics.
        
        Args:
            model_results: Dictionary of model evaluation results
            save_comparison: Whether to save comparison results
            
        Returns:
            Dictionary containing comparison results
        """
        self.logger.info("Comparing multiple models")
        
        # Extract metrics for comparison
        comparison_data = []
        for model_name, results in model_results.items():
            overall_metrics = results['overall_metrics']
            comparison_data.append({
                'model_name': model_name,
                'r2_score': overall_metrics['r2_score'],
                'rmse': overall_metrics['rmse'],
                'mae': overall_metrics['mae'],
                'correlation': overall_metrics['correlation']
            })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Calculate ranking
        ranking = self._calculate_model_ranking(comparison_df)
        
        # Prepare comparison results
        comparison_results = {
            'comparison_data': comparison_data,
            'comparison_df': comparison_df,
            'ranking': ranking,
            'best_model': ranking[0]['model_name'] if ranking else None,
            'comparison_metadata': {
                'num_models': len(model_results),
                'models': list(model_results.keys())
            }
        }
        
        # Store results
        self.comparison_results = comparison_results
        
        # Save comparison if requested
        if save_comparison:
            self._save_comparison_results(comparison_results)
        
        self.logger.info(f"Model comparison completed. Best model: {comparison_results['best_model']}")
        
        return comparison_results
    
    def _calculate_model_ranking(self, comparison_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Calculate model ranking based on multiple metrics."""
        # Normalize metrics (higher is better for R² and correlation, lower is better for RMSE and MAE)
        normalized_df = comparison_df.copy()
        normalized_df['r2_score_norm'] = normalized_df['r2_score']
        normalized_df['correlation_norm'] = normalized_df['correlation']
        normalized_df['rmse_norm'] = 1 / (1 + normalized_df['rmse'])  # Inverse for ranking
        normalized_df['mae_norm'] = 1 / (1 + normalized_df['mae'])  # Inverse for ranking
        
        # Calculate composite score
        normalized_df['composite_score'] = (
            normalized_df['r2_score_norm'] * 0.3 +
            normalized_df['correlation_norm'] * 0.3 +
            normalized_df['rmse_norm'] * 0.2 +
            normalized_df['mae_norm'] * 0.2
        )
        
        # Sort by composite score
        ranked_df = normalized_df.sort_values('composite_score', ascending=False)
        
        # Create ranking list
        ranking = []
        for i, (_, row) in enumerate(ranked_df.iterrows()):
            ranking.append({
                'rank': i + 1,
                'model_name': row['model_name'],
                'composite_score': row['composite_score'],
                'r2_score': row['r2_score'],
                'rmse': row['rmse'],
                'mae': row['mae'],
                'correlation': row['correlation']
            })
        
        return ranking
    
    def create_evaluation_plots(
        self,
        model_results: Dict[str, Dict[str, Any]],
        save_plots: bool = True
    ) -> Dict[str, str]:
        """
        Create evaluation plots for model comparison.
        
        Args:
            model_results: Dictionary of model evaluation results
            save_plots: Whether to save the plots
            
        Returns:
            Dictionary containing paths to saved plots
        """
        self.logger.info("Creating evaluation plots")
        
        plot_paths = {}
        
        # 1. Model comparison bar chart
        comparison_plot = self._create_model_comparison_plot(model_results)
        if save_plots:
            comparison_path = self.output_dir / "model_comparison.png"
            comparison_plot.savefig(comparison_path, dpi=300, bbox_inches='tight')
            plot_paths['comparison'] = str(comparison_path)
        
        # 2. Prediction vs target scatter plots
        scatter_plots = self._create_prediction_scatter_plots(model_results)
        if save_plots:
            for model_name, fig in scatter_plots.items():
                scatter_path = self.output_dir / f"{model_name}_predictions.png"
                fig.savefig(scatter_path, dpi=300, bbox_inches='tight')
                plot_paths[f'{model_name}_scatter'] = str(scatter_path)
        
        # 3. Target-specific performance plots
        target_plots = self._create_target_performance_plots(model_results)
        if save_plots:
            for target_name, fig in target_plots.items():
                target_path = self.output_dir / f"target_{target_name}_performance.png"
                fig.savefig(target_path, dpi=300, bbox_inches='tight')
                plot_paths[f'target_{target_name}'] = str(target_path)
        
        # 4. Interactive comparison plot
        interactive_plot = self._create_interactive_comparison_plot(model_results)
        if save_plots:
            interactive_path = self.output_dir / "interactive_comparison.html"
            interactive_plot.write_html(interactive_path)
            plot_paths['interactive'] = str(interactive_path)
        
        self.logger.info(f"Created {len(plot_paths)} evaluation plots")
        
        return plot_paths
    
    def _create_model_comparison_plot(self, model_results: Dict[str, Dict[str, Any]]) -> plt.Figure:
        """Create a bar chart comparing models across metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # Extract data for plotting
        model_names = list(model_results.keys())
        metrics = ['r2_score', 'rmse', 'mae', 'correlation']
        metric_labels = ['R² Score', 'RMSE', 'MAE', 'Correlation']
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[i//2, i%2]
            values = [model_results[name]['overall_metrics'][metric] for name in model_names]
            
            bars = ax.bar(model_names, values, alpha=0.7)
            ax.set_title(f'{label} Comparison')
            ax.set_ylabel(label)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def _create_prediction_scatter_plots(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, plt.Figure]:
        """Create scatter plots of predictions vs targets for each model."""
        scatter_plots = {}
        
        for model_name, results in model_results.items():
            predictions = results['predictions']
            targets = results['targets']
            
            # Flatten for overall scatter plot
            pred_flat = predictions.flatten()
            target_flat = targets.flatten()
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(target_flat, pred_flat, alpha=0.6, s=20)
            
            # Add diagonal line
            min_val = min(target_flat.min(), pred_flat.min())
            max_val = max(target_flat.max(), pred_flat.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            # Calculate and display R²
            r2 = results['overall_metrics']['r2_score']
            ax.set_title(f'{model_name} - Predictions vs Targets (R² = {r2:.3f})')
            ax.set_xlabel('True Values')
            ax.set_ylabel('Predicted Values')
            ax.grid(True, alpha=0.3)
            
            scatter_plots[model_name] = fig
        
        return scatter_plots
    
    def _create_target_performance_plots(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, plt.Figure]:
        """Create performance plots for each target."""
        target_plots = {}
        
        # Get target names from first model
        first_model = list(model_results.keys())[0]
        target_names = list(model_results[first_model]['target_metrics'].keys())
        
        for target_name in target_names:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Target Performance: {target_name}', fontsize=16)
            
            # Extract metrics for this target
            model_names = list(model_results.keys())
            metrics = ['r2_score', 'rmse', 'mae', 'correlation']
            metric_labels = ['R² Score', 'RMSE', 'MAE', 'Correlation']
            
            for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                ax = axes[i//2, i%2]
                values = [model_results[name]['target_metrics'][target_name][metric] for name in model_names]
                
                bars = ax.bar(model_names, values, alpha=0.7)
                ax.set_title(f'{label} for {target_name}')
                ax.set_ylabel(label)
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            target_plots[target_name] = fig
        
        return target_plots
    
    def _create_interactive_comparison_plot(self, model_results: Dict[str, Dict[str, Any]]) -> go.Figure:
        """Create an interactive comparison plot using Plotly."""
        # Prepare data for plotting
        model_names = list(model_results.keys())
        metrics = ['r2_score', 'rmse', 'mae', 'correlation']
        metric_labels = ['R² Score', 'RMSE', 'MAE', 'Correlation']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=metric_labels,
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = [model_results[name]['overall_metrics'][metric] for name in model_names]
            
            fig.add_trace(
                go.Bar(
                    x=model_names,
                    y=values,
                    name=label,
                    text=[f'{v:.3f}' for v in values],
                    textposition='auto'
                ),
                row=i//2 + 1, col=i%2 + 1
            )
        
        fig.update_layout(
            title="Interactive Model Performance Comparison",
            showlegend=False,
            height=600
        )
        
        return fig
    
    def _save_evaluation_results(self, results: Dict[str, Any], model_name: str):
        """Save evaluation results to file."""
        results_path = self.output_dir / f"evaluation_{model_name}.json"
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = self._prepare_results_for_json(results)
            json.dump(json_results, f, indent=2)
        
        self.logger.info(f"Saved evaluation results to {results_path}")
    
    def _save_comparison_results(self, results: Dict[str, Any]):
        """Save comparison results to file."""
        comparison_path = self.output_dir / "model_comparison.json"
        with open(comparison_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Saved comparison results to {comparison_path}")
    
    def _prepare_results_for_json(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare results for JSON serialization."""
        json_results = results.copy()
        
        # Convert numpy arrays to lists
        if 'predictions' in json_results:
            json_results['predictions'] = json_results['predictions'].tolist()
        if 'targets' in json_results:
            json_results['targets'] = json_results['targets'].tolist()
        if 'losses' in json_results:
            json_results['losses'] = json_results['losses']
        
        return json_results
    
    def generate_evaluation_report(
        self,
        model_results: Dict[str, Dict[str, Any]],
        comparison_results: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            model_results: Dictionary of model evaluation results
            comparison_results: Model comparison results
            output_path: Path to save the report
            
        Returns:
            Path to the generated report
        """
        if output_path is None:
            output_path = self.output_dir / "evaluation_report.md"
        else:
            output_path = Path(output_path)
        
        # Generate report content
        report_content = self._generate_report_content(model_results, comparison_results)
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"Generated evaluation report: {output_path}")
        
        return str(output_path)
    
    def _generate_report_content(
        self,
        model_results: Dict[str, Dict[str, Any]],
        comparison_results: Dict[str, Any]
    ) -> str:
        """Generate the content for the evaluation report."""
        report_lines = [
            "# Model Evaluation Report",
            "",
            f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            ""
        ]
        
        # Add executive summary
        if comparison_results and 'best_model' in comparison_results:
            best_model = comparison_results['best_model']
            report_lines.extend([
                f"- **Best Model:** {best_model}",
                f"- **Number of Models Evaluated:** {len(model_results)}",
                f"- **Evaluation Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}",
                ""
            ])
        
        # Add model comparison table
        if comparison_results and 'comparison_df' in comparison_results:
            report_lines.extend([
                "## Model Performance Comparison",
                "",
                "| Model | R² Score | RMSE | MAE | Correlation |",
                "|-------|----------|------|-----|-------------|"
            ])
            
            for _, row in comparison_results['comparison_df'].iterrows():
                report_lines.append(
                    f"| {row['model_name']} | {row['r2_score']:.4f} | "
                    f"{row['rmse']:.4f} | {row['mae']:.4f} | {row['correlation']:.4f} |"
                )
            
            report_lines.append("")
        
        # Add detailed results for each model
        report_lines.extend([
            "## Detailed Model Results",
            ""
        ])
        
        for model_name, results in model_results.items():
            report_lines.extend([
                f"### {model_name}",
                "",
                "#### Overall Performance",
                f"- **R² Score:** {results['overall_metrics']['r2_score']:.4f}",
                f"- **RMSE:** {results['overall_metrics']['rmse']:.4f}",
                f"- **MAE:** {results['overall_metrics']['mae']:.4f}",
                f"- **Correlation:** {results['overall_metrics']['correlation']:.4f}",
                ""
            ])
            
            # Add target-specific results
            if 'target_metrics' in results:
                report_lines.extend([
                    "#### Target-Specific Performance",
                    "",
                    "| Target | R² Score | RMSE | MAE | Correlation |",
                    "|--------|----------|------|-----|-------------|"
                ])
                
                for target_name, target_metrics in results['target_metrics'].items():
                    report_lines.append(
                        f"| {target_name} | {target_metrics['r2_score']:.4f} | "
                        f"{target_metrics['rmse']:.4f} | {target_metrics['mae']:.4f} | "
                        f"{target_metrics['correlation']:.4f} |"
                    )
                
                report_lines.append("")
        
        # Add conclusions
        report_lines.extend([
            "## Conclusions",
            "",
            "### Key Findings",
            "- Model performance varies across different targets",
            "- Some models may perform better on specific target types",
            "- Consider ensemble approaches for improved performance",
            "",
            "### Recommendations",
            "- Use the best performing model for production",
            "- Consider target-specific model selection",
            "- Implement ensemble methods for critical applications",
            ""
        ])
        
        return "\n".join(report_lines)
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get a summary of all evaluation results."""
        summary = {
            'evaluation_results': self.evaluation_results,
            'comparison_results': self.comparison_results,
            'num_models_evaluated': len(self.evaluation_results),
            'evaluation_metadata': {
                'output_dir': str(self.output_dir),
                'log_dir': str(self.log_dir)
            }
        }
        
        return summary
