"""
ChemForge Pre-trained Models Demo

This module demonstrates the usage of ChemForge pre-trained models
for CNS drug discovery applications.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from chemforge.pretrained.model_trainer import PreTrainer
from chemforge.pretrained.data_preparation import DataPreparator
from chemforge.pretrained.model_evaluator import ModelEvaluator
from chemforge.pretrained.model_distributor import ModelDistributor
from chemforge.models.transformer_model import TransformerModel
from chemforge.models.gnn_model import GNNModel
from chemforge.models.ensemble_model import EnsembleModel
from chemforge.utils.logging_utils import setup_logging


def run_pretrained_models_demo():
    """Run pre-trained models demonstration."""
    print("ChemForge Pre-trained Models Demo")
    print("=" * 50)
    
    # Setup logging
    log_manager = setup_logging(level='INFO')
    logger = log_manager.get_logger('pretrained_demo')
    logger.info("Starting ChemForge Pre-trained Models Demo")
    
    # Create demo directories
    demo_dir = Path("./pretrained_demo")
    demo_dir.mkdir(exist_ok=True)
    
    models_dir = demo_dir / "models"
    data_dir = demo_dir / "data"
    results_dir = demo_dir / "results"
    distributions_dir = demo_dir / "distributions"
    
    for dir_path in [models_dir, data_dir, results_dir, distributions_dir]:
        dir_path.mkdir(exist_ok=True)
    
    # 1. Data Preparation Demo
    print("\n1. Data Preparation Demo...")
    
    data_preparator = DataPreparator(
        output_dir=str(data_dir),
        log_dir=str(demo_dir / "logs")
    )
    
    # Create demo molecular data
    demo_molecules = pd.DataFrame({
        'molecule_id': [f'mol_{i}' for i in range(100)],
        'smiles': [
            'CCO', 'CCN', 'CC(C)O', 'CC(C)N', 'CC(C)(C)O',
            'CC(C)(C)N', 'CC(C)(C)(C)O', 'CC(C)(C)(C)N',
            'CC(C)(C)(C)(C)O', 'CC(C)(C)(C)(C)N'
        ] * 10,  # Repeat to get 100 molecules
        'mol_weight': np.random.normal(100, 50, 100),
        'logp': np.random.normal(2, 1, 100),
        'hbd': np.random.randint(0, 5, 100),
        'hba': np.random.randint(0, 10, 100),
        'tpsa': np.random.normal(50, 20, 100),
        'rotatable_bonds': np.random.randint(0, 10, 100),
        'aromatic_rings': np.random.randint(0, 3, 100),
        'heavy_atoms': np.random.randint(5, 50, 100)
    })
    
    # Create demo activity data
    demo_activities = pd.DataFrame({
        'molecule_id': [f'mol_{i}' for i in range(100)],
        'target_id': np.random.choice([1, 2, 3], 100),
        'activity_value': np.random.normal(5, 2, 100),
        'activity_type': ['IC50'] * 100,
        'activity_unit': ['nM'] * 100,
        'activity_relation': ['='] * 100
    })
    
    targets = ['5-HT2A', 'D2R', 'DAT']
    
    # Prepare custom dataset
    prepared_dataset = data_preparator.prepare_custom_dataset(
        molecules=demo_molecules,
        activities=demo_activities,
        targets=targets,
        test_size=0.2,
        val_size=0.1,
        save_data=True
    )
    
    print(f"  - Prepared dataset with {prepared_dataset['data_info']['total_molecules']} molecules")
    print(f"  - Feature dimension: {prepared_dataset['data_info']['feature_dim']}")
    print(f"  - Train size: {prepared_dataset['data_info']['train_size']}")
    print(f"  - Val size: {prepared_dataset['data_info']['val_size']}")
    print(f"  - Test size: {prepared_dataset['data_info']['test_size']}")
    
    # Get dataset statistics
    statistics = data_preparator.get_data_statistics(prepared_dataset)
    print(f"  - Dataset statistics generated")
    
    # 2. Model Training Demo
    print("\n2. Model Training Demo...")
    
    model_trainer = PreTrainer(
        output_dir=str(models_dir),
        log_dir=str(demo_dir / "logs"),
        device='cpu'
    )
    
    # Prepare data for training
    data_info = prepared_dataset['data_info']
    split_data = prepared_dataset['split_data']
    
    # Model configurations
    model_configs = {
        'transformer': {
            'model_type': 'transformer',
            'input_dim': data_info['feature_dim'],
            'output_dim': len(targets),
            'hidden_dim': 256,
            'num_layers': 6,
            'num_heads': 8,
            'dropout': 0.1,
            'use_pwa_pet': True,
            'pwa_buckets': {'trivial': 2, 'fund': 10, 'adj': 4},
            'use_rope': True,
            'use_pet': True,
            'pet_curv_reg': 1e-4
        },
        'gnn': {
            'model_type': 'gnn',
            'input_dim': data_info['feature_dim'],
            'output_dim': len(targets),
            'gnn_type': 'gat',
            'gnn_layers': 3,
            'hidden_dim': 128,
            'dropout': 0.1
        },
        'ensemble': {
            'model_type': 'ensemble',
            'input_dim': data_info['feature_dim'],
            'output_dim': len(targets),
            'ensemble_models': ['transformer', 'gnn'],
            'ensemble_weights': [0.6, 0.4]
        }
    }
    
    training_config = {
        'epochs': 5,  # Reduced for demo
        'batch_size': 16,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'optimizer': 'adam',
        'scheduler': 'cosine',
        'checkpoint_interval': 2
    }
    
    # Train all models
    training_results = model_trainer.train_all_models(
        targets=targets,
        model_configs=model_configs,
        training_config=training_config,
        save_models=True
    )
    
    print(f"  - Trained {len(training_results)} models")
    for model_name, results in training_results.items():
        if 'error' not in results:
            print(f"    - {model_name}: Training completed successfully")
        else:
            print(f"    - {model_name}: Training failed - {results['error']}")
    
    # 3. Model Evaluation Demo
    print("\n3. Model Evaluation Demo...")
    
    model_evaluator = ModelEvaluator(
        output_dir=str(results_dir),
        log_dir=str(demo_dir / "logs")
    )
    
    # Create mock evaluation results for demonstration
    mock_evaluation_results = {}
    
    for model_name in ['transformer', 'gnn', 'ensemble']:
        if model_name in training_results and 'error' not in training_results[model_name]:
            # Create mock evaluation results
            mock_predictions = np.random.normal(5, 1, (20, len(targets)))
            mock_targets = np.random.normal(5, 1, (20, len(targets)))
            
            mock_results = {
                'model_name': model_name,
                'overall_metrics': {
                    'mse': np.random.uniform(0.5, 2.0),
                    'rmse': np.random.uniform(0.7, 1.4),
                    'mae': np.random.uniform(0.6, 1.2),
                    'r2_score': np.random.uniform(0.6, 0.9),
                    'correlation': np.random.uniform(0.7, 0.95)
                },
                'target_metrics': {
                    target: {
                        'mse': np.random.uniform(0.5, 2.0),
                        'rmse': np.random.uniform(0.7, 1.4),
                        'mae': np.random.uniform(0.6, 1.2),
                        'r2_score': np.random.uniform(0.6, 0.9),
                        'correlation': np.random.uniform(0.7, 0.95)
                    } for target in targets
                },
                'predictions': mock_predictions,
                'targets': mock_targets,
                'losses': [np.random.uniform(0.5, 2.0) for _ in range(20)],
                'evaluation_metadata': {
                    'num_samples': 20,
                    'num_targets': len(targets),
                    'targets': targets
                }
            }
            
            mock_evaluation_results[model_name] = mock_results
    
    # Compare models
    if mock_evaluation_results:
        comparison_results = model_evaluator.compare_models(mock_evaluation_results)
        print(f"  - Model comparison completed")
        print(f"  - Best model: {comparison_results['best_model']}")
        
        # Create evaluation plots
        plot_paths = model_evaluator.create_evaluation_plots(mock_evaluation_results)
        print(f"  - Created {len(plot_paths)} evaluation plots")
        
        # Generate evaluation report
        report_path = model_evaluator.generate_evaluation_report(
            mock_evaluation_results, comparison_results
        )
        print(f"  - Generated evaluation report: {report_path}")
    
    # 4. Model Distribution Demo
    print("\n4. Model Distribution Demo...")
    
    model_distributor = ModelDistributor(
        models_dir=str(models_dir),
        distribution_dir=str(distributions_dir),
        log_dir=str(demo_dir / "logs")
    )
    
    # Create model packages for each trained model
    package_paths = {}
    
    for model_name in ['transformer', 'gnn', 'ensemble']:
        if model_name in training_results and 'error' not in training_results[model_name]:
            # Find the model file
            model_files = list(models_dir.glob(f"{model_name}_pretrained_*.pt"))
            if model_files:
                model_path = str(model_files[0])
                
                # Create model package
                package_path = model_distributor.create_model_package(
                    model_path=model_path,
                    model_name=f"chemforge_{model_name}",
                    version="1.0.0",
                    description=f"ChemForge {model_name} model for CNS drug discovery",
                    author="ChemForge Development Team",
                    license="MIT",
                    dependencies=["torch>=1.9.0", "numpy>=1.21.0", "pandas>=1.3.0"],
                    include_data=True,
                    include_examples=True
                )
                
                package_paths[model_name] = package_path
                print(f"  - Created package for {model_name}: {package_path}")
    
    # Create distribution catalog
    if package_paths:
        catalog_path = model_distributor.create_distribution_catalog(
            catalog_name="chemforge_models",
            description="ChemForge Pre-trained Models Catalog"
        )
        print(f"  - Created distribution catalog: {catalog_path}")
        
        # Get available models
        available_models = model_distributor.get_available_models()
        print(f"  - Available models: {len(available_models)}")
        
        # Get distribution summary
        summary = model_distributor.get_distribution_summary()
        print(f"  - Distribution summary: {summary['total_models']} models")
    
    # 5. Model Usage Demo
    print("\n5. Model Usage Demo...")
    
    # Demonstrate loading and using a pre-trained model
    if 'transformer' in package_paths:
        print("  - Demonstrating model loading and usage...")
        
        # This would normally load the actual model
        # For demo purposes, we'll show the structure
        print("    - Model loading structure:")
        print("      ```python")
        print("      import torch")
        print("      from chemforge.models.transformer_model import TransformerModel")
        print("      ")
        print("      # Load model")
        print("      model_data = torch.load('chemforge_transformer.pt')")
        print("      model = TransformerModel(**model_data['model_config'])")
        print("      model.load_state_dict(model_data['model_state_dict'])")
        print("      model.eval()")
        print("      ")
        print("      # Make predictions")
        print("      features = torch.randn(1, 200)  # Example features")
        print("      predictions = model(features)")
        print("      ```")
    
    # 6. Data Export Demo
    print("\n6. Data Export Demo...")
    
    # Export dataset in different formats
    export_formats = ['csv', 'json', 'parquet']
    
    for format_type in export_formats:
        try:
            export_path = data_preparator.export_dataset(
                prepared_dataset, format_type
            )
            print(f"  - Exported dataset to {format_type}: {export_path}")
        except Exception as e:
            print(f"  - Failed to export {format_type}: {e}")
    
    # 7. Cleanup Demo
    print("\n7. Cleanup Demo...")
    
    # Show available datasets
    available_datasets = data_preparator.get_available_datasets()
    print(f"  - Available datasets: {len(available_datasets)}")
    
    # Show training history
    training_history = model_trainer.get_training_history()
    print(f"  - Training history: {len(training_history)} models")
    
    # Show evaluation summary
    evaluation_summary = model_evaluator.get_evaluation_summary()
    print(f"  - Evaluation summary: {evaluation_summary['num_models_evaluated']} models evaluated")
    
    # Show distribution summary
    distribution_summary = model_distributor.get_distribution_summary()
    print(f"  - Distribution summary: {distribution_summary['total_models']} models distributed")
    
    print("\nDemo completed successfully!")
    logger.info("ChemForge Pre-trained Models Demo completed")
    
    # Print summary
    print("\n" + "=" * 50)
    print("DEMO SUMMARY")
    print("=" * 50)
    print(f"Data Preparation: ✅ {prepared_dataset['data_info']['total_molecules']} molecules processed")
    print(f"Model Training: ✅ {len(training_results)} models trained")
    print(f"Model Evaluation: ✅ {len(mock_evaluation_results)} models evaluated")
    print(f"Model Distribution: ✅ {len(package_paths)} packages created")
    print(f"Data Export: ✅ {len(export_formats)} formats supported")
    print(f"Results Directory: {results_dir}")
    print(f"Models Directory: {models_dir}")
    print(f"Distributions Directory: {distributions_dir}")


if __name__ == "__main__":
    run_pretrained_models_demo()
