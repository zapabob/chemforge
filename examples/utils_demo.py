"""
ChemForge Utils Demo

This module demonstrates the usage of ChemForge utility functions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import tempfile
import os

from chemforge.utils.database import ChEMBLDatabase, LocalDatabase
from chemforge.utils.visualization import (
    MolecularVisualizer,
    ADMETVisualizer,
    CNSMPOVisualizer,
    ScaffoldVisualizer,
    TrainingVisualizer
)
from chemforge.utils.file_utils import FileManager, DataExporter, DataImporter
from chemforge.utils.config_utils import ConfigManager, ModelConfig, TrainingConfig
from chemforge.utils.logging_utils import setup_logging, TrainingLogger
from chemforge.utils.validation import DataValidator, ModelValidator, PredictionValidator


def run_utils_demo():
    """Run utilities demonstration."""
    print("ChemForge Utils Demo")
    print("=" * 50)
    
    # Setup logging
    log_manager = setup_logging(level='INFO')
    logger = log_manager.get_logger('utils_demo')
    logger.info("Starting ChemForge Utils Demo")
    
    # Create demo data
    print("\n1. Creating demo data...")
    demo_data = pd.DataFrame({
        'smiles': ['CCO', 'CCN', 'CC(C)O', 'CC(C)N', 'CC(C)(C)O'],
        'molecule_id': [f'mol_{i}' for i in range(5)],
        'mol_weight': [46.07, 45.08, 60.10, 59.11, 74.12],
        'logp': [0.31, 0.16, 0.05, 0.10, -0.17],
        'hbd': [1, 2, 1, 2, 1],
        'hba': [1, 1, 1, 1, 1],
        'tpsa': [20.23, 26.02, 20.23, 26.02, 20.23],
        'rotatable_bonds': [0, 0, 0, 0, 0],
        'aromatic_rings': [0, 0, 0, 0, 0],
        'heavy_atoms': [2, 2, 3, 3, 4],
        'scaffold_type': ['trivial', 'fund', 'trivial', 'fund', 'trivial'],
        'cns_mpo_score': [4.5, 3.8, 5.2, 2.1, 4.8]
    })
    
    # Add ADMET properties
    demo_data['absorption'] = [0.8, 0.7, 0.9, 0.6, 0.85]
    demo_data['distribution'] = [0.7, 0.8, 0.6, 0.9, 0.75]
    demo_data['metabolism'] = [0.6, 0.7, 0.8, 0.5, 0.65]
    demo_data['excretion'] = [0.9, 0.8, 0.7, 0.6, 0.85]
    demo_data['toxicity'] = [0.3, 0.4, 0.2, 0.5, 0.25]
    
    print(f"Created demo data with {len(demo_data)} molecules")
    
    # 1. Database utilities demo
    print("\n2. Database utilities demo...")
    with tempfile.TemporaryDirectory() as temp_dir:
        # ChEMBL database demo
        chembl_db_path = os.path.join(temp_dir, 'chembl.db')
        chembl_db = ChEMBLDatabase(chembl_db_path)
        chembl_db.connect()
        chembl_db.create_tables()
        
        # Insert demo data
        for _, row in demo_data.iterrows():
            molecule_data = {
                'chembl_id': row['molecule_id'],
                'smiles': row['smiles'],
                'mol_weight': row['mol_weight'],
                'logp': row['logp'],
                'hbd': row['hbd'],
                'hba': row['hba'],
                'tpsa': row['tpsa'],
                'rotatable_bonds': row['rotatable_bonds'],
                'aromatic_rings': row['aromatic_rings'],
                'heavy_atoms': row['heavy_atoms']
            }
            chembl_db.insert_molecule(molecule_data)
        
        # Insert target
        target_data = {
            'chembl_id': 'CHEMBL1234',
            'target_name': '5-HT2A',
            'target_type': 'SINGLE PROTEIN',
            'organism': 'Homo sapiens',
            'uniprot_id': 'P28223',
            'gene_name': 'HTR2A'
        }
        target_id = chembl_db.insert_target(target_data)
        
        # Insert activities
        for i, (_, row) in enumerate(demo_data.iterrows()):
            activity_data = {
                'molecule_id': i + 1,
                'target_id': target_id,
                'activity_type': 'IC50',
                'activity_value': 5.0 + i * 0.5,
                'activity_unit': 'nM',
                'activity_relation': '='
            }
            chembl_db.insert_activity(activity_data)
        
        # Retrieve data
        molecules = chembl_db.get_molecules()
        targets = chembl_db.get_targets()
        activities = chembl_db.get_activities()
        
        print(f"  - Inserted {len(molecules)} molecules")
        print(f"  - Inserted {len(targets)} targets")
        print(f"  - Inserted {len(activities)} activities")
        
        chembl_db.disconnect()
        
        # Local database demo
        local_db_path = os.path.join(temp_dir, 'local.db')
        local_db = LocalDatabase(local_db_path)
        local_db.connect()
        local_db.create_tables()
        
        # Insert custom molecules
        for _, row in demo_data.iterrows():
            custom_molecule_data = {
                'molecule_id': row['molecule_id'],
                'smiles': row['smiles'],
                'features': {
                    'mol_weight': row['mol_weight'],
                    'logp': row['logp'],
                    'hbd': row['hbd'],
                    'hba': row['hba']
                },
                'properties': {
                    'cns_mpo_score': row['cns_mpo_score'],
                    'scaffold_type': row['scaffold_type']
                }
            }
            local_db.insert_custom_molecule(custom_molecule_data)
        
        # Insert custom predictions
        for _, row in demo_data.iterrows():
            prediction_data = {
                'molecule_id': row['molecule_id'],
                'target_name': '5-HT2A',
                'prediction_value': 5.0 + np.random.normal(0, 0.5),
                'confidence': 0.8 + np.random.normal(0, 0.1),
                'model_name': 'demo_model'
            }
            local_db.insert_custom_prediction(prediction_data)
        
        # Retrieve data
        custom_molecules = local_db.get_custom_molecules()
        custom_predictions = local_db.get_custom_predictions()
        
        print(f"  - Inserted {len(custom_molecules)} custom molecules")
        print(f"  - Inserted {len(custom_predictions)} custom predictions")
        
        local_db.disconnect()
    
    # 2. Visualization utilities demo
    print("\n3. Visualization utilities demo...")
    
    # Molecular visualization
    mol_visualizer = MolecularVisualizer()
    fig1 = mol_visualizer.plot_molecular_properties(demo_data)
    fig1.savefig('molecular_properties.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print("  - Created molecular properties plot")
    
    fig2 = mol_visualizer.plot_molecular_scatter(demo_data, 'mol_weight', 'logp', color_prop='hbd')
    fig2.savefig('molecular_scatter.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print("  - Created molecular scatter plot")
    
    fig3 = mol_visualizer.plot_molecular_correlation(demo_data)
    fig3.savefig('molecular_correlation.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print("  - Created molecular correlation matrix")
    
    # ADMET visualization
    admet_visualizer = ADMETVisualizer()
    fig4 = admet_visualizer.plot_admet_radar(demo_data, 'mol_0')
    fig4.savefig('admet_radar.png', dpi=150, bbox_inches='tight')
    plt.close(fig4)
    print("  - Created ADMET radar chart")
    
    fig5 = admet_visualizer.plot_admet_distribution(demo_data)
    fig5.savefig('admet_distribution.png', dpi=150, bbox_inches='tight')
    plt.close(fig5)
    print("  - Created ADMET distribution plot")
    
    fig6 = admet_visualizer.plot_admet_heatmap(demo_data)
    fig6.savefig('admet_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig6)
    print("  - Created ADMET heatmap")
    
    # CNS-MPO visualization
    cns_mpo_visualizer = CNSMPOVisualizer()
    fig7 = cns_mpo_visualizer.plot_cns_mpo_scores(demo_data)
    fig7.savefig('cns_mpo_scores.png', dpi=150, bbox_inches='tight')
    plt.close(fig7)
    print("  - Created CNS-MPO scores plot")
    
    fig8 = cns_mpo_visualizer.plot_cns_mpo_vs_properties(demo_data)
    fig8.savefig('cns_mpo_vs_properties.png', dpi=150, bbox_inches='tight')
    plt.close(fig8)
    print("  - Created CNS-MPO vs properties plot")
    
    # Scaffold visualization
    scaffold_visualizer = ScaffoldVisualizer()
    fig9 = scaffold_visualizer.plot_scaffold_distribution(demo_data)
    fig9.savefig('scaffold_distribution.png', dpi=150, bbox_inches='tight')
    plt.close(fig9)
    print("  - Created scaffold distribution plot")
    
    fig10 = scaffold_visualizer.plot_scaffold_properties(demo_data)
    fig10.savefig('scaffold_properties.png', dpi=150, bbox_inches='tight')
    plt.close(fig10)
    print("  - Created scaffold properties plot")
    
    # Training visualization
    training_visualizer = TrainingVisualizer()
    training_data = {
        'train_loss': [1.0, 0.8, 0.6, 0.4, 0.2],
        'val_loss': [1.1, 0.9, 0.7, 0.5, 0.3],
        'train_acc': [0.5, 0.6, 0.7, 0.8, 0.9],
        'val_acc': [0.4, 0.5, 0.6, 0.7, 0.8]
    }
    
    fig11 = training_visualizer.plot_training_curves(training_data)
    fig11.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig11)
    print("  - Created training curves plot")
    
    lr_schedule = [0.001, 0.0008, 0.0006, 0.0004, 0.0002]
    fig12 = training_visualizer.plot_learning_rate_schedule(lr_schedule)
    fig12.savefig('learning_rate_schedule.png', dpi=150, bbox_inches='tight')
    plt.close(fig12)
    print("  - Created learning rate schedule plot")
    
    model_results = {
        'Model A': {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.88},
        'Model B': {'accuracy': 0.87, 'precision': 0.85, 'recall': 0.89},
        'Model C': {'accuracy': 0.83, 'precision': 0.80, 'recall': 0.86}
    }
    
    fig13 = training_visualizer.plot_model_comparison(model_results)
    fig13.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig13)
    print("  - Created model comparison plot")
    
    # 3. File utilities demo
    print("\n4. File utilities demo...")
    
    file_manager = FileManager()
    data_exporter = DataExporter()
    data_importer = DataImporter()
    
    # Export data in different formats
    csv_path = 'demo_data.csv'
    data_exporter.export_csv(demo_data, csv_path)
    print(f"  - Exported data to {csv_path}")
    
    json_path = 'demo_data.json'
    demo_dict = demo_data.to_dict('records')
    data_exporter.export_json(demo_dict, json_path)
    print(f"  - Exported data to {json_path}")
    
    pickle_path = 'demo_data.pkl'
    data_exporter.export_pickle(demo_data, pickle_path)
    print(f"  - Exported data to {pickle_path}")
    
    # Import data
    loaded_csv = data_importer.import_csv(csv_path)
    print(f"  - Imported {len(loaded_csv)} rows from CSV")
    
    loaded_json = data_importer.import_json(json_path)
    print(f"  - Imported {len(loaded_json)} records from JSON")
    
    loaded_pickle = data_importer.import_pickle(pickle_path)
    print(f"  - Imported data from pickle")
    
    # 4. Configuration utilities demo
    print("\n5. Configuration utilities demo...")
    
    config_manager = ConfigManager()
    default_config = config_manager.create_default_config()
    
    # Create custom model configuration
    model_config = ModelConfig(
        model_type='transformer',
        input_dim=200,
        output_dim=5,
        hidden_dim=512,
        num_layers=8,
        num_heads=16,
        dropout=0.2,
        use_pwa_pet=True,
        pwa_buckets={'trivial': 2, 'fund': 10, 'adj': 4},
        use_rope=True,
        use_pet=True,
        pet_curv_reg=1e-4
    )
    
    config_manager.set_model_config(model_config)
    print("  - Set custom model configuration")
    
    # Create custom training configuration
    training_config = TrainingConfig(
        epochs=200,
        batch_size=64,
        learning_rate=1e-4,
        weight_decay=1e-5,
        optimizer='adamw',
        scheduler='cosine',
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        random_seed=123,
        use_amp=True,
        gradient_clip=1.0,
        early_stopping=True,
        patience=20,
        checkpoint_interval=20,
        save_best=True,
        save_last=True,
        log_interval=20,
        log_level='INFO'
    )
    
    config_manager.set_training_config(training_config)
    print("  - Set custom training configuration")
    
    # Save configuration
    config_path = 'demo_config.json'
    config_manager.save_config(config_path)
    print(f"  - Saved configuration to {config_path}")
    
    # Load configuration
    loaded_config = config_manager.load_config(config_path)
    print(f"  - Loaded configuration from {config_path}")
    
    # 5. Logging utilities demo
    print("\n6. Logging utilities demo...")
    
    training_logger = TrainingLogger('demo_training', log_dir='./logs')
    
    # Simulate training process
    config = {
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'model_type': 'transformer'
    }
    
    training_logger.start_training(config)
    print("  - Started training logging")
    
    # Simulate epochs
    for epoch in range(5):
        training_logger.start_epoch(epoch, 100)
        
        # Simulate batch processing
        for batch in range(10):
            training_logger.log_batch(batch, 32, 0.5 - epoch * 0.1)
        
        # Simulate epoch end
        metrics = {
            'train_loss': 0.5 - epoch * 0.1,
            'val_loss': 0.6 - epoch * 0.1,
            'train_acc': 0.5 + epoch * 0.1,
            'val_acc': 0.4 + epoch * 0.1
        }
        training_logger.end_epoch(epoch, metrics)
        
        # Simulate validation
        val_metrics = {
            'accuracy': 0.4 + epoch * 0.1,
            'precision': 0.4 + epoch * 0.1,
            'recall': 0.4 + epoch * 0.1
        }
        training_logger.log_validation(val_metrics)
        
        # Simulate checkpoint
        if epoch % 2 == 0:
            training_logger.log_checkpoint(epoch, f'checkpoint_epoch_{epoch}.pt')
    
    # Simulate training end
    final_metrics = {
        'final_accuracy': 0.9,
        'final_loss': 0.1,
        'best_epoch': 4
    }
    training_logger.end_training(final_metrics)
    print("  - Completed training logging")
    
    # 6. Validation utilities demo
    print("\n7. Validation utilities demo...")
    
    data_validator = DataValidator()
    model_validator = ModelValidator()
    prediction_validator = PredictionValidator()
    
    # Validate molecular data
    mol_validation = data_validator.validate_molecular_data(demo_data)
    print(f"  - Molecular data validation: {mol_validation['valid']} ({len(mol_validation['errors'])} errors, {len(mol_validation['warnings'])} warnings)")
    
    # Validate activity data
    activity_data = pd.DataFrame({
        'molecule_id': [1, 2, 3, 4, 5],
        'target_id': [1, 1, 2, 2, 3],
        'activity_value': [5.0, 6.0, 4.5, 5.5, 4.0],
        'activity_type': ['IC50', 'IC50', 'IC50', 'IC50', 'IC50'],
        'activity_unit': ['nM', 'nM', 'nM', 'nM', 'nM'],
        'activity_relation': ['=', '=', '=', '=', '=']
    })
    
    activity_validation = data_validator.validate_activity_data(activity_data)
    print(f"  - Activity data validation: {activity_validation['valid']} ({len(activity_validation['errors'])} errors, {len(activity_validation['warnings'])} warnings)")
    
    # Validate model configuration
    model_config_dict = {
        'model_type': 'transformer',
        'input_dim': 200,
        'output_dim': 5,
        'hidden_dim': 512,
        'num_layers': 8,
        'num_heads': 16,
        'dropout': 0.2,
        'use_pwa_pet': True,
        'pwa_buckets': {'trivial': 2, 'fund': 10, 'adj': 4},
        'use_rope': True,
        'use_pet': True,
        'pet_curv_reg': 1e-4
    }
    
    model_validation = model_validator.validate_model_config(model_config_dict)
    print(f"  - Model configuration validation: {model_validation['valid']} ({len(model_validation['errors'])} errors, {len(model_validation['warnings'])} warnings)")
    
    # Validate training configuration
    training_config_dict = {
        'epochs': 200,
        'batch_size': 64,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'train_split': 0.7,
        'val_split': 0.15,
        'test_split': 0.15
    }
    
    training_validation = model_validator.validate_training_config(training_config_dict)
    print(f"  - Training configuration validation: {training_validation['valid']} ({len(training_validation['errors'])} errors, {len(training_validation['warnings'])} warnings)")
    
    # Validate predictions
    predictions = np.array([5.0, 6.0, 4.5, 5.5, 4.0])
    confidence = np.array([0.8, 0.9, 0.7, 0.85, 0.75])
    
    prediction_validation = prediction_validator.validate_predictions(predictions, confidence)
    print(f"  - Prediction validation: {prediction_validation['valid']} ({len(prediction_validation['errors'])} errors, {len(prediction_validation['warnings'])} warnings)")
    
    # Validate ADMET predictions
    admet_predictions = {
        'absorption': demo_data['absorption'].values,
        'distribution': demo_data['distribution'].values,
        'metabolism': demo_data['metabolism'].values,
        'excretion': demo_data['excretion'].values,
        'toxicity': demo_data['toxicity'].values
    }
    
    admet_validation = prediction_validator.validate_admet_predictions(admet_predictions)
    print(f"  - ADMET prediction validation: {admet_validation['valid']} ({len(admet_validation['errors'])} errors, {len(admet_validation['warnings'])} warnings)")
    
    # Validate CNS-MPO scores
    cns_mpo_scores = demo_data['cns_mpo_score'].values
    cns_mpo_validation = prediction_validator.validate_cns_mpo_scores(cns_mpo_scores)
    print(f"  - CNS-MPO score validation: {cns_mpo_validation['valid']} ({len(cns_mpo_validation['errors'])} errors, {len(cns_mpo_validation['warnings'])} warnings)")
    
    # Cleanup
    print("\n8. Cleaning up demo files...")
    demo_files = [
        'molecular_properties.png',
        'molecular_scatter.png',
        'molecular_correlation.png',
        'admet_radar.png',
        'admet_distribution.png',
        'admet_heatmap.png',
        'cns_mpo_scores.png',
        'cns_mpo_vs_properties.png',
        'scaffold_distribution.png',
        'scaffold_properties.png',
        'training_curves.png',
        'learning_rate_schedule.png',
        'model_comparison.png',
        'demo_data.csv',
        'demo_data.json',
        'demo_data.pkl',
        'demo_config.json'
    ]
    
    for file_path in demo_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"  - Removed {file_path}")
    
    # Remove logs directory
    import shutil
    if os.path.exists('./logs'):
        shutil.rmtree('./logs')
        print("  - Removed logs directory")
    
    print("\nDemo completed successfully!")
    logger.info("ChemForge Utils Demo completed")


if __name__ == "__main__":
    run_utils_demo()
