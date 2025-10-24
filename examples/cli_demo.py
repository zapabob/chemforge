"""
ChemForge CLI Demo

This module demonstrates the usage of ChemForge CLI commands.
"""

import subprocess
import sys
from pathlib import Path
import pandas as pd
import json


def run_cli_demo():
    """Run CLI demonstration."""
    print("ChemForge CLI Demo")
    print("=" * 50)
    
    # Create demo data
    print("Creating demo data...")
    demo_data = pd.DataFrame({
        'smiles': [
            'CCO',  # Ethanol
            'CCN',  # Ethylamine
            'CC(C)O',  # Isopropanol
            'CC(C)N',  # Isopropylamine
            'CC(C)(C)O',  # tert-Butanol
            'CC(C)(C)N',  # tert-Butylamine
            'CC(C)(C)CO',  # 2-Methyl-2-propanol
            'CC(C)(C)CN',  # 2-Methyl-2-propylamine
            'CC(C)(C)CCO',  # 2-Methyl-2-butanol
            'CC(C)(C)CCN'   # 2-Methyl-2-butylamine
        ],
        'molecule_id': [f'mol_{i}' for i in range(10)],
        'target_5HT2A': [5.0, 6.0, 4.5, 5.5, 4.0, 5.0, 3.5, 4.5, 3.0, 4.0],
        'target_D2': [4.5, 5.5, 4.0, 5.0, 3.5, 4.5, 3.0, 4.0, 2.5, 3.5],
        'target_CB1': [4.0, 5.0, 3.5, 4.5, 3.0, 4.0, 2.5, 3.5, 2.0, 3.0]
    })
    
    # Save demo data
    demo_data_path = Path("demo_data.csv")
    demo_data.to_csv(demo_data_path, index=False)
    print(f"Demo data saved to {demo_data_path}")
    
    # Create demo model checkpoint
    print("Creating demo model checkpoint...")
    demo_checkpoint = {
        'model_type': 'transformer',
        'model_config': {
            'input_dim': 100,
            'output_dim': 3,
            'use_pwa_pet': True,
            'pwa_buckets': {'trivial': 1, 'fund': 5, 'adj': 2},
            'pet_curv_reg': 1e-5
        },
        'model_state_dict': {},
        'optimizer_state_dict': {},
        'epoch': 0,
        'best_score': 0.0
    }
    
    demo_model_path = Path("demo_model.pt")
    import torch
    torch.save(demo_checkpoint, demo_model_path)
    print(f"Demo model checkpoint saved to {demo_model_path}")
        
    # Create demo target profile
    print("Creating demo target profile...")
    target_profile = {
        '5HT2A': 6.0,
        'D2': 5.5,
        'CB1': 5.0
    }
    
    target_profile_path = Path("demo_target_profile.json")
    with open(target_profile_path, 'w') as f:
        json.dump(target_profile, f, indent=2)
    print(f"Target profile saved to {target_profile_path}")
        
    # Demo CLI commands
    print("\nDemo CLI Commands:")
    print("-" * 30)
    
    # 1. Train command demo
    print("\n1. Train Command Demo:")
    print("   chemforge train --data-path demo_data.csv --model-type transformer --epochs 2 --batch-size 2")
        
    # 2. Predict command demo
    print("\n2. Predict Command Demo:")
    print("   chemforge predict --model-path demo_model.pt --data-path demo_data.csv --output-path predictions.csv")
        
    # 3. ADMET command demo
    print("\n3. ADMET Command Demo:")
    print("   chemforge admet --data-path demo_data.csv --properties all --output-path admet_predictions.csv")
    
    # 4. Generate command demo
    print("\n4. Generate Command Demo:")
    print("   chemforge generate --model-path demo_model.pt --num-molecules 5 --generation-method vae --output-path generated_molecules.csv")
        
    # 5. Optimize command demo
    print("\n5. Optimize Command Demo:")
    print("   chemforge optimize --model-path demo_model.pt --data-path demo_data.csv --optimization-method genetic --num-generations 2 --output-path optimized_molecules.csv")
        
    # Run actual CLI commands
    print("\nRunning CLI Commands:")
    print("-" * 30)
    
    # Run train command
    print("\nRunning train command...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "chemforge.cli.main", "train",
            "--data-path", str(demo_data_path),
            "--model-type", "transformer",
            "--epochs", "2",
            "--batch-size", "2",
            "--verbose"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✓ Train command completed successfully")
        else:
            print(f"✗ Train command failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("✗ Train command timed out")
    except Exception as e:
        print(f"✗ Train command error: {e}")
        
    # Run predict command
    print("\nRunning predict command...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "chemforge.cli.main", "predict",
            "--model-path", str(demo_model_path),
            "--data-path", str(demo_data_path),
            "--output-path", "demo_predictions.csv"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✓ Predict command completed successfully")
        else:
            print(f"✗ Predict command failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("✗ Predict command timed out")
    except Exception as e:
        print(f"✗ Predict command error: {e}")
    
    # Run ADMET command
    print("\nRunning ADMET command...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "chemforge.cli.main", "admet",
            "--data-path", str(demo_data_path),
            "--properties", "all",
            "--output-path", "demo_admet_predictions.csv"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✓ ADMET command completed successfully")
        else:
            print(f"✗ ADMET command failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("✗ ADMET command timed out")
    exceptexcept Exception as e:
        print(f"✗ ADMET command error: {e}")
    
    # Run generate command
    print("\nRunning generate command...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "chemforge.cli.main", "generate",
            "--model-path", str(demo_model_path),
            "--num-molecules", "3",
                "--generation-method", "vae",
            "--output-path", "demo_generated_molecules.csv"
        ], capture_output=True, text=True, timeout=30)
            
        if result.returncode == 0:
            print("✓ Generate command completed successfully")
        else:
            print(f"✗ Generate command failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("✗ Generate command timed out")
    exceptexcept Exception as e:
        print(f"✗ Generate command error: {e}")
        
    # Run optimize command
    print("\nRunning optimize command...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "chemforge.cli.main", "optimize",
            "--model-path", str(demo_model_path),
            "--data-path", str(demo_data_path),
                "--optimization-method", "genetic",
                "--num-generations", "2",
            "--output-path", "demo_optimized_molecules.csv"
        ], capture_output=True, text=True, timeout=30)
            
        if result.returncode == 0:
            print("✓ Optimize command completed successfully")
        else:
            print(f"✗ Optimize command failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("✗ Optimize command timed out")
    exceptexcept Exception as e:
        print(f"✗ Optimize command error: {e}")
    
    # Cleanup
    print("\nCleaning up demo files...")
    demo_files = [
        demo_data_path,
        demo_model_path,
        target_profile_path,
        Path("demo_predictions.csv"),
        Path("demo_admet_predictions.csv"),
        Path("demo_generated_molecules.csv"),
        Path("demo_optimized_molecules.csv")
    ]
    
    for file_path in demo_files:
        if file_path.exists():
            file_path.unlink()
            print(f"  Removed {file_path}")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    run_cli_demo()