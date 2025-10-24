"""
ChEMBL Integration Example

Demonstrates how to use the multi-target-pIC50-predictor library with correct ChEMBL database IDs.
"""

import pandas as pd
import numpy as np
from chemforge import MultiTargetPredictor
from chemforge.targets import get_chembl_targets, get_target_info
from chemforge.data import ChEMBLLoader


def main():
    """Main example function."""
    print("🧬 Multi-Target pIC50 Predictor - ChEMBL Integration Example")
    print("=" * 70)
    
    # 1. Initialize ChEMBL targets
    print("\n📋 1. ChEMBL Targets")
    targets = get_chembl_targets()
    available_targets = targets.get_available_targets()
    print(f"Available targets: {', '.join(available_targets[:5])}...")
    
    # 2. Get target information
    print("\n🎯 2. Target Information")
    target = 'DAT'
    info = get_target_info(target)
    print(f"Target: {target}")
    print(f"ChEMBL ID: {info['chembl_id']}")
    print(f"Name: {info['name']}")
    print(f"Function: {info['function']}")
    print(f"Diseases: {', '.join(info['diseases'])}")
    print(f"Drugs: {', '.join(info['drugs'])}")
    
    # 3. Initialize ChEMBL data loader
    print("\n📊 3. ChEMBL Data Loader")
    loader = ChEMBLLoader()
    
    # 4. Get target data (simulated)
    print("\n📈 4. Target Data")
    print("Fetching data from ChEMBL database...")
    
    # Simulate data fetching
    sample_data = {
        'smiles': [
            'CC(CC1=CC=CC=C1)NC',
            'COc1ccc(CCN(C)C)cc1',
            'CCN(CC)CCc1ccc(O)cc1',
            'COc1ccc(CCNCC)cc1',
            'CCN(CC)CCc1ccc(F)cc1'
        ],
        'pIC50': [7.2, 6.8, 6.5, 6.3, 6.1],
        'target': ['DAT'] * 5
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Retrieved {len(df)} records for {target}")
    print(f"pIC50 range: {df['pIC50'].min():.1f} - {df['pIC50'].max():.1f}")
    
    # 5. Initialize predictor
    print("\n🤖 5. Multi-Target Predictor")
    predictor = MultiTargetPredictor(
        targets=['DAT', '5HT2A', 'CB1', 'CB2', 'MOR'],
        device='auto'
    )
    
    print(f"Predictor initialized with {len(predictor.targets)} targets")
    print(f"Available targets: {predictor.get_available_targets()}")
    
    # 6. Make predictions
    print("\n🔮 6. Predictions")
    smiles = "CC(CC1=CC=CC=C1)NC"
    
    # Single target prediction
    print(f"\nSingle target prediction for {target}:")
    prediction = predictor.predict(smiles, target=target)
    print(f"pIC50: {prediction[target]['pIC50']:.2f}")
    print(f"Uncertainty: {prediction[target]['uncertainty']:.2f}")
    print(f"Confidence: {prediction[target]['confidence']:.2f}")
    
    # Multi-target prediction
    print(f"\nMulti-target prediction:")
    multi_predictions = predictor.predict_multi_target(
        smiles=smiles,
        targets=['DAT', '5HT2A', 'CB1']
    )
    
    for target_name, pred in multi_predictions.items():
        print(f"  {target_name}: pIC50 = {pred['pIC50']:.2f} ± {pred['uncertainty']:.2f}")
    
    # 7. Batch predictions
    print("\n📊 7. Batch Predictions")
    smiles_list = [
        "CC(CC1=CC=CC=C1)NC",
        "COc1ccc(CCN(C)C)cc1",
        "CCN(CC)CCc1ccc(O)cc1"
    ]
    
    batch_predictions = predictor.predict_batch(
        smiles_list=smiles_list,
        target=target
    )
    
    for i, (smiles, pred) in enumerate(zip(smiles_list, batch_predictions)):
        print(f"Molecule {i+1}: {smiles}")
        print(f"  pIC50: {pred[target]['pIC50']:.2f}")
        print(f"  Confidence: {pred[target]['confidence']:.2f}")
    
    # 8. Target-specific analysis
    print("\n🎯 8. Target-specific Analysis")
    families = {
        'serotonin': ['5HT2A', '5HT1A'],
        'dopamine': ['D1', 'D2'],
        'cannabinoid': ['CB1', 'CB2'],
        'opioid': ['MOR', 'DOR', 'KOR', 'NOP'],
        'transporter': ['SERT', 'DAT', 'NET']
    }
    
    for family, target_list in families.items():
        print(f"\n{family.capitalize()} family:")
        for target in target_list[:2]:  # Show first 2 targets
            info = get_target_info(target)
            print(f"  {target}: {info['name']}")
            print(f"    Function: {info['function']}")
            print(f"    ChEMBL ID: {info['chembl_id']}")
    
    # 9. Model evaluation
    print("\n📈 9. Model Evaluation")
    test_data = pd.DataFrame({
        'smiles': ['CC(CC1=CC=CC=C1)NC', 'COc1ccc(CCN(C)C)cc1'],
        'pIC50': [7.2, 6.8]
    })
    
    try:
        evaluation = predictor.evaluate(
            test_data=test_data,
            target=target,
            metrics=['mse', 'mae', 'r2']
        )
        
        print("Evaluation results:")
        for metric, value in evaluation.items():
            print(f"  {metric}: {value:.4f}")
    except Exception as e:
        print(f"Evaluation error: {e}")
    
    print("\n🎉 ChEMBL Integration Example Complete!")
    print("なんｊ魂で最後まで頑張った結果や！めっちゃ嬉しいで〜！💪")


if __name__ == "__main__":
    main()
