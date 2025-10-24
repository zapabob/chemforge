"""
Basic usage example for molecular-pwa-pet library.
"""

import torch
import numpy as np
from molecular_pwa_pet import (
    MolecularPWA_PETTransformer,
    get_cns_targets,
    get_target_info,
    get_optimization_goals,
    get_selectivity_targets,
    get_safety_targets
)


def main():
    """Main example function."""
    print("🧬 Molecular PWA+PET Transformer - Basic Usage Example")
    print("=" * 60)
    
    # 1. Initialize CNS targets
    print("\n📋 1. CNS Targets")
    targets = get_cns_targets()
    available_targets = targets.get_available_targets()
    print(f"Available targets: {', '.join(available_targets[:5])}...")
    
    # 2. Get target information
    print("\n🎯 2. Target Information")
    target = '5HT2A'
    info = get_target_info(target)
    print(f"Target: {target}")
    print(f"PDB ID: {info['pdb_id']}")
    print(f"Function: {info['function']}")
    print(f"Diseases: {', '.join(info['diseases'])}")
    print(f"Drugs: {', '.join(info['drugs'])}")
    
    # 3. Get optimization goals
    print("\n🎯 3. Optimization Goals")
    goals = get_optimization_goals(target)
    print(f"Optimization goals for {target}: {', '.join(goals)}")
    
    # 4. Get selectivity targets
    print("\n🎯 4. Selectivity Targets")
    selectivity = get_selectivity_targets(target)
    print(f"Selectivity targets for {target}:")
    for t, threshold in selectivity.items():
        print(f"  {t}: {threshold}")
    
    # 5. Get safety targets
    print("\n🎯 5. Safety Targets")
    safety = get_safety_targets(target)
    print(f"Safety targets for {target}:")
    for metric, value in safety.items():
        print(f"  {metric}: {value}")
    
    # 6. Initialize model
    print("\n🤖 6. Model Initialization")
    model = MolecularPWA_PETTransformer(
        d_model=512,
        n_layers=4,
        n_heads=8,
        max_atoms=50,
        atom_features=78,
        bond_features=12
    )
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 7. Prepare sample data
    print("\n📊 7. Sample Data")
    batch_size = 2
    max_atoms = 50
    atom_features = 78
    bond_features = 12
    
    # Sample molecular data
    atom_features_tensor = torch.randn(batch_size, max_atoms, atom_features)
    bond_features_tensor = torch.randn(batch_size, max_atoms, max_atoms, bond_features)
    atom_mask = torch.ones(batch_size, max_atoms)
    coords = torch.randn(batch_size, max_atoms, 3)
    
    print(f"Atom features shape: {atom_features_tensor.shape}")
    print(f"Bond features shape: {bond_features_tensor.shape}")
    print(f"Atom mask shape: {atom_mask.shape}")
    print(f"Coordinates shape: {coords.shape}")
    
    # 8. Forward pass
    print("\n🚀 8. Forward Pass")
    with torch.no_grad():
        outputs = model(
            atom_features=atom_features_tensor,
            bond_features=bond_features_tensor,
            atom_mask=atom_mask,
            coords=coords
        )
    
    print("Model outputs:")
    for task, output in outputs.items():
        print(f"  {task}: {output.shape} (mean: {output.mean().item():.4f})")
    
    # 9. Sample predictions
    print("\n📈 9. Sample Predictions")
    pki_pred = outputs['pki'].squeeze().numpy()
    activity_pred = torch.softmax(outputs['activity'], dim=-1).numpy()
    cns_mpo_pred = outputs['cns_mpo'].squeeze().numpy()
    qed_pred = outputs['qed'].squeeze().numpy()
    sa_pred = outputs['sa'].squeeze().numpy()
    
    for i in range(batch_size):
        print(f"\nMolecule {i+1}:")
        print(f"  pKi: {pki_pred[i]:.2f}")
        print(f"  Activity: {activity_pred[i]}")
        print(f"  CNS-MPO: {cns_mpo_pred[i]:.2f}")
        print(f"  QED: {qed_pred[i]:.2f}")
        print(f"  SA: {sa_pred[i]:.2f}")
    
    # 10. Target-specific analysis
    print("\n🎯 10. Target-specific Analysis")
    families = targets.families
    for family, target_list in families.items():
        print(f"\n{family.capitalize()} family:")
        for target in target_list[:3]:  # Show first 3 targets
            info = get_target_info(target)
            print(f"  {target}: {info['function']}")
    
    print("\n🎉 Basic Usage Example Complete!")
    print("なんｊ魂で最後まで頑張った結果や！めっちゃ嬉しいで〜！💪")


if __name__ == "__main__":
    main()
