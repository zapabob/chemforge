"""
CNS Targets Example

Example usage of extended CNS targets including D1, CB1/CB2, and opioid receptors.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from molecular_pwa_pet.targets import (
    CNSTargets,
    get_cns_targets,
    get_target_info,
    get_optimization_goals,
    get_selectivity_targets,
    get_safety_targets
)


def main():
    """Main example function."""
    print("🧬 CNS Targets Example")
    print("=" * 50)
    
    # Initialize CNS targets
    targets = get_cns_targets()
    
    # 1. List all available targets
    print("\n📋 Available CNS Targets:")
    available_targets = targets.get_available_targets()
    for i, target in enumerate(available_targets, 1):
        print(f"  {i:2d}. {target}")
    
    # 2. Group targets by family
    print("\n👥 Targets by Family:")
    families = targets.families
    for family, target_list in families.items():
        print(f"  {family.capitalize()}: {', '.join(target_list)}")
    
    # 3. Example: D1 receptor
    print("\n🎯 D1 Receptor Example:")
    d1_info = get_target_info('D1')
    print(f"  PDB ID: {d1_info['pdb_id']}")
    print(f"  Function: {d1_info['function']}")
    print(f"  Diseases: {', '.join(d1_info['diseases'])}")
    print(f"  Drugs: {', '.join(d1_info['drugs'])}")
    
    # 4. Optimization goals for D1
    print("\n🎯 D1 Optimization Goals:")
    d1_goals = get_optimization_goals('D1')
    print(f"  Goals: {', '.join(d1_goals)}")
    
    # 5. Selectivity targets for D1
    print("\n🎯 D1 Selectivity Targets:")
    d1_selectivity = get_selectivity_targets('D1')
    for target, threshold in d1_selectivity.items():
        print(f"  {target}: {threshold}")
    
    # 6. Safety targets for D1
    print("\n🎯 D1 Safety Targets:")
    d1_safety = get_safety_targets('D1')
    for metric, value in d1_safety.items():
        print(f"  {metric}: {value}")
    
    # 7. Example: CB1 receptor
    print("\n🎯 CB1 Receptor Example:")
    cb1_info = get_target_info('CB1')
    print(f"  PDB ID: {cb1_info['pdb_id']}")
    print(f"  Function: {cb1_info['function']}")
    print(f"  Diseases: {', '.join(cb1_info['diseases'])}")
    print(f"  Drugs: {', '.join(cb1_info['drugs'])}")
    
    # 8. Optimization goals for CB1
    print("\n🎯 CB1 Optimization Goals:")
    cb1_goals = get_optimization_goals('CB1')
    print(f"  Goals: {', '.join(cb1_goals)}")
    
    # 9. Selectivity targets for CB1
    print("\n🎯 CB1 Selectivity Targets:")
    cb1_selectivity = get_selectivity_targets('CB1')
    for target, threshold in cb1_selectivity.items():
        print(f"  {target}: {threshold}")
    
    # 10. Safety targets for CB1
    print("\n🎯 CB1 Safety Targets:")
    cb1_safety = get_safety_targets('CB1')
    for metric, value in cb1_safety.items():
        print(f"  {metric}: {value}")
    
    # 11. Example: MOR receptor
    print("\n🎯 MOR Receptor Example:")
    mor_info = get_target_info('MOR')
    print(f"  PDB ID: {mor_info['pdb_id']}")
    print(f"  Function: {mor_info['function']}")
    print(f"  Diseases: {', '.join(mor_info['diseases'])}")
    print(f"  Drugs: {', '.join(mor_info['drugs'])}")
    
    # 12. Optimization goals for MOR
    print("\n🎯 MOR Optimization Goals:")
    mor_goals = get_optimization_goals('MOR')
    print(f"  Goals: {', '.join(mor_goals)}")
    
    # 13. Selectivity targets for MOR
    print("\n🎯 MOR Selectivity Targets:")
    mor_selectivity = get_selectivity_targets('MOR')
    for target, threshold in mor_selectivity.items():
        print(f"  {target}: {threshold}")
    
    # 14. Safety targets for MOR
    print("\n🎯 MOR Safety Targets:")
    mor_safety = get_safety_targets('MOR')
    for metric, value in mor_safety.items():
        print(f"  {metric}: {value}")
    
    # 15. Targets by disease
    print("\n🎯 Targets for Depression:")
    depression_targets = targets.get_targets_by_disease('depression')
    print(f"  Targets: {', '.join(depression_targets)}")
    
    # 16. Targets by drug
    print("\n🎯 Targets for Morphine:")
    morphine_targets = targets.get_targets_by_drug('morphine')
    print(f"  Targets: {', '.join(morphine_targets)}")
    
    print("\n🎉 CNS Targets Example Complete!")
    print("なんｊ魂で最後まで頑張った結果や！めっちゃ嬉しいで〜！💪")


if __name__ == "__main__":
    main()
