"""
ChemForge Molecular Generation Demo

This module demonstrates the usage of ChemForge molecular generation
capabilities using VAE, RL, and GA approaches.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from chemforge.generation.molecular_generator import MolecularGenerator
from chemforge.utils.logging_utils import setup_logging


def run_molecular_generation_demo():
    """Run molecular generation demonstration."""
    print("ChemForge Molecular Generation Demo")
    print("=" * 50)
    
    # Setup logging
    log_manager = setup_logging(level='INFO')
    logger = log_manager.get_logger('molecular_generation_demo')
    logger.info("Starting ChemForge Molecular Generation Demo")
    
    # Create demo directories
    demo_dir = Path("./molecular_generation_demo")
    demo_dir.mkdir(exist_ok=True)
    
    results_dir = demo_dir / "results"
    plots_dir = demo_dir / "plots"
    
    for dir_path in [results_dir, plots_dir]:
        dir_path.mkdir(exist_ok=True)
    
    # Initialize molecular generator
    generator = MolecularGenerator(
        output_dir=str(demo_dir),
        log_dir=str(demo_dir / "logs")
    )
    
    # 1. VAE Generation Demo
    print("\n1. VAE Generation Demo...")
    
    # Setup VAE generator
    vae_generator = generator.setup_vae_generator(
        input_dim=200,
        latent_dim=64,
        hidden_dim=256,
        max_length=50,
        vocab_size=100,
        device='cpu'
    )
    
    # Create demo training data
    demo_smiles = [
        'CCO', 'CCN', 'CC(C)O', 'CC(C)N', 'CC(C)(C)O',
        'CC(C)(C)N', 'CC(C)(C)(C)O', 'CC(C)(C)(C)N',
        'CC(C)(C)(C)(C)O', 'CC(C)(C)(C)(C)N',
        'CCOC', 'CCNC', 'CC(C)OC', 'CC(C)NC',
        'CC(C)(C)OC', 'CC(C)(C)NC', 'CC(C)(C)(C)OC',
        'CC(C)(C)(C)NC', 'CC(C)(C)(C)(C)OC', 'CC(C)(C)(C)(C)NC'
    ] * 5  # 100 molecules
    
    demo_features = np.random.randn(100, 200)
    
    # Generate molecules using VAE
    vae_results = generator.generate_with_vae(
        training_smiles=demo_smiles,
        training_features=demo_features,
        num_molecules=50,
        epochs=5,  # Reduced for demo
        temperature=1.0
    )
    
    print(f"  - Generated {len(vae_results['generated_molecules'])} molecules using VAE")
    print(f"  - Training completed in {len(vae_results['training_results']['train_losses'])} epochs")
    
    # 2. RL Optimization Demo
    print("\n2. RL Optimization Demo...")
    
    # Setup RL optimizer
    rl_optimizer = generator.setup_rl_optimizer(
        state_dim=200,
        action_dim=10,
        hidden_dim=128,
        learning_rate=1e-3,
        gamma=0.99,
        device='cpu'
    )
    
    # Define reward function
    def reward_function(smiles):
        """Simple reward function based on molecule length."""
        try:
            return len(smiles) * 0.1 + np.random.normal(0, 0.1)  # Add some noise
        except:
            return 0.0
    
    # Optimize molecules using RL
    initial_molecules = ['CCO', 'CCN', 'CC(C)O', 'CC(C)N', 'CC(C)(C)O']
    
    rl_results = generator.optimize_with_rl(
        initial_molecules=initial_molecules,
        reward_function=reward_function,
        max_steps=20,  # Reduced for demo
        temperature=1.0
    )
    
    print(f"  - Optimized {len(rl_results['optimized_results'])} molecules using RL")
    
    # Calculate average improvement
    improvements = [result['final_reward'] - reward_function(result['initial_smiles']) 
                   for result in rl_results['optimized_results']]
    avg_improvement = np.mean(improvements)
    print(f"  - Average improvement: {avg_improvement:.4f}")
    
    # 3. Genetic Algorithm Demo
    print("\n3. Genetic Algorithm Demo...")
    
    # Setup genetic optimizer
    genetic_optimizer = generator.setup_genetic_optimizer(
        population_size=50,
        mutation_rate=0.1,
        crossover_rate=0.8,
        elite_size=5,
        max_generations=20
    )
    
    # Define fitness function
    def fitness_function(smiles):
        """Simple fitness function based on molecule length."""
        try:
            return len(smiles) * 0.1 + np.random.normal(0, 0.1)  # Add some noise
        except:
            return 0.0
    
    # Optimize molecules using genetic algorithm
    genetic_results = generator.optimize_with_genetic(
        initial_molecules=initial_molecules,
        fitness_function=fitness_function,
        maximize=True
    )
    
    print(f"  - Generated {len(genetic_results['optimization_results']['best_molecules'])} molecules using GA")
    
    # Calculate average fitness
    best_molecules = genetic_results['optimization_results']['best_molecules']
    fitness_scores = [fitness_function(mol) for mol in best_molecules]
    avg_fitness = np.mean(fitness_scores)
    print(f"  - Average fitness: {avg_fitness:.4f}")
    
    # 4. Method Comparison Demo
    print("\n4. Method Comparison Demo...")
    
    # Define evaluation function
    def evaluation_function(smiles):
        """Evaluation function for comparing methods."""
        try:
            return len(smiles) * 0.1 + np.random.normal(0, 0.1)  # Add some noise
        except:
            return 0.0
    
    # Compare methods
    comparison_results = generator.compare_methods(
        test_molecules=initial_molecules,
        evaluation_function=evaluation_function,
        methods=['vae', 'rl', 'genetic']
    )
    
    print(f"  - Compared {len(comparison_results['methods'])} methods")
    print(f"  - Best method: {comparison_results['summary']['best_method']}")
    
    # Print method scores
    for method, results in comparison_results.items():
        if method != 'summary':
            print(f"    - {method}: {results['avg_score']:.4f} (best: {results['best_score']:.4f})")
    
    # 5. Visualization Demo
    print("\n5. Visualization Demo...")
    
    # Plot generation history for each method
    for method in ['vae', 'rl', 'genetic']:
        if method in generator.generation_history:
            plot_path = plots_dir / f"{method}_history.png"
            try:
                generator.plot_generation_history(method, save_path=str(plot_path))
                print(f"  - Saved {method} history plot to {plot_path}")
            except Exception as e:
                print(f"  - Failed to create {method} plot: {e}")
    
    # 6. Results Analysis Demo
    print("\n6. Results Analysis Demo...")
    
    # Analyze VAE results
    vae_molecules = vae_results['generated_molecules']
    vae_lengths = [len(mol) for mol in vae_molecules]
    print(f"  - VAE generated molecules length: {np.mean(vae_lengths):.2f} ± {np.std(vae_lengths):.2f}")
    
    # Analyze RL results
    rl_improvements = [result['final_reward'] - reward_function(result['initial_smiles']) 
                      for result in rl_results['optimized_results']]
    print(f"  - RL improvement: {np.mean(rl_improvements):.4f} ± {np.std(rl_improvements):.4f}")
    
    # Analyze GA results
    ga_fitness = [fitness_function(mol) for mol in best_molecules]
    print(f"  - GA fitness: {np.mean(ga_fitness):.4f} ± {np.std(ga_fitness):.4f}")
    
    # 7. Save Results Demo
    print("\n7. Save Results Demo...")
    
    # Save generation results
    results_path = results_dir / "generation_results.json"
    generator.save_generation_results(str(results_path))
    print(f"  - Saved generation results to {results_path}")
    
    # Get generation summary
    summary = generator.get_generation_summary()
    print(f"  - Total generations: {summary['total_generations']}")
    print(f"  - Methods used: {summary['methods_used']}")
    
    # 8. Advanced Features Demo
    print("\n8. Advanced Features Demo...")
    
    # VAE interpolation demo
    if 'vae' in generator.generation_history:
        print("  - VAE interpolation demo...")
        try:
            interpolated = generator.vae_generator.interpolate_molecules('CCO', 'CCN', num_steps=5)
            print(f"    - Interpolated between CCO and CCN: {interpolated}")
        except Exception as e:
            print(f"    - VAE interpolation failed: {e}")
    
    # GA diversity analysis
    if 'genetic' in generator.generation_history:
        print("  - GA diversity analysis...")
        try:
            final_population = genetic_results['optimization_results']['final_population']
            diversity = generator.genetic_optimizer._calculate_diversity(final_population)
            print(f"    - Final population diversity: {diversity:.4f}")
        except Exception as e:
            print(f"    - GA diversity analysis failed: {e}")
    
    # 9. Performance Metrics Demo
    print("\n9. Performance Metrics Demo...")
    
    # Calculate performance metrics
    metrics = {
        'vae': {
            'num_generated': len(vae_results['generated_molecules']),
            'avg_length': np.mean(vae_lengths),
            'std_length': np.std(vae_lengths)
        },
        'rl': {
            'num_optimized': len(rl_results['optimized_results']),
            'avg_improvement': np.mean(rl_improvements),
            'std_improvement': np.std(rl_improvements)
        },
        'genetic': {
            'num_generated': len(best_molecules),
            'avg_fitness': np.mean(ga_fitness),
            'std_fitness': np.std(ga_fitness)
        }
    }
    
    # Save metrics
    metrics_path = results_dir / "performance_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  - Saved performance metrics to {metrics_path}")
    
    # 10. Cleanup Demo
    print("\n10. Cleanup Demo...")
    
    # Show available results
    available_results = list(results_dir.glob("*.json"))
    available_plots = list(plots_dir.glob("*.png"))
    
    print(f"  - Available results: {len(available_results)} files")
    print(f"  - Available plots: {len(available_plots)} files")
    
    print("\nDemo completed successfully!")
    logger.info("ChemForge Molecular Generation Demo completed")
    
    # Print summary
    print("\n" + "=" * 50)
    print("DEMO SUMMARY")
    print("=" * 50)
    print(f"VAE Generation: ✅ {len(vae_results['generated_molecules'])} molecules generated")
    print(f"RL Optimization: ✅ {len(rl_results['optimized_results'])} molecules optimized")
    print(f"GA Optimization: ✅ {len(best_molecules)} molecules generated")
    print(f"Method Comparison: ✅ {len(comparison_results['methods'])} methods compared")
    print(f"Visualization: ✅ {len(available_plots)} plots created")
    print(f"Results Directory: {results_dir}")
    print(f"Plots Directory: {plots_dir}")


if __name__ == "__main__":
    run_molecular_generation_demo()
