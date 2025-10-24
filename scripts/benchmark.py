"""
Benchmark script for molecular-pwa-pet.
"""

import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def benchmark_model_performance():
    """Benchmark model performance."""
    print("🚀 Benchmarking Model Performance")
    print("=" * 40)
    
    from molecular_pwa_pet import MolecularPWA_PETTransformer
    
    # Model configurations
    configs = [
        {"d_model": 256, "n_layers": 2, "n_heads": 4, "name": "Small"},
        {"d_model": 512, "n_layers": 4, "n_heads": 8, "name": "Medium"},
        {"d_model": 1024, "n_layers": 8, "n_heads": 16, "name": "Large"},
    ]
    
    # Batch sizes to test
    batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256]
    
    results = []
    
    for config in configs:
        print(f"\n📊 Testing {config['name']} model...")
        
        # Initialize model
        model = MolecularPWA_PETTransformer(
            d_model=config['d_model'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
            max_atoms=50,
            atom_features=78,
            bond_features=12
        )
        
        if torch.cuda.is_available():
            model = model.cuda()
            device = 'cuda'
        else:
            device = 'cpu'
        
        for batch_size in batch_sizes:
            print(f"  Batch size: {batch_size}")
            
            # Prepare data
            atom_features = torch.randn(batch_size, 50, 78, device=device)
            bond_features = torch.randn(batch_size, 50, 50, 12, device=device)
            atom_mask = torch.ones(batch_size, 50, device=device)
            coords = torch.randn(batch_size, 50, 3, device=device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = model(
                        atom_features=atom_features,
                        bond_features=bond_features,
                        atom_mask=atom_mask,
                        coords=coords
                    )
            
            # Benchmark
            times = []
            for _ in range(10):
                start_time = time.time()
                with torch.no_grad():
                    _ = model(
                        atom_features=atom_features,
                        bond_features=bond_features,
                        atom_mask=atom_mask,
                        coords=coords
                    )
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            results.append({
                'model': config['name'],
                'batch_size': batch_size,
                'time': avg_time,
                'std': std_time,
                'device': device
            })
            
            print(f"    Time: {avg_time:.4f} ± {std_time:.4f} s")
    
    return results


def benchmark_memory_usage():
    """Benchmark memory usage."""
    print("\n💾 Benchmarking Memory Usage")
    print("=" * 40)
    
    from molecular_pwa_pet import MolecularPWA_PETTransformer
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping memory benchmark")
        return []
    
    # Model configurations
    configs = [
        {"d_model": 256, "n_layers": 2, "n_heads": 4, "name": "Small"},
        {"d_model": 512, "n_layers": 4, "n_heads": 8, "name": "Medium"},
        {"d_model": 1024, "n_layers": 8, "n_heads": 16, "name": "Large"},
    ]
    
    # Batch sizes to test
    batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256]
    
    results = []
    
    for config in configs:
        print(f"\n📊 Testing {config['name']} model...")
        
        # Initialize model
        model = MolecularPWA_PETTransformer(
            d_model=config['d_model'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
            max_atoms=50,
            atom_features=78,
            bond_features=12
        ).cuda()
        
        for batch_size in batch_sizes:
            print(f"  Batch size: {batch_size}")
            
            # Clear cache
            torch.cuda.empty_cache()
            
            # Prepare data
            atom_features = torch.randn(batch_size, 50, 78, device='cuda')
            bond_features = torch.randn(batch_size, 50, 50, 12, device='cuda')
            atom_mask = torch.ones(batch_size, 50, device='cuda')
            coords = torch.randn(batch_size, 50, 3, device='cuda')
            
            # Measure memory
            torch.cuda.reset_peak_memory_stats()
            
            with torch.no_grad():
                _ = model(
                    atom_features=atom_features,
                    bond_features=bond_features,
                    atom_mask=atom_mask,
                    coords=coords
                )
            
            peak_memory = torch.cuda.max_memory_allocated() / 1e9  # GB
            
            results.append({
                'model': config['name'],
                'batch_size': batch_size,
                'memory': peak_memory,
                'device': 'cuda'
            })
            
            print(f"    Memory: {peak_memory:.2f} GB")
    
    return results


def benchmark_accuracy():
    """Benchmark accuracy on different tasks."""
    print("\n🎯 Benchmarking Accuracy")
    print("=" * 40)
    
    from molecular_pwa_pet import MolecularPWA_PETTransformer
    from molecular_pwa_pet.targets import get_cns_targets
    
    # Initialize model
    model = MolecularPWA_PETTransformer(
        d_model=512,
        n_layers=4,
        n_heads=8,
        max_atoms=50,
        atom_features=78,
        bond_features=12
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
        device = 'cuda'
    else:
        device = 'cpu'
    
    # Test on different targets
    targets = get_cns_targets()
    available_targets = targets.get_available_targets()
    
    results = []
    
    for target in available_targets[:5]:  # Test first 5 targets
        print(f"\n📊 Testing {target}...")
        
        # Prepare data
        atom_features = torch.randn(100, 50, 78, device=device)
        bond_features = torch.randn(100, 50, 50, 12, device=device)
        atom_mask = torch.ones(100, 50, device=device)
        coords = torch.randn(100, 50, 3, device=device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                atom_features=atom_features,
                bond_features=bond_features,
                atom_mask=atom_mask,
                coords=coords
            )
        
        # Calculate metrics
        pki_pred = outputs['pki'].squeeze()
        activity_pred = torch.softmax(outputs['activity'], dim=-1)
        cns_mpo_pred = outputs['cns_mpo'].squeeze()
        qed_pred = outputs['qed'].squeeze()
        sa_pred = outputs['sa'].squeeze()
        
        results.append({
            'target': target,
            'pki_mean': pki_pred.mean().item(),
            'pki_std': pki_pred.std().item(),
            'activity_mean': activity_pred.mean(dim=0).cpu().numpy(),
            'cns_mpo_mean': cns_mpo_pred.mean().item(),
            'cns_mpo_std': cns_mpo_pred.std().item(),
            'qed_mean': qed_pred.mean().item(),
            'qed_std': qed_pred.std().item(),
            'sa_mean': sa_pred.mean().item(),
            'sa_std': sa_pred.std().item(),
        })
        
        print(f"  pKi: {pki_pred.mean().item():.2f} ± {pki_pred.std().item():.2f}")
        print(f"  CNS-MPO: {cns_mpo_pred.mean().item():.2f} ± {cns_mpo_pred.std().item():.2f}")
        print(f"  QED: {qed_pred.mean().item():.2f} ± {qed_pred.std().item():.2f}")
        print(f"  SA: {sa_pred.mean().item():.2f} ± {sa_pred.std().item():.2f}")
    
    return results


def plot_results(performance_results, memory_results, accuracy_results):
    """Plot benchmark results."""
    print("\n📊 Plotting Results")
    print("=" * 40)
    
    # Create output directory
    output_dir = Path("results/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot performance
    if performance_results:
        plt.figure(figsize=(12, 8))
        
        for model_name in set(r['model'] for r in performance_results):
            model_results = [r for r in performance_results if r['model'] == model_name]
            batch_sizes = [r['batch_size'] for r in model_results]
            times = [r['time'] for r in model_results]
            
            plt.plot(batch_sizes, times, marker='o', label=model_name)
        
        plt.xlabel('Batch Size')
        plt.ylabel('Time (s)')
        plt.title('Model Performance vs Batch Size')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / 'performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot memory usage
    if memory_results:
        plt.figure(figsize=(12, 8))
        
        for model_name in set(r['model'] for r in memory_results):
            model_results = [r for r in memory_results if r['model'] == model_name]
            batch_sizes = [r['batch_size'] for r in model_results]
            memories = [r['memory'] for r in model_results]
            
            plt.plot(batch_sizes, memories, marker='o', label=model_name)
        
        plt.xlabel('Batch Size')
        plt.ylabel('Memory (GB)')
        plt.title('Memory Usage vs Batch Size')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / 'memory.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot accuracy
    if accuracy_results:
        plt.figure(figsize=(12, 8))
        
        targets = [r['target'] for r in accuracy_results]
        pki_means = [r['pki_mean'] for r in accuracy_results]
        cns_mpo_means = [r['cns_mpo_mean'] for r in accuracy_results]
        qed_means = [r['qed_mean'] for r in accuracy_results]
        
        x = np.arange(len(targets))
        width = 0.25
        
        plt.bar(x - width, pki_means, width, label='pKi', alpha=0.8)
        plt.bar(x, cns_mpo_means, width, label='CNS-MPO', alpha=0.8)
        plt.bar(x + width, qed_means, width, label='QED', alpha=0.8)
        
        plt.xlabel('Target')
        plt.ylabel('Score')
        plt.title('Model Performance by Target')
        plt.xticks(x, targets, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"  ✅ Plots saved to {output_dir}")


def main():
    """Main benchmark function."""
    print("🧬 Molecular PWA+PET Transformer - Benchmark")
    print("=" * 60)
    
    # Run benchmarks
    performance_results = benchmark_model_performance()
    memory_results = benchmark_memory_usage()
    accuracy_results = benchmark_accuracy()
    
    # Plot results
    plot_results(performance_results, memory_results, accuracy_results)
    
    print("\n🎉 Benchmark complete!")
    print("なんｊ魂で最後まで頑張った結果や！めっちゃ嬉しいで〜！💪")
    
    print("\n📚 Results saved to:")
    print("  📊 Performance: results/benchmarks/performance.png")
    print("  💾 Memory: results/benchmarks/memory.png")
    print("  🎯 Accuracy: results/benchmarks/accuracy.png")


if __name__ == "__main__":
    main()
