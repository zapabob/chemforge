"""
Environment setup script for molecular-pwa-pet.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def check_python_version():
    """Check Python version."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA {torch.version.cuda} available")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  CUDA not available, using CPU")
    except ImportError:
        print("⚠️  PyTorch not installed")


def install_requirements():
    """Install requirements."""
    print("\n📦 Installing requirements...")
    
    # Install basic requirements
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Install development requirements if requested
    if "--dev" in sys.argv:
        print("📦 Installing development requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements-dev.txt"])
    
    # Install pre-commit hooks if available
    if Path(".pre-commit-config.yaml").exists():
        print("📦 Installing pre-commit hooks...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pre-commit"])
        subprocess.run(["pre-commit", "install"])


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = [
        "data/raw",
        "data/processed",
        "data/external",
        "models/pretrained",
        "models/checkpoints",
        "results/experiments",
        "results/benchmarks",
        "logs",
        "outputs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {directory}")


def setup_git_hooks():
    """Setup git hooks."""
    print("\n🔧 Setting up git hooks...")
    
    if Path(".git").exists():
        # Install pre-commit hooks
        if Path(".pre-commit-config.yaml").exists():
            subprocess.run(["pre-commit", "install"])
            print("  ✅ Pre-commit hooks installed")
        else:
            print("  ⚠️  No pre-commit configuration found")
    else:
        print("  ⚠️  Not a git repository")


def check_dependencies():
    """Check optional dependencies."""
    print("\n🔍 Checking optional dependencies...")
    
    optional_deps = {
        "rdkit": "RDKit for molecular processing",
        "openmm": "OpenMM for molecular dynamics",
        "mdtraj": "MDTraj for trajectory analysis",
        "sphinx": "Sphinx for documentation",
        "jupyter": "Jupyter for notebooks"
    }
    
    for dep, description in optional_deps.items():
        try:
            __import__(dep)
            print(f"  ✅ {dep}: {description}")
        except ImportError:
            print(f"  ⚠️  {dep}: {description} (not installed)")


def main():
    """Main setup function."""
    print("🧬 Molecular PWA+PET Transformer - Environment Setup")
    print("=" * 60)
    
    # Check Python version
    check_python_version()
    
    # Check CUDA
    check_cuda()
    
    # Install requirements
    install_requirements()
    
    # Create directories
    create_directories()
    
    # Setup git hooks
    setup_git_hooks()
    
    # Check dependencies
    check_dependencies()
    
    print("\n🎉 Environment setup complete!")
    print("なんｊ魂で最後まで頑張った結果や！めっちゃ嬉しいで〜！💪")
    
    print("\n📚 Next steps:")
    print("  1. Run tests: python -m pytest tests/")
    print("  2. Try examples: python examples/basic_usage.py")
    print("  3. Read documentation: docs/")
    print("  4. Start developing: molecular_pwa_pet/")


if __name__ == "__main__":
    main()
