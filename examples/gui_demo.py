"""
ChemForge GUI Demo

This module demonstrates the usage of ChemForge GUI applications
including Streamlit and Dash interfaces.
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import pandas as pd
import numpy as np

from chemforge.utils.logging_utils import setup_logging


def run_streamlit_demo():
    """Run Streamlit demo."""
    print("Starting Streamlit Demo...")
    print("=" * 50)
    
    # Setup logging
    log_manager = setup_logging(level='INFO')
    logger = log_manager.get_logger('gui_demo')
    logger.info("Starting ChemForge Streamlit Demo")
    
    # Create demo data
    demo_data = create_demo_data()
    
    # Save demo data
    demo_dir = Path("./gui_demo")
    demo_dir.mkdir(exist_ok=True)
    
    data_path = demo_dir / "demo_data.csv"
    demo_data.to_csv(data_path, index=False)
    print(f"Demo data saved to {data_path}")
    
    # Instructions for running Streamlit
    print("\nStreamlit Demo Instructions:")
    print("1. Open a new terminal")
    print("2. Navigate to the project directory")
    print("3. Run: streamlit run chemforge/gui/streamlit_app.py")
    print("4. Open the URL shown in the terminal (usually http://localhost:8501)")
    print("5. Upload the demo data file or enter SMILES strings")
    print("6. Explore the different tabs and features")
    
    # Try to run Streamlit automatically
    try:
        print("\nAttempting to start Streamlit automatically...")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "chemforge/gui/streamlit_app.py", "--server.port", "8501"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Could not start Streamlit automatically: {e}")
        print("Please run the command manually as shown above")
    except FileNotFoundError:
        print("Streamlit not found. Please install it with: pip install streamlit")
        print("Then run the command manually as shown above")


def run_dash_demo():
    """Run Dash demo."""
    print("Starting Dash Demo...")
    print("=" * 50)
    
    # Setup logging
    log_manager = setup_logging(level='INFO')
    logger = log_manager.get_logger('gui_demo')
    logger.info("Starting ChemForge Dash Demo")
    
    # Create demo data
    demo_data = create_demo_data()
    
    # Save demo data
    demo_dir = Path("./gui_demo")
    demo_dir.mkdir(exist_ok=True)
    
    data_path = demo_dir / "demo_data.csv"
    demo_data.to_csv(data_path, index=False)
    print(f"Demo data saved to {data_path}")
    
    # Instructions for running Dash
    print("\nDash Demo Instructions:")
    print("1. Open a new terminal")
    print("2. Navigate to the project directory")
    print("3. Run: python chemforge/gui/dash_app.py")
    print("4. Open the URL shown in the terminal (usually http://localhost:8050)")
    print("5. Upload the demo data file or enter SMILES strings")
    print("6. Explore the different tabs and features")
    
    # Try to run Dash automatically
    try:
        print("\nAttempting to start Dash automatically...")
        from chemforge.gui.dash_app import run_dash_app
        run_dash_app()
    except ImportError as e:
        print(f"Could not start Dash automatically: {e}")
        print("Please run the command manually as shown above")
    except Exception as e:
        print(f"Error starting Dash: {e}")
        print("Please run the command manually as shown above")


def create_demo_data():
    """Create demo data for GUI applications."""
    print("Creating demo data...")
    
    # Create sample molecules with properties
    molecules = [
        'CCO',  # Ethanol
        'CCN',  # Ethylamine
        'CC(C)O',  # Isopropanol
        'CC(C)N',  # Isopropylamine
        'CC(C)(C)O',  # tert-Butanol
        'CC(C)(C)N',  # tert-Butylamine
        'CCOC',  # Ethyl methyl ether
        'CCNC',  # N-Methylethylamine
        'CC(C)OC',  # Isopropyl methyl ether
        'CC(C)NC',  # N-Methylisopropylamine
        'C1=CC=CC=C1',  # Benzene
        'C1=CC=CC=C1O',  # Phenol
        'C1=CC=CC=C1N',  # Aniline
        'C1=CC=CC=C1C',  # Toluene
        'C1=CC=CC=C1CC',  # Ethylbenzene
    ]
    
    # Create properties for each molecule
    data = []
    for i, smiles in enumerate(molecules):
        # Generate realistic properties
        mw = np.random.uniform(50, 500)
        logp = np.random.uniform(-2, 5)
        hbd = np.random.randint(0, 5)
        hba = np.random.randint(0, 10)
        tpsa = np.random.uniform(0, 200)
        
        data.append({
            'ID': f'Molecule_{i+1}',
            'SMILES': smiles,
            'MW': round(mw, 2),
            'LogP': round(logp, 2),
            'HBD': hbd,
            'HBA': hba,
            'TPSA': round(tpsa, 2),
            'Length': len(smiles)
        })
    
    return pd.DataFrame(data)


def run_comparison_demo():
    """Run comparison demo between Streamlit and Dash."""
    print("ChemForge GUI Comparison Demo")
    print("=" * 50)
    
    print("This demo compares Streamlit and Dash implementations:")
    print()
    print("Streamlit Features:")
    print("- Simple, Pythonic interface")
    print("- Automatic widget state management")
    print("- Built-in caching and session state")
    print("- Easy deployment with Streamlit Cloud")
    print("- Great for rapid prototyping")
    print()
    print("Dash Features:")
    print("- More flexible layout control")
    print("- Custom CSS and styling")
    print("- Better performance for large datasets")
    print("- More control over callbacks")
    print("- Better for production applications")
    print()
    print("Both implementations provide:")
    print("- Molecular analysis and visualization")
    print("- AI predictions and ADMET analysis")
    print("- Molecular generation capabilities")
    print("- Data management and export")
    print("- Interactive plots and charts")
    print()
    print("Choose the implementation that best fits your needs!")


def run_gui_tests():
    """Run GUI tests."""
    print("Running GUI Tests...")
    print("=" * 50)
    
    # Setup logging
    log_manager = setup_logging(level='INFO')
    logger = log_manager.get_logger('gui_demo')
    logger.info("Running ChemForge GUI Tests")
    
    try:
        # Run Streamlit tests
        print("Running Streamlit tests...")
        import subprocess
        result = subprocess.run([
            sys.executable, "-m", "pytest", "tests/test_gui_streamlit_app.py", "-v"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Streamlit tests passed")
        else:
            print("❌ Streamlit tests failed")
            print(result.stdout)
            print(result.stderr)
        
        # Run Dash tests
        print("\nRunning Dash tests...")
        result = subprocess.run([
            sys.executable, "-m", "pytest", "tests/test_gui_dash_app.py", "-v"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dash tests passed")
        else:
            print("❌ Dash tests failed")
            print(result.stdout)
            print(result.stderr)
        
        # Run GUI utils tests
        print("\nRunning GUI utils tests...")
        result = subprocess.run([
            sys.executable, "-m", "pytest", "tests/test_gui_gui_utils.py", "-v"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ GUI utils tests passed")
        else:
            print("❌ GUI utils tests failed")
            print(result.stdout)
            print(result.stderr)
        
        print("\nGUI tests completed!")
        
    except Exception as e:
        print(f"Error running tests: {e}")
        print("Please run tests manually with: pytest tests/test_gui_*.py -v")


def main():
    """Main demo function."""
    print("ChemForge GUI Demo")
    print("=" * 50)
    print()
    print("Available demos:")
    print("1. Streamlit Demo")
    print("2. Dash Demo")
    print("3. Comparison Demo")
    print("4. Run Tests")
    print("5. All Demos")
    print()
    
    choice = input("Enter your choice (1-5): ").strip()
    
    if choice == "1":
        run_streamlit_demo()
    elif choice == "2":
        run_dash_demo()
    elif choice == "3":
        run_comparison_demo()
    elif choice == "4":
        run_gui_tests()
    elif choice == "5":
        print("Running all demos...")
        run_streamlit_demo()
        print("\n" + "="*50 + "\n")
        run_dash_demo()
        print("\n" + "="*50 + "\n")
        run_comparison_demo()
        print("\n" + "="*50 + "\n")
        run_gui_tests()
    else:
        print("Invalid choice. Please run the demo again.")


if __name__ == "__main__":
    main()
