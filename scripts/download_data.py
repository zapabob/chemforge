"""
Data download script for molecular-pwa-pet.
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm


def download_file(url, filename, description="Downloading"):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=description,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = file.write(chunk)
            progress_bar.update(size)


def download_pdsp_data():
    """Download PDSP data."""
    print("📥 Downloading PDSP data...")
    
    # Create data directory
    data_dir = Path("data/raw/pdsp")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # PDSP URLs (example - replace with actual URLs)
    pdsp_urls = {
        "5HT2A": "https://example.com/pdsp/5ht2a_data.csv",
        "5HT1A": "https://example.com/pdsp/5ht1a_data.csv",
        "D1": "https://example.com/pdsp/d1_data.csv",
        "D2": "https://example.com/pdsp/d2_data.csv",
    }
    
    for target, url in pdsp_urls.items():
        filename = data_dir / f"{target.lower()}_data.csv"
        if not filename.exists():
            try:
                download_file(url, filename, f"Downloading {target} data")
                print(f"  ✅ {target} data downloaded")
            except Exception as e:
                print(f"  ⚠️  Failed to download {target} data: {e}")
        else:
            print(f"  ✅ {target} data already exists")


def download_chembl_data():
    """Download ChEMBL data."""
    print("📥 Downloading ChEMBL data...")
    
    # Create data directory
    data_dir = Path("data/raw/chembl")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # ChEMBL URLs (example - replace with actual URLs)
    chembl_urls = {
        "CB1": "https://example.com/chembl/cb1_data.csv",
        "CB2": "https://example.com/chembl/cb2_data.csv",
        "MOR": "https://example.com/chembl/mor_data.csv",
        "DOR": "https://example.com/chembl/dor_data.csv",
        "KOR": "https://example.com/chembl/kor_data.csv",
        "NOP": "https://example.com/chembl/nop_data.csv",
    }
    
    for target, url in chembl_urls.items():
        filename = data_dir / f"{target.lower()}_data.csv"
        if not filename.exists():
            try:
                download_file(url, filename, f"Downloading {target} data")
                print(f"  ✅ {target} data downloaded")
            except Exception as e:
                print(f"  ⚠️  Failed to download {target} data: {e}")
        else:
            print(f"  ✅ {target} data already exists")


def download_moses_data():
    """Download MOSES data."""
    print("📥 Downloading MOSES data...")
    
    # Create data directory
    data_dir = Path("data/raw/moses")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # MOSES URLs
    moses_urls = {
        "train": "https://github.com/molecularsets/moses/raw/master/data/dataset_v1.csv",
        "test": "https://github.com/molecularsets/moses/raw/master/data/test.csv",
        "scaffolds": "https://github.com/molecularsets/moses/raw/master/data/scaffolds.csv",
    }
    
    for name, url in moses_urls.items():
        filename = data_dir / f"{name}.csv"
        if not filename.exists():
            try:
                download_file(url, filename, f"Downloading {name} data")
                print(f"  ✅ {name} data downloaded")
            except Exception as e:
                print(f"  ⚠️  Failed to download {name} data: {e}")
        else:
            print(f"  ✅ {name} data already exists")


def download_guacamol_data():
    """Download GuacaMol data."""
    print("📥 Downloading GuacaMol data...")
    
    # Create data directory
    data_dir = Path("data/raw/guacamol")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # GuacaMol URLs
    guacamol_urls = {
        "train": "https://github.com/BenevolentAI/guacamol/raw/master/guacamol/data/guacamol_v1_train.smiles",
        "valid": "https://github.com/BenevolentAI/guacamol/raw/master/guacamol/data/guacamol_v1_valid.smiles",
        "test": "https://github.com/BenevolentAI/guacamol/raw/master/guacamol/data/guacamol_v1_test.smiles",
    }
    
    for name, url in guacamol_urls.items():
        filename = data_dir / f"{name}.smiles"
        if not filename.exists():
            try:
                download_file(url, filename, f"Downloading {name} data")
                print(f"  ✅ {name} data downloaded")
            except Exception as e:
                print(f"  ⚠️  Failed to download {name} data: {e}")
        else:
            print(f"  ✅ {name} data already exists")


def download_pretrained_models():
    """Download pretrained models."""
    print("📥 Downloading pretrained models...")
    
    # Create models directory
    models_dir = Path("models/pretrained")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Model URLs (example - replace with actual URLs)
    model_urls = {
        "pwa_pet_base": "https://example.com/models/pwa_pet_base.pt",
        "pwa_pet_large": "https://example.com/models/pwa_pet_large.pt",
        "pwa_pet_cns": "https://example.com/models/pwa_pet_cns.pt",
    }
    
    for name, url in model_urls.items():
        filename = models_dir / f"{name}.pt"
        if not filename.exists():
            try:
                download_file(url, filename, f"Downloading {name} model")
                print(f"  ✅ {name} model downloaded")
            except Exception as e:
                print(f"  ⚠️  Failed to download {name} model: {e}")
        else:
            print(f"  ✅ {name} model already exists")


def main():
    """Main download function."""
    print("🧬 Molecular PWA+PET Transformer - Data Download")
    print("=" * 60)
    
    # Download data
    download_pdsp_data()
    download_chembl_data()
    download_moses_data()
    download_guacamol_data()
    download_pretrained_models()
    
    print("\n🎉 Data download complete!")
    print("なんｊ魂で最後まで頑張った結果や！めっちゃ嬉しいで〜！💪")
    
    print("\n📚 Next steps:")
    print("  1. Check downloaded data: ls -la data/raw/")
    print("  2. Check models: ls -la models/pretrained/")
    print("  3. Run preprocessing: python scripts/preprocess_data.py")
    print("  4. Start training: molecular-pwa-pet train --target 5HT2A")


if __name__ == "__main__":
    main()
