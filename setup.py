"""
Setup script for chemforge package.
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements file
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="chemforge",
    version="0.1.0",
    author="ChemForge Development Team",
    author_email="chemforge@example.com",
    description="Advanced CNS drug discovery platform with PWA+PET Transformer",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/zapabob/chemforge",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.800",
            "pre-commit>=2.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "nbsphinx>=0.8.0",
            "sphinx-autodoc-typehints>=1.0.0",
        ],
        "gui": [
            "streamlit>=1.0.0",
            "plotly>=5.0.0",
            "dash>=2.0.0",
        ],
        "docking": [
            "rdkit>=2022.03.1",
            "openmm>=7.6.0",
            "mdtraj>=1.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "chemforge=chemforge.cli.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "chemforge": [
            "config/*.yaml",
            "config/targets/*.yaml",
            "data/*.csv",
            "models/*.pt",
        ],
    },
    keywords=[
        "pIC50",
        "QSAR",
        "molecular",
        "prediction",
        "ChEMBL",
        "drug discovery",
        "cheminformatics",
        "transformer",
        "GNN",
        "ensemble",
        "DAT",
        "5HT2A",
        "CB1",
        "CB2",
        "opioid",
        "receptor",
        "pharmacology",
    ],
    project_urls={
        "Homepage": "https://github.com/zapabob/chemforge",
        "Documentation": "https://chemforge.readthedocs.io/",
        "Repository": "https://github.com/zapabob/chemforge.git",
        "Issues": "https://github.com/zapabob/chemforge/issues",
        "Changelog": "https://github.com/zapabob/chemforge/blob/main/CHANGELOG.md",
    },
)