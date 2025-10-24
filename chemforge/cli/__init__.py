"""
ChemForge CLI Module

This module provides command-line interface for ChemForge library.
"""

from .main import main
from .train import main as train_main
from .predict import main as predict_main
from .admet import main as admet_main
from .generate import main as generate_main
from .optimize import main as optimize_main

__all__ = [
    "main",
    "train_main",
    "predict_main", 
    "admet_main",
    "generate_main",
    "optimize_main"
]