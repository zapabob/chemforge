"""
ChemForge CLI Module

This module provides command-line interface for ChemForge library.
"""

from .main import main
from .train import train_command
from .predict import predict_command
from .admet import admet_command
from .generate import generate_command
from .optimize import optimize_command

__all__ = [
    "main",
    "train_command",
    "predict_command", 
    "admet_command",
    "generate_command",
    "optimize_command"
]