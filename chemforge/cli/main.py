"""
ChemForge CLI Main Module

This module provides the main CLI interface for ChemForge library.
"""

import click
import sys
from pathlib import Path
from typing import Optional

from .train import train_command
from .predict import predict_command
from .admet import admet_command
from .generate import generate_command
from .optimize import optimize_command


@click.group()
@click.version_option(version="0.1.0", prog_name="chemforge")
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose output"
)
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file"
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(path_type=Path),
    default=Path("output"),
    help="Output directory for results"
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda", "mps"]),
    default="auto",
    help="Device to use for computation"
)
@click.pass_context
def main(
    ctx: click.Context,
    verbose: bool,
    config: Optional[Path],
    output_dir: Path,
    device: str
) -> None:
    """
    ChemForge: Advanced CNS drug discovery platform with PWA+PET Transformer
    
    A comprehensive drug discovery platform for CNS targets including:
    - Multi-target pIC50 prediction
    - ADMET property prediction
    - Molecular generation and optimization
    - Scaffold detection and analysis
    """
    # Set context variables
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["config"] = config
    ctx.obj["output_dir"] = output_dir
    ctx.obj["device"] = device
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        click.echo(f"ChemForge CLI initialized")
        click.echo(f"Output directory: {output_dir}")
        click.echo(f"Device: {device}")


# Add subcommands
main.add_command(train_command)
main.add_command(predict_command)
main.add_command(admet_command)
main.add_command(generate_command)
main.add_command(optimize_command)


if __name__ == "__main__":
    main()
