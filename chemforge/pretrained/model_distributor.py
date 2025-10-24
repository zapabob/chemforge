"""
Model distributor for pre-trained models.

This module provides functionality for distributing pre-trained models
and making them available for download and use.
"""

import os
import json
import pickle
import shutil
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import tarfile
import zipfile
import torch
import pandas as pd
from packaging import version

from chemforge.utils.logging_utils import Logger
from chemforge.utils.file_utils import FileManager, DataExporter


class ModelDistributor:
    """
    Model distributor for pre-trained models.
    
    This class handles the distribution of pre-trained models,
    including packaging, versioning, and metadata management.
    """
    
    def __init__(
        self,
        models_dir: str = "./pretrained_models",
        distribution_dir: str = "./distributions",
        log_dir: str = "./logs"
    ):
        """
        Initialize the model distributor.
        
        Args:
            models_dir: Directory containing pre-trained models
            distribution_dir: Directory for distribution packages
            log_dir: Directory for logs
        """
        self.models_dir = Path(models_dir)
        self.distribution_dir = Path(distribution_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.distribution_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.logger = Logger('model_distributor', log_dir=str(self.log_dir))
        self.file_manager = FileManager()
        self.data_exporter = DataExporter()
        
        # Distribution metadata
        self.distribution_metadata = {}
        
        self.logger.info("ModelDistributor initialized")
    
    def create_model_package(
        self,
        model_path: str,
        model_name: str,
        version: str = "1.0.0",
        description: str = "",
        author: str = "ChemForge Team",
        license: str = "MIT",
        dependencies: List[str] = None,
        include_data: bool = True,
        include_examples: bool = True
    ) -> str:
        """
        Create a distribution package for a pre-trained model.
        
        Args:
            model_path: Path to the model file
            model_name: Name of the model
            version: Version of the model
            description: Description of the model
            author: Author of the model
            license: License of the model
            dependencies: List of dependencies
            include_data: Whether to include training data
            include_examples: Whether to include usage examples
            
        Returns:
            Path to the created package
        """
        self.logger.info(f"Creating model package for {model_name} v{version}")
        
        # Create package directory
        package_name = f"{model_name}_v{version}"
        package_dir = self.distribution_dir / package_name
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model file
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        shutil.copy2(model_file, package_dir / f"{model_name}.pt")
        
        # Create model metadata
        model_metadata = self._create_model_metadata(
            model_path, model_name, version, description, author, license
        )
        
        # Save metadata
        metadata_path = package_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        # Include training data if requested
        if include_data:
            self._include_training_data(package_dir, model_path)
        
        # Include examples if requested
        if include_examples:
            self._include_usage_examples(package_dir, model_name)
        
        # Create requirements file
        if dependencies:
            self._create_requirements_file(package_dir, dependencies)
        
        # Create README
        self._create_readme(package_dir, model_name, version, description)
        
        # Create package archive
        package_archive = self._create_package_archive(package_dir, package_name)
        
        # Store distribution metadata
        self.distribution_metadata[package_name] = {
            'model_name': model_name,
            'version': version,
            'package_path': str(package_archive),
            'created_date': datetime.now().isoformat(),
            'metadata': model_metadata
        }
        
        self.logger.info(f"Created model package: {package_archive}")
        
        return str(package_archive)
    
    def _create_model_metadata(
        self,
        model_path: str,
        model_name: str,
        version: str,
        description: str,
        author: str,
        license: str
    ) -> Dict[str, Any]:
        """Create metadata for the model."""
        # Load model to get configuration
        model_data = torch.load(model_path, map_location='cpu')
        
        # Calculate file hash
        file_hash = self._calculate_file_hash(model_path)
        
        # Get file size
        file_size = os.path.getsize(model_path)
        
        metadata = {
            'model_name': model_name,
            'version': version,
            'description': description,
            'author': author,
            'license': license,
            'created_date': datetime.now().isoformat(),
            'file_info': {
                'path': model_path,
                'size_bytes': file_size,
                'hash_md5': file_hash
            },
            'model_info': {
                'model_config': model_data.get('model_config', {}),
                'training_config': model_data.get('training_config', {}),
                'data_info': model_data.get('data_info', {}),
                'training_results': model_data.get('training_results', {})
            },
            'requirements': {
                'python_version': '>=3.8',
                'torch_version': '>=1.9.0',
                'chemforge_version': '>=0.1.0'
            }
        }
        
        return metadata
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _include_training_data(self, package_dir: Path, model_path: str):
        """Include training data in the package."""
        # Look for associated data files
        model_dir = Path(model_path).parent
        data_files = list(model_dir.glob("*dataset*.pkl")) + list(model_dir.glob("*data*.pkl"))
        
        if data_files:
            data_dir = package_dir / "data"
            data_dir.mkdir(exist_ok=True)
            
            for data_file in data_files:
                shutil.copy2(data_file, data_dir / data_file.name)
            
            self.logger.info(f"Included {len(data_files)} data files in package")
    
    def _include_usage_examples(self, package_dir: Path, model_name: str):
        """Include usage examples in the package."""
        examples_dir = package_dir / "examples"
        examples_dir.mkdir(exist_ok=True)
        
        # Create basic usage example
        example_code = f'''"""
Usage example for {model_name} pre-trained model.
"""

import torch
import json
from chemforge.models.transformer_model import TransformerModel
from chemforge.models.gnn_model import GNNModel
from chemforge.models.ensemble_model import EnsembleModel

def load_pretrained_model(model_path: str, model_type: str = "transformer"):
    """
    Load a pre-trained model.
    
    Args:
        model_path: Path to the model file
        model_type: Type of model ('transformer', 'gnn', 'ensemble')
    
    Returns:
        Loaded model
    """
    # Load model data
    model_data = torch.load(model_path, map_location='cpu')
    
    # Get model configuration
    model_config = model_data['model_config']
    
    # Initialize model based on type
    if model_type == "transformer":
        model = TransformerModel(**model_config)
    elif model_type == "gnn":
        model = GNNModel(**model_config)
    elif model_type == "ensemble":
        model = EnsembleModel(**model_config)
    else:
        raise ValueError(f"Unknown model type: {{model_type}}")
    
    # Load state dict
    model.load_state_dict(model_data['model_state_dict'])
    model.eval()
    
    return model

def make_predictions(model, features):
    """
    Make predictions using the pre-trained model.
    
    Args:
        model: Pre-trained model
        features: Input features
    
    Returns:
        Model predictions
    """
    with torch.no_grad():
        predictions = model(features)
    
    return predictions

# Example usage
if __name__ == "__main__":
    # Load model
    model = load_pretrained_model("{model_name}.pt")
    
    # Example features (replace with actual features)
    features = torch.randn(1, 200)  # Example: 1 sample, 200 features
    
    # Make predictions
    predictions = make_predictions(model, features)
    print(f"Predictions: {{predictions}}")
'''
        
        example_path = examples_dir / "usage_example.py"
        with open(example_path, 'w') as f:
            f.write(example_code)
        
        # Create advanced usage example
        advanced_example = f'''"""
Advanced usage example for {model_name} pre-trained model.
"""

import torch
import pandas as pd
import numpy as np
from chemforge.data.molecular_features import MolecularFeatures
from chemforge.data.rdkit_descriptors import RDKitDescriptors

def preprocess_molecules(smiles_list):
    """
    Preprocess molecules for prediction.
    
    Args:
        smiles_list: List of SMILES strings
    
    Returns:
        Preprocessed features
    """
    # Create molecular features
    mol_features = MolecularFeatures()
    rdkit_descriptors = RDKitDescriptors()
    
    # Create molecules DataFrame
    molecules = pd.DataFrame({{'smiles': smiles_list}})
    
    # Extract features
    basic_features = mol_features.extract_features(molecules)
    rdkit_features = rdkit_descriptors.calculate_descriptors(molecules)
    
    # Combine features
    if len(basic_features) > 0 and len(rdkit_features) > 0:
        features = np.hstack([basic_features, rdkit_features])
    elif len(basic_features) > 0:
        features = basic_features
    elif len(rdkit_features) > 0:
        features = rdkit_features
    else:
        raise ValueError("No features extracted")
    
    return torch.tensor(features, dtype=torch.float32)

def batch_predictions(model, smiles_list, batch_size=32):
    """
    Make batch predictions for a list of molecules.
    
    Args:
        model: Pre-trained model
        smiles_list: List of SMILES strings
        batch_size: Batch size for processing
    
    Returns:
        Predictions for all molecules
    """
    all_predictions = []
    
    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i:i+batch_size]
        batch_features = preprocess_molecules(batch_smiles)
        
        with torch.no_grad():
            batch_predictions = model(batch_features)
        
        all_predictions.append(batch_predictions.numpy())
    
    return np.concatenate(all_predictions, axis=0)

# Example usage
if __name__ == "__main__":
    # Example SMILES
    smiles_list = [
        "CCO",  # Ethanol
        "CCN",  # Ethylamine
        "CC(C)O",  # Isopropanol
        "CC(C)N"  # Isopropylamine
    ]
    
    # Load model
    model = load_pretrained_model("{model_name}.pt")
    
    # Make predictions
    predictions = batch_predictions(model, smiles_list)
    print(f"Predictions shape: {{predictions.shape}}")
    print(f"Predictions: {{predictions}}")
'''
        
        advanced_path = examples_dir / "advanced_usage.py"
        with open(advanced_path, 'w') as f:
            f.write(advanced_example)
        
        self.logger.info("Created usage examples")
    
    def _create_requirements_file(self, package_dir: Path, dependencies: List[str]):
        """Create requirements.txt file for the package."""
        requirements_path = package_dir / "requirements.txt"
        
        with open(requirements_path, 'w') as f:
            f.write("# Core dependencies\n")
            f.write("torch>=1.9.0\n")
            f.write("numpy>=1.21.0\n")
            f.write("pandas>=1.3.0\n")
            f.write("scikit-learn>=1.0.0\n")
            f.write("rdkit>=2022.3.0\n")
            f.write("chemforge>=0.1.0\n")
            f.write("\n# Additional dependencies\n")
            for dep in dependencies:
                f.write(f"{dep}\n")
        
        self.logger.info("Created requirements.txt")
    
    def _create_readme(self, package_dir: Path, model_name: str, version: str, description: str):
        """Create README.md file for the package."""
        readme_content = f"""# {model_name} Pre-trained Model

**Version:** {version}  
**Description:** {description}

## Overview

This package contains a pre-trained {model_name} model for CNS drug discovery applications.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
import torch
from chemforge.models.transformer_model import TransformerModel

# Load the model
model_data = torch.load('{model_name}.pt')
model = TransformerModel(**model_data['model_config'])
model.load_state_dict(model_data['model_state_dict'])
model.eval()

# Make predictions
features = torch.randn(1, 200)  # Example features
predictions = model(features)
```

### Advanced Usage

See the examples directory for more detailed usage examples.

## Model Information

- **Model Type:** {model_name}
- **Version:** {version}
- **Training Data:** ChEMBL database
- **Targets:** CNS receptors and transporters
- **Features:** Molecular descriptors and RDKit features

## Performance

The model performance metrics are included in the model metadata.

## License

This model is distributed under the MIT License.

## Citation

If you use this model in your research, please cite:

```
ChemForge: A Comprehensive Platform for CNS Drug Discovery
ChemForge Development Team
https://github.com/zapabob/chemforge
```

## Support

For questions and support, please visit the ChemForge GitHub repository.
"""
        
        readme_path = package_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        self.logger.info("Created README.md")
    
    def _create_package_archive(self, package_dir: Path, package_name: str) -> Path:
        """Create a compressed archive of the package."""
        archive_path = self.distribution_dir / f"{package_name}.tar.gz"
        
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(package_dir, arcname=package_name)
        
        self.logger.info(f"Created package archive: {archive_path}")
        
        return archive_path
    
    def create_distribution_catalog(
        self,
        catalog_name: str = "chemforge_models",
        description: str = "ChemForge Pre-trained Models Catalog"
    ) -> str:
        """
        Create a catalog of all available model distributions.
        
        Args:
            catalog_name: Name of the catalog
            description: Description of the catalog
            
        Returns:
            Path to the catalog file
        """
        self.logger.info(f"Creating distribution catalog: {catalog_name}")
        
        # Create catalog metadata
        catalog_metadata = {
            'catalog_name': catalog_name,
            'description': description,
            'created_date': datetime.now().isoformat(),
            'version': '1.0.0',
            'models': list(self.distribution_metadata.values())
        }
        
        # Save catalog
        catalog_path = self.distribution_dir / f"{catalog_name}.json"
        with open(catalog_path, 'w') as f:
            json.dump(catalog_metadata, f, indent=2)
        
        # Create catalog README
        readme_content = f"""# {catalog_name}

{description}

## Available Models

"""
        
        for model_info in self.distribution_metadata.values():
            readme_content += f"""### {model_info['model_name']} v{model_info['version']}

- **Description:** {model_info['metadata'].get('description', 'No description available')}
- **Author:** {model_info['metadata'].get('author', 'Unknown')}
- **License:** {model_info['metadata'].get('license', 'Unknown')}
- **Created:** {model_info['created_date']}
- **Package:** {model_info['package_path']}

"""
        
        readme_content += """
## Installation

Download the desired model package and follow the installation instructions in each package.

## Usage

Each model package includes detailed usage examples and documentation.

## Support

For questions and support, please visit the ChemForge GitHub repository.
"""
        
        readme_path = self.distribution_dir / f"{catalog_name}_README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        self.logger.info(f"Created distribution catalog: {catalog_path}")
        
        return str(catalog_path)
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available model distributions."""
        return list(self.distribution_metadata.values())
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        for model_info in self.distribution_metadata.values():
            if model_info['model_name'] == model_name:
                return model_info
        return None
    
    def validate_model_package(self, package_path: str) -> Dict[str, Any]:
        """
        Validate a model package.
        
        Args:
            package_path: Path to the model package
            
        Returns:
            Validation results
        """
        self.logger.info(f"Validating model package: {package_path}")
        
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'package_info': {}
        }
        
        try:
            # Check if package exists
            if not os.path.exists(package_path):
                validation_results['valid'] = False
                validation_results['errors'].append(f"Package not found: {package_path}")
                return validation_results
            
            # Extract package if it's an archive
            if package_path.endswith('.tar.gz'):
                with tarfile.open(package_path, 'r:gz') as tar:
                    # Check for required files
                    required_files = ['model_metadata.json', 'README.md']
                    file_list = tar.getnames()
                    
                    for req_file in required_files:
                        if req_file not in file_list:
                            validation_results['warnings'].append(f"Missing file: {req_file}")
            
            # Load and validate metadata
            if os.path.exists(package_path.replace('.tar.gz', '/model_metadata.json')):
                metadata_path = package_path.replace('.tar.gz', '/model_metadata.json')
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                validation_results['package_info'] = metadata
                
                # Validate metadata structure
                required_fields = ['model_name', 'version', 'description', 'author']
                for field in required_fields:
                    if field not in metadata:
                        validation_results['warnings'].append(f"Missing metadata field: {field}")
            
        except Exception as e:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Validation error: {str(e)}")
        
        self.logger.info(f"Package validation completed: {validation_results['valid']}")
        
        return validation_results
    
    def cleanup_old_versions(self, model_name: str, keep_latest: int = 3):
        """
        Clean up old versions of a model, keeping only the latest N versions.
        
        Args:
            model_name: Name of the model
            keep_latest: Number of latest versions to keep
        """
        self.logger.info(f"Cleaning up old versions of {model_name}")
        
        # Find all versions of the model
        model_versions = []
        for package_name, model_info in self.distribution_metadata.items():
            if model_info['model_name'] == model_name:
                model_versions.append((package_name, model_info))
        
        # Sort by version (assuming semantic versioning)
        model_versions.sort(key=lambda x: version.parse(x[1]['version']), reverse=True)
        
        # Remove old versions
        if len(model_versions) > keep_latest:
            versions_to_remove = model_versions[keep_latest:]
            
            for package_name, model_info in versions_to_remove:
                package_path = model_info['package_path']
                
                # Remove package file
                if os.path.exists(package_path):
                    os.remove(package_path)
                    self.logger.info(f"Removed old version: {package_path}")
                
                # Remove from metadata
                del self.distribution_metadata[package_name]
    
    def get_distribution_summary(self) -> Dict[str, Any]:
        """Get a summary of all distributions."""
        summary = {
            'total_models': len(self.distribution_metadata),
            'models_by_name': {},
            'latest_versions': {},
            'distribution_metadata': self.distribution_metadata
        }
        
        # Group models by name
        for model_info in self.distribution_metadata.values():
            model_name = model_info['model_name']
            if model_name not in summary['models_by_name']:
                summary['models_by_name'][model_name] = []
            summary['models_by_name'][model_name].append(model_info)
        
        # Find latest versions
        for model_name, versions in summary['models_by_name'].items():
            latest_version = max(versions, key=lambda x: version.parse(x['version']))
            summary['latest_versions'][model_name] = latest_version
        
        return summary
