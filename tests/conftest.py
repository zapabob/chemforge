"""
Test configuration and fixtures for molecular-pwa-pet.
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any

from molecular_pwa_pet import MolecularPWA_PETTransformer
from molecular_pwa_pet.targets import get_cns_targets


@pytest.fixture
def sample_molecular_data():
    """Sample molecular data for testing."""
    batch_size = 2
    max_atoms = 50
    atom_features = 78
    bond_features = 12
    
    atom_features = torch.randn(batch_size, max_atoms, atom_features)
    bond_features = torch.randn(batch_size, max_atoms, max_atoms, bond_features)
    atom_mask = torch.ones(batch_size, max_atoms)
    coords = torch.randn(batch_size, max_atoms, 3)
    
    return {
        'atom_features': atom_features,
        'bond_features': bond_features,
        'atom_mask': atom_mask,
        'coords': coords
    }


@pytest.fixture
def sample_model():
    """Sample model for testing."""
    return MolecularPWA_PETTransformer(
        d_model=512,
        n_layers=4,
        n_heads=8,
        max_atoms=50,
        atom_features=78,
        bond_features=12
    )


@pytest.fixture
def cns_targets():
    """CNS targets for testing."""
    return get_cns_targets()


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'model': {
            'd_model': 512,
            'n_layers': 4,
            'n_heads': 8,
            'max_atoms': 50,
            'atom_features': 78,
            'bond_features': 12,
            'buckets': {"trivial": 1, "fund": 5, "adj": 2},
            'pet_curv_reg': 1e-6
        },
        'training': {
            'epochs': 10,
            'batch_size': 256,
            'learning_rate': 3e-4,
            'weight_decay': 0.03
        },
        'data': {
            'max_atoms': 50,
            'atom_features': 78,
            'bond_features': 12
        }
    }


@pytest.fixture
def sample_smiles():
    """Sample SMILES strings for testing."""
    return [
        "CCN(CC)CCc1ccc(O)cc1",
        "COc1ccc(CCN(C)C)cc1",
        "CCN(CC)CCc1ccc(OC)cc1",
        "COc1ccc(CCNCC)cc1",
        "CCN(CC)CCc1ccc(F)cc1"
    ]


@pytest.fixture
def sample_targets():
    """Sample CNS targets for testing."""
    return ['5HT2A', '5HT1A', 'D1', 'D2', 'CB1', 'CB2', 'MOR', 'DOR']


@pytest.fixture
def sample_metrics():
    """Sample evaluation metrics for testing."""
    return {
        'accuracy': 0.9902,
        'rmse': 0.15,
        'r2': 0.95,
        'cns_mpo': 4.2,
        'qed': 0.7,
        'sa': 5.8
    }


@pytest.fixture
def sample_optimization_goals():
    """Sample optimization goals for testing."""
    return {
        '5HT2A': ['pki', 'cns_mpo', 'qed', 'sa'],
        'D1': ['pki', 'selectivity', 'cns_mpo', 'qed'],
        'CB1': ['pki', 'selectivity', 'cns_mpo', 'qed', 'sa', 'safety'],
        'MOR': ['pki', 'selectivity', 'cns_mpo', 'qed', 'sa', 'safety', 'addiction_potential']
    }


@pytest.fixture
def sample_selectivity_targets():
    """Sample selectivity targets for testing."""
    return {
        'D1': {'D1': 6.0, 'D2': 4.0},
        'CB1': {'CB1': 6.0, 'CB2': 4.0},
        'MOR': {'MOR': 6.0, 'DOR': 4.0, 'KOR': 4.0}
    }


@pytest.fixture
def sample_safety_targets():
    """Sample safety targets for testing."""
    return {
        'D1': {'psychosis': 'low', 'dyskinesia': 'low', 'tolerance': 'low'},
        'CB1': {'psychoactivity': 'low', 'tolerance': 'low', 'withdrawal': 'low'},
        'MOR': {'addiction_potential': 'low', 'respiratory_depression': 'low', 'tolerance': 'low'}
    }
