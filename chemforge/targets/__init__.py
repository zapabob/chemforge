"""
ChEMBL Targets Module

Correct ChEMBL database IDs for human receptors used in pIC50 prediction.
"""

from .chembl_targets import (
    ChEMBLTargets,
    get_chembl_targets,
    get_target_info,
    get_chembl_id,
    get_target_data,
    CHEMBL_TARGETS,
    TARGET_CONFIGS
)

__all__ = [
    'ChEMBLTargets',
    'get_chembl_targets',
    'get_target_info',
    'get_chembl_id',
    'get_target_data',
    'CHEMBL_TARGETS',
    'TARGET_CONFIGS'
]
