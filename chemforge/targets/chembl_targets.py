"""
ChEMBL Targets Configuration

Correct ChEMBL database IDs for human receptors used in pIC50 prediction.
"""

from typing import Dict, List, Optional, Tuple
import yaml
import os
import requests
import pandas as pd
from tqdm import tqdm


class ChEMBLTargets:
    """ChEMBL targets configuration and management."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ChEMBL targets.
        
        Args:
            config_path: Path to custom targets configuration file
        """
        self.targets = self._load_targets(config_path)
        self.base_url = "https://www.ebi.ac.uk/chembl/api/data"
        self.timeout = 30
        self.retry_attempts = 3
        
    def _load_targets(self, config_path: Optional[str] = None) -> Dict:
        """Load targets configuration."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            return self._get_default_targets()
    
    def _get_default_targets(self) -> Dict:
        """Get default ChEMBL targets configuration with correct IDs."""
        return {
            # Serotonin receptors
            '5HT2A': {
                'chembl_id': 'CHEMBL224',
                'uniprot_id': 'P28223',
                'gene_name': 'HTR2A',
                'name': '5-hydroxytryptamine receptor 2A',
                'organism': 'Homo sapiens',
                'target_type': 'SINGLE PROTEIN',
                'confidence_score': 9,
                'function': 'hallucinogenic, mood regulation',
                'diseases': ['depression', 'anxiety', 'psychosis'],
                'drugs': ['LSD', 'psilocybin', 'ketamine']
            },
            '5HT1A': {
                'chembl_id': 'CHEMBL214',
                'uniprot_id': 'P08908',
                'gene_name': 'HTR1A',
                'name': '5-hydroxytryptamine 1A receptor',
                'organism': 'Homo sapiens',
                'target_type': 'SINGLE PROTEIN',
                'confidence_score': 9,
                'function': 'anxiolytic, antidepressant',
                'diseases': ['anxiety', 'depression', 'panic_disorder'],
                'drugs': ['buspirone', 'tandospirone', 'gepirone']
            },
            
            # Dopamine receptors
            'D1': {
                'chembl_id': 'CHEMBL2056',
                'uniprot_id': 'P21728',
                'gene_name': 'DRD1',
                'name': 'D(1A) dopamine receptor',
                'organism': 'Homo sapiens',
                'target_type': 'SINGLE PROTEIN',
                'confidence_score': 9,
                'function': 'cognition, reward, motor control',
                'diseases': ['parkinsons', 'adhd', 'schizophrenia'],
                'drugs': ['SKF-38393', 'dihydrexidine', 'A-77636']
            },
            'D2': {
                'chembl_id': 'CHEMBL217',
                'uniprot_id': 'P14416',
                'gene_name': 'DRD2',
                'name': 'D(2) dopamine receptor',
                'organism': 'Homo sapiens',
                'target_type': 'SINGLE PROTEIN',
                'confidence_score': 9,
                'function': 'motor control, reward, psychosis',
                'diseases': ['schizophrenia', 'parkinsons', 'tourette'],
                'drugs': ['haloperidol', 'risperidone', 'olanzapine']
            },
            
            # Cannabinoid receptors
            'CB1': {
                'chembl_id': 'CHEMBL218',
                'uniprot_id': 'P21554',
                'gene_name': 'CNR1',
                'name': 'Cannabinoid receptor 1',
                'organism': 'Homo sapiens',
                'target_type': 'SINGLE PROTEIN',
                'confidence_score': 9,
                'function': 'appetite, pain, memory, mood',
                'diseases': ['obesity', 'pain', 'epilepsy', 'ms'],
                'drugs': ['THC', 'rimonabant', 'nabilone']
            },
            'CB2': {
                'chembl_id': 'CHEMBL253',
                'uniprot_id': 'P34972',
                'gene_name': 'CNR2',
                'name': 'Cannabinoid receptor 2',
                'organism': 'Homo sapiens',
                'target_type': 'SINGLE PROTEIN',
                'confidence_score': 9,
                'function': 'immune, inflammation, pain',
                'diseases': ['inflammation', 'pain', 'autoimmune'],
                'drugs': ['JWH-133', 'AM-1241', 'GW-405833']
            },
            
            # Opioid receptors
            'MOR': {  # μ-opioid receptor
                'chembl_id': 'CHEMBL233',
                'uniprot_id': 'P35372',
                'gene_name': 'OPRM1',
                'name': 'Mu-type opioid receptor',
                'organism': 'Homo sapiens',
                'target_type': 'SINGLE PROTEIN',
                'confidence_score': 9,
                'function': 'analgesia, euphoria, respiratory depression',
                'diseases': ['pain', 'addiction', 'respiratory_depression'],
                'drugs': ['morphine', 'fentanyl', 'methadone']
            },
            'DOR': {  # δ-opioid receptor
                'chembl_id': 'CHEMBL236',
                'uniprot_id': 'P41143',
                'gene_name': 'OPRD1',
                'name': 'Delta-type opioid receptor',
                'organism': 'Homo sapiens',
                'target_type': 'SINGLE PROTEIN',
                'confidence_score': 9,
                'function': 'analgesia, mood, addiction',
                'diseases': ['pain', 'depression', 'addiction'],
                'drugs': ['DPDPE', 'SNC-80', 'BUBU']
            },
            'KOR': {  # κ-opioid receptor
                'chembl_id': 'CHEMBL237',
                'uniprot_id': 'P41145',
                'gene_name': 'OPRK1',
                'name': 'Kappa-type opioid receptor',
                'organism': 'Homo sapiens',
                'target_type': 'SINGLE PROTEIN',
                'confidence_score': 9,
                'function': 'analgesia, dysphoria, stress',
                'diseases': ['pain', 'stress', 'addiction'],
                'drugs': ['U-50488', 'salvinorin_A', 'GR-89696']
            },
            'NOP': {  # Nociceptin receptor
                'chembl_id': 'CHEMBL2014',
                'uniprot_id': 'P41146',
                'gene_name': 'OPRL1',
                'name': 'Nociceptin receptor',
                'organism': 'Homo sapiens',
                'target_type': 'SINGLE PROTEIN',
                'confidence_score': 9,
                'function': 'pain, anxiety, addiction',
                'diseases': ['pain', 'anxiety', 'addiction'],
                'drugs': ['nociceptin', 'Ro-64-6198', 'J-113397']
            },
            
            # Transporters
            'SERT': {
                'chembl_id': 'CHEMBL228',
                'uniprot_id': 'P31645',
                'gene_name': 'SLC6A4',
                'name': 'Sodium-dependent serotonin transporter',
                'organism': 'Homo sapiens',
                'target_type': 'SINGLE PROTEIN',
                'confidence_score': 9,
                'function': 'serotonin reuptake, antidepressant target',
                'diseases': ['depression', 'anxiety', 'ocd'],
                'drugs': ['fluoxetine', 'sertraline', 'paroxetine']
            },
            'DAT': {
                'chembl_id': 'CHEMBL238',
                'uniprot_id': 'Q01959',
                'gene_name': 'SLC6A3',
                'name': 'Sodium-dependent dopamine transporter',
                'organism': 'Homo sapiens',
                'target_type': 'SINGLE PROTEIN',
                'confidence_score': 9,
                'function': 'dopamine reuptake, ADHD, addiction',
                'diseases': ['adhd', 'addiction', 'parkinsons'],
                'drugs': ['methylphenidate', 'amphetamine', 'cocaine']
            },
            'NET': {
                'chembl_id': 'CHEMBL222',
                'uniprot_id': 'P23975',
                'gene_name': 'SLC6A2',
                'name': 'Sodium-dependent noradrenaline transporter',
                'organism': 'Homo sapiens',
                'target_type': 'SINGLE PROTEIN',
                'confidence_score': 9,
                'function': 'norepinephrine reuptake, antidepressant target',
                'diseases': ['depression', 'anxiety', 'adhd'],
                'drugs': ['atomoxetine', 'reboxetine', 'desipramine']
            },
            
            # Glutamate receptors
            'AMPA': {
                'chembl_id': 'CHEMBL4205',
                'uniprot_id': 'P42261',
                'gene_name': 'GRIA1',
                'name': 'Glutamate receptor ionotropic, AMPA 1',
                'organism': 'Homo sapiens',
                'target_type': 'SINGLE PROTEIN',
                'confidence_score': 9,
                'function': 'excitatory neurotransmission, synaptic plasticity',
                'diseases': ['epilepsy', 'alzheimers', 'stroke', 'autism'],
                'drugs': ['perampanel', 'topiramate', 'levetiracetam']
            },
            'NMDA': {
                'chembl_id': 'CHEMBL240',
                'uniprot_id': 'Q05586',
                'gene_name': 'GRIN1',
                'name': 'Glutamate receptor ionotropic, NMDA 1',
                'organism': 'Homo sapiens',
                'target_type': 'SINGLE PROTEIN',
                'confidence_score': 9,
                'function': 'excitatory neurotransmission, learning and memory',
                'diseases': ['schizophrenia', 'depression', 'alzheimers', 'stroke'],
                'drugs': ['memantine', 'ketamine', 'phencyclidine']
            },
            
            # GABA receptors
            'GABA-A': {
                'chembl_id': 'CHEMBL2093872',
                'uniprot_id': 'P14867',
                'gene_name': 'GABRA1',
                'name': 'Gamma-aminobutyric acid receptor subunit alpha-1',
                'organism': 'Homo sapiens',
                'target_type': 'SINGLE PROTEIN',
                'confidence_score': 9,
                'function': 'inhibitory neurotransmission, anxiolytic effects',
                'diseases': ['anxiety', 'epilepsy', 'insomnia', 'alcoholism'],
                'drugs': ['diazepam', 'lorazepam', 'zolpidem', 'barbiturates']
            },
            'GABA-B': {
                'chembl_id': 'CHEMBL2093873',
                'uniprot_id': 'Q9UBS5',
                'gene_name': 'GABBR1',
                'name': 'Gamma-aminobutyric acid type B receptor subunit 1',
                'organism': 'Homo sapiens',
                'target_type': 'SINGLE PROTEIN',
                'confidence_score': 9,
                'function': 'inhibitory neurotransmission, presynaptic modulation',
                'diseases': ['anxiety', 'depression', 'epilepsy', 'pain'],
                'drugs': ['baclofen', 'gabapentin', 'pregabalin']
            }
        }
    
    def get_target_info(self, target: str) -> Dict:
        """Get target information."""
        return self.targets.get(target, {})
    
    def get_chembl_id(self, target: str) -> str:
        """Get ChEMBL ID for target."""
        target_info = self.get_target_info(target)
        return target_info.get('chembl_id', '')
    
    def get_available_targets(self) -> List[str]:
        """Get list of available targets."""
        return list(self.targets.keys())
    
    def get_targets_by_family(self, family: str) -> List[str]:
        """Get targets by family."""
        families = {
            'serotonin': ['5HT2A', '5HT1A'],
            'dopamine': ['D1', 'D2'],
            'cannabinoid': ['CB1', 'CB2'],
            'opioid': ['MOR', 'DOR', 'KOR', 'NOP'],
            'transporter': ['SERT', 'DAT', 'NET'],
            'glutamate': ['AMPA', 'NMDA'],
            'gaba': ['GABA-A', 'GABA-B']
        }
        return families.get(family, [])
    
    def get_target_data(self, target: str, limit: int = 1000) -> pd.DataFrame:
        """
        Get target data from ChEMBL database.
        
        Args:
            target: Target name or ChEMBL ID
            limit: Maximum number of records to retrieve
            
        Returns:
            DataFrame with target data
        """
        chembl_id = self.get_chembl_id(target) if target in self.targets else target
        
        if not chembl_id:
            raise ValueError(f"Target {target} not found")
        
        # Get target data from ChEMBL
        url = f"{self.base_url}/activity"
        params = {
            'target_chembl_id': chembl_id,
            'limit': limit,
            'format': 'json'
        }
        
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data['activities'])
            
            # Filter for human data
            df = df[df['target_organism'] == 'Homo sapiens']
            
            # Convert IC50 to pIC50
            if 'standard_value' in df.columns:
                import numpy as np
                df['pIC50'] = -np.log10(df['standard_value'].astype(float) * 1e-9)  # Convert nM to pIC50
                df = df.dropna(subset=['pIC50'])
            
            return df
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {target}: {e}")
            return pd.DataFrame()
    
    def get_multi_target_data(self, targets: List[str], limit: int = 1000) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple targets.
        
        Args:
            targets: List of target names
            limit: Maximum number of records per target
            
        Returns:
            Dictionary with target data
        """
        data = {}
        
        for target in tqdm(targets, desc="Fetching target data"):
            try:
                df = self.get_target_data(target, limit)
                if not df.empty:
                    data[target] = df
                    print(f"Retrieved {len(df)} records for {target}")
                else:
                    print(f"No data found for {target}")
            except Exception as e:
                print(f"Error fetching data for {target}: {e}")
        
        return data
    
    def save_config(self, path: str):
        """Save configuration to file."""
        with open(path, 'w') as f:
            yaml.dump(self.targets, f, default_flow_style=False)
    
    def load_config(self, path: str):
        """Load configuration from file."""
        with open(path, 'r') as f:
            self.targets = yaml.safe_load(f)


# Convenience functions
def get_chembl_targets() -> ChEMBLTargets:
    """Get ChEMBL targets instance."""
    return ChEMBLTargets()

def get_target_info(target: str) -> Dict:
    """Get target information."""
    targets = ChEMBLTargets()
    return targets.get_target_info(target)

def get_chembl_id(target: str) -> str:
    """Get ChEMBL ID for target."""
    targets = ChEMBLTargets()
    return targets.get_chembl_id(target)

def get_target_data(target: str, limit: int = 1000) -> pd.DataFrame:
    """Get target data from ChEMBL database."""
    targets = ChEMBLTargets()
    return targets.get_target_data(target, limit)

# Constants
CHEMBL_TARGETS = {
    '5HT2A': 'CHEMBL224',
    '5HT1A': 'CHEMBL214',
    'D1': 'CHEMBL2056',
    'D2': 'CHEMBL217',
    'CB1': 'CHEMBL218',
    'CB2': 'CHEMBL253',
    'MOR': 'CHEMBL233',
    'DOR': 'CHEMBL236',
    'KOR': 'CHEMBL237',
    'NOP': 'CHEMBL2014',
    'SERT': 'CHEMBL228',
    'DAT': 'CHEMBL238',
    'NET': 'CHEMBL222'
}

TARGET_CONFIGS = {
    'DAT': {
        'chembl_id': 'CHEMBL238',
        'name': 'Sodium-dependent dopamine transporter',
        'organism': 'Homo sapiens',
        'target_type': 'SINGLE PROTEIN',
        'confidence_score': 9
    },
    '5HT2A': {
        'chembl_id': 'CHEMBL224',
        'name': '5-hydroxytryptamine receptor 2A',
        'organism': 'Homo sapiens',
        'target_type': 'SINGLE PROTEIN',
        'confidence_score': 9
    },
    'CB1': {
        'chembl_id': 'CHEMBL218',
        'name': 'Cannabinoid receptor 1',
        'organism': 'Homo sapiens',
        'target_type': 'SINGLE PROTEIN',
        'confidence_score': 9
    },
    'CB2': {
        'chembl_id': 'CHEMBL253',
        'name': 'Cannabinoid receptor 2',
        'organism': 'Homo sapiens',
        'target_type': 'SINGLE PROTEIN',
        'confidence_score': 9
    },
    'MOR': {
        'chembl_id': 'CHEMBL233',
        'name': 'Mu-type opioid receptor',
        'organism': 'Homo sapiens',
        'target_type': 'SINGLE PROTEIN',
        'confidence_score': 9
    }
}
