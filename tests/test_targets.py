"""
Tests for CNS targets functionality.
"""

import pytest
from molecular_pwa_pet.targets import (
    CNSTargets,
    get_cns_targets,
    get_target_info,
    get_optimization_goals,
    get_selectivity_targets,
    get_safety_targets
)


class TestCNSTargets:
    """Test CNSTargets class."""
    
    def test_initialization(self):
        """Test CNSTargets initialization."""
        targets = CNSTargets()
        assert isinstance(targets.targets, dict)
        assert len(targets.targets) > 0
    
    def test_get_available_targets(self, cns_targets):
        """Test getting available targets."""
        targets = cns_targets.get_available_targets()
        assert isinstance(targets, list)
        assert len(targets) > 0
        assert '5HT2A' in targets
        assert 'D1' in targets
        assert 'CB1' in targets
        assert 'MOR' in targets
    
    def test_get_targets_by_family(self, cns_targets):
        """Test getting targets by family."""
        # Test serotonin family
        serotonin_targets = cns_targets.get_targets_by_family('serotonin')
        assert isinstance(serotonin_targets, list)
        assert '5HT2A' in serotonin_targets
        assert '5HT1A' in serotonin_targets
        
        # Test dopamine family
        dopamine_targets = cns_targets.get_targets_by_family('dopamine')
        assert isinstance(dopamine_targets, list)
        assert 'D1' in dopamine_targets
        assert 'D2' in dopamine_targets
        
        # Test cannabinoid family
        cannabinoid_targets = cns_targets.get_targets_by_family('cannabinoid')
        assert isinstance(cannabinoid_targets, list)
        assert 'CB1' in cannabinoid_targets
        assert 'CB2' in cannabinoid_targets
        
        # Test opioid family
        opioid_targets = cns_targets.get_targets_by_family('opioid')
        assert isinstance(opioid_targets, list)
        assert 'MOR' in opioid_targets
        assert 'DOR' in opioid_targets
        assert 'KOR' in opioid_targets
        assert 'NOP' in opioid_targets
    
    def test_get_target_info(self, cns_targets):
        """Test getting target information."""
        # Test 5HT2A
        info = cns_targets.get_target_info('5HT2A')
        assert isinstance(info, dict)
        assert 'pdb_id' in info
        assert 'function' in info
        assert 'diseases' in info
        assert 'drugs' in info
        assert info['family'] == 'serotonin'
        
        # Test D1
        info = cns_targets.get_target_info('D1')
        assert isinstance(info, dict)
        assert info['family'] == 'dopamine'
        
        # Test CB1
        info = cns_targets.get_target_info('CB1')
        assert isinstance(info, dict)
        assert info['family'] == 'cannabinoid'
        
        # Test MOR
        info = cns_targets.get_target_info('MOR')
        assert isinstance(info, dict)
        assert info['family'] == 'opioid'
    
    def test_get_targets_by_disease(self, cns_targets):
        """Test getting targets by disease."""
        # Test depression
        depression_targets = cns_targets.get_targets_by_disease('depression')
        assert isinstance(depression_targets, list)
        assert '5HT1A' in depression_targets
        assert 'SERT' in depression_targets
        
        # Test pain
        pain_targets = cns_targets.get_targets_by_disease('pain')
        assert isinstance(pain_targets, list)
        assert 'CB1' in pain_targets
        assert 'CB2' in pain_targets
        assert 'MOR' in pain_targets
    
    def test_get_targets_by_drug(self, cns_targets):
        """Test getting targets by drug."""
        # Test morphine
        morphine_targets = cns_targets.get_targets_by_drug('morphine')
        assert isinstance(morphine_targets, list)
        assert 'MOR' in morphine_targets
        
        # Test THC
        thc_targets = cns_targets.get_targets_by_drug('THC')
        assert isinstance(thc_targets, list)
        assert 'CB1' in thc_targets


class TestTargetFunctions:
    """Test target utility functions."""
    
    def test_get_target_info(self):
        """Test get_target_info function."""
        info = get_target_info('5HT2A')
        assert isinstance(info, dict)
        assert 'pdb_id' in info
        assert 'function' in info
    
    def test_get_optimization_goals(self):
        """Test get_optimization_goals function."""
        # Test dopamine targets
        d1_goals = get_optimization_goals('D1')
        assert isinstance(d1_goals, list)
        assert 'pki' in d1_goals
        assert 'selectivity' in d1_goals
        
        # Test cannabinoid targets
        cb1_goals = get_optimization_goals('CB1')
        assert isinstance(cb1_goals, list)
        assert 'pki' in cb1_goals
        assert 'selectivity' in cb1_goals
        assert 'safety' in cb1_goals
        
        # Test opioid targets
        mor_goals = get_optimization_goals('MOR')
        assert isinstance(mor_goals, list)
        assert 'pki' in mor_goals
        assert 'selectivity' in mor_goals
        assert 'safety' in mor_goals
        assert 'addiction_potential' in mor_goals
    
    def test_get_selectivity_targets(self):
        """Test get_selectivity_targets function."""
        # Test D1 selectivity
        d1_selectivity = get_selectivity_targets('D1')
        assert isinstance(d1_selectivity, dict)
        assert 'D1' in d1_selectivity
        assert 'D2' in d1_selectivity
        assert d1_selectivity['D1'] > d1_selectivity['D2']
        
        # Test CB1 selectivity
        cb1_selectivity = get_selectivity_targets('CB1')
        assert isinstance(cb1_selectivity, dict)
        assert 'CB1' in cb1_selectivity
        assert 'CB2' in cb1_selectivity
        assert cb1_selectivity['CB1'] > cb1_selectivity['CB2']
        
        # Test MOR selectivity
        mor_selectivity = get_selectivity_targets('MOR')
        assert isinstance(mor_selectivity, dict)
        assert 'MOR' in mor_selectivity
        assert 'DOR' in mor_selectivity
        assert 'KOR' in mor_selectivity
        assert mor_selectivity['MOR'] > mor_selectivity['DOR']
        assert mor_selectivity['MOR'] > mor_selectivity['KOR']
    
    def test_get_safety_targets(self):
        """Test get_safety_targets function."""
        # Test D1 safety
        d1_safety = get_safety_targets('D1')
        assert isinstance(d1_safety, dict)
        assert 'psychosis' in d1_safety
        assert 'dyskinesia' in d1_safety
        assert 'tolerance' in d1_safety
        
        # Test CB1 safety
        cb1_safety = get_safety_targets('CB1')
        assert isinstance(cb1_safety, dict)
        assert 'psychoactivity' in cb1_safety
        assert 'tolerance' in cb1_safety
        assert 'withdrawal' in cb1_safety
        
        # Test MOR safety
        mor_safety = get_safety_targets('MOR')
        assert isinstance(mor_safety, dict)
        assert 'addiction_potential' in mor_safety
        assert 'respiratory_depression' in mor_safety
        assert 'tolerance' in mor_safety
