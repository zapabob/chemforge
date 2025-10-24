"""
Unit tests for genetic optimizer.
"""

import unittest
import tempfile
import os
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

from chemforge.generation.genetic_optimizer import GeneticOptimizer


class TestGeneticOptimizer(unittest.TestCase):
    """Test GeneticOptimizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, 'output')
        self.log_dir = os.path.join(self.temp_dir, 'logs')
        
        self.optimizer = GeneticOptimizer(
            population_size=50,
            mutation_rate=0.1,
            crossover_rate=0.8,
            elite_size=5,
            max_generations=20,
            output_dir=self.output_dir,
            log_dir=self.log_dir
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test GeneticOptimizer initialization."""
        self.assertEqual(self.optimizer.population_size, 50)
        self.assertEqual(self.optimizer.mutation_rate, 0.1)
        self.assertEqual(self.optimizer.crossover_rate, 0.8)
        self.assertEqual(self.optimizer.elite_size, 5)
        self.assertEqual(self.optimizer.max_generations, 20)
        self.assertEqual(len(self.optimizer.population), 0)
        self.assertEqual(len(self.optimizer.fitness_history), 0)
        self.assertEqual(len(self.optimizer.best_individuals), 0)
    
    def test_initialize_population(self):
        """Test population initialization."""
        initial_molecules = ['CCO', 'CCN', 'CC(C)O', 'CC(C)N', 'CC(C)(C)O']
        
        population = self.optimizer.initialize_population(initial_molecules, target_size=20)
        
        # Verify population
        self.assertEqual(len(population), 20)
        self.assertTrue(all(isinstance(mol, str) for mol in population))
        
        # Verify all initial molecules are included
        for mol in initial_molecules:
            self.assertIn(mol, population)
    
    def test_initialize_population_large_target(self):
        """Test population initialization with large target size."""
        initial_molecules = ['CCO', 'CCN']
        
        population = self.optimizer.initialize_population(initial_molecules, target_size=100)
        
        # Verify population
        self.assertEqual(len(population), 100)
        self.assertTrue(all(isinstance(mol, str) for mol in population))
    
    def test_select_parent(self):
        """Test parent selection."""
        # Initialize population
        self.optimizer.population = ['CCO', 'CCN', 'CC(C)O', 'CC(C)N', 'CC(C)(C)O']
        fitness_scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        # Test selection
        parent = self.optimizer._select_parent(fitness_scores, maximize=True)
        
        # Verify parent is from population
        self.assertIn(parent, self.optimizer.population)
    
    def test_crossover(self):
        """Test crossover operation."""
        parent1 = "CCO"
        parent2 = "CCN"
        
        child1, child2 = self.optimizer._crossover(parent1, parent2)
        
        # Verify children are strings
        self.assertIsInstance(child1, str)
        self.assertIsInstance(child2, str)
        
        # Verify children are not empty
        self.assertGreater(len(child1), 0)
        self.assertGreater(len(child2), 0)
    
    def test_mutate_molecule(self):
        """Test molecule mutation."""
        molecule = "CCO"
        
        mutated = self.optimizer._mutate_molecule(molecule)
        
        # Verify mutated molecule is a string
        self.assertIsInstance(mutated, str)
        
        # Verify mutated molecule is not empty
        self.assertGreater(len(mutated), 0)
    
    def test_mutate_molecule_short(self):
        """Test mutation of short molecule."""
        molecule = "C"
        
        mutated = self.optimizer._mutate_molecule(molecule)
        
        # Verify mutated molecule is a string
        self.assertIsInstance(mutated, str)
    
    def test_optimize(self):
        """Test genetic optimization."""
        def mock_fitness_function(smiles):
            return len(smiles) * 0.1  # Simple fitness function
        
        initial_molecules = ['CCO', 'CCN', 'CC(C)O', 'CC(C)N', 'CC(C)(C)O']
        
        results = self.optimizer.optimize(
            initial_molecules, mock_fitness_function, maximize=True, save_results=False
        )
        
        # Verify results
        self.assertIsInstance(results, dict)
        self.assertIn('best_molecules', results)
        self.assertIn('fitness_history', results)
        self.assertIn('final_population', results)
        self.assertIn('optimization_parameters', results)
        
        # Verify best molecules
        self.assertIsInstance(results['best_molecules'], list)
        self.assertGreater(len(results['best_molecules']), 0)
        
        # Verify fitness history
        self.assertIsInstance(results['fitness_history'], list)
        self.assertEqual(len(results['fitness_history']), self.optimizer.max_generations)
        
        # Verify final population
        self.assertIsInstance(results['final_population'], list)
        self.assertEqual(len(results['final_population']), self.optimizer.population_size)
    
    def test_evaluate_population(self):
        """Test population evaluation."""
        def mock_fitness_function(smiles):
            return len(smiles) * 0.1  # Simple fitness function
        
        # Set population
        self.optimizer.population = ['CCO', 'CCN', 'CC(C)O', 'CC(C)N', 'CC(C)(C)O']
        
        results = self.optimizer.evaluate_population(mock_fitness_function)
        
        # Verify results
        self.assertIsInstance(results, dict)
        self.assertIn('fitness_scores', results)
        self.assertIn('valid_molecules', results)
        self.assertIn('best_fitness', results)
        self.assertIn('worst_fitness', results)
        self.assertIn('avg_fitness', results)
        self.assertIn('std_fitness', results)
        self.assertIn('diversity', results)
        
        # Verify fitness scores
        self.assertEqual(len(results['fitness_scores']), len(self.optimizer.population))
        self.assertEqual(len(results['valid_molecules']), len(self.optimizer.population))
    
    def test_calculate_diversity(self):
        """Test diversity calculation."""
        molecules = ['CCO', 'CCN', 'CC(C)O', 'CC(C)N', 'CC(C)(C)O']
        
        diversity = self.optimizer._calculate_diversity(molecules)
        
        # Verify diversity is a number
        self.assertIsInstance(diversity, float)
        self.assertGreaterEqual(diversity, 0.0)
    
    def test_calculate_diversity_single_molecule(self):
        """Test diversity calculation with single molecule."""
        molecules = ['CCO']
        
        diversity = self.optimizer._calculate_diversity(molecules)
        
        # Verify diversity is 0 for single molecule
        self.assertEqual(diversity, 0.0)
    
    def test_string_distance(self):
        """Test string distance calculation."""
        s1 = "CCO"
        s2 = "CCN"
        
        distance = self.optimizer._string_distance(s1, s2)
        
        # Verify distance is a number
        self.assertIsInstance(distance, float)
        self.assertGreaterEqual(distance, 0.0)
    
    def test_string_distance_identical(self):
        """Test string distance with identical strings."""
        s1 = "CCO"
        s2 = "CCO"
        
        distance = self.optimizer._string_distance(s1, s2)
        
        # Verify distance is 0 for identical strings
        self.assertEqual(distance, 0.0)
    
    def test_string_distance_empty(self):
        """Test string distance with empty strings."""
        s1 = ""
        s2 = "CCO"
        
        distance = self.optimizer._string_distance(s1, s2)
        
        # Verify distance is length of non-empty string
        self.assertEqual(distance, len(s2))
    
    def test_plot_optimization_history(self):
        """Test optimization history plotting."""
        # Set mock fitness history
        self.optimizer.fitness_history = [
            {
                'generation': 0,
                'best_fitness': 0.5,
                'avg_fitness': 0.3,
                'worst_fitness': 0.1
            },
            {
                'generation': 1,
                'best_fitness': 0.7,
                'avg_fitness': 0.4,
                'worst_fitness': 0.2
            }
        ]
        
        # Test plotting (should not raise exception)
        try:
            self.optimizer.plot_optimization_history()
        except Exception as e:
            # If plotting fails due to display issues, that's okay
            self.assertIn('display', str(e).lower())
    
    def test_save_results(self):
        """Test saving optimization results."""
        # Set mock results
        results = {
            'best_molecules': ['CCO', 'CCN', 'CC(C)O'],
            'fitness_history': [
                {'generation': 0, 'best_fitness': 0.5},
                {'generation': 1, 'best_fitness': 0.7}
            ],
            'final_population': ['CCO', 'CCN', 'CC(C)O', 'CC(C)N', 'CC(C)(C)O'],
            'optimization_parameters': {
                'population_size': 50,
                'mutation_rate': 0.1,
                'crossover_rate': 0.8
            }
        }
        
        # Save results
        self.optimizer._save_results(results)
        
        # Verify file was created
        results_path = os.path.join(self.output_dir, 'optimization_results.json')
        self.assertTrue(os.path.exists(results_path))
    
    def test_load_results(self):
        """Test loading optimization results."""
        # Create mock results file
        results = {
            'best_molecules': ['CCO', 'CCN', 'CC(C)O'],
            'fitness_history': [
                {'generation': 0, 'best_fitness': 0.5},
                {'generation': 1, 'best_fitness': 0.7}
            ]
        }
        
        results_path = os.path.join(self.output_dir, 'optimization_results.json')
        import json
        with open(results_path, 'w') as f:
            json.dump(results, f)
        
        # Load results
        loaded_results = self.optimizer.load_results(results_path)
        
        # Verify loading
        self.assertIsInstance(loaded_results, dict)
        self.assertIn('best_molecules', loaded_results)
        self.assertIn('fitness_history', loaded_results)
    
    def test_get_optimization_history(self):
        """Test getting optimization history."""
        # Set mock history
        self.optimizer.fitness_history = [{'generation': 0, 'best_fitness': 0.5}]
        self.optimizer.best_individuals = ['CCO', 'CCN']
        
        history = self.optimizer.get_optimization_history()
        
        # Verify history
        self.assertIsInstance(history, dict)
        self.assertIn('fitness_history', history)
        self.assertIn('best_individuals', history)
        self.assertIn('population_size', history)
        self.assertIn('mutation_rate', history)
        self.assertIn('crossover_rate', history)
        self.assertIn('elite_size', history)
        self.assertIn('max_generations', history)


if __name__ == '__main__':
    unittest.main()
