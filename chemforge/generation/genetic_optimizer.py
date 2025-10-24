"""
Genetic Algorithm (GA) optimizer for molecular optimization.

This module provides GA-based molecular optimization capabilities
for improving molecular properties through evolutionary optimization.
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import pandas as pd
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from chemforge.utils.logging_utils import Logger
from chemforge.utils.validation import DataValidator


class GeneticOptimizer:
    """
    Genetic Algorithm-based molecular optimizer.
    
    This class provides GA-based molecular optimization capabilities
    for improving molecular properties through evolutionary optimization.
    """
    
    def __init__(
        self,
        population_size: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elite_size: int = 10,
        max_generations: int = 100,
        output_dir: str = "./ga_models",
        log_dir: str = "./logs"
    ):
        """
        Initialize the genetic optimizer.
        
        Args:
            population_size: Size of the population
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elite_size: Number of elite individuals to preserve
            max_generations: Maximum number of generations
            output_dir: Directory to save models
            log_dir: Directory for logs
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.max_generations = max_generations
        
        # Create directories
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.logger = Logger('genetic_optimizer', log_dir=str(self.log_dir))
        self.data_validator = DataValidator()
        
        # Optimization state
        self.population = []
        self.fitness_history = []
        self.best_individuals = []
        self.optimization_history = {}
        
        self.logger.info(f"Genetic Optimizer initialized with population size {population_size}")
    
    def initialize_population(
        self,
        initial_molecules: List[str],
        target_size: Optional[int] = None
    ) -> List[str]:
        """
        Initialize the population with molecules.
        
        Args:
            initial_molecules: List of initial molecules
            target_size: Target population size
            
        Returns:
            Initial population
        """
        if target_size is None:
            target_size = self.population_size
        
        # Start with initial molecules
        population = initial_molecules.copy()
        
        # Generate additional molecules if needed
        while len(population) < target_size:
            # Randomly select a molecule to mutate
            parent = random.choice(initial_molecules)
            child = self._mutate_molecule(parent)
            population.append(child)
        
        # Trim to target size
        population = population[:target_size]
        
        self.population = population
        self.logger.info(f"Initialized population with {len(population)} molecules")
        
        return population
    
    def optimize(
        self,
        initial_molecules: List[str],
        fitness_function: Callable[[str], float],
        maximize: bool = True,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run genetic algorithm optimization.
        
        Args:
            initial_molecules: List of initial molecules
            fitness_function: Function to calculate fitness
            maximize: Whether to maximize fitness
            save_results: Whether to save optimization results
            
        Returns:
            Optimization results dictionary
        """
        self.logger.info("Starting genetic algorithm optimization")
        
        # Initialize population
        self.population = self.initialize_population(initial_molecules)
        
        # Optimization loop
        for generation in range(self.max_generations):
            # Calculate fitness for all individuals
            fitness_scores = []
            for molecule in self.population:
                try:
                    fitness = fitness_function(molecule)
                    fitness_scores.append(fitness)
                except Exception as e:
                    self.logger.warning(f"Fitness calculation failed for {molecule}: {e}")
                    fitness_scores.append(0.0)
            
            # Store fitness history
            self.fitness_history.append({
                'generation': generation,
                'fitness_scores': fitness_scores.copy(),
                'best_fitness': max(fitness_scores) if maximize else min(fitness_scores),
                'avg_fitness': np.mean(fitness_scores),
                'worst_fitness': min(fitness_scores) if maximize else max(fitness_scores)
            })
            
            # Select best individuals
            if maximize:
                sorted_indices = np.argsort(fitness_scores)[::-1]
            else:
                sorted_indices = np.argsort(fitness_scores)
            
            best_individuals = [self.population[i] for i in sorted_indices[:self.elite_size]]
            self.best_individuals.extend(best_individuals)
            
            # Create new population
            new_population = []
            
            # Keep elite individuals
            new_population.extend(best_individuals)
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Select parents
                parent1 = self._select_parent(fitness_scores, maximize)
                parent2 = self._select_parent(fitness_scores, maximize)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2
                
                # Mutation
                if random.random() < self.mutation_rate:
                    child1 = self._mutate_molecule(child1)
                if random.random() < self.mutation_rate:
                    child2 = self._mutate_molecule(child2)
                
                new_population.extend([child1, child2])
            
            # Update population
            self.population = new_population[:self.population_size]
            
            # Log progress
            if generation % 10 == 0:
                best_fitness = self.fitness_history[-1]['best_fitness']
                avg_fitness = self.fitness_history[-1]['avg_fitness']
                self.logger.info(f"Generation {generation}: Best = {best_fitness:.4f}, Avg = {avg_fitness:.4f}")
        
        # Prepare results
        results = {
            'best_molecules': self.best_individuals[:10],  # Top 10
            'fitness_history': self.fitness_history,
            'final_population': self.population,
            'optimization_parameters': {
                'population_size': self.population_size,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'elite_size': self.elite_size,
                'max_generations': self.max_generations
            }
        }
        
        # Save results if requested
        if save_results:
            self._save_results(results)
        
        self.logger.info("Genetic algorithm optimization completed")
        
        return results
    
    def _select_parent(self, fitness_scores: List[float], maximize: bool = True) -> str:
        """
        Select a parent using tournament selection.
        
        Args:
            fitness_scores: List of fitness scores
            maximize: Whether to maximize fitness
            
        Returns:
            Selected parent molecule
        """
        tournament_size = 3
        tournament_indices = random.sample(range(len(fitness_scores)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        
        if maximize:
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        else:
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        
        return self.population[winner_idx]
    
    def _crossover(self, parent1: str, parent2: str) -> Tuple[str, str]:
        """
        Perform crossover between two parents.
        
        Args:
            parent1: First parent molecule
            parent2: Second parent molecule
            
        Returns:
            Tuple of (child1, child2)
        """
        # Simple crossover: randomly combine parts of parents
        if len(parent1) < 2 or len(parent2) < 2:
            return parent1, parent2
        
        # Random crossover point
        crossover_point1 = random.randint(1, len(parent1) - 1)
        crossover_point2 = random.randint(1, len(parent2) - 1)
        
        # Create children
        child1 = parent1[:crossover_point1] + parent2[crossover_point2:]
        child2 = parent2[:crossover_point2] + parent1[crossover_point1:]
        
        return child1, child2
    
    def _mutate_molecule(self, molecule: str) -> str:
        """
        Mutate a molecule.
        
        Args:
            molecule: Molecule to mutate
            
        Returns:
            Mutated molecule
        """
        if len(molecule) < 2:
            return molecule
        
        # Random mutation operations
        mutation_type = random.choice(['substitute', 'insert', 'delete', 'swap'])
        
        if mutation_type == 'substitute':
            # Substitute a random character
            pos = random.randint(0, len(molecule) - 1)
            new_char = random.choice('CNOSFClBrI()[]=#-+')
            molecule = molecule[:pos] + new_char + molecule[pos+1:]
        
        elif mutation_type == 'insert':
            # Insert a random character
            pos = random.randint(0, len(molecule))
            new_char = random.choice('CNOSFClBrI()[]=#-+')
            molecule = molecule[:pos] + new_char + molecule[pos:]
        
        elif mutation_type == 'delete':
            # Delete a random character
            if len(molecule) > 1:
                pos = random.randint(0, len(molecule) - 1)
                molecule = molecule[:pos] + molecule[pos+1:]
        
        elif mutation_type == 'swap':
            # Swap two random characters
            if len(molecule) > 1:
                pos1, pos2 = random.sample(range(len(molecule)), 2)
                molecule_list = list(molecule)
                molecule_list[pos1], molecule_list[pos2] = molecule_list[pos2], molecule_list[pos1]
                molecule = ''.join(molecule_list)
        
        return molecule
    
    def evaluate_population(
        self,
        fitness_function: Callable[[str], float]
    ) -> Dict[str, Any]:
        """
        Evaluate the current population.
        
        Args:
            fitness_function: Function to calculate fitness
            
        Returns:
            Evaluation results dictionary
        """
        self.logger.info(f"Evaluating population of {len(self.population)} molecules")
        
        fitness_scores = []
        valid_molecules = []
        
        for molecule in self.population:
            try:
                fitness = fitness_function(molecule)
                fitness_scores.append(fitness)
                valid_molecules.append(molecule)
            except Exception as e:
                self.logger.warning(f"Fitness calculation failed for {molecule}: {e}")
                fitness_scores.append(0.0)
                valid_molecules.append(molecule)
        
        # Calculate statistics
        results = {
            'fitness_scores': fitness_scores,
            'valid_molecules': valid_molecules,
            'best_fitness': max(fitness_scores),
            'worst_fitness': min(fitness_scores),
            'avg_fitness': np.mean(fitness_scores),
            'std_fitness': np.std(fitness_scores),
            'diversity': self._calculate_diversity(valid_molecules)
        }
        
        self.logger.info(f"Population evaluation completed. Best fitness: {results['best_fitness']:.4f}")
        
        return results
    
    def _calculate_diversity(self, molecules: List[str]) -> float:
        """
        Calculate population diversity.
        
        Args:
            molecules: List of molecules
            
        Returns:
            Diversity score
        """
        if len(molecules) < 2:
            return 0.0
        
        # Calculate pairwise distances (simplified)
        distances = []
        for i in range(len(molecules)):
            for j in range(i + 1, len(molecules)):
                # Simple distance based on string similarity
                distance = self._string_distance(molecules[i], molecules[j])
                distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    def _string_distance(self, s1: str, s2: str) -> float:
        """
        Calculate distance between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Distance between strings
        """
        # Simple Levenshtein distance
        if len(s1) < len(s2):
            return self._string_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """
        Plot optimization history.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.fitness_history:
            self.logger.warning("No fitness history available for plotting")
            return
        
        generations = [h['generation'] for h in self.fitness_history]
        best_fitness = [h['best_fitness'] for h in self.fitness_history]
        avg_fitness = [h['avg_fitness'] for h in self.fitness_history]
        worst_fitness = [h['worst_fitness'] for h in self.fitness_history]
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(generations, best_fitness, label='Best Fitness', color='green')
        plt.plot(generations, avg_fitness, label='Average Fitness', color='blue')
        plt.plot(generations, worst_fitness, label='Worst Fitness', color='red')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        fitness_std = [np.std(h['fitness_scores']) for h in self.fitness_history]
        plt.plot(generations, fitness_std, label='Fitness Standard Deviation', color='purple')
        plt.xlabel('Generation')
        plt.ylabel('Standard Deviation')
        plt.title('Fitness Diversity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved optimization history plot to {save_path}")
        
        plt.show()
    
    def _save_results(self, results: Dict[str, Any]):
        """Save optimization results."""
        results_path = self.output_dir / "optimization_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                json_results[key] = [v.tolist() for v in value]
            else:
                json_results[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        self.logger.info(f"Saved optimization results to {results_path}")
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """Load optimization results."""
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        self.logger.info(f"Loaded optimization results from {filepath}")
        
        return results
    
    def get_optimization_history(self) -> Dict[str, Any]:
        """Get the optimization history."""
        return {
            'fitness_history': self.fitness_history,
            'best_individuals': self.best_individuals,
            'population_size': self.population_size,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'elite_size': self.elite_size,
            'max_generations': self.max_generations
        }
