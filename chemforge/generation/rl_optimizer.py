"""
Reinforcement Learning (RL) optimizer for molecular optimization.

This module provides RL-based molecular optimization capabilities
for improving molecular properties through iterative optimization.
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

from chemforge.utils.logging_utils import Logger
from chemforge.utils.validation import DataValidator


class RLPolicy(nn.Module):
    """
    Policy network for RL-based molecular optimization.
    
    This network learns to predict actions for molecular modification
    to optimize desired properties.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        """
        Initialize the RL policy network.
        
        Args:
            state_dim: State dimension (molecular features)
            action_dim: Action dimension (modification types)
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super(RLPolicy, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self.logger = Logger('rl_policy')
        self.logger.info(f"RL Policy initialized: {state_dim} -> {action_dim}")
    
    def forward(self, state):
        """
        Forward pass through the policy network.
        
        Args:
            state: Current state (molecular features)
            
        Returns:
            Tuple of (action_logits, value)
        """
        action_logits = self.policy_net(state)
        value = self.value_net(state)
        
        return action_logits, value
    
    def get_action(self, state, temperature: float = 1.0):
        """
        Get action from policy.
        
        Args:
            state: Current state
            temperature: Sampling temperature
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        action_logits, value = self.forward(state)
        
        # Sample action
        action_probs = F.softmax(action_logits / temperature, dim=-1)
        action = torch.multinomial(action_probs, 1)
        
        # Calculate log probability
        log_prob = F.log_softmax(action_logits, dim=-1)
        log_prob = log_prob.gather(1, action)
        
        return action, log_prob, value


class RLReplayBuffer:
    """
    Replay buffer for RL training.
    
    This buffer stores experiences for training the RL agent.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum buffer capacity
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
        self.logger = Logger('rl_replay_buffer')
        self.logger.info(f"RL Replay Buffer initialized with capacity {capacity}")
    
    def push(self, state, action, reward, next_state, done):
        """
        Push experience to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        """
        Sample batch from buffer.
        
        Args:
            batch_size: Batch size to sample
            
        Returns:
            Batch of experiences
        """
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states = torch.stack([self.buffer[i][0] for i in batch])
        actions = torch.stack([self.buffer[i][1] for i in batch])
        rewards = torch.stack([self.buffer[i][2] for i in batch])
        next_states = torch.stack([self.buffer[i][3] for i in batch])
        dones = torch.stack([self.buffer[i][4] for i in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return buffer size."""
        return len(self.buffer)


class RLOptimizer:
    """
    RL-based molecular optimizer.
    
    This class provides RL-based molecular optimization capabilities
    for improving molecular properties through iterative optimization.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        device: str = 'auto',
        output_dir: str = "./rl_models",
        log_dir: str = "./logs"
    ):
        """
        Initialize the RL optimizer.
        
        Args:
            state_dim: State dimension (molecular features)
            action_dim: Action dimension (modification types)
            hidden_dim: Hidden layer dimension
            learning_rate: Learning rate
            gamma: Discount factor
            device: Device to use for training
            output_dir: Directory to save models
            log_dir: Directory for logs
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.device = self._get_device(device)
        
        # Create directories
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.logger = Logger('rl_optimizer', log_dir=str(self.log_dir))
        self.data_validator = DataValidator()
        
        # Initialize policy network
        self.policy = RLPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=learning_rate
        )
        
        # Initialize replay buffer
        self.replay_buffer = RLReplayBuffer()
        
        # Training state
        self.training_history = {}
        self.episode_rewards = []
        
        self.logger.info(f"RL Optimizer initialized with device: {self.device}")
    
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device for training."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("cpu")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def optimize_molecule(
        self,
        initial_smiles: str,
        reward_function: Callable[[str], float],
        max_steps: int = 100,
        temperature: float = 1.0
    ) -> Tuple[str, List[float], List[str]]:
        """
        Optimize a molecule using RL.
        
        Args:
            initial_smiles: Initial SMILES string
            reward_function: Function to calculate reward
            max_steps: Maximum optimization steps
            temperature: Sampling temperature
            
        Returns:
            Tuple of (optimized_smiles, rewards, smiles_history)
        """
        self.logger.info(f"Optimizing molecule: {initial_smiles}")
        
        current_smiles = initial_smiles
        smiles_history = [current_smiles]
        rewards = []
        
        for step in range(max_steps):
            # Get current state (molecular features)
            state = self._smiles_to_features(current_smiles)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Get action from policy
            action, log_prob, value = self.policy.get_action(state_tensor, temperature)
            
            # Apply action to molecule
            new_smiles = self._apply_action(current_smiles, action.item())
            
            # Calculate reward
            reward = reward_function(new_smiles)
            
            # Store experience
            next_state = self._smiles_to_features(new_smiles)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            self.replay_buffer.push(
                state_tensor, action, torch.tensor(reward), next_state_tensor, torch.tensor(False)
            )
            
            # Update current molecule
            current_smiles = new_smiles
            smiles_history.append(current_smiles)
            rewards.append(reward)
            
            # Train policy if buffer has enough experiences
            if len(self.replay_buffer) > 32:
                self._train_policy()
            
            # Log progress
            if step % 10 == 0:
                self.logger.info(f"Step {step}: Reward = {reward:.4f}, SMILES = {current_smiles}")
        
        self.logger.info(f"Optimization completed. Final reward: {rewards[-1]:.4f}")
        
        return current_smiles, rewards, smiles_history
    
    def _smiles_to_features(self, smiles: str) -> np.ndarray:
        """
        Convert SMILES to molecular features.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Molecular features array
        """
        # This is a simplified implementation
        # In practice, you would use proper molecular feature extraction
        features = np.random.randn(self.state_dim)
        return features
    
    def _apply_action(self, smiles: str, action: int) -> str:
        """
        Apply action to modify molecule.
        
        Args:
            smiles: Current SMILES string
            action: Action to apply
            
        Returns:
            Modified SMILES string
        """
        # This is a simplified implementation
        # In practice, you would implement proper molecular modifications
        return smiles  # Placeholder
    
    def _train_policy(self, batch_size: int = 32):
        """
        Train the policy network.
        
        Args:
            batch_size: Batch size for training
        """
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Calculate target values
        with torch.no_grad():
            _, next_values = self.policy(next_states)
            target_values = rewards + self.gamma * next_values * (1 - dones)
        
        # Calculate current values
        _, current_values = self.policy(states)
        
        # Calculate value loss
        value_loss = F.mse_loss(current_values, target_values)
        
        # Calculate policy loss
        action_logits, _ = self.policy(states)
        log_probs = F.log_softmax(action_logits, dim=-1)
        log_probs = log_probs.gather(1, actions)
        
        policy_loss = -(log_probs * (target_values - current_values.detach())).mean()
        
        # Total loss
        total_loss = value_loss + policy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
    
    def train_on_dataset(
        self,
        molecules: List[str],
        reward_function: Callable[[str], float],
        episodes: int = 1000,
        max_steps_per_episode: int = 50,
        save_model: bool = True
    ) -> Dict[str, Any]:
        """
        Train the RL agent on a dataset of molecules.
        
        Args:
            molecules: List of initial molecules
            reward_function: Function to calculate reward
            episodes: Number of training episodes
            max_steps_per_episode: Maximum steps per episode
            save_model: Whether to save the trained model
            
        Returns:
            Training results dictionary
        """
        self.logger.info(f"Training RL agent on {len(molecules)} molecules")
        
        episode_rewards = []
        
        for episode in range(episodes):
            # Select random molecule
            initial_smiles = np.random.choice(molecules)
            
            # Optimize molecule
            _, rewards, _ = self.optimize_molecule(
                initial_smiles, reward_function, max_steps_per_episode
            )
            
            # Store episode reward
            episode_rewards.append(sum(rewards))
            
            # Log progress
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                self.logger.info(f"Episode {episode}: Average Reward = {avg_reward:.4f}")
        
        # Store training history
        self.training_history = {
            'episode_rewards': episode_rewards,
            'episodes': episodes,
            'max_steps_per_episode': max_steps_per_episode,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma
        }
        
        # Save model if requested
        if save_model:
            self.save_model()
        
        self.logger.info("RL training completed")
        
        return self.training_history
    
    def evaluate_policy(
        self,
        test_molecules: List[str],
        reward_function: Callable[[str], float],
        max_steps: int = 100
    ) -> Dict[str, Any]:
        """
        Evaluate the trained policy.
        
        Args:
            test_molecules: List of test molecules
            reward_function: Function to calculate reward
            max_steps: Maximum optimization steps
            
        Returns:
            Evaluation results dictionary
        """
        self.logger.info(f"Evaluating policy on {len(test_molecules)} molecules")
        
        results = {
            'initial_rewards': [],
            'final_rewards': [],
            'improvements': [],
            'optimized_molecules': []
        }
        
        for smiles in test_molecules:
            # Calculate initial reward
            initial_reward = reward_function(smiles)
            
            # Optimize molecule
            optimized_smiles, rewards, _ = self.optimize_molecule(
                smiles, reward_function, max_steps
            )
            
            # Calculate final reward
            final_reward = reward_function(optimized_smiles)
            
            # Store results
            results['initial_rewards'].append(initial_reward)
            results['final_rewards'].append(final_reward)
            results['improvements'].append(final_reward - initial_reward)
            results['optimized_molecules'].append(optimized_smiles)
        
        # Calculate statistics
        results['avg_initial_reward'] = np.mean(results['initial_rewards'])
        results['avg_final_reward'] = np.mean(results['final_rewards'])
        results['avg_improvement'] = np.mean(results['improvements'])
        results['improvement_rate'] = np.mean([imp > 0 for imp in results['improvements']])
        
        self.logger.info(f"Evaluation completed. Average improvement: {results['avg_improvement']:.4f}")
        
        return results
    
    def save_model(self, filepath: Optional[str] = None):
        """Save the trained RL model."""
        if filepath is None:
            filepath = self.output_dir / "rl_model.pt"
        
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dim': self.hidden_dim,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'training_history': self.training_history
        }, filepath)
        
        self.logger.info(f"Saved RL model to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained RL model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', {})
        
        self.logger.info(f"Loaded RL model from {filepath}")
    
    def get_training_history(self) -> Dict[str, Any]:
        """Get the training history."""
        return self.training_history
