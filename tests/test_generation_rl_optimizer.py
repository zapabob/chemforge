"""
Unit tests for RL optimizer.
"""

import unittest
import tempfile
import os
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

from chemforge.generation.rl_optimizer import RLOptimizer, RLPolicy, RLReplayBuffer


class TestRLPolicy(unittest.TestCase):
    """Test RLPolicy class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.state_dim = 100
        self.action_dim = 10
        self.hidden_dim = 256
        
        self.policy = RLPolicy(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim
        )
    
    def test_init(self):
        """Test RLPolicy initialization."""
        self.assertEqual(self.policy.state_dim, self.state_dim)
        self.assertEqual(self.policy.action_dim, self.action_dim)
        self.assertEqual(self.policy.hidden_dim, self.hidden_dim)
    
    def test_forward(self):
        """Test RLPolicy forward pass."""
        batch_size = 32
        state = torch.randn(batch_size, self.state_dim)
        
        action_logits, value = self.policy(state)
        
        self.assertEqual(action_logits.shape, (batch_size, self.action_dim))
        self.assertEqual(value.shape, (batch_size, 1))
    
    def test_get_action(self):
        """Test RLPolicy action selection."""
        batch_size = 32
        state = torch.randn(batch_size, self.state_dim)
        
        action, log_prob, value = self.policy.get_action(state)
        
        self.assertEqual(action.shape, (batch_size, 1))
        self.assertEqual(log_prob.shape, (batch_size, 1))
        self.assertEqual(value.shape, (batch_size, 1))


class TestRLReplayBuffer(unittest.TestCase):
    """Test RLReplayBuffer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.capacity = 1000
        self.buffer = RLReplayBuffer(capacity=self.capacity)
    
    def test_init(self):
        """Test RLReplayBuffer initialization."""
        self.assertEqual(self.buffer.capacity, self.capacity)
        self.assertEqual(len(self.buffer), 0)
    
    def test_push(self):
        """Test pushing experience to buffer."""
        state = torch.randn(10)
        action = torch.tensor(1)
        reward = torch.tensor(0.5)
        next_state = torch.randn(10)
        done = torch.tensor(False)
        
        self.buffer.push(state, action, reward, next_state, done)
        
        self.assertEqual(len(self.buffer), 1)
    
    def test_push_overflow(self):
        """Test buffer overflow behavior."""
        # Fill buffer beyond capacity
        for i in range(self.capacity + 10):
            state = torch.randn(10)
            action = torch.tensor(i % 5)
            reward = torch.tensor(i * 0.1)
            next_state = torch.randn(10)
            done = torch.tensor(i % 10 == 0)
            
            self.buffer.push(state, action, reward, next_state, done)
        
        # Buffer should not exceed capacity
        self.assertEqual(len(self.buffer), self.capacity)
    
    def test_sample(self):
        """Test sampling from buffer."""
        # Add some experiences
        for i in range(10):
            state = torch.randn(10)
            action = torch.tensor(i % 5)
            reward = torch.tensor(i * 0.1)
            next_state = torch.randn(10)
            done = torch.tensor(i % 3 == 0)
            
            self.buffer.push(state, action, reward, next_state, done)
        
        # Sample batch
        batch_size = 5
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        self.assertEqual(states.shape[0], batch_size)
        self.assertEqual(actions.shape[0], batch_size)
        self.assertEqual(rewards.shape[0], batch_size)
        self.assertEqual(next_states.shape[0], batch_size)
        self.assertEqual(dones.shape[0], batch_size)


class TestRLOptimizer(unittest.TestCase):
    """Test RLOptimizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, 'output')
        self.log_dir = os.path.join(self.temp_dir, 'logs')
        
        self.optimizer = RLOptimizer(
            state_dim=100,
            action_dim=10,
            hidden_dim=256,
            learning_rate=1e-3,
            gamma=0.99,
            device='cpu',
            output_dir=self.output_dir,
            log_dir=self.log_dir
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test RLOptimizer initialization."""
        self.assertEqual(self.optimizer.state_dim, 100)
        self.assertEqual(self.optimizer.action_dim, 10)
        self.assertEqual(self.optimizer.hidden_dim, 256)
        self.assertEqual(self.optimizer.learning_rate, 1e-3)
        self.assertEqual(self.optimizer.gamma, 0.99)
        self.assertEqual(self.optimizer.device, torch.device('cpu'))
        self.assertIsNotNone(self.optimizer.policy)
        self.assertIsNotNone(self.optimizer.optimizer)
        self.assertIsNotNone(self.optimizer.replay_buffer)
    
    def test_get_device_auto_cpu(self):
        """Test device selection with auto on CPU."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=False):
                device = self.optimizer._get_device('auto')
                self.assertEqual(device, torch.device('cpu'))
    
    def test_get_device_auto_cuda(self):
        """Test device selection with auto on CUDA."""
        with patch('torch.cuda.is_available', return_value=True):
            device = self.optimizer._get_device('auto')
            self.assertEqual(device, torch.device('cuda'))
    
    def test_smiles_to_features(self):
        """Test SMILES to features conversion."""
        smiles = "CCO"
        features = self.optimizer._smiles_to_features(smiles)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), self.optimizer.state_dim)
    
    def test_apply_action(self):
        """Test applying action to molecule."""
        smiles = "CCO"
        action = 5
        
        modified_smiles = self.optimizer._apply_action(smiles, action)
        
        self.assertIsInstance(modified_smiles, str)
    
    def test_optimize_molecule(self):
        """Test molecule optimization."""
        def mock_reward_function(smiles):
            return len(smiles) * 0.1  # Simple reward function
        
        initial_smiles = "CCO"
        optimized_smiles, rewards, smiles_history = self.optimizer.optimize_molecule(
            initial_smiles, mock_reward_function, max_steps=10
        )
        
        # Verify results
        self.assertIsInstance(optimized_smiles, str)
        self.assertIsInstance(rewards, list)
        self.assertIsInstance(smiles_history, list)
        self.assertEqual(len(rewards), 10)
        self.assertEqual(len(smiles_history), 11)  # Initial + 10 steps
    
    def test_train_on_dataset(self):
        """Test training on dataset."""
        def mock_reward_function(smiles):
            return len(smiles) * 0.1  # Simple reward function
        
        molecules = ["CCO", "CCN", "CC(C)O", "CC(C)N", "CC(C)(C)O"]
        
        training_results = self.optimizer.train_on_dataset(
            molecules, mock_reward_function, episodes=10, max_steps_per_episode=5, save_model=False
        )
        
        # Verify training results
        self.assertIsInstance(training_results, dict)
        self.assertIn('episode_rewards', training_results)
        self.assertIn('episodes', training_results)
        self.assertEqual(len(training_results['episode_rewards']), 10)
    
    def test_evaluate_policy(self):
        """Test policy evaluation."""
        def mock_reward_function(smiles):
            return len(smiles) * 0.1  # Simple reward function
        
        test_molecules = ["CCO", "CCN", "CC(C)O"]
        
        evaluation_results = self.optimizer.evaluate_policy(
            test_molecules, mock_reward_function, max_steps=5
        )
        
        # Verify evaluation results
        self.assertIsInstance(evaluation_results, dict)
        self.assertIn('initial_rewards', evaluation_results)
        self.assertIn('final_rewards', evaluation_results)
        self.assertIn('improvements', evaluation_results)
        self.assertIn('optimized_molecules', evaluation_results)
        self.assertEqual(len(evaluation_results['initial_rewards']), 3)
        self.assertEqual(len(evaluation_results['final_rewards']), 3)
        self.assertEqual(len(evaluation_results['improvements']), 3)
        self.assertEqual(len(evaluation_results['optimized_molecules']), 3)
    
    def test_train_policy(self):
        """Test policy training."""
        # Add some experiences to replay buffer
        for i in range(50):
            state = torch.randn(100)
            action = torch.tensor(i % 10)
            reward = torch.tensor(i * 0.1)
            next_state = torch.randn(100)
            done = torch.tensor(i % 10 == 0)
            
            self.optimizer.replay_buffer.push(state, action, reward, next_state, done)
        
        # Train policy
        self.optimizer._train_policy(batch_size=32)
        
        # Verify training completed without errors
        self.assertTrue(True)  # If we get here, training succeeded
    
    def test_save_model(self):
        """Test model saving."""
        # Set mock training history
        self.optimizer.training_history = {'episode_rewards': [1.0, 0.8]}
        
        # Save model
        model_path = os.path.join(self.temp_dir, 'test_model.pt')
        self.optimizer.save_model(model_path)
        
        # Verify file was created
        self.assertTrue(os.path.exists(model_path))
    
    def test_load_model(self):
        """Test model loading."""
        # Create mock checkpoint
        checkpoint = {
            'policy_state_dict': self.optimizer.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.optimizer.state_dict(),
            'state_dim': 100,
            'action_dim': 10,
            'hidden_dim': 256,
            'learning_rate': 1e-3,
            'gamma': 0.99,
            'training_history': {'episode_rewards': [1.0, 0.8]}
        }
        
        model_path = os.path.join(self.temp_dir, 'test_model.pt')
        torch.save(checkpoint, model_path)
        
        # Load model
        self.optimizer.load_model(model_path)
        
        # Verify loading
        self.assertIsNotNone(self.optimizer.training_history)
    
    def test_get_training_history(self):
        """Test getting training history."""
        # Set mock training history
        self.optimizer.training_history = {'episode_rewards': [1.0, 0.8]}
        
        history = self.optimizer.get_training_history()
        
        self.assertIsInstance(history, dict)
        self.assertIn('episode_rewards', history)


if __name__ == '__main__':
    unittest.main()
