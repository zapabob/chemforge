"""
Tests for ChemForge CLI Module

This module contains tests for the CLI functionality.
"""

import pytest
import click
import tempfile
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from chemforge.cli.main import main
from chemforge.cli.train import train_command
from chemforge.cli.predict import predict_command
from chemforge.cli.admet import admet_command
from chemforge.cli.generate import generate_command
from chemforge.cli.optimize import optimize_command


class TestCLIMain:
    """Test CLI main module."""
    
    def test_main_initialization(self):
        """Test main CLI initialization."""
        with click.testing.CliRunner() as runner:
            result = runner.invoke(main, ["--help"])
            assert result.exit_code == 0
            assert "ChemForge" in result.output
    
    def test_main_with_options(self):
        """Test main CLI with options."""
        with click.testing.CliRunner() as runner:
            result = runner.invoke(main, [
                "--verbose",
                "--output-dir", "test_output",
                "--device", "cpu"
            ])
            assert result.exit_code == 0


class TestCLITrain:
    """Test CLI train command."""
    
    def test_train_command_help(self):
        """Test train command help."""
        with click.testing.CliRunner() as runner:
            result = runner.invoke(train_command, ["--help"])
            assert result.exit_code == 0
            assert "Train a model" in result.output
    
    def test_train_command_validation(self):
        """Test train command validation."""
        with click.testing.CliRunner() as runner:
            result = runner.invoke(train_command, [])
            assert result.exit_code != 0  # Should fail without required data-path
    
    @patch('chemforge.cli.train.ChEMBLLoader')
    @patch('chemforge.cli.train.DataPreprocessor')
    @patch('chemforge.cli.train.TransformerRegressor')
    @patch('chemforge.cli.train.Trainer')
    def test_train_command_success(self, mock_trainer, mock_model, mock_preprocessor, mock_loader):
        """Test successful train command execution."""
        # Mock data
        mock_data = pd.DataFrame({
            'smiles': ['CCO', 'CCN'],
            'target_1': [5.0, 6.0],
            'target_2': [4.0, 5.0]
        })
        
        # Mock loader
        mock_loader_instance = Mock()
        mock_loader_instance.load_data.return_value = mock_data
        mock_loader.return_value = mock_loader_instance
        
        # Mock preprocessor
        mock_preprocessor_instance = Mock()
        mock_preprocessor_instance.preprocess_data.return_value = mock_data
        mock_preprocessor_instance.split_data.return_value = (mock_data, mock_data)
        mock_preprocessor_instance.create_data_loader.return_value = Mock()
        mock_preprocessor.return_value = mock_preprocessor_instance
        
        # Mock model
        mock_model_instance = Mock()
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.return_value = mock_model_instance
        
        # Mock trainer
        mock_trainer_instance = Mock()
        mock_trainer_instance.setup_optimizer.return_value = None
        mock_trainer_instance.setup_scheduler.return_value = None
        mock_trainer_instance.setup_loss_function.return_value = None
        mock_trainer_instance.train.return_value = {'loss': [0.1, 0.05], 'val_loss': [0.2, 0.1]}
        mock_trainer.return_value = mock_trainer_instance
        
        # Create temporary data file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            mock_data.to_csv(f.name, index=False)
            data_path = f.name
        
        try:
            with click.testing.CliRunner() as runner:
                result = runner.invoke(train_command, [
                    "--data-path", data_path,
                    "--model-type", "transformer",
                    "--epochs", "2",
                    "--batch-size", "1"
                ])
                assert result.exit_code == 0
        finally:
            Path(data_path).unlink()


class TestCLIPredict:
    """Test CLI predict command."""
    
    def test_predict_command_help(self):
        """Test predict command help."""
        with click.testing.CliRunner() as runner:
            result = runner.invoke(predict_command, ["--help"])
            assert result.exit_code == 0
            assert "Predict pIC50 values" in result.output
    
    def test_predict_command_validation(self):
        """Test predict command validation."""
        with click.testing.CliRunner() as runner:
            result = runner.invoke(predict_command, [])
            assert result.exit_code != 0  # Should fail without required options
    
    @patch('chemforge.cli.predict.torch.load')
    @patch('chemforge.cli.predict.TransformerRegressor')
    @patch('chemforge.cli.predict.Trainer')
    def test_predict_command_success(self, mock_trainer, mock_model, mock_torch_load):
        """Test successful predict command execution."""
        # Mock torch.load
        mock_checkpoint = {
            'model_type': 'transformer',
            'model_config': {'input_dim': 100, 'output_dim': 2},
            'model_state_dict': {}
        }
        mock_torch_load.return_value = mock_checkpoint
        
        # Mock model
        mock_model_instance = Mock()
        mock_model_instance.load_state_dict.return_value = None
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_instance.eval.return_value = None
        mock_model.return_value = mock_model_instance
        
        # Mock trainer
        mock_trainer_instance = Mock()
        mock_trainer_instance.predict.return_value = [[5.0, 4.0], [6.0, 5.0]]
        mock_trainer.return_value = mock_trainer_instance
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            pd.DataFrame({'smiles': ['CCO', 'CCN']}).to_csv(f.name, index=False)
            data_path = f.name
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            model_path = f.name
        
        try:
            with click.testing.CliRunner() as runner:
                result = runner.invoke(predict_command, [
                    "--model-path", model_path,
                    "--data-path", data_path
                ])
                assert result.exit_code == 0
        finally:
            Path(data_path).unlink()
            Path(model_path).unlink()


class TestCLIADMET:
    """Test CLI ADMET command."""
    
    def test_admet_command_help(self):
        """Test ADMET command help."""
        with click.testing.CliRunner() as runner:
            result = runner.invoke(admet_command, ["--help"])
            assert result.exit_code == 0
            assert "Predict ADMET properties" in result.output
    
    def test_admet_command_validation(self):
        """Test ADMET command validation."""
        with click.testing.CliRunner() as runner:
            result = runner.invoke(admet_command, [])
            assert result.exit_code != 0  # Should fail without required data-path
    
    @patch('chemforge.cli.admet.ADMETPredictor')
    @patch('chemforge.cli.admet.PropertyPredictor')
    @patch('chemforge.cli.admet.ToxicityPredictor')
    @patch('chemforge.cli.admet.DrugLikeness')
    @patch('chemforge.cli.admet.CNS_MPO_Calculator')
    def test_admet_command_success(self, mock_cns_mpo, mock_drug_likeness, mock_toxicity, mock_property, mock_admet):
        """Test successful ADMET command execution."""
        # Mock predictors
        mock_admet_instance = Mock()
        mock_admet_instance.predict_pharmacokinetic_properties.return_value = {'absorption': [0.8, 0.9]}
        mock_admet.return_value = mock_admet_instance
        
        mock_property_instance = Mock()
        mock_property_instance.predict_properties.return_value = {'mw': [100, 120], 'logp': [2.0, 3.0]}
        mock_property.return_value = mock_property_instance
        
        mock_toxicity_instance = Mock()
        mock_toxicity_instance.predict_toxicity.return_value = {'hepatotoxicity': [0.1, 0.2]}
        mock_toxicity.return_value = mock_toxicity_instance
        
        mock_drug_likeness_instance = Mock()
        mock_drug_likeness_instance.calculate_drug_likeness.return_value = {'qed': [0.8, 0.9]}
        mock_drug_likeness.return_value = mock_drug_likeness_instance
        
        mock_cns_mpo_instance = Mock()
        mock_cns_mpo_instance.calculate_cns_mpo.return_value = [4.0, 5.0]
        mock_cns_mpo.return_value = mock_cns_mpo_instance
        
        # Create temporary data file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            pd.DataFrame({'smiles': ['CCO', 'CCN']}).to_csv(f.name, index=False)
            data_path = f.name
        
        try:
            with click.testing.CliRunner() as runner:
                result = runner.invoke(admet_command, [
                    "--data-path", data_path,
                    "--properties", "all"
                ])
                assert result.exit_code == 0
        finally:
            Path(data_path).unlink()


class TestCLIGenerate:
    """Test CLI generate command."""
    
    def test_generate_command_help(self):
        """Test generate command help."""
        with click.testing.CliRunner() as runner:
            result = runner.invoke(generate_command, ["--help"])
            assert result.exit_code == 0
            assert "Generate new molecules" in result.output
    
    def test_generate_command_validation(self):
        """Test generate command validation."""
        with click.testing.CliRunner() as runner:
            result = runner.invoke(generate_command, [])
            assert result.exit_code != 0  # Should fail without required options
    
    @patch('chemforge.cli.generate.torch.load')
    @patch('chemforge.cli.generate.TransformerRegressor')
    @patch('chemforge.cli.generate.Trainer')
    def test_generate_command_success(self, mock_trainer, mock_model, mock_torch_load):
        """Test successful generate command execution."""
        # Mock torch.load
        mock_checkpoint = {
            'model_type': 'transformer',
            'model_config': {'input_dim': 100, 'output_dim': 2},
            'model_state_dict': {}
        }
        mock_torch_load.return_value = mock_checkpoint
        
        # Mock model
        mock_model_instance = Mock()
        mock_model_instance.load_state_dict.return_value = None
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_instance.eval.return_value = None
        mock_model_instance.generate_vae.return_value = (['CCO', 'CCN'], None, None)
        mock_model.return_value = mock_model_instance
        
        # Mock trainer
        mock_trainer_instance = Mock()
        mock_trainer_instance.predict.return_value = [[5.0, 4.0], [6.0, 5.0]]
        mock_trainer.return_value = mock_trainer_instance
        
        # Create temporary model file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            model_path = f.name
        
        try:
            with click.testing.CliRunner() as runner:
                result = runner.invoke(generate_command, [
                    "--model-path", model_path,
                    "--num-molecules", "2",
                    "--generation-method", "vae"
                ])
                assert result.exit_code == 0
        finally:
            Path(model_path).unlink()


class TestCLIOptimize:
    """Test CLI optimize command."""
    
    def test_optimize_command_help(self):
        """Test optimize command help."""
        with click.testing.CliRunner() as runner:
            result = runner.invoke(optimize_command, ["--help"])
            assert result.exit_code == 0
            assert "Optimize molecules" in result.output
    
    def test_optimize_command_validation(self):
        """Test optimize command validation."""
        with click.testing.CliRunner() as runner:
            result = runner.invoke(optimize_command, [])
            assert result.exit_code != 0  # Should fail without required options
    
    @patch('chemforge.cli.optimize.torch.load')
    @patch('chemforge.cli.optimize.TransformerRegressor')
    @patch('chemforge.cli.optimize.Trainer')
    def test_optimize_command_success(self, mock_trainer, mock_model, mock_torch_load):
        """Test successful optimize command execution."""
        # Mock torch.load
        mock_checkpoint = {
            'model_type': 'transformer',
            'model_config': {'input_dim': 100, 'output_dim': 2},
            'model_state_dict': {}
        }
        mock_torch_load.return_value = mock_checkpoint
        
        # Mock model
        mock_model_instance = Mock()
        mock_model_instance.load_state_dict.return_value = None
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_instance.eval.return_value = None
        mock_model_instance.optimize_genetic.return_value = (['CCO', 'CCN'], [{'fitness': 0.8}], [0.8, 0.9])
        mock_model.return_value = mock_model_instance
        
        # Mock trainer
        mock_trainer_instance = Mock()
        mock_trainer_instance.predict.return_value = [[5.0, 4.0], [6.0, 5.0]]
        mock_trainer.return_value = mock_trainer_instance
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            pd.DataFrame({'smiles': ['CCO', 'CCN']}).to_csv(f.name, index=False)
            data_path = f.name
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            model_path = f.name
        
        try:
            with click.testing.CliRunner() as runner:
                result = runner.invoke(optimize_command, [
                    "--model-path", model_path,
                    "--data-path", data_path,
                    "--optimization-method", "genetic",
                    "--num-generations", "2"
                ])
                assert result.exit_code == 0
        finally:
            Path(data_path).unlink()
            Path(model_path).unlink()


class TestCLIIntegration:
    """Test CLI integration."""
    
    def test_cli_integration(self):
        """Test CLI integration."""
        with click.testing.CliRunner() as runner:
            # Test main command
            result = runner.invoke(main, ["--help"])
            assert result.exit_code == 0
            
            # Test subcommands
            result = runner.invoke(main, ["train", "--help"])
            assert result.exit_code == 0
            
            result = runner.invoke(main, ["predict", "--help"])
            assert result.exit_code == 0
            
            result = runner.invoke(main, ["admet", "--help"])
            assert result.exit_code == 0
            
            result = runner.invoke(main, ["generate", "--help"])
            assert result.exit_code == 0
            
            result = runner.invoke(main, ["optimize", "--help"])
            assert result.exit_code == 0


if __name__ == "__main__":
    pytest.main([__file__])

