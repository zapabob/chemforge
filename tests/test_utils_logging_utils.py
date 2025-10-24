"""
Unit tests for logging utilities.
"""

import unittest
import tempfile
import os
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

from chemforge.utils.logging_utils import (
    Logger,
    LogManager,
    TrainingLogger,
    PredictionLogger,
    log_context,
    setup_logging
)


class TestLogger(unittest.TestCase):
    """Test Logger class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = os.path.join(self.temp_dir, 'logs')
        self.logger = Logger('test_logger', log_dir=self.log_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test Logger initialization."""
        self.assertEqual(self.logger.name, 'test_logger')
        self.assertEqual(self.logger.level, logging.INFO)
        self.assertEqual(self.logger.log_dir, Path(self.log_dir))
        self.assertIsNotNone(self.logger.logger)
    
    def test_init_custom_level(self):
        """Test Logger initialization with custom level."""
        logger = Logger('test_logger', level='DEBUG', log_dir=self.log_dir)
        self.assertEqual(logger.level, logging.DEBUG)
    
    def test_debug(self):
        """Test debug logging."""
        with patch.object(self.logger.logger, 'debug') as mock_debug:
            self.logger.debug("Debug message")
            mock_debug.assert_called_once_with("Debug message")
    
    def test_info(self):
        """Test info logging."""
        with patch.object(self.logger.logger, 'info') as mock_info:
            self.logger.info("Info message")
            mock_info.assert_called_once_with("Info message")
    
    def test_warning(self):
        """Test warning logging."""
        with patch.object(self.logger.logger, 'warning') as mock_warning:
            self.logger.warning("Warning message")
            mock_warning.assert_called_once_with("Warning message")
    
    def test_error(self):
        """Test error logging."""
        with patch.object(self.logger.logger, 'error') as mock_error:
            self.logger.error("Error message")
            mock_error.assert_called_once_with("Error message")
    
    def test_critical(self):
        """Test critical logging."""
        with patch.object(self.logger.logger, 'critical') as mock_critical:
            self.logger.critical("Critical message")
            mock_critical.assert_called_once_with("Critical message")
    
    def test_log_performance(self):
        """Test performance logging."""
        with patch.object(self.logger.logger, 'info') as mock_info:
            self.logger.log_performance("test_operation", 1.5)
            mock_info.assert_called_once_with("Performance - test_operation: 1.5000s")
    
    def test_log_model_metrics(self):
        """Test model metrics logging."""
        metrics = {'accuracy': 0.85, 'loss': 0.15}
        
        with patch.object(self.logger.logger, 'info') as mock_info:
            self.logger.log_model_metrics(metrics)
            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            self.assertIn('Model Metrics', call_args)
            self.assertIn('accuracy', call_args)
            self.assertIn('loss', call_args)
    
    def test_log_model_metrics_with_epoch(self):
        """Test model metrics logging with epoch."""
        metrics = {'accuracy': 0.85, 'loss': 0.15}
        
        with patch.object(self.logger.logger, 'info') as mock_info:
            self.logger.log_model_metrics(metrics, epoch=10)
            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            self.assertIn('Epoch 10', call_args)
            self.assertIn('Model Metrics', call_args)
    
    def test_log_data_info(self):
        """Test data info logging."""
        data_info = {'total_samples': 1000, 'features': 50}
        
        with patch.object(self.logger.logger, 'info') as mock_info:
            self.logger.log_data_info(data_info)
            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            self.assertIn('Data Info', call_args)
            self.assertIn('total_samples', call_args)
    
    def test_log_config(self):
        """Test configuration logging."""
        config = {'model_type': 'transformer', 'epochs': 100}
        
        with patch.object(self.logger.logger, 'info') as mock_info:
            self.logger.log_config(config)
            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            self.assertIn('Configuration', call_args)
            self.assertIn('model_type', call_args)
    
    def test_log_error_with_traceback(self):
        """Test error logging with traceback."""
        with patch.object(self.logger.logger, 'error') as mock_error:
            self.logger.log_error_with_traceback("Error message", exc_info=True)
            mock_error.assert_called_once_with("Error message", exc_info=True)
    
    def test_set_level(self):
        """Test logging level setting."""
        self.logger.set_level('DEBUG')
        self.assertEqual(self.logger.level, logging.DEBUG)
    
    def test_add_file_handler(self):
        """Test additional file handler."""
        log_file = os.path.join(self.temp_dir, 'additional.log')
        self.logger.add_file_handler(log_file, 'DEBUG')
        
        # Check if file handler was added
        self.assertGreater(len(self.logger.logger.handlers), 1)
    
    def test_remove_handler(self):
        """Test handler removal."""
        # Add a file handler
        log_file = os.path.join(self.temp_dir, 'test.log')
        self.logger.add_file_handler(log_file, 'DEBUG')
        
        initial_handlers = len(self.logger.logger.handlers)
        
        # Remove file handler
        self.logger.remove_handler('FileHandler')
        
        # Check if handler was removed
        self.assertLess(len(self.logger.logger.handlers), initial_handlers)
    
    def test_get_log_files(self):
        """Test log file retrieval."""
        log_files = self.logger.get_log_files()
        
        # Should have at least the main log file
        self.assertGreaterEqual(len(log_files), 0)
    
    def test_clear_logs(self):
        """Test log clearing."""
        # Create some log files
        log_file = self.logger.log_dir / f"{self.logger.name}.log"
        log_file.write_text("test log content")
        
        # Clear logs
        self.logger.clear_logs()
        
        # Check if log files were cleared
        self.assertFalse(log_file.exists())


class TestLogManager(unittest.TestCase):
    """Test LogManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = os.path.join(self.temp_dir, 'logs')
        self.log_manager = LogManager(self.log_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test LogManager initialization."""
        self.assertEqual(self.log_manager.log_dir, Path(self.log_dir))
        self.assertEqual(self.log_manager.loggers, {})
    
    def test_get_logger(self):
        """Test logger retrieval."""
        logger = self.log_manager.get_logger('test_logger')
        
        self.assertIsInstance(logger, Logger)
        self.assertEqual(logger.name, 'test_logger')
        self.assertIn('test_logger', self.log_manager.loggers)
    
    def test_get_logger_existing(self):
        """Test logger retrieval for existing logger."""
        logger1 = self.log_manager.get_logger('test_logger')
        logger2 = self.log_manager.get_logger('test_logger')
        
        self.assertIs(logger1, logger2)
    
    def test_get_logger_custom_level(self):
        """Test logger retrieval with custom level."""
        logger = self.log_manager.get_logger('test_logger', 'DEBUG')
        
        self.assertEqual(logger.level, logging.DEBUG)
    
    def test_get_all_loggers(self):
        """Test all loggers retrieval."""
        logger1 = self.log_manager.get_logger('logger1')
        logger2 = self.log_manager.get_logger('logger2')
        
        all_loggers = self.log_manager.get_all_loggers()
        
        self.assertEqual(len(all_loggers), 2)
        self.assertIn('logger1', all_loggers)
        self.assertIn('logger2', all_loggers)
    
    def test_set_level_all(self):
        """Test level setting for all loggers."""
        logger1 = self.log_manager.get_logger('logger1')
        logger2 = self.log_manager.get_logger('logger2')
        
        self.log_manager.set_level_all('DEBUG')
        
        self.assertEqual(logger1.level, logging.DEBUG)
        self.assertEqual(logger2.level, logging.DEBUG)
    
    def test_clear_all_logs(self):
        """Test log clearing for all loggers."""
        logger1 = self.log_manager.get_logger('logger1')
        logger2 = self.log_manager.get_logger('logger2')
        
        # Create some log files
        log_file1 = logger1.log_dir / f"{logger1.name}.log"
        log_file2 = logger2.log_dir / f"{logger2.name}.log"
        log_file1.write_text("test log content")
        log_file2.write_text("test log content")
        
        # Clear all logs
        self.log_manager.clear_all_logs()
        
        # Check if log files were cleared
        self.assertFalse(log_file1.exists())
        self.assertFalse(log_file2.exists())
    
    def test_get_log_summary(self):
        """Test log summary retrieval."""
        logger1 = self.log_manager.get_logger('logger1')
        logger2 = self.log_manager.get_logger('logger2')
        
        # Create some log files
        log_file1 = logger1.log_dir / f"{logger1.name}.log"
        log_file2 = logger2.log_dir / f"{logger2.name}.log"
        log_file1.write_text("test log content")
        log_file2.write_text("test log content")
        
        summary = self.log_manager.get_log_summary()
        
        self.assertIn('logger1', summary)
        self.assertIn('logger2', summary)
        self.assertIn('log_files', summary['logger1'])
        self.assertIn('total_size', summary['logger1'])
        self.assertIn('level', summary['logger1'])


class TestTrainingLogger(unittest.TestCase):
    """Test TrainingLogger class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = os.path.join(self.temp_dir, 'logs')
        self.training_logger = TrainingLogger(log_dir=self.log_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test TrainingLogger initialization."""
        self.assertIsInstance(self.training_logger.logger, Logger)
        self.assertEqual(self.training_logger.logger.name, 'training')
        self.assertIsNone(self.training_logger.training_start)
        self.assertIsNone(self.training_logger.epoch_start)
    
    def test_start_training(self):
        """Test training start logging."""
        config = {'epochs': 100, 'batch_size': 32}
        
        with patch.object(self.training_logger.logger, 'info') as mock_info:
            self.training_logger.start_training(config)
            
            # Check if training start was logged
            self.assertIsNotNone(self.training_logger.training_start)
            self.assertGreater(mock_info.call_count, 0)
    
    def test_end_training(self):
        """Test training end logging."""
        self.training_logger.training_start = datetime.now()
        final_metrics = {'accuracy': 0.85, 'loss': 0.15}
        
        with patch.object(self.training_logger.logger, 'info') as mock_info:
            self.training_logger.end_training(final_metrics)
            
            # Check if training end was logged
            self.assertGreater(mock_info.call_count, 0)
    
    def test_start_epoch(self):
        """Test epoch start logging."""
        with patch.object(self.training_logger.logger, 'info') as mock_info:
            self.training_logger.start_epoch(5, 100)
            
            # Check if epoch start was logged
            self.assertIsNotNone(self.training_logger.epoch_start)
            mock_info.assert_called_once_with("Epoch 5/100 started")
    
    def test_end_epoch(self):
        """Test epoch end logging."""
        self.training_logger.epoch_start = datetime.now()
        metrics = {'accuracy': 0.85, 'loss': 0.15}
        
        with patch.object(self.training_logger.logger, 'log_performance') as mock_perf:
            with patch.object(self.training_logger.logger, 'log_model_metrics') as mock_metrics:
                self.training_logger.end_epoch(5, metrics)
                
                # Check if epoch end was logged
                mock_perf.assert_called_once()
                mock_metrics.assert_called_once_with(metrics, 5)
    
    def test_log_batch(self):
        """Test batch logging."""
        with patch.object(self.training_logger.logger, 'debug') as mock_debug:
            self.training_logger.log_batch(10, 32, 0.5)
            mock_debug.assert_called_once_with("Batch 10 (size: 32) - Loss: 0.500000")
    
    def test_log_validation(self):
        """Test validation logging."""
        metrics = {'accuracy': 0.85, 'loss': 0.15}
        
        with patch.object(self.training_logger.logger, 'info') as mock_info:
            self.training_logger.log_validation(metrics)
            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            self.assertIn('Validation Metrics', call_args)
    
    def test_log_checkpoint(self):
        """Test checkpoint logging."""
        with patch.object(self.training_logger.logger, 'info') as mock_info:
            self.training_logger.log_checkpoint(10, '/path/to/checkpoint.pt')
            mock_info.assert_called_once_with("Checkpoint saved at epoch 10: /path/to/checkpoint.pt")
    
    def test_log_early_stopping(self):
        """Test early stopping logging."""
        with patch.object(self.training_logger.logger, 'warning') as mock_warning:
            self.training_logger.log_early_stopping(50, 10)
            mock_warning.assert_called_once_with("Early stopping triggered at epoch 50 (patience: 10)")
    
    def test_log_error(self):
        """Test error logging."""
        error = Exception("Test error")
        
        with patch.object(self.training_logger.logger, 'log_error_with_traceback') as mock_error:
            self.training_logger.log_error(error, "test context")
            mock_error.assert_called_once()
            call_args = mock_error.call_args[0][0]
            self.assertIn('Training error test context', call_args)


class TestPredictionLogger(unittest.TestCase):
    """Test PredictionLogger class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = os.path.join(self.temp_dir, 'logs')
        self.prediction_logger = PredictionLogger(log_dir=self.log_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test PredictionLogger initialization."""
        self.assertIsInstance(self.prediction_logger.logger, Logger)
        self.assertEqual(self.prediction_logger.logger.name, 'prediction')
        self.assertIsNone(self.prediction_logger.prediction_start)
    
    def test_start_prediction(self):
        """Test prediction start logging."""
        with patch.object(self.prediction_logger.logger, 'info') as mock_info:
            self.prediction_logger.start_prediction('/path/to/model.pt', '/path/to/data.csv')
            
            # Check if prediction start was logged
            self.assertIsNotNone(self.prediction_logger.prediction_start)
            self.assertGreater(mock_info.call_count, 0)
    
    def test_end_prediction(self):
        """Test prediction end logging."""
        self.prediction_logger.prediction_start = datetime.now()
        
        with patch.object(self.prediction_logger.logger, 'log_performance') as mock_perf:
            with patch.object(self.prediction_logger.logger, 'info') as mock_info:
                self.prediction_logger.end_prediction(100, '/path/to/output.csv')
                
                # Check if prediction end was logged
                mock_perf.assert_called_once()
                self.assertGreater(mock_info.call_count, 0)
    
    def test_log_prediction_batch(self):
        """Test prediction batch logging."""
        with patch.object(self.prediction_logger.logger, 'debug') as mock_debug:
            self.prediction_logger.log_prediction_batch(10, 32)
            mock_debug.assert_called_once_with("Processing batch 10 (size: 32)")
    
    def test_log_prediction_metrics(self):
        """Test prediction metrics logging."""
        metrics = {'accuracy': 0.85, 'precision': 0.82}
        
        with patch.object(self.prediction_logger.logger, 'log_model_metrics') as mock_metrics:
            self.prediction_logger.log_prediction_metrics(metrics)
            mock_metrics.assert_called_once_with(metrics)
    
    def test_log_error(self):
        """Test error logging."""
        error = Exception("Test error")
        
        with patch.object(self.prediction_logger.logger, 'log_error_with_traceback') as mock_error:
            self.prediction_logger.log_error(error, "test context")
            mock_error.assert_called_once()
            call_args = mock_error.call_args[0][0]
            self.assertIn('Prediction error test context', call_args)


class TestLogContext(unittest.TestCase):
    """Test log_context context manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = os.path.join(self.temp_dir, 'logs')
        self.logger = Logger('test_logger', log_dir=self.log_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_log_context_success(self):
        """Test log_context with successful operation."""
        with patch.object(self.logger, 'info') as mock_info:
            with patch.object(self.logger, 'log_performance') as mock_perf:
                with log_context(self.logger, 'test_operation'):
                    pass
                
                # Check if operation was logged
                mock_info.assert_called()
                mock_perf.assert_called_once()
    
    def test_log_context_error(self):
        """Test log_context with error."""
        with patch.object(self.logger, 'log_performance') as mock_perf:
            with patch.object(self.logger, 'log_error_with_traceback') as mock_error:
                with self.assertRaises(ValueError):
                    with log_context(self.logger, 'test_operation'):
                        raise ValueError("Test error")
                
                # Check if error was logged
                mock_perf.assert_called_once()
                mock_error.assert_called_once()


class TestSetupLogging(unittest.TestCase):
    """Test setup_logging function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = os.path.join(self.temp_dir, 'logs')
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_setup_logging(self):
        """Test logging setup."""
        log_manager = setup_logging(level='DEBUG', log_dir=self.log_dir)
        
        self.assertIsInstance(log_manager, LogManager)
        self.assertEqual(log_manager.log_dir, Path(self.log_dir))
        
        # Check if main logger was created
        app_logger = log_manager.get_logger('chemforge')
        self.assertEqual(app_logger.name, 'chemforge')
        self.assertEqual(app_logger.level, logging.DEBUG)


if __name__ == '__main__':
    unittest.main()
