"""
Logging utilities for ChemForge platform.

This module provides logging functionality including log management, formatting,
and file handling.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import json
import sys
from contextlib import contextmanager


class Logger:
    """Enhanced logger class."""
    
    def __init__(self, name: str, level: str = 'INFO', 
                 log_dir: Optional[str] = None,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            level: Logging level
            log_dir: Log directory
            max_file_size: Maximum log file size in bytes
            backup_count: Number of backup files to keep
        """
        self.name = name
        self.level = getattr(logging, level.upper())
        self.log_dir = Path(log_dir) if log_dir else Path('./logs')
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        self.console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        self.file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Setup handlers
        self._setup_console_handler()
        self._setup_file_handler()
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def _setup_console_handler(self):
        """Setup console handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_handler.setFormatter(self.console_formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(self):
        """Setup file handler with rotation."""
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file path
        log_file = self.log_dir / f"{self.name}.log"
        
        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        file_handler.setLevel(self.level)
        file_handler.setFormatter(self.file_formatter)
        self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(message, **kwargs)
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics."""
        self.info(f"Performance - {operation}: {duration:.4f}s", **kwargs)
    
    def log_model_metrics(self, metrics: Dict[str, float], epoch: int = None):
        """Log model metrics."""
        if epoch is not None:
            self.info(f"Epoch {epoch} - Model Metrics: {json.dumps(metrics, indent=2)}")
        else:
            self.info(f"Model Metrics: {json.dumps(metrics, indent=2)}")
    
    def log_data_info(self, data_info: Dict[str, Any]):
        """Log data information."""
        self.info(f"Data Info: {json.dumps(data_info, indent=2)}")
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration."""
        self.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    def log_error_with_traceback(self, message: str, exc_info: bool = True):
        """Log error with traceback."""
        self.logger.error(message, exc_info=exc_info)
    
    def set_level(self, level: str):
        """Set logging level."""
        self.level = getattr(logging, level.upper())
        self.logger.setLevel(self.level)
        
        # Update all handlers
        for handler in self.logger.handlers:
            handler.setLevel(self.level)
    
    def add_file_handler(self, log_file: str, level: str = 'INFO'):
        """Add additional file handler."""
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(self.file_formatter)
        self.logger.addHandler(file_handler)
    
    def remove_handler(self, handler_type: str):
        """Remove handler by type."""
        handlers_to_remove = []
        for handler in self.logger.handlers:
            if isinstance(handler, getattr(logging, handler_type)):
                handlers_to_remove.append(handler)
        
        for handler in handlers_to_remove:
            self.logger.removeHandler(handler)
            handler.close()
    
    def get_log_files(self) -> List[Path]:
        """Get list of log files."""
        log_files = []
        
        # Get main log file
        main_log = self.log_dir / f"{self.name}.log"
        if main_log.exists():
            log_files.append(main_log)
        
        # Get backup log files
        for i in range(1, self.backup_count + 1):
            backup_log = self.log_dir / f"{self.name}.log.{i}"
            if backup_log.exists():
                log_files.append(backup_log)
        
        return log_files
    
    def clear_logs(self):
        """Clear all log files."""
        log_files = self.get_log_files()
        for log_file in log_files:
            if log_file.exists():
                log_file.unlink()
        
        self.info("Cleared all log files")


class LogManager:
    """Log manager for multiple loggers."""
    
    def __init__(self, log_dir: str = './logs'):
        """
        Initialize log manager.
        
        Args:
            log_dir: Log directory
        """
        self.log_dir = Path(log_dir)
        self.loggers: Dict[str, Logger] = {}
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def get_logger(self, name: str, level: str = 'INFO') -> Logger:
        """
        Get or create logger.
        
        Args:
            name: Logger name
            level: Logging level
            
        Returns:
            Logger instance
        """
        if name not in self.loggers:
            self.loggers[name] = Logger(
                name=name,
                level=level,
                log_dir=str(self.log_dir)
            )
        
        return self.loggers[name]
    
    def get_all_loggers(self) -> Dict[str, Logger]:
        """Get all loggers."""
        return self.loggers.copy()
    
    def set_level_all(self, level: str):
        """Set level for all loggers."""
        for logger in self.loggers.values():
            logger.set_level(level)
    
    def clear_all_logs(self):
        """Clear logs for all loggers."""
        for logger in self.loggers.values():
            logger.clear_logs()
    
    def get_log_summary(self) -> Dict[str, Any]:
        """Get log summary for all loggers."""
        summary = {}
        
        for name, logger in self.loggers.items():
            log_files = logger.get_log_files()
            total_size = sum(f.stat().st_size for f in log_files if f.exists())
            
            summary[name] = {
                'log_files': len(log_files),
                'total_size': total_size,
                'level': logger.level.name
            }
        
        return summary


class TrainingLogger:
    """Specialized logger for training processes."""
    
    def __init__(self, name: str = 'training', log_dir: str = './logs'):
        """
        Initialize training logger.
        
        Args:
            name: Logger name
            log_dir: Log directory
        """
        self.logger = Logger(name, log_dir=log_dir)
        self.training_start = None
        self.epoch_start = None
    
    def start_training(self, config: Dict[str, Any]):
        """Log training start."""
        self.training_start = datetime.now()
        self.logger.info("=" * 50)
        self.logger.info("TRAINING STARTED")
        self.logger.info("=" * 50)
        self.logger.log_config(config)
    
    def end_training(self, final_metrics: Dict[str, float]):
        """Log training end."""
        if self.training_start:
            duration = (datetime.now() - self.training_start).total_seconds()
            self.logger.info(f"Training duration: {duration:.2f} seconds")
        
        self.logger.info("=" * 50)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info("=" * 50)
        self.logger.log_model_metrics(final_metrics)
    
    def start_epoch(self, epoch: int, total_epochs: int):
        """Log epoch start."""
        self.epoch_start = datetime.now()
        self.logger.info(f"Epoch {epoch}/{total_epochs} started")
    
    def end_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Log epoch end."""
        if self.epoch_start:
            duration = (datetime.now() - self.epoch_start).total_seconds()
            self.logger.log_performance(f"Epoch {epoch}", duration)
        
        self.logger.log_model_metrics(metrics, epoch)
    
    def log_batch(self, batch_idx: int, batch_size: int, loss: float):
        """Log batch information."""
        self.logger.debug(f"Batch {batch_idx} (size: {batch_size}) - Loss: {loss:.6f}")
    
    def log_validation(self, metrics: Dict[str, float]):
        """Log validation metrics."""
        self.logger.info(f"Validation Metrics: {json.dumps(metrics, indent=2)}")
    
    def log_checkpoint(self, epoch: int, checkpoint_path: str):
        """Log checkpoint save."""
        self.logger.info(f"Checkpoint saved at epoch {epoch}: {checkpoint_path}")
    
    def log_early_stopping(self, epoch: int, patience: int):
        """Log early stopping."""
        self.logger.warning(f"Early stopping triggered at epoch {epoch} (patience: {patience})")
    
    def log_error(self, error: Exception, context: str = ""):
        """Log training error."""
        self.logger.log_error_with_traceback(f"Training error {context}: {str(error)}")


class PredictionLogger:
    """Specialized logger for prediction processes."""
    
    def __init__(self, name: str = 'prediction', log_dir: str = './logs'):
        """
        Initialize prediction logger.
        
        Args:
            name: Logger name
            log_dir: Log directory
        """
        self.logger = Logger(name, log_dir=log_dir)
        self.prediction_start = None
    
    def start_prediction(self, model_path: str, data_path: str):
        """Log prediction start."""
        self.prediction_start = datetime.now()
        self.logger.info("=" * 50)
        self.logger.info("PREDICTION STARTED")
        self.logger.info("=" * 50)
        self.logger.info(f"Model: {model_path}")
        self.logger.info(f"Data: {data_path}")
    
    def end_prediction(self, num_predictions: int, output_path: str):
        """Log prediction end."""
        if self.prediction_start:
            duration = (datetime.now() - self.prediction_start).total_seconds()
            self.logger.log_performance("Prediction", duration)
        
        self.logger.info(f"Predictions completed: {num_predictions} predictions")
        self.logger.info(f"Output saved to: {output_path}")
        self.logger.info("=" * 50)
        self.logger.info("PREDICTION COMPLETED")
        self.logger.info("=" * 50)
    
    def log_prediction_batch(self, batch_idx: int, batch_size: int):
        """Log prediction batch."""
        self.logger.debug(f"Processing batch {batch_idx} (size: {batch_size})")
    
    def log_prediction_metrics(self, metrics: Dict[str, float]):
        """Log prediction metrics."""
        self.logger.log_model_metrics(metrics)
    
    def log_error(self, error: Exception, context: str = ""):
        """Log prediction error."""
        self.logger.log_error_with_traceback(f"Prediction error {context}: {str(error)}")


@contextmanager
def log_context(logger: Logger, operation: str):
    """Context manager for logging operations."""
    start_time = datetime.now()
    logger.info(f"Starting {operation}")
    
    try:
        yield
        duration = (datetime.now() - start_time).total_seconds()
        logger.log_performance(operation, duration)
        logger.info(f"Completed {operation}")
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.log_performance(f"{operation} (failed)", duration)
        logger.log_error_with_traceback(f"Error in {operation}: {str(e)}")
        raise


def setup_logging(level: str = 'INFO', log_dir: str = './logs') -> LogManager:
    """
    Setup logging for the application.
    
    Args:
        level: Logging level
        log_dir: Log directory
        
    Returns:
        Log manager instance
    """
    log_manager = LogManager(log_dir)
    
    # Create main application logger
    app_logger = log_manager.get_logger('chemforge', level)
    app_logger.info("ChemForge logging system initialized")
    
    return log_manager
