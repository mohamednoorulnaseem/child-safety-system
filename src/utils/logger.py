"""
Logging configuration for Child Safety System
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Import settings
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import LOGGING, LOG_DIR


def setup_logger(name: str = 'ChildSafetySystem') -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOGGING['level']))
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    formatter = logging.Formatter(LOGGING['format'])
    
    # File handler with rotation
    log_file = LOGGING['file']
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=LOGGING['max_bytes'],
        backupCount=LOGGING['backup_count']
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# Create default logger
logger = setup_logger()
