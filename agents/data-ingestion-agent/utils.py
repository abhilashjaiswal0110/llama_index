"""
Utility functions for data ingestion agent
"""

import logging
import yaml
from pathlib import Path
from typing import Any, Dict
from rich.logging import RichHandler

from agent import IngestionConfig


def setup_logging(level: int = logging.INFO, log_file: str = "ingestion.log"):
    """
    Setup logging configuration
    
    Args:
        level: Logging level
        log_file: Log file path
    """
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    
    # Rich console handler
    console_handler = RichHandler(rich_tracebacks=True)
    console_handler.setLevel(level)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def load_config(config_path: str) -> IngestionConfig:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        IngestionConfig object
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Flatten nested config structure
    flat_config = {}
    for section, values in config_dict.items():
        if isinstance(values, dict):
            flat_config.update(values)
        else:
            flat_config[section] = values
    
    return IngestionConfig(**flat_config)


def validate_path(path: str) -> Path:
    """
    Validate and return Path object
    
    Args:
        path: Path string
        
    Returns:
        Path object
        
    Raises:
        FileNotFoundError: If path doesn't exist
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    return p


def format_size(bytes: int) -> str:
    """
    Format bytes as human-readable size
    
    Args:
        bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} PB"


def estimate_tokens(text: str) -> int:
    """
    Estimate number of tokens in text
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    # Rough estimation: ~4 characters per token
    return len(text) // 4


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe filesystem usage
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    import re
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Limit length
    return sanitized[:255]


def extract_metadata_from_path(file_path: Path) -> Dict[str, Any]:
    """
    Extract metadata from file path
    
    Args:
        file_path: Path object
        
    Returns:
        Dictionary of metadata
    """
    import os
    from datetime import datetime
    
    stat = file_path.stat()
    
    return {
        'filename': file_path.name,
        'file_extension': file_path.suffix,
        'file_size': stat.st_size,
        'file_size_human': format_size(stat.st_size),
        'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
        'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        'absolute_path': str(file_path.absolute()),
    }


def load_environment():
    """Load environment variables from .env file"""
    from dotenv import load_dotenv
    load_dotenv()


def create_checkpoint(data: Dict[str, Any], checkpoint_file: str):
    """
    Create checkpoint file for resuming
    
    Args:
        data: Checkpoint data
        checkpoint_file: Checkpoint file path
    """
    import json
    
    with open(checkpoint_file, 'w') as f:
        json.dump(data, f, indent=2)


def load_checkpoint(checkpoint_file: str) -> Dict[str, Any]:
    """
    Load checkpoint data
    
    Args:
        checkpoint_file: Checkpoint file path
        
    Returns:
        Checkpoint data dictionary
    """
    import json
    
    path = Path(checkpoint_file)
    if not path.exists():
        return {}
    
    with open(path, 'r') as f:
        return json.load(f)
