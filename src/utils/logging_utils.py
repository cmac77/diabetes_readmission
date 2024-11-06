# src/utils/logging_utils.py
import logging
from json_log_formatter import JSONFormatter
from pathlib import Path


def setup_logging(log_file: Path):
    """
    Set up structured JSON logging.

    Args:
        log_file (Path): Path to the log file.
    """
    # Create a JSON formatter
    formatter = JSONFormatter()

    # File handler to write logs to a file
    json_handler = logging.FileHandler(log_file)
    json_handler.setFormatter(formatter)

    # Console handler to output logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Configure the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(json_handler)
    logger.addHandler(console_handler)

    return logger
