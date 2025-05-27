# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import logging
import sys
import os
from typing import Optional


def get_logger(name: str = "rf-detr", level: Optional[int] = None) -> logging.Logger:
    """Creates and configures a logger with stdout and stderr handlers.
    
    This function creates a logger that sends INFO and DEBUG level logs to stdout,
    and WARNING, ERROR, and CRITICAL level logs to stderr. If the logger already
    has handlers, it returns the existing logger without adding new handlers.
    
    The log level can be specified directly or through the LOG_LEVEL environment
    variable.
    
    Args:
        name: The name of the logger. Defaults to "rf-detr".
        level: The logging level to set. If None, uses the LOG_LEVEL environment
            variable, defaulting to INFO if not set.
    
    Returns:
        A configured logging.Logger instance.
    """
    if level is None:
        level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.addFilter(lambda r: r.levelno <= logging.INFO)
        stdout_handler.setFormatter(formatter)

        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)
        stderr_handler.setFormatter(formatter)

        logger.addHandler(stdout_handler)
        logger.addHandler(stderr_handler)
        logger.propagate = False

    return logger
