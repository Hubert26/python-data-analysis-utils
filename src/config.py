# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 16:00:13 2024

@author: Hubert Szewczyk
"""

from pathlib import Path
import logging
from utils.file_utils import create_directory

#Directories

# Define directories
ROOT_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT_DIR / 'logs'
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# Create directories if they don't exist
for directory in [LOGS_DIR, DATA_DIR, RESULTS_DIR, PLOTS_DIR]:
    create_directory(directory)
    
#Logger
    
LOGGING_LVL = 'INFO'  # Can be changed to 'DEBUG', 'ERROR', 'INFO', 'WARNING' , etc.
LOG_FILE = LOGS_DIR / 'app.log'

# Clear any existing handlers to avoid duplication
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
    
# Map string levels to logging constants
log_level = getattr(logging, LOGGING_LVL.upper(), logging.WARNING)

# Set up logging
logging.basicConfig(
    level=log_level,  # Set the log level
    format="%(asctime)s [%(levelname)s] %(message)s",  # Define log format
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w')  # Log to file, overwrite each time
    ]
)

# Get the logger instance
logger = logging.getLogger(__name__)
logger.info("Logging has been set up successfully.")