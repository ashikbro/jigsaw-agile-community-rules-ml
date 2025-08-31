import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'outputs'
MODEL_DIR = BASE_DIR / 'models'

# Create directories if they don't exist
for directory in [DATA_DIR, OUTPUT_DIR, MODEL_DIR]:
    os.makedirs(directory, exist_ok=True)

# Model parameters
MODEL_CONFIG = {
    'model_name': 'bert-base-uncased',  # Default model
    'max_length': 256,  # Maximum sequence length
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 3,
    'num_labels': 2,  # Binary classification
    'class_weights': [1.0, 1.0]  # Can be adjusted based on class imbalance
}

# Training parameters
TRAIN_CONFIG = {
    'train_frac': 0.8,  # Fraction of data to use for training
    'val_frac': 0.1,    # Fraction of data to use for validation
    'test_frac': 0.1,   # Fraction of data to use for testing
    'random_state': 42,  # For reproducibility
}

# Data processing parameters
PROCESSING_CONFIG = {
    'text_columns': ['comment_text', 'rule_text'],  # Update based on actual data
    'label_column': 'label',  # Update based on actual data
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'INFO',
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': OUTPUT_DIR / 'training.log',
            'formatter': 'standard',
            'level': 'DEBUG',
        },
    },
    'loggers': {
        '': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': True
        },
    }
}
