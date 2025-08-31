"""
Improved Prediction Script for Jigsaw Competition

This script generates predictions using the trained model
with enhanced error handling and logging.
"""

import os
import sys
import logging
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Set up paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODEL_DIR = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'outputs'
SUBMISSIONS_DIR = BASE_DIR / 'submissions'

# Create directories if they don't exist
SUBMISSIONS_DIR.mkdir(exist_ok=True, parents=True)

# Set up logging
def setup_logging():
    """Configure logging for the application."""
    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create handlers
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    
    # Create logs directory if it doesn't exist
    log_dir = BASE_DIR / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    # Create file handler with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'prediction_{timestamp}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    return log_file

logger = logging.getLogger(__name__)
log_file = setup_logging()

class ModelPredictor:
    """Handles model loading and prediction."""
    
    def __init__(self, model_dir: Optional[Path] = None):
        """Initialize with path to the trained model directory."""
        self.model_dir = self._find_latest_model(model_dir) if model_dir is None else Path(model_dir)
        self.embedding_model = None
        self.classifier = None
        self.label_encoder = None
        self.model_info = None
        self.logger = logging.getLogger(__name__)
        
        # Load model components
        self._load_model()
    
    def _find_latest_model(self, model_dir: Optional[Path] = None) -> Path:
        """Find the most recent model directory."""
        if model_dir is None:
            model_dirs = list(MODEL_DIR.glob('*_*_*'))  # Matches timestamped dirs
            if not model_dirs:
                raise FileNotFoundError("No model directories found in models/")
            
            # Sort by creation time (newest first)
            model_dirs.sort(key=os.path.getmtime, reverse=True)
            return model_dirs[0]
        return Path(model_dir)
    
    def _load_model(self):
        """Load the trained model and related components."""
        try:
            self.logger.info(f"Loading model from {self.model_dir}")
            
            # Load model info
            with open(self.model_dir / 'model_info.json', 'r') as f:
                self.model_info = json.load(f)
            
            # Load classifier
            self.classifier = joblib.load(self.model_dir / 'classifier.joblib')
            
            # Load label encoder
            self.label_encoder = joblib.load(self.model_dir / 'label_encoder.joblib')
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(self.model_info['model_name'])
            
            self.logger.info("Model loaded successfully!")
            self.logger.info(f"Model info: {json.dumps(self.model_info, indent=2)}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}", exc_info=True)
            raise
    
    def predict(self, texts: List[str], batch_size: int = 32) -> Dict[str, np.ndarray]:
        """Generate predictions for a list of texts."""
        try:
            # Generate embeddings
            self.logger.info(f"Generating embeddings for {len(texts)} texts...")
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Make predictions
            self.logger.info("Generating predictions...")
            predictions = self.classifier.predict(embeddings)
            probabilities = self.classifier.predict_proba(embeddings)
            
            # Convert to human-readable labels
            labels = self.label_encoder.inverse_transform(predictions)
            
            return {
                'labels': labels,
                'predictions': predictions,
                'probabilities': probabilities,
                'class_names': self.label_encoder.classes_.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}", exc_info=True)
            raise

def load_test_data() -> pd.DataFrame:
    """Load and prepare the test data."""
    try:
        test_path = DATA_DIR / 'test.csv'
        processed_test_path = PROCESSED_DATA_DIR / 'test_processed.csv'
        
        if processed_test_path.exists():
            logger.info(f"Loading processed test data from {processed_test_path}")
            df = pd.read_csv(processed_test_path)
        else:
            logger.info(f"Loading raw test data from {test_path}")
            df = pd.read_csv(test_path)
            
            # Basic preprocessing
            if 'text' not in df.columns and 'body' in df.columns:
                df['text'] = df['body']
        
        # Ensure we have the required columns
        if 'text' not in df.columns:
            raise ValueError("Test data must contain a 'text' or 'body' column")
            
        logger.info(f"Loaded {len(df)} test examples")
        return df
        
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}", exc_info=True)
        raise

def create_submission(predictions: Dict[str, np.ndarray], test_df: pd.DataFrame) -> Path:
    """Create a submission file in the required format."""
    try:
        # Create submission dataframe
        submission = pd.DataFrame({
            'row_id': test_df['row_id'] if 'row_id' in test_df.columns else test_df.index,
            'prediction': predictions['probabilities'][:, 1]  # Probability of positive class
        })
        
        # Save to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        submission_file = SUBMISSIONS_DIR / f'submission_{timestamp}.csv'
        submission.to_csv(submission_file, index=False)
        
        logger.info(f"Submission file created at: {submission_file}")
        return submission_file
        
    except Exception as e:
        logger.error(f"Error creating submission: {str(e)}", exc_info=True)
        raise

def main():
    """Main function to run the prediction pipeline."""
    try:
        logger.info("Starting prediction pipeline...")
        
        # Load test data
        test_df = load_test_data()
        
        # Initialize predictor
        predictor = ModelPredictor()
        
        # Make predictions
        predictions = predictor.predict(test_df['text'].tolist())
        
        # Create submission file
        submission_file = create_submission(predictions, test_df)
        
        logger.info("Prediction pipeline completed successfully!")
        logger.info(f"Log file: {log_file}")
        logger.info(f"Submission file: {submission_file}")
        
        return {
            'status': 'success',
            'submission_file': str(submission_file),
            'log_file': str(log_file),
            'num_predictions': len(test_df)
        }
        
    except Exception as e:
        logger.error(f"Prediction pipeline failed: {str(e)}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e),
            'log_file': str(log_file)
        }

if __name__ == "__main__":
    main()
