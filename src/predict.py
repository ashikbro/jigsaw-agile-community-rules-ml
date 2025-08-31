import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import List, Dict, Union, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Set up paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODEL_DIR = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'outputs'

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JigsawPredictor:
    """Class for making predictions with the Jigsaw baseline model."""
    
    def __init__(self, model_path: str = None, vectorizer_path: str = None):
        """Initialize the predictor with trained model and vectorizer.
        
        Args:
            model_path: Path to the trained model (.joblib)
            vectorizer_path: Path to the trained vectorizer (.joblib)
        """
        # Set default paths if not provided
        if model_path is None:
            model_path = MODEL_DIR / 'baseline_model.joblib'
        if vectorizer_path is None:
            vectorizer_path = MODEL_DIR / 'baseline_model_vectorizer.joblib'
            
        # Load model and vectorizer
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        
        logging.info(f"Loaded model from {model_path}")
        logging.info(f"Loaded vectorizer from {vectorizer_path}")
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess a single text input.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Basic cleaning (can be expanded based on preprocessing steps)
        if not isinstance(text, str):
            return ""
        return text.strip()
    
    def predict_proba(self, text: str) -> float:
        """Predict the probability of the positive class.
        
        Args:
            text: Input text to predict on
            
        Returns:
            Probability of the positive class (rule violation)
        """
        # Preprocess the input
        processed_text = self.preprocess_text(text)
        
        # Transform text using the trained vectorizer
        text_tfidf = self.vectorizer.transform([processed_text])
        
        # Make prediction
        prob = self.model.predict_proba(text_tfidf)[0][1]
        
        return prob
    
    def predict(self, text: str, threshold: float = 0.5) -> int:
        """Predict the class label.
        
        Args:
            text: Input text to predict on
            threshold: Decision threshold
            
        Returns:
            Predicted class (0 for no violation, 1 for violation)
        """
        prob = self.predict_proba(text)
        return 1 if prob >= threshold else 0
    
    def batch_predict(self, texts: List[str], threshold: float = 0.5) -> List[Dict]:
        """Make predictions on a batch of texts.
        
        Args:
            texts: List of input texts
            threshold: Decision threshold
            
        Returns:
            List of prediction dictionaries with 'text', 'probability', and 'prediction' keys
        """
        # Preprocess all texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Transform texts using the trained vectorizer
        texts_tfidf = self.vectorizer.transform(processed_texts)
        
        # Make predictions in batch
        probas = self.model.predict_proba(texts_tfidf)[:, 1]
        predictions = (probas >= threshold).astype(int)
        
        # Prepare results
        results = []
        for text, prob, pred in zip(texts, probas, predictions):
            results.append({
                'text': text,
                'probability': float(prob),
                'prediction': int(pred)
            })
        
        return results
    
    def save_predictions(self, predictions: List[Dict], original_df: pd.DataFrame, 
                        output_file: str, include_text: bool = False):
        """Save predictions to a CSV file with original data.
        
        Args:
            predictions: List of prediction dictionaries
            original_df: Original DataFrame with test data
            output_file: Path to save the predictions
            include_text: Whether to include the original text in the output
        """
        # Convert predictions to DataFrame
        preds_df = pd.DataFrame(predictions)
        
        # Add predictions to original DataFrame
        result_df = original_df.copy()
        result_df['prediction'] = preds_df['prediction']
        result_df['probability'] = preds_df['probability']
        
        # Include original text if requested
        if include_text and 'text' in preds_df.columns:
            result_df['prediction_text'] = preds_df['text']
        
        # Select columns to save
        columns_to_save = ['row_id', 'prediction', 'probability']
        if include_text and 'body' in result_df.columns:
            columns_to_save.append('body')
        
        # Save to CSV
        result_df[columns_to_save].to_csv(output_file, index=False)
        logging.info(f"Saved predictions to {output_file}")
        
        # Print some statistics
        pos_percent = (result_df['prediction'].mean() * 100)
        logging.info(f"Predicted {pos_percent:.2f}% of comments as rule violations")
        
        return result_df


def load_test_data(file_path: str) -> Tuple[List[str], pd.DataFrame]:
    """Load test data from a CSV file.
    
    Args:
        file_path: Path to the test data file
        
    Returns:
        Tuple of (list of texts, original DataFrame)
    """
    # Read the test data
    df = pd.read_csv(file_path)
    
    # Get the text column (assuming it's named 'cleaned_text' from preprocessing)
    text_column = 'cleaned_text' if 'cleaned_text' in df.columns else 'body'
    texts = df[text_column].fillna('').astype(str).tolist()
    
    return texts, df


def main():
    """Main function for making predictions."""
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Make predictions with the Jigsaw baseline model')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the trained model (.joblib)')
    parser.add_argument('--vectorizer_path', type=str, default=None,
                        help='Path to the trained vectorizer (.joblib)')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to the input CSV file')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save the predictions (default: outputs/predictions_<timestamp>.csv)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Decision threshold for classification (default: 0.5)')
    parser.add_argument('--include_text', action='store_true',
                        help='Include original text in the output')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(OUTPUT_DIR / 'prediction.log')
        ]
    )
    
    try:
        # Set default output file if not provided
        if args.output_file is None:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            args.output_file = OUTPUT_DIR / f'predictions_{timestamp}.csv'
        
        # Initialize predictor
        logging.info("Initializing predictor...")
        predictor = JigsawPredictor(
            model_path=args.model_path,
            vectorizer_path=args.vectorizer_path
        )
        
        # Load test data
        logging.info(f"Loading test data from {args.input_file}")
        test_texts, test_df = load_test_data(args.input_file)
        logging.info(f"Loaded {len(test_texts)} test examples")
        
        # Make predictions
        logging.info(f"Making predictions with threshold={args.threshold}...")
        predictions = predictor.batch_predict(test_texts, threshold=args.threshold)
        
        # Save predictions
        logging.info(f"Saving predictions to {args.output_file}")
        result_df = predictor.save_predictions(
            predictions=predictions,
            original_df=test_df,
            output_file=args.output_file,
            include_text=args.include_text
        )
        
        # Print some statistics
        pred_counts = result_df['prediction'].value_counts().to_dict()
        logging.info(f"Prediction counts: {pred_counts}")
        
        logging.info("Prediction completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
