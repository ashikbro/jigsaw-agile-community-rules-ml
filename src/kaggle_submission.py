"""Kaggle Submission System for Reddit Comment Rule Violation Detection.

This module provides a complete system for preparing and submitting solutions
to the Kaggle competition, including proper evaluation metrics and constraints.
"""

import pandas as pd
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import time
import json
import warnings
from dataclasses import dataclass, asdict
import gc
import psutil
import os

# Import our custom modules
from reddit_data_pipeline import RedditDataPipeline
from reddit_classifier import (
    RedditClassifierTrainer, EnsembleClassifier, ModelConfig
)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class KaggleConfig:
    """Configuration for Kaggle submission."""
    # Competition constraints
    max_runtime_hours: int = 12
    memory_limit_gb: int = 16
    
    # Model configuration
    model_name: str = 'bert-base-uncased'
    max_length: int = 256  # Reduced for memory efficiency
    batch_size: int = 8    # Smaller batch size for memory
    num_epochs: int = 2    # Fewer epochs due to time constraint
    learning_rate: float = 2e-5
    
    # Cross-validation
    n_folds: int = 3       # Fewer folds for time efficiency
    
    # Ensemble weights
    transformer_weight: float = 0.7
    rf_weight: float = 0.2
    lr_weight: float = 0.1
    
    # Output paths
    submission_file: str = 'submission.csv'
    model_save_path: str = 'models/reddit_classifier'
    
    # Performance monitoring
    memory_check_interval: int = 100  # Check memory every N batches
    
class PerformanceMonitor:
    """Monitor system performance during training."""
    
    def __init__(self, config: KaggleConfig):
        self.config = config
        self.start_time = time.time()
        self.memory_usage = []
        self.process = psutil.Process(os.getpid())
    
    def check_time_limit(self) -> bool:
        """Check if we're approaching time limit."""
        elapsed_hours = (time.time() - self.start_time) / 3600
        return elapsed_hours < (self.config.max_runtime_hours - 0.5)  # 30min buffer
    
    def check_memory_usage(self) -> Dict[str, float]:
        """Check current memory usage."""
        memory_info = self.process.memory_info()
        memory_gb = memory_info.rss / (1024 ** 3)
        
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)
        
        usage = {
            'ram_gb': memory_gb,
            'gpu_gb': gpu_memory,
            'ram_percent': (memory_gb / self.config.memory_limit_gb) * 100
        }
        
        self.memory_usage.append(usage)
        return usage
    
    def cleanup_memory(self):
        """Clean up memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_remaining_time(self) -> float:
        """Get remaining time in hours."""
        elapsed_hours = (time.time() - self.start_time) / 3600
        return max(0, self.config.max_runtime_hours - elapsed_hours)

class KaggleEvaluator:
    """Evaluator for Kaggle competition metrics."""
    
    @staticmethod
    def column_averaged_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate column-averaged AUC as specified in competition.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted probabilities
            
        Returns:
            Column-averaged AUC score
        """
        if len(y_true.shape) == 1:
            # Single column case
            return roc_auc_score(y_true, y_pred)
        
        # Multi-column case (if applicable)
        aucs = []
        for col in range(y_true.shape[1]):
            if len(np.unique(y_true[:, col])) > 1:  # Check if both classes present
                auc = roc_auc_score(y_true[:, col], y_pred[:, col])
                aucs.append(auc)
        
        return np.mean(aucs) if aucs else 0.0
    
    @staticmethod
    def validate_submission(submission_df: pd.DataFrame, 
                          expected_rows: Optional[int] = None) -> Dict[str, bool]:
        """Validate submission format.
        
        Args:
            submission_df: Submission DataFrame
            expected_rows: Expected number of rows
            
        Returns:
            Dictionary with validation results
        """
        checks = {
            'has_required_columns': all(col in submission_df.columns 
                                      for col in ['row_id', 'rule_violation']),
            'no_missing_values': not submission_df.isnull().any().any(),
            'valid_probabilities': all(
                (submission_df['rule_violation'] >= 0) & 
                (submission_df['rule_violation'] <= 1)
            ),
            'correct_row_count': True if expected_rows is None 
                               else len(submission_df) == expected_rows,
            'unique_row_ids': submission_df['row_id'].nunique() == len(submission_df)
        }
        
        return checks

class KaggleSubmissionPipeline:
    """Complete pipeline for Kaggle submission."""
    
    def __init__(self, config: KaggleConfig):
        self.config = config
        self.monitor = PerformanceMonitor(config)
        self.evaluator = KaggleEvaluator()
        self.data_pipeline = RedditDataPipeline(config.model_name)
        
        # Model configuration
        self.model_config = ModelConfig(
            model_name=config.model_name,
            max_length=config.max_length,
            batch_size=config.batch_size,
            num_epochs=config.num_epochs,
            learning_rate=config.learning_rate
        )
        
        logger.info(f"Initialized Kaggle submission pipeline")
        logger.info(f"Max runtime: {config.max_runtime_hours} hours")
        logger.info(f"Memory limit: {config.memory_limit_gb} GB")
    
    def load_competition_data(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load competition data.
        
        Args:
            train_path: Path to training data
            test_path: Path to test data
            
        Returns:
            Training and test DataFrames
        """
        logger.info("Loading competition data...")
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        logger.info(f"Training data: {len(train_df)} samples")
        logger.info(f"Test data: {len(test_df)} samples")
        
        return train_df, test_df
    
    def prepare_data(self, train_df: pd.DataFrame, 
                    text_column: str = 'comment_text',
                    target_column: str = 'rule_violation') -> Tuple[np.ndarray, List[str], List[int]]:
        """Prepare data for training.
        
        Args:
            train_df: Training DataFrame
            text_column: Name of text column
            target_column: Name of target column
            
        Returns:
            Features, texts, and labels
        """
        logger.info("Preparing training data...")
        
        # Process comments
        processed_df = self.data_pipeline.process_comments(
            train_df, text_column, target_column
        )
        
        # Create features
        features, feature_names = self.data_pipeline.create_features(processed_df)
        
        # Extract texts and labels
        texts = processed_df[text_column].tolist()
        labels = processed_df[target_column].tolist()
        
        logger.info(f"Feature matrix shape: {features.shape}")
        logger.info(f"Number of positive samples: {sum(labels)}")
        
        return features, texts, labels
    
    def train_with_cross_validation(self, features: np.ndarray, 
                                   texts: List[str], labels: List[int]) -> Dict[str, float]:
        """Train model with cross-validation.
        
        Args:
            features: Feature matrix
            texts: List of text strings
            labels: List of labels
            
        Returns:
            Cross-validation results
        """
        logger.info(f"Starting {self.config.n_folds}-fold cross-validation...")
        
        skf = StratifiedKFold(n_splits=self.config.n_folds, shuffle=True, random_state=42)
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
            if not self.monitor.check_time_limit():
                logger.warning("Approaching time limit, stopping cross-validation")
                break
            
            logger.info(f"Training fold {fold + 1}/{self.config.n_folds}")
            
            # Split data
            train_texts = [texts[i] for i in train_idx]
            val_texts = [texts[i] for i in val_idx]
            train_labels = [labels[i] for i in train_idx]
            val_labels = [labels[i] for i in val_idx]
            train_features = features[train_idx]
            val_features = features[val_idx]
            
            # Train ensemble
            ensemble = EnsembleClassifier(self.model_config)
            ensemble.weights = [
                self.config.transformer_weight,
                self.config.rf_weight,
                self.config.lr_weight
            ]
            
            try:
                ensemble.train(train_texts, train_labels, train_features)
                
                # Predict and evaluate
                val_preds = ensemble.predict(val_texts, val_features)
                fold_auc = self.evaluator.column_averaged_auc(np.array(val_labels), val_preds)
                fold_scores.append(fold_auc)
                
                logger.info(f"Fold {fold + 1} AUC: {fold_auc:.4f}")
                
                # Memory cleanup
                self.monitor.cleanup_memory()
                
            except Exception as e:
                logger.error(f"Error in fold {fold + 1}: {str(e)}")
                continue
        
        cv_results = {
            'mean_auc': np.mean(fold_scores) if fold_scores else 0.0,
            'std_auc': np.std(fold_scores) if fold_scores else 0.0,
            'fold_scores': fold_scores,
            'completed_folds': len(fold_scores)
        }
        
        logger.info(f"Cross-validation completed: {cv_results['mean_auc']:.4f} ± {cv_results['std_auc']:.4f}")
        
        return cv_results
    
    def train_final_model(self, features: np.ndarray, 
                         texts: List[str], labels: List[int]) -> EnsembleClassifier:
        """Train final model on all data.
        
        Args:
            features: Feature matrix
            texts: List of text strings
            labels: List of labels
            
        Returns:
            Trained ensemble classifier
        """
        logger.info("Training final model on all data...")
        
        ensemble = EnsembleClassifier(self.model_config)
        ensemble.weights = [
            self.config.transformer_weight,
            self.config.rf_weight,
            self.config.lr_weight
        ]
        
        ensemble.train(texts, labels, features)
        
        # Save model
        model_path = Path(self.config.model_save_path)
        model_path.mkdir(parents=True, exist_ok=True)
        
        ensemble.transformer_trainer.save_model(model_path / 'transformer')
        
        # Save traditional ML models
        import pickle
        with open(model_path / 'rf_model.pkl', 'wb') as f:
            pickle.dump(ensemble.rf_model, f)
        
        with open(model_path / 'lr_model.pkl', 'wb') as f:
            pickle.dump(ensemble.lr_model, f)
        
        logger.info(f"Final model saved to {model_path}")
        
        return ensemble
    
    def generate_predictions(self, ensemble: EnsembleClassifier,
                           test_df: pd.DataFrame,
                           text_column: str = 'comment_text') -> np.ndarray:
        """Generate predictions for test data.
        
        Args:
            ensemble: Trained ensemble classifier
            test_df: Test DataFrame
            text_column: Name of text column
            
        Returns:
            Array of predictions
        """
        logger.info("Generating predictions for test data...")
        
        # Process test data
        processed_test = self.data_pipeline.process_comments(
            test_df, text_column, target_column=None
        )
        
        # Create features
        test_features, _ = self.data_pipeline.create_features(processed_test)
        
        # Extract texts
        test_texts = processed_test[text_column].tolist()
        
        # Generate predictions
        predictions = ensemble.predict(test_texts, test_features)
        
        logger.info(f"Generated {len(predictions)} predictions")
        
        return predictions
    
    def create_submission(self, test_df: pd.DataFrame, predictions: np.ndarray,
                         row_id_column: str = 'row_id') -> pd.DataFrame:
        """Create submission DataFrame.
        
        Args:
            test_df: Test DataFrame
            predictions: Array of predictions
            row_id_column: Name of row ID column
            
        Returns:
            Submission DataFrame
        """
        submission_df = pd.DataFrame({
            'row_id': test_df[row_id_column],
            'rule_violation': predictions
        })
        
        # Validate submission
        validation_results = self.evaluator.validate_submission(
            submission_df, len(test_df)
        )
        
        logger.info("Submission validation:")
        for check, passed in validation_results.items():
            status = "✓" if passed else "✗"
            logger.info(f"  {status} {check}")
        
        if not all(validation_results.values()):
            raise ValueError("Submission validation failed")
        
        return submission_df
    
    def run_complete_pipeline(self, train_path: str, test_path: str,
                             text_column: str = 'comment_text',
                             target_column: str = 'rule_violation',
                             row_id_column: str = 'row_id') -> Dict[str, any]:
        """Run the complete Kaggle submission pipeline.
        
        Args:
            train_path: Path to training data
            test_path: Path to test data
            text_column: Name of text column
            target_column: Name of target column
            row_id_column: Name of row ID column
            
        Returns:
            Pipeline results
        """
        logger.info("Starting complete Kaggle submission pipeline...")
        
        try:
            # Load data
            train_df, test_df = self.load_competition_data(train_path, test_path)
            
            # Prepare training data
            features, texts, labels = self.prepare_data(
                train_df, text_column, target_column
            )
            
            # Cross-validation
            cv_results = self.train_with_cross_validation(features, texts, labels)
            
            # Train final model
            ensemble = self.train_final_model(features, texts, labels)
            
            # Generate predictions
            predictions = self.generate_predictions(
                ensemble, test_df, text_column
            )
            
            # Create submission
            submission_df = self.create_submission(
                test_df, predictions, row_id_column
            )
            
            # Save submission
            submission_df.to_csv(self.config.submission_file, index=False)
            logger.info(f"Submission saved to {self.config.submission_file}")
            
            # Final performance check
            final_memory = self.monitor.check_memory_usage()
            remaining_time = self.monitor.get_remaining_time()
            
            results = {
                'cv_results': cv_results,
                'submission_file': self.config.submission_file,
                'final_memory_usage': final_memory,
                'remaining_time_hours': remaining_time,
                'total_predictions': len(predictions)
            }
            
            logger.info("Pipeline completed successfully!")
            logger.info(f"Cross-validation AUC: {cv_results['mean_auc']:.4f}")
            logger.info(f"Remaining time: {remaining_time:.2f} hours")
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

def create_sample_data(output_dir: str = 'data/sample'):
    """Create sample data for testing the pipeline."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sample training data
    train_data = {
        'row_id': range(1000),
        'comment_text': [
            "This is a great post, thanks for sharing!",
            "You're an idiot if you believe this nonsense.",
            "Check out my website at www.example.com for more info!",
            "This is completely unrelated but I love pizza.",
            "What a load of crap, this is fake news.",
            "lol",
            "I completely agree with your analysis.",
            "This violates community guidelines.",
            "Spam spam spam buy now!",
            "Great discussion, very informative."
        ] * 100,
        'rule_violation': [0, 1, 1, 1, 1, 1, 0, 1, 1, 0] * 100
    }
    
    # Sample test data
    test_data = {
        'row_id': range(1000, 1500),
        'comment_text': [
            "Another great comment here.",
            "This is stupid and you're wrong.",
            "Visit my site for deals!",
            "Random off-topic comment.",
            "Fake news everywhere."
        ] * 100
    }
    
    # Save data
    pd.DataFrame(train_data).to_csv(output_path / 'train.csv', index=False)
    pd.DataFrame(test_data).to_csv(output_path / 'test.csv', index=False)
    
    logger.info(f"Sample data created in {output_path}")

def main():
    """Example usage of the Kaggle submission pipeline."""
    # Create sample data
    create_sample_data()
    
    # Configuration
    config = KaggleConfig(
        max_runtime_hours=1,  # Short for demo
        batch_size=4,         # Small for demo
        num_epochs=1,         # Quick training
        n_folds=2             # Fewer folds for demo
    )
    
    # Initialize pipeline
    pipeline = KaggleSubmissionPipeline(config)
    
    # Run pipeline
    results = pipeline.run_complete_pipeline(
        train_path='data/sample/train.csv',
        test_path='data/sample/test.csv'
    )
    
    print("Pipeline Results:")
    print(f"Cross-validation AUC: {results['cv_results']['mean_auc']:.4f}")
    print(f"Submission file: {results['submission_file']}")
    print(f"Total predictions: {results['total_predictions']}")

if __name__ == '__main__':
    main()