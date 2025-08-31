"""
Improved Model Training for Jigsaw Competition

This script trains a model using sentence-transformers and scikit-learn
for the Jigsaw Agile Community Rules Classification task with enhanced features.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional
from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score,
    cross_validate
)
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, make_scorer, precision_recall_curve, auc
)
from sentence_transformers import SentenceTransformer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import random
from typing import List, Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Set up paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODEL_DIR = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'outputs'

# Create directories if they don't exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Configuration
CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'n_splits': 5,  # for k-fold cross-validation
    'batch_size': 32,
    'models': {
        'all-mpnet-base-v2': {
            'name': 'all-mpnet-base-v2',
            'classifier': 'logistic',  # 'logistic' or 'random_forest'
            'params': {
                'logistic': {'max_iter': 1000, 'random_state': 42},
                'random_forest': {'n_estimators': 100, 'random_state': 42}
            }
        },
        'roberta-base': {
            'name': 'roberta-base',
            'classifier': 'logistic',
            'params': {
                'logistic': {'max_iter': 1000, 'random_state': 42}
            }
        },
        'distilbert-base-uncased': {
            'name': 'distilbert-base-uncased',
            'classifier': 'logistic',
            'params': {
                'logistic': {'max_iter': 1000, 'random_state': 42}
            }
        }
    },
    'selected_model': 'all-mpnet-base-v2',
    'augmentation': {
        'enabled': True,
        'methods': ['synonym_replacement', 'random_deletion', 'random_swap'],
        'augment_ratio': 0.1  # 10% of the data will be augmented
    }
}

# Set random seed for reproducibility
np.random.seed(CONFIG['random_state'])
random.seed(CONFIG['random_state'])

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
    log_file = log_dir / f'training_{timestamp}.log'
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

class DataLoader:
    """Handles loading and preprocessing of data."""
    
    def __init__(self, data_dir: Path):
        """Initialize with data directory."""
        self.data_dir = data_dir
        self.processed_dir = data_dir / 'processed'
        self.processed_dir.mkdir(exist_ok=True)
    
    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the data."""
        try:
            # Try to load processed data first
            processed_path = self.processed_dir / 'train_processed.csv'
            if processed_path.exists():
                logger.info(f"Loading processed data from {processed_path}")
                df = pd.read_csv(processed_path)
                return df
            
            # Load raw data if processed not available
            logger.info("Processed data not found. Loading raw data...")
            train_path = self.data_dir / 'train.csv'
            
            if not train_path.exists():
                raise FileNotFoundError(f"Training data not found at {train_path}")
            
            df = pd.read_csv(train_path)
            
            # Basic preprocessing
            if 'text' not in df.columns and 'body' in df.columns:
                df['text'] = df['body']
            
            # Save processed data
            df.to_csv(processed_path, index=False)
            logger.info(f"Saved processed data to {processed_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

class TextAugmenter:
    """Handles text data augmentation."""
    
    @staticmethod
    def synonym_replacement(text: str, n: int = 1) -> str:
        """Replace n words in the text with their synonyms."""
        # Simple implementation - in practice, use a library like nlpaug
        words = text.split()
        if len(words) <= n:
            return text
            
        indices = random.sample(range(len(words)), n)
        for idx in indices:
            # In a real implementation, replace with actual synonym lookup
            pass
            
        return ' '.join(words)
    
    @staticmethod
    def random_deletion(text: str, p: float = 0.1) -> str:
        """Randomly delete words with probability p."""
        words = text.split()
        if len(words) <= 1:
            return text
            
        new_words = [word for word in words if random.random() > p]
        return ' '.join(new_words) if new_words else words[0]
    
    @staticmethod
    def random_swap(text: str, n: int = 1) -> str:
        """Randomly swap n pairs of words in the text."""
        words = text.split()
        if len(words) < 2:
            return text
            
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
            
        return ' '.join(words)
    
    @classmethod
    def augment_text(cls, text: str, method: str, **kwargs) -> str:
        """Apply specified augmentation method to text."""
        if method == 'synonym_replacement':
            return cls.synonym_replacement(text, **kwargs)
        elif method == 'random_deletion':
            return cls.random_deletion(text, **kwargs)
        elif method == 'random_swap':
            return cls.random_swap(text, **kwargs)
        else:
            raise ValueError(f"Unknown augmentation method: {method}")
    
    @classmethod
    def augment_dataframe(
        cls,
        df: pd.DataFrame,
        text_col: str = 'text',
        label_col: str = 'label',
        methods: List[str] = None,
        augment_ratio: float = 0.1,
        **kwargs
    ) -> pd.DataFrame:
        """Augment the dataset with specified methods."""
        if methods is None:
            methods = ['synonym_replacement', 'random_deletion', 'random_swap']
            
        n_augment = int(len(df) * augment_ratio)
        if n_augment < 1:
            return df
            
        logger.info(f"Augmenting {n_augment} samples...")
        
        augmented_samples = []
        sampled_indices = np.random.choice(df.index, n_augment, replace=False)
        
        for idx in tqdm(sampled_indices, desc="Augmenting data"):
            text = df.at[idx, text_col]
            label = df.at[idx, label_col]
            
            # Apply a random augmentation method
            method = random.choice(methods)
            try:
                augmented_text = cls.augment_text(text, method, **kwargs)
                augmented_samples.append({
                    text_col: augmented_text,
                    label_col: label,
                    'is_augmented': True,
                    'augmentation_method': method
                })
            except Exception as e:
                logger.warning(f"Error augmenting text: {e}")
        
        # Combine original and augmented data
        augmented_df = pd.DataFrame(augmented_samples)
        df['is_augmented'] = False
        df['augmentation_method'] = None
        
        return pd.concat([df, augmented_df], ignore_index=True)

class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self.config = config
        self.model_name = config['selected_model']
        self.model_config = config['models'][self.model_name]
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self.embedding_model = None
        self.classifier = None
        self.label_encoder = LabelEncoder()
        
        # Results storage
        self.results = {
            'model_name': self.model_name,
            'training_time': None,
            'metrics': {}
        }
    
    def _init_embedding_model(self):
        """Initialize the embedding model."""
        try:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            self.embedding_model = SentenceTransformer(self.model_name)
            self.logger.info("Embedding model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    def _init_classifier(self):
        """Initialize the classifier."""
        classifier_type = self.model_config['classifier']
        params = self.model_config['params'].get(classifier_type, {})
        
        if classifier_type == 'logistic':
            self.classifier = LogisticRegression(**params)
        elif classifier_type == 'random_forest':
            self.classifier = RandomForestClassifier(**params)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        self.logger.info(f"Initialized {classifier_type} classifier with params: {params}")
    
    def _encode_labels(self, y):
        """Encode string labels to integers."""
        return self.label_encoder.fit_transform(y)
    
    def _decode_labels(self, y_encoded):
        """Decode integer labels back to strings."""
        return self.label_encoder.inverse_transform(y_encoded)
    
    def _get_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        self.logger.info(f"Generating embeddings for {len(texts)} texts...")
        try:
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            self.logger.info(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def _evaluate_model(self, y_true, y_pred, y_proba=None) -> Dict[str, float]:
        """Evaluate model performance and return metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        if y_proba is not None and len(np.unique(y_true)) > 1:  # Check if we have multiple classes
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
            except Exception as e:
                self.logger.warning(f"Could not compute ROC AUC: {str(e)}")
        
        # Log metrics
        self.logger.info("\n=== Classification Report ===")
        self.logger.info(classification_report(y_true, y_pred, zero_division=0))
        
        self.logger.info("\n=== Model Performance ===")
        for name, value in metrics.items():
            self.logger.info(f"{name}: {value:.4f}")
        
        return metrics
    
    def _plot_confusion_matrix(self, y_true, y_pred, output_path: Path):
        """Plot and save confusion matrix."""
        try:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=self.label_encoder.classes_,
                yticklabels=self.label_encoder.classes_
            )
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            self.logger.info(f"Confusion matrix saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error plotting confusion matrix: {str(e)}")
    
    def _save_model(self, output_dir: Path):
        """Save the trained model and related artifacts."""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save classifier
            joblib.dump(self.classifier, output_dir / 'classifier.joblib')
            
            # Save label encoder
            joblib.dump(self.label_encoder, output_dir / 'label_encoder.joblib')
            
            # Save model info
            model_info = {
                'model_name': self.model_name,
                'classifier_type': self.model_config['classifier'],
                'classifier_params': self.model_config['params'].get(self.model_config['classifier'], {}),
                'training_date': datetime.now().isoformat(),
                'label_classes': self.label_encoder.classes_.tolist()
            }
            
            with open(output_dir / 'model_info.json', 'w') as f:
                json.dump(model_info, f, indent=2)
            
            self.logger.info(f"Model saved to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    def _cross_validate(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Dict[str, float]:
        """Perform k-fold cross-validation."""
        self.logger.info(f"Performing {n_splits}-fold cross-validation...")
        
        # Define metrics to compute
        scoring = {
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score, average='weighted', zero_division=0),
            'recall': make_scorer(recall_score, average='weighted', zero_division=0),
            'f1': make_scorer(f1_score, average='weighted', zero_division=0),
            'roc_auc': make_scorer(roc_auc_score, multi_class='ovr', needs_proba=True)
        }
        
        # Perform cross-validation
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=self.config['random_state'])
        cv_results = cross_validate(
            self.classifier,
            X,
            y,
            cv=kfold,
            scoring=scoring,
            return_train_score=True,
            return_estimator=True,
            n_jobs=-1
        )
        
        # Calculate mean and std of metrics
        metrics = {}
        for metric in scoring.keys():
            train_scores = cv_results[f'train_{metric}']
            val_scores = cv_results[f'test_{metric}']
            
            metrics[f'cv_train_{metric}_mean'] = np.mean(train_scores)
            metrics[f'cv_train_{metric}_std'] = np.std(train_scores)
            metrics[f'cv_val_{metric}_mean'] = np.mean(val_scores)
            metrics[f'cv_val_{metric}_std'] = np.std(val_scores)
            
            self.logger.info(f"CV {metric} - Train: {metrics[f'cv_train_{metric}_mean']:.4f} (±{metrics[f'cv_train_{metric}_std']:.4f})")
            self.logger.info(f"CV {metric} - Val: {metrics[f'cv_val_{metric}_mean']:.4f} (±{metrics[f'cv_val_{metric}_std']:.4f})")
        
        return metrics
    
    def train(self, X_train: List[str], y_train: List[str], X_val: List[str] = None, y_val: List[str] = None):
        """Train the model with k-fold cross-validation and optional validation."""
        start_time = datetime.now()
        
        try:
            # Initialize models
            self._init_embedding_model()
            self._init_classifier()
            
            # Encode labels
            y_train_encoded = self._encode_labels(y_train)
            
            # Generate embeddings
            X_train_emb = self._get_embeddings(X_train)
            
            # Perform k-fold cross-validation
            cv_metrics = self._cross_validate(X_train_emb, y_train_encoded, self.config['n_splits'])
            self.results['metrics'].update(cv_metrics)
            
            # Train final model on full training data
            self.logger.info("Training final model on full training set...")
            self.classifier.fit(X_train_emb, y_train_encoded)
            
            # Evaluate on training set
            y_train_pred = self.classifier.predict(X_train_emb)
            y_train_pred_proba = self.classifier.predict_proba(X_train_emb)
            
            train_metrics = self._evaluate_model(
                y_train_encoded, y_train_pred, y_train_pred_proba
            )
            self.results['metrics']['train'] = train_metrics
            
            # Evaluate on validation set if provided
            if X_val is not None and y_val is not None:
                X_val_emb = self._get_embeddings(X_val)
                y_val_encoded = self._encode_labels(y_val)
                y_val_pred = self.classifier.predict(X_val_emb)
                y_val_pred_proba = self.classifier.predict_proba(X_val_emb)
                
                val_metrics = self._evaluate_model(
                    y_val_encoded, y_val_pred, y_val_pred_proba
                )
                self.results['metrics']['val'] = val_metrics
                
                # Plot confusion matrix for validation set
                self._plot_confusion_matrix(
                    y_val_encoded,
                    y_val_pred,
                    OUTPUT_DIR / f'confusion_matrix_{self.model_name.replace("/", "_")}.png'
                )
            
            # Save model
            model_dir = MODEL_DIR / f"{self.model_name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self._save_model(model_dir)
            
            # Record training time
            self.results['training_time'] = str(datetime.now() - start_time)
            
            # Save results
            results_file = OUTPUT_DIR / f'training_results_{self.model_name.replace("/", "_")}.json'
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            self.logger.info(f"Training completed in {self.results['training_time']}")
            self.logger.info(f"Results saved to {results_file}")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}", exc_info=True)
            raise

def main():
    """Main function to run the training pipeline."""
    try:
        logger.info("Starting training pipeline...")
        logger.info(f"Configuration: {json.dumps(CONFIG, indent=2)}")
        
        # Load and prepare data
        data_loader = DataLoader(DATA_DIR)
        df = data_loader.load_data()
        
        # Basic data validation
        required_columns = ['text', 'label']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Apply text augmentation if enabled
        if CONFIG['augmentation']['enabled']:
            logger.info("Applying text augmentation...")
            df = TextAugmenter.augment_dataframe(
                df,
                text_col='text',
                label_col='label',
                methods=CONFIG['augmentation']['methods'],
                augment_ratio=CONFIG['augmentation']['augment_ratio']
            )
        
        # Split data
        train_df, val_df = train_test_split(
            df,
            test_size=CONFIG['test_size'],
            random_state=CONFIG['random_state'],
            stratify=df['label']
        )
        
        logger.info(f"Training samples: {len(train_df)}")
        logger.info(f"Validation samples: {len(val_df)}")
        
        # Initialize and train model
        trainer = ModelTrainer(CONFIG)
        results = trainer.train(
            X_train=train_df['text'].tolist(),
            y_train=train_df['label'].tolist(),
            X_val=val_df['text'].tolist(),
            y_val=val_df['label'].tolist()
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Log file: {log_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
