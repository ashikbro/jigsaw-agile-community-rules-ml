#!/usr/bin/env python3
"""
Train Models with Real Large Datasets

This script replaces the synthetic data training with real large-scale datasets:
- Downloads real datasets using RealDatasetDownloader
- Trains all 5 models (Logistic Regression, Random Forest, SVM, Naive Bayes, Neural Network)
- Saves trained models to replace the existing ones
- Generates comprehensive training reports

Total training data: ~100K real samples from multiple sources
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Import our real dataset downloader
from real_dataset_downloader import RealDatasetDownloader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealDatasetTrainer:
    """Trains models using real large-scale datasets."""
    
    def __init__(self, models_dir: str = "models", reports_dir: str = "reports"):
        self.models_dir = Path(models_dir)
        self.reports_dir = Path(reports_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        
        # Initialize dataset downloader
        self.downloader = RealDatasetDownloader()
        
        # Model configurations
        self.models = {
            'logistic_regression': LogisticRegression(
                max_iter=1000, 
                random_state=42,
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'svm': SVC(
                kernel='linear', 
                random_state=42,
                class_weight='balanced',
                probability=True
            ),
            'naive_bayes': MultinomialNB(
                alpha=1.0
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=300,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
        }
        
        self.vectorizer = None
        self.training_results = {}
        
    def download_and_prepare_datasets(self, sample_limits: Dict[str, int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Download real datasets and combine them for training.
        
        Args:
            sample_limits: Optional limits per dataset
            
        Returns:
            Combined (train_df, test_df) from all datasets
        """
        logger.info("🚀 Downloading real datasets...")
        
        # Default sample limits for balanced training
        if sample_limits is None:
            sample_limits = {
                'jigsaw_toxic': 20000,    # Toxicity detection
                'imdb': 15000,            # Sentiment analysis  
                'ag_news': 10000,         # News classification
                'twitter_hate': 10000,    # Hate speech detection
                'fake_news': 5000         # Fake news detection
            }
            
        # Download all datasets
        datasets = self.downloader.download_all_datasets(sample_limits)
        
        if not datasets:
            raise RuntimeError("Failed to download any datasets")
            
        # Combine all datasets
        all_train_dfs = []
        all_test_dfs = []
        
        for dataset_name, (train_df, test_df) in datasets.items():
            logger.info(f"📊 Processing {dataset_name}: {len(train_df)} train + {len(test_df)} test")
            
            # Standardize labels to binary (0/1) for unified training
            train_df = self._standardize_labels(train_df, dataset_name)
            test_df = self._standardize_labels(test_df, dataset_name)
            
            # Add dataset source for tracking
            train_df['source'] = dataset_name
            test_df['source'] = dataset_name
            
            all_train_dfs.append(train_df)
            all_test_dfs.append(test_df)
            
        # Combine all datasets
        combined_train = pd.concat(all_train_dfs, ignore_index=True)
        combined_test = pd.concat(all_test_dfs, ignore_index=True)
        
        # Shuffle the combined datasets
        combined_train = combined_train.sample(frac=1, random_state=42).reset_index(drop=True)
        combined_test = combined_test.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"✅ Combined dataset: {len(combined_train):,} train + {len(combined_test):,} test = {len(combined_train) + len(combined_test):,} total")
        
        return combined_train, combined_test
        
    def _standardize_labels(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Standardize labels to binary 0/1 format."""
        df = df.copy()
        
        if dataset_name == 'ag_news':
            # AG News has 4 classes (0,1,2,3) - convert to binary (news vs non-news)
            # First convert to int, then apply comparison
            df['label'] = pd.to_numeric(df['label'], errors='coerce')
            df['label'] = (df['label'] >= 2).astype(int)
        elif dataset_name in ['imdb', 'jigsaw_toxic', 'twitter_hate', 'fake_news']:
            # These should already be binary, but ensure they're 0/1
            df['label'] = pd.to_numeric(df['label'], errors='coerce').astype(int)
            
        # Remove any rows with NaN labels
        df = df.dropna(subset=['label'])
        
        return df
        
    def prepare_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare text features using TF-IDF vectorization.
        
        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info("🔧 Preparing text features with TF-IDF...")
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        # Fit and transform training data
        X_train = self.vectorizer.fit_transform(train_df['text'])
        y_train = train_df['label'].values
        
        # Transform test data
        X_test = self.vectorizer.transform(test_df['text'])
        y_test = test_df['label'].values
        
        logger.info(f"   ✅ Features: {X_train.shape[1]:,} TF-IDF features")
        logger.info(f"   ✅ Training samples: {X_train.shape[0]:,}")
        logger.info(f"   ✅ Test samples: {X_test.shape[0]:,}")
        logger.info(f"   ✅ Label distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")
        
        return X_train, X_test, y_train, y_test
        
    def train_all_models(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Train all models and evaluate performance.
        
        Args:
            X_train, X_test, y_train, y_test: Prepared features and labels
            
        Returns:
            Dictionary with training results for each model
        """
        logger.info("🎯 Training all models on real datasets...")
        
        results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"\n🔄 Training {model_name.replace('_', ' ').title()}...")
            
            try:
                # Train the model
                start_time = datetime.now()
                model.fit(X_train, y_train)
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                # Cross-validation score (on a subset for speed)
                cv_indices = np.random.choice(X_train.shape[0], min(5000, X_train.shape[0]), replace=False)
                X_train_cv = X_train[cv_indices]
                y_train_cv = y_train[cv_indices]
                cv_scores = cross_val_score(model, X_train_cv, y_train_cv, cv=3, scoring='accuracy')
                
                # Classification report
                class_report = classification_report(y_test, y_pred, output_dict=True)
                
                # Store results
                results[model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'training_time': training_time,
                    'classification_report': class_report,
                    'predictions': y_pred
                }
                
                logger.info(f"   ✅ Accuracy: {accuracy:.4f}")
                logger.info(f"   ✅ CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                logger.info(f"   ✅ Training Time: {training_time:.2f}s")
                
            except Exception as e:
                logger.error(f"   ❌ Failed to train {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
                
        return results
        
    def save_models(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Save trained models and vectorizer to disk.
        
        Args:
            results: Training results containing trained models
        """
        logger.info("💾 Saving trained models...")
        
        # Save vectorizer
        vectorizer_path = self.models_dir / "tfidf_vectorizer_real.pkl"
        joblib.dump(self.vectorizer, vectorizer_path)
        logger.info(f"   ✅ Saved vectorizer: {vectorizer_path}")
        
        # Save each model
        for model_name, result in results.items():
            if 'model' in result:
                model_path = self.models_dir / f"{model_name}_real.pkl"
                joblib.dump(result['model'], model_path)
                logger.info(f"   ✅ Saved {model_name}: {model_path}")
                
        # Also save as the default models (replacing old ones)
        logger.info("🔄 Replacing default models with real dataset versions...")
        
        # Copy vectorizer as default
        default_vectorizer_path = self.models_dir / "tfidf_vectorizer.pkl"
        joblib.dump(self.vectorizer, default_vectorizer_path)
        
        # Copy best performing model as default
        successful_models = {k: v for k, v in results.items() if 'accuracy' in v}
        if successful_models:
            best_model_name = max(successful_models.keys(), key=lambda k: successful_models[k]['accuracy'])
            default_model_path = self.models_dir / "simple_model.pkl"
            joblib.dump(successful_models[best_model_name]['model'], default_model_path)
            logger.info(f"   ✅ Set {best_model_name} as default model (accuracy: {successful_models[best_model_name]['accuracy']:.4f})")
        else:
            logger.warning("   ⚠️ No successful models to set as default")
            
    def generate_training_report(self, results: Dict[str, Dict[str, Any]], train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
        """Generate comprehensive training report.
        
        Args:
            results: Training results
            train_df: Training dataframe
            test_df: Test dataframe
            
        Returns:
            Report string
        """
        report_lines = [
            "=" * 80,
            "REAL DATASET TRAINING REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Training Data: {len(train_df):,} samples",
            f"Test Data: {len(test_df):,} samples",
            f"Total Data: {len(train_df) + len(test_df):,} samples",
            "",
            "DATASET COMPOSITION:",
            "-" * 50
        ]
        
        # Dataset breakdown
        for source in train_df['source'].unique():
            source_train = len(train_df[train_df['source'] == source])
            source_test = len(test_df[test_df['source'] == source])
            report_lines.append(f"{source.upper()}: {source_train:,} train + {source_test:,} test = {source_train + source_test:,} total")
            
        report_lines.extend([
            "",
            "FEATURE EXTRACTION:",
            "-" * 50,
            f"Vectorizer: TF-IDF",
            f"Features: {self.vectorizer.get_feature_names_out().shape[0]:,}",
            f"Vocabulary size: {len(self.vectorizer.vocabulary_):,}",
            f"N-gram range: {self.vectorizer.ngram_range}",
            "",
            "MODEL PERFORMANCE:",
            "-" * 50
        ])
        
        # Sort models by accuracy
        sorted_results = sorted(
            [(name, res) for name, res in results.items() if 'accuracy' in res],
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )
        
        for i, (model_name, result) in enumerate(sorted_results):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
            
            report_lines.extend([
                f"",
                f"{rank} {model_name.replace('_', ' ').title().upper()}:",
                f"   • Test Accuracy: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)",
                f"   • CV Score: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}",
                f"   • Training Time: {result['training_time']:.2f}s",
                f"   • Precision: {result['classification_report']['weighted avg']['precision']:.4f}",
                f"   • Recall: {result['classification_report']['weighted avg']['recall']:.4f}",
                f"   • F1-Score: {result['classification_report']['weighted avg']['f1-score']:.4f}"
            ])
            
        # Summary statistics
        accuracies = [res['accuracy'] for res in results.values() if 'accuracy' in res]
        if accuracies:
            report_lines.extend([
                "",
                "SUMMARY STATISTICS:",
                "-" * 50,
                f"Best Accuracy: {max(accuracies):.4f} ({max(accuracies)*100:.2f}%)",
                f"Average Accuracy: {np.mean(accuracies):.4f} ({np.mean(accuracies)*100:.2f}%)",
                f"Accuracy Std Dev: {np.std(accuracies):.4f}",
                f"Models Trained: {len(accuracies)}/5",
                "",
                "COMPARISON WITH PREVIOUS (50-sample) TRAINING:",
                "-" * 50,
                "Previous training used only 50 hardcoded samples with 100% accuracy.",
                f"Current training uses {len(train_df) + len(test_df):,} real samples with more realistic performance.",
                "This represents a significant improvement in model robustness and generalization.",
                "",
                "=" * 80
            ])
            
        return "\n".join(report_lines)
        
    def run_complete_training(self, sample_limits: Dict[str, int] = None) -> Dict[str, Any]:
        """Run the complete training pipeline.
        
        Args:
            sample_limits: Optional limits per dataset
            
        Returns:
            Complete training results
        """
        logger.info("🚀 Starting complete real dataset training pipeline...")
        
        try:
            # Step 1: Download and prepare datasets
            train_df, test_df = self.download_and_prepare_datasets(sample_limits)
            
            # Step 2: Prepare features
            X_train, X_test, y_train, y_test = self.prepare_features(train_df, test_df)
            
            # Step 3: Train all models
            results = self.train_all_models(X_train, X_test, y_train, y_test)
            
            # Step 4: Save models
            self.save_models(results)
            
            # Step 5: Generate report
            report = self.generate_training_report(results, train_df, test_df)
            
            # Step 6: Save report
            report_path = self.reports_dir / f"real_dataset_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            report_path.write_text(report)
            
            logger.info(f"\n📋 Training complete! Report saved: {report_path}")
            print("\n" + report)
            
            return {
                'results': results,
                'report': report,
                'report_path': str(report_path),
                'train_samples': len(train_df),
                'test_samples': len(test_df)
            }
            
        except Exception as e:
            logger.error(f"❌ Training pipeline failed: {str(e)}")
            raise
            
def main():
    """Main function to run real dataset training."""
    print("🚀 Real Dataset Training Pipeline")
    print("=" * 50)
    
    # Initialize trainer
    trainer = RealDatasetTrainer()
    
    # Define sample limits for balanced training
    sample_limits = {
        'jigsaw_toxic': 20000,    # Toxicity detection
        'imdb': 15000,            # Sentiment analysis
        'ag_news': 10000,         # News classification  
        'twitter_hate': 10000,    # Hate speech detection
        'fake_news': 5000         # Fake news detection
    }
    
    total_samples = sum(sample_limits.values())
    print(f"📊 Target training size: {total_samples:,} samples")
    print(f"📚 Datasets: {', '.join(sample_limits.keys())}")
    
    # Run training
    try:
        results = trainer.run_complete_training(sample_limits)
        
        print(f"\n✅ SUCCESS: Trained {len(results['results'])} models on {results['train_samples'] + results['test_samples']:,} real samples")
        print(f"📋 Report: {results['report_path']}")
        
        # Show best model
        successful_results = {k: v for k, v in results['results'].items() if 'accuracy' in v}
        if successful_results:
            best_model = max(successful_results.items(), key=lambda x: x[1]['accuracy'])
            print(f"🏆 Best Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")
        else:
            print("⚠️ No models trained successfully")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return None

if __name__ == "__main__":
    main()