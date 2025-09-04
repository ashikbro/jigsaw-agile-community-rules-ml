#!/usr/bin/env python3
"""
Large-Scale Multi-Dataset Training Script
Trains models on multiple datasets with large sample sizes (10K-50K samples each)
Demonstrates comprehensive preprocessing and feature extraction pipeline
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import logging
import time
from datetime import datetime
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import project modules
from src.dataset_integration import DatasetManager
from src.preprocess import DataPreprocessor
from src.batch_processor import MemoryEfficientBatchProcessor
from src.evaluation_metrics import KaggleEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reports/large_scale_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LargeScaleTrainer:
    """Large-scale multi-dataset trainer with comprehensive preprocessing."""
    
    def __init__(self, sample_sizes: Dict[str, int] = None, output_dir: str = "reports"):
        """
        Initialize the large-scale trainer.
        
        Args:
            sample_sizes: Dictionary mapping dataset names to sample sizes
            output_dir: Directory to save results
        """
        self.sample_sizes = sample_sizes or {
            'jigsaw_toxic': 20000,
            'ag_news': 30000,
            'imdb': 25000,
            'twitter_hate': 15000,
            'fake_news': 20000
        }
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.dataset_manager = DatasetManager()
        self.preprocessor = DataPreprocessor()
        self.batch_processor = MemoryEfficientBatchProcessor()
        self.evaluator = KaggleEvaluator()
        
        # Initialize models
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'svm_linear': SVC(kernel='linear', random_state=42, probability=True),
            'naive_bayes': MultinomialNB()
        }
        
        # Results storage
        self.results = {}
        self.trained_models = {}
        self.vectorizers = {}
        
    def load_and_preprocess_datasets(self) -> Dict[str, Dict[str, Any]]:
        """
        Load and preprocess all datasets with comprehensive pipeline demonstration.
        
        Returns:
            Dictionary containing processed datasets
        """
        logger.info("🚀 Starting large-scale dataset loading and preprocessing...")
        
        processed_datasets = {}
        
        for dataset_name, sample_size in self.sample_sizes.items():
            logger.info(f"\n📊 Processing {dataset_name.upper()} with {sample_size:,} samples...")
            
            try:
                # Step 1: Load raw data
                logger.info("  Step 1: Loading raw data...")
                train_df, test_df = self.dataset_manager.load_dataset(dataset_name, sample_size=sample_size)
                
                # Step 2: Standardize dataset format
                logger.info("  Step 2: Standardizing dataset format...")
                std_train = self.dataset_manager.standardize_dataset(train_df, dataset_name)
                std_test = self.dataset_manager.standardize_dataset(test_df, dataset_name)
                
                # Step 3: Text preprocessing
                logger.info("  Step 3: Applying text preprocessing...")
                
                # Show preprocessing examples
                print(f"\n    📝 Preprocessing Examples for {dataset_name}:")
                sample_texts = std_train['text'].head(3).tolist()
                
                print("    Before preprocessing:")
                for i, text in enumerate(sample_texts):
                    print(f"    {i+1}. {text[:100]}...")
                
                # Apply preprocessing
                processed_train = self.preprocessor.preprocess_dataframe(std_train)
                processed_test = self.preprocessor.preprocess_dataframe(std_test)
                
                print("    After preprocessing:")
                for i, text in enumerate(processed_train['text'].head(3)):
                    print(f"    {i+1}. {text[:100]}...")
                
                # Step 4: Feature extraction with TF-IDF
                logger.info("  Step 4: Extracting TF-IDF features...")
                
                vectorizer = TfidfVectorizer(
                    max_features=10000,  # Increased for large datasets
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=3,  # Increased for large datasets
                    max_df=0.95,
                    sublinear_tf=True  # Better for large datasets
                )
                
                # Fit on training data and transform both sets
                X_train = vectorizer.fit_transform(processed_train['text'])
                X_test = vectorizer.transform(processed_test['text'])
                
                # Encode labels
                label_encoder = LabelEncoder()
                y_train = label_encoder.fit_transform(processed_train['label'])
                y_test = label_encoder.transform(processed_test['label'])
                
                # Step 5: Create validation split
                logger.info("  Step 5: Creating validation split...")
                X_train_split, X_val, y_train_split, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
                )
                
                # Store processed data
                processed_datasets[dataset_name] = {
                    'X_train': X_train_split,
                    'X_val': X_val,
                    'X_test': X_test,
                    'y_train': y_train_split,
                    'y_val': y_val,
                    'y_test': y_test,
                    'vectorizer': vectorizer,
                    'label_encoder': label_encoder,
                    'feature_count': X_train.shape[1],
                    'train_samples': X_train_split.shape[0],
                    'val_samples': X_val.shape[0],
                    'test_samples': X_test.shape[0],
                    'classes': label_encoder.classes_
                }
                
                # Log statistics
                logger.info(f"  ✅ {dataset_name} processed successfully:")
                logger.info(f"     • Features: {X_train.shape[1]:,}")
                logger.info(f"     • Train samples: {X_train_split.shape[0]:,}")
                logger.info(f"     • Validation samples: {X_val.shape[0]:,}")
                logger.info(f"     • Test samples: {X_test.shape[0]:,}")
                logger.info(f"     • Classes: {len(label_encoder.classes_)}")
                logger.info(f"     • Matrix density: {X_train.nnz / (X_train.shape[0] * X_train.shape[1]):.4f}")
                
            except Exception as e:
                logger.error(f"  ❌ Failed to process {dataset_name}: {e}")
                continue
        
        logger.info(f"\n🎯 Dataset preprocessing complete! Processed {len(processed_datasets)} datasets.")
        return processed_datasets
    
    def train_models_on_datasets(self, processed_datasets: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Train models on all processed datasets.
        
        Args:
            processed_datasets: Dictionary of processed datasets
            
        Returns:
            Dictionary containing training results
        """
        logger.info("\n🤖 Starting large-scale model training...")
        
        all_results = {}
        
        for dataset_name, data in processed_datasets.items():
            logger.info(f"\n📈 Training models on {dataset_name.upper()}...")
            
            dataset_results = {}
            
            for model_name, model in self.models.items():
                logger.info(f"  Training {model_name}...")
                
                try:
                    # Train model
                    start_time = time.time()
                    model.fit(data['X_train'], data['y_train'])
                    training_time = time.time() - start_time
                    
                    # Predictions
                    y_val_pred = model.predict(data['X_val'])
                    y_test_pred = model.predict(data['X_test'])
                    
                    # Probabilities (if available)
                    try:
                        y_val_proba = model.predict_proba(data['X_val'])[:, 1] if len(data['classes']) == 2 else None
                        y_test_proba = model.predict_proba(data['X_test'])[:, 1] if len(data['classes']) == 2 else None
                    except:
                        y_val_proba = None
                        y_test_proba = None
                    
                    # Calculate metrics
                    val_accuracy = accuracy_score(data['y_val'], y_val_pred)
                    test_accuracy = accuracy_score(data['y_test'], y_test_pred)
                    
                    val_f1 = f1_score(data['y_val'], y_val_pred, average='weighted')
                    test_f1 = f1_score(data['y_test'], y_test_pred, average='weighted')
                    
                    # AUC for binary classification
                    val_auc = None
                    test_auc = None
                    if len(data['classes']) == 2 and y_val_proba is not None:
                        try:
                            val_auc = roc_auc_score(data['y_val'], y_val_proba)
                            test_auc = roc_auc_score(data['y_test'], y_test_proba)
                        except:
                            pass
                    
                    # Store results
                    dataset_results[model_name] = {
                        'val_accuracy': val_accuracy,
                        'test_accuracy': test_accuracy,
                        'val_f1': val_f1,
                        'test_f1': test_f1,
                        'val_auc': val_auc,
                        'test_auc': test_auc,
                        'training_time': training_time,
                        'train_samples': data['train_samples'],
                        'val_samples': data['val_samples'],
                        'test_samples': data['test_samples'],
                        'features': data['feature_count']
                    }
                    
                    logger.info(f"    ✅ {model_name}: Test Acc={test_accuracy:.4f}, Test F1={test_f1:.4f}, Time={training_time:.1f}s")
                    
                except Exception as e:
                    logger.error(f"    ❌ {model_name} failed: {e}")
                    continue
            
            all_results[dataset_name] = dataset_results
        
        return all_results
    
    def generate_comprehensive_report(self, results: Dict[str, Dict[str, Any]], processed_datasets: Dict[str, Dict[str, Any]]):
        """
        Generate comprehensive evaluation report.
        
        Args:
            results: Training results
            processed_datasets: Processed datasets information
        """
        logger.info("\n📊 Generating comprehensive evaluation report...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"large_scale_training_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("LARGE-SCALE MULTI-DATASET TRAINING REPORT\n")
            f.write("=" * 100 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Dataset Summary
            f.write("DATASET SUMMARY\n")
            f.write("-" * 50 + "\n")
            
            total_samples = 0
            total_features = 0
            
            for dataset_name, data in processed_datasets.items():
                samples = data['train_samples'] + data['val_samples'] + data['test_samples']
                total_samples += samples
                total_features = max(total_features, data['feature_count'])
                
                f.write(f"\n{dataset_name.upper()}:\n")
                f.write(f"  • Total samples: {samples:,}\n")
                f.write(f"  • Train: {data['train_samples']:,}, Val: {data['val_samples']:,}, Test: {data['test_samples']:,}\n")
                f.write(f"  • Features: {data['feature_count']:,}\n")
                f.write(f"  • Classes: {len(data['classes'])} ({', '.join(map(str, data['classes']))})\n")
            
            f.write(f"\nOVERALL STATISTICS:\n")
            f.write(f"  • Total samples across all datasets: {total_samples:,}\n")
            f.write(f"  • Maximum features: {total_features:,}\n")
            f.write(f"  • Datasets processed: {len(processed_datasets)}\n")
            
            # Model Performance Summary
            f.write("\n\nMODEL PERFORMANCE SUMMARY\n")
            f.write("-" * 50 + "\n")
            
            # Create summary table
            summary_data = []
            for dataset_name, dataset_results in results.items():
                for model_name, metrics in dataset_results.items():
                    summary_data.append({
                        'Dataset': dataset_name,
                        'Model': model_name,
                        'Test_Accuracy': metrics['test_accuracy'],
                        'Test_F1': metrics['test_f1'],
                        'Test_AUC': metrics['test_auc'] if metrics['test_auc'] else 'N/A',
                        'Training_Time': metrics['training_time'],
                        'Samples': metrics['train_samples']
                    })
            
            summary_df = pd.DataFrame(summary_data)
            f.write(summary_df.to_string(index=False))
            
            # Best performing models
            f.write("\n\nBEST PERFORMING MODELS\n")
            f.write("-" * 50 + "\n")
            
            for dataset_name in results.keys():
                dataset_results = results[dataset_name]
                best_model = max(dataset_results.keys(), key=lambda x: dataset_results[x]['test_f1'])
                best_f1 = dataset_results[best_model]['test_f1']
                
                f.write(f"\n{dataset_name.upper()}: {best_model} (F1: {best_f1:.4f})\n")
        
        # Save results as CSV
        csv_file = self.output_dir / f"large_scale_results_{timestamp}.csv"
        summary_df.to_csv(csv_file, index=False)
        
        logger.info(f"📄 Report saved to: {report_file}")
        logger.info(f"📊 CSV results saved to: {csv_file}")
        
        return report_file, csv_file
    
    def run_large_scale_training(self):
        """
        Run the complete large-scale training pipeline.
        """
        logger.info("🚀 Starting Large-Scale Multi-Dataset Training Pipeline...")
        
        try:
            # Step 1: Load and preprocess datasets
            processed_datasets = self.load_and_preprocess_datasets()
            
            if not processed_datasets:
                logger.error("❌ No datasets were successfully processed!")
                return
            
            # Step 2: Train models
            results = self.train_models_on_datasets(processed_datasets)
            
            # Step 3: Generate comprehensive report
            report_file, csv_file = self.generate_comprehensive_report(results, processed_datasets)
            
            # Step 4: Summary
            logger.info("\n" + "=" * 80)
            logger.info("🎯 LARGE-SCALE TRAINING COMPLETE!")
            logger.info("=" * 80)
            
            total_samples = sum(data['train_samples'] + data['val_samples'] + data['test_samples'] 
                              for data in processed_datasets.values())
            
            logger.info(f"\n✅ Successfully trained on {total_samples:,} total samples")
            logger.info(f"✅ Processed {len(processed_datasets)} datasets")
            logger.info(f"✅ Trained {len(self.models)} model types")
            logger.info(f"✅ Generated comprehensive reports")
            
            logger.info(f"\n📊 Results available at:")
            logger.info(f"   • Report: {report_file}")
            logger.info(f"   • CSV: {csv_file}")
            
        except Exception as e:
            logger.error(f"❌ Large-scale training failed: {e}")
            raise

def main():
    """Main function to run large-scale training."""
    
    # Configure sample sizes (10K-50K per dataset)
    sample_sizes = {
        'jigsaw_toxic': 20000,
        'ag_news': 30000,
        'imdb': 25000,
        'twitter_hate': 15000,
        'fake_news': 20000
    }
    
    print("🚀 Large-Scale Multi-Dataset Training")
    print("=" * 50)
    print(f"Sample sizes configured:")
    for dataset, size in sample_sizes.items():
        print(f"  • {dataset}: {size:,} samples")
    
    total_samples = sum(sample_sizes.values())
    print(f"\nTotal samples across all datasets: {total_samples:,}")
    print("\nThis will demonstrate:")
    print("  ✓ Large-scale dataset loading (110K total samples)")
    print("  ✓ Comprehensive preprocessing pipeline")
    print("  ✓ Feature extraction with TF-IDF (10K features)")
    print("  ✓ Memory-efficient batch processing")
    print("  ✓ Multi-model training and evaluation")
    print("  ✓ Detailed performance reporting")
    
    # Initialize and run trainer
    trainer = LargeScaleTrainer(sample_sizes=sample_sizes)
    trainer.run_large_scale_training()

if __name__ == "__main__":
    main()