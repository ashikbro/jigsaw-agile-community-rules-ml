#!/usr/bin/env python3
"""
Large Dataset Demonstration Script
Demonstrates training with larger datasets and comprehensive preprocessing pipeline
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import time
from datetime import datetime
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import project modules
from src.dataset_integration import DatasetManager
from src.preprocess import DataPreprocessor
from src.batch_processor import MemoryEfficientBatchProcessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demonstrate_large_dataset_training():
    """Demonstrate training with larger datasets and preprocessing."""
    
    print("=" * 80)
    print("🚀 LARGE DATASET TRAINING DEMONSTRATION")
    print("=" * 80)
    
    # Initialize components
    dataset_manager = DatasetManager()
    preprocessor = DataPreprocessor()
    batch_processor = MemoryEfficientBatchProcessor()
    
    # Configure larger sample sizes
    datasets_config = {
        'imdb': 10000,      # 10K movie reviews
        'ag_news': 15000,   # 15K news articles
        'jigsaw_toxic': 8000  # 8K toxic comments
    }
    
    print(f"\n📊 Dataset Configuration:")
    total_samples = sum(datasets_config.values())
    for dataset, size in datasets_config.items():
        print(f"  • {dataset.upper()}: {size:,} samples")
    print(f"  • TOTAL: {total_samples:,} samples")
    
    print(f"\n🖥️ System Configuration:")
    memory_info = batch_processor._monitor_memory()
    print(f"  • Batch size: {batch_processor.batch_size:,}")
    print(f"  • Workers: {batch_processor.n_jobs}")
    print(f"  • Available memory: {memory_info['available_gb']:.1f} GB")
    
    results = {}
    
    # Process each dataset
    for dataset_name, sample_size in datasets_config.items():
        print(f"\n" + "=" * 60)
        print(f"📈 PROCESSING {dataset_name.upper()} ({sample_size:,} samples)")
        print("=" * 60)
        
        try:
            # Step 1: Load dataset
            print(f"\n🔄 Step 1: Loading {dataset_name} dataset...")
            start_time = time.time()
            train_df, test_df = dataset_manager.load_dataset(dataset_name, sample_size=sample_size)
            load_time = time.time() - start_time
            
            print(f"  ✅ Loaded in {load_time:.2f}s: {len(train_df):,} train, {len(test_df):,} test")
            
            # Step 2: Standardize format
            print(f"\n🔧 Step 2: Standardizing dataset format...")
            std_train = dataset_manager.standardize_dataset(train_df, dataset_name)
            std_test = dataset_manager.standardize_dataset(test_df, dataset_name)
            
            # Show sample data
            print(f"\n📝 Sample Data (before preprocessing):")
            for i, (text, label) in enumerate(zip(std_train['text'].head(2), std_train['label'].head(2))):
                print(f"  {i+1}. [{label}] {text[:100]}...")
            
            # Step 3: Preprocessing
            print(f"\n🧹 Step 3: Applying comprehensive preprocessing...")
            start_time = time.time()
            
            # Apply text cleaning
            processed_train = preprocessor.preprocess_dataframe(std_train)
            processed_test = preprocessor.preprocess_dataframe(std_test)
            
            preprocess_time = time.time() - start_time
            
            print(f"  ✅ Preprocessing completed in {preprocess_time:.2f}s")
            
            # Show processed data
            print(f"\n📝 Sample Data (after preprocessing):")
            for i, (text, label) in enumerate(zip(processed_train['text'].head(2), processed_train['label'].head(2))):
                print(f"  {i+1}. [{label}] {text[:100]}...")
            
            # Text statistics
            text_lengths = processed_train['text'].str.len()
            print(f"\n📊 Text Statistics:")
            print(f"  • Average length: {text_lengths.mean():.1f} characters")
            print(f"  • Max length: {text_lengths.max()} characters")
            print(f"  • Min length: {text_lengths.min()} characters")
            
            # Step 4: Feature extraction
            print(f"\n⚙️ Step 4: TF-IDF Feature Extraction...")
            start_time = time.time()
            
            # Configure TF-IDF for large datasets
            vectorizer = TfidfVectorizer(
                max_features=5000,  # 5K features for faster processing
                stop_words='english',
                ngram_range=(1, 2),
                min_df=3,
                max_df=0.95,
                sublinear_tf=True
            )
            
            # Transform data
            X_train = vectorizer.fit_transform(processed_train['text'])
            X_test = vectorizer.transform(processed_test['text'])
            
            # Encode labels
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(processed_train['label'])
            y_test = label_encoder.transform(processed_test['label'])
            
            feature_time = time.time() - start_time
            
            print(f"  ✅ Feature extraction completed in {feature_time:.2f}s")
            print(f"  • Feature matrix shape: {X_train.shape}")
            print(f"  • Matrix density: {X_train.nnz / (X_train.shape[0] * X_train.shape[1]):.4f}")
            print(f"  • Classes: {len(label_encoder.classes_)} ({', '.join(label_encoder.classes_)})")
            
            # Step 5: Model training
            print(f"\n🤖 Step 5: Training models...")
            
            # Create validation split
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            
            models = {
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            }
            
            dataset_results = {}
            
            for model_name, model in models.items():
                print(f"\n  🔄 Training {model_name}...")
                start_time = time.time()
                
                # Train model
                model.fit(X_train_split, y_train_split)
                train_time = time.time() - start_time
                
                # Evaluate
                val_pred = model.predict(X_val)
                test_pred = model.predict(X_test)
                
                val_acc = accuracy_score(y_val, val_pred)
                test_acc = accuracy_score(y_test, test_pred)
                val_f1 = f1_score(y_val, val_pred, average='weighted')
                test_f1 = f1_score(y_test, test_pred, average='weighted')
                
                dataset_results[model_name] = {
                    'val_accuracy': val_acc,
                    'test_accuracy': test_acc,
                    'val_f1': val_f1,
                    'test_f1': test_f1,
                    'train_time': train_time,
                    'train_samples': X_train_split.shape[0],
                    'val_samples': X_val.shape[0],
                    'test_samples': X_test.shape[0]
                }
                
                print(f"    ✅ {model_name}:")
                print(f"       • Training time: {train_time:.2f}s")
                print(f"       • Validation Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
                print(f"       • Test Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
            
            results[dataset_name] = {
                'dataset_info': {
                    'total_samples': len(train_df) + len(test_df),
                    'train_samples': X_train_split.shape[0],
                    'val_samples': X_val.shape[0],
                    'test_samples': X_test.shape[0],
                    'features': X_train.shape[1],
                    'classes': len(label_encoder.classes_),
                    'load_time': load_time,
                    'preprocess_time': preprocess_time,
                    'feature_time': feature_time
                },
                'model_results': dataset_results
            }
            
        except Exception as e:
            print(f"  ❌ Error processing {dataset_name}: {e}")
            continue
    
    # Generate summary report
    print(f"\n" + "=" * 80)
    print("📊 COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 80)
    
    if results:
        # Overall statistics
        total_samples = sum(r['dataset_info']['total_samples'] for r in results.values())
        total_features = max(r['dataset_info']['features'] for r in results.values())
        
        print(f"\n🎯 Overall Statistics:")
        print(f"  • Total samples processed: {total_samples:,}")
        print(f"  • Maximum features extracted: {total_features:,}")
        print(f"  • Datasets successfully processed: {len(results)}")
        
        # Performance summary
        print(f"\n🏆 Model Performance Summary:")
        print("-" * 60)
        print(f"{'Dataset':<15} {'Model':<20} {'Test Acc':<10} {'Test F1':<10} {'Time(s)':<8}")
        print("-" * 60)
        
        for dataset_name, dataset_results in results.items():
            for model_name, metrics in dataset_results['model_results'].items():
                print(f"{dataset_name:<15} {model_name:<20} {metrics['test_accuracy']:<10.4f} {metrics['test_f1']:<10.4f} {metrics['train_time']:<8.2f}")
        
        # Save results
        os.makedirs('reports', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create detailed report
        report_file = f"reports/large_dataset_demo_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write("LARGE DATASET TRAINING DEMONSTRATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("DATASET PROCESSING SUMMARY:\n")
            f.write("-" * 30 + "\n")
            for dataset_name, dataset_results in results.items():
                info = dataset_results['dataset_info']
                f.write(f"\n{dataset_name.upper()}:\n")
                f.write(f"  • Total samples: {info['total_samples']:,}\n")
                f.write(f"  • Features extracted: {info['features']:,}\n")
                f.write(f"  • Processing times: Load={info['load_time']:.2f}s, Preprocess={info['preprocess_time']:.2f}s, Features={info['feature_time']:.2f}s\n")
            
            f.write("\nMODEL PERFORMANCE:\n")
            f.write("-" * 30 + "\n")
            for dataset_name, dataset_results in results.items():
                f.write(f"\n{dataset_name.upper()}:\n")
                for model_name, metrics in dataset_results['model_results'].items():
                    f.write(f"  {model_name}: Test Acc={metrics['test_accuracy']:.4f}, Test F1={metrics['test_f1']:.4f}\n")
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        print(f"\n✅ DEMONSTRATION COMPLETE!")
        print(f"\n🎯 Key Achievements:")
        print(f"  • Successfully processed {total_samples:,} samples across {len(results)} datasets")
        print(f"  • Demonstrated comprehensive preprocessing pipeline")
        print(f"  • Extracted up to {total_features:,} TF-IDF features")
        print(f"  • Trained and evaluated multiple models")
        print(f"  • Generated detailed performance reports")
        
    else:
        print("❌ No datasets were successfully processed.")

def main():
    """Main demonstration function."""
    print("🚀 Starting Large Dataset Training Demonstration...")
    print("\nThis demonstration will show:")
    print("  ✓ Loading datasets with 8K-15K samples each (33K total)")
    print("  ✓ Comprehensive text preprocessing pipeline")
    print("  ✓ TF-IDF feature extraction with 5K features")
    print("  ✓ Memory-efficient batch processing")
    print("  ✓ Multi-model training and evaluation")
    print("  ✓ Detailed performance reporting")
    
    input("\nPress Enter to continue...")
    
    demonstrate_large_dataset_training()

if __name__ == "__main__":
    main()