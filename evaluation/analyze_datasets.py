#!/usr/bin/env python3
"""
Dataset Analysis Script
Analyzes all available datasets and their capabilities for large-scale training
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import project modules
from src.dataset_integration import DatasetManager
from src.preprocess import DataPreprocessor
from src.batch_processor import MemoryEfficientBatchProcessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_dataset_capabilities():
    """Analyze all available datasets and their sample size capabilities."""
    print("=" * 80)
    print("DATASET ANALYSIS - Available Datasets and Sample Sizes")
    print("=" * 80)
    
    # Initialize dataset manager
    manager = DatasetManager()
    
    # Get dataset information
    dataset_info = manager.get_dataset_info()
    
    print("\n📊 Available Datasets:")
    print("-" * 50)
    for name, info in dataset_info.items():
        print(f"• {name.upper()}:")
        print(f"  - Description: {info['description']}")
        print(f"  - Task: {info['task']}")
        print(f"  - Text Column: {info['text_column']}")
        print(f"  - Label Column: {info['label_column']}")
        print(f"  - Classes: {info['classes']}")
        print()
    
    # Test different sample sizes
    sample_sizes = [1000, 5000, 10000, 20000, 50000]
    
    print("\n📈 Sample Size Analysis:")
    print("-" * 50)
    
    results = {}
    for sample_size in sample_sizes:
        print(f"\nTesting sample size: {sample_size:,}")
        dataset_results = {}
        
        for dataset_name in dataset_info.keys():
            try:
                train_df, test_df = manager.load_dataset(dataset_name, sample_size=sample_size)
                total_samples = len(train_df) + len(test_df)
                dataset_results[dataset_name] = {
                    'train_size': len(train_df),
                    'test_size': len(test_df),
                    'total_size': total_samples
                }
                print(f"  ✓ {dataset_name}: {total_samples:,} samples (train: {len(train_df):,}, test: {len(test_df):,})")
            except Exception as e:
                print(f"  ✗ {dataset_name}: Failed - {str(e)}")
                dataset_results[dataset_name] = {'error': str(e)}
        
        results[sample_size] = dataset_results
    
    return results, dataset_info

def demonstrate_preprocessing_pipeline():
    """Demonstrate the preprocessing pipeline with actual data."""
    print("\n" + "=" * 80)
    print("PREPROCESSING PIPELINE DEMONSTRATION")
    print("=" * 80)
    
    # Initialize components
    manager = DatasetManager()
    preprocessor = DataPreprocessor()
    
    # Load a sample dataset
    print("\n🔄 Loading sample data for preprocessing demonstration...")
    train_df, test_df = manager.load_dataset('imdb', sample_size=100)
    
    print(f"\n📝 Original Data Sample:")
    print("-" * 30)
    sample_text = train_df.iloc[0]['review']
    print(f"Original text: {sample_text[:200]}...")
    
    # Demonstrate text cleaning
    print("\n🧹 Text Cleaning Steps:")
    print("-" * 30)
    
    # Step 1: Raw text
    print(f"1. Raw text: {sample_text[:100]}...")
    
    # Step 2: Cleaned text
    cleaned_text = preprocessor.clean_text(sample_text)
    print(f"2. Cleaned text: {cleaned_text[:100]}...")
    
    # Step 3: Preprocess entire dataframe
    print("\n📊 Processing entire dataset...")
    standardized_train = manager.standardize_dataset(train_df, 'imdb')
    processed_train = preprocessor.preprocess_dataframe(standardized_train)
    
    print(f"Dataset shape: {processed_train.shape}")
    print(f"Columns: {list(processed_train.columns)}")
    print(f"Sample processed text: {processed_train.iloc[0]['text'][:100]}...")
    
    # Show text length distribution
    text_lengths = processed_train['text'].str.len()
    print(f"\n📏 Text Length Statistics:")
    print(f"  - Mean length: {text_lengths.mean():.1f} characters")
    print(f"  - Median length: {text_lengths.median():.1f} characters")
    print(f"  - Max length: {text_lengths.max()} characters")
    print(f"  - Min length: {text_lengths.min()} characters")
    
    return processed_train

def demonstrate_batch_processing():
    """Demonstrate memory-efficient batch processing capabilities."""
    print("\n" + "=" * 80)
    print("BATCH PROCESSING CAPABILITIES")
    print("=" * 80)
    
    # Initialize batch processor
    batch_processor = MemoryEfficientBatchProcessor()
    
    print(f"\n⚙️ Batch Processor Configuration:")
    print(f"  - Batch size: {batch_processor.batch_size:,}")
    print(f"  - Number of workers: {batch_processor.n_jobs}")
    print(f"  - Memory threshold: {batch_processor.memory_threshold * 100}%")
    
    # Monitor current memory
    memory_info = batch_processor._monitor_memory()
    print(f"\n💾 Current Memory Status:")
    print(f"  - Total memory: {memory_info['total_gb']:.1f} GB")
    print(f"  - Available memory: {memory_info['available_gb']:.1f} GB")
    print(f"  - Memory usage: {memory_info['used_percent']:.1f}%")
    if memory_info['warning']:
        print(f"  ⚠️ Memory usage is high!")
    else:
        print(f"  ✅ Memory usage is normal")
    
    return batch_processor

def create_visualization(results: Dict, dataset_info: Dict):
    """Create visualizations of dataset analysis results."""
    print("\n📊 Creating visualizations...")
    
    # Create output directory
    output_dir = Path('reports')
    output_dir.mkdir(exist_ok=True)
    
    # Prepare data for visualization
    sample_sizes = list(results.keys())
    datasets = list(dataset_info.keys())
    
    # Create a matrix of total sample sizes
    size_matrix = []
    for sample_size in sample_sizes:
        row = []
        for dataset in datasets:
            if dataset in results[sample_size] and 'total_size' in results[sample_size][dataset]:
                row.append(results[sample_size][dataset]['total_size'])
            else:
                row.append(0)
        size_matrix.append(row)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(size_matrix, 
                xticklabels=[d.replace('_', ' ').title() for d in datasets],
                yticklabels=[f"{s:,}" for s in sample_sizes],
                annot=True, 
                fmt='d',
                cmap='YlOrRd',
                cbar_kws={'label': 'Total Samples Generated'})
    
    plt.title('Dataset Sample Size Analysis\nActual Samples Generated vs Requested Sample Size', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Dataset Type', fontweight='bold')
    plt.ylabel('Requested Sample Size', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save visualization
    viz_path = output_dir / 'dataset_analysis_heatmap.png'
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"📈 Visualization saved to: {viz_path}")
    plt.close()

def main():
    """Main analysis function."""
    print("🚀 Starting comprehensive dataset analysis...")
    
    try:
        # Analyze dataset capabilities
        results, dataset_info = analyze_dataset_capabilities()
        
        # Demonstrate preprocessing
        processed_data = demonstrate_preprocessing_pipeline()
        
        # Demonstrate batch processing
        batch_processor = demonstrate_batch_processing()
        
        # Create visualizations
        create_visualization(results, dataset_info)
        
        # Summary
        print("\n" + "=" * 80)
        print("ANALYSIS SUMMARY")
        print("=" * 80)
        print("\n✅ Dataset Analysis Complete!")
        print(f"\n📋 Key Findings:")
        print(f"  • {len(dataset_info)} datasets available for training")
        print(f"  • Sample sizes tested: 1K to 50K samples per dataset")
        print(f"  • All datasets support configurable sample sizes")
        print(f"  • Preprocessing pipeline includes text cleaning and standardization")
        print(f"  • Batch processing supports memory-efficient handling of large datasets")
        print(f"  • Optimal batch size: {batch_processor.batch_size:,} samples")
        
        print(f"\n🎯 Recommendations for Large-Scale Training:")
        print(f"  • Use 10K-50K samples per dataset for robust training")
        print(f"  • Combine multiple datasets for cross-domain evaluation")
        print(f"  • Utilize batch processing for memory efficiency")
        print(f"  • Apply comprehensive preprocessing pipeline")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()