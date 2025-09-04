#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Report Generator
Generates detailed evaluation reports using the larger multilingual dataset
"""

import os
import pandas as pd
from datetime import datetime
from training_pipeline import MLTrainingPipeline
from src.evaluation_metrics import KaggleEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def generate_comprehensive_report():
    """
    Generate a comprehensive evaluation report with the larger dataset
    """
    print("=" * 80)
    print("COMPREHENSIVE MODEL EVALUATION REPORT GENERATOR")
    print("=" * 80)
    
    # Create output directory
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize pipeline
    print("\n1. Initializing training pipeline...")
    pipeline = MLTrainingPipeline()
    
    # Load data
    print("2. Loading multilingual dataset...")
    data = pipeline.load_data()
    print(f"   Loaded {len(data)} samples")
    print(f"   Class distribution: {data['target'].value_counts().to_dict()}")
    
    # Train models
    print("\n3. Training models with comprehensive evaluation...")
    models_to_evaluate = [
        'logistic_regression',
        'random_forest', 
        'svm_linear',
        'multinomial_nb',
        'gradient_boosting'
    ]
    
    results = pipeline.train_pipeline(
        df=data,
        models_to_train=models_to_evaluate,
        use_grid_search=True  # Enable grid search for best performance
    )
    
    # Generate timestamp for report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate detailed report
    print("\n4. Generating comprehensive report...")
    report_content = generate_detailed_report(results, data, timestamp)
    
    # Save report
    report_path = os.path.join(output_dir, f"comprehensive_evaluation_report_{timestamp}.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n✅ Comprehensive report generated: {report_path}")
    
    # Generate visualizations
    print("\n5. Generating visualizations...")
    generate_visualizations(results, output_dir, timestamp)
    
    # Generate CSV summary
    print("\n6. Generating CSV summary...")
    generate_csv_summary(results, output_dir, timestamp)
    
    print("\n" + "=" * 80)
    print("REPORT GENERATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"📁 Reports saved in: {output_dir}/")
    print(f"📊 Dataset size: {len(data)} samples")
    print(f"🔄 Train/Val/Test split: {results['train_size']}/{results['val_size']}/{results['test_size']}")
    print(f"🎯 Models evaluated: {len(models_to_evaluate)}")
    
    return report_path

def generate_detailed_report(results, data, timestamp):
    """
    Generate detailed text report
    """
    report = []
    report.append("=" * 80)
    report.append("COMPREHENSIVE MODEL EVALUATION REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Report ID: {timestamp}")
    report.append("")
    
    # Dataset Information
    report.append("DATASET INFORMATION")
    report.append("-" * 40)
    report.append(f"Total samples: {len(data)}")
    report.append(f"Class distribution: {data['target'].value_counts().to_dict()}")
    report.append(f"Train samples: {results['train_size']}")
    report.append(f"Validation samples: {results['val_size']}")
    report.append(f"Test samples: {results['test_size']}")
    report.append(f"Feature matrix shape: {results['feature_shape']}")
    report.append("")
    
    # Configuration
    report.append("CONFIGURATION")
    report.append("-" * 40)
    report.append(f"Preprocessing: {results['preprocessing_config']}")
    report.append(f"Feature extraction: {results['feature_config']}")
    report.append("")
    
    # Validation Results
    report.append("VALIDATION RESULTS")
    report.append("-" * 40)
    for model_name, metrics in results['validation_results'].items():
        report.append(f"\n{model_name.upper()}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                report.append(f"  {metric}: {value:.4f}")
            else:
                report.append(f"  {metric}: {value}")
    report.append("")
    
    # Test Results
    report.append("TEST RESULTS")
    report.append("-" * 40)
    for model_name, metrics in results['test_results'].items():
        report.append(f"\n{model_name.upper()}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                report.append(f"  {metric}: {value:.4f}")
            else:
                report.append(f"  {metric}: {value}")
    report.append("")
    
    # Model Comparison
    report.append("MODEL COMPARISON (TEST SET)")
    report.append("-" * 40)
    test_results = results['test_results']
    
    # Find best models
    best_accuracy = max(test_results.values(), key=lambda x: x.get('accuracy', 0))
    best_f1 = max(test_results.values(), key=lambda x: x.get('f1_score', 0))
    best_auc = max(test_results.values(), key=lambda x: x.get('auc_score', 0))
    
    best_acc_model = [name for name, metrics in test_results.items() if metrics.get('accuracy', 0) == best_accuracy['accuracy']][0]
    best_f1_model = [name for name, metrics in test_results.items() if metrics.get('f1_score', 0) == best_f1['f1_score']][0]
    best_auc_model = [name for name, metrics in test_results.items() if metrics.get('auc_score', 0) == best_auc['auc_score']][0]
    
    report.append(f"Best Accuracy: {best_acc_model} ({best_accuracy['accuracy']:.4f})")
    report.append(f"Best F1-Score: {best_f1_model} ({best_f1['f1_score']:.4f})")
    report.append(f"Best AUC-Score: {best_auc_model} ({best_auc['auc_score']:.4f})")
    report.append("")
    
    # Summary
    report.append("SUMMARY")
    report.append("-" * 40)
    report.append(f"✅ Successfully evaluated {len(test_results)} models")
    report.append(f"📊 Used {len(data)} samples (vs. previous 60 synthetic samples)")
    report.append(f"🎯 Realistic performance metrics obtained")
    report.append(f"🔍 Feature extraction: {results['feature_shape'][1]} features")
    report.append("")
    
    return "\n".join(report)

def generate_visualizations(results, output_dir, timestamp):
    """
    Generate visualization plots
    """
    # Model comparison plot
    test_results = results['test_results']
    models = list(test_results.keys())
    accuracies = [test_results[model]['accuracy'] for model in models]
    f1_scores = [test_results[model]['f1_score'] for model in models]
    auc_scores = [test_results[model]['auc_score'] for model in models]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Accuracy comparison
    ax1.bar(models, accuracies, color='skyblue', alpha=0.7)
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1.1)
    ax1.tick_params(axis='x', rotation=45)
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # F1-Score comparison
    ax2.bar(models, f1_scores, color='lightgreen', alpha=0.7)
    ax2.set_title('Model F1-Score Comparison')
    ax2.set_ylabel('F1-Score')
    ax2.set_ylim(0, 1.1)
    ax2.tick_params(axis='x', rotation=45)
    for i, v in enumerate(f1_scores):
        ax2.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # AUC-Score comparison
    ax3.bar(models, auc_scores, color='lightcoral', alpha=0.7)
    ax3.set_title('Model AUC-Score Comparison')
    ax3.set_ylabel('AUC-Score')
    ax3.set_ylim(0, 1.1)
    ax3.tick_params(axis='x', rotation=45)
    for i, v in enumerate(auc_scores):
        ax3.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"model_comparison_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   📊 Model comparison plot saved: {plot_path}")

def generate_csv_summary(results, output_dir, timestamp):
    """
    Generate CSV summary of results
    """
    # Prepare data for CSV
    csv_data = []
    
    for model_name, metrics in results['test_results'].items():
        row = {
            'model': model_name,
            'accuracy': metrics.get('accuracy', 0),
            'f1_score': metrics.get('f1_score', 0),
            'auc_score': metrics.get('auc_score', 0),
            'dataset_size': results['train_size'] + results['val_size'] + results['test_size'],
            'train_size': results['train_size'],
            'val_size': results['val_size'],
            'test_size': results['test_size'],
            'feature_count': results['feature_shape'][1],
            'timestamp': timestamp
        }
        csv_data.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(csv_data)
    csv_path = os.path.join(output_dir, f"evaluation_summary_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"   📋 CSV summary saved: {csv_path}")

if __name__ == "__main__":
    try:
        report_path = generate_comprehensive_report()
        print(f"\n🎉 Report generation completed successfully!")
        print(f"📄 Main report: {report_path}")
    except Exception as e:
        print(f"\n❌ Error generating report: {str(e)}")
        import traceback
        traceback.print_exc()