"""Automated Benchmarking System for Text Classification and XAI.

This module provides comprehensive automated benchmarking capabilities for comparing
multiple models, datasets, and explainable AI techniques systematically.
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import project modules
from src.dataset_integration import DatasetManager
from src.explainable_ai import (
    extract_feature_importance, predict_with_explanation,
    visualize_feature_importance, explain_model
)
from src.transfer_learning import TransferLearningFramework, PretrainedEmbeddingTransformer

# Try to import XAI libraries
try:
    import lime
    import lime.lime_text
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class BenchmarkResult:
    """Container for benchmark results."""
    
    def __init__(self, dataset_name: str, model_name: str, embedding_type: str = 'tfidf'):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.embedding_type = embedding_type
        self.timestamp = datetime.now()
        
        # Performance metrics
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1_score = 0.0
        self.roc_auc = 0.0
        self.cv_scores = []
        self.cv_mean = 0.0
        self.cv_std = 0.0
        
        # Training metrics
        self.training_time = 0.0
        self.prediction_time = 0.0
        self.model_size = 0
        
        # XAI metrics
        self.xai_available = False
        self.lime_time = 0.0
        self.shap_time = 0.0
        self.feature_importance_available = False
        self.top_features = []
        
        # Additional info
        self.train_samples = 0
        self.test_samples = 0
        self.feature_count = 0
        self.class_distribution = {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'dataset_name': self.dataset_name,
            'model_name': self.model_name,
            'embedding_type': self.embedding_type,
            'timestamp': self.timestamp.isoformat(),
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'roc_auc': self.roc_auc,
            'cv_mean': self.cv_mean,
            'cv_std': self.cv_std,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time,
            'model_size': self.model_size,
            'xai_available': self.xai_available,
            'lime_time': self.lime_time,
            'shap_time': self.shap_time,
            'feature_importance_available': self.feature_importance_available,
            'top_features': self.top_features,
            'train_samples': self.train_samples,
            'test_samples': self.test_samples,
            'feature_count': self.feature_count,
            'class_distribution': self.class_distribution
        }


class AutomatedBenchmark:
    """Comprehensive automated benchmarking system."""
    
    def __init__(self, output_dir: str = "outputs/benchmarks"):
        """
        Initialize the automated benchmarking system.
        
        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_manager = DatasetManager()
        self.results = []
        self.datasets = {}
        
        # Default models to benchmark
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'svm_linear': SVC(kernel='linear', probability=True, random_state=42),
            'naive_bayes': MultinomialNB()
        }
        
        # Embedding types to test
        self.embedding_types = ['tfidf', 'sentence_transformer']
        
        print(f"Automated Benchmark initialized")
        print(f"Output directory: {self.output_dir}")
        print(f"Available XAI libraries: LIME={LIME_AVAILABLE}, SHAP={SHAP_AVAILABLE}")
    
    def load_datasets(self, dataset_names: List[str], sample_size: int = 1000) -> None:
        """Load datasets for benchmarking."""
        print(f"Loading {len(dataset_names)} datasets for benchmarking...")
        
        for dataset_name in dataset_names:
            try:
                print(f"Loading {dataset_name}...")
                train_df, test_df = self.dataset_manager.load_dataset(dataset_name, sample_size=sample_size)
                
                # Standardize format
                train_df = self.dataset_manager.standardize_dataset(train_df, dataset_name)
                test_df = self.dataset_manager.standardize_dataset(test_df, dataset_name)
                
                self.datasets[dataset_name] = {
                    'train': train_df,
                    'test': test_df
                }
                
                print(f"  ✓ {dataset_name}: Train {train_df.shape}, Test {test_df.shape}")
                
            except Exception as e:
                print(f"  ✗ Error loading {dataset_name}: {e}")
        
        print(f"Successfully loaded {len(self.datasets)} datasets")
    
    def benchmark_single_configuration(self, dataset_name: str, model_name: str, 
                                     embedding_type: str = 'tfidf') -> BenchmarkResult:
        """Benchmark a single dataset-model-embedding configuration."""
        print(f"Benchmarking: {dataset_name} + {model_name} + {embedding_type}")
        
        result = BenchmarkResult(dataset_name, model_name, embedding_type)
        
        try:
            # Get dataset
            dataset = self.datasets[dataset_name]
            train_df = dataset['train']
            test_df = dataset['test']
            
            # Prepare data
            X_train = train_df['text'].tolist()
            y_train = train_df['label'].values
            X_test = test_df['text'].tolist()
            y_test = test_df['label'].values
            
            result.train_samples = len(X_train)
            result.test_samples = len(X_test)
            result.class_distribution = dict(pd.Series(y_train).value_counts())
            
            # Create embedding transformer
            if embedding_type == 'tfidf':
                embedding_transformer = TfidfVectorizer(
                    max_features=5000,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95
                )
            else:
                embedding_transformer = PretrainedEmbeddingTransformer(
                    embedding_type=embedding_type
                )
            
            # Transform data
            print("  Creating embeddings...")
            start_time = time.time()
            
            if embedding_type == 'tfidf':
                X_train_transformed = embedding_transformer.fit_transform(X_train)
                X_test_transformed = embedding_transformer.transform(X_test)
                if hasattr(X_train_transformed, 'toarray'):
                    X_train_transformed = X_train_transformed.toarray()
                    X_test_transformed = X_test_transformed.toarray()
            else:
                embedding_transformer.fit(X_train)
                X_train_transformed = embedding_transformer.transform(X_train)
                X_test_transformed = embedding_transformer.transform(X_test)
            
            embedding_time = time.time() - start_time
            result.feature_count = X_train_transformed.shape[1]
            
            # Train model
            print("  Training model...")
            model = self.models[model_name]
            
            start_time = time.time()
            model.fit(X_train_transformed, y_train)
            result.training_time = time.time() - start_time
            
            # Evaluate model
            print("  Evaluating model...")
            start_time = time.time()
            y_pred = model.predict(X_test_transformed)
            result.prediction_time = time.time() - start_time
            
            # Calculate metrics
            result.accuracy = accuracy_score(y_test, y_pred)
            result.precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            result.recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            result.f1_score = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # ROC AUC (for binary classification or with probability)
            try:
                if hasattr(model, 'predict_proba') and len(np.unique(y_test)) == 2:
                    y_proba = model.predict_proba(X_test_transformed)[:, 1]
                    result.roc_auc = roc_auc_score(y_test, y_proba)
                elif hasattr(model, 'decision_function') and len(np.unique(y_test)) == 2:
                    y_scores = model.decision_function(X_test_transformed)
                    result.roc_auc = roc_auc_score(y_test, y_scores)
            except Exception as e:
                print(f"    Warning: Could not calculate ROC AUC: {e}")
            
            # Cross-validation
            print("  Running cross-validation...")
            try:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X_train_transformed, y_train, cv=cv, scoring='accuracy')
                result.cv_scores = cv_scores.tolist()
                result.cv_mean = cv_scores.mean()
                result.cv_std = cv_scores.std()
            except Exception as e:
                print(f"    Warning: Cross-validation failed: {e}")
            
            # XAI Analysis
            print("  Analyzing explainability...")
            self._analyze_xai(model, embedding_transformer, X_train, X_test, y_test, result)
            
            # Model size estimation
            try:
                import pickle
                result.model_size = len(pickle.dumps(model))
            except:
                result.model_size = 0
            
            print(f"  ✓ Completed: Accuracy={result.accuracy:.4f}, F1={result.f1_score:.4f}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
        
        return result
    
    def _analyze_xai(self, model, embedding_transformer, X_train: List[str], 
                    X_test: List[str], y_test: np.ndarray, result: BenchmarkResult) -> None:
        """Analyze explainability for the model."""
        try:
            # Feature importance (for compatible models)
            if hasattr(model, 'coef_') or hasattr(model, 'feature_importances_'):
                result.feature_importance_available = True
                
                if isinstance(embedding_transformer, TfidfVectorizer):
                    feature_names = embedding_transformer.get_feature_names_out()
                    importance_scores, feature_names_out = extract_feature_importance(
                        model, feature_names
                    )
                    result.top_features = list(zip(feature_names_out[:10], importance_scores[:10]))
            
            # LIME analysis (sample a few examples)
            if LIME_AVAILABLE and isinstance(embedding_transformer, TfidfVectorizer):
                try:
                    start_time = time.time()
                    
                    # Create LIME explainer
                    explainer = lime.lime_text.LimeTextExplainer(
                        class_names=['0', '1'],
                        mode='classification'
                    )
                    
                    # Test on a few samples
                    n_samples = min(5, len(X_test))
                    indices = np.random.choice(len(X_test), n_samples, replace=False)
                    
                    for idx in indices:
                        text = X_test[idx]
                        
                        def predict_fn(texts):
                            transformed = embedding_transformer.transform(texts)
                            if hasattr(transformed, 'toarray'):
                                transformed = transformed.toarray()
                            if hasattr(model, 'predict_proba'):
                                return model.predict_proba(transformed)
                            else:
                                # Convert decision function to probabilities
                                scores = model.decision_function(transformed)
                                return np.column_stack([1-scores, scores])
                        
                        exp = explainer.explain_instance(
                            text, predict_fn, num_features=10, num_samples=100
                        )
                        break  # Just test one for timing
                    
                    result.lime_time = time.time() - start_time
                    result.xai_available = True
                    
                except Exception as e:
                    print(f"    LIME analysis failed: {e}")
            
            # SHAP analysis (for compatible models)
            if SHAP_AVAILABLE and hasattr(model, 'predict_proba'):
                try:
                    start_time = time.time()
                    
                    # Sample data for SHAP (it can be slow)
                    if isinstance(embedding_transformer, TfidfVectorizer):
                        X_sample = embedding_transformer.transform(X_test[:10])
                        if hasattr(X_sample, 'toarray'):
                            X_sample = X_sample.toarray()
                        
                        explainer = shap.Explainer(model.predict_proba, X_sample)
                        shap_values = explainer(X_sample[:5])  # Just a few samples
                    
                    result.shap_time = time.time() - start_time
                    result.xai_available = True
                    
                except Exception as e:
                    print(f"    SHAP analysis failed: {e}")
                    
        except Exception as e:
            print(f"    XAI analysis failed: {e}")
    
    def run_comprehensive_benchmark(self, dataset_names: List[str], 
                                  model_names: Optional[List[str]] = None,
                                  embedding_types: Optional[List[str]] = None,
                                  sample_size: int = 1000) -> List[BenchmarkResult]:
        """Run comprehensive benchmark across all configurations."""
        print("🚀 Starting comprehensive benchmark...")
        
        # Use defaults if not specified
        if model_names is None:
            model_names = list(self.models.keys())
        if embedding_types is None:
            embedding_types = self.embedding_types
        
        # Load datasets
        self.load_datasets(dataset_names, sample_size)
        
        # Calculate total configurations
        total_configs = len(self.datasets) * len(model_names) * len(embedding_types)
        print(f"Total configurations to benchmark: {total_configs}")
        
        # Run benchmarks
        results = []
        config_count = 0
        
        for dataset_name in self.datasets.keys():
            for model_name in model_names:
                for embedding_type in embedding_types:
                    config_count += 1
                    print(f"\n[{config_count}/{total_configs}] ", end="")
                    
                    result = self.benchmark_single_configuration(
                        dataset_name, model_name, embedding_type
                    )
                    results.append(result)
        
        self.results.extend(results)
        
        print(f"\n✅ Benchmark completed! {len(results)} configurations tested.")
        return results
    
    def save_results(self, filename: str = None) -> str:
        """Save benchmark results to files."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}"
        
        # Save as JSON
        json_file = self.output_dir / f"{filename}.json"
        with open(json_file, 'w') as f:
            json.dump([result.to_dict() for result in self.results], f, indent=2)
        
        # Save as CSV
        csv_file = self.output_dir / f"{filename}.csv"
        df = pd.DataFrame([result.to_dict() for result in self.results])
        df.to_csv(csv_file, index=False)
        
        print(f"Results saved to:")
        print(f"  JSON: {json_file}")
        print(f"  CSV: {csv_file}")
        
        return str(csv_file)
    
    def generate_performance_report(self) -> pd.DataFrame:
        """Generate performance comparison report."""
        if not self.results:
            print("No results available. Run benchmark first.")
            return pd.DataFrame()
        
        # Create summary DataFrame
        summary_data = []
        
        for result in self.results:
            summary_data.append({
                'Dataset': result.dataset_name,
                'Model': result.model_name,
                'Embedding': result.embedding_type,
                'Accuracy': result.accuracy,
                'F1 Score': result.f1_score,
                'Precision': result.precision,
                'Recall': result.recall,
                'ROC AUC': result.roc_auc,
                'CV Mean': result.cv_mean,
                'CV Std': result.cv_std,
                'Training Time (s)': result.training_time,
                'Prediction Time (s)': result.prediction_time,
                'Features': result.feature_count,
                'XAI Available': result.xai_available
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        summary_file = self.output_dir / "performance_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        print(f"Performance summary saved to: {summary_file}")
        return summary_df
    
    def visualize_results(self) -> None:
        """Create comprehensive visualizations of benchmark results."""
        if not self.results:
            print("No results available. Run benchmark first.")
            return
        
        # Create summary DataFrame
        df = self.generate_performance_report()
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Accuracy comparison by model
        plt.subplot(3, 3, 1)
        model_accuracy = df.groupby('Model')['Accuracy'].mean().sort_values(ascending=False)
        model_accuracy.plot(kind='bar')
        plt.title('Average Accuracy by Model')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        
        # 2. Accuracy comparison by dataset
        plt.subplot(3, 3, 2)
        dataset_accuracy = df.groupby('Dataset')['Accuracy'].mean().sort_values(ascending=False)
        dataset_accuracy.plot(kind='bar')
        plt.title('Average Accuracy by Dataset')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        
        # 3. Embedding type comparison
        plt.subplot(3, 3, 3)
        embedding_accuracy = df.groupby('Embedding')['Accuracy'].mean().sort_values(ascending=False)
        embedding_accuracy.plot(kind='bar')
        plt.title('Average Accuracy by Embedding Type')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        
        # 4. Training time vs Accuracy
        plt.subplot(3, 3, 4)
        plt.scatter(df['Training Time (s)'], df['Accuracy'], alpha=0.6)
        plt.xlabel('Training Time (seconds)')
        plt.ylabel('Accuracy')
        plt.title('Training Time vs Accuracy')
        
        # 5. F1 Score heatmap
        plt.subplot(3, 3, 5)
        pivot_f1 = df.pivot_table(values='F1 Score', index='Dataset', columns='Model', aggfunc='mean')
        sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='RdYlBu_r')
        plt.title('F1 Score Heatmap')
        
        # 6. Cross-validation stability
        plt.subplot(3, 3, 6)
        plt.scatter(df['CV Mean'], df['CV Std'], alpha=0.6)
        plt.xlabel('CV Mean Accuracy')
        plt.ylabel('CV Standard Deviation')
        plt.title('Model Stability (CV Mean vs Std)')
        
        # 7. Feature count vs Performance
        plt.subplot(3, 3, 7)
        plt.scatter(df['Features'], df['Accuracy'], alpha=0.6)
        plt.xlabel('Number of Features')
        plt.ylabel('Accuracy')
        plt.title('Feature Count vs Accuracy')
        
        # 8. XAI Availability
        plt.subplot(3, 3, 8)
        xai_counts = df['XAI Available'].value_counts()
        xai_counts.plot(kind='pie', autopct='%1.1f%%')
        plt.title('XAI Availability')
        plt.ylabel('')
        
        # 9. Performance distribution
        plt.subplot(3, 3, 9)
        df['Accuracy'].hist(bins=20, alpha=0.7)
        plt.xlabel('Accuracy')
        plt.ylabel('Frequency')
        plt.title('Accuracy Distribution')
        
        plt.tight_layout()
        
        # Save visualization
        viz_file = self.output_dir / "benchmark_visualization.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualization saved to: {viz_file}")
    
    def get_best_configurations(self, metric: str = 'accuracy', top_k: int = 10) -> pd.DataFrame:
        """Get the best performing configurations."""
        if not self.results:
            print("No results available. Run benchmark first.")
            return pd.DataFrame()
        
        # Create DataFrame
        data = []
        for result in self.results:
            data.append({
                'Dataset': result.dataset_name,
                'Model': result.model_name,
                'Embedding': result.embedding_type,
                'Accuracy': result.accuracy,
                'F1 Score': result.f1_score,
                'Precision': result.precision,
                'Recall': result.recall,
                'Training Time': result.training_time,
                'XAI Available': result.xai_available
            })
        
        df = pd.DataFrame(data)
        
        # Sort by specified metric
        metric_map = {
            'accuracy': 'Accuracy',
            'f1': 'F1 Score',
            'precision': 'Precision',
            'recall': 'Recall'
        }
        
        sort_column = metric_map.get(metric.lower(), 'Accuracy')
        best_configs = df.nlargest(top_k, sort_column)
        
        print(f"\nTop {top_k} configurations by {sort_column}:")
        print("=" * 80)
        print(best_configs.to_string(index=False))
        
        return best_configs
    
    def generate_academic_insights(self) -> Dict[str, Any]:
        """Generate academic insights from benchmark results."""
        if not self.results:
            return {}
        
        df = self.generate_performance_report()
        
        insights = {
            'total_configurations': len(self.results),
            'datasets_tested': df['Dataset'].nunique(),
            'models_tested': df['Model'].nunique(),
            'embedding_types_tested': df['Embedding'].nunique(),
            
            'performance_statistics': {
                'mean_accuracy': df['Accuracy'].mean(),
                'std_accuracy': df['Accuracy'].std(),
                'max_accuracy': df['Accuracy'].max(),
                'min_accuracy': df['Accuracy'].min(),
                'mean_f1': df['F1 Score'].mean(),
                'std_f1': df['F1 Score'].std()
            },
            
            'best_model_overall': df.loc[df['Accuracy'].idxmax(), 'Model'],
            'best_embedding_overall': df.loc[df['Accuracy'].idxmax(), 'Embedding'],
            'most_challenging_dataset': df.groupby('Dataset')['Accuracy'].mean().idxmin(),
            'easiest_dataset': df.groupby('Dataset')['Accuracy'].mean().idxmax(),
            
            'efficiency_insights': {
                'fastest_training_model': df.loc[df['Training Time (s)'].idxmin(), 'Model'],
                'slowest_training_model': df.loc[df['Training Time (s)'].idxmax(), 'Model'],
                'mean_training_time': df['Training Time (s)'].mean(),
                'training_time_std': df['Training Time (s)'].std()
            },
            
            'xai_insights': {
                'xai_availability_rate': df['XAI Available'].mean(),
                'models_with_xai': df[df['XAI Available']]['Model'].unique().tolist(),
                'models_without_xai': df[~df['XAI Available']]['Model'].unique().tolist()
            }
        }
        
        # Save insights
        insights_file = self.output_dir / "academic_insights.json"
        with open(insights_file, 'w') as f:
            json.dump(insights, f, indent=2)
        
        print(f"Academic insights saved to: {insights_file}")
        return insights


def main():
    """Main function to run automated benchmarking."""
    # Initialize benchmark system
    benchmark = AutomatedBenchmark()
    
    # Define datasets and configurations
    dataset_names = ['imdb', 'ag_news', 'jigsaw_toxic']  # Adjust based on available datasets
    model_names = ['logistic_regression', 'random_forest', 'svm_linear']
    embedding_types = ['tfidf']  # Start with TF-IDF for reliability
    
    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark(
        dataset_names=dataset_names,
        model_names=model_names,
        embedding_types=embedding_types,
        sample_size=500
    )
    
    # Save and analyze results
    benchmark.save_results()
    benchmark.visualize_results()
    best_configs = benchmark.get_best_configurations()
    insights = benchmark.generate_academic_insights()
    
    print("\n" + "="*80)
    print("🎉 AUTOMATED BENCHMARKING COMPLETED!")
    print("="*80)
    print(f"Total configurations tested: {len(results)}")
    print(f"Results saved to: {benchmark.output_dir}")
    
    return benchmark, results


if __name__ == "__main__":
    benchmark, results = main()