"""Transfer Learning Implementation for Text Classification.

This module implements various transfer learning techniques for text classification,
including pre-trained embeddings, domain adaptation, and multi-source learning.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import project modules
from src.dataset_integration import DatasetManager
from src.explainable_ai import extract_feature_importance

# Try to import sentence transformers for advanced embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")

# Try to import transformers for BERT-like models
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers torch")


class PretrainedEmbeddingTransformer(BaseEstimator, TransformerMixin):
    """Transform text using pre-trained embeddings."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', embedding_type: str = 'sentence_transformer'):
        """
        Initialize the embedding transformer.
        
        Args:
            model_name: Name of the pre-trained model
            embedding_type: Type of embedding ('sentence_transformer', 'transformers', 'tfidf')
        """
        self.model_name = model_name
        self.embedding_type = embedding_type
        self.model = None
        self.tokenizer = None
        self.tfidf_vectorizer = None
        
    def fit(self, X, y=None):
        """Fit the embedding model."""
        if self.embedding_type == 'sentence_transformer' and SENTENCE_TRANSFORMERS_AVAILABLE:
            self.model = SentenceTransformer(self.model_name)
            
        elif self.embedding_type == 'transformers' and TRANSFORMERS_AVAILABLE:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
        elif self.embedding_type == 'tfidf':
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            self.tfidf_vectorizer.fit(X)
            
        else:
            # Fallback to TF-IDF
            print(f"Warning: {self.embedding_type} not available, falling back to TF-IDF")
            self.embedding_type = 'tfidf'
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.tfidf_vectorizer.fit(X)
            
        return self
    
    def transform(self, X):
        """Transform text to embeddings."""
        if self.embedding_type == 'sentence_transformer' and self.model is not None:
            embeddings = self.model.encode(X, show_progress_bar=False)
            return embeddings
            
        elif self.embedding_type == 'transformers' and self.model is not None:
            embeddings = []
            for text in X:
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use CLS token embedding
                    embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
                    embeddings.append(embedding)
            return np.array(embeddings)
            
        elif self.embedding_type == 'tfidf' and self.tfidf_vectorizer is not None:
            return self.tfidf_vectorizer.transform(X).toarray()
            
        else:
            raise ValueError(f"Model not properly initialized for {self.embedding_type}")


class DomainAdversarialClassifier(BaseEstimator, ClassifierMixin):
    """Simple domain adversarial classifier for domain adaptation."""
    
    def __init__(self, base_classifier=None, domain_classifier=None, lambda_domain=0.1):
        """
        Initialize domain adversarial classifier.
        
        Args:
            base_classifier: Main classification model
            domain_classifier: Domain classification model
            lambda_domain: Weight for domain adversarial loss
        """
        self.base_classifier = base_classifier or LogisticRegression(random_state=42, max_iter=1000)
        self.domain_classifier = domain_classifier or LogisticRegression(random_state=42, max_iter=1000)
        self.lambda_domain = lambda_domain
        
    def fit(self, X_source, y_source, X_target=None, y_target=None):
        """Fit the domain adversarial model."""
        # Train base classifier on source domain
        self.base_classifier.fit(X_source, y_source)
        
        # If target domain is available, train domain classifier
        if X_target is not None:
            # Create domain labels (0 for source, 1 for target)
            X_combined = np.vstack([X_source, X_target])
            domain_labels = np.hstack([
                np.zeros(len(X_source)),
                np.ones(len(X_target))
            ])
            
            # Train domain classifier
            self.domain_classifier.fit(X_combined, domain_labels)
            
        return self
    
    def predict(self, X):
        """Make predictions."""
        return self.base_classifier.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if hasattr(self.base_classifier, 'predict_proba'):
            return self.base_classifier.predict_proba(X)
        else:
            # Convert decision function to probabilities
            decision = self.base_classifier.decision_function(X)
            return np.column_stack([1 - decision, decision])
    
    def score(self, X, y):
        """Calculate accuracy score."""
        return accuracy_score(y, self.predict(X))


class TransferLearningFramework:
    """Comprehensive transfer learning framework for text classification."""
    
    def __init__(self, embedding_type: str = 'sentence_transformer'):
        """
        Initialize the transfer learning framework.
        
        Args:
            embedding_type: Type of embeddings to use
        """
        self.embedding_type = embedding_type
        self.dataset_manager = DatasetManager()
        self.datasets = {}
        self.embeddings = {}
        self.models = {}
        self.results = {}
        
        # Create output directory
        self.output_dir = Path("outputs/transfer_learning")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Transfer Learning Framework initialized with {embedding_type} embeddings")
        print(f"Output directory: {self.output_dir}")
    
    def load_datasets(self, dataset_names: List[str], sample_size: int = 1000) -> None:
        """Load multiple datasets for transfer learning experiments."""
        print(f"Loading {len(dataset_names)} datasets...")
        
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
    
    def create_embeddings(self, model_name: str = 'all-MiniLM-L6-v2') -> None:
        """Create embeddings for all loaded datasets."""
        print(f"Creating {self.embedding_type} embeddings...")
        
        for dataset_name, data in self.datasets.items():
            print(f"Processing {dataset_name}...")
            
            # Initialize embedding transformer
            embedding_transformer = PretrainedEmbeddingTransformer(
                model_name=model_name,
                embedding_type=self.embedding_type
            )
            
            # Fit on training data
            train_texts = data['train']['text'].tolist()
            test_texts = data['test']['text'].tolist()
            
            embedding_transformer.fit(train_texts)
            
            # Transform both training and test data
            train_embeddings = embedding_transformer.transform(train_texts)
            test_embeddings = embedding_transformer.transform(test_texts)
            
            self.embeddings[dataset_name] = {
                'train_X': train_embeddings,
                'train_y': data['train']['label'].values,
                'test_X': test_embeddings,
                'test_y': data['test']['label'].values,
                'transformer': embedding_transformer
            }
            
            print(f"  ✓ {dataset_name}: {train_embeddings.shape} train, {test_embeddings.shape} test")
        
        print("Embeddings created successfully!")
    
    def baseline_evaluation(self) -> Dict[str, Dict[str, float]]:
        """Evaluate baseline performance (within-domain)."""
        print("Running baseline evaluation (within-domain)...")
        
        baseline_results = {}
        
        for dataset_name, embeddings in self.embeddings.items():
            print(f"Evaluating {dataset_name}...")
            
            # Train models
            models = {
                'logistic': LogisticRegression(random_state=42, max_iter=1000),
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'svm': SVC(kernel='linear', probability=True, random_state=42)
            }
            
            dataset_results = {}
            
            for model_name, model in models.items():
                # Train and evaluate
                model.fit(embeddings['train_X'], embeddings['train_y'])
                accuracy = model.score(embeddings['test_X'], embeddings['test_y'])
                
                dataset_results[model_name] = accuracy
                print(f"  {model_name}: {accuracy:.4f}")
            
            baseline_results[dataset_name] = dataset_results
        
        self.results['baseline'] = baseline_results
        return baseline_results
    
    def cross_domain_evaluation(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Evaluate cross-domain transfer performance."""
        print("Running cross-domain evaluation...")
        
        cross_domain_results = {}
        dataset_names = list(self.embeddings.keys())
        
        for source_dataset in dataset_names:
            cross_domain_results[source_dataset] = {}
            
            for target_dataset in dataset_names:
                if source_dataset == target_dataset:
                    continue
                
                print(f"Transfer: {source_dataset} → {target_dataset}")
                
                # Get embeddings
                source_embeddings = self.embeddings[source_dataset]
                target_embeddings = self.embeddings[target_dataset]
                
                # Train models on source, evaluate on target
                models = {
                    'logistic': LogisticRegression(random_state=42, max_iter=1000),
                    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                    'svm': SVC(kernel='linear', probability=True, random_state=42)
                }
                
                transfer_results = {}
                
                for model_name, model in models.items():
                    # Train on source
                    model.fit(source_embeddings['train_X'], source_embeddings['train_y'])
                    
                    # Evaluate on target
                    accuracy = model.score(target_embeddings['test_X'], target_embeddings['test_y'])
                    transfer_results[model_name] = accuracy
                    
                    print(f"  {model_name}: {accuracy:.4f}")
                
                cross_domain_results[source_dataset][target_dataset] = transfer_results
        
        self.results['cross_domain'] = cross_domain_results
        return cross_domain_results
    
    def domain_adaptation_evaluation(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Evaluate domain adaptation techniques."""
        print("Running domain adaptation evaluation...")
        
        adaptation_results = {}
        dataset_names = list(self.embeddings.keys())
        
        for source_dataset in dataset_names:
            adaptation_results[source_dataset] = {}
            
            for target_dataset in dataset_names:
                if source_dataset == target_dataset:
                    continue
                
                print(f"Domain adaptation: {source_dataset} → {target_dataset}")
                
                # Get embeddings
                source_embeddings = self.embeddings[source_dataset]
                target_embeddings = self.embeddings[target_dataset]
                
                # Test different adaptation techniques
                techniques = {
                    'baseline': LogisticRegression(random_state=42, max_iter=1000),
                    'domain_adversarial': DomainAdversarialClassifier()
                }
                
                technique_results = {}
                
                for technique_name, model in techniques.items():
                    if technique_name == 'baseline':
                        # Standard transfer learning
                        model.fit(source_embeddings['train_X'], source_embeddings['train_y'])
                        accuracy = model.score(target_embeddings['test_X'], target_embeddings['test_y'])
                    
                    elif technique_name == 'domain_adversarial':
                        # Domain adversarial training
                        model.fit(
                            source_embeddings['train_X'], source_embeddings['train_y'],
                            target_embeddings['train_X']
                        )
                        accuracy = model.score(target_embeddings['test_X'], target_embeddings['test_y'])
                    
                    technique_results[technique_name] = accuracy
                    print(f"  {technique_name}: {accuracy:.4f}")
                
                adaptation_results[source_dataset][target_dataset] = technique_results
        
        self.results['domain_adaptation'] = adaptation_results
        return adaptation_results
    
    def visualize_transfer_matrix(self, results_type: str = 'cross_domain', model_type: str = 'logistic') -> None:
        """Visualize transfer learning results as a matrix."""
        if results_type not in self.results:
            print(f"Results for {results_type} not available. Run evaluation first.")
            return
        
        results = self.results[results_type]
        dataset_names = list(self.embeddings.keys())
        n_datasets = len(dataset_names)
        
        # Create transfer matrix
        transfer_matrix = np.zeros((n_datasets, n_datasets))
        
        for i, source in enumerate(dataset_names):
            for j, target in enumerate(dataset_names):
                if source == target:
                    # Use baseline results for diagonal
                    if 'baseline' in self.results:
                        transfer_matrix[i, j] = self.results['baseline'][source][model_type]
                else:
                    # Use transfer results
                    if source in results and target in results[source]:
                        if results_type == 'domain_adaptation':
                            transfer_matrix[i, j] = results[source][target]['domain_adversarial']
                        else:
                            transfer_matrix[i, j] = results[source][target][model_type]
        
        # Visualize
        plt.figure(figsize=(10, 8))
        im = plt.imshow(transfer_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
        
        plt.title(f'Transfer Learning Matrix: {results_type.replace("_", " ").title()} ({model_type})')
        plt.xlabel('Target Dataset')
        plt.ylabel('Source Dataset')
        
        plt.xticks(range(n_datasets), dataset_names, rotation=45)
        plt.yticks(range(n_datasets), dataset_names)
        
        # Add text annotations
        for i in range(n_datasets):
            for j in range(n_datasets):
                text = plt.text(j, i, f'{transfer_matrix[i, j]:.3f}',
                               ha='center', va='center', 
                               color='white' if transfer_matrix[i, j] < 0.5 else 'black')
        
        plt.colorbar(im, label='Accuracy')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{results_type}_{model_type}_matrix.png')
        plt.show()
    
    def analyze_embedding_spaces(self) -> None:
        """Analyze and visualize embedding spaces across domains."""
        print("Analyzing embedding spaces...")
        
        # Combine embeddings from all datasets
        all_embeddings = []
        all_labels = []
        all_datasets = []
        
        for dataset_name, embeddings in self.embeddings.items():
            # Use a subset for visualization
            n_samples = min(200, len(embeddings['train_X']))
            indices = np.random.choice(len(embeddings['train_X']), n_samples, replace=False)
            
            all_embeddings.append(embeddings['train_X'][indices])
            all_labels.extend(embeddings['train_y'][indices])
            all_datasets.extend([dataset_name] * n_samples)
        
        all_embeddings = np.vstack(all_embeddings)
        all_labels = np.array(all_labels)
        all_datasets = np.array(all_datasets)
        
        # Dimensionality reduction
        print("Performing PCA...")
        pca = PCA(n_components=2, random_state=42)
        embeddings_pca = pca.fit_transform(all_embeddings)
        
        print("Performing t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_tsne = tsne.fit_transform(all_embeddings[:1000])  # Limit for t-SNE
        
        # Visualize
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # PCA by dataset
        unique_datasets = np.unique(all_datasets)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_datasets)))
        
        for i, dataset in enumerate(unique_datasets):
            mask = all_datasets == dataset
            axes[0, 0].scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1], 
                              c=[colors[i]], label=dataset, alpha=0.6)
        
        axes[0, 0].set_title('PCA: Embeddings by Dataset')
        axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        axes[0, 0].legend()
        
        # PCA by label
        unique_labels = np.unique(all_labels)
        label_colors = ['red', 'blue']
        
        for i, label in enumerate(unique_labels):
            mask = all_labels == label
            axes[0, 1].scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1], 
                              c=label_colors[i], label=f'Label {label}', alpha=0.6)
        
        axes[0, 1].set_title('PCA: Embeddings by Label')
        axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        axes[0, 1].legend()
        
        # t-SNE by dataset (limited samples)
        tsne_datasets = all_datasets[:1000]
        tsne_labels = all_labels[:1000]
        
        for i, dataset in enumerate(unique_datasets):
            mask = tsne_datasets == dataset
            if np.any(mask):
                axes[1, 0].scatter(embeddings_tsne[mask, 0], embeddings_tsne[mask, 1], 
                                  c=[colors[i]], label=dataset, alpha=0.6)
        
        axes[1, 0].set_title('t-SNE: Embeddings by Dataset')
        axes[1, 0].set_xlabel('t-SNE 1')
        axes[1, 0].set_ylabel('t-SNE 2')
        axes[1, 0].legend()
        
        # t-SNE by label
        for i, label in enumerate(unique_labels):
            mask = tsne_labels == label
            if np.any(mask):
                axes[1, 1].scatter(embeddings_tsne[mask, 0], embeddings_tsne[mask, 1], 
                                  c=label_colors[i], label=f'Label {label}', alpha=0.6)
        
        axes[1, 1].set_title('t-SNE: Embeddings by Label')
        axes[1, 1].set_xlabel('t-SNE 1')
        axes[1, 1].set_ylabel('t-SNE 2')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'embedding_space_analysis.png')
        plt.show()
        
        print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive transfer learning report."""
        print("Generating comprehensive transfer learning report...")
        
        report = {
            'datasets': list(self.datasets.keys()),
            'embedding_type': self.embedding_type,
            'num_datasets': len(self.datasets),
            'results_summary': {}
        }
        
        # Baseline performance summary
        if 'baseline' in self.results:
            baseline_results = self.results['baseline']
            avg_baseline = {}
            
            for model_type in ['logistic', 'random_forest', 'svm']:
                scores = [results[model_type] for results in baseline_results.values() if model_type in results]
                if scores:
                    avg_baseline[model_type] = {
                        'mean': np.mean(scores),
                        'std': np.std(scores),
                        'min': np.min(scores),
                        'max': np.max(scores)
                    }
            
            report['results_summary']['baseline'] = avg_baseline
        
        # Cross-domain performance summary
        if 'cross_domain' in self.results:
            cross_results = self.results['cross_domain']
            avg_cross = {}
            
            for model_type in ['logistic', 'random_forest', 'svm']:
                scores = []
                for source_results in cross_results.values():
                    for target_results in source_results.values():
                        if model_type in target_results:
                            scores.append(target_results[model_type])
                
                if scores:
                    avg_cross[model_type] = {
                        'mean': np.mean(scores),
                        'std': np.std(scores),
                        'min': np.min(scores),
                        'max': np.max(scores)
                    }
            
            report['results_summary']['cross_domain'] = avg_cross
        
        # Calculate transfer gaps
        if 'baseline' in report['results_summary'] and 'cross_domain' in report['results_summary']:
            transfer_gaps = {}
            
            for model_type in ['logistic', 'random_forest', 'svm']:
                if (model_type in report['results_summary']['baseline'] and 
                    model_type in report['results_summary']['cross_domain']):
                    
                    baseline_mean = report['results_summary']['baseline'][model_type]['mean']
                    cross_mean = report['results_summary']['cross_domain'][model_type]['mean']
                    transfer_gaps[model_type] = baseline_mean - cross_mean
            
            report['results_summary']['transfer_gaps'] = transfer_gaps
        
        # Save report
        report_df = pd.DataFrame({
            'Metric': ['Datasets', 'Embedding Type', 'Baseline Accuracy (LR)', 'Cross-Domain Accuracy (LR)', 'Transfer Gap (LR)'],
            'Value': [
                len(self.datasets),
                self.embedding_type,
                f"{report['results_summary'].get('baseline', {}).get('logistic', {}).get('mean', 0):.4f}",
                f"{report['results_summary'].get('cross_domain', {}).get('logistic', {}).get('mean', 0):.4f}",
                f"{report['results_summary'].get('transfer_gaps', {}).get('logistic', 0):.4f}"
            ]
        })
        
        report_df.to_csv(self.output_dir / 'transfer_learning_report.csv', index=False)
        
        print("\n" + "="*60)
        print("TRANSFER LEARNING REPORT SUMMARY")
        print("="*60)
        print(report_df.to_string(index=False))
        
        return report
    
    def run_complete_evaluation(self, dataset_names: List[str], sample_size: int = 1000) -> Dict[str, Any]:
        """Run complete transfer learning evaluation pipeline."""
        print("🚀 Starting complete transfer learning evaluation...")
        
        try:
            # Load datasets
            self.load_datasets(dataset_names, sample_size)
            
            # Create embeddings
            self.create_embeddings()
            
            # Run evaluations
            self.baseline_evaluation()
            self.cross_domain_evaluation()
            self.domain_adaptation_evaluation()
            
            # Visualizations
            self.visualize_transfer_matrix('cross_domain', 'logistic')
            self.visualize_transfer_matrix('domain_adaptation', 'logistic')
            self.analyze_embedding_spaces()
            
            # Generate report
            report = self.generate_comprehensive_report()
            
            print("\n" + "="*60)
            print("🎉 TRANSFER LEARNING EVALUATION COMPLETED!")
            print("="*60)
            print(f"Results saved to: {self.output_dir}")
            print("\nKey Findings:")
            
            if 'transfer_gaps' in report['results_summary']:
                for model, gap in report['results_summary']['transfer_gaps'].items():
                    print(f"• {model.title()} transfer gap: {gap:.4f}")
            
            print("\nNext Steps:")
            print("• Analyze the generated visualizations")
            print("• Experiment with different embedding models")
            print("• Try advanced domain adaptation techniques")
            print("• Investigate domain-specific features")
            
            return report
            
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return {}


def main():
    """Main function to run transfer learning experiments."""
    # Initialize framework
    framework = TransferLearningFramework(embedding_type='sentence_transformer')
    
    # Define datasets to use
    dataset_names = ['imdb', 'ag_news', 'jigsaw_toxic']  # Adjust based on available datasets
    
    # Run complete evaluation
    report = framework.run_complete_evaluation(dataset_names, sample_size=500)
    
    return framework, report


if __name__ == "__main__":
    framework, report = main()