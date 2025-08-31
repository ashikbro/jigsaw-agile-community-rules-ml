"""Hands-on XAI Lab Exercises Implementation.

This module provides step-by-step implementations of Explainable AI techniques
for educational purposes, designed to give students practical experience
with feature importance, LIME, SHAP, and decision boundary visualization.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import project modules
from src.dataset_integration import DatasetManager
from src.explainable_ai import (
    extract_feature_importance,
    visualize_feature_importance,
    has_xai_libs
)

# Try to import XAI libraries
try:
    import lime
    from lime.lime_text import LimeTextExplainer
    import shap
    XAI_AVAILABLE = True
except ImportError:
    XAI_AVAILABLE = False
    print("Warning: LIME and SHAP libraries not available. Install with: pip install lime shap")


class XAILabExercises:
    """Class containing hands-on XAI lab exercises."""
    
    def __init__(self, dataset_name: str = 'imdb', sample_size: int = 1000):
        """Initialize the lab exercises.
        
        Args:
            dataset_name: Name of dataset to use for exercises
            sample_size: Number of samples to use
        """
        self.dataset_name = dataset_name
        self.sample_size = sample_size
        self.dataset_manager = DatasetManager()
        self.vectorizer = None
        self.models = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_vec = None
        self.X_test_vec = None
        
        # Create output directory
        self.output_dir = Path("outputs/xai_lab")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"XAI Lab Exercises initialized with {dataset_name} dataset")
        print(f"Output directory: {self.output_dir}")
    
    def exercise_1_data_loading_and_exploration(self) -> None:
        """Exercise 1: Load and explore the dataset."""
        print("\n" + "="*60)
        print("EXERCISE 1: DATA LOADING AND EXPLORATION")
        print("="*60)
        
        # Step 1: Load the dataset
        print("\nStep 1: Loading dataset...")
        train_df, test_df = self.dataset_manager.load_dataset(
            self.dataset_name, 
            sample_size=self.sample_size
        )
        
        # Step 2: Standardize the dataset
        print("\nStep 2: Standardizing dataset format...")
        train_df = self.dataset_manager.standardize_dataset(train_df, self.dataset_name)
        test_df = self.dataset_manager.standardize_dataset(test_df, self.dataset_name)
        
        # Step 3: Basic exploration
        print("\nStep 3: Basic dataset exploration")
        print(f"Training set shape: {train_df.shape}")
        print(f"Test set shape: {test_df.shape}")
        print(f"\nLabel distribution in training set:")
        print(train_df['label'].value_counts())
        
        # Step 4: Text length analysis
        print("\nStep 4: Text length analysis")
        train_df['text_length'] = train_df['text'].str.len()
        test_df['text_length'] = test_df['text'].str.len()
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        train_df['text_length'].hist(bins=30, alpha=0.7)
        plt.title('Distribution of Text Lengths')
        plt.xlabel('Text Length (characters)')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        for label in train_df['label'].unique():
            subset = train_df[train_df['label'] == label]
            plt.hist(subset['text_length'], bins=20, alpha=0.7, label=str(label))
        plt.title('Text Length by Label')
        plt.xlabel('Text Length (characters)')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'exercise1_text_length_analysis.png')
        plt.show()
        
        # Store data for later exercises
        self.X_train = train_df['text'].values
        self.X_test = test_df['text'].values
        self.y_train = train_df['label'].values
        self.y_test = test_df['label'].values
        
        print("\n✓ Exercise 1 completed! Data loaded and explored.")
        print(f"Sample text: '{self.X_train[0][:100]}...'")
        print(f"Sample label: {self.y_train[0]}")
    
    def exercise_2_text_preprocessing_and_vectorization(self) -> None:
        """Exercise 2: Text preprocessing and vectorization."""
        print("\n" + "="*60)
        print("EXERCISE 2: TEXT PREPROCESSING AND VECTORIZATION")
        print("="*60)
        
        # Step 1: Initialize TF-IDF vectorizer
        print("\nStep 1: Initializing TF-IDF vectorizer...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        # Step 2: Fit and transform the text data
        print("\nStep 2: Vectorizing text data...")
        self.X_train_vec = self.vectorizer.fit_transform(self.X_train)
        self.X_test_vec = self.vectorizer.transform(self.X_test)
        
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        print(f"Training matrix shape: {self.X_train_vec.shape}")
        print(f"Test matrix shape: {self.X_test_vec.shape}")
        
        # Step 3: Analyze feature names
        print("\nStep 3: Analyzing feature names...")
        feature_names = self.vectorizer.get_feature_names_out()
        print(f"Sample features: {feature_names[:20]}")
        
        # Step 4: Visualize TF-IDF statistics
        print("\nStep 4: TF-IDF statistics visualization")
        
        # Calculate feature statistics
        tfidf_sum = np.array(self.X_train_vec.sum(axis=0)).flatten()
        tfidf_mean = np.array(self.X_train_vec.mean(axis=0)).flatten()
        
        # Create feature statistics DataFrame
        feature_stats = pd.DataFrame({
            'feature': feature_names,
            'tfidf_sum': tfidf_sum,
            'tfidf_mean': tfidf_mean
        }).sort_values('tfidf_sum', ascending=False)
        
        # Plot top features
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        top_features = feature_stats.head(20)
        plt.barh(range(len(top_features)), top_features['tfidf_sum'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.title('Top 20 Features by TF-IDF Sum')
        plt.xlabel('TF-IDF Sum')
        
        plt.subplot(2, 2, 2)
        plt.hist(tfidf_sum, bins=50, alpha=0.7)
        plt.title('Distribution of TF-IDF Sums')
        plt.xlabel('TF-IDF Sum')
        plt.ylabel('Frequency')
        plt.yscale('log')
        
        plt.subplot(2, 2, 3)
        plt.scatter(tfidf_sum, tfidf_mean, alpha=0.5)
        plt.xlabel('TF-IDF Sum')
        plt.ylabel('TF-IDF Mean')
        plt.title('TF-IDF Sum vs Mean')
        
        plt.subplot(2, 2, 4)
        # Document frequency analysis
        doc_freq = np.array((self.X_train_vec > 0).sum(axis=0)).flatten()
        plt.hist(doc_freq, bins=50, alpha=0.7)
        plt.title('Document Frequency Distribution')
        plt.xlabel('Number of Documents')
        plt.ylabel('Number of Features')
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'exercise2_tfidf_analysis.png')
        plt.show()
        
        print("\n✓ Exercise 2 completed! Text data vectorized and analyzed.")
    
    def exercise_3_model_training_and_evaluation(self) -> None:
        """Exercise 3: Train multiple models and evaluate performance."""
        print("\n" + "="*60)
        print("EXERCISE 3: MODEL TRAINING AND EVALUATION")
        print("="*60)
        
        # Step 1: Initialize models
        print("\nStep 1: Initializing models...")
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='linear', probability=True, random_state=42),
            'Naive Bayes': MultinomialNB()
        }
        
        # Step 2: Train and evaluate models
        print("\nStep 2: Training and evaluating models...")
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(self.X_train_vec, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test_vec)
            
            # Calculate accuracy
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train_vec, self.y_train, cv=5)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        # Step 3: Visualize model comparison
        print("\nStep 3: Visualizing model comparison...")
        
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        cv_means = [results[name]['cv_mean'] for name in model_names]
        cv_stds = [results[name]['cv_std'] for name in model_names]
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.bar(model_names, accuracies)
        plt.title('Test Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 2)
        plt.errorbar(model_names, cv_means, yerr=cv_stds, fmt='o', capsize=5)
        plt.title('Cross-Validation Scores')
        plt.ylabel('CV Accuracy')
        plt.xticks(rotation=45)
        
        # Confusion matrices for best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
        best_predictions = results[best_model_name]['predictions']
        
        plt.subplot(2, 2, 3)
        cm = confusion_matrix(self.y_test, best_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.subplot(2, 2, 4)
        # Learning curve for best model
        best_model = results[best_model_name]['model']
        train_sizes, train_scores, val_scores = learning_curve(
            best_model, self.X_train_vec, self.y_train, cv=3, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Training')
        plt.plot(train_sizes, val_scores.mean(axis=1), 'o-', label='Validation')
        plt.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                         train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
        plt.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                         val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)
        plt.title(f'Learning Curve - {best_model_name}')
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'exercise3_model_comparison.png')
        plt.show()
        
        print(f"\n✓ Exercise 3 completed! Best model: {best_model_name} (Accuracy: {results[best_model_name]['accuracy']:.4f})")
    
    def exercise_4_feature_importance_analysis(self) -> None:
        """Exercise 4: Analyze feature importance for different models."""
        print("\n" + "="*60)
        print("EXERCISE 4: FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        # Step 1: Extract feature importance for linear model
        print("\nStep 1: Analyzing Logistic Regression feature importance...")
        lr_model = self.models['Logistic Regression']
        lr_importance = extract_feature_importance(lr_model, self.vectorizer)
        
        print("Top 10 positive features (indicating positive class):")
        print(lr_importance.head(10)[['Feature', 'Importance']])
        print("\nTop 10 negative features (indicating negative class):")
        print(lr_importance.tail(10)[['Feature', 'Importance']])
        
        # Step 2: Extract feature importance for tree-based model
        print("\nStep 2: Analyzing Random Forest feature importance...")
        rf_model = self.models['Random Forest']
        rf_importance = extract_feature_importance(rf_model, self.vectorizer)
        
        print("Top 10 most important features:")
        print(rf_importance.head(10)[['Feature', 'Importance']])
        
        # Step 3: Visualize and compare feature importance
        print("\nStep 3: Visualizing feature importance...")
        
        plt.figure(figsize=(15, 10))
        
        # Logistic Regression coefficients
        plt.subplot(2, 2, 1)
        top_lr = lr_importance.head(15)
        colors = ['red' if x < 0 else 'blue' for x in top_lr['Importance']]
        plt.barh(range(len(top_lr)), top_lr['Importance'], color=colors)
        plt.yticks(range(len(top_lr)), top_lr['Feature'])
        plt.title('Logistic Regression: Top 15 Features')
        plt.xlabel('Coefficient Value')
        
        plt.subplot(2, 2, 2)
        bottom_lr = lr_importance.tail(15)
        colors = ['red' if x < 0 else 'blue' for x in bottom_lr['Importance']]
        plt.barh(range(len(bottom_lr)), bottom_lr['Importance'], color=colors)
        plt.yticks(range(len(bottom_lr)), bottom_lr['Feature'])
        plt.title('Logistic Regression: Bottom 15 Features')
        plt.xlabel('Coefficient Value')
        
        # Random Forest importance
        plt.subplot(2, 2, 3)
        top_rf = rf_importance.head(20)
        plt.barh(range(len(top_rf)), top_rf['Importance'])
        plt.yticks(range(len(top_rf)), top_rf['Feature'])
        plt.title('Random Forest: Top 20 Features')
        plt.xlabel('Importance Value')
        
        # Feature importance comparison
        plt.subplot(2, 2, 4)
        # Find common features
        lr_features = set(lr_importance.head(50)['Feature'])
        rf_features = set(rf_importance.head(50)['Feature'])
        common_features = lr_features.intersection(rf_features)
        
        lr_common = lr_importance[lr_importance['Feature'].isin(common_features)].set_index('Feature')['Abs_Importance']
        rf_common = rf_importance[rf_importance['Feature'].isin(common_features)].set_index('Feature')['Importance']
        
        # Align the series
        common_df = pd.DataFrame({
            'LR_Importance': lr_common,
            'RF_Importance': rf_common
        }).fillna(0)
        
        plt.scatter(common_df['LR_Importance'], common_df['RF_Importance'], alpha=0.7)
        plt.xlabel('Logistic Regression |Importance|')
        plt.ylabel('Random Forest Importance')
        plt.title(f'Feature Importance Comparison\n({len(common_features)} common features)')
        
        # Add diagonal line
        max_val = max(common_df['LR_Importance'].max(), common_df['RF_Importance'].max())
        plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'exercise4_feature_importance.png')
        plt.show()
        
        # Step 4: Analyze feature importance insights
        print("\nStep 4: Feature importance insights")
        print(f"Number of common top features: {len(common_features)}")
        print(f"Correlation between importance scores: {common_df.corr().iloc[0, 1]:.4f}")
        
        print("\n✓ Exercise 4 completed! Feature importance analyzed and visualized.")
    
    def exercise_5_lime_explanations(self) -> None:
        """Exercise 5: Generate and analyze LIME explanations."""
        print("\n" + "="*60)
        print("EXERCISE 5: LIME EXPLANATIONS")
        print("="*60)
        
        if not XAI_AVAILABLE:
            print("❌ LIME not available. Please install: pip install lime")
            return
        
        # Step 1: Initialize LIME explainer
        print("\nStep 1: Initializing LIME explainer...")
        explainer = LimeTextExplainer(class_names=['negative', 'positive'])
        
        # Step 2: Select model for explanation
        model = self.models['Logistic Regression']
        
        # Create prediction function for LIME
        def predict_proba_fn(texts):
            vectors = self.vectorizer.transform(texts)
            return model.predict_proba(vectors)
        
        # Step 3: Generate explanations for sample texts
        print("\nStep 2: Generating LIME explanations...")
        
        # Select diverse examples
        sample_indices = [0, 10, 50, 100, 200]  # Different examples
        explanations = []
        
        plt.figure(figsize=(15, 12))
        
        for i, idx in enumerate(sample_indices):
            text = self.X_test[idx]
            true_label = self.y_test[idx]
            pred_label = model.predict(self.vectorizer.transform([text]))[0]
            pred_proba = model.predict_proba(self.vectorizer.transform([text]))[0]
            
            print(f"\nExample {i+1}:")
            print(f"Text: '{text[:100]}...'")
            print(f"True label: {true_label}, Predicted: {pred_label}")
            print(f"Prediction probability: {pred_proba}")
            
            # Generate LIME explanation
            explanation = explainer.explain_instance(
                text, predict_proba_fn, num_features=10
            )
            explanations.append(explanation)
            
            # Plot explanation
            plt.subplot(3, 2, i+1)
            exp_list = explanation.as_list()
            words = [item[0] for item in exp_list]
            scores = [item[1] for item in exp_list]
            
            colors = ['red' if score < 0 else 'green' for score in scores]
            plt.barh(range(len(words)), scores, color=colors)
            plt.yticks(range(len(words)), words)
            plt.title(f'LIME Explanation {i+1}\nTrue: {true_label}, Pred: {pred_label}')
            plt.xlabel('LIME Score')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'exercise5_lime_explanations.png')
        plt.show()
        
        # Step 4: Analyze explanation consistency
        print("\nStep 3: Analyzing explanation consistency...")
        
        # Collect all explanation words and scores
        all_explanations = []
        for explanation in explanations:
            all_explanations.extend(explanation.as_list())
        
        # Create DataFrame of explanations
        exp_df = pd.DataFrame(all_explanations, columns=['word', 'score'])
        word_stats = exp_df.groupby('word')['score'].agg(['mean', 'std', 'count']).reset_index()
        word_stats = word_stats[word_stats['count'] >= 2]  # Words appearing in multiple explanations
        word_stats = word_stats.sort_values('mean', key=abs, ascending=False)
        
        print("Most consistent explanation words:")
        print(word_stats.head(10))
        
        # Step 5: Save explanations
        print("\nStep 4: Saving LIME explanations...")
        for i, explanation in enumerate(explanations):
            explanation.save_to_file(str(self.output_dir / f'lime_explanation_{i+1}.html'))
        
        print("\n✓ Exercise 5 completed! LIME explanations generated and analyzed.")
    
    def exercise_6_shap_analysis(self) -> None:
        """Exercise 6: Generate and analyze SHAP explanations."""
        print("\n" + "="*60)
        print("EXERCISE 6: SHAP ANALYSIS")
        print("="*60)
        
        if not XAI_AVAILABLE:
            print("❌ SHAP not available. Please install: pip install shap")
            return
        
        # Step 1: Initialize SHAP explainer
        print("\nStep 1: Initializing SHAP explainer...")
        model = self.models['Logistic Regression']
        
        # Create SHAP explainer for linear model
        explainer = shap.LinearExplainer(model, self.X_train_vec)
        
        # Step 2: Calculate SHAP values for test samples
        print("\nStep 2: Calculating SHAP values...")
        sample_size = min(100, len(self.X_test))  # Limit for computational efficiency
        X_test_sample = self.X_test_vec[:sample_size]
        shap_values = explainer.shap_values(X_test_sample)
        
        # Step 3: Create SHAP summary plot
        print("\nStep 3: Creating SHAP summary plot...")
        feature_names = self.vectorizer.get_feature_names_out()
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'exercise6_shap_summary.png')
        plt.show()
        
        # Step 4: Individual SHAP explanations
        print("\nStep 4: Individual SHAP explanations...")
        
        # Select a few examples for detailed analysis
        example_indices = [0, 5, 10]
        
        for i, idx in enumerate(example_indices):
            plt.figure(figsize=(12, 6))
            
            # Force plot
            shap.force_plot(
                explainer.expected_value, 
                shap_values[idx], 
                X_test_sample[idx], 
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
            
            plt.title(f'SHAP Force Plot - Example {i+1}')
            plt.tight_layout()
            plt.savefig(self.output_dir / f'exercise6_shap_force_{i+1}.png')
            plt.show()
        
        # Step 5: Feature importance from SHAP
        print("\nStep 5: Global feature importance from SHAP...")
        
        # Calculate mean absolute SHAP values
        shap_importance = np.abs(shap_values).mean(axis=0)
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': shap_importance
        }).sort_values('shap_importance', ascending=False)
        
        # Plot top features
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(20)
        plt.barh(range(len(top_features)), top_features['shap_importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.title('Top 20 Features by SHAP Importance')
        plt.xlabel('Mean |SHAP Value|')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'exercise6_shap_importance.png')
        plt.show()
        
        print("Top 10 most important features by SHAP:")
        print(importance_df.head(10))
        
        print("\n✓ Exercise 6 completed! SHAP analysis performed and visualized.")
    
    def exercise_7_decision_boundary_visualization(self) -> None:
        """Exercise 7: Visualize decision boundaries using dimensionality reduction."""
        print("\n" + "="*60)
        print("EXERCISE 7: DECISION BOUNDARY VISUALIZATION")
        print("="*60)
        
        # Step 1: Prepare data for visualization
        print("\nStep 1: Preparing data for visualization...")
        
        # Use a subset for computational efficiency
        n_samples = min(500, len(self.X_test))
        X_subset = self.X_test_vec[:n_samples].toarray()
        y_subset = self.y_test[:n_samples]
        
        # Step 2: PCA visualization
        print("\nStep 2: PCA-based visualization...")
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_subset)
        
        # Get model predictions
        model = self.models['Logistic Regression']
        y_pred = model.predict(self.X_test_vec[:n_samples])
        
        plt.figure(figsize=(15, 5))
        
        # Plot 1: True labels
        plt.subplot(1, 3, 1)
        unique_labels = np.unique(y_subset)
        colors = ['red', 'blue']
        for i, label in enumerate(unique_labels):
            mask = y_subset == label
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=colors[i], label=f'True {label}', alpha=0.6)
        plt.title('PCA: True Labels')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.legend()
        
        # Plot 2: Predicted labels
        plt.subplot(1, 3, 2)
        for i, label in enumerate(unique_labels):
            mask = y_pred == label
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=colors[i], label=f'Pred {label}', alpha=0.6)
        plt.title('PCA: Predicted Labels')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.legend()
        
        # Plot 3: Correct vs incorrect predictions
        plt.subplot(1, 3, 3)
        correct = y_subset == y_pred
        plt.scatter(X_pca[correct, 0], X_pca[correct, 1], 
                   c='green', label='Correct', alpha=0.6)
        plt.scatter(X_pca[~correct, 0], X_pca[~correct, 1], 
                   c='red', label='Incorrect', alpha=0.6)
        plt.title('PCA: Prediction Accuracy')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'exercise7_pca_visualization.png')
        plt.show()
        
        # Step 3: t-SNE visualization
        print("\nStep 3: t-SNE visualization...")
        
        # Use even smaller subset for t-SNE (computationally expensive)
        n_tsne = min(300, n_samples)
        X_tsne_subset = X_subset[:n_tsne]
        y_tsne_subset = y_subset[:n_tsne]
        y_pred_tsne = y_pred[:n_tsne]
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_tsne_subset)
        
        plt.figure(figsize=(15, 5))
        
        # Plot 1: True labels
        plt.subplot(1, 3, 1)
        for i, label in enumerate(unique_labels):
            mask = y_tsne_subset == label
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                       c=colors[i], label=f'True {label}', alpha=0.6)
        plt.title('t-SNE: True Labels')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend()
        
        # Plot 2: Predicted labels
        plt.subplot(1, 3, 2)
        for i, label in enumerate(unique_labels):
            mask = y_pred_tsne == label
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                       c=colors[i], label=f'Pred {label}', alpha=0.6)
        plt.title('t-SNE: Predicted Labels')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend()
        
        # Plot 3: Correct vs incorrect predictions
        plt.subplot(1, 3, 3)
        correct_tsne = y_tsne_subset == y_pred_tsne
        plt.scatter(X_tsne[correct_tsne, 0], X_tsne[correct_tsne, 1], 
                   c='green', label='Correct', alpha=0.6)
        plt.scatter(X_tsne[~correct_tsne, 0], X_tsne[~correct_tsne, 1], 
                   c='red', label='Incorrect', alpha=0.6)
        plt.title('t-SNE: Prediction Accuracy')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'exercise7_tsne_visualization.png')
        plt.show()
        
        print(f"\nPCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
        print(f"Number of misclassified samples: {(~correct).sum()} / {len(correct)}")
        
        print("\n✓ Exercise 7 completed! Decision boundaries visualized using PCA and t-SNE.")
    
    def run_all_exercises(self) -> None:
        """Run all XAI lab exercises in sequence."""
        print("🚀 Starting XAI Lab Exercises")
        print(f"Dataset: {self.dataset_name}")
        print(f"Sample size: {self.sample_size}")
        print(f"XAI libraries available: {XAI_AVAILABLE}")
        
        try:
            self.exercise_1_data_loading_and_exploration()
            self.exercise_2_text_preprocessing_and_vectorization()
            self.exercise_3_model_training_and_evaluation()
            self.exercise_4_feature_importance_analysis()
            
            if XAI_AVAILABLE:
                self.exercise_5_lime_explanations()
                self.exercise_6_shap_analysis()
            else:
                print("\n⚠️ Skipping LIME and SHAP exercises (libraries not available)")
            
            self.exercise_7_decision_boundary_visualization()
            
            print("\n" + "="*60)
            print("🎉 ALL XAI LAB EXERCISES COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"\nResults saved to: {self.output_dir}")
            print("\nNext steps:")
            print("1. Review the generated visualizations")
            print("2. Analyze the model explanations")
            print("3. Try different datasets using DatasetManager")
            print("4. Experiment with different model parameters")
            print("5. Compare explanations across different models")
            
        except Exception as e:
            print(f"\n❌ Error during exercises: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function to run XAI lab exercises."""
    # You can change the dataset and sample size here
    lab = XAILabExercises(dataset_name='imdb', sample_size=1000)
    lab.run_all_exercises()


if __name__ == "__main__":
    main()