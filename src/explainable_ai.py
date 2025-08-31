"""Explainable AI module for the Jigsaw Agile Community Rules ML project.

This module provides tools for explaining model predictions using various XAI techniques.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import XAI libraries
try:
    from lime.lime_text import LimeTextExplainer
    import shap
    has_xai_libs = True
except ImportError:
    has_xai_libs = False
    print("Warning: XAI libraries not found. Install with: pip install lime shap")

# Set up paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / 'outputs' / 'explanations'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

 
def extract_feature_importance(model, vectorizer, top_n=50):
    """Extract feature importance from a trained model.
    
    Args:
        model: Trained model (linear or tree-based)
        vectorizer: Text vectorizer used for training
        top_n: Number of top features to return
        
    Returns:
        DataFrame with feature names and importance scores
    """
    try:
        # Check for tree-based models first (more specific)
        if hasattr(model, 'feature_importances_') and not hasattr(model, 'coef_'):
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Get feature importances
            importances = model.feature_importances_
            
            # Ensure lengths match
            min_length = min(len(feature_names), len(importances))
            feature_names = feature_names[:min_length]
            importances = importances[:min_length]
            
            # Create DataFrame for visualization
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            })
            
            # Sort by importance
            feature_importance = feature_importance.sort_values('Importance', ascending=False)
            
        elif hasattr(model, 'coef_'):  # Linear models like LogisticRegression, SVM
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Get coefficients - handle both 1D and 2D arrays
            coefficients = model.coef_
            if hasattr(coefficients, 'ndim') and coefficients.ndim > 1:
                coefficients = coefficients[0]
            
            # Ensure lengths match
            min_length = min(len(feature_names), len(coefficients))
            feature_names = feature_names[:min_length]
            coefficients = coefficients[:min_length]
            
            # Create DataFrame for visualization
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': coefficients
            })
            
            # Sort by absolute importance
            feature_importance['Abs_Importance'] = np.abs(feature_importance['Importance'])
            feature_importance = feature_importance.sort_values('Abs_Importance', ascending=False)
            
        # If model has feature_importances_ but also coef_ (shouldn't happen in practice)
        elif hasattr(model, 'feature_importances_'):
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Get feature importances
            importances = model.feature_importances_
            
            # Ensure lengths match
            min_length = min(len(feature_names), len(importances))
            feature_names = feature_names[:min_length]
            importances = importances[:min_length]
            
            # Create DataFrame for visualization
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            })
            
            # Sort by importance
            feature_importance = feature_importance.sort_values('Importance', ascending=False)
        else:
            # Return empty DataFrame if model doesn't have interpretable features
            return pd.DataFrame(columns=['Feature', 'Importance'])
        
        return feature_importance.head(top_n)
    
    except Exception as e:
        print(f"Error extracting feature importance: {e}")
        return pd.DataFrame(columns=['Feature', 'Importance'])


def visualize_feature_importance(feature_importance, model_name, is_linear=True, output_path=None):
    """Visualize feature importance.
    
    Args:
        feature_importance: DataFrame with feature importance
        model_name: Name of the model for plot title
        is_linear: Whether the model is linear (with positive/negative coefficients)
        output_path: Path to save the visualization (optional)
    """
    try:
        fig = plt.figure(figsize=(12, 8))
        
        if is_linear and len(feature_importance) > 0:  # For linear models with positive/negative coefficients
            # Top positive
            plt.subplot(1, 2, 1)
            top_positive = feature_importance[feature_importance['Importance'] > 0].head(15)
            if len(top_positive) > 0:
                # Sort by importance for better visualization
                top_positive = top_positive.sort_values('Importance', ascending=True)
                plt.barh(range(len(top_positive)), top_positive['Importance'])
                plt.yticks(range(len(top_positive)), top_positive['Feature'])
                plt.title(f'Top Features for Violation ({model_name})')
                plt.xlabel('Coefficient Value')
            
            # Top negative
            plt.subplot(1, 2, 2)
            top_negative = feature_importance[feature_importance['Importance'] < 0].head(15)
            if len(top_negative) > 0:
                # Sort by absolute importance for better visualization
                top_negative = top_negative.sort_values('Importance', ascending=False)
                plt.barh(range(len(top_negative)), top_negative['Importance'])
                plt.yticks(range(len(top_negative)), top_negative['Feature'])
                plt.title(f'Top Features for Non-Violation ({model_name})')
                plt.xlabel('Coefficient Value')
        else:  # For tree-based models with only positive importance
            if len(feature_importance) > 0:
                top_features = feature_importance.head(20)
                top_features = top_features.sort_values('Importance', ascending=True)
                plt.barh(range(len(top_features)), top_features['Importance'])
                plt.yticks(range(len(top_features)), top_features['Feature'])
                plt.title(f'Top Features by Importance ({model_name})')
                plt.xlabel('Importance')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
        else:
            plt.savefig(OUTPUT_DIR / f'feature_importance_{model_name}.png')
        
        return fig
    except Exception as e:
        print(f"Error in visualize_feature_importance: {e}")
        # Return a simple empty figure
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f'Visualization Error: {str(e)}', 
                ha='center', va='center', transform=ax.transAxes)
        return fig


def explain_prediction_with_lime(model, vectorizer, text, class_names=['Not Violation', 'Violation'], output_path=None):
    """Explain a single prediction using LIME.
    
    Args:
        model: Trained model
        vectorizer: Text vectorizer
        text: Text to explain
        class_names: Names of the classes
        output_path: Path to save the visualization (optional)
        
    Returns:
        LIME explanation object
    """
    if not has_xai_libs:
        print("LIME not available. Install with: pip install lime")
        return None
    
    # Create a prediction function that takes raw text
    def predict_proba(texts):
        # Vectorize the texts
        vectorized_texts = vectorizer.transform(texts)
        # Get prediction probabilities
        return model.predict_proba(vectorized_texts)
    
    # Create LIME explainer
    explainer = LimeTextExplainer(class_names=class_names)
    
    # Generate explanation
    explanation = explainer.explain_instance(
        text, 
        predict_proba, 
        num_features=10,
        top_labels=1
    )
    
    # Create visualization
    fig = explanation.as_pyplot_figure()
    
    if output_path:
        fig.savefig(output_path)
    else:
        fig.savefig(OUTPUT_DIR / f'lime_explanation_{model.__class__.__name__}.png')
    
    return explanation


def explain_prediction_with_shap(model, vectorizer, text, background_texts=None, output_path=None):
    """Explain a single prediction using SHAP.
    
    Args:
        model: Trained model
        vectorizer: Text vectorizer
        text: Text to explain
        background_texts: Background texts for SHAP explainer
        output_path: Path to save the visualization (optional)
        
    Returns:
        SHAP values
    """
    if not has_xai_libs:
        print("SHAP not available. Install with: pip install shap")
        return None
    
    # Create a prediction function
    def predict(texts):
        vectorized_texts = vectorizer.transform(texts)
        return model.predict_proba(vectorized_texts)[:, 1]
    
    # If no background texts provided, create a simple one
    if background_texts is None:
        background_texts = [""]
    
    # Create explainer
    explainer = shap.Explainer(predict, background_texts)
    
    # Calculate SHAP values
    shap_values = explainer([text])
    
    # Visualize
    plt.figure(figsize=(10, 6))
    shap.plots.text(shap_values)
    
    if output_path:
        plt.savefig(output_path)
    else:
        plt.savefig(OUTPUT_DIR / f'shap_explanation_{model.__class__.__name__}.png')
    
    return shap_values


def visualize_decision_boundary(model, X, y, method='pca', sample_size=1000, output_path=None):
    """Visualize decision boundary using dimensionality reduction.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: Target vector
        method: Dimensionality reduction method ('pca' or 'tsne')
        sample_size: Number of samples to use for visualization
        output_path: Path to save the visualization (optional)
    """
    # Sample data if it's too large
    if X.shape[0] > sample_size:
        indices = np.random.choice(X.shape[0], sample_size, replace=False)
        X_sample = X[indices]
        y_sample = y[indices]
    else:
        X_sample = X
        y_sample = y
    
    # Reduce to 2 dimensions
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")
    
    # Transform data
    X_2d = reducer.fit_transform(X_sample.toarray())
    
    # Create a meshgrid for decision boundary
    h = 0.02  # Step size
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Train a new model on the 2D data
    model_2d = type(model)(**model.get_params())
    model_2d.fit(X_2d, y_sample)
    
    # Predict on the meshgrid
    Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_sample, cmap='coolwarm', edgecolors='k')
    plt.title(f'Decision Boundary Visualization using {method.upper()} - {model.__class__.__name__}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar()
    
    if output_path:
        plt.savefig(output_path)
    else:
        plt.savefig(OUTPUT_DIR / f'decision_boundary_{method}_{model.__class__.__name__}.png')
    
    return plt.gcf()


def create_xai_summary_plot(model, vectorizer, texts, true_labels, pred_labels, model_name, output_path=None):
    """Create a summary plot for XAI presentation.
    
    Args:
        model: Trained model
        vectorizer: Text vectorizer
        texts: Text data
        true_labels: True labels
        pred_labels: Predicted labels
        model_name: Name of the model
        output_path: Path to save the visualization (optional)
    """
    if not has_xai_libs:
        print("XAI libraries not available. Install with: pip install lime shap")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Feature importance
    feature_importance = extract_feature_importance(model, vectorizer, top_n=10)
    is_linear = hasattr(model, 'coef_')
    
    if is_linear:
        # Top positive features
        top_positive = feature_importance[feature_importance['Importance'] > 0].head(10)
        sns.barplot(x='Importance', y='Feature', data=top_positive, ax=axes[0, 0])
        axes[0, 0].set_title('Top Features for Violation')
        
        # Top negative features
        top_negative = feature_importance[feature_importance['Importance'] < 0].head(10)
        sns.barplot(x='Importance', y='Feature', data=top_negative, ax=axes[0, 1])
        axes[0, 1].set_title('Top Features for Non-Violation')
    else:
        # Feature importance for tree-based models
        sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=axes[0, 0])
        axes[0, 0].set_title('Top Features by Importance')
        axes[0, 1].axis('off')
    
    # 2. Sample explanation
    sample_idx = np.random.choice(len(texts))
    sample_text = texts.iloc[sample_idx]
    
    # LIME explanation visualization (simplified for presentation)
    explainer = LimeTextExplainer(class_names=['Not Violation', 'Violation'])
    
    def predict_proba(texts):
        vectorized_texts = vectorizer.transform(texts)
        return model.predict_proba(vectorized_texts)
    
    exp = explainer.explain_instance(sample_text, predict_proba, num_features=6)
    exp.as_pyplot_figure(label=1, ax=axes[1, 0])
    axes[1, 0].set_title('LIME Explanation for Sample Text')
    
    # 3. Confusion matrix with explanation summary
    cm = confusion_matrix(true_labels, pred_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
                xticklabels=['Not Violation', 'Violation'],
                yticklabels=['Not Violation', 'Violation'])
    axes[1, 1].set_title('Confusion Matrix')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.savefig(OUTPUT_DIR / f'xai_summary_{model_name}.png', dpi=300, bbox_inches='tight')
    
    return fig


def analyze_misclassifications(model, vectorizer, texts, true_labels, pred_labels, n_samples=3):
    """Analyze misclassified examples using XAI techniques.
    
    Args:
        model: Trained model
        vectorizer: Text vectorizer
        texts: Text data
        true_labels: True labels
        pred_labels: Predicted labels
        n_samples: Number of misclassified samples to analyze
        
    Returns:
        List of dictionaries with misclassification analysis
    """
    if not has_xai_libs:
        print("XAI libraries not available. Install with: pip install lime shap")
        return None
    
    # Find misclassified examples
    misclassified_indices = np.where(pred_labels != true_labels)[0]
    
    if len(misclassified_indices) == 0:
        print("No misclassifications found.")
        return []
    
    # Limit to n_samples
    misclassified_indices = misclassified_indices[:n_samples]
    
    results = []
    
    for idx in misclassified_indices:
        text = texts.iloc[idx]
        true_label = true_labels.iloc[idx]
        pred_label = pred_labels[idx]
        
        # Generate LIME explanation
        lime_exp = explain_prediction_with_lime(
            model, 
            vectorizer, 
            text,
            output_path=OUTPUT_DIR / f'misclassified_{idx}_lime.png'
        )
        
        # Store results
        results.append({
            'index': idx,
            'text': text,
            'true_label': true_label,
            'predicted_label': pred_label,
            'lime_explanation': lime_exp
        })
    
    return results


def explain_model(model, vectorizer, X, y, texts, sample_indices=None):
    """Generate comprehensive explanations for a model.
    
    Args:
        model: Trained model
        vectorizer: Text vectorizer
        X: Feature matrix
        y: Target vector
        texts: Text data
        sample_indices: Indices of samples to explain (optional)
        
    Returns:
        Dictionary with explanation results
    """
    # If no sample indices provided, randomly select some
    if sample_indices is None:
        sample_indices = np.random.choice(len(texts), min(5, len(texts)), replace=False)
    
    results = {}
    
    # 1. Feature importance
    feature_importance = extract_feature_importance(model, vectorizer)
    is_linear = hasattr(model, 'coef_')
    visualize_feature_importance(
        feature_importance, 
        type(model).__name__, 
        is_linear,
        OUTPUT_DIR / f'feature_importance_{type(model).__name__}.png'
    )
    results['feature_importance'] = feature_importance
    
    # 2. Sample explanations
    sample_explanations = []
    for idx in sample_indices:
        text = texts.iloc[idx]
        true_label = y.iloc[idx]
        
        explanation = {
            'text': text,
            'true_label': true_label
        }
        
        # LIME explanation
        if has_xai_libs:
            lime_exp = explain_prediction_with_lime(
                model, 
                vectorizer, 
                text,
                output_path=OUTPUT_DIR / f'lime_explanation_sample_{idx}.png'
            )
            explanation['lime'] = lime_exp
        
        # SHAP explanation
        if has_xai_libs:
            shap_exp = explain_prediction_with_shap(
                model, 
                vectorizer, 
                text,
                output_path=OUTPUT_DIR / f'shap_explanation_sample_{idx}.png'
            )
            explanation['shap'] = shap_exp
        
        sample_explanations.append(explanation)
    
    results['sample_explanations'] = sample_explanations
    
    # 3. Decision boundary
    visualize_decision_boundary(
        model, 
        X, 
        y, 
        method='pca',
        output_path=OUTPUT_DIR / f'decision_boundary_pca_{type(model).__name__}.png'
    )
    
    # Only run t-SNE if the dataset is small enough (it's computationally expensive)
    if X.shape[0] < 5000:
        visualize_decision_boundary(
            model, 
            X, 
            y, 
            method='tsne',
            output_path=OUTPUT_DIR / f'decision_boundary_tsne_{type(model).__name__}.png'
        )
    
    return results


def predict_with_explanation(text, model, vectorizer):
    """Predict with explanation.
    
    Args:
        text: Text to predict
        model: Trained model
        vectorizer: Text vectorizer
        
    Returns:
        Dictionary with prediction and explanations
    """
    # Make prediction
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)[0]
    probability = model.predict_proba(vectorized_text)[0][1]
    
    # Generate explanations
    lime_explanation = None
    shap_explanation = None
    
    if has_xai_libs:
        lime_explanation = explain_prediction_with_lime(model, vectorizer, text)
        shap_explanation = explain_prediction_with_shap(model, vectorizer, text)
    
    return {
        'text': text,
        'prediction': int(prediction),
        'probability': float(probability),
        'lime_explanation': lime_explanation,
        'shap_explanation': shap_explanation
    }


if __name__ == "__main__":
    print("Explainable AI module loaded.")
    print(f"XAI libraries available: {has_xai_libs}")
    print(f"Output directory: {OUTPUT_DIR}")