# Explainable AI Implementation Guide

This document outlines how to implement Explainable AI (XAI) techniques in the Jigsaw Agile Community Rules ML project, based on the concepts covered in CSE 445.

## 1. Introduction to Explainable AI

Explainable AI refers to methods and techniques that make the results of machine learning solutions understandable to humans. For text classification tasks like community rule violation detection, XAI helps stakeholders understand why certain comments are flagged as violations, which is crucial for:

- Building user trust in the system
- Identifying and mitigating biases
- Providing actionable feedback to users
- Meeting regulatory requirements
- Improving model performance through better understanding

## 2. XAI Techniques for Text Classification

### 2.1 Feature Importance Methods

#### 2.1.1 Model-Specific Feature Importance

```python
def extract_feature_importance(model, vectorizer, top_n=20):
    """Extract feature importance from a trained model."""
    if hasattr(model, 'coef_'):  # Linear models like Logistic Regression
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Get coefficients
        coefficients = model.coef_[0]
        
        # Create DataFrame for visualization
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': coefficients
        })
        
        # Sort by absolute importance
        feature_importance['Abs_Importance'] = np.abs(feature_importance['Importance'])
        feature_importance = feature_importance.sort_values('Abs_Importance', ascending=False)
        
    elif hasattr(model, 'feature_importances_'):  # Tree-based models like Random Forest
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Create DataFrame for visualization
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    return feature_importance.head(top_n)
```

#### 2.1.2 Visualizing Feature Importance

```python
def visualize_feature_importance(feature_importance, model_name, is_linear=True):
    """Visualize feature importance."""
    plt.figure(figsize=(12, 8))
    
    if is_linear:  # For linear models with positive/negative coefficients
        # Top positive
        plt.subplot(1, 2, 1)
        top_positive = feature_importance[feature_importance['Importance'] > 0].head(15)
        sns.barplot(x='Importance', y='Feature', data=top_positive)
        plt.title(f'Top 15 Features for Violation ({model_name})')
        plt.xlabel('Coefficient Value')
        
        # Top negative
        plt.subplot(1, 2, 2)
        top_negative = feature_importance[feature_importance['Importance'] < 0].head(15)
        sns.barplot(x='Importance', y='Feature', data=top_negative)
        plt.title(f'Top 15 Features for Non-Violation ({model_name})')
        plt.xlabel('Coefficient Value')
    else:  # For tree-based models with only positive importance
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
        plt.title(f'Top 20 Features by Importance ({model_name})')
        plt.xlabel('Importance')
    
    plt.tight_layout()
    plt.savefig(f'outputs/feature_importance_{model_name}.png')
    plt.show()
```

### 2.2 LIME (Local Interpretable Model-agnostic Explanations)

LIME explains individual predictions by approximating the model locally with an interpretable model.

#### 2.2.1 LIME Implementation for Text

```python
from lime.lime_text import LimeTextExplainer

def explain_prediction_with_lime(model, vectorizer, text, class_names=['Not Violation', 'Violation']):
    """Explain a single prediction using LIME."""
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
    
    # Visualize explanation
    explanation.show_in_notebook(text=True)
    
    # Save explanation to file
    fig = explanation.as_pyplot_figure()
    fig.savefig('outputs/lime_explanation.png')
    
    return explanation
```

### 2.3 SHAP (SHapley Additive exPlanations)

SHAP assigns each feature an importance value for a particular prediction based on game theory concepts.

#### 2.3.1 SHAP Implementation for Text

```python
import shap

def explain_prediction_with_shap(model, vectorizer, text, background_texts=None):
    """Explain a single prediction using SHAP."""
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
    shap.plots.text(shap_values)
    
    # Save visualization
    plt.savefig('outputs/shap_explanation.png')
    
    return shap_values
```

### 2.4 Decision Boundary Visualization

For text data, we can use dimensionality reduction to visualize decision boundaries.

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualize_decision_boundary(model, X, y, method='pca'):
    """Visualize decision boundary using dimensionality reduction."""
    # Reduce to 2 dimensions
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")
    
    # Transform data
    X_2d = reducer.fit_transform(X.toarray())
    
    # Create a meshgrid for decision boundary
    h = 0.02  # Step size
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Train a new model on the 2D data
    model_2d = type(model)(**model.get_params())
    model_2d.fit(X_2d, y)
    
    # Predict on the meshgrid
    Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    plt.title(f'Decision Boundary Visualization using {method.upper()}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar()
    plt.savefig(f'outputs/decision_boundary_{method}.png')
    plt.show()
```

## 3. Integration with the ML Pipeline

### 3.1 XAI Module Implementation

Create a new file `src/explainable_ai.py` with the following structure:

```python
"""
Explainable AI module for the Jigsaw Agile Community Rules ML project.

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

# Feature importance functions
def extract_feature_importance(model, vectorizer, top_n=20):
    """Extract feature importance from a trained model."""
    # Implementation as shown above
    pass

def visualize_feature_importance(feature_importance, model_name, is_linear=True):
    """Visualize feature importance."""
    # Implementation as shown above
    pass

# LIME functions
def explain_prediction_with_lime(model, vectorizer, text, class_names=['Not Violation', 'Violation']):
    """Explain a single prediction using LIME."""
    if not has_xai_libs:
        print("LIME not available. Install with: pip install lime")
        return None
    
    # Implementation as shown above
    pass

# SHAP functions
def explain_prediction_with_shap(model, vectorizer, text, background_texts=None):
    """Explain a single prediction using SHAP."""
    if not has_xai_libs:
        print("SHAP not available. Install with: pip install shap")
        return None
    
    # Implementation as shown above
    pass

# Decision boundary visualization
def visualize_decision_boundary(model, X, y, method='pca'):
    """Visualize decision boundary using dimensionality reduction."""
    # Implementation as shown above
    pass

# Main explanation function
def explain_model(model, vectorizer, X, y, texts, sample_indices=None):
    """Generate comprehensive explanations for a model."""
    # If no sample indices provided, randomly select some
    if sample_indices is None:
        sample_indices = np.random.choice(len(texts), min(5, len(texts)), replace=False)
    
    results = {}
    
    # 1. Feature importance
    feature_importance = extract_feature_importance(model, vectorizer)
    is_linear = hasattr(model, 'coef_')
    visualize_feature_importance(feature_importance, type(model).__name__, is_linear)
    results['feature_importance'] = feature_importance
    
    # 2. Sample explanations
    sample_explanations = []
    for idx in sample_indices:
        text = texts[idx]
        true_label = y[idx]
        
        explanation = {
            'text': text,
            'true_label': true_label
        }
        
        # LIME explanation
        if has_xai_libs:
            lime_exp = explain_prediction_with_lime(model, vectorizer, text)
            explanation['lime'] = lime_exp
        
        # SHAP explanation
        if has_xai_libs:
            shap_exp = explain_prediction_with_shap(model, vectorizer, text)
            explanation['shap'] = shap_exp
        
        sample_explanations.append(explanation)
    
    results['sample_explanations'] = sample_explanations
    
    # 3. Decision boundary
    visualize_decision_boundary(model, X, y, method='pca')
    visualize_decision_boundary(model, X, y, method='tsne')
    
    return results
```

### 3.2 Integration with Prediction Script

Update `src/predict_improved.py` to include explanations:

```python
# Add to imports
from src.explainable_ai import explain_prediction_with_lime, explain_prediction_with_shap

# Add to prediction function
def predict_with_explanation(text, model, vectorizer):
    """Predict with explanation."""
    # Make prediction
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)[0]
    probability = model.predict_proba(vectorized_text)[0][1]
    
    # Generate explanations
    lime_explanation = explain_prediction_with_lime(model, vectorizer, text)
    shap_explanation = explain_prediction_with_shap(model, vectorizer, text)
    
    return {
        'text': text,
        'prediction': int(prediction),
        'probability': float(probability),
        'lime_explanation': lime_explanation,
        'shap_explanation': shap_explanation
    }
```

### 3.3 Integration with API

Update `src/api.py` to include an explanation endpoint:

```python
# Add to imports
from src.explainable_ai import explain_prediction_with_lime

# Add new endpoint
@app.post("/explain/")
async def explain_prediction(request: PredictionRequest):
    """Explain a prediction for the given text."""
    try:
        # Make prediction
        vectorized_text = vectorizer.transform([request.text])
        prediction = model.predict(vectorized_text)[0]
        probability = model.predict_proba(vectorized_text)[0][1]
        
        # Generate explanation
        explanation = explain_prediction_with_lime(model, vectorizer, request.text)
        
        # Convert explanation to JSON-serializable format
        explanation_data = {
            'text': request.text,
            'prediction': int(prediction),
            'probability': float(probability),
            'explanation': {
                'intercept': explanation.intercept[1],
                'features': []
            }
        }
        
        # Add feature contributions
        for feature, weight in explanation.as_list(label=1):
            explanation_data['explanation']['features'].append({
                'feature': feature,
                'weight': weight
            })
        
        return explanation_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## 4. Academic Experiments with XAI

### 4.1 Comparing Model Interpretability

Create a notebook `notebooks/model_interpretability_comparison.ipynb` to compare the interpretability of different models:

```python
# Compare feature importance across models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='linear', probability=True, random_state=42)
}

for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Extract and visualize feature importance
    feature_importance = extract_feature_importance(model, vectorizer)
    visualize_feature_importance(
        feature_importance, 
        name, 
        is_linear=(name != 'Random Forest')
    )
```

### 4.2 Analyzing Misclassifications

```python
# Find misclassified examples
y_pred = model.predict(X_test)
misclassified_indices = np.where(y_pred != y_test)[0]

# Explain misclassifications
for idx in misclassified_indices[:5]:  # Analyze first 5 misclassifications
    text = X_test_raw[idx]
    true_label = y_test[idx]
    pred_label = y_pred[idx]
    
    print(f"Text: {text}")
    print(f"True label: {true_label}, Predicted label: {pred_label}")
    
    # Generate explanation
    explanation = explain_prediction_with_lime(model, vectorizer, text)
    # Analyze explanation
```

## 5. Academic Presentation of XAI Results

### 5.1 Visualization Templates

```python
def create_xai_summary_plot(model, vectorizer, texts, true_labels, pred_labels):
    """Create a summary plot for XAI presentation."""
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
    sample_text = texts[sample_idx]
    
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
    plt.savefig('outputs/xai_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
```

### 5.2 Academic Report Template

Create a template for reporting XAI findings in academic presentations:

```markdown
# Explainable AI Results

## 1. Model Interpretability Analysis

### 1.1 Feature Importance

[Insert feature importance visualization]

**Key Findings:**
- The most predictive features for rule violations are: [list top features]
- These align with community guidelines by focusing on [explain alignment]
- Potential bias concerns: [identify any concerning patterns]

### 1.2 Decision Boundary Analysis

[Insert decision boundary visualization]

**Key Findings:**
- The model effectively separates classes with [describe boundary characteristics]
- Areas of uncertainty include [identify regions with potential misclassifications]

## 2. Individual Prediction Explanations

### 2.1 LIME Analysis

[Insert LIME visualization]

**Key Findings:**
- For correctly classified examples, the model focuses on [describe patterns]
- For misclassified examples, the model is confused by [describe patterns]

### 2.2 SHAP Analysis

[Insert SHAP visualization]

**Key Findings:**
- Global feature impact shows [describe patterns]
- Individual predictions are most influenced by [describe patterns]

## 3. Academic Implications

- The model demonstrates [high/medium/low] interpretability
- Key challenges in explaining this text classification task include [list challenges]
- Potential improvements for interpretability: [list suggestions]
- Connection to XAI literature: [reference relevant papers/concepts]
```

## 6. Requirements and Dependencies

Add the following to `requirements.txt`:

```
lime>=0.2.0
shap>=0.41.0
mpl-scatter-density>=0.7
```

## 7. Conclusion

Implementing Explainable AI techniques in the Jigsaw Agile Community Rules ML project enhances its academic value by connecting to the XAI module in CSE 445. These implementations demonstrate practical applications of theoretical concepts and provide deeper insights into model behavior.

By following this guide, you can create comprehensive explanations for your text classification models, analyze their behavior, and present the results in an academically rigorous manner.