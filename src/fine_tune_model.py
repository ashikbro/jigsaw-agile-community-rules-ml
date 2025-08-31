"""
Model Fine-tuning Script

This script fine-tunes the baseline model by:
1. Trying different hyperparameters
2. Using cross-validation
3. Experimenting with different feature extraction techniques
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Set up paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODEL_DIR = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'outputs'

# Create directories if they don't exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Set plot style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

def load_data():
    """Load the processed data."""
    data_path = PROCESSED_DATA_DIR / 'processed_data.csv'
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found at {data_path}")
    
    df = pd.read_csv(data_path)
    return df

def preprocess_data(df):
    """Preprocess the data for modeling."""
    # Split into features and target
    X = df['comment_text'].fillna('').astype(str)
    y = df['label']
    
    return X, y

def evaluate_model(model, X_test, y_test, model_name='model'):
    """Evaluate the model and return metrics."""
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba) if len(set(y_test)) > 1 else 0.5
    
    # Print classification report
    print(f"\n=== {model_name} Classification Report ===")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, model_name)
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Violation', 'Violation'],
                yticklabels=['Not Violation', 'Violation'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    
    # Save the plot
    plot_path = OUTPUT_DIR / f'confusion_matrix_{model_name}.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Confusion matrix saved to: {plot_path}")

def save_model(model, model_name):
    """Save the trained model."""
    model_path = MODEL_DIR / f'{model_name}.joblib'
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    return model_path

def fine_tune_logistic_regression(X, y):
    """Fine-tune a logistic regression model with different parameters."""
    print("\n=== Fine-tuning Logistic Regression ===")
    
    # Define the pipeline
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', LogisticRegression(random_state=RANDOM_STATE))
    ])
    
    # Define parameter grid
    param_grid = {
        'vectorizer__max_features': [1000, 5000, 10000],
        'vectorizer__ngram_range': [(1, 1), (1, 2)],
        'vectorizer__stop_words': [None, 'english'],
        'classifier__C': [0.1, 1, 10],
        'classifier__class_weight': [None, 'balanced']
    }
    
    # For small datasets, use a simple train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    
    # Manually try different parameter combinations
    best_score = -1
    best_params = {}
    
    # Simple parameter grid iteration
    from itertools import product
    
    # Get all parameter combinations
    param_names = param_grid.keys()
    param_values = [param_grid[name] for name in param_names]
    
    print("\nTrying parameter combinations...")
    for i, values in enumerate(product(*param_values)):
        params = dict(zip(param_names, values))
        
        # Set parameters
        pipeline.set_params(**params)
        
        # Fit and evaluate
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        score = f1_score(y_test, y_pred, zero_division=0)
        
        # Update best score
        if score > best_score:
            best_score = score
            best_params = params
            
        if (i + 1) % 5 == 0:
            print(f"Tried {i+1} combinations, best F1: {best_score:.4f}")
    
    # Train final model with best parameters on full data
    print("\nBest parameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    print(f"\nBest validation F1 score: {best_score:.4f}")
    
    # Train on full data with best parameters
    pipeline.set_params(**best_params)
    pipeline.fit(X, y)
    
    return pipeline
    
    # Fit the model
    grid_search.fit(X, y)
    
    # Print best parameters and score
    print("\nBest parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
    
    print(f"\nBest cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def try_different_models(X, y):
    """Try different models and return the best one."""
    print("\n=== Trying Different Models ===")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Define models to try
    models = [
        {
            'name': 'LogisticRegression_TFIDF',
            'pipeline': Pipeline([
                ('vectorizer', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ('classifier', LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced'))
            ])
        },
        {
            'name': 'LogisticRegression_Count',
            'pipeline': Pipeline([
                ('vectorizer', CountVectorizer(max_features=5000, ngram_range=(1, 2))),
                ('classifier', LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced'))
            ])
        },
        {
            'name': 'RandomForest',
            'pipeline': Pipeline([
                ('vectorizer', TfidfVectorizer(max_features=5000)),
                ('classifier', RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'))
            ])
        }
    ]
    
    best_score = -1
    best_model = None
    best_name = ""
    
    # Evaluate each model
    for model_info in models:
        name = model_info['name']
        model = model_info['pipeline']
        
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        metrics = evaluate_model(model, X_test, y_test, name)
        
        # Update best model
        if metrics['f1'] > best_score:
            best_score = metrics['f1']
            best_model = model
            best_name = name
    
    print(f"\nBest model: {best_name} with F1 score: {best_score:.4f}")
    return best_model, best_name

def main():
    """Main function to run the model fine-tuning."""
    print("Starting model fine-tuning...")
    
    try:
        # Load the data
        print("\nLoading data...")
        df = load_data()
        
        # Preprocess the data
        print("\nPreprocessing data...")
        X, y = preprocess_data(df)
        
        # Try different models
        print("\nTrying different models...")
        best_model, best_model_name = try_different_models(X, y)
        
        # Fine-tune the best model
        print("\nFine-tuning the best model...")
        if 'LogisticRegression' in best_model_name:
            best_model = fine_tune_logistic_regression(X, y)
        
        # Save the best model
        print("\nSaving the best model...")
        model_path = save_model(best_model, 'fine_tuned_model')
        
        print("\nModel fine-tuning completed successfully!")
        print(f"Best model saved to: {model_path}")
        
    except Exception as e:
        print(f"\nAn error occurred during model fine-tuning: {str(e)}")
        raise

if __name__ == "__main__":
    main()
