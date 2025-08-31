"""
Baseline Model Training Script

This script trains a baseline model for the Jigsaw Agile Community Rules Classification task.
It uses a simple TF-IDF vectorizer with a logistic regression classifier.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve, auc
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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
    train_path = PROCESSED_DATA_DIR / 'train_processed.csv'
    if not train_path.exists():
        raise FileNotFoundError(f"Processed training data not found at {train_path}")
    
    df = pd.read_csv(train_path)
    # Filter out test data (if any)
    df = df[df['is_test'] == 0].copy()
    
    # Filter out rows with missing or empty cleaned_text
    df = df[df['cleaned_text'].notna() & (df['cleaned_text'].str.len() > 0)]
    
    return df

def preprocess_data(df):
    """
    Preprocess the data for modeling.
    
    Args:
        df: DataFrame containing the processed data
        
    Returns:
        X_train, X_val, y_train, y_val: Train/validation split of the data
    """
    # Split into features and target
    X = df['cleaned_text'].fillna('').astype(str)
    y = df['rule_violation']
    
    # Split into train and validation sets (80/20 split)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    return X_train, X_val, y_train, y_val

def train_baseline_model(X_train, y_train):
    """
    Train a baseline model using TF-IDF and Logistic Regression.
    
    Args:
        X_train: Training text data
        y_train: Training labels
        
    Returns:
        model: Trained model pipeline
        vectorizer: Fitted TF-IDF vectorizer
    """
    print("\nInitializing TF-IDF vectorizer...")
    # Initialize and fit the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=10000,  # Increased from 5000
        ngram_range=(1, 2),  # Use unigrams and bigrams
        stop_words='english',
        min_df=3,  # Increased from 2 to reduce vocabulary size
        max_df=0.85,  # Reduced from 0.9 to filter out more common terms
        sublinear_tf=True  # Use sublinear tf scaling
    )
    
    print("Transforming training data...")
    # Transform the training data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    print("Training logistic regression model...")
    # Initialize and train the logistic regression model with class weights
    model = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
        class_weight='balanced',  # Handle class imbalance
        solver='liblinear',  # Better for small datasets
        C=1.0  # Regularization strength
    )
    
    # Perform grid search for hyperparameter tuning
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'penalty': ['l1', 'l2']
    }
    
    print("Performing grid search for hyperparameter tuning...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='f1',  # Optimize for F1 score
        n_jobs=-1,  # Use all available cores
        verbose=1
    )
    
    grid_search.fit(X_train_tfidf, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
    
    return best_model, vectorizer

def evaluate_model(model, vectorizer, X_val, y_val):
    """
    Evaluate the model on the validation set.
    
    Args:
        model: Trained model
        vectorizer: Fitted TF-IDF vectorizer
        X_val: Validation text data
        y_val: True labels for validation set
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    print("\nEvaluating model on validation set...")
    # Transform the validation data
    X_val_tfidf = vectorizer.transform(X_val)
    
    # Make predictions
    y_pred = model.predict(X_val_tfidf)
    y_pred_proba = model.predict_proba(X_val_tfidf)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, zero_division=0),
        'recall': recall_score(y_val, y_pred, zero_division=0),
        'f1': f1_score(y_val, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_val, y_pred_proba) if len(set(y_val)) > 1 else 0.5
    }
    
    # Print metrics
    print("\n=== Model Evaluation Metrics ===")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    # Print classification report
    print("\n=== Classification Report ===")
    print(classification_report(y_val, y_pred, zero_division=0, target_names=['Not Violation', 'Violation']))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_val, y_pred)
    
    # Plot ROC curve
    plot_roc_curve(y_val, y_pred_proba)
    
    # Plot feature importance
    plot_feature_importance(model, vectorizer, top_n=20)
    
    return metrics

def plot_confusion_matrix(y_true, y_pred):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Violation', 'Violation'],
                yticklabels=['Not Violation', 'Violation'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save the plot
    plot_path = OUTPUT_DIR / 'confusion_matrix.png'
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"\nConfusion matrix saved to: {plot_path}")

def plot_roc_curve(y_true, y_pred_proba):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # Save the plot
    plot_path = OUTPUT_DIR / 'roc_curve.png'
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"ROC curve saved to: {plot_path}")

def plot_feature_importance(model, vectorizer, top_n=20):
    """Plot and save feature importance."""
    # Get feature importance (coefficients)
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    
    # Create a DataFrame of feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': coefs
    })
    
    # Sort by absolute importance
    feature_importance['abs_importance'] = np.abs(feature_importance['importance'])
    top_features = feature_importance.sort_values('abs_importance', ascending=False).head(top_n)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=top_features.sort_values('importance', ascending=False))
    plt.title(f'Top {top_n} Most Important Features')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    
    # Save the plot
    plot_path = OUTPUT_DIR / 'feature_importance.png'
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Feature importance plot saved to: {plot_path}")

def save_model(model, vectorizer, model_name='baseline_model'):
    """Save the trained model and vectorizer."""
    # Create models directory if it doesn't exist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save the model
    model_path = MODEL_DIR / f'{model_name}.joblib'
    joblib.dump(model, model_path)
    
    # Save the vectorizer
    vectorizer_path = MODEL_DIR / f'{model_name}_vectorizer.joblib'
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Vectorizer saved to: {vectorizer_path}")
    
    return model_path, vectorizer_path

def main():
    """Main function to run the baseline model training."""
    print("Starting baseline model training...")
    
    try:
        # Load the data
        print("\nLoading data...")
        df = load_data()
        print(f"Loaded {len(df)} training examples")
        
        # Check class distribution
        print("\nClass distribution:")
        print(df['rule_violation'].value_counts(normalize=True))
        
        # Preprocess the data
        print("\nPreprocessing data...")
        X_train, X_val, y_train, y_val = preprocess_data(df)
        print(f"Training set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")
        
        # Train the baseline model
        print("\nTraining baseline model...")
        model, vectorizer = train_baseline_model(X_train, y_train)
        
        # Evaluate the model
        print("\nEvaluating model...")
        metrics = evaluate_model(model, vectorizer, X_val, y_val)
        
        # Print metrics
        print("\n=== Model Performance ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Save the model
        print("\nSaving model...")
        model_path, vectorizer_path = save_model(model, vectorizer)
        
        print("\nBaseline model training completed successfully!")
        print(f"Model artifacts saved to: {MODEL_DIR}")
        
    except Exception as e:
        print(f"\nAn error occurred during model training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
