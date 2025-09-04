#!/usr/bin/env python3
"""
Fast Model Training Script
Quickly trains models on medium and large datasets without complex optimization loops
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
from datetime import datetime
import argparse

def create_dataset(size='medium'):
    """Create dataset with specified size"""
    print(f"Creating {size} dataset...")
    
    # Base examples
    safe_examples = [
        "I disagree with your opinion but respect your right to have it",
        "That's an interesting perspective, thanks for sharing",
        "I would like to kill some time before the movie starts",
        "This is a great discussion, thank you for the insights",
        "I appreciate your thoughtful response to this topic",
        "Could you please provide more information about this?",
        "I think we should consider different viewpoints here",
        "This article raises some important questions",
        "Thank you for sharing your experience with us",
        "I learned something new from your comment today",
        "Great point, I hadn't thought of it that way",
        "This is a constructive conversation",
        "I value your input on this matter",
        "Let's work together to find a solution",
        "Your expertise in this area is helpful"
    ]
    
    violation_examples = [
        "You're such an idiot, go kill yourself",
        "I hate you so much, you should die",
        "Stupid moron, nobody likes you",
        "You're a pathetic loser and everyone knows it",
        "Shut up you worthless piece of garbage",
        "I hope you get cancer and die slowly",
        "You're so ugly, no wonder you're alone",
        "Kill yourself, the world would be better",
        "You're a waste of oxygen, just disappear",
        "I wish you would just die already",
        "You're a disgusting piece of trash",
        "Nobody cares about your stupid opinion",
        "You should be ashamed of yourself",
        "Go jump off a bridge, loser",
        "You're the worst person I've ever met"
    ]
    
    # Determine dataset size
    if size == 'small':
        target_size = 1000
    elif size == 'medium':
        target_size = 5000
    else:  # large
        target_size = 20000
    
    # Generate dataset
    texts = []
    labels = []
    
    samples_per_class = target_size // 2
    
    # Generate safe examples
    for i in range(samples_per_class):
        base_text = safe_examples[i % len(safe_examples)]
        # Add some variation
        if i % 3 == 0:
            text = f"I think {base_text.lower()}"
        elif i % 3 == 1:
            text = f"{base_text} Really appreciate it."
        else:
            text = base_text
        texts.append(text)
        labels.append(0)
    
    # Generate violation examples
    for i in range(samples_per_class):
        base_text = violation_examples[i % len(violation_examples)]
        # Add some variation
        if i % 3 == 0:
            text = f"{base_text} Seriously."
        elif i % 3 == 1:
            text = f"Honestly, {base_text.lower()}"
        else:
            text = base_text
        texts.append(text)
        labels.append(1)
    
    df = pd.DataFrame({'text': texts, 'label': labels})
    print(f"Dataset created with {len(df)} samples ({sum(labels)} violations, {len(labels) - sum(labels)} safe)")
    
    return df

def train_fast_models(size='medium'):
    """Train models quickly without complex optimization"""
    print(f"\n=== Fast Training for {size.upper()} Dataset ===")
    
    # Create dataset
    df = create_dataset(size)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Vectorize text
    print("\nVectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train models
    models = {
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'svm': SVC(kernel='linear', random_state=42, probability=True),
        'naive_bayes': MultinomialNB()
    }
    
    results = {}
    trained_models = {}
    
    print("\nTraining models...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_vec, y_train)
        
        # Predict
        y_pred = model.predict(X_test_vec)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': classification_report(y_test, y_pred)
        }
        
        trained_models[name] = model
        
        print(f"  {name}: Accuracy={accuracy:.3f}, F1={f1:.3f}")
    
    # Save models and results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"models/{size}_dataset_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save vectorizer
    joblib.dump(vectorizer, f"{model_dir}/vectorizer.joblib")
    
    # Save models
    for name, model in trained_models.items():
        joblib.dump(model, f"{model_dir}/{name}.joblib")
    
    # Save results
    joblib.dump(results, f"{model_dir}/results.joblib")
    
    # Save dataset
    df.to_parquet(f"{model_dir}/dataset.parquet")
    
    print(f"\n✅ {size.upper()} dataset training completed successfully!")
    print(f"Models saved to: {model_dir}")
    print(f"Dataset size: {len(df)} samples")
    
    # Print best model
    best_model = max(results.keys(), key=lambda k: results[k]['f1_score'])
    print(f"Best model: {best_model} (F1: {results[best_model]['f1_score']:.3f})")
    
    return trained_models, results, model_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fast model training')
    parser.add_argument('--size', choices=['small', 'medium', 'large'], 
                       default='medium', help='Dataset size')
    parser.add_argument('--all', action='store_true', help='Train all sizes')
    
    args = parser.parse_args()
    
    try:
        if args.all:
            for size in ['medium', 'large']:
                train_fast_models(size)
                print("\n" + "="*60 + "\n")
        else:
            train_fast_models(args.size)
        
        print("\n🎉 All training completed successfully!")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()