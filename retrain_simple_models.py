#!/usr/bin/env python3
"""
Simple Model Retraining Script
Retrains models with consistent TF-IDF vectorization to fix feature mismatch issues
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
from datetime import datetime
import os

def create_training_dataset():
    """
    Create a balanced training dataset with diverse examples
    """
    print("Creating training dataset...")
    
    # Safe/non-toxic examples
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
        "Your expertise in this area is helpful",
        "I understand your concern about this issue",
        "That makes sense from your perspective",
        "I'm glad we can have this discussion",
        "Thanks for clarifying that point",
        "I see what you mean now",
        "Good question, let me think about that",
        "I appreciate you taking the time to explain",
        "That's a valid concern to raise",
        "I hadn't considered that angle before",
        "Thanks for the helpful feedback"
    ]
    
    # Toxic/violation examples
    toxic_examples = [
        "You're such an idiot, go kill yourself",
        "I hate you so much, you should die",
        "Stupid moron, nobody likes you",
        "You're a pathetic loser and everyone knows it",
        "Shut up you worthless piece of garbage",
        "I hope you get cancer and die slowly",
        "You're so ugly, no wonder you're alone",
        "Kill yourself, the world would be better",
        "You're a waste of oxygen, just disappear",
        "I wish someone would beat you up",
        "I'm going to find you and hurt you",
        "Watch your back, I know where you live",
        "I'll make sure you regret this comment",
        "You better watch out, I'm coming for you",
        "I'm going to destroy your life",
        "All people from that country are terrorists",
        "Those people don't deserve to live here",
        "I hate all members of that religion",
        "People like you are ruining our society",
        "Your kind should be banned from this place",
        "You're a complete waste of space",
        "I hope bad things happen to you",
        "You deserve all the pain you get",
        "Nobody cares about your stupid opinion",
        "You're the worst person I've ever seen"
    ]
    
    # Create expanded dataset
    data = []
    
    # Add safe examples (label 0)
    for text in safe_examples:
        for i in range(40):  # 40 variations each
            variations = [
                text,
                text.lower(),
                text.upper(),
                f"{text} Thanks!",
                f"I think {text.lower()}",
                f"Actually, {text}",
                f"{text} What do you think?",
                f"Honestly, {text.lower()}"
            ]
            for variation in variations:
                data.append((variation, 0))
    
    # Add toxic examples (label 1)
    for text in toxic_examples:
        for i in range(40):  # 40 variations each
            variations = [
                text,
                text.lower(),
                text.upper(),
                f"{text}!!!",
                f"Seriously, {text.lower()}",
                f"{text} Idiot!",
                f"Listen here, {text.lower()}",
                f"{text} Moron!"
            ]
            for variation in variations:
                data.append((variation, 1))
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=['text', 'label'])
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Created dataset with {len(df)} samples")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    return df

def train_models():
    """
    Train models with consistent TF-IDF vectorization
    """
    print("\n=== RETRAINING MODELS WITH CONSISTENT VECTORIZATION ===")
    
    # Create dataset
    df = create_training_dataset()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], 
        test_size=0.2, 
        random_state=42, 
        stratify=df['label']
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Define consistent TF-IDF parameters
    tfidf_params = {
        'max_features': 5000,
        'ngram_range': (1, 2),
        'min_df': 2,
        'max_df': 0.95,
        'stop_words': 'english',
        'lowercase': True,
        'sublinear_tf': True
    }
    
    # Define models
    models = {
        'logistic_regression': LogisticRegression(
            random_state=42, 
            max_iter=1000,
            class_weight='balanced'
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            class_weight='balanced'
        ),
        'svm': SVC(
            kernel='linear', 
            random_state=42, 
            probability=True,
            class_weight='balanced'
        ),
        'naive_bayes': MultinomialNB(alpha=1.0)
    }
    
    results = {}
    
    # Train individual models (non-pipeline)
    print("\nTraining individual models...")
    vectorizer = TfidfVectorizer(**tfidf_params)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train_vec, y_train)
        
        # Predict
        y_pred = model.predict(X_test_vec)
        y_pred_proba = model.predict_proba(X_test_vec) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'model': model,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  F1 Score: {f1:.3f}")
        
        # Save individual model and vectorizer
        model_path = f'models/{name}_model.joblib'
        vectorizer_path = f'models/{name}_vectorizer.joblib'
        
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        
        print(f"  Saved: {model_path} and {vectorizer_path}")
    
    # Train multilingual pipeline models
    print("\nTraining multilingual pipeline models...")
    
    for name, model in models.items():
        print(f"\nTraining {name} pipeline...")
        
        # Create pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf_params)),
            ('classifier', model)
        ])
        
        # Train pipeline
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test) if hasattr(pipeline, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        pipeline_results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'pipeline': pipeline,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  F1 Score: {f1:.3f}")
        
        # Save pipeline
        pipeline_path = f'models/{name}_multilingual.joblib'
        joblib.dump(pipeline, pipeline_path)
        
        print(f"  Saved: {pipeline_path}")
        
        results[f'{name}_pipeline'] = pipeline_results
    
    # Save shared vectorizer
    joblib.dump(vectorizer, 'models/vectorizer.joblib')
    print(f"\nSaved shared vectorizer: models/vectorizer.joblib")
    
    return results

def test_models():
    """
    Test the retrained models to ensure they work properly
    """
    print("\n=== TESTING RETRAINED MODELS ===")
    
    test_texts = [
        "This is a great discussion, thank you!",
        "I appreciate your thoughtful response",
        "You're an idiot, go kill yourself",
        "I hate you so much, you should die",
        "That's an interesting perspective"
    ]
    
    expected_labels = [0, 0, 1, 1, 0]  # 0=safe, 1=toxic
    
    # Test individual models
    print("\nTesting individual models:")
    model_names = ['logistic_regression', 'random_forest', 'svm', 'naive_bayes']
    
    for name in model_names:
        try:
            model = joblib.load(f'models/{name}_model.joblib')
            vectorizer = joblib.load(f'models/{name}_vectorizer.joblib')
            
            X_test = vectorizer.transform(test_texts)
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            print(f"\n{name.upper()}:")
            for i, (text, pred, exp) in enumerate(zip(test_texts, predictions, expected_labels)):
                prob_str = f" (prob: {probabilities[i][1]:.3f})" if probabilities is not None else ""
                status = "✓" if pred == exp else "✗"
                print(f"  {status} '{text[:50]}...' -> {pred}{prob_str}")
                
        except Exception as e:
            print(f"  ✗ Error loading {name}: {e}")
    
    # Test pipeline models
    print("\nTesting pipeline models:")
    
    for name in model_names:
        try:
            pipeline = joblib.load(f'models/{name}_multilingual.joblib')
            
            predictions = pipeline.predict(test_texts)
            probabilities = pipeline.predict_proba(test_texts) if hasattr(pipeline, 'predict_proba') else None
            
            print(f"\n{name.upper()} PIPELINE:")
            for i, (text, pred, exp) in enumerate(zip(test_texts, predictions, expected_labels)):
                prob_str = f" (prob: {probabilities[i][1]:.3f})" if probabilities is not None else ""
                status = "✓" if pred == exp else "✗"
                print(f"  {status} '{text[:50]}...' -> {pred}{prob_str}")
                
        except Exception as e:
            print(f"  ✗ Error loading {name} pipeline: {e}")

if __name__ == "__main__":
    try:
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Train models
        results = train_models()
        
        # Test models
        test_models()
        
        print("\n✅ Model retraining completed successfully!")
        print("\nSummary:")
        print("- All models use consistent TF-IDF vectorization (5000 features)")
        print("- Individual models saved with separate vectorizers")
        print("- Pipeline models include vectorization internally")
        print("- Feature mismatch issue should be resolved")
        
    except Exception as e:
        print(f"\n❌ Retraining failed: {e}")
        import traceback
        traceback.print_exc()