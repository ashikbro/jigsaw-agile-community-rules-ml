#!/usr/bin/env python3
"""
Complete Model Training Script
Trains the advanced ML pipeline with comprehensive datasets
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from datetime import datetime

# Import our modules
from research_pipeline import ResearchMLPipeline

def create_comprehensive_dataset(size='large'):
    """
    Create a comprehensive dataset for training
    """
    print(f"Creating {size} dataset...")
    
    # Base examples with various rule violations
    base_examples = [
        # Safe comments
        ("I disagree with your opinion but respect your right to have it", 0),
        ("That's an interesting perspective, thanks for sharing", 0),
        ("I would like to kill some time before the movie starts", 0),
        ("This is a great discussion, thank you for the insights", 0),
        ("I appreciate your thoughtful response to this topic", 0),
        ("Could you please provide more information about this?", 0),
        ("I think we should consider different viewpoints here", 0),
        ("This article raises some important questions", 0),
        ("Thank you for sharing your experience with us", 0),
        ("I learned something new from your comment today", 0),
        
        # Rule violations - harassment/toxicity
        ("You're such an idiot, go kill yourself", 1),
        ("I hate you so much, you should die", 1),
        ("Stupid moron, nobody likes you", 1),
        ("You're a pathetic loser and everyone knows it", 1),
        ("Shut up you worthless piece of garbage", 1),
        ("I hope you get cancer and die slowly", 1),
        ("You're so ugly, no wonder you're alone", 1),
        ("Kill yourself, the world would be better", 1),
        ("You're a waste of oxygen, just disappear", 1),
        ("I wish someone would beat you up", 1),
        
        # Rule violations - threats
        ("I'm going to find you and hurt you", 1),
        ("Watch your back, I know where you live", 1),
        ("I'll make sure you regret this comment", 1),
        ("You better watch out, I'm coming for you", 1),
        ("I'm going to destroy your life", 1),
        
        # Rule violations - hate speech
        ("All people from that country are terrorists", 1),
        ("Those people don't deserve to live here", 1),
        ("I hate all members of that religion", 1),
        ("People like you are ruining our society", 1),
        ("Your kind should be banned from this place", 1),
        
        # Rule violations - spam/off-topic
        ("BUY CHEAP VIAGRA NOW!!! CLICK HERE!!!", 1),
        ("Make $5000 a day working from home!!!", 1),
        ("Hot singles in your area want to meet you", 1),
        ("URGENT: Your account will be deleted unless you click here", 1),
        ("Free money! No questions asked! Click now!", 1),
    ]
    
    # Expand dataset based on size (optimized for memory)
    if size == 'small':
        multiplier = 5
    elif size == 'medium':
        multiplier = 15
    else:  # large
        multiplier = 25
    
    # Generate variations
    expanded_data = []
    for text, label in base_examples:
        for i in range(multiplier):
            # Add some variations to make dataset more diverse
            variations = [
                text,
                text.lower(),
                text.upper(),
                text + " " + str(i),
                f"Comment #{i}: {text}",
                f"{text} (posted at {i}:00)",
            ]
            
            for variation in variations:
                expanded_data.append((variation, label))
    
    # Create DataFrame
    df = pd.DataFrame(expanded_data, columns=['comment_text', 'target'])
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Dataset created with {len(df)} samples")
    print(f"Safe comments: {len(df[df['target'] == 0])}")
    print(f"Violations: {len(df[df['target'] == 1])}")
    
    return df

def train_complete_model(dataset_size='large'):
    """
    Train the complete model with comprehensive configuration
    """
    print("=" * 60)
    print("COMPLETE MODEL TRAINING")
    print("=" * 60)
    
    # Create comprehensive dataset
    df = create_comprehensive_dataset(dataset_size)
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
    
    print(f"\nTraining set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    # Prepare data
    texts_train = train_df['comment_text'].tolist()
    y_train = train_df['target'].values
    texts_test = test_df['comment_text'].tolist()
    y_test = test_df['target'].values
    
    # Complete configuration for production
    config = {
        'feature_extraction': {
            'use_hybrid_features': True,
            'use_domain_features': True,
            'use_topic_modeling': True,
            'tfidf_params': {
                'max_features': 5000,
                'ngram_range': (1, 2),
                'min_df': 3,
                'max_df': 0.9
            },
            'topic_modeling_params': {
                'n_topics_lda': 20,
                'n_topics_nmf': 15
            }
        },
        'ensemble_methods': {
            'use_diverse_ensemble': True,
            'use_adaptive_ensemble': True,
            'use_hierarchical_ensemble': True,
            'voting_weights': 'uniform'
        },
        'imbalance_handling': {
            'strategy': 'combined',
            'sampling_method': 'smote',
            'cost_sensitive': True,
            'threshold_optimization': True
        },
        'meta_learning': {
            'adversarial_validation': True,
            'pseudo_labeling': True,
            'unsupervised_features': True
        },
        'explainability': {
            'methods': ['rule_based', 'lime', 'shap'],
            'generate_reports': True
        },
        'evaluation': {
            'cv_folds': 5,
            'metrics': ['f1', 'precision', 'recall', 'roc_auc', 'accuracy'],
            'detailed_analysis': True
        }
    }
    
    # Initialize and train pipeline
    print("\nInitializing Research ML Pipeline...")
    pipeline = ResearchMLPipeline(config=config, random_state=42)
    
    print("\nStarting comprehensive training...")
    results = pipeline.run_comprehensive_experiment(
        X_train=texts_train,
        y_train=y_train,
        X_test=texts_test,
        y_test=y_test,
        texts_train=texts_train,
        texts_test=texts_test,
        experiment_name="complete_production_model"
    )
    
    # Save the trained model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/best_model.joblib'
    
    print(f"\nSaving model to {model_path}...")
    joblib.dump(pipeline, model_path)
    
    # Save results
    results_path = f'models/training_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.joblib'
    joblib.dump(results, results_path)
    
    print(f"\nTraining completed successfully!")
    print(f"Model saved to: {model_path}")
    print(f"Results saved to: {results_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Dataset size: {len(df)} samples")
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Features extracted: Multiple feature types combined")
    print(f"Models trained: Advanced ensemble methods")
    print(f"Configuration: Complete production setup")
    
    return pipeline, results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train complete model')
    parser.add_argument('--size', choices=['small', 'medium', 'large'], 
                       default='large', help='Dataset size')
    
    args = parser.parse_args()
    
    try:
        pipeline, results = train_complete_model(args.size)
        print("\n✅ Complete model training finished successfully!")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()