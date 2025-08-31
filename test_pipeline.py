#!/usr/bin/env python3
"""
Test script to verify the complete ML pipeline works correctly.
"""

import pandas as pd
from training_pipeline import MLTrainingPipeline
from model_evaluation import ModelEvaluator
import os

def test_complete_pipeline():
    """
    Test the complete ML pipeline with sample data.
    """
    print("Testing Complete ML Pipeline")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = MLTrainingPipeline()
    
    # Load and prepare data
    data_path = "data/raw/sample_data.csv"
    print(f"Loading data from {data_path}...")
    
    df = pipeline.load_data(data_path, text_column='comment_text', label_column='label')
    
    # Train the complete pipeline
    print("\nTraining models...")
    models_to_train = ['logistic_regression', 'multinomial_nb', 'svm_linear']
    
    feature_config = {
        'max_features_word': 1000,
        'max_features_char': 500,
        'word_ngram_range': (1, 2),
        'char_ngram_range': (3, 5),
        'feature_selection_k': 800
    }
    
    training_results = pipeline.train_pipeline(
        df=df,
        models_to_train=models_to_train,
        use_grid_search=False,  # Set to False for faster testing
        feature_config=feature_config
    )
    
    # Get test results from training history
    test_results = training_results['test_results']
    
    # Print results
    print("\nTest Results:")
    print("-" * 30)
    for model_name, metrics in test_results.items():
        print(f"{model_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  F1-Score: {metrics['f1_score']:.3f}")
        if metrics['auc_score'] is not None:
            print(f"  AUC: {metrics['auc_score']:.3f}")
        else:
            print(f"  AUC: N/A")
        print()
    
    # Test prediction on new text
    print("Testing predictions on new text:")
    print("-" * 30)
    
    test_texts = [
        "This is a great post, thanks for sharing!",
        "You're such an idiot, go kill yourself!",
        "I disagree with your opinion but respect your right to have it.",
        "This is completely stupid and worthless trash!"
    ]
    
    for text in test_texts:
        prediction = pipeline.predict_text(text)
        print(f"Text: '{text[:50]}...'")
        prob = prediction['probability']
        label = prediction['prediction']
        model = prediction['model_used']
        prob_str = f"{prob:.3f}" if prob is not None else "N/A"
        print(f"  {model}: {label} (prob: {prob_str})")
        print()
    
    # Save pipeline
    save_path = "models/test_pipeline"
    os.makedirs("models", exist_ok=True)
    pipeline.save_pipeline(save_path)
    print(f"Pipeline saved to {save_path}")
    
    print("\n✅ Pipeline test completed successfully!")
    return pipeline

if __name__ == "__main__":
    try:
        pipeline = test_complete_pipeline()
    except Exception as e:
        print(f"❌ Pipeline test failed: {str(e)}")
        import traceback
        traceback.print_exc()