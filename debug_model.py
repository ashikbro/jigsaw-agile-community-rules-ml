#!/usr/bin/env python3
"""
Debug Model Performance
Directly test the loaded models to understand prediction behavior
"""

import joblib
import numpy as np
from pathlib import Path

def test_model_directly():
    """Test the model directly to understand its behavior"""
    print("=" * 60)
    print("DIRECT MODEL TESTING")
    print("=" * 60)
    
    # Load the multilingual random forest model
    model_path = Path("models/random_forest_multilingual.joblib")
    vectorizer_path = Path("models/random_forest_vectorizer.joblib")
    
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return
    
    if not vectorizer_path.exists():
        print(f"❌ Vectorizer file not found: {vectorizer_path}")
        return
    
    try:
        # Load model and vectorizer
        print("Loading model and vectorizer...")
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        print(f"✓ Model type: {type(model)}")
        print(f"✓ Vectorizer type: {type(vectorizer)}")
        print(f"✓ Model has predict_proba: {hasattr(model, 'predict_proba')}")
        
        # Test cases
        test_texts = [
            "You are an idiot!",  # Should be toxic
            "Thank you for your help.",  # Should be safe
            "This is a test message.",  # Should be safe
            "I hate you so much!",  # Should be toxic
            "The weather is nice today.",  # Should be safe
        ]
        
        print("\nTesting predictions:")
        print("-" * 40)
        
        for i, text in enumerate(test_texts, 1):
            print(f"\nTest {i}: {text}")
            
            # Vectorize text
            text_vectorized = vectorizer.transform([text])
            print(f"  Vectorized shape: {text_vectorized.shape}")
            print(f"  Vectorized type: {type(text_vectorized)}")
            
            # Make prediction
            prediction = model.predict(text_vectorized)[0]
            print(f"  Prediction: {prediction} ({'violation' if prediction == 1 else 'safe'})")
            
            # Get probabilities if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(text_vectorized)[0]
                print(f"  Probabilities: safe={probabilities[0]:.4f}, violation={probabilities[1]:.4f}")
                print(f"  Max probability: {max(probabilities):.4f}")
            else:
                print("  No predict_proba method available")
        
        # Test model properties
        print("\n" + "=" * 60)
        print("MODEL ANALYSIS")
        print("=" * 60)
        
        if hasattr(model, 'feature_importances_'):
            print(f"Feature importances shape: {model.feature_importances_.shape}")
            print(f"Top 5 feature importances: {sorted(model.feature_importances_, reverse=True)[:5]}")
        
        if hasattr(model, 'n_estimators'):
            print(f"Number of estimators: {model.n_estimators}")
        
        if hasattr(model, 'classes_'):
            print(f"Model classes: {model.classes_}")
        
        # Test with a batch of identical texts to see if results vary
        print("\nTesting consistency with identical inputs:")
        identical_texts = ["This is a test"] * 5
        vectorized_batch = vectorizer.transform(identical_texts)
        predictions_batch = model.predict(vectorized_batch)
        
        print(f"Identical text predictions: {predictions_batch}")
        print(f"All predictions same: {len(set(predictions_batch)) == 1}")
        
        if hasattr(model, 'predict_proba'):
            proba_batch = model.predict_proba(vectorized_batch)
            print(f"Probability variations: {np.std(proba_batch, axis=0)}")
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()

def check_training_data():
    """Check if we can find information about training data"""
    print("\n" + "=" * 60)
    print("TRAINING DATA ANALYSIS")
    print("=" * 60)
    
    # Look for training logs or data files
    data_paths = [
        Path("data"),
        Path("bengali_generated"),
        Path("cache"),
    ]
    
    for path in data_paths:
        if path.exists():
            print(f"\n📁 Found directory: {path}")
            try:
                items = list(path.iterdir())
                print(f"   Contains {len(items)} items")
                for item in items[:5]:  # Show first 5 items
                    print(f"   - {item.name}")
                if len(items) > 5:
                    print(f"   ... and {len(items) - 5} more")
            except Exception as e:
                print(f"   Error reading directory: {e}")
        else:
            print(f"❌ Directory not found: {path}")

if __name__ == "__main__":
    test_model_directly()
    check_training_data()