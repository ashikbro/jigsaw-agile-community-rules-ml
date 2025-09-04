import joblib
import os
import glob

# Find models that actually work properly
print("FINDING WORKING MODELS")
print("="*50)

working_models = []
test_texts = [
    "This is great!",
    "You are stupid", 
    "Hello world",
    "I hate you",
    "Thank you"
]

# Test individual models with their correct vectorizers
model_vectorizer_pairs = [
    ("models/logistic_regression_model.joblib", "models/logistic_regression_vectorizer.joblib"),
    ("models/naive_bayes_model.joblib", "models/naive_bayes_vectorizer.joblib"),
    ("models/random_forest_model.joblib", "models/random_forest_vectorizer.joblib"),
    ("models/svm_model.joblib", "models/svm_vectorizer.joblib")
]

for model_file, vectorizer_file in model_vectorizer_pairs:
    if os.path.exists(model_file) and os.path.exists(vectorizer_file):
        print(f"\nTesting: {os.path.basename(model_file)}")
        try:
            model = joblib.load(model_file)
            vectorizer = joblib.load(vectorizer_file)
            
            predictions = []
            all_same = True
            first_prob = None
            
            for text in test_texts:
                try:
                    vectorized = vectorizer.transform([text])
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(vectorized)[0]
                        prob_violation = proba[1] if len(proba) > 1 else proba[0]
                        predictions.append(prob_violation)
                        
                        if first_prob is None:
                            first_prob = prob_violation
                        elif abs(prob_violation - first_prob) > 0.01:  # Allow small numerical differences
                            all_same = False
                            
                        print(f"  '{text}' -> {prob_violation:.3f}")
                    else:
                        pred = model.predict(vectorized)[0]
                        predictions.append(pred)
                        print(f"  '{text}' -> {pred}")
                except Exception as e:
                    print(f"  Error with '{text}': {e}")
                    break
            else:
                # If we got here, all texts were processed successfully
                if not all_same:
                    working_models.append((model_file, vectorizer_file))
                    print(f"  ✓ WORKING - Predictions vary appropriately")
                else:
                    print(f"  ✗ BROKEN - All predictions are the same ({first_prob:.3f})")
                    
        except Exception as e:
            print(f"  Error loading: {e}")
    else:
        print(f"\nSkipping {model_file} - files not found")

# Test multilingual pipeline models
print("\n" + "="*50)
print("TESTING MULTILINGUAL PIPELINE MODELS")
print("="*50)

multilingual_models = glob.glob("models/*_multilingual.joblib")
for model_file in multilingual_models:
    print(f"\nTesting: {os.path.basename(model_file)}")
    try:
        model = joblib.load(model_file)
        
        if hasattr(model, 'steps'):
            predictions = []
            all_same = True
            first_prob = None
            
            for text in test_texts:
                try:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba([text])[0]
                        prob_violation = proba[1] if len(proba) > 1 else proba[0]
                        predictions.append(prob_violation)
                        
                        if first_prob is None:
                            first_prob = prob_violation
                        elif abs(prob_violation - first_prob) > 0.01:
                            all_same = False
                            
                        print(f"  '{text}' -> {prob_violation:.3f}")
                    else:
                        pred = model.predict([text])[0]
                        predictions.append(pred)
                        print(f"  '{text}' -> {pred}")
                except Exception as e:
                    print(f"  Error with '{text}': {e}")
                    break
            else:
                if not all_same:
                    working_models.append((model_file, None))  # Pipeline doesn't need separate vectorizer
                    print(f"  ✓ WORKING - Predictions vary appropriately")
                else:
                    print(f"  ✗ BROKEN - All predictions are the same ({first_prob:.3f})")
        else:
            print(f"  Not a pipeline model")
            
    except Exception as e:
        print(f"  Error loading: {e}")

print("\n" + "="*50)
print("SUMMARY OF WORKING MODELS")
print("="*50)

if working_models:
    for model_file, vectorizer_file in working_models:
        model_name = os.path.basename(model_file).replace('.joblib', '')
        if vectorizer_file:
            print(f"✓ {model_name} (with separate vectorizer)")
        else:
            print(f"✓ {model_name} (pipeline)")
else:
    print("No working models found!")

print(f"\nTotal working models: {len(working_models)}")