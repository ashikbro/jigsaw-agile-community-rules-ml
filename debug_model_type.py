import joblib
import os

# Load the multilingual random forest model and vectorizer
model_path = "models/random_forest_multilingual.joblib"
vectorizer_path = "models/random_forest_vectorizer.joblib"

print("Loading model and vectorizer...")
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

print(f"Model type: {type(model)}")
print(f"Model attributes: {dir(model)}")
print(f"Vectorizer type: {type(vectorizer)}")
print(f"Vectorizer attributes: {dir(vectorizer)}")

# Check if model has any preprocessing methods
if hasattr(model, 'preprocess'):
    print("Model has preprocess method")
if hasattr(model, 'preprocessor'):
    print("Model has preprocessor attribute")
if hasattr(model, 'steps'):
    print(f"Model has steps (pipeline): {model.steps}")

# Check if vectorizer has any preprocessing methods
if hasattr(vectorizer, 'preprocess'):
    print("Vectorizer has preprocess method")
if hasattr(vectorizer, 'preprocessor'):
    print(f"Vectorizer has preprocessor attribute: {vectorizer.preprocessor}")

# Test with simple text
test_text = "This is a test"
print(f"\nTesting with: '{test_text}'")

try:
    # Transform text with vectorizer
    vectorized = vectorizer.transform([test_text])
    print(f"Vectorized shape: {vectorized.shape}")
    print(f"Vectorized type: {type(vectorized)}")
    
    # Try to predict
    prediction = model.predict(vectorized)
    print(f"Prediction: {prediction}")
    
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(vectorized)
        print(f"Probabilities: {proba}")
        
except Exception as e:
    print(f"Error during prediction: {e}")
    import traceback
    traceback.print_exc()