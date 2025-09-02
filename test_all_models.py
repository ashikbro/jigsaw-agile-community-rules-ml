import joblib
import os
import glob

# Test all available models to see which ones work properly
model_files = glob.glob("models/*.joblib")
print("Available model files:")
for f in model_files:
    print(f"  {f}")

print("\n" + "="*60)
print("TESTING ALL MODELS")
print("="*60)

test_texts = [
    "This is great!",
    "You are stupid",
    "Hello world"
]

for model_file in model_files:
    if "vectorizer" in model_file:
        continue
        
    print(f"\nTesting model: {model_file}")
    try:
        model = joblib.load(model_file)
        print(f"  Model type: {type(model)}")
        print(f"  Is pipeline: {hasattr(model, 'steps')}")
        
        if hasattr(model, 'steps'):
            print(f"  Pipeline steps: {[step[0] for step in model.steps]}")
            
            # Test predictions
            for text in test_texts:
                try:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba([text])[0]
                        print(f"    '{text}' -> {proba[1]:.3f}")
                    else:
                        pred = model.predict([text])[0]
                        print(f"    '{text}' -> {pred}")
                except Exception as e:
                    print(f"    Error with '{text}': {e}")
        else:
            # Non-pipeline model, need vectorizer
            vectorizer_file = model_file.replace(".joblib", "_vectorizer.joblib")
            if not os.path.exists(vectorizer_file):
                vectorizer_file = "models/vectorizer.joblib"
            
            if os.path.exists(vectorizer_file):
                try:
                    vectorizer = joblib.load(vectorizer_file)
                    print(f"  Using vectorizer: {vectorizer_file}")
                    
                    for text in test_texts:
                        try:
                            vectorized = vectorizer.transform([text])
                            if hasattr(model, 'predict_proba'):
                                proba = model.predict_proba(vectorized)[0]
                                print(f"    '{text}' -> {proba[1]:.3f}")
                            else:
                                pred = model.predict(vectorized)[0]
                                print(f"    '{text}' -> {pred}")
                        except Exception as e:
                            print(f"    Error with '{text}': {e}")
                except Exception as e:
                    print(f"  Error loading vectorizer: {e}")
            else:
                print(f"  No vectorizer found for {model_file}")
                
    except Exception as e:
        print(f"  Error loading model: {e}")

# Test the regular (non-multilingual) random forest
print("\n" + "="*60)
print("TESTING REGULAR RANDOM FOREST")
print("="*60)

try:
    model = joblib.load("models/random_forest_model.joblib")
    vectorizer = joblib.load("models/random_forest_vectorizer.joblib")
    
    print(f"Model type: {type(model)}")
    print(f"Vectorizer type: {type(vectorizer)}")
    
    for text in test_texts:
        try:
            vectorized = vectorizer.transform([text])
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(vectorized)[0]
                print(f"'{text}' -> {proba[1]:.3f}")
            else:
                pred = model.predict(vectorized)[0]
                print(f"'{text}' -> {pred}")
        except Exception as e:
            print(f"Error with '{text}': {e}")
except Exception as e:
    print(f"Error testing regular random forest: {e}")