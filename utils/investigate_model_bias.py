import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Load the multilingual random forest model
model_path = "models/random_forest_multilingual.joblib"
print("Loading multilingual random forest model...")
model = joblib.load(model_path)

print(f"Model type: {type(model)}")
print(f"Is pipeline: {hasattr(model, 'steps')}")

if hasattr(model, 'steps'):
    print(f"Pipeline steps: {[step[0] for step in model.steps]}")
    
    # Get the classifier (last step)
    classifier = model.steps[-1][1]
    print(f"Classifier type: {type(classifier)}")
    
    if hasattr(classifier, 'n_estimators'):
        print(f"Number of estimators: {classifier.n_estimators}")
    if hasattr(classifier, 'classes_'):
        print(f"Classes: {classifier.classes_}")
    if hasattr(classifier, 'feature_importances_'):
        print(f"Feature importances shape: {classifier.feature_importances_.shape}")
        print(f"Top 10 feature importances: {sorted(classifier.feature_importances_, reverse=True)[:10]}")

# Test with various texts to see prediction patterns
test_texts = [
    "This is great!",
    "You are stupid",
    "Hello world",
    "I hate you",
    "Thank you very much",
    "Go kill yourself",
    "Nice weather today",
    "You're an idiot",
    "Have a good day",
    "Fuck off"
]

print("\n" + "="*60)
print("TESTING PREDICTION PATTERNS")
print("="*60)

for text in test_texts:
    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba([text])[0]
            prediction = model.predict([text])[0]
            print(f"Text: '{text}'")
            print(f"  Prediction: {prediction}")
            print(f"  Probabilities: {proba}")
            print(f"  Violation prob: {proba[1]:.3f}")
            print()
    except Exception as e:
        print(f"Error with text '{text}': {e}")

# Check if there's training data available to analyze
print("\n" + "="*60)
print("CHECKING TRAINING DATA")
print("="*60)

try:
    # Look for training data
    training_data_path = "data/multilingual_training_data.csv"
    df = pd.read_csv(training_data_path)
    print(f"Training data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check label distribution
    if 'label' in df.columns:
        label_dist = df['label'].value_counts()
        print(f"Label distribution: {label_dist}")
        print(f"Label proportions: {df['label'].value_counts(normalize=True)}")
    elif 'toxic' in df.columns:
        label_dist = df['toxic'].value_counts()
        print(f"Toxic distribution: {label_dist}")
        print(f"Toxic proportions: {df['toxic'].value_counts(normalize=True)}")
    
    print(f"Sample texts:")
    print(df.head())
    
except Exception as e:
    print(f"Could not load training data: {e}")

# Check model's default prediction behavior
print("\n" + "="*60)
print("ANALYZING MODEL BEHAVIOR")
print("="*60)

# Test with empty/minimal inputs
minimal_tests = [
    "",
    " ",
    "a",
    "the",
    "and",
    "."
]

for text in minimal_tests:
    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba([text])[0]
            print(f"'{text}' -> Violation prob: {proba[1]:.3f}")
    except Exception as e:
        print(f"Error with minimal text '{text}': {e}")