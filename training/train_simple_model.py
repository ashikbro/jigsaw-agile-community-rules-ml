from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Create a simple dataset
texts = [
    "This is a positive message",
    "This is a negative message",
    "Great news everyone!",
    "This is terrible news",
    "I love this project",
    "I don't like this at all"
]
labels = [1, 0, 1, 0, 1, 0]  # 1 for positive, 0 for negative

# Create and save the vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train a simple model
model = LogisticRegression()
model.fit(X, labels)

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the model and vectorizer
joblib.dump(model, "models/simple_model.joblib")
joblib.dump(vectorizer, "models/vectorizer.joblib")

print("Model and vectorizer saved successfully!")
