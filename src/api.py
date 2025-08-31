"""API for serving the trained model.

This module provides a FastAPI application for serving the trained model.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import uvicorn

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.predict_improved import TextClassifier

# Import explainable AI module
try:
    from src.explainable_ai import predict_with_explanation, has_xai_libs
except ImportError:
    has_xai_libs = False
    print("Warning: Explainable AI module not found. XAI features will be disabled.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set up paths
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / 'models'

# Initialize FastAPI app
app = FastAPI(
    title="Jigsaw Rule Violation Classifier API",
    description="API for classifying text according to community rule violations",
    version="0.1.0"
)

# Load the model
model_path = MODEL_DIR / 'improved_model.joblib'
if not model_path.exists():
    model_path = MODEL_DIR / 'baseline_model.joblib'

try:
    classifier = TextClassifier.load(model_path)
    logger.info(f"Loaded model from {model_path}")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    classifier = None

# Define request and response models
class PredictionRequest(BaseModel):
    """Model for prediction request."""
    text: str
    rule_text: str = None

class ExplanationRequest(BaseModel):
    """Model for explanation request."""
    text: str
    rule_text: str = None
    explanation_type: str = "lime"  # Options: "lime", "shap", "both"

class PredictionResponse(BaseModel):
    """Model for prediction response."""
    prediction: int
    probability: float
    text: str

@app.get("/")
def read_root():
    """Root endpoint."""
    return {"message": "Jigsaw Rule Violation Classifier API"}

@app.get("/health")
def health_check():
    """Health check endpoint."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Predict whether text violates a rule."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Combine text and rule_text if provided
        text = request.text
        if request.rule_text:
            text = f"Text: {request.text}\nRule: {request.rule_text}"
        
        # Get prediction
        probability = classifier.predict_proba(text)
        prediction = 1 if probability >= 0.5 else 0
        
        return PredictionResponse(
            prediction=prediction,
            probability=float(probability),
            text=request.text
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain")
def explain_prediction(request: ExplanationRequest):
    """Predict with explanation for the given text."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not has_xai_libs:
        raise HTTPException(
            status_code=501, 
            detail="Explainable AI libraries not available. Install with: pip install lime shap"
        )
    
    try:
        # Combine text and rule_text if provided
        text = request.text
        if request.rule_text:
            text = f"Text: {request.text}\nRule: {request.rule_text}"
        
        # Get prediction with explanation
        result = predict_with_explanation(text, classifier, request.explanation_type)
        
        # Add the original text to the response
        result["text"] = request.text
        
        return result
    except Exception as e:
        logger.error(f"Explanation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def start_server(host="0.0.0.0", port=8000):
    """Start the API server."""
    uvicorn.run("src.api:app", host=host, port=port, reload=True)

if __name__ == "__main__":
    start_server()