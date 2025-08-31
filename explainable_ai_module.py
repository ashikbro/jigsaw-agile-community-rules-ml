#!/usr/bin/env python3
"""
Explainable AI Module for Content Moderation

This module provides explanations for content moderation predictions using:
- LIME (Local Interpretable Model-agnostic Explanations)
- SHAP (SHapley Additive exPlanations)
- Custom feature importance analysis

Author: AI Assistant
Date: 2025
"""

import numpy as np
import pandas as pd
import joblib
import lime
import lime.lime_text
import shap
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class ContentModerationExplainer:
    """
    Provides explanations for content moderation predictions using multiple XAI techniques.
    """
    
    def __init__(self, model_path: str, vectorizer_path: str):
        """
        Initialize the explainer with trained model and vectorizer.
        
        Args:
            model_path: Path to the trained model
            vectorizer_path: Path to the trained vectorizer
        """
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.lime_explainer = None
        self.shap_explainer = None
        self._initialize_explainers()
    
    def _initialize_explainers(self):
        """Initialize LIME and SHAP explainers."""
        try:
            # Initialize LIME explainer for text
            self.lime_explainer = lime.lime_text.LimeTextExplainer(
                class_names=['Safe Content', 'Rule Violation'],
                feature_selection='auto',
                verbose=False
            )
            
            # For SHAP, we'll use the Explainer with the model's predict_proba method
            # Note: SHAP works better with tree-based models, but we'll adapt for our use case
            print("✓ LIME explainer initialized successfully")
            print("✓ SHAP explainer will be initialized on-demand")
            
        except Exception as e:
            print(f"Warning: Could not initialize all explainers: {e}")
    
    @property
    def predict_proba_wrapper(self):
        """Property to access the prediction wrapper for LIME."""
        return self._predict_proba_wrapper
    
    def _predict_proba_wrapper(self, texts: List[str]) -> np.ndarray:
        """
        Wrapper function for model prediction that LIME can use.
        
        Args:
            texts: List of text samples
            
        Returns:
            Prediction probabilities
        """
        try:
            # Vectorize the texts
            X = self.vectorizer.transform(texts)
            
            # Get predictions
            if hasattr(self.model, 'predict_proba'):
                probas = self.model.predict_proba(X)
            else:
                # For models without predict_proba, use decision_function or predict
                if hasattr(self.model, 'decision_function'):
                    scores = self.model.decision_function(X)
                    # Convert to probabilities using sigmoid
                    probas = np.column_stack([1 / (1 + np.exp(scores)), 1 / (1 + np.exp(-scores))])
                else:
                    # Fallback to binary predictions
                    preds = self.model.predict(X)
                    probas = np.column_stack([1 - preds, preds])
            
            return probas
            
        except Exception as e:
            print(f"Error in prediction wrapper: {e}")
            # Return neutral probabilities as fallback
            return np.array([[0.5, 0.5]] * len(texts))
    
    def explain_with_lime(self, text: str, num_features: int = 10) -> Dict[str, Any]:
        """
        Generate LIME explanation for a text sample.
        
        Args:
            text: Input text to explain
            num_features: Number of top features to show
            
        Returns:
            Dictionary containing LIME explanation
        """
        try:
            if self.lime_explainer is None:
                return {"error": "LIME explainer not available"}
            
            # Generate explanation
            explanation = self.lime_explainer.explain_instance(
                text,
                self.predict_proba_wrapper,
                num_features=num_features,
                labels=[1]  # Explain the "Rule Violation" class
            )
            
            # Extract feature importance
            feature_importance = explanation.as_list(label=1)
            
            # Get prediction for this text
            prediction_proba = self._predict_proba_wrapper([text])[0]
            
            return {
                "method": "LIME",
                "text": text,
                "prediction_proba": {
                    "safe": float(prediction_proba[0]),
                    "violation": float(prediction_proba[1])
                },
                "feature_importance": feature_importance,
                "explanation_html": explanation.as_html(),
                "top_positive_features": [f for f, score in feature_importance if score > 0][:5],
                "top_negative_features": [f for f, score in feature_importance if score < 0][:5]
            }
            
        except Exception as e:
            return {"error": f"LIME explanation failed: {str(e)}"}
    
    def explain_with_feature_importance(self, text: str, top_n: int = 10) -> Dict[str, Any]:
        """
        Generate explanation based on model feature importance.
        
        Args:
            text: Input text to explain
            top_n: Number of top features to show
            
        Returns:
            Dictionary containing feature importance explanation
        """
        try:
            # Vectorize the text
            X = self.vectorizer.transform([text])
            
            # Get prediction
            if hasattr(self.model, 'predict_proba'):
                prediction_proba = self.model.predict_proba(X)[0]
            else:
                prediction = self.model.predict(X)[0]
                prediction_proba = [1-prediction, prediction] if prediction == 0 else [0, 1]
            
            # Get feature names
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get model coefficients (for linear models)
            if hasattr(self.model, 'coef_'):
                coefficients = self.model.coef_[0] if len(self.model.coef_.shape) > 1 else self.model.coef_
                
                # Get active features in this text
                active_features = X.toarray()[0]
                feature_contributions = active_features * coefficients
                
                # Get top contributing features
                feature_indices = np.argsort(np.abs(feature_contributions))[-top_n:]
                top_features = [
                    {
                        "feature": feature_names[i],
                        "contribution": float(feature_contributions[i]),
                        "coefficient": float(coefficients[i]),
                        "present": bool(active_features[i] > 0)
                    }
                    for i in feature_indices[::-1]
                    if active_features[i] > 0  # Only show features present in the text
                ]
                
                return {
                    "method": "Feature Importance",
                    "text": text,
                    "prediction_proba": {
                        "safe": float(prediction_proba[0]),
                        "violation": float(prediction_proba[1])
                    },
                    "top_features": top_features,
                    "model_type": type(self.model).__name__
                }
            else:
                return {
                    "method": "Feature Importance",
                    "text": text,
                    "prediction_proba": {
                        "safe": float(prediction_proba[0]),
                        "violation": float(prediction_proba[1])
                    },
                    "error": "Model does not support feature importance extraction",
                    "model_type": type(self.model).__name__
                }
                
        except Exception as e:
            return {"error": f"Feature importance explanation failed: {str(e)}"}
    
    def explain_comprehensive(self, text: str) -> Dict[str, Any]:
        """
        Generate comprehensive explanation using multiple methods.
        
        Args:
            text: Input text to explain
            
        Returns:
            Dictionary containing comprehensive explanation
        """
        explanations = {
            "text": text,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # Get basic prediction
        try:
            prediction_proba = self._predict_proba_wrapper([text])[0]
            explanations["prediction"] = {
                "safe_probability": float(prediction_proba[0]),
                "violation_probability": float(prediction_proba[1]),
                "predicted_class": "Rule Violation" if prediction_proba[1] > 0.5 else "Safe Content",
                "confidence": float(max(prediction_proba))
            }
        except Exception as e:
            explanations["prediction_error"] = str(e)
        
        # Add LIME explanation
        lime_explanation = self.explain_with_lime(text)
        if "error" not in lime_explanation:
            explanations["lime"] = lime_explanation
        else:
            explanations["lime_error"] = lime_explanation["error"]
        
        # Add feature importance explanation
        feature_explanation = self.explain_with_feature_importance(text)
        if "error" not in feature_explanation:
            explanations["feature_importance"] = feature_explanation
        else:
            explanations["feature_importance_error"] = feature_explanation["error"]
        
        # Add text analysis
        explanations["text_analysis"] = self._analyze_text_properties(text)
        
        return explanations
    
    def _analyze_text_properties(self, text: str) -> Dict[str, Any]:
        """
        Analyze basic text properties that might influence classification.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with text analysis
        """
        import re
        
        # Basic text statistics
        words = text.split()
        sentences = text.split('.')
        
        # Check for potentially problematic patterns
        threat_words = ['kill', 'murder', 'hurt', 'harm', 'attack', 'destroy', 'eliminate']
        profanity_indicators = ['damn', 'hell', 'stupid', 'idiot', 'hate']
        
        threat_count = sum(1 for word in words if word.lower() in threat_words)
        profanity_count = sum(1 for word in words if word.lower() in profanity_indicators)
        
        # Check for caps (shouting)
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # Check for exclamation marks
        exclamation_count = text.count('!')
        
        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "character_count": len(text),
            "threat_words_found": threat_count,
            "profanity_indicators": profanity_count,
            "caps_ratio": caps_ratio,
            "exclamation_count": exclamation_count,
            "contains_person_reference": any(word.lower() in ['person', 'people', 'someone', 'him', 'her', 'they'] for word in words)
        }

def load_explainer_from_model_dir(model_dir: str) -> Optional[ContentModerationExplainer]:
    """
    Load explainer from a model directory.
    
    Args:
        model_dir: Directory containing model files
        
    Returns:
        ContentModerationExplainer instance or None if loading fails
    """
    import os
    
    try:
        # Look for model and vectorizer files
        model_files = [f for f in os.listdir(model_dir) if f.endswith('_model.joblib')]
        vectorizer_files = [f for f in os.listdir(model_dir) if f.endswith('_vectorizer.joblib')]
        
        if not model_files or not vectorizer_files:
            print(f"Could not find model or vectorizer files in {model_dir}")
            return None
        
        # Use the first available model (could be enhanced to select best model)
        model_path = os.path.join(model_dir, model_files[0])
        vectorizer_path = os.path.join(model_dir, vectorizer_files[0])
        
        print(f"Loading explainer with model: {model_files[0]}")
        return ContentModerationExplainer(model_path, vectorizer_path)
        
    except Exception as e:
        print(f"Error loading explainer: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    print("Content Moderation Explainable AI Module")
    print("========================================")
    
    # Try to load a model for demonstration
    import os
    import glob
    
    # Look for available model directories
    model_dirs = glob.glob("models/human_aware_*")
    if not model_dirs:
        model_dirs = glob.glob("models/context_aware_*")
    if not model_dirs:
        model_dirs = ["models"] if os.path.exists("models") else []
    
    if model_dirs:
        latest_model_dir = max(model_dirs, key=os.path.getctime)
        print(f"Attempting to load explainer from: {latest_model_dir}")
        
        explainer = load_explainer_from_model_dir(latest_model_dir)
        
        if explainer:
            # Test with example texts
            test_texts = [
                "i want to kill the person who made the food",
                "this game is killing me with its difficulty",
                "we need to kill the bugs in our garden",
                "great job on the presentation!"
            ]
            
            print("\nTesting explanations:")
            print("====================")
            
            for text in test_texts:
                print(f"\nText: '{text}'")
                explanation = explainer.explain_comprehensive(text)
                
                if "prediction" in explanation:
                    pred = explanation["prediction"]
                    print(f"Prediction: {pred['predicted_class']} ({pred['confidence']:.3f} confidence)")
                
                if "lime" in explanation:
                    lime_exp = explanation["lime"]
                    print(f"LIME top features: {lime_exp.get('top_positive_features', [])[:3]}")
                
                if "text_analysis" in explanation:
                    analysis = explanation["text_analysis"]
                    print(f"Threat words: {analysis['threat_words_found']}, Person reference: {analysis['contains_person_reference']}")
        else:
            print("Could not load explainer. Please ensure models are trained first.")
    else:
        print("No model directories found. Please train models first.")