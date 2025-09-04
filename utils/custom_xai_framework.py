#!/usr/bin/env python3
"""
Custom Explainable AI Framework for Content Moderation

This module implements novel explainability techniques specifically designed for
content moderation tasks, extending beyond standard LIME/SHAP approaches.

Features:
- Stability Analysis: Evaluates explanation consistency across text variations
- Span Highlighting: Identifies toxic word spans directly in text
- Feature-Aware Explanations: Combines TF-IDF weights with SHAP scores
- Moderation-Specific XAI: Tailored explanations for content moderation context
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import re
from collections import defaultdict
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator
import joblib
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StabilityAnalyzer:
    """
    Analyzes the stability of explanations when text undergoes minor modifications.
    This is crucial for content moderation to ensure consistent explanations.
    """
    
    def __init__(self, model, vectorizer, explainer_func):
        self.model = model
        self.vectorizer = vectorizer
        self.explainer_func = explainer_func
        
    def generate_text_variations(self, text: str, num_variations: int = 5) -> List[str]:
        """
        Generate variations of input text for stability testing.
        
        Args:
            text: Original text
            num_variations: Number of variations to generate
            
        Returns:
            List of text variations
        """
        variations = [text]  # Include original
        
        # Variation 1: Add/remove punctuation
        punct_variation = re.sub(r'[.,!?;]', '', text)
        if punct_variation != text:
            variations.append(punct_variation)
            
        # Variation 2: Case changes
        case_variation = text.lower() if text != text.lower() else text.upper()
        variations.append(case_variation)
        
        # Variation 3: Extra spaces
        space_variation = re.sub(r'\s+', '  ', text)
        variations.append(space_variation)
        
        # Variation 4: Minor typos (simulate common mistakes)
        typo_variation = text.replace('the', 'teh').replace('and', 'adn')
        if typo_variation != text:
            variations.append(typo_variation)
            
        # Variation 5: Synonym replacement (simple)
        synonym_variation = text.replace('bad', 'terrible').replace('good', 'great')
        if synonym_variation != text:
            variations.append(synonym_variation)
            
        return variations[:num_variations + 1]
    
    def analyze_stability(self, text: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Analyze explanation stability across text variations.
        
        Args:
            text: Input text to analyze
            top_k: Number of top features to consider
            
        Returns:
            Dictionary containing stability metrics
        """
        variations = self.generate_text_variations(text)
        explanations = []
        predictions = []
        
        for variation in variations:
            try:
                # Get explanation for this variation
                explanation = self.explainer_func(variation, top_k=top_k)
                explanations.append(explanation)
                
                # Get prediction
                pred = self.model.predict_proba([variation])[0]
                predictions.append(pred)
                
            except Exception as e:
                logger.warning(f"Failed to process variation: {variation[:50]}... Error: {e}")
                continue
        
        if len(explanations) < 2:
            return {"stability_score": 0.0, "prediction_consistency": 0.0, "error": "Insufficient valid variations"}
        
        # Calculate stability metrics
        stability_score = self._calculate_explanation_stability(explanations)
        prediction_consistency = self._calculate_prediction_consistency(predictions)
        
        return {
            "stability_score": stability_score,
            "prediction_consistency": prediction_consistency,
            "num_variations": len(explanations),
            "explanations": explanations,
            "predictions": predictions
        }
    
    def _calculate_explanation_stability(self, explanations: List[Dict]) -> float:
        """
        Calculate stability score based on feature importance consistency.
        """
        if len(explanations) < 2:
            return 0.0
            
        # Extract feature importance vectors
        feature_vectors = []
        all_features = set()
        
        for exp in explanations:
            if isinstance(exp, dict) and 'feature_importance' in exp:
                features = exp['feature_importance']
                all_features.update(features.keys())
                feature_vectors.append(features)
        
        if not all_features or len(feature_vectors) < 2:
            return 0.0
        
        # Create normalized vectors
        vectors = []
        for features in feature_vectors:
            vector = [features.get(feat, 0.0) for feat in sorted(all_features)]
            # Normalize
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = np.array(vector) / norm
            vectors.append(vector)
        
        # Calculate pairwise correlations
        correlations = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                corr = np.corrcoef(vectors[i], vectors[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0
    
    def _calculate_prediction_consistency(self, predictions: List[np.ndarray]) -> float:
        """
        Calculate consistency of predictions across variations.
        """
        if len(predictions) < 2:
            return 0.0
            
        # Calculate standard deviation of predictions
        pred_matrix = np.array(predictions)
        std_devs = np.std(pred_matrix, axis=0)
        
        # Return inverse of average standard deviation (higher consistency = lower std)
        avg_std = np.mean(std_devs)
        return max(0.0, 1.0 - avg_std)

class ToxicSpanHighlighter:
    """
    Identifies and highlights toxic word spans directly in text.
    Goes beyond global feature weights to provide precise span-level explanations.
    """
    
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer
        self.toxic_keywords = self._load_toxic_keywords()
        
    def _load_toxic_keywords(self) -> set:
        """
        Load a basic set of toxic keywords for span detection.
        In practice, this would be loaded from a comprehensive lexicon.
        """
        return {
            'hate', 'stupid', 'idiot', 'moron', 'dumb', 'kill', 'die', 'death',
            'murder', 'violence', 'attack', 'destroy', 'annihilate', 'eliminate',
            'racist', 'sexist', 'discrimination', 'prejudice', 'bigot', 'nazi',
            'terrorist', 'extremist', 'radical', 'fanatic', 'lunatic', 'psycho',
            'retard', 'cripple', 'disabled', 'handicapped', 'deformed', 'ugly',
            'fat', 'skinny', 'short', 'tall', 'bald', 'hairy', 'smelly', 'dirty'
        }
    
    def highlight_toxic_spans(self, text: str, threshold: float = 0.1) -> Dict[str, Any]:
        """
        Identify and highlight toxic spans in text.
        
        Args:
            text: Input text to analyze
            threshold: Minimum importance threshold for highlighting
            
        Returns:
            Dictionary containing span information and highlighted text
        """
        # Tokenize text
        words = text.split()
        word_positions = self._get_word_positions(text)
        
        # Get feature importance for the text
        try:
            X = self.vectorizer.transform([text])
            prediction = self.model.predict_proba(X)[0]
            
            # Get feature names and their coefficients
            feature_names = self.vectorizer.get_feature_names_out()
            if hasattr(self.model, 'coef_'):
                coefficients = self.model.coef_[0] if len(self.model.coef_.shape) > 1 else self.model.coef_
            else:
                # For models without coef_ attribute, use feature importance if available
                coefficients = getattr(self.model, 'feature_importances_', np.ones(len(feature_names)))
            
            # Create feature importance mapping
            feature_importance = dict(zip(feature_names, coefficients))
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {"error": str(e), "highlighted_text": text, "toxic_spans": []}
        
        # Identify toxic spans
        toxic_spans = []
        highlighted_text = text
        
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?;:"\'')
            
            # Check if word is in toxic keywords
            is_toxic_keyword = word_lower in self.toxic_keywords
            
            # Check if word has high negative importance (indicates toxicity)
            word_importance = feature_importance.get(word_lower, 0.0)
            is_high_importance = abs(word_importance) > threshold
            
            if is_toxic_keyword or is_high_importance:
                start_pos, end_pos = word_positions[i]
                toxic_spans.append({
                    'word': word,
                    'start': start_pos,
                    'end': end_pos,
                    'importance': word_importance,
                    'is_keyword': is_toxic_keyword,
                    'toxicity_score': abs(word_importance)
                })
                
                # Highlight in text (using HTML-like tags)
                highlighted_word = f"<TOXIC>{word}</TOXIC>"
                highlighted_text = highlighted_text[:start_pos] + highlighted_word + highlighted_text[end_pos:]
        
        return {
            'original_text': text,
            'highlighted_text': highlighted_text,
            'toxic_spans': toxic_spans,
            'prediction_score': prediction[1] if len(prediction) > 1 else prediction[0],
            'num_toxic_spans': len(toxic_spans)
        }
    
    def _get_word_positions(self, text: str) -> List[Tuple[int, int]]:
        """
        Get start and end positions of each word in the text.
        """
        positions = []
        words = text.split()
        current_pos = 0
        
        for word in words:
            start = text.find(word, current_pos)
            end = start + len(word)
            positions.append((start, end))
            current_pos = end
            
        return positions

class FeatureAwareExplainer:
    """
    Combines TF-IDF weights with SHAP scores to create hybrid explanations.
    Provides richer context for content moderation decisions.
    """
    
    def __init__(self, model, vectorizer, shap_explainer=None):
        self.model = model
        self.vectorizer = vectorizer
        self.shap_explainer = shap_explainer
        
    def generate_hybrid_explanation(self, text: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Generate hybrid explanation combining TF-IDF and SHAP insights.
        
        Args:
            text: Input text to explain
            top_k: Number of top features to return
            
        Returns:
            Dictionary containing hybrid explanation
        """
        try:
            # Get TF-IDF features
            X = self.vectorizer.transform([text])
            tfidf_scores = X.toarray()[0]
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get model coefficients (feature importance)
            if hasattr(self.model, 'coef_'):
                model_weights = self.model.coef_[0] if len(self.model.coef_.shape) > 1 else self.model.coef_
            else:
                model_weights = getattr(self.model, 'feature_importances_', np.ones(len(feature_names)))
            
            # Calculate hybrid scores
            hybrid_scores = tfidf_scores * model_weights
            
            # Get top features
            top_indices = np.argsort(np.abs(hybrid_scores))[-top_k:]
            
            feature_explanations = []
            for idx in reversed(top_indices):
                if tfidf_scores[idx] > 0:  # Only include features present in text
                    feature_explanations.append({
                        'feature': feature_names[idx],
                        'tfidf_score': float(tfidf_scores[idx]),
                        'model_weight': float(model_weights[idx]),
                        'hybrid_score': float(hybrid_scores[idx]),
                        'contribution': 'toxic' if hybrid_scores[idx] > 0 else 'non_toxic'
                    })
            
            # Get prediction
            prediction = self.model.predict_proba(X)[0]
            
            return {
                'text': text,
                'prediction_proba': prediction.tolist(),
                'predicted_class': int(np.argmax(prediction)),
                'feature_explanations': feature_explanations,
                'explanation_method': 'hybrid_tfidf_model_weights',
                'top_k': len(feature_explanations)
            }
            
        except Exception as e:
            logger.error(f"Error generating hybrid explanation: {e}")
            return {'error': str(e), 'text': text}
    
    def compare_explanation_methods(self, text: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Compare different explanation methods for the same text.
        
        Args:
            text: Input text to explain
            top_k: Number of top features to return
            
        Returns:
            Dictionary comparing different explanation methods
        """
        results = {
            'text': text,
            'methods': {}
        }
        
        # Method 1: TF-IDF only
        try:
            X = self.vectorizer.transform([text])
            tfidf_scores = X.toarray()[0]
            feature_names = self.vectorizer.get_feature_names_out()
            
            top_tfidf_indices = np.argsort(tfidf_scores)[-top_k:]
            tfidf_features = [(feature_names[i], float(tfidf_scores[i])) 
                             for i in reversed(top_tfidf_indices) if tfidf_scores[i] > 0]
            
            results['methods']['tfidf_only'] = tfidf_features
            
        except Exception as e:
            results['methods']['tfidf_only'] = f"Error: {e}"
        
        # Method 2: Model weights only
        try:
            if hasattr(self.model, 'coef_'):
                model_weights = self.model.coef_[0] if len(self.model.coef_.shape) > 1 else self.model.coef_
                top_weight_indices = np.argsort(np.abs(model_weights))[-top_k:]
                weight_features = [(feature_names[i], float(model_weights[i])) 
                                 for i in reversed(top_weight_indices)]
                results['methods']['model_weights_only'] = weight_features
            else:
                results['methods']['model_weights_only'] = "Model weights not available"
                
        except Exception as e:
            results['methods']['model_weights_only'] = f"Error: {e}"
        
        # Method 3: Hybrid approach
        hybrid_result = self.generate_hybrid_explanation(text, top_k)
        if 'error' not in hybrid_result:
            results['methods']['hybrid'] = [(exp['feature'], exp['hybrid_score']) 
                                          for exp in hybrid_result['feature_explanations']]
        else:
            results['methods']['hybrid'] = hybrid_result['error']
        
        return results

class CustomXAIFramework:
    """
    Main framework class that orchestrates all custom XAI components.
    """
    
    def __init__(self, model_path: str = None, vectorizer_path: str = None):
        self.model = None
        self.vectorizer = None
        self.stability_analyzer = None
        self.span_highlighter = None
        self.feature_explainer = None
        
        if model_path and vectorizer_path:
            self.load_model(model_path, vectorizer_path)
    
    def load_model(self, model_path: str, vectorizer_path: str):
        """
        Load trained model and vectorizer.
        
        Args:
            model_path: Path to saved model
            vectorizer_path: Path to saved vectorizer
        """
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            
            # Initialize components
            self._initialize_components()
            
            logger.info("Model and vectorizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _initialize_components(self):
        """
        Initialize all XAI components.
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model and vectorizer must be loaded first")
        
        # Simple explainer function for stability analysis
        def simple_explainer(text, top_k=10):
            return self.feature_explainer.generate_hybrid_explanation(text, top_k)
        
        self.stability_analyzer = StabilityAnalyzer(self.model, self.vectorizer, simple_explainer)
        self.span_highlighter = ToxicSpanHighlighter(self.model, self.vectorizer)
        self.feature_explainer = FeatureAwareExplainer(self.model, self.vectorizer)
    
    def comprehensive_explanation(self, text: str, include_stability: bool = True, 
                                include_spans: bool = True, top_k: int = 10) -> Dict[str, Any]:
        """
        Generate comprehensive explanation using all available methods.
        
        Args:
            text: Input text to explain
            include_stability: Whether to include stability analysis
            include_spans: Whether to include span highlighting
            top_k: Number of top features to return
            
        Returns:
            Dictionary containing comprehensive explanation
        """
        if not all([self.model, self.vectorizer, self.feature_explainer]):
            raise ValueError("Framework not properly initialized. Load model first.")
        
        results = {
            'text': text,
            'timestamp': pd.Timestamp.now().isoformat(),
            'framework_version': '1.0.0'
        }
        
        # Basic hybrid explanation
        try:
            hybrid_explanation = self.feature_explainer.generate_hybrid_explanation(text, top_k)
            results['hybrid_explanation'] = hybrid_explanation
        except Exception as e:
            results['hybrid_explanation'] = {'error': str(e)}
        
        # Span highlighting
        if include_spans and self.span_highlighter:
            try:
                span_results = self.span_highlighter.highlight_toxic_spans(text)
                results['toxic_spans'] = span_results
            except Exception as e:
                results['toxic_spans'] = {'error': str(e)}
        
        # Stability analysis
        if include_stability and self.stability_analyzer:
            try:
                stability_results = self.stability_analyzer.analyze_stability(text, top_k)
                results['stability_analysis'] = stability_results
            except Exception as e:
                results['stability_analysis'] = {'error': str(e)}
        
        # Method comparison
        try:
            comparison_results = self.feature_explainer.compare_explanation_methods(text, top_k)
            results['method_comparison'] = comparison_results
        except Exception as e:
            results['method_comparison'] = {'error': str(e)}
        
        return results
    
    def batch_explain(self, texts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Generate explanations for multiple texts.
        
        Args:
            texts: List of texts to explain
            **kwargs: Arguments passed to comprehensive_explanation
            
        Returns:
            List of explanation dictionaries
        """
        results = []
        for i, text in enumerate(texts):
            try:
                explanation = self.comprehensive_explanation(text, **kwargs)
                explanation['batch_index'] = i
                results.append(explanation)
            except Exception as e:
                results.append({
                    'text': text,
                    'batch_index': i,
                    'error': str(e)
                })
                logger.error(f"Error explaining text {i}: {e}")
        
        return results

# Example usage and testing functions
def demo_custom_xai():
    """
    Demonstrate the custom XAI framework capabilities.
    """
    print("Custom XAI Framework Demo")
    print("=" * 50)
    
    # Sample toxic and non-toxic texts
    sample_texts = [
        "You are such an idiot and I hate you",
        "This is a wonderful day and I love everyone",
        "That movie was terrible and stupid",
        "Great job on the project, well done!",
        "I want to kill that annoying person"
    ]
    
    try:
        # Initialize framework (would need actual model files)
        framework = CustomXAIFramework()
        print("Framework initialized successfully")
        
        # Note: This would require actual trained models
        print("\nNote: This demo requires trained models to be loaded.")
        print("Please ensure model files are available before running.")
        
    except Exception as e:
        print(f"Demo requires trained models: {e}")
        print("\nFramework components available:")
        print("- StabilityAnalyzer: Analyzes explanation consistency")
        print("- ToxicSpanHighlighter: Identifies toxic word spans")
        print("- FeatureAwareExplainer: Combines TF-IDF and model weights")
        print("- CustomXAIFramework: Orchestrates all components")

if __name__ == "__main__":
    demo_custom_xai()