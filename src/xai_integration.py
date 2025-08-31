#!/usr/bin/env python3
"""
Explainable AI Integration for Reddit Comment Moderation

This module integrates explainable AI features to provide transparency and interpretability
for content moderation decisions in the Kaggle competition.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import shap
import lime
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
import json
import re
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExplanationResult:
    """Container for explanation results."""
    text: str
    prediction: float
    rule_violated: str
    word_importance: Dict[str, float]
    sentence_importance: Dict[str, float]
    feature_importance: Dict[str, float]
    lime_explanation: Optional[Any] = None
    shap_values: Optional[np.ndarray] = None
    attention_weights: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'text': self.text,
            'prediction': self.prediction,
            'rule_violated': self.rule_violated,
            'word_importance': self.word_importance,
            'sentence_importance': self.sentence_importance,
            'feature_importance': self.feature_importance
        }

class AttentionExtractor:
    """Extract attention weights from transformer models."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
    
    def extract_attention(self, text: str, layer: int = -1) -> Tuple[np.ndarray, List[str]]:
        """Extract attention weights for a given text."""
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model outputs with attention
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        # Extract attention weights
        attention_weights = outputs.attentions[layer][0]  # First sample, specified layer
        attention_weights = attention_weights.mean(dim=0)  # Average across heads
        attention_weights = attention_weights.cpu().numpy()
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        return attention_weights, tokens
    
    def visualize_attention(self, text: str, save_path: Optional[str] = None):
        """Visualize attention weights as a heatmap."""
        attention_weights, tokens = self.extract_attention(text)
        
        # Create attention matrix (token to token)
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot heatmap
        sns.heatmap(
            attention_weights, 
            xticklabels=tokens, 
            yticklabels=tokens,
            cmap='Blues', 
            ax=ax
        )
        
        ax.set_title(f'Attention Weights Visualization')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return attention_weights, tokens

class LIMEExplainer:
    """LIME-based explanations for text classification."""
    
    def __init__(self, model, tokenizer=None, class_names=None):
        self.model = model
        self.tokenizer = tokenizer
        self.class_names = class_names or ['No Violation', 'Violation']
        self.explainer = LimeTextExplainer(
            class_names=self.class_names,
            mode='classification'
        )
    
    def predict_proba_wrapper(self, texts: List[str]) -> np.ndarray:
        """Wrapper function for model predictions."""
        if hasattr(self.model, 'predict_proba'):
            # Sklearn-style model
            return self.model.predict_proba(texts)
        else:
            # Custom model - implement prediction logic
            predictions = []
            for text in texts:
                if self.tokenizer:
                    inputs = self.tokenizer(
                        text, 
                        return_tensors='pt', 
                        truncation=True, 
                        padding=True, 
                        max_length=512
                    )
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    pred = torch.sigmoid(outputs.logits).cpu().numpy()[0]
                else:
                    # Fallback for non-transformer models
                    pred = self.model.predict([text])[0]
                
                # Convert to binary classification probabilities
                if isinstance(pred, (int, float)):
                    predictions.append([1-pred, pred])
                else:
                    predictions.append(pred)
            
            return np.array(predictions)
    
    def explain_instance(self, text: str, num_features: int = 10) -> Tuple[Any, Dict[str, float]]:
        """Generate LIME explanation for a single instance."""
        explanation = self.explainer.explain_instance(
            text, 
            self.predict_proba_wrapper, 
            num_features=num_features
        )
        
        # Extract feature importance
        feature_importance = dict(explanation.as_list())
        
        return explanation, feature_importance
    
    def visualize_explanation(self, text: str, save_path: Optional[str] = None):
        """Visualize LIME explanation."""
        explanation, _ = self.explain_instance(text)
        
        # Save as HTML
        if save_path:
            explanation.save_to_file(save_path)
        
        return explanation

class SHAPExplainer:
    """SHAP-based explanations for text classification."""
    
    def __init__(self, model, tokenizer=None, background_data=None):
        self.model = model
        self.tokenizer = tokenizer
        self.background_data = background_data
        self.explainer = None
        self._setup_explainer()
    
    def _setup_explainer(self):
        """Setup SHAP explainer based on model type."""
        if hasattr(self.model, 'predict_proba'):
            # For sklearn models
            self.explainer = shap.Explainer(self.model.predict_proba, self.background_data)
        else:
            # For deep learning models
            self.explainer = shap.Explainer(self._predict_wrapper, self.background_data)
    
    def _predict_wrapper(self, texts):
        """Wrapper for model predictions compatible with SHAP."""
        predictions = []
        for text in texts:
            if self.tokenizer:
                inputs = self.tokenizer(
                    text, 
                    return_tensors='pt', 
                    truncation=True, 
                    padding=True, 
                    max_length=512
                )
                with torch.no_grad():
                    outputs = self.model(**inputs)
                pred = torch.sigmoid(outputs.logits).cpu().numpy()[0]
            else:
                pred = self.model.predict([text])[0]
            
            predictions.append(pred)
        
        return np.array(predictions)
    
    def explain_instance(self, text: str) -> np.ndarray:
        """Generate SHAP explanation for a single instance."""
        shap_values = self.explainer([text])
        return shap_values.values[0]
    
    def visualize_explanation(self, text: str, save_path: Optional[str] = None):
        """Visualize SHAP explanation."""
        shap_values = self.explain_instance(text)
        
        # Create waterfall plot
        shap.plots.waterfall(shap_values, show=False)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return shap_values

class RuleBasedExplainer:
    """Rule-based explanations using keyword and pattern matching."""
    
    def __init__(self, rules_config: Dict[str, Any]):
        self.rules_config = rules_config
        self.keyword_patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for each rule."""
        compiled_patterns = {}
        
        for rule_name, rule_info in self.rules_config.items():
            patterns = []
            
            # Add keyword patterns
            for keyword in rule_info.get('keywords', []):
                pattern = re.compile(r'\b' + re.escape(keyword.lower()) + r'\b', re.IGNORECASE)
                patterns.append(pattern)
            
            # Add regex patterns
            for regex in rule_info.get('patterns', []):
                pattern = re.compile(regex, re.IGNORECASE)
                patterns.append(pattern)
            
            compiled_patterns[rule_name] = patterns
        
        return compiled_patterns
    
    def explain_instance(self, text: str) -> Dict[str, Any]:
        """Generate rule-based explanation for a single instance."""
        text_lower = text.lower()
        explanations = {}
        
        for rule_name, patterns in self.keyword_patterns.items():
            matches = []
            match_positions = []
            
            for pattern in patterns:
                for match in pattern.finditer(text):
                    matches.append(match.group())
                    match_positions.append((match.start(), match.end()))
            
            if matches:
                explanations[rule_name] = {
                    'matches': matches,
                    'positions': match_positions,
                    'severity': self.rules_config[rule_name].get('severity', 1),
                    'description': self.rules_config[rule_name].get('description', '')
                }
        
        return explanations
    
    def highlight_violations(self, text: str) -> str:
        """Highlight rule violations in text with HTML tags."""
        highlighted_text = text
        offset = 0
        
        explanations = self.explain_instance(text)
        
        # Sort positions by start index (reverse order for proper offset handling)
        all_positions = []
        for rule_name, rule_data in explanations.items():
            for pos in rule_data['positions']:
                all_positions.append((pos[0], pos[1], rule_name))
        
        all_positions.sort(key=lambda x: x[0], reverse=True)
        
        # Add HTML highlighting
        for start, end, rule_name in all_positions:
            severity = self.rules_config[rule_name].get('severity', 1)
            color = self._get_severity_color(severity)
            
            highlighted_text = (
                highlighted_text[:start + offset] + 
                f'<span style="background-color: {color}; padding: 2px; border-radius: 3px;" title="{rule_name}">' +
                highlighted_text[start + offset:end + offset] + 
                '</span>' +
                highlighted_text[end + offset:]
            )
            offset += len(f'<span style="background-color: {color}; padding: 2px; border-radius: 3px;" title="{rule_name}">') + len('</span>')
        
        return highlighted_text
    
    def _get_severity_color(self, severity: int) -> str:
        """Get color based on severity level."""
        colors = {
            1: '#ffeb3b',  # Yellow
            2: '#ff9800',  # Orange
            3: '#f44336',  # Red
            4: '#9c27b0',  # Purple
            5: '#000000'   # Black
        }
        return colors.get(severity, '#ffeb3b')

class XAIIntegrator:
    """Main class integrating all explainability methods."""
    
    def __init__(self, model, tokenizer=None, rules_config=None, background_data=None):
        self.model = model
        self.tokenizer = tokenizer
        self.rules_config = rules_config or self._default_rules_config()
        
        # Initialize explainers
        self.lime_explainer = LIMEExplainer(model, tokenizer)
        self.shap_explainer = SHAPExplainer(model, tokenizer, background_data)
        self.rule_explainer = RuleBasedExplainer(self.rules_config)
        
        if tokenizer and hasattr(model, 'transformer'):
            self.attention_extractor = AttentionExtractor(model.transformer, tokenizer)
        else:
            self.attention_extractor = None
    
    def _default_rules_config(self) -> Dict[str, Any]:
        """Default rules configuration."""
        return {
            'harassment': {
                'keywords': ['idiot', 'stupid', 'moron', 'retard', 'kill yourself'],
                'patterns': [r'you\s+are\s+(?:an?\s+)?(?:idiot|moron|stupid)', r'go\s+(?:kill\s+yourself|die)'],
                'severity': 5,
                'description': 'Personal attacks and harassment'
            },
            'spam': {
                'keywords': ['buy now', 'click here', 'subscribe', 'follow me'],
                'patterns': [r'(?:buy|purchase)\s+(?:now|today)', r'(?:subscribe|follow)\s+(?:me|my|us)'],
                'severity': 3,
                'description': 'Spam and self-promotion'
            },
            'personal_info': {
                'keywords': ['phone number', 'address', 'social security'],
                'patterns': [r'\b\d{3}-\d{3}-\d{4}\b', r'\b\d{3}\s\d{3}\s\d{4}\b'],
                'severity': 5,
                'description': 'Sharing personal information'
            }
        }
    
    def comprehensive_explanation(self, text: str, rule_name: str = 'unknown') -> ExplanationResult:
        """Generate comprehensive explanation using all methods."""
        logger.info(f"Generating comprehensive explanation for text: {text[:50]}...")
        
        # Get model prediction
        prediction = self._get_prediction(text)
        
        # LIME explanation
        lime_explanation, lime_features = self.lime_explainer.explain_instance(text)
        
        # SHAP explanation (if available)
        shap_values = None
        try:
            shap_values = self.shap_explainer.explain_instance(text)
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
        
        # Rule-based explanation
        rule_explanations = self.rule_explainer.explain_instance(text)
        
        # Attention weights (if available)
        attention_weights = None
        if self.attention_extractor:
            try:
                attention_weights, tokens = self.attention_extractor.extract_attention(text)
            except Exception as e:
                logger.warning(f"Attention extraction failed: {e}")
        
        # Combine explanations
        word_importance = self._extract_word_importance(text, lime_features, shap_values)
        sentence_importance = self._extract_sentence_importance(text, word_importance)
        feature_importance = lime_features
        
        return ExplanationResult(
            text=text,
            prediction=prediction,
            rule_violated=rule_name,
            word_importance=word_importance,
            sentence_importance=sentence_importance,
            feature_importance=feature_importance,
            lime_explanation=lime_explanation,
            shap_values=shap_values,
            attention_weights=attention_weights
        )
    
    def _get_prediction(self, text: str) -> float:
        """Get model prediction for text."""
        if hasattr(self.model, 'predict_proba'):
            pred = self.model.predict_proba([text])[0]
            return pred[1] if len(pred) > 1 else pred[0]
        else:
            # Custom model prediction
            if self.tokenizer:
                inputs = self.tokenizer(
                    text, 
                    return_tensors='pt', 
                    truncation=True, 
                    padding=True, 
                    max_length=512
                )
                with torch.no_grad():
                    outputs = self.model(**inputs)
                return torch.sigmoid(outputs.logits).cpu().numpy()[0][0]
            else:
                return self.model.predict([text])[0]
    
    def _extract_word_importance(self, text: str, lime_features: Dict[str, float], 
                                shap_values: Optional[np.ndarray]) -> Dict[str, float]:
        """Extract word-level importance scores."""
        words = text.split()
        word_importance = {}
        
        # Use LIME features as base
        for word in words:
            word_lower = word.lower().strip('.,!?;:')
            word_importance[word] = lime_features.get(word_lower, 0.0)
        
        # Enhance with SHAP values if available
        if shap_values is not None and len(shap_values) == len(words):
            for i, word in enumerate(words):
                word_importance[word] = max(word_importance.get(word, 0), abs(shap_values[i]))
        
        return word_importance
    
    def _extract_sentence_importance(self, text: str, word_importance: Dict[str, float]) -> Dict[str, float]:
        """Extract sentence-level importance scores."""
        sentences = re.split(r'[.!?]+', text)
        sentence_importance = {}
        
        for sentence in sentences:
            if sentence.strip():
                words = sentence.split()
                sentence_score = np.mean([word_importance.get(word, 0) for word in words])
                sentence_importance[sentence.strip()] = sentence_score
        
        return sentence_importance
    
    def generate_explanation_report(self, text: str, rule_name: str = 'unknown', 
                                  save_path: Optional[str] = None) -> str:
        """Generate a comprehensive explanation report."""
        explanation = self.comprehensive_explanation(text, rule_name)
        
        # Generate HTML report
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>XAI Explanation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
                .highlight {{ background-color: #ffeb3b; padding: 2px; border-radius: 3px; }}
                .word-importance {{ display: inline-block; margin: 2px; padding: 2px 5px; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>XAI Explanation Report</h1>
                <p><strong>Prediction:</strong> {explanation.prediction:.4f}</p>
                <p><strong>Rule Violated:</strong> {explanation.rule_violated}</p>
            </div>
            
            <div class="section">
                <h2>Original Text</h2>
                <p>{explanation.text}</p>
            </div>
            
            <div class="section">
                <h2>Highlighted Text (Rule-based)</h2>
                <p>{self.rule_explainer.highlight_violations(explanation.text)}</p>
            </div>
            
            <div class="section">
                <h2>Word Importance</h2>
                <div>
        """
        
        # Add word importance visualization
        for word, importance in sorted(explanation.word_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:20]:
            color_intensity = min(255, int(abs(importance) * 255))
            color = f'rgba(255, 0, 0, {abs(importance)})' if importance < 0 else f'rgba(0, 255, 0, {abs(importance)})'
            html_report += f'<span class="word-importance" style="background-color: {color};">{word} ({importance:.3f})</span>'
        
        html_report += """
                </div>
            </div>
            
            <div class="section">
                <h2>Feature Importance (LIME)</h2>
                <ul>
        """
        
        for feature, importance in sorted(explanation.feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
            html_report += f'<li>{feature}: {importance:.4f}</li>'
        
        html_report += """
                </ul>
            </div>
        </body>
        </html>
        """
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(html_report)
        
        return html_report

def main():
    """Example usage of XAI integration."""
    # Create a simple mock model for demonstration
    class MockModel:
        def predict_proba(self, texts):
            # Simple rule-based mock predictions
            predictions = []
            for text in texts:
                text_lower = text.lower()
                score = 0.1  # Base score
                
                # Check for harassment keywords
                harassment_words = ['idiot', 'stupid', 'moron']
                for word in harassment_words:
                    if word in text_lower:
                        score += 0.3
                
                # Check for spam keywords
                spam_words = ['buy now', 'click here']
                for word in spam_words:
                    if word in text_lower:
                        score += 0.2
                
                score = min(score, 0.9)
                predictions.append([1-score, score])
            
            return np.array(predictions)
    
    # Initialize XAI integrator
    model = MockModel()
    xai_integrator = XAIIntegrator(model)
    
    # Test texts
    test_texts = [
        "You are such an idiot for posting this garbage",
        "Great post! Thanks for sharing this information",
        "Buy now and get 50% off! Click here for amazing deals!",
        "This is a normal comment without any violations"
    ]
    
    # Generate explanations
    for i, text in enumerate(test_texts):
        logger.info(f"\nAnalyzing text {i+1}: {text}")
        
        explanation = xai_integrator.comprehensive_explanation(text, f'rule_{i+1}')
        
        logger.info(f"Prediction: {explanation.prediction:.4f}")
        logger.info(f"Top important words: {dict(list(explanation.word_importance.items())[:5])}")
        
        # Generate report
        report_path = f'explanation_report_{i+1}.html'
        xai_integrator.generate_explanation_report(text, f'rule_{i+1}', report_path)
        logger.info(f"Report saved to: {report_path}")

if __name__ == '__main__':
    main()