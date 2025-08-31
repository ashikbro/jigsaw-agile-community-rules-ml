import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP and LIME (optional dependencies)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("LIME not available. Install with: pip install lime")

class RuleSpecificExplainer:
    """
    Rule-specific explainer that provides interpretations tailored to
    community moderation and rule violation detection.
    """
    
    def __init__(self):
        """
        Initialize the rule-specific explainer.
        """
        # Rule violation categories and their associated keywords
        self.rule_categories = {
            'harassment': {
                'keywords': ['harass', 'bully', 'attack', 'abuse', 'threaten', 'intimidate'],
                'description': 'Content that targets individuals with harmful intent',
                'severity': 'high'
            },
            'hate_speech': {
                'keywords': ['hate', 'racist', 'discrimination', 'bigot', 'supremacist'],
                'description': 'Content expressing hatred toward groups based on identity',
                'severity': 'high'
            },
            'threats': {
                'keywords': ['kill', 'die', 'death', 'violence', 'harm', 'hurt'],
                'description': 'Content threatening physical or emotional harm',
                'severity': 'critical'
            },
            'spam': {
                'keywords': ['buy', 'sell', 'promotion', 'advertisement', 'click'],
                'description': 'Unwanted promotional or commercial content',
                'severity': 'medium'
            },
            'trolling': {
                'keywords': ['troll', 'bait', 'flame', 'provoke', 'annoy'],
                'description': 'Content intended to provoke or disrupt discussion',
                'severity': 'medium'
            },
            'off_topic': {
                'keywords': ['topic', 'relevant', 'belong', 'appropriate'],
                'description': 'Content not relevant to the community discussion',
                'severity': 'low'
            }
        }
        
        # Positive indicators (content that suggests compliance)
        self.positive_indicators = {
            'politeness': ['please', 'thank', 'sorry', 'excuse me'],
            'constructive': ['help', 'support', 'contribute', 'improve'],
            'respectful': ['respect', 'understand', 'appreciate', 'agree']
        }
    
    def analyze_keywords(self, text: str, feature_importance: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze keywords in the context of rule violations.
        
        Args:
            text: Input text to analyze
            feature_importance: Dictionary of feature names and their importance scores
            
        Returns:
            Dictionary with keyword analysis
        """
        text_lower = text.lower()
        analysis = {
            'detected_violations': [],
            'positive_indicators': [],
            'keyword_explanations': {},
            'overall_risk': 'low',
            'recommendations': []
        }
        
        # Analyze rule violations
        total_violation_score = 0
        
        for category, info in self.rule_categories.items():
            category_score = 0
            detected_keywords = []
            
            for keyword in info['keywords']:
                if keyword in text_lower:
                    detected_keywords.append(keyword)
                    # Check if this keyword has high importance in the model
                    keyword_importance = feature_importance.get(keyword, 0)
                    category_score += keyword_importance
            
            if detected_keywords:
                violation_info = {
                    'category': category,
                    'keywords': detected_keywords,
                    'description': info['description'],
                    'severity': info['severity'],
                    'score': category_score
                }
                analysis['detected_violations'].append(violation_info)
                total_violation_score += category_score
        
        # Analyze positive indicators
        for category, keywords in self.positive_indicators.items():
            detected_positive = [kw for kw in keywords if kw in text_lower]
            if detected_positive:
                analysis['positive_indicators'].append({
                    'category': category,
                    'keywords': detected_positive
                })
        
        # Determine overall risk
        if total_violation_score > 0.7:
            analysis['overall_risk'] = 'critical'
        elif total_violation_score > 0.4:
            analysis['overall_risk'] = 'high'
        elif total_violation_score > 0.2:
            analysis['overall_risk'] = 'medium'
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Generate actionable recommendations based on the analysis.
        
        Args:
            analysis: Analysis results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if analysis['overall_risk'] == 'critical':
            recommendations.append("Immediate moderation action required")
            recommendations.append("Consider temporary or permanent ban")
        elif analysis['overall_risk'] == 'high':
            recommendations.append("Issue formal warning to user")
            recommendations.append("Monitor user activity closely")
        elif analysis['overall_risk'] == 'medium':
            recommendations.append("Consider informal warning")
            recommendations.append("Add to watchlist for future monitoring")
        
        # Specific recommendations based on violations
        for violation in analysis['detected_violations']:
            if violation['severity'] == 'critical':
                recommendations.append(f"Address {violation['category']} immediately")
            elif violation['severity'] == 'high':
                recommendations.append(f"Take action on {violation['category']} content")
        
        # Positive recommendations
        if analysis['positive_indicators']:
            recommendations.append("User shows positive engagement patterns")
            recommendations.append("Consider educational approach over punitive")
        
        return recommendations

class ModelExplainer:
    """
    Comprehensive model explainer that combines SHAP, LIME, and rule-specific analysis.
    """
    
    def __init__(self, 
                 model: BaseEstimator,
                 feature_extractor: BaseEstimator,
                 feature_names: Optional[List[str]] = None):
        """
        Initialize the model explainer.
        
        Args:
            model: Trained model to explain
            feature_extractor: Feature extractor used with the model
            feature_names: Names of features (optional)
        """
        self.model = model
        self.feature_extractor = feature_extractor
        self.feature_names = feature_names
        self.rule_explainer = RuleSpecificExplainer()
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        
        if SHAP_AVAILABLE:
            self._initialize_shap()
        
        if LIME_AVAILABLE:
            self._initialize_lime()
    
    def _initialize_shap(self):
        """
        Initialize SHAP explainer.
        """
        try:
            # Create a pipeline for SHAP
            pipeline = Pipeline([
                ('features', self.feature_extractor),
                ('classifier', self.model)
            ])
            
            # Use appropriate SHAP explainer based on model type
            if hasattr(self.model, 'predict_proba'):
                self.shap_explainer = shap.Explainer(pipeline.predict_proba)
            else:
                self.shap_explainer = shap.Explainer(pipeline.predict)
                
        except Exception as e:
            print(f"Could not initialize SHAP explainer: {e}")
            self.shap_explainer = None
    
    def _initialize_lime(self):
        """
        Initialize LIME explainer.
        """
        try:
            self.lime_explainer = LimeTextExplainer(
                class_names=['Safe', 'Violation'],
                mode='classification'
            )
        except Exception as e:
            print(f"Could not initialize LIME explainer: {e}")
            self.lime_explainer = None
    
    def explain_prediction(self, 
                          text: str,
                          methods: List[str] = ['rule_based', 'lime', 'shap']) -> Dict[str, Any]:
        """
        Explain a single prediction using multiple methods.
        
        Args:
            text: Text to explain
            methods: List of explanation methods to use
            
        Returns:
            Dictionary with explanations from different methods
        """
        explanations = {
            'text': text,
            'prediction': None,
            'probability': None,
            'explanations': {}
        }
        
        # Get model prediction
        X_features = self.feature_extractor.transform([text])
        prediction = self.model.predict(X_features)[0]
        
        if hasattr(self.model, 'predict_proba'):
            probability = self.model.predict_proba(X_features)[0]
            explanations['probability'] = probability
        
        explanations['prediction'] = prediction
        
        # Rule-based explanation
        if 'rule_based' in methods:
            explanations['explanations']['rule_based'] = self._explain_rule_based(text, X_features)
        
        # LIME explanation
        if 'lime' in methods and self.lime_explainer is not None:
            explanations['explanations']['lime'] = self._explain_lime(text)
        
        # SHAP explanation
        if 'shap' in methods and self.shap_explainer is not None:
            explanations['explanations']['shap'] = self._explain_shap(text)
        
        return explanations
    
    def _explain_rule_based(self, text: str, X_features) -> Dict[str, Any]:
        """
        Provide rule-based explanation.
        
        Args:
            text: Input text
            X_features: Extracted features
            
        Returns:
            Rule-based explanation
        """
        # Get feature importance (simplified approach)
        feature_importance = {}
        
        if hasattr(self.model, 'coef_') and self.feature_names:
            # For linear models
            coefficients = self.model.coef_[0] if len(self.model.coef_.shape) > 1 else self.model.coef_
            for i, coef in enumerate(coefficients):
                if i < len(self.feature_names):
                    feature_importance[self.feature_names[i]] = abs(coef)
        
        elif hasattr(self.model, 'feature_importances_') and self.feature_names:
            # For tree-based models
            for i, importance in enumerate(self.model.feature_importances_):
                if i < len(self.feature_names):
                    feature_importance[self.feature_names[i]] = importance
        
        # Use rule-specific analyzer
        rule_analysis = self.rule_explainer.analyze_keywords(text, feature_importance)
        
        return {
            'method': 'rule_based',
            'analysis': rule_analysis,
            'feature_importance': feature_importance
        }
    
    def _explain_lime(self, text: str) -> Dict[str, Any]:
        """
        Provide LIME explanation.
        
        Args:
            text: Input text
            
        Returns:
            LIME explanation
        """
        try:
            # Create prediction function for LIME
            def predict_fn(texts):
                X_features = self.feature_extractor.transform(texts)
                if hasattr(self.model, 'predict_proba'):
                    return self.model.predict_proba(X_features)
                else:
                    predictions = self.model.predict(X_features)
                    # Convert to probabilities (simplified)
                    proba = np.zeros((len(predictions), 2))
                    proba[predictions == 0, 0] = 1
                    proba[predictions == 1, 1] = 1
                    return proba
            
            # Generate explanation
            explanation = self.lime_explainer.explain_instance(
                text, predict_fn, num_features=10
            )
            
            # Extract important features
            lime_features = explanation.as_list()
            
            return {
                'method': 'lime',
                'important_features': lime_features,
                'explanation_object': explanation
            }
            
        except Exception as e:
            return {
                'method': 'lime',
                'error': str(e),
                'important_features': []
            }
    
    def _explain_shap(self, text: str) -> Dict[str, Any]:
        """
        Provide SHAP explanation.
        
        Args:
            text: Input text
            
        Returns:
            SHAP explanation
        """
        try:
            # Generate SHAP values
            shap_values = self.shap_explainer([text])
            
            # Extract important features
            if hasattr(shap_values, 'values'):
                values = shap_values.values[0]
                if len(values.shape) > 1:
                    values = values[:, 1]  # Positive class for binary classification
            else:
                values = shap_values[0]
            
            # Get feature names if available
            if hasattr(shap_values, 'data'):
                feature_data = shap_values.data[0]
            else:
                feature_data = None
            
            return {
                'method': 'shap',
                'shap_values': values,
                'feature_data': feature_data,
                'explanation_object': shap_values
            }
            
        except Exception as e:
            return {
                'method': 'shap',
                'error': str(e),
                'shap_values': []
            }
    
    def visualize_explanation(self, 
                            explanation: Dict[str, Any],
                            save_path: Optional[str] = None) -> None:
        """
        Create visualizations for the explanation.
        
        Args:
            explanation: Explanation dictionary from explain_prediction
            save_path: Path to save the visualization (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Explanation for: '{explanation['text'][:50]}...'")
        
        # Rule-based visualization
        if 'rule_based' in explanation['explanations']:
            self._plot_rule_based(explanation['explanations']['rule_based'], axes[0, 0])
        
        # LIME visualization
        if 'lime' in explanation['explanations']:
            self._plot_lime(explanation['explanations']['lime'], axes[0, 1])
        
        # SHAP visualization
        if 'shap' in explanation['explanations']:
            self._plot_shap(explanation['explanations']['shap'], axes[1, 0])
        
        # Summary visualization
        self._plot_summary(explanation, axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _plot_rule_based(self, rule_explanation: Dict[str, Any], ax):
        """
        Plot rule-based explanation.
        
        Args:
            rule_explanation: Rule-based explanation data
            ax: Matplotlib axis
        """
        analysis = rule_explanation['analysis']
        
        # Plot violation categories
        if analysis['detected_violations']:
            categories = [v['category'] for v in analysis['detected_violations']]
            scores = [v['score'] for v in analysis['detected_violations']]
            
            bars = ax.bar(categories, scores)
            ax.set_title('Rule Violation Analysis')
            ax.set_ylabel('Violation Score')
            ax.tick_params(axis='x', rotation=45)
            
            # Color bars by severity
            for bar, violation in zip(bars, analysis['detected_violations']):
                if violation['severity'] == 'critical':
                    bar.set_color('red')
                elif violation['severity'] == 'high':
                    bar.set_color('orange')
                else:
                    bar.set_color('yellow')
        else:
            ax.text(0.5, 0.5, 'No violations detected', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Rule Violation Analysis')
    
    def _plot_lime(self, lime_explanation: Dict[str, Any], ax):
        """
        Plot LIME explanation.
        
        Args:
            lime_explanation: LIME explanation data
            ax: Matplotlib axis
        """
        if 'important_features' in lime_explanation and lime_explanation['important_features']:
            features = [f[0] for f in lime_explanation['important_features'][:10]]
            importances = [f[1] for f in lime_explanation['important_features'][:10]]
            
            colors = ['red' if imp < 0 else 'green' for imp in importances]
            bars = ax.barh(features, importances, color=colors)
            ax.set_title('LIME Feature Importance')
            ax.set_xlabel('Importance Score')
        else:
            ax.text(0.5, 0.5, 'LIME explanation not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('LIME Feature Importance')
    
    def _plot_shap(self, shap_explanation: Dict[str, Any], ax):
        """
        Plot SHAP explanation.
        
        Args:
            shap_explanation: SHAP explanation data
            ax: Matplotlib axis
        """
        if 'shap_values' in shap_explanation and len(shap_explanation['shap_values']) > 0:
            values = shap_explanation['shap_values']
            
            # Simple bar plot of SHAP values
            indices = range(len(values))
            colors = ['red' if v < 0 else 'green' for v in values]
            
            ax.bar(indices, values, color=colors)
            ax.set_title('SHAP Values')
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('SHAP Value')
        else:
            ax.text(0.5, 0.5, 'SHAP explanation not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('SHAP Values')
    
    def _plot_summary(self, explanation: Dict[str, Any], ax):
        """
        Plot summary of all explanations.
        
        Args:
            explanation: Full explanation dictionary
            ax: Matplotlib axis
        """
        # Create summary text
        summary_text = f"Prediction: {explanation['prediction']}\n"
        
        if explanation['probability'] is not None:
            prob = explanation['probability']
            if len(prob) == 2:
                summary_text += f"Confidence: {prob[1]:.3f}\n\n"
        
        # Add rule-based summary
        if 'rule_based' in explanation['explanations']:
            rule_analysis = explanation['explanations']['rule_based']['analysis']
            summary_text += f"Risk Level: {rule_analysis['overall_risk']}\n"
            summary_text += f"Violations: {len(rule_analysis['detected_violations'])}\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=12)
        ax.set_title('Prediction Summary')
        ax.axis('off')
    
    def batch_explain(self, 
                     texts: List[str],
                     methods: List[str] = ['rule_based']) -> List[Dict[str, Any]]:
        """
        Explain multiple predictions in batch.
        
        Args:
            texts: List of texts to explain
            methods: List of explanation methods to use
            
        Returns:
            List of explanation dictionaries
        """
        explanations = []
        
        for text in texts:
            explanation = self.explain_prediction(text, methods)
            explanations.append(explanation)
        
        return explanations
    
    def generate_explanation_report(self, 
                                  explanations: List[Dict[str, Any]],
                                  save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive explanation report.
        
        Args:
            explanations: List of explanations from batch_explain
            save_path: Path to save the report (optional)
            
        Returns:
            Report as string
        """
        report = "# Model Explanation Report\n\n"
        
        # Summary statistics
        total_predictions = len(explanations)
        violation_predictions = sum(1 for exp in explanations if exp['prediction'] == 1)
        
        report += f"## Summary\n"
        report += f"- Total predictions: {total_predictions}\n"
        report += f"- Violation predictions: {violation_predictions}\n"
        report += f"- Safe predictions: {total_predictions - violation_predictions}\n\n"
        
        # Detailed explanations
        report += "## Detailed Explanations\n\n"
        
        for i, explanation in enumerate(explanations):
            report += f"### Prediction {i+1}\n"
            report += f"**Text:** {explanation['text'][:100]}...\n"
            report += f"**Prediction:** {explanation['prediction']}\n"
            
            if explanation['probability'] is not None:
                prob = explanation['probability']
                if len(prob) == 2:
                    report += f"**Confidence:** {prob[1]:.3f}\n"
            
            # Rule-based explanation
            if 'rule_based' in explanation['explanations']:
                rule_analysis = explanation['explanations']['rule_based']['analysis']
                report += f"**Risk Level:** {rule_analysis['overall_risk']}\n"
                
                if rule_analysis['detected_violations']:
                    report += "**Detected Violations:**\n"
                    for violation in rule_analysis['detected_violations']:
                        report += f"- {violation['category']}: {violation['keywords']}\n"
                
                if rule_analysis['recommendations']:
                    report += "**Recommendations:**\n"
                    for rec in rule_analysis['recommendations']:
                        report += f"- {rec}\n"
            
            report += "\n---\n\n"
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report