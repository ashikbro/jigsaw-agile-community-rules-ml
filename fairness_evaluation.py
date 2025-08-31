#!/usr/bin/env python3
"""
Fairness and Bias Evaluation Module for Content Moderation

This module implements comprehensive fairness evaluation for ML-based content moderation,
including bias detection across languages, demographic groups, and linguistic styles.

Novelty: Bias & fairness evaluation of classical ML moderation models across
multiple languages with focus on linguistic diversity and cultural context.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
from collections import defaultdict, Counter
import warnings
import logging
import re
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# ML imports
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, average_precision_score
)
from sklearn.model_selection import cross_val_predict
from sklearn.base import BaseEstimator, ClassifierMixin
import scipy.stats as stats

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DemographicBiasDetector:
    """
    Detects bias in model predictions across different demographic groups.
    """
    
    def __init__(self):
        self.bias_metrics = {}
        self.group_performance = {}
        self.statistical_tests = {}
    
    def evaluate_demographic_bias(self, predictions: np.ndarray, true_labels: np.ndarray,
                                 demographic_groups: List[str], 
                                 group_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate bias across demographic groups.
        
        Args:
            predictions: Model predictions
            true_labels: Ground truth labels
            demographic_groups: List of demographic group identifiers for each sample
            group_names: Optional names for demographic groups
            
        Returns:
            Dictionary containing bias evaluation results
        """
        unique_groups = list(set(demographic_groups))
        if group_names is None:
            group_names = [f"Group_{g}" for g in unique_groups]
        
        results = {
            'groups_evaluated': unique_groups,
            'group_names': group_names,
            'group_performance': {},
            'bias_metrics': {},
            'statistical_tests': {},
            'recommendations': []
        }
        
        # Calculate performance for each group
        for i, group in enumerate(unique_groups):
            group_mask = np.array(demographic_groups) == group
            group_preds = predictions[group_mask]
            group_true = true_labels[group_mask]
            
            if len(group_preds) == 0:
                continue
            
            # Basic metrics
            accuracy = accuracy_score(group_true, group_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                group_true, group_preds, average='weighted', zero_division=0
            )
            
            # False positive and false negative rates
            tn, fp, fn, tp = confusion_matrix(group_true, group_preds, labels=[0, 1]).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            group_name = group_names[i] if i < len(group_names) else f"Group_{group}"
            
            results['group_performance'][group_name] = {
                'group_id': group,
                'sample_count': len(group_preds),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'false_positive_rate': fpr,
                'false_negative_rate': fnr,
                'positive_prediction_rate': np.mean(group_preds)
            }
        
        # Calculate bias metrics
        results['bias_metrics'] = self._calculate_bias_metrics(results['group_performance'])
        
        # Statistical significance tests
        results['statistical_tests'] = self._perform_statistical_tests(
            predictions, true_labels, demographic_groups, unique_groups
        )
        
        # Generate recommendations
        results['recommendations'] = self._generate_bias_recommendations(results)
        
        self.bias_metrics = results
        return results
    
    def _calculate_bias_metrics(self, group_performance: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Calculate various bias metrics.
        
        Args:
            group_performance: Performance metrics for each group
            
        Returns:
            Dictionary of bias metrics
        """
        metrics = {
            'equalized_odds_difference': {},
            'demographic_parity_difference': {},
            'equal_opportunity_difference': {},
            'accuracy_disparity': {},
            'max_bias_score': 0.0
        }
        
        group_names = list(group_performance.keys())
        
        if len(group_names) < 2:
            return metrics
        
        # Calculate pairwise differences
        for i, group1 in enumerate(group_names):
            for j, group2 in enumerate(group_names[i+1:], i+1):
                perf1 = group_performance[group1]
                perf2 = group_performance[group2]
                
                pair_key = f"{group1}_vs_{group2}"
                
                # Demographic parity (difference in positive prediction rates)
                demo_parity_diff = abs(perf1['positive_prediction_rate'] - 
                                     perf2['positive_prediction_rate'])
                metrics['demographic_parity_difference'][pair_key] = demo_parity_diff
                
                # Equal opportunity (difference in true positive rates/recall)
                equal_opp_diff = abs(perf1['recall'] - perf2['recall'])
                metrics['equal_opportunity_difference'][pair_key] = equal_opp_diff
                
                # Equalized odds (max difference in FPR and TPR)
                fpr_diff = abs(perf1['false_positive_rate'] - perf2['false_positive_rate'])
                tpr_diff = equal_opp_diff  # TPR is recall
                equalized_odds_diff = max(fpr_diff, tpr_diff)
                metrics['equalized_odds_difference'][pair_key] = equalized_odds_diff
                
                # Accuracy disparity
                accuracy_diff = abs(perf1['accuracy'] - perf2['accuracy'])
                metrics['accuracy_disparity'][pair_key] = accuracy_diff
                
                # Update max bias score
                max_current_bias = max(demo_parity_diff, equal_opp_diff, 
                                     equalized_odds_diff, accuracy_diff)
                metrics['max_bias_score'] = max(metrics['max_bias_score'], max_current_bias)
        
        return metrics
    
    def _perform_statistical_tests(self, predictions: np.ndarray, true_labels: np.ndarray,
                                  demographic_groups: List[str], unique_groups: List) -> Dict[str, Any]:
        """
        Perform statistical significance tests for bias.
        
        Args:
            predictions: Model predictions
            true_labels: Ground truth labels
            demographic_groups: Demographic group identifiers
            unique_groups: List of unique groups
            
        Returns:
            Statistical test results
        """
        tests = {
            'chi_square_test': {},
            'fisher_exact_test': {},
            'permutation_test': {}
        }
        
        # Chi-square test for independence
        try:
            contingency_table = []
            for group in unique_groups:
                group_mask = np.array(demographic_groups) == group
                group_preds = predictions[group_mask]
                
                if len(group_preds) > 0:
                    pos_preds = np.sum(group_preds == 1)
                    neg_preds = np.sum(group_preds == 0)
                    contingency_table.append([pos_preds, neg_preds])
            
            if len(contingency_table) >= 2:
                chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
                tests['chi_square_test'] = {
                    'statistic': chi2,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        except Exception as e:
            logger.warning(f"Chi-square test failed: {e}")
        
        # Permutation test for accuracy differences
        if len(unique_groups) == 2:
            try:
                group1_mask = np.array(demographic_groups) == unique_groups[0]
                group2_mask = np.array(demographic_groups) == unique_groups[1]
                
                group1_acc = accuracy_score(true_labels[group1_mask], predictions[group1_mask])
                group2_acc = accuracy_score(true_labels[group2_mask], predictions[group2_mask])
                
                observed_diff = abs(group1_acc - group2_acc)
                
                # Permutation test
                n_permutations = 1000
                perm_diffs = []
                
                combined_preds = np.concatenate([predictions[group1_mask], predictions[group2_mask]])
                combined_true = np.concatenate([true_labels[group1_mask], true_labels[group2_mask]])
                
                for _ in range(n_permutations):
                    perm_indices = np.random.permutation(len(combined_preds))
                    perm_preds = combined_preds[perm_indices]
                    
                    n_group1 = np.sum(group1_mask)
                    perm_group1_acc = accuracy_score(combined_true[:n_group1], perm_preds[:n_group1])
                    perm_group2_acc = accuracy_score(combined_true[n_group1:], perm_preds[n_group1:])
                    
                    perm_diffs.append(abs(perm_group1_acc - perm_group2_acc))
                
                p_value = np.mean(np.array(perm_diffs) >= observed_diff)
                
                tests['permutation_test'] = {
                    'observed_difference': observed_diff,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'n_permutations': n_permutations
                }
                
            except Exception as e:
                logger.warning(f"Permutation test failed: {e}")
        
        return tests
    
    def _generate_bias_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on bias evaluation results.
        
        Args:
            results: Bias evaluation results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        max_bias = results['bias_metrics']['max_bias_score']
        
        if max_bias > 0.2:
            recommendations.append("HIGH BIAS DETECTED: Significant disparities found between groups.")
            recommendations.append("Consider rebalancing training data or applying bias mitigation techniques.")
        elif max_bias > 0.1:
            recommendations.append("MODERATE BIAS: Some disparities detected between groups.")
            recommendations.append("Monitor model performance and consider bias-aware training.")
        else:
            recommendations.append("LOW BIAS: Model shows relatively fair performance across groups.")
        
        # Check statistical significance
        if results['statistical_tests'].get('chi_square_test', {}).get('significant', False):
            recommendations.append("Statistical tests indicate significant bias in predictions.")
        
        if results['statistical_tests'].get('permutation_test', {}).get('significant', False):
            recommendations.append("Permutation test confirms significant accuracy differences between groups.")
        
        return recommendations

class LanguageBiasEvaluator:
    """
    Evaluates bias specifically across different languages and linguistic styles.
    """
    
    def __init__(self):
        self.language_patterns = self._initialize_language_patterns()
        self.linguistic_features = {}
    
    def _initialize_language_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Initialize patterns for different linguistic styles and dialects.
        
        Returns:
            Dictionary of language patterns
        """
        return {
            'en': {
                'formal': ['therefore', 'furthermore', 'consequently', 'nevertheless', 'moreover'],
                'informal': ['gonna', 'wanna', 'gotta', 'kinda', 'sorta', 'yeah', 'nah'],
                'slang': ['lit', 'fire', 'sick', 'dope', 'savage', 'salty', 'shade'],
                'aave': ['finna', 'bout', 'aint', 'yall', 'deadass', 'periodt'],
                'sarcasm_markers': ['oh sure', 'yeah right', 'totally', 'obviously', 'clearly']
            },
            'es': {
                'formal': ['por lo tanto', 'además', 'sin embargo', 'no obstante'],
                'informal': ['pues', 'bueno', 'vale', 'oye', 'mira'],
                'slang': ['chido', 'padre', 'genial', 'bacán', 'chevere']
            },
            'fr': {
                'formal': ['par conséquent', 'néanmoins', 'cependant', 'toutefois'],
                'informal': ['bon', 'ben', 'ouais', 'nan', 'bah'],
                'slang': ['cool', 'sympa', 'génial', 'super']
            }
        }
    
    def detect_linguistic_style(self, text: str, language: str = 'en') -> Dict[str, float]:
        """
        Detect linguistic style characteristics in text.
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            Dictionary of style scores
        """
        text_lower = text.lower()
        patterns = self.language_patterns.get(language, {})
        
        style_scores = {}
        
        for style, markers in patterns.items():
            score = sum(1 for marker in markers if marker in text_lower)
            style_scores[style] = score / len(text.split()) if text.split() else 0
        
        return style_scores
    
    def evaluate_language_bias(self, texts: List[str], predictions: np.ndarray, 
                              true_labels: np.ndarray, languages: List[str]) -> Dict[str, Any]:
        """
        Evaluate bias across different languages and linguistic styles.
        
        Args:
            texts: Input texts
            predictions: Model predictions
            true_labels: Ground truth labels
            languages: Language codes for each text
            
        Returns:
            Language bias evaluation results
        """
        results = {
            'language_performance': {},
            'style_bias_analysis': {},
            'cross_language_consistency': {},
            'recommendations': []
        }
        
        unique_languages = list(set(languages))
        
        # Evaluate performance by language
        for lang in unique_languages:
            lang_mask = np.array(languages) == lang
            lang_texts = [texts[i] for i in range(len(texts)) if lang_mask[i]]
            lang_preds = predictions[lang_mask]
            lang_true = true_labels[lang_mask]
            
            if len(lang_preds) == 0:
                continue
            
            # Basic performance metrics
            accuracy = accuracy_score(lang_true, lang_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                lang_true, lang_preds, average='weighted', zero_division=0
            )
            
            # False positive rate analysis
            tn, fp, fn, tp = confusion_matrix(lang_true, lang_preds, labels=[0, 1]).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            results['language_performance'][lang] = {
                'sample_count': len(lang_preds),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'false_positive_rate': fpr,
                'positive_prediction_rate': np.mean(lang_preds)
            }
            
            # Analyze linguistic style bias within language
            style_analysis = self._analyze_style_bias(lang_texts, lang_preds, lang_true, lang)
            results['style_bias_analysis'][lang] = style_analysis
        
        # Cross-language consistency analysis
        results['cross_language_consistency'] = self._analyze_cross_language_consistency(
            results['language_performance']
        )
        
        # Generate recommendations
        results['recommendations'] = self._generate_language_bias_recommendations(results)
        
        return results
    
    def _analyze_style_bias(self, texts: List[str], predictions: np.ndarray, 
                           true_labels: np.ndarray, language: str) -> Dict[str, Any]:
        """
        Analyze bias across different linguistic styles within a language.
        
        Args:
            texts: Texts in the language
            predictions: Predictions for these texts
            true_labels: True labels
            language: Language code
            
        Returns:
            Style bias analysis results
        """
        style_analysis = {
            'style_performance': {},
            'style_bias_scores': {},
            'dominant_style_bias': None
        }
        
        # Detect styles for each text
        text_styles = []
        for text in texts:
            style_scores = self.detect_linguistic_style(text, language)
            # Assign dominant style (or 'neutral' if no clear style)
            if style_scores:
                max_style = max(style_scores.items(), key=lambda x: x[1])
                dominant_style = max_style[0] if max_style[1] > 0 else 'neutral'
            else:
                dominant_style = 'neutral'
            text_styles.append(dominant_style)
        
        # Analyze performance by style
        unique_styles = list(set(text_styles))
        
        for style in unique_styles:
            style_mask = np.array(text_styles) == style
            style_preds = predictions[style_mask]
            style_true = true_labels[style_mask]
            
            if len(style_preds) == 0:
                continue
            
            accuracy = accuracy_score(style_true, style_preds)
            fpr = self._calculate_fpr(style_true, style_preds)
            
            style_analysis['style_performance'][style] = {
                'sample_count': len(style_preds),
                'accuracy': accuracy,
                'false_positive_rate': fpr,
                'positive_prediction_rate': np.mean(style_preds)
            }
        
        # Calculate style bias scores
        if len(style_analysis['style_performance']) > 1:
            style_performances = style_analysis['style_performance']
            styles = list(style_performances.keys())
            
            max_fpr_diff = 0
            max_acc_diff = 0
            
            for i, style1 in enumerate(styles):
                for style2 in styles[i+1:]:
                    fpr_diff = abs(style_performances[style1]['false_positive_rate'] - 
                                 style_performances[style2]['false_positive_rate'])
                    acc_diff = abs(style_performances[style1]['accuracy'] - 
                                 style_performances[style2]['accuracy'])
                    
                    max_fpr_diff = max(max_fpr_diff, fpr_diff)
                    max_acc_diff = max(max_acc_diff, acc_diff)
            
            style_analysis['style_bias_scores'] = {
                'max_fpr_difference': max_fpr_diff,
                'max_accuracy_difference': max_acc_diff,
                'overall_style_bias': max(max_fpr_diff, max_acc_diff)
            }
        
        return style_analysis
    
    def _calculate_fpr(self, true_labels: np.ndarray, predictions: np.ndarray) -> float:
        """
        Calculate false positive rate.
        
        Args:
            true_labels: Ground truth labels
            predictions: Model predictions
            
        Returns:
            False positive rate
        """
        try:
            tn, fp, fn, tp = confusion_matrix(true_labels, predictions, labels=[0, 1]).ravel()
            return fp / (fp + tn) if (fp + tn) > 0 else 0
        except:
            return 0.0
    
    def _analyze_cross_language_consistency(self, language_performance: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Analyze consistency of model performance across languages.
        
        Args:
            language_performance: Performance metrics by language
            
        Returns:
            Cross-language consistency analysis
        """
        consistency = {
            'accuracy_variance': 0.0,
            'fpr_variance': 0.0,
            'max_accuracy_difference': 0.0,
            'max_fpr_difference': 0.0,
            'consistency_score': 0.0
        }
        
        if len(language_performance) < 2:
            return consistency
        
        accuracies = [perf['accuracy'] for perf in language_performance.values()]
        fprs = [perf['false_positive_rate'] for perf in language_performance.values()]
        
        consistency['accuracy_variance'] = np.var(accuracies)
        consistency['fpr_variance'] = np.var(fprs)
        consistency['max_accuracy_difference'] = max(accuracies) - min(accuracies)
        consistency['max_fpr_difference'] = max(fprs) - min(fprs)
        
        # Overall consistency score (lower is better)
        consistency['consistency_score'] = (consistency['max_accuracy_difference'] + 
                                          consistency['max_fpr_difference']) / 2
        
        return consistency
    
    def _generate_language_bias_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations for language bias mitigation.
        
        Args:
            results: Language bias evaluation results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check cross-language consistency
        consistency = results['cross_language_consistency']
        if consistency['consistency_score'] > 0.2:
            recommendations.append("HIGH CROSS-LANGUAGE INCONSISTENCY: Model performance varies significantly across languages.")
            recommendations.append("Consider language-specific model tuning or multilingual training approaches.")
        elif consistency['consistency_score'] > 0.1:
            recommendations.append("MODERATE CROSS-LANGUAGE INCONSISTENCY: Some variation in performance across languages.")
        
        # Check for style bias within languages
        high_style_bias_langs = []
        for lang, style_analysis in results['style_bias_analysis'].items():
            if 'style_bias_scores' in style_analysis:
                bias_score = style_analysis['style_bias_scores'].get('overall_style_bias', 0)
                if bias_score > 0.15:
                    high_style_bias_langs.append(lang)
        
        if high_style_bias_langs:
            recommendations.append(f"LINGUISTIC STYLE BIAS detected in languages: {', '.join(high_style_bias_langs)}")
            recommendations.append("Consider training data augmentation with diverse linguistic styles.")
        
        # Check false positive rate disparities
        max_fpr_diff = consistency['max_fpr_difference']
        if max_fpr_diff > 0.1:
            recommendations.append("SIGNIFICANT FPR DISPARITY: Some languages have much higher false positive rates.")
            recommendations.append("This may indicate cultural or linguistic bias in toxicity definitions.")
        
        return recommendations

class FairnessMetricsCalculator:
    """
    Calculates comprehensive fairness metrics for content moderation models.
    """
    
    def __init__(self):
        self.supported_metrics = [
            'demographic_parity',
            'equalized_odds',
            'equal_opportunity',
            'calibration',
            'individual_fairness'
        ]
    
    def calculate_all_fairness_metrics(self, predictions: np.ndarray, 
                                     probabilities: np.ndarray,
                                     true_labels: np.ndarray,
                                     protected_attributes: List[str]) -> Dict[str, Any]:
        """
        Calculate comprehensive fairness metrics.
        
        Args:
            predictions: Binary predictions
            probabilities: Prediction probabilities
            true_labels: Ground truth labels
            protected_attributes: Protected attribute values for each sample
            
        Returns:
            Dictionary of fairness metrics
        """
        results = {
            'demographic_parity': self._demographic_parity(predictions, protected_attributes),
            'equalized_odds': self._equalized_odds(predictions, true_labels, protected_attributes),
            'equal_opportunity': self._equal_opportunity(predictions, true_labels, protected_attributes),
            'calibration': self._calibration_metrics(probabilities, true_labels, protected_attributes),
            'overall_fairness_score': 0.0
        }
        
        # Calculate overall fairness score
        fairness_scores = []
        for metric_name, metric_result in results.items():
            if isinstance(metric_result, dict) and 'fairness_violation' in metric_result:
                fairness_scores.append(1.0 - metric_result['fairness_violation'])
        
        if fairness_scores:
            results['overall_fairness_score'] = np.mean(fairness_scores)
        
        return results
    
    def _demographic_parity(self, predictions: np.ndarray, 
                           protected_attributes: List[str]) -> Dict[str, Any]:
        """
        Calculate demographic parity metrics.
        
        Args:
            predictions: Model predictions
            protected_attributes: Protected attribute values
            
        Returns:
            Demographic parity results
        """
        unique_groups = list(set(protected_attributes))
        group_rates = {}
        
        for group in unique_groups:
            group_mask = np.array(protected_attributes) == group
            group_preds = predictions[group_mask]
            
            if len(group_preds) > 0:
                positive_rate = np.mean(group_preds)
                group_rates[group] = positive_rate
        
        if len(group_rates) < 2:
            return {'fairness_violation': 0.0, 'group_rates': group_rates}
        
        rates = list(group_rates.values())
        max_diff = max(rates) - min(rates)
        
        return {
            'group_rates': group_rates,
            'max_difference': max_diff,
            'fairness_violation': max_diff,
            'is_fair': max_diff < 0.1  # 10% threshold
        }
    
    def _equalized_odds(self, predictions: np.ndarray, true_labels: np.ndarray,
                       protected_attributes: List[str]) -> Dict[str, Any]:
        """
        Calculate equalized odds metrics.
        
        Args:
            predictions: Model predictions
            true_labels: Ground truth labels
            protected_attributes: Protected attribute values
            
        Returns:
            Equalized odds results
        """
        unique_groups = list(set(protected_attributes))
        group_metrics = {}
        
        for group in unique_groups:
            group_mask = np.array(protected_attributes) == group
            group_preds = predictions[group_mask]
            group_true = true_labels[group_mask]
            
            if len(group_preds) > 0:
                # Calculate TPR and FPR
                try:
                    tn, fp, fn, tp = confusion_matrix(group_true, group_preds, labels=[0, 1]).ravel()
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    
                    group_metrics[group] = {'tpr': tpr, 'fpr': fpr}
                except:
                    group_metrics[group] = {'tpr': 0, 'fpr': 0}
        
        if len(group_metrics) < 2:
            return {'fairness_violation': 0.0, 'group_metrics': group_metrics}
        
        # Calculate max differences
        tprs = [metrics['tpr'] for metrics in group_metrics.values()]
        fprs = [metrics['fpr'] for metrics in group_metrics.values()]
        
        max_tpr_diff = max(tprs) - min(tprs)
        max_fpr_diff = max(fprs) - min(fprs)
        
        fairness_violation = max(max_tpr_diff, max_fpr_diff)
        
        return {
            'group_metrics': group_metrics,
            'max_tpr_difference': max_tpr_diff,
            'max_fpr_difference': max_fpr_diff,
            'fairness_violation': fairness_violation,
            'is_fair': fairness_violation < 0.1
        }
    
    def _equal_opportunity(self, predictions: np.ndarray, true_labels: np.ndarray,
                         protected_attributes: List[str]) -> Dict[str, Any]:
        """
        Calculate equal opportunity metrics.
        
        Args:
            predictions: Model predictions
            true_labels: Ground truth labels
            protected_attributes: Protected attribute values
            
        Returns:
            Equal opportunity results
        """
        unique_groups = list(set(protected_attributes))
        group_tprs = {}
        
        for group in unique_groups:
            group_mask = np.array(protected_attributes) == group
            group_preds = predictions[group_mask]
            group_true = true_labels[group_mask]
            
            if len(group_preds) > 0:
                # Calculate TPR (recall)
                try:
                    tn, fp, fn, tp = confusion_matrix(group_true, group_preds, labels=[0, 1]).ravel()
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    group_tprs[group] = tpr
                except:
                    group_tprs[group] = 0
        
        if len(group_tprs) < 2:
            return {'fairness_violation': 0.0, 'group_tprs': group_tprs}
        
        tprs = list(group_tprs.values())
        max_diff = max(tprs) - min(tprs)
        
        return {
            'group_tprs': group_tprs,
            'max_difference': max_diff,
            'fairness_violation': max_diff,
            'is_fair': max_diff < 0.1
        }
    
    def _calibration_metrics(self, probabilities: np.ndarray, true_labels: np.ndarray,
                           protected_attributes: List[str]) -> Dict[str, Any]:
        """
        Calculate calibration metrics across groups.
        
        Args:
            probabilities: Prediction probabilities
            true_labels: Ground truth labels
            protected_attributes: Protected attribute values
            
        Returns:
            Calibration results
        """
        unique_groups = list(set(protected_attributes))
        group_calibration = {}
        
        for group in unique_groups:
            group_mask = np.array(protected_attributes) == group
            group_probs = probabilities[group_mask]
            group_true = true_labels[group_mask]
            
            if len(group_probs) > 10:  # Need sufficient samples
                # Simple calibration: bin probabilities and check accuracy in each bin
                bins = np.linspace(0, 1, 11)
                bin_accuracies = []
                bin_confidences = []
                
                for i in range(len(bins) - 1):
                    bin_mask = (group_probs >= bins[i]) & (group_probs < bins[i + 1])
                    if np.sum(bin_mask) > 0:
                        bin_acc = np.mean(group_true[bin_mask])
                        bin_conf = np.mean(group_probs[bin_mask])
                        bin_accuracies.append(bin_acc)
                        bin_confidences.append(bin_conf)
                
                if bin_accuracies:
                    # Expected Calibration Error (ECE)
                    ece = np.mean([abs(acc - conf) for acc, conf in zip(bin_accuracies, bin_confidences)])
                    group_calibration[group] = ece
        
        if len(group_calibration) < 2:
            return {'fairness_violation': 0.0, 'group_calibration': group_calibration}
        
        calibration_values = list(group_calibration.values())
        max_diff = max(calibration_values) - min(calibration_values)
        
        return {
            'group_calibration': group_calibration,
            'max_difference': max_diff,
            'fairness_violation': max_diff,
            'is_fair': max_diff < 0.05  # 5% threshold for calibration
        }

class ComprehensiveFairnessEvaluator:
    """
    Main class that orchestrates comprehensive fairness evaluation.
    """
    
    def __init__(self):
        self.demographic_detector = DemographicBiasDetector()
        self.language_evaluator = LanguageBiasEvaluator()
        self.metrics_calculator = FairnessMetricsCalculator()
        self.evaluation_results = {}
    
    def comprehensive_fairness_evaluation(self, 
                                        texts: List[str],
                                        predictions: np.ndarray,
                                        probabilities: np.ndarray,
                                        true_labels: np.ndarray,
                                        languages: List[str],
                                        demographic_groups: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive fairness evaluation.
        
        Args:
            texts: Input texts
            predictions: Model predictions
            probabilities: Prediction probabilities
            true_labels: Ground truth labels
            languages: Language codes
            demographic_groups: Optional demographic group identifiers
            
        Returns:
            Comprehensive fairness evaluation results
        """
        results = {
            'evaluation_metadata': {
                'total_samples': len(texts),
                'unique_languages': list(set(languages)),
                'evaluation_timestamp': pd.Timestamp.now().isoformat()
            },
            'language_bias_evaluation': {},
            'demographic_bias_evaluation': {},
            'fairness_metrics': {},
            'overall_fairness_assessment': {},
            'recommendations': []
        }
        
        # Language bias evaluation
        logger.info("Evaluating language bias...")
        results['language_bias_evaluation'] = self.language_evaluator.evaluate_language_bias(
            texts, predictions, true_labels, languages
        )
        
        # Demographic bias evaluation (if demographic data available)
        if demographic_groups:
            logger.info("Evaluating demographic bias...")
            results['demographic_bias_evaluation'] = self.demographic_detector.evaluate_demographic_bias(
                predictions, true_labels, demographic_groups
            )
        
        # Comprehensive fairness metrics
        logger.info("Calculating fairness metrics...")
        protected_attributes = demographic_groups if demographic_groups else languages
        results['fairness_metrics'] = self.metrics_calculator.calculate_all_fairness_metrics(
            predictions, probabilities, true_labels, protected_attributes
        )
        
        # Overall assessment
        results['overall_fairness_assessment'] = self._generate_overall_assessment(results)
        
        # Consolidated recommendations
        results['recommendations'] = self._generate_consolidated_recommendations(results)
        
        self.evaluation_results = results
        return results
    
    def _generate_overall_assessment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate overall fairness assessment.
        
        Args:
            results: Evaluation results
            
        Returns:
            Overall assessment
        """
        assessment = {
            'overall_fairness_score': 0.0,
            'fairness_level': 'UNKNOWN',
            'critical_issues': [],
            'strengths': []
        }
        
        # Collect fairness scores
        fairness_scores = []
        
        # Language consistency score
        lang_consistency = results['language_bias_evaluation'].get('cross_language_consistency', {})
        if 'consistency_score' in lang_consistency:
            consistency_score = 1.0 - min(lang_consistency['consistency_score'], 1.0)
            fairness_scores.append(consistency_score)
        
        # Demographic fairness score
        if 'demographic_bias_evaluation' in results and results['demographic_bias_evaluation']:
            demo_bias = results['demographic_bias_evaluation'].get('bias_metrics', {})
            if 'max_bias_score' in demo_bias:
                demo_fairness = 1.0 - min(demo_bias['max_bias_score'], 1.0)
                fairness_scores.append(demo_fairness)
        
        # Overall fairness metrics score
        fairness_metrics = results.get('fairness_metrics', {})
        if 'overall_fairness_score' in fairness_metrics:
            fairness_scores.append(fairness_metrics['overall_fairness_score'])
        
        # Calculate overall score
        if fairness_scores:
            assessment['overall_fairness_score'] = np.mean(fairness_scores)
        
        # Determine fairness level
        score = assessment['overall_fairness_score']
        if score >= 0.8:
            assessment['fairness_level'] = 'HIGH'
            assessment['strengths'].append("Model demonstrates high fairness across evaluated dimensions.")
        elif score >= 0.6:
            assessment['fairness_level'] = 'MODERATE'
            assessment['strengths'].append("Model shows acceptable fairness with room for improvement.")
        elif score >= 0.4:
            assessment['fairness_level'] = 'LOW'
            assessment['critical_issues'].append("Model shows significant fairness concerns.")
        else:
            assessment['fairness_level'] = 'CRITICAL'
            assessment['critical_issues'].append("Model has critical fairness violations.")
        
        return assessment
    
    def _generate_consolidated_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """
        Generate consolidated recommendations from all evaluations.
        
        Args:
            results: All evaluation results
            
        Returns:
            List of consolidated recommendations
        """
        recommendations = []
        
        # Add language bias recommendations
        lang_recs = results['language_bias_evaluation'].get('recommendations', [])
        recommendations.extend([f"[LANGUAGE] {rec}" for rec in lang_recs])
        
        # Add demographic bias recommendations
        if 'demographic_bias_evaluation' in results and results['demographic_bias_evaluation']:
            demo_recs = results['demographic_bias_evaluation'].get('recommendations', [])
            recommendations.extend([f"[DEMOGRAPHIC] {rec}" for rec in demo_recs])
        
        # Add general fairness recommendations
        overall_assessment = results.get('overall_fairness_assessment', {})
        if overall_assessment.get('fairness_level') in ['LOW', 'CRITICAL']:
            recommendations.append("[GENERAL] Consider implementing bias mitigation techniques during training.")
            recommendations.append("[GENERAL] Increase diversity in training data across all protected attributes.")
            recommendations.append("[GENERAL] Implement post-processing fairness corrections.")
        
        return recommendations
    
    def generate_fairness_report(self, results: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a comprehensive fairness evaluation report.
        
        Args:
            results: Optional evaluation results (uses stored results if None)
            
        Returns:
            Formatted report string
        """
        if results is None:
            results = self.evaluation_results
        
        if not results:
            return "No evaluation results available. Run comprehensive_fairness_evaluation() first."
        
        report = []
        report.append("Comprehensive Fairness Evaluation Report")
        report.append("=" * 50)
        report.append("")
        
        # Metadata
        metadata = results.get('evaluation_metadata', {})
        report.append(f"Total Samples: {metadata.get('total_samples', 'N/A')}")
        report.append(f"Languages: {', '.join(metadata.get('unique_languages', []))}")
        report.append(f"Evaluation Date: {metadata.get('evaluation_timestamp', 'N/A')}")
        report.append("")
        
        # Overall Assessment
        assessment = results.get('overall_fairness_assessment', {})
        report.append("OVERALL FAIRNESS ASSESSMENT")
        report.append("-" * 30)
        report.append(f"Fairness Level: {assessment.get('fairness_level', 'UNKNOWN')}")
        report.append(f"Overall Score: {assessment.get('overall_fairness_score', 0.0):.3f}")
        report.append("")
        
        # Critical Issues
        critical_issues = assessment.get('critical_issues', [])
        if critical_issues:
            report.append("CRITICAL ISSUES:")
            for issue in critical_issues:
                report.append(f"• {issue}")
            report.append("")
        
        # Strengths
        strengths = assessment.get('strengths', [])
        if strengths:
            report.append("STRENGTHS:")
            for strength in strengths:
                report.append(f"• {strength}")
            report.append("")
        
        # Language Bias Summary
        lang_bias = results.get('language_bias_evaluation', {})
        if lang_bias:
            report.append("LANGUAGE BIAS ANALYSIS")
            report.append("-" * 25)
            
            consistency = lang_bias.get('cross_language_consistency', {})
            if consistency:
                report.append(f"Cross-Language Consistency Score: {consistency.get('consistency_score', 0.0):.3f}")
                report.append(f"Max Accuracy Difference: {consistency.get('max_accuracy_difference', 0.0):.3f}")
                report.append(f"Max FPR Difference: {consistency.get('max_fpr_difference', 0.0):.3f}")
            report.append("")
        
        # Recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            report.append("RECOMMENDATIONS")
            report.append("-" * 15)
            for rec in recommendations:
                report.append(f"• {rec}")
        
        return "\n".join(report)
    
    def save_evaluation_results(self, filepath: str, results: Optional[Dict[str, Any]] = None):
        """
        Save evaluation results to file.
        
        Args:
            filepath: Path to save results
            results: Optional results to save (uses stored results if None)
        """
        if results is None:
            results = self.evaluation_results
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Fairness evaluation results saved to {filepath}")

# Example usage and testing functions
def demo_fairness_evaluation():
    """
    Demonstrate the fairness evaluation framework capabilities.
    """
    print("Comprehensive Fairness Evaluation Framework Demo")
    print("=" * 60)
    
    # Initialize framework
    evaluator = ComprehensiveFairnessEvaluator()
    
    print("\nFramework components:")
    print("- DemographicBiasDetector: Detects bias across demographic groups")
    print("- LanguageBiasEvaluator: Evaluates bias across languages and linguistic styles")
    print("- FairnessMetricsCalculator: Calculates comprehensive fairness metrics")
    print("- ComprehensiveFairnessEvaluator: Orchestrates all evaluations")
    
    print("\nSupported fairness metrics:")
    print("- Demographic Parity")
    print("- Equalized Odds")
    print("- Equal Opportunity")
    print("- Calibration Metrics")
    print("- Cross-Language Consistency")
    print("- Linguistic Style Bias")
    
    print("\nNote: Full evaluation requires real datasets with predictions and labels.")
    print("This framework provides comprehensive bias detection and mitigation guidance.")

if __name__ == "__main__":
    demo_fairness_evaluation()