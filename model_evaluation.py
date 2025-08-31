#!/usr/bin/env python3
"""
Model Evaluation Module for Binary Text Classification
Provides comprehensive evaluation metrics and analysis tools
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    average_precision_score
)
from sklearn.calibration import calibration_curve
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    Comprehensive model evaluation for binary classification
    """
    
    def __init__(self):
        self.evaluation_results = {}
        
    def evaluate_single_model(self, 
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             y_pred_proba: Optional[np.ndarray] = None,
                             model_name: str = "model") -> Dict[str, Any]:
        """
        Evaluate a single model with comprehensive metrics
        """
        results = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        # Add probability-based metrics if available
        if y_pred_proba is not None:
            results.update({
                'roc_auc': roc_auc_score(y_true, y_pred_proba),
                'average_precision': average_precision_score(y_true, y_pred_proba)
            })
        
        # Store detailed classification report
        results['classification_report'] = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        
        self.evaluation_results[model_name] = results
        return results
    
    def evaluate_multiple_models(self,
                               y_true: np.ndarray,
                               predictions: Dict[str, np.ndarray],
                               probabilities: Optional[Dict[str, np.ndarray]] = None) -> pd.DataFrame:
        """
        Evaluate multiple models and return comparison DataFrame
        """
        if probabilities is None:
            probabilities = {}
        
        comparison_data = []
        
        for model_name, y_pred in predictions.items():
            y_pred_proba = probabilities.get(model_name)
            
            results = self.evaluate_single_model(
                y_true, y_pred, y_pred_proba, model_name
            )
            
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1 Score': results['f1_score'],
                'ROC AUC': results.get('roc_auc', 'N/A'),
                'Avg Precision': results.get('average_precision', 'N/A')
            })
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values('F1 Score', ascending=False)
    
    def plot_confusion_matrices(self, 
                              models: Optional[List[str]] = None,
                              figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot confusion matrices for specified models
        """
        if models is None:
            models = list(self.evaluation_results.keys())
        
        n_models = len(models)
        if n_models == 0:
            print("No models to plot")
            return
        
        # Calculate subplot layout
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, model_name in enumerate(models):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            if model_name in self.evaluation_results:
                cm = self.evaluation_results[model_name]['confusion_matrix']
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=['Safe', 'Violation'],
                           yticklabels=['Safe', 'Violation'])
                ax.set_title(f'{model_name}\nConfusion Matrix')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
        
        # Hide empty subplots
        for i in range(n_models, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self,
                       y_true: np.ndarray,
                       probabilities: Dict[str, np.ndarray],
                       figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot ROC curves for multiple models
        """
        plt.figure(figsize=figsize)
        
        for model_name, y_pred_proba in probabilities.items():
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            auc_score = roc_auc_score(y_true, y_pred_proba)
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_precision_recall_curves(self,
                                   y_true: np.ndarray,
                                   probabilities: Dict[str, np.ndarray],
                                   figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot Precision-Recall curves for multiple models
        """
        plt.figure(figsize=figsize)
        
        for model_name, y_pred_proba in probabilities.items():
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            avg_precision = average_precision_score(y_true, y_pred_proba)
            
            plt.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.3f})', linewidth=2)
        
        # Baseline (random classifier)
        baseline = np.mean(y_true)
        plt.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, 
                   label=f'Random Classifier (AP = {baseline:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_calibration_curves(self,
                              y_true: np.ndarray,
                              probabilities: Dict[str, np.ndarray],
                              n_bins: int = 10,
                              figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot calibration curves to assess probability calibration
        """
        plt.figure(figsize=figsize)
        
        for model_name, y_pred_proba in probabilities.items():
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred_proba, n_bins=n_bins
            )
            
            plt.plot(mean_predicted_value, fraction_of_positives, "s-",
                    label=f'{model_name}', linewidth=2, markersize=8)
        
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def analyze_errors(self,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      texts: List[str],
                      model_name: str = "model",
                      n_examples: int = 10) -> Dict[str, List[Dict]]:
        """
        Analyze prediction errors with text examples
        """
        # Get prediction errors
        errors = y_true != y_pred
        
        # False positives (predicted violation, actually safe)
        fp_mask = (y_pred == 1) & (y_true == 0)
        fp_indices = np.where(fp_mask)[0]
        
        # False negatives (predicted safe, actually violation)
        fn_mask = (y_pred == 0) & (y_true == 1)
        fn_indices = np.where(fn_mask)[0]
        
        # Get examples
        fp_examples = []
        for i in fp_indices[:n_examples]:
            fp_examples.append({
                'text': texts[i],
                'true_label': int(y_true[i]),
                'predicted_label': int(y_pred[i]),
                'index': int(i)
            })
        
        fn_examples = []
        for i in fn_indices[:n_examples]:
            fn_examples.append({
                'text': texts[i],
                'true_label': int(y_true[i]),
                'predicted_label': int(y_pred[i]),
                'index': int(i)
            })
        
        error_analysis = {
            'model_name': model_name,
            'total_errors': int(np.sum(errors)),
            'false_positives': int(np.sum(fp_mask)),
            'false_negatives': int(np.sum(fn_mask)),
            'fp_examples': fp_examples,
            'fn_examples': fn_examples
        }
        
        return error_analysis
    
    def print_detailed_report(self, model_name: str) -> None:
        """
        Print detailed evaluation report for a model
        """
        if model_name not in self.evaluation_results:
            print(f"Model {model_name} not found in evaluation results")
            return
        
        results = self.evaluation_results[model_name]
        
        print(f"\n{'='*50}")
        print(f"DETAILED EVALUATION REPORT: {model_name.upper()}")
        print(f"{'='*50}")
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1 Score:  {results['f1_score']:.4f}")
        
        if 'roc_auc' in results:
            print(f"  ROC AUC:   {results['roc_auc']:.4f}")
        if 'average_precision' in results:
            print(f"  Avg Precision: {results['average_precision']:.4f}")
        
        print(f"\nConfusion Matrix:")
        cm = results['confusion_matrix']
        print(f"  True Negatives:  {cm[0,0]}")
        print(f"  False Positives: {cm[0,1]}")
        print(f"  False Negatives: {cm[1,0]}")
        print(f"  True Positives:  {cm[1,1]}")
        
        print(f"\nPer-Class Metrics:")
        report = results['classification_report']
        for class_name, metrics in report.items():
            if class_name in ['0', '1']:
                class_label = 'Safe' if class_name == '0' else 'Violation'
                print(f"  {class_label}:")
                print(f"    Precision: {metrics['precision']:.4f}")
                print(f"    Recall:    {metrics['recall']:.4f}")
                print(f"    F1-score:  {metrics['f1-score']:.4f}")
                print(f"    Support:   {metrics['support']}")
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics for all evaluated models
        """
        if not self.evaluation_results:
            return pd.DataFrame()
        
        summary_data = []
        
        for model_name, results in self.evaluation_results.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1 Score': results['f1_score'],
                'ROC AUC': results.get('roc_auc', 'N/A'),
                'Avg Precision': results.get('average_precision', 'N/A')
            })
        
        df = pd.DataFrame(summary_data)
        return df.sort_values('F1 Score', ascending=False)
    
    def export_results(self, filepath: str) -> None:
        """
        Export evaluation results to file
        """
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        export_data = {}
        for model_name, results in self.evaluation_results.items():
            export_results = results.copy()
            export_results['confusion_matrix'] = results['confusion_matrix'].tolist()
            export_data[model_name] = export_results
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"Evaluation results exported to {filepath}")

# Utility functions for quick evaluation
def quick_evaluate(y_true: np.ndarray, 
                  y_pred: np.ndarray, 
                  y_pred_proba: Optional[np.ndarray] = None,
                  model_name: str = "model") -> None:
    """
    Quick evaluation with printed results
    """
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_single_model(y_true, y_pred, y_pred_proba, model_name)
    evaluator.print_detailed_report(model_name)

def compare_models(y_true: np.ndarray,
                  predictions: Dict[str, np.ndarray],
                  probabilities: Optional[Dict[str, np.ndarray]] = None) -> pd.DataFrame:
    """
    Quick model comparison
    """
    evaluator = ModelEvaluator()
    comparison = evaluator.evaluate_multiple_models(y_true, predictions, probabilities)
    print("\nModel Comparison:")
    print(comparison.to_string(index=False))
    return comparison

# Example usage
if __name__ == "__main__":
    # Generate sample data for demonstration
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    
    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train sample models
    lr = LogisticRegression(random_state=42)
    nb = MultinomialNB()
    
    lr.fit(X_train, y_train)
    nb.fit(X_train, y_train)
    
    # Make predictions
    lr_pred = lr.predict(X_test)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    
    nb_pred = nb.predict(X_test)
    nb_proba = nb.predict_proba(X_test)[:, 1]
    
    # Evaluate models
    evaluator = ModelEvaluator()
    
    # Single model evaluation
    evaluator.evaluate_single_model(y_test, lr_pred, lr_proba, "Logistic Regression")
    evaluator.evaluate_single_model(y_test, nb_pred, nb_proba, "Naive Bayes")
    
    # Print detailed reports
    evaluator.print_detailed_report("Logistic Regression")
    evaluator.print_detailed_report("Naive Bayes")
    
    # Model comparison
    predictions = {
        "Logistic Regression": lr_pred,
        "Naive Bayes": nb_pred
    }
    
    probabilities = {
        "Logistic Regression": lr_proba,
        "Naive Bayes": nb_proba
    }
    
    comparison = evaluator.evaluate_multiple_models(y_test, predictions, probabilities)
    print("\nModel Comparison:")
    print(comparison)