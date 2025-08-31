#!/usr/bin/env python3
"""
Evaluation Metrics System for Reddit Comment Moderation

This module implements comprehensive evaluation metrics including column-averaged AUC,
cross-validation, and performance monitoring for the Kaggle competition.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, 
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, average_precision_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import psutil
import logging
from pathlib import Path
import json
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    column_averaged_auc: float
    individual_aucs: Dict[str, float]
    precision_scores: Dict[str, float]
    recall_scores: Dict[str, float]
    f1_scores: Dict[str, float]
    average_precision_scores: Dict[str, float]
    confusion_matrices: Dict[str, np.ndarray]
    roc_curves: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
    pr_curves: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
    evaluation_time: float
    memory_usage: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'column_averaged_auc': self.column_averaged_auc,
            'individual_aucs': self.individual_aucs,
            'precision_scores': self.precision_scores,
            'recall_scores': self.recall_scores,
            'f1_scores': self.f1_scores,
            'average_precision_scores': self.average_precision_scores,
            'evaluation_time': self.evaluation_time,
            'memory_usage': self.memory_usage
        }

class PerformanceMonitor:
    """Monitor system performance during evaluation."""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.process = psutil.Process()
    
    def start(self):
        """Start monitoring."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
    
    def stop(self) -> Tuple[float, float]:
        """Stop monitoring and return elapsed time and memory usage."""
        if self.start_time is None:
            return 0.0, 0.0
        
        elapsed_time = time.time() - self.start_time
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = current_memory - self.start_memory
        
        return elapsed_time, memory_usage

class KaggleEvaluator:
    """Main evaluator for Kaggle competition metrics."""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.monitor = PerformanceMonitor()
    
    def calculate_column_averaged_auc(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    column_names: Optional[List[str]] = None) -> Tuple[float, Dict[str, float]]:
        """Calculate column-averaged AUC as specified in Kaggle competition."""
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
        
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
            y_pred = y_pred.reshape(-1, 1)
        
        n_columns = y_true.shape[1]
        if column_names is None:
            column_names = [f'column_{i}' for i in range(n_columns)]
        
        individual_aucs = {}
        valid_aucs = []
        
        for i, col_name in enumerate(column_names):
            y_true_col = y_true[:, i]
            y_pred_col = y_pred[:, i]
            
            # Check if both classes are present
            unique_labels = np.unique(y_true_col)
            if len(unique_labels) < 2:
                logger.warning(f"Column {col_name} has only one class. Skipping AUC calculation.")
                individual_aucs[col_name] = np.nan
                continue
            
            try:
                auc = roc_auc_score(y_true_col, y_pred_col)
                individual_aucs[col_name] = auc
                valid_aucs.append(auc)
            except Exception as e:
                logger.warning(f"Error calculating AUC for column {col_name}: {e}")
                individual_aucs[col_name] = np.nan
        
        # Calculate column-averaged AUC
        column_averaged_auc = np.mean(valid_aucs) if valid_aucs else 0.0
        
        return column_averaged_auc, individual_aucs
    
    def comprehensive_evaluation(self, y_true: np.ndarray, y_pred: np.ndarray,
                               column_names: Optional[List[str]] = None) -> EvaluationResult:
        """Perform comprehensive evaluation with all metrics."""
        self.monitor.start()
        
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
            y_pred = y_pred.reshape(-1, 1)
        
        n_columns = y_true.shape[1]
        if column_names is None:
            column_names = [f'column_{i}' for i in range(n_columns)]
        
        # Calculate column-averaged AUC
        column_averaged_auc, individual_aucs = self.calculate_column_averaged_auc(
            y_true, y_pred, column_names
        )
        
        # Initialize result containers
        precision_scores = {}
        recall_scores = {}
        f1_scores = {}
        average_precision_scores = {}
        confusion_matrices = {}
        roc_curves = {}
        pr_curves = {}
        
        # Calculate metrics for each column
        for i, col_name in enumerate(column_names):
            y_true_col = y_true[:, i]
            y_pred_col = y_pred[:, i]
            y_pred_binary = (y_pred_col >= self.threshold).astype(int)
            
            # Skip if only one class present
            if len(np.unique(y_true_col)) < 2:
                continue
            
            try:
                # Precision, Recall, F1
                precision_scores[col_name] = precision_score(y_true_col, y_pred_binary, zero_division=0)
                recall_scores[col_name] = recall_score(y_true_col, y_pred_binary, zero_division=0)
                f1_scores[col_name] = f1_score(y_true_col, y_pred_binary, zero_division=0)
                
                # Average Precision (PR AUC)
                average_precision_scores[col_name] = average_precision_score(y_true_col, y_pred_col)
                
                # Confusion Matrix
                confusion_matrices[col_name] = confusion_matrix(y_true_col, y_pred_binary)
                
                # ROC Curve
                fpr, tpr, roc_thresholds = roc_curve(y_true_col, y_pred_col)
                roc_curves[col_name] = (fpr, tpr, roc_thresholds)
                
                # Precision-Recall Curve
                precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true_col, y_pred_col)
                pr_curves[col_name] = (precision_curve, recall_curve, pr_thresholds)
                
            except Exception as e:
                logger.warning(f"Error calculating metrics for column {col_name}: {e}")
        
        # Stop monitoring
        evaluation_time, memory_usage = self.monitor.stop()
        
        return EvaluationResult(
            column_averaged_auc=column_averaged_auc,
            individual_aucs=individual_aucs,
            precision_scores=precision_scores,
            recall_scores=recall_scores,
            f1_scores=f1_scores,
            average_precision_scores=average_precision_scores,
            confusion_matrices=confusion_matrices,
            roc_curves=roc_curves,
            pr_curves=pr_curves,
            evaluation_time=evaluation_time,
            memory_usage=memory_usage
        )
    
    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray, 
                           cv_folds: int = 5, column_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Perform cross-validation with column-averaged AUC."""
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Use first column for stratification if multi-label
        if len(y.shape) > 1 and y.shape[1] > 1:
            y_stratify = y[:, 0]
        else:
            y_stratify = y.ravel()
        
        cv_scores = []
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_stratify)):
            logger.info(f"Cross-validation fold {fold + 1}/{cv_folds}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict_proba(X_val)
            if hasattr(y_pred, '__len__') and len(y_pred) > 1 and hasattr(y_pred[0], 'shape'):
                # Handle multi-output case
                y_pred = np.column_stack([pred[:, 1] for pred in y_pred])
            elif len(y_pred.shape) > 1 and y_pred.shape[1] == 2:
                y_pred = y_pred[:, 1]
            
            # Evaluate
            fold_result = self.comprehensive_evaluation(y_val, y_pred, column_names)
            fold_results.append(fold_result)
            cv_scores.append(fold_result.column_averaged_auc)
            
            logger.info(f"Fold {fold + 1} AUC: {fold_result.column_averaged_auc:.4f}")
        
        return {
            'mean_auc': np.mean(cv_scores),
            'std_auc': np.std(cv_scores),
            'fold_scores': cv_scores,
            'fold_results': fold_results,
            'confidence_interval': stats.t.interval(
                0.95, len(cv_scores)-1, 
                loc=np.mean(cv_scores), 
                scale=stats.sem(cv_scores)
            )
        }
    
    def validate_submission_format(self, submission_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate Kaggle submission format."""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required columns
        required_columns = ['row_id', 'rule_violation']
        for col in required_columns:
            if col not in submission_df.columns:
                validation_results['is_valid'] = False
                validation_results['errors'].append(f"Missing required column: {col}")
        
        if not validation_results['is_valid']:
            return validation_results
        
        # Check data types
        if not pd.api.types.is_numeric_dtype(submission_df['rule_violation']):
            validation_results['is_valid'] = False
            validation_results['errors'].append("rule_violation column must be numeric")
        
        # Check value ranges
        rule_violation_values = submission_df['rule_violation']
        if rule_violation_values.min() < 0 or rule_violation_values.max() > 1:
            validation_results['warnings'].append(
                "rule_violation values should be probabilities between 0 and 1"
            )
        
        # Check for missing values
        if submission_df.isnull().any().any():
            validation_results['is_valid'] = False
            validation_results['errors'].append("Submission contains missing values")
        
        # Check for duplicates
        if submission_df['row_id'].duplicated().any():
            validation_results['is_valid'] = False
            validation_results['errors'].append("Duplicate row_ids found")
        
        return validation_results

class EvaluationVisualizer:
    """Create visualizations for evaluation results."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')
    
    def plot_roc_curves(self, evaluation_result: EvaluationResult, save_path: Optional[str] = None):
        """Plot ROC curves for all columns."""
        n_curves = len(evaluation_result.roc_curves)
        if n_curves == 0:
            logger.warning("No ROC curves to plot")
            return
        
        fig, axes = plt.subplots(1, min(n_curves, 3), figsize=self.figsize)
        if n_curves == 1:
            axes = [axes]
        
        for i, (col_name, (fpr, tpr, _)) in enumerate(evaluation_result.roc_curves.items()):
            if i >= 3:  # Limit to 3 plots
                break
            
            auc_score = evaluation_result.individual_aucs.get(col_name, 0)
            axes[i].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
            axes[i].plot([0, 1], [0, 1], 'k--', label='Random')
            axes[i].set_xlabel('False Positive Rate')
            axes[i].set_ylabel('True Positive Rate')
            axes[i].set_title(f'ROC Curve - {col_name}')
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curves(self, evaluation_result: EvaluationResult, save_path: Optional[str] = None):
        """Plot Precision-Recall curves for all columns."""
        n_curves = len(evaluation_result.pr_curves)
        if n_curves == 0:
            logger.warning("No PR curves to plot")
            return
        
        fig, axes = plt.subplots(1, min(n_curves, 3), figsize=self.figsize)
        if n_curves == 1:
            axes = [axes]
        
        for i, (col_name, (precision, recall, _)) in enumerate(evaluation_result.pr_curves.items()):
            if i >= 3:  # Limit to 3 plots
                break
            
            ap_score = evaluation_result.average_precision_scores.get(col_name, 0)
            axes[i].plot(recall, precision, label=f'PR Curve (AP = {ap_score:.3f})')
            axes[i].set_xlabel('Recall')
            axes[i].set_ylabel('Precision')
            axes[i].set_title(f'Precision-Recall Curve - {col_name}')
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrices(self, evaluation_result: EvaluationResult, save_path: Optional[str] = None):
        """Plot confusion matrices for all columns."""
        n_matrices = len(evaluation_result.confusion_matrices)
        if n_matrices == 0:
            logger.warning("No confusion matrices to plot")
            return
        
        fig, axes = plt.subplots(1, min(n_matrices, 3), figsize=self.figsize)
        if n_matrices == 1:
            axes = [axes]
        
        for i, (col_name, cm) in enumerate(evaluation_result.confusion_matrices.items()):
            if i >= 3:  # Limit to 3 plots
                break
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'Confusion Matrix - {col_name}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_metrics_comparison(self, evaluation_result: EvaluationResult, save_path: Optional[str] = None):
        """Plot comparison of different metrics across columns."""
        columns = list(evaluation_result.individual_aucs.keys())
        if not columns:
            logger.warning("No metrics to plot")
            return
        
        metrics_data = {
            'AUC': [evaluation_result.individual_aucs.get(col, 0) for col in columns],
            'Precision': [evaluation_result.precision_scores.get(col, 0) for col in columns],
            'Recall': [evaluation_result.recall_scores.get(col, 0) for col in columns],
            'F1': [evaluation_result.f1_scores.get(col, 0) for col in columns],
            'AP': [evaluation_result.average_precision_scores.get(col, 0) for col in columns]
        }
        
        x = np.arange(len(columns))
        width = 0.15
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for i, (metric, values) in enumerate(metrics_data.items()):
            ax.bar(x + i * width, values, width, label=metric)
        
        ax.set_xlabel('Columns')
        ax.set_ylabel('Score')
        ax.set_title('Metrics Comparison Across Columns')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(columns, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Example usage of the evaluation system."""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_columns = 3
    
    # Generate synthetic data
    y_true = np.random.randint(0, 2, size=(n_samples, n_columns))
    y_pred = np.random.random(size=(n_samples, n_columns))
    
    # Add some correlation to make it more realistic
    for i in range(n_columns):
        mask = y_true[:, i] == 1
        y_pred[mask, i] += 0.3
        y_pred[:, i] = np.clip(y_pred[:, i], 0, 1)
    
    column_names = ['harassment', 'spam', 'off_topic']
    
    # Initialize evaluator
    evaluator = KaggleEvaluator()
    
    # Perform comprehensive evaluation
    logger.info("Performing comprehensive evaluation...")
    result = evaluator.comprehensive_evaluation(y_true, y_pred, column_names)
    
    # Print results
    logger.info(f"Column-averaged AUC: {result.column_averaged_auc:.4f}")
    logger.info(f"Individual AUCs: {result.individual_aucs}")
    logger.info(f"Evaluation time: {result.evaluation_time:.2f} seconds")
    logger.info(f"Memory usage: {result.memory_usage:.2f} MB")
    
    # Create visualizations
    visualizer = EvaluationVisualizer()
    visualizer.plot_roc_curves(result)
    visualizer.plot_precision_recall_curves(result)
    visualizer.plot_confusion_matrices(result)
    visualizer.plot_metrics_comparison(result)
    
    # Validate sample submission
    submission_df = pd.DataFrame({
        'row_id': range(100),
        'rule_violation': np.random.random(100)
    })
    
    validation_result = evaluator.validate_submission_format(submission_df)
    logger.info(f"Submission validation: {validation_result}")

if __name__ == '__main__':
    main()