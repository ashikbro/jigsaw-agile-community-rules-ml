import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    precision_recall_curve, roc_curve, f1_score, 
    precision_score, recall_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Try to import SMOTE (optional dependency)
try:
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
    from imblearn.combine import SMOTETomek, SMOTEENN
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    IMBALANCED_LEARN_AVAILABLE = False
    print("imbalanced-learn not available. Install with: pip install imbalanced-learn")

class SparseDataSMOTE:
    """
    Custom SMOTE implementation optimized for sparse TF-IDF data.
    """
    
    def __init__(self, 
                 sampling_strategy: Union[str, float, Dict] = 'auto',
                 k_neighbors: int = 5,
                 random_state: Optional[int] = None):
        """
        Initialize Sparse Data SMOTE.
        
        Args:
            sampling_strategy: Sampling strategy for SMOTE
            k_neighbors: Number of nearest neighbors
            random_state: Random state for reproducibility
        """
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit_resample(self, X: Union[np.ndarray, sparse.csr_matrix], y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit and resample the data using sparse-optimized SMOTE.
        
        Args:
            X: Feature matrix (can be sparse)
            y: Target labels
            
        Returns:
            Resampled X and y
        """
        if IMBALANCED_LEARN_AVAILABLE:
            # Use standard SMOTE if available
            smote = SMOTE(
                sampling_strategy=self.sampling_strategy,
                k_neighbors=self.k_neighbors,
                random_state=self.random_state
            )
            return smote.fit_resample(X, y)
        else:
            # Fallback to simple oversampling for sparse data
            return self._simple_oversample(X, y)
    
    def _simple_oversample(self, X: Union[np.ndarray, sparse.csr_matrix], y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple oversampling fallback for when SMOTE is not available.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Oversampled X and y
        """
        # Count classes
        class_counts = Counter(y)
        majority_class = max(class_counts, key=class_counts.get)
        majority_count = class_counts[majority_class]
        
        X_resampled = []
        y_resampled = []
        
        for class_label in class_counts:
            class_indices = np.where(y == class_label)[0]
            class_X = X[class_indices]
            
            if class_counts[class_label] < majority_count:
                # Oversample minority class
                n_samples_needed = majority_count - class_counts[class_label]
                
                # Random oversampling with noise
                oversample_indices = np.random.choice(
                    len(class_indices), 
                    size=n_samples_needed, 
                    replace=True
                )
                
                if sparse.issparse(X):
                    oversampled_X = class_X[oversample_indices]
                    # Add small random noise to sparse data
                    noise = sparse.random(
                        oversampled_X.shape[0], 
                        oversampled_X.shape[1], 
                        density=0.01, 
                        random_state=self.random_state
                    ) * 0.1
                    oversampled_X = oversampled_X + noise
                else:
                    oversampled_X = class_X[oversample_indices]
                    # Add small random noise
                    noise = np.random.normal(0, 0.01, oversampled_X.shape)
                    oversampled_X = oversampled_X + noise
                
                X_resampled.append(class_X)
                X_resampled.append(oversampled_X)
                y_resampled.extend([class_label] * class_counts[class_label])
                y_resampled.extend([class_label] * n_samples_needed)
            else:
                X_resampled.append(class_X)
                y_resampled.extend([class_label] * class_counts[class_label])
        
        if sparse.issparse(X):
            X_final = sparse.vstack(X_resampled)
        else:
            X_final = np.vstack(X_resampled)
        
        y_final = np.array(y_resampled)
        
        return X_final, y_final

class CostSensitiveLearning:
    """
    Cost-sensitive learning implementation with automatic class weight computation.
    """
    
    def __init__(self, 
                 cost_matrix: Optional[np.ndarray] = None,
                 class_weight_method: str = 'balanced'):
        """
        Initialize cost-sensitive learning.
        
        Args:
            cost_matrix: Custom cost matrix (optional)
            class_weight_method: Method for computing class weights
        """
        self.cost_matrix = cost_matrix
        self.class_weight_method = class_weight_method
        self.class_weights_ = None
    
    def compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Compute class weights based on the target distribution.
        
        Args:
            y: Target labels
            
        Returns:
            Dictionary of class weights
        """
        if self.class_weight_method == 'balanced':
            classes = np.unique(y)
            weights = compute_class_weight('balanced', classes=classes, y=y)
            self.class_weights_ = dict(zip(classes, weights))
        
        elif self.class_weight_method == 'balanced_subsample':
            # Custom balanced approach for imbalanced data
            class_counts = Counter(y)
            total_samples = len(y)
            n_classes = len(class_counts)
            
            weights = {}
            for class_label, count in class_counts.items():
                weights[class_label] = total_samples / (n_classes * count)
            
            self.class_weights_ = weights
        
        elif self.class_weight_method == 'inverse_frequency':
            # Inverse frequency weighting
            class_counts = Counter(y)
            total_samples = len(y)
            
            weights = {}
            for class_label, count in class_counts.items():
                weights[class_label] = total_samples / count
            
            self.class_weights_ = weights
        
        elif self.class_weight_method == 'custom' and self.cost_matrix is not None:
            # Use custom cost matrix
            classes = np.unique(y)
            weights = {}
            
            for i, class_label in enumerate(classes):
                # Weight based on misclassification costs
                weights[class_label] = np.sum(self.cost_matrix[i, :]) / len(classes)
            
            self.class_weights_ = weights
        
        else:
            # Equal weights
            classes = np.unique(y)
            self.class_weights_ = {class_label: 1.0 for class_label in classes}
        
        return self.class_weights_
    
    def get_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """
        Get sample weights based on class weights.
        
        Args:
            y: Target labels
            
        Returns:
            Array of sample weights
        """
        if self.class_weights_ is None:
            self.compute_class_weights(y)
        
        sample_weights = np.array([self.class_weights_[label] for label in y])
        return sample_weights

class ThresholdOptimizer:
    """
    Optimize classification threshold for better performance on imbalanced data.
    """
    
    def __init__(self, 
                 metric: str = 'f1',
                 cv_folds: int = 5,
                 threshold_range: Tuple[float, float] = (0.1, 0.9),
                 n_thresholds: int = 100):
        """
        Initialize threshold optimizer.
        
        Args:
            metric: Metric to optimize ('f1', 'precision', 'recall', 'balanced_accuracy')
            cv_folds: Number of cross-validation folds
            threshold_range: Range of thresholds to test
            n_thresholds: Number of thresholds to test
        """
        self.metric = metric
        self.cv_folds = cv_folds
        self.threshold_range = threshold_range
        self.n_thresholds = n_thresholds
        self.optimal_threshold_ = 0.5
        self.threshold_scores_ = None
    
    def optimize_threshold(self, 
                          model: BaseEstimator,
                          X: np.ndarray,
                          y: np.ndarray) -> float:
        """
        Find optimal threshold using cross-validation.
        
        Args:
            model: Trained model with predict_proba method
            X: Feature matrix
            y: Target labels
            
        Returns:
            Optimal threshold
        """
        if not hasattr(model, 'predict_proba'):
            print("Model does not have predict_proba method. Using default threshold 0.5")
            return 0.5
        
        # Generate threshold candidates
        thresholds = np.linspace(
            self.threshold_range[0], 
            self.threshold_range[1], 
            self.n_thresholds
        )
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        threshold_scores = []
        
        for threshold in thresholds:
            scores = []
            
            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Fit model on training data
                model.fit(X_train, y_train)
                
                # Get probabilities on validation data
                y_proba = model.predict_proba(X_val)[:, 1]
                
                # Apply threshold
                y_pred = (y_proba >= threshold).astype(int)
                
                # Calculate metric
                if self.metric == 'f1':
                    score = f1_score(y_val, y_pred)
                elif self.metric == 'precision':
                    score = precision_score(y_val, y_pred, zero_division=0)
                elif self.metric == 'recall':
                    score = recall_score(y_val, y_pred, zero_division=0)
                elif self.metric == 'balanced_accuracy':
                    from sklearn.metrics import balanced_accuracy_score
                    score = balanced_accuracy_score(y_val, y_pred)
                else:
                    score = f1_score(y_val, y_pred)
                
                scores.append(score)
            
            threshold_scores.append(np.mean(scores))
        
        # Find optimal threshold
        optimal_idx = np.argmax(threshold_scores)
        self.optimal_threshold_ = thresholds[optimal_idx]
        self.threshold_scores_ = list(zip(thresholds, threshold_scores))
        
        return self.optimal_threshold_
    
    def plot_threshold_curve(self, save_path: Optional[str] = None):
        """
        Plot threshold optimization curve.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if self.threshold_scores_ is None:
            print("No threshold scores available. Run optimize_threshold first.")
            return
        
        thresholds, scores = zip(*self.threshold_scores_)
        
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, scores, 'b-', linewidth=2)
        plt.axvline(x=self.optimal_threshold_, color='r', linestyle='--', 
                   label=f'Optimal Threshold: {self.optimal_threshold_:.3f}')
        plt.xlabel('Threshold')
        plt.ylabel(f'{self.metric.capitalize()} Score')
        plt.title(f'Threshold Optimization - {self.metric.capitalize()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

class AdvancedImbalanceHandler:
    """
    Comprehensive imbalance handling with multiple strategies.
    """
    
    def __init__(self, 
                 strategy: str = 'combined',
                 sampling_method: str = 'smote',
                 cost_sensitive: bool = True,
                 threshold_optimization: bool = True,
                 random_state: Optional[int] = None):
        """
        Initialize advanced imbalance handler.
        
        Args:
            strategy: Overall strategy ('sampling', 'cost_sensitive', 'threshold', 'combined')
            sampling_method: Sampling method ('smote', 'borderline_smote', 'adasyn', 'random')
            cost_sensitive: Whether to use cost-sensitive learning
            threshold_optimization: Whether to optimize threshold
            random_state: Random state for reproducibility
        """
        self.strategy = strategy
        self.sampling_method = sampling_method
        self.cost_sensitive = cost_sensitive
        self.threshold_optimization = threshold_optimization
        self.random_state = random_state
        
        # Initialize components
        self.sampler = None
        self.cost_learner = None
        self.threshold_optimizer = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """
        Initialize imbalance handling components.
        """
        # Initialize sampler
        if self.strategy in ['sampling', 'combined']:
            if self.sampling_method == 'smote':
                self.sampler = SparseDataSMOTE(random_state=self.random_state)
            elif self.sampling_method == 'borderline_smote' and IMBALANCED_LEARN_AVAILABLE:
                self.sampler = BorderlineSMOTE(random_state=self.random_state)
            elif self.sampling_method == 'adasyn' and IMBALANCED_LEARN_AVAILABLE:
                self.sampler = ADASYN(random_state=self.random_state)
            else:
                self.sampler = SparseDataSMOTE(random_state=self.random_state)
        
        # Initialize cost-sensitive learning
        if self.strategy in ['cost_sensitive', 'combined'] and self.cost_sensitive:
            self.cost_learner = CostSensitiveLearning()
        
        # Initialize threshold optimizer
        if self.strategy in ['threshold', 'combined'] and self.threshold_optimization:
            self.threshold_optimizer = ThresholdOptimizer()
    
    def handle_imbalance(self, 
                        X: np.ndarray, 
                        y: np.ndarray,
                        model: Optional[BaseEstimator] = None) -> Dict[str, Any]:
        """
        Apply imbalance handling strategies.
        
        Args:
            X: Feature matrix
            y: Target labels
            model: Model for threshold optimization (optional)
            
        Returns:
            Dictionary with processed data and parameters
        """
        result = {
            'X_processed': X.copy(),
            'y_processed': y.copy(),
            'class_weights': None,
            'sample_weights': None,
            'optimal_threshold': 0.5,
            'sampling_applied': False,
            'original_distribution': Counter(y),
            'processed_distribution': None
        }
        
        # Apply sampling
        if self.sampler is not None:
            print(f"Applying {self.sampling_method} sampling...")
            X_resampled, y_resampled = self.sampler.fit_resample(
                result['X_processed'], result['y_processed']
            )
            result['X_processed'] = X_resampled
            result['y_processed'] = y_resampled
            result['sampling_applied'] = True
            print(f"Original distribution: {result['original_distribution']}")
            print(f"Resampled distribution: {Counter(y_resampled)}")
        
        result['processed_distribution'] = Counter(result['y_processed'])
        
        # Apply cost-sensitive learning
        if self.cost_learner is not None:
            print("Computing class weights for cost-sensitive learning...")
            class_weights = self.cost_learner.compute_class_weights(result['y_processed'])
            sample_weights = self.cost_learner.get_sample_weights(result['y_processed'])
            
            result['class_weights'] = class_weights
            result['sample_weights'] = sample_weights
            print(f"Class weights: {class_weights}")
        
        # Optimize threshold
        if self.threshold_optimizer is not None and model is not None:
            print("Optimizing classification threshold...")
            optimal_threshold = self.threshold_optimizer.optimize_threshold(
                model, result['X_processed'], result['y_processed']
            )
            result['optimal_threshold'] = optimal_threshold
            print(f"Optimal threshold: {optimal_threshold:.3f}")
        
        return result
    
    def evaluate_strategies(self, 
                          X: np.ndarray, 
                          y: np.ndarray,
                          models: List[BaseEstimator],
                          model_names: List[str]) -> pd.DataFrame:
        """
        Evaluate different imbalance handling strategies.
        
        Args:
            X: Feature matrix
            y: Target labels
            models: List of models to evaluate
            model_names: Names of the models
            
        Returns:
            DataFrame with evaluation results
        """
        strategies = [
            ('baseline', {'strategy': 'none'}),
            ('sampling_only', {'strategy': 'sampling'}),
            ('cost_sensitive_only', {'strategy': 'cost_sensitive'}),
            ('threshold_only', {'strategy': 'threshold'}),
            ('combined', {'strategy': 'combined'})
        ]
        
        results = []
        
        for strategy_name, strategy_params in strategies:
            print(f"\nEvaluating strategy: {strategy_name}")
            
            # Create handler for this strategy
            if strategy_name == 'baseline':
                handler = None
            else:
                handler = AdvancedImbalanceHandler(
                    strategy=strategy_params['strategy'],
                    random_state=self.random_state
                )
            
            for model, model_name in zip(models, model_names):
                # Apply imbalance handling
                if handler is not None:
                    processed_data = handler.handle_imbalance(X, y, model)
                    X_proc = processed_data['X_processed']
                    y_proc = processed_data['y_processed']
                    class_weights = processed_data['class_weights']
                    optimal_threshold = processed_data['optimal_threshold']
                else:
                    X_proc, y_proc = X, y
                    class_weights = None
                    optimal_threshold = 0.5
                
                # Cross-validation evaluation
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = {'f1': [], 'precision': [], 'recall': [], 'roc_auc': []}
                
                for train_idx, val_idx in cv.split(X_proc, y_proc):
                    X_train, X_val = X_proc[train_idx], X_proc[val_idx]
                    y_train, y_val = y_proc[train_idx], y_proc[val_idx]
                    
                    # Fit model
                    if class_weights is not None and hasattr(model, 'class_weight'):
                        model.set_params(class_weight=class_weights)
                    
                    model.fit(X_train, y_train)
                    
                    # Predict
                    if hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(X_val)[:, 1]
                        y_pred = (y_proba >= optimal_threshold).astype(int)
                        cv_scores['roc_auc'].append(roc_auc_score(y_val, y_proba))
                    else:
                        y_pred = model.predict(X_val)
                        cv_scores['roc_auc'].append(0)  # Not available
                    
                    # Calculate metrics
                    cv_scores['f1'].append(f1_score(y_val, y_pred, zero_division=0))
                    cv_scores['precision'].append(precision_score(y_val, y_pred, zero_division=0))
                    cv_scores['recall'].append(recall_score(y_val, y_pred, zero_division=0))
                
                # Store results
                result_row = {
                    'strategy': strategy_name,
                    'model': model_name,
                    'f1_mean': np.mean(cv_scores['f1']),
                    'f1_std': np.std(cv_scores['f1']),
                    'precision_mean': np.mean(cv_scores['precision']),
                    'precision_std': np.std(cv_scores['precision']),
                    'recall_mean': np.mean(cv_scores['recall']),
                    'recall_std': np.std(cv_scores['recall']),
                    'roc_auc_mean': np.mean(cv_scores['roc_auc']),
                    'roc_auc_std': np.std(cv_scores['roc_auc']),
                    'optimal_threshold': optimal_threshold
                }
                
                results.append(result_row)
        
        return pd.DataFrame(results)
    
    def plot_evaluation_results(self, 
                              results_df: pd.DataFrame,
                              save_path: Optional[str] = None):
        """
        Plot evaluation results for different strategies.
        
        Args:
            results_df: Results DataFrame from evaluate_strategies
            save_path: Path to save the plot (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Imbalance Handling Strategy Comparison')
        
        metrics = ['f1_mean', 'precision_mean', 'recall_mean', 'roc_auc_mean']
        titles = ['F1 Score', 'Precision', 'Recall', 'ROC AUC']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i // 2, i % 2]
            
            # Create pivot table for heatmap
            pivot_data = results_df.pivot(index='strategy', columns='model', values=metric)
            
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
            ax.set_title(title)
            ax.set_xlabel('Model')
            ax.set_ylabel('Strategy')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_best_strategy(self, results_df: pd.DataFrame, metric: str = 'f1_mean') -> Dict[str, Any]:
        """
        Get the best strategy based on a specific metric.
        
        Args:
            results_df: Results DataFrame from evaluate_strategies
            metric: Metric to optimize
            
        Returns:
            Dictionary with best strategy information
        """
        best_idx = results_df[metric].idxmax()
        best_result = results_df.iloc[best_idx]
        
        return {
            'strategy': best_result['strategy'],
            'model': best_result['model'],
            'score': best_result[metric],
            'optimal_threshold': best_result['optimal_threshold'],
            'full_result': best_result.to_dict()
        }