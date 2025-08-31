#!/usr/bin/env python3
"""
Classical Machine Learning Models for Text Classification
Implements Logistic Regression, Naive Bayes, SVM, and XGBoost
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import csr_matrix
from typing import Dict, Any, List, Tuple, Optional
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

class ClassicalMLPipeline:
    """
    Pipeline for training and evaluating classical ML models
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_params = {}
        self.results = {}
        
    def get_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get model configurations with hyperparameter grids"""
        configs = {
            'logistic_regression': {
                'model': LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,
                    class_weight='balanced'
                ),
                'param_grid': {
                    'C': [0.1, 1.0, 10.0, 100.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            },
            'sgd_classifier': {
                'model': SGDClassifier(
                    random_state=self.random_state,
                    class_weight='balanced',
                    loss='log_loss'
                ),
                'param_grid': {
                    'alpha': [0.0001, 0.001, 0.01, 0.1],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'learning_rate': ['constant', 'optimal', 'adaptive']
                }
            },
            'multinomial_nb': {
                'model': MultinomialNB(),
                'param_grid': {
                    'alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
                    'fit_prior': [True, False]
                }
            },
            'bernoulli_nb': {
                'model': BernoulliNB(),
                'param_grid': {
                    'alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
                    'fit_prior': [True, False]
                }
            },
            'svm_linear': {
                'model': SVC(
                    kernel='linear',
                    random_state=self.random_state,
                    class_weight='balanced',
                    probability=True
                ),
                'param_grid': {
                    'C': [0.1, 1.0, 10.0, 100.0],
                    'gamma': ['scale', 'auto']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(
                    random_state=self.random_state,
                    class_weight='balanced',
                    n_jobs=-1
                ),
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            }
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            configs['xgboost'] = {
                'model': xgb.XGBClassifier(
                    random_state=self.random_state,
                    eval_metric='logloss',
                    use_label_encoder=False
                ),
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            }
        
        return configs
    
    def train_single_model(self, 
                          model_name: str,
                          X_train: csr_matrix,
                          y_train: np.ndarray,
                          X_val: csr_matrix,
                          y_val: np.ndarray,
                          use_grid_search: bool = True,
                          cv_folds: int = 3) -> Dict[str, Any]:
        """Train a single model with hyperparameter tuning"""
        
        print(f"Training {model_name}...")
        
        configs = self.get_model_configs()
        if model_name not in configs:
            raise ValueError(f"Model {model_name} not supported")
        
        config = configs[model_name]
        base_model = config['model']
        param_grid = config['param_grid']
        
        if use_grid_search and len(param_grid) > 0:
            # Use RandomizedSearchCV for efficiency with large parameter spaces
            if len(param_grid) > 3 or any(len(v) > 4 for v in param_grid.values()):
                search = RandomizedSearchCV(
                    base_model,
                    param_grid,
                    n_iter=20,
                    cv=cv_folds,
                    scoring='f1',
                    n_jobs=-1,
                    random_state=self.random_state
                )
            else:
                search = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=cv_folds,
                    scoring='f1',
                    n_jobs=-1
                )
            
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
            
        else:
            # Train with default parameters
            best_model = base_model
            best_model.fit(X_train, y_train)
            best_params = {}
        
        # Make predictions
        y_pred = best_model.predict(X_val)
        y_pred_proba = best_model.predict_proba(X_val)[:, 1] if hasattr(best_model, 'predict_proba') else None
        
        # Calculate metrics
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba) if y_pred_proba is not None else None
        
        # Store results
        result = {
            'model': best_model,
            'best_params': best_params,
            'f1_score': f1,
            'auc_score': auc,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }
        
        self.models[model_name] = best_model
        self.best_params[model_name] = best_params
        self.results[model_name] = result
        
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        print(f"{model_name} - F1: {f1:.4f}, AUC: {auc_str}")
        
        return result
    
    def train_all_models(self,
                        X_train: csr_matrix,
                        y_train: np.ndarray,
                        X_val: csr_matrix,
                        y_val: np.ndarray,
                        models_to_train: Optional[List[str]] = None,
                        use_grid_search: bool = True) -> Dict[str, Dict[str, Any]]:
        """Train all specified models"""
        
        if models_to_train is None:
            models_to_train = list(self.get_model_configs().keys())
        
        print(f"Training {len(models_to_train)} models...")
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        print("-" * 50)
        
        results = {}
        for model_name in models_to_train:
            try:
                result = self.train_single_model(
                    model_name, X_train, y_train, X_val, y_val, use_grid_search
                )
                results[model_name] = result
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                continue
        
        return results
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Get comparison of all trained models"""
        if not self.results:
            raise ValueError("No models have been trained yet")
        
        comparison_data = []
        for model_name, result in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'F1 Score': result['f1_score'],
                'AUC Score': result['auc_score'] if result['auc_score'] else 'N/A',
                'Best Params': str(result['best_params'])
            })
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values('F1 Score', ascending=False)
    
    def get_best_model(self, metric: str = 'f1_score') -> Tuple[str, Any]:
        """Get the best performing model"""
        if not self.results:
            raise ValueError("No models have been trained yet")
        
        best_score = -1
        best_model_name = None
        
        for model_name, result in self.results.items():
            score = result.get(metric)
            if score is not None and score > best_score:
                best_score = score
                best_model_name = model_name
        
        if best_model_name is None:
            raise ValueError(f"No models have {metric} available")
        
        return best_model_name, self.models[best_model_name]
    
    def predict(self, model_name: str, X: csr_matrix) -> np.ndarray:
        """Make predictions with a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        return self.models[model_name].predict(X)
    
    def predict_proba(self, model_name: str, X: csr_matrix) -> np.ndarray:
        """Get prediction probabilities with a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)[:, 1]
        else:
            raise ValueError(f"Model {model_name} doesn't support probability predictions")
    
    def save_models(self, directory: str) -> None:
        """Save all trained models"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for model_name, model in self.models.items():
            filepath = os.path.join(directory, f"{model_name}.joblib")
            joblib.dump(model, filepath)
        
        # Save results and parameters
        results_filepath = os.path.join(directory, "training_results.joblib")
        joblib.dump({
            'results': self.results,
            'best_params': self.best_params
        }, results_filepath)
        
        print(f"Models saved to {directory}")
    
    def load_models(self, directory: str) -> None:
        """Load previously trained models"""
        import os
        
        # Load results and parameters
        results_filepath = os.path.join(directory, "training_results.joblib")
        if os.path.exists(results_filepath):
            data = joblib.load(results_filepath)
            self.results = data['results']
            self.best_params = data['best_params']
        
        # Load models
        for filename in os.listdir(directory):
            if filename.endswith('.joblib') and filename != 'training_results.joblib':
                model_name = filename.replace('.joblib', '')
                filepath = os.path.join(directory, filename)
                self.models[model_name] = joblib.load(filepath)
        
        print(f"Loaded {len(self.models)} models from {directory}")

# Example usage
if __name__ == "__main__":
    # This would typically be used with real data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from scipy.sparse import csr_matrix
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=100, n_classes=2, random_state=42)
    X = csr_matrix(X)  # Convert to sparse matrix
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize pipeline
    pipeline = ClassicalMLPipeline()
    
    # Train a few models (not all for demo)
    models_to_train = ['logistic_regression', 'multinomial_nb', 'svm_linear']
    results = pipeline.train_all_models(X_train, y_train, X_val, y_val, models_to_train)
    
    # Get comparison
    comparison = pipeline.get_model_comparison()
    print("\nModel Comparison:")
    print(comparison)
    
    # Get best model
    best_name, best_model = pipeline.get_best_model()
    print(f"\nBest model: {best_name}")