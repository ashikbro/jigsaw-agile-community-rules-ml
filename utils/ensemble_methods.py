#!/usr/bin/env python3
"""
Ensemble Methods for Binary Text Classification
Implements blending, stacking, and voting techniques
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.sparse import csr_matrix, hstack
from typing import Dict, List, Tuple, Any, Optional, Union
import joblib
import warnings
warnings.filterwarnings('ignore')

class SimpleBlender(BaseEstimator, ClassifierMixin):
    """
    Simple weighted average blending of model predictions
    """
    
    def __init__(self, weights: Optional[List[float]] = None):
        self.weights = weights
        self.models = {}
        self.model_names = []
        
    def add_model(self, name: str, model: Any) -> None:
        """Add a trained model to the ensemble"""
        self.models[name] = model
        self.model_names.append(name)
        
    def fit(self, X, y):
        """Fit is not needed as models are already trained"""
        if not self.models:
            raise ValueError("No models added to the blender")
        
        # Set equal weights if not provided
        if self.weights is None:
            self.weights = [1.0 / len(self.models)] * len(self.models)
        elif len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities using weighted average"""
        if not self.models:
            raise ValueError("No models in ensemble")
        
        predictions = []
        for name in self.model_names:
            model = self.models[name]
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(X)[:, 1]
            else:
                # For models without predict_proba, use decision function or predictions
                if hasattr(model, 'decision_function'):
                    scores = model.decision_function(X)
                    # Convert to probabilities using sigmoid
                    pred_proba = 1 / (1 + np.exp(-scores))
                else:
                    pred_proba = model.predict(X).astype(float)
            
            predictions.append(pred_proba)
        
        # Weighted average
        blended_proba = np.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            blended_proba += self.weights[i] * pred
        
        # Return as 2D array for compatibility
        return np.column_stack([1 - blended_proba, blended_proba])
    
    def predict(self, X):
        """Make binary predictions"""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

class StackingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Stacking ensemble with cross-validation
    """
    
    def __init__(self, 
                 base_models: Dict[str, Any],
                 meta_model: Any = None,
                 cv_folds: int = 5,
                 use_original_features: bool = True,
                 random_state: int = 42):
        self.base_models = base_models
        self.meta_model = meta_model if meta_model else LogisticRegression(random_state=random_state)
        self.cv_folds = cv_folds
        self.use_original_features = use_original_features
        self.random_state = random_state
        self.fitted_base_models = {}
        self.fitted_meta_model = None
        
    def fit(self, X, y):
        """Fit the stacking ensemble"""
        print(f"Training stacking ensemble with {len(self.base_models)} base models...")
        
        # Step 1: Generate out-of-fold predictions for meta-features
        meta_features = []
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        for name, model in self.base_models.items():
            print(f"Generating meta-features for {name}...")
            
            # Get out-of-fold predictions
            oof_predictions = cross_val_predict(
                model, X, y, cv=cv, method='predict_proba'
            )[:, 1]  # Get probability of positive class
            
            meta_features.append(oof_predictions.reshape(-1, 1))
        
        # Combine meta-features
        meta_X = np.hstack(meta_features)
        
        # Add original features if requested
        if self.use_original_features:
            if hasattr(X, 'toarray'):  # Sparse matrix
                original_features = X.toarray()
            else:
                original_features = X
            meta_X = np.hstack([meta_X, original_features])
        
        # Step 2: Train meta-model on meta-features
        print("Training meta-model...")
        self.fitted_meta_model = self.meta_model.fit(meta_X, y)
        
        # Step 3: Retrain base models on full dataset
        print("Retraining base models on full dataset...")
        for name, model in self.base_models.items():
            self.fitted_base_models[name] = model.fit(X, y)
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities using stacking"""
        if not self.fitted_base_models or self.fitted_meta_model is None:
            raise ValueError("Ensemble not fitted yet")
        
        # Generate base model predictions
        meta_features = []
        for name, model in self.fitted_base_models.items():
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(X)[:, 1]
            else:
                if hasattr(model, 'decision_function'):
                    scores = model.decision_function(X)
                    pred_proba = 1 / (1 + np.exp(-scores))
                else:
                    pred_proba = model.predict(X).astype(float)
            
            meta_features.append(pred_proba.reshape(-1, 1))
        
        # Combine meta-features
        meta_X = np.hstack(meta_features)
        
        # Add original features if used during training
        if self.use_original_features:
            if hasattr(X, 'toarray'):  # Sparse matrix
                original_features = X.toarray()
            else:
                original_features = X
            meta_X = np.hstack([meta_X, original_features])
        
        # Get meta-model predictions
        return self.fitted_meta_model.predict_proba(meta_X)
    
    def predict(self, X):
        """Make binary predictions"""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

class EnsembleManager:
    """
    Manager class for creating and evaluating different ensemble methods
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.ensembles = {}
        self.base_models = {}
        self.evaluation_results = {}
        
    def add_base_model(self, name: str, model: Any) -> None:
        """Add a base model to the ensemble pool"""
        self.base_models[name] = model
        
    def create_voting_ensemble(self, 
                             model_names: List[str],
                             voting: str = 'soft',
                             weights: Optional[List[float]] = None) -> VotingClassifier:
        """Create a voting ensemble"""
        if not all(name in self.base_models for name in model_names):
            missing = [name for name in model_names if name not in self.base_models]
            raise ValueError(f"Models not found: {missing}")
        
        estimators = [(name, self.base_models[name]) for name in model_names]
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting,
            weights=weights
        )
        
        ensemble_name = f"voting_{voting}_{'_'.join(model_names)}"
        self.ensembles[ensemble_name] = ensemble
        
        return ensemble
    
    def create_blending_ensemble(self, 
                               model_names: List[str],
                               weights: Optional[List[float]] = None) -> SimpleBlender:
        """Create a blending ensemble"""
        if not all(name in self.base_models for name in model_names):
            missing = [name for name in model_names if name not in self.base_models]
            raise ValueError(f"Models not found: {missing}")
        
        blender = SimpleBlender(weights=weights)
        for name in model_names:
            blender.add_model(name, self.base_models[name])
        
        ensemble_name = f"blending_{'_'.join(model_names)}"
        self.ensembles[ensemble_name] = blender
        
        return blender
    
    def create_stacking_ensemble(self,
                               model_names: List[str],
                               meta_model: Any = None,
                               cv_folds: int = 5,
                               use_original_features: bool = True) -> StackingEnsemble:
        """Create a stacking ensemble"""
        if not all(name in self.base_models for name in model_names):
            missing = [name for name in model_names if name not in self.base_models]
            raise ValueError(f"Models not found: {missing}")
        
        selected_models = {name: self.base_models[name] for name in model_names}
        
        stacker = StackingEnsemble(
            base_models=selected_models,
            meta_model=meta_model,
            cv_folds=cv_folds,
            use_original_features=use_original_features,
            random_state=self.random_state
        )
        
        ensemble_name = f"stacking_{'_'.join(model_names)}"
        self.ensembles[ensemble_name] = stacker
        
        return stacker
    
    def optimize_blending_weights(self,
                                X_val: csr_matrix,
                                y_val: np.ndarray,
                                model_names: List[str],
                                metric: str = 'f1') -> Tuple[List[float], float]:
        """Optimize blending weights using validation data"""
        from scipy.optimize import minimize
        
        # Get base model predictions
        predictions = []
        for name in model_names:
            model = self.base_models[name]
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X_val)[:, 1]
            else:
                if hasattr(model, 'decision_function'):
                    scores = model.decision_function(X_val)
                    pred = 1 / (1 + np.exp(-scores))
                else:
                    pred = model.predict(X_val).astype(float)
            predictions.append(pred)
        
        predictions = np.array(predictions).T
        
        def objective(weights):
            # Normalize weights
            weights = weights / np.sum(weights)
            
            # Weighted average
            blended_proba = np.dot(predictions, weights)
            blended_pred = (blended_proba >= 0.5).astype(int)
            
            if metric == 'f1':
                return -f1_score(y_val, blended_pred)
            elif metric == 'auc':
                return -roc_auc_score(y_val, blended_proba)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
        
        # Initial weights (equal)
        initial_weights = np.ones(len(model_names)) / len(model_names)
        
        # Constraints: weights sum to 1 and are non-negative
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(len(model_names))]
        
        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        optimal_weights = result.x.tolist()
        best_score = -result.fun
        
        print(f"Optimal weights for {model_names}: {optimal_weights}")
        print(f"Best {metric} score: {best_score:.4f}")
        
        return optimal_weights, best_score
    
    def evaluate_ensemble(self,
                         ensemble_name: str,
                         X_val: csr_matrix,
                         y_val: np.ndarray,
                         X_test: Optional[csr_matrix] = None,
                         y_test: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Evaluate an ensemble on validation and test data"""
        if ensemble_name not in self.ensembles:
            raise ValueError(f"Ensemble {ensemble_name} not found")
        
        ensemble = self.ensembles[ensemble_name]
        
        # Validation predictions
        val_pred = ensemble.predict(X_val)
        val_proba = ensemble.predict_proba(X_val)[:, 1] if hasattr(ensemble, 'predict_proba') else None
        
        results = {
            'ensemble_name': ensemble_name,
            'val_f1': f1_score(y_val, val_pred),
            'val_auc': roc_auc_score(y_val, val_proba) if val_proba is not None else None
        }
        
        # Test predictions if provided
        if X_test is not None and y_test is not None:
            test_pred = ensemble.predict(X_test)
            test_proba = ensemble.predict_proba(X_test)[:, 1] if hasattr(ensemble, 'predict_proba') else None
            
            results.update({
                'test_f1': f1_score(y_test, test_pred),
                'test_auc': roc_auc_score(y_test, test_proba) if test_proba is not None else None
            })
        
        self.evaluation_results[ensemble_name] = results
        return results
    
    def compare_ensembles(self) -> pd.DataFrame:
        """Compare all evaluated ensembles"""
        if not self.evaluation_results:
            return pd.DataFrame()
        
        comparison_data = []
        for name, results in self.evaluation_results.items():
            comparison_data.append({
                'Ensemble': name,
                'Val F1': results['val_f1'],
                'Val AUC': results.get('val_auc', 'N/A'),
                'Test F1': results.get('test_f1', 'N/A'),
                'Test AUC': results.get('test_auc', 'N/A')
            })
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values('Val F1', ascending=False)
    
    def get_best_ensemble(self, metric: str = 'val_f1') -> Tuple[str, Any]:
        """Get the best performing ensemble"""
        if not self.evaluation_results:
            raise ValueError("No ensembles evaluated yet")
        
        best_score = -1
        best_name = None
        
        for name, results in self.evaluation_results.items():
            score = results.get(metric)
            if score is not None and score > best_score:
                best_score = score
                best_name = name
        
        if best_name is None:
            raise ValueError(f"No ensembles have {metric} available")
        
        return best_name, self.ensembles[best_name]
    
    def save_ensembles(self, directory: str) -> None:
        """Save all ensembles"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for name, ensemble in self.ensembles.items():
            filepath = os.path.join(directory, f"{name}.joblib")
            joblib.dump(ensemble, filepath)
        
        # Save evaluation results
        results_filepath = os.path.join(directory, "ensemble_results.joblib")
        joblib.dump(self.evaluation_results, results_filepath)
        
        print(f"Ensembles saved to {directory}")
    
    def load_ensembles(self, directory: str) -> None:
        """Load previously saved ensembles"""
        import os
        
        # Load evaluation results
        results_filepath = os.path.join(directory, "ensemble_results.joblib")
        if os.path.exists(results_filepath):
            self.evaluation_results = joblib.load(results_filepath)
        
        # Load ensembles
        for filename in os.listdir(directory):
            if filename.endswith('.joblib') and filename != 'ensemble_results.joblib':
                name = filename.replace('.joblib', '')
                filepath = os.path.join(directory, filename)
                self.ensembles[name] = joblib.load(filepath)
        
        print(f"Loaded {len(self.ensembles)} ensembles from {directory}")

# Example usage
if __name__ == "__main__":
    # This would typically be used with real trained models
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import SVC
    from scipy.sparse import csr_matrix
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=100, n_classes=2, random_state=42)
    X = csr_matrix(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    
    # Train base models
    lr = LogisticRegression(random_state=42)
    nb = MultinomialNB()
    svm = SVC(probability=True, random_state=42)
    
    lr.fit(X_train, y_train)
    nb.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    
    # Create ensemble manager
    manager = EnsembleManager()
    manager.add_base_model('logistic', lr)
    manager.add_base_model('naive_bayes', nb)
    manager.add_base_model('svm', svm)
    
    # Create different ensembles
    voting_ensemble = manager.create_voting_ensemble(['logistic', 'naive_bayes', 'svm'])
    voting_ensemble.fit(X_train, y_train)
    
    # Optimize blending weights
    optimal_weights, _ = manager.optimize_blending_weights(
        X_val, y_val, ['logistic', 'naive_bayes', 'svm']
    )
    
    blending_ensemble = manager.create_blending_ensemble(
        ['logistic', 'naive_bayes', 'svm'], weights=optimal_weights
    )
    blending_ensemble.fit(X_train, y_train)
    
    stacking_ensemble = manager.create_stacking_ensemble(['logistic', 'naive_bayes', 'svm'])
    stacking_ensemble.fit(X_train, y_train)
    
    # Evaluate ensembles
    for name in manager.ensembles.keys():
        manager.evaluate_ensemble(name, X_val, y_val, X_test, y_test)
    
    # Compare results
    comparison = manager.compare_ensembles()
    print("\nEnsemble Comparison:")
    print(comparison)