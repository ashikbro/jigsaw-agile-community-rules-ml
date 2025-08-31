import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

class DiverseFeatureEnsemble(BaseEstimator, ClassifierMixin):
    """
    Advanced ensemble that uses different feature extractors with different models
    to create diverse predictions that are then combined.
    """
    
    def __init__(self, 
                 feature_model_pairs: List[Tuple[Any, Any]],
                 meta_classifier=None,
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        Initialize the diverse feature ensemble.
        
        Args:
            feature_model_pairs: List of (feature_extractor, model) pairs
            meta_classifier: Meta-classifier for stacking (default: LogisticRegression)
            cv_folds: Number of CV folds for meta-features
            random_state: Random state for reproducibility
        """
        self.feature_model_pairs = feature_model_pairs
        self.meta_classifier = meta_classifier or LogisticRegression(random_state=random_state)
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Fitted components
        self.fitted_pairs = []
        self.is_fitted = False
    
    def fit(self, X: List[str], y: np.ndarray):
        """
        Fit the diverse feature ensemble.
        
        Args:
            X: List of text documents
            y: Target labels
            
        Returns:
            self
        """
        # Fit each feature extractor and model pair
        self.fitted_pairs = []
        meta_features = []
        
        for feature_extractor, model in self.feature_model_pairs:
            # Clone to avoid modifying original objects
            feat_ext = clone(feature_extractor)
            clf = clone(model)
            
            # Fit feature extractor and transform data
            X_features = feat_ext.fit_transform(X)
            
            # Generate meta-features using cross-validation
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            meta_pred = cross_val_predict(clf, X_features, y, cv=cv, method='predict_proba')
            
            # Store meta-features (probabilities)
            if meta_pred.shape[1] == 2:  # Binary classification
                meta_features.append(meta_pred[:, 1])  # Positive class probability
            else:
                meta_features.append(meta_pred)
            
            # Fit the model on full data
            clf.fit(X_features, y)
            
            # Store fitted pair
            self.fitted_pairs.append((feat_ext, clf))
        
        # Combine meta-features and fit meta-classifier
        X_meta = np.column_stack(meta_features)
        self.meta_classifier.fit(X_meta, y)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: List[str]) -> np.ndarray:
        """
        Make predictions using the ensemble.
        
        Args:
            X: List of text documents
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Generate predictions from each pair
        meta_features = []
        
        for feat_ext, clf in self.fitted_pairs:
            X_features = feat_ext.transform(X)
            pred_proba = clf.predict_proba(X_features)
            
            if pred_proba.shape[1] == 2:  # Binary classification
                meta_features.append(pred_proba[:, 1])
            else:
                meta_features.append(pred_proba)
        
        # Combine meta-features and predict
        X_meta = np.column_stack(meta_features)
        return self.meta_classifier.predict(X_meta)
    
    def predict_proba(self, X: List[str]) -> np.ndarray:
        """
        Predict class probabilities using the ensemble.
        
        Args:
            X: List of text documents
            
        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Generate predictions from each pair
        meta_features = []
        
        for feat_ext, clf in self.fitted_pairs:
            X_features = feat_ext.transform(X)
            pred_proba = clf.predict_proba(X_features)
            
            if pred_proba.shape[1] == 2:  # Binary classification
                meta_features.append(pred_proba[:, 1])
            else:
                meta_features.append(pred_proba)
        
        # Combine meta-features and predict probabilities
        X_meta = np.column_stack(meta_features)
        return self.meta_classifier.predict_proba(X_meta)

class AdaptiveWeightedEnsemble(BaseEstimator, ClassifierMixin):
    """
    Ensemble that adaptively weights different models based on their
    performance on different types of violations.
    """
    
    def __init__(self, 
                 models: List[Tuple[str, Any]],
                 feature_extractor: Any,
                 adaptation_method: str = 'performance_based',
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        Initialize the adaptive weighted ensemble.
        
        Args:
            models: List of (name, model) pairs
            feature_extractor: Feature extractor to use
            adaptation_method: Method for weight adaptation
            cv_folds: Number of CV folds for weight calculation
            random_state: Random state for reproducibility
        """
        self.models = models
        self.feature_extractor = feature_extractor
        self.adaptation_method = adaptation_method
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Fitted components
        self.fitted_models = []
        self.model_weights = {}
        self.is_fitted = False
    
    def fit(self, X: List[str], y: np.ndarray):
        """
        Fit the adaptive weighted ensemble.
        
        Args:
            X: List of text documents
            y: Target labels
            
        Returns:
            self
        """
        # Fit feature extractor
        self.feature_extractor = clone(self.feature_extractor)
        X_features = self.feature_extractor.fit_transform(X)
        
        # Fit models and calculate weights
        self.fitted_models = []
        model_scores = {}
        
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        for name, model in self.models:
            # Clone model
            clf = clone(model)
            
            # Calculate cross-validation scores
            cv_predictions = cross_val_predict(clf, X_features, y, cv=cv)
            cv_probabilities = cross_val_predict(clf, X_features, y, cv=cv, method='predict_proba')
            
            # Calculate performance metrics
            f1 = f1_score(y, cv_predictions)
            accuracy = accuracy_score(y, cv_predictions)
            
            if cv_probabilities.shape[1] == 2:
                auc = roc_auc_score(y, cv_probabilities[:, 1])
            else:
                auc = 0.5  # Default for multiclass
            
            # Store scores
            model_scores[name] = {
                'f1': f1,
                'accuracy': accuracy,
                'auc': auc,
                'combined': (f1 + accuracy + auc) / 3
            }
            
            # Fit model on full data
            clf.fit(X_features, y)
            self.fitted_models.append((name, clf))
        
        # Calculate adaptive weights
        self._calculate_weights(model_scores)
        
        self.is_fitted = True
        return self
    
    def _calculate_weights(self, model_scores: Dict[str, Dict[str, float]]):
        """
        Calculate adaptive weights for models based on their performance.
        
        Args:
            model_scores: Dictionary of model performance scores
        """
        if self.adaptation_method == 'performance_based':
            # Weight based on combined performance
            total_score = sum(scores['combined'] for scores in model_scores.values())
            
            for name, scores in model_scores.items():
                self.model_weights[name] = scores['combined'] / total_score
        
        elif self.adaptation_method == 'inverse_error':
            # Weight inversely proportional to error
            errors = {name: 1 - scores['accuracy'] for name, scores in model_scores.items()}
            total_inv_error = sum(1 / (error + 1e-10) for error in errors.values())
            
            for name, error in errors.items():
                self.model_weights[name] = (1 / (error + 1e-10)) / total_inv_error
        
        else:  # Equal weights
            n_models = len(model_scores)
            for name in model_scores.keys():
                self.model_weights[name] = 1.0 / n_models
    
    def predict(self, X: List[str]) -> np.ndarray:
        """
        Make predictions using the adaptive weighted ensemble.
        
        Args:
            X: List of text documents
            
        Returns:
            Predicted labels
        """
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] > 0.5).astype(int)
    
    def predict_proba(self, X: List[str]) -> np.ndarray:
        """
        Predict class probabilities using the adaptive weighted ensemble.
        
        Args:
            X: List of text documents
            
        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Transform features
        X_features = self.feature_extractor.transform(X)
        
        # Get weighted predictions
        weighted_probabilities = None
        
        for name, clf in self.fitted_models:
            pred_proba = clf.predict_proba(X_features)
            weight = self.model_weights[name]
            
            if weighted_probabilities is None:
                weighted_probabilities = weight * pred_proba
            else:
                weighted_probabilities += weight * pred_proba
        
        return weighted_probabilities

class HierarchicalEnsemble(BaseEstimator, ClassifierMixin):
    """
    Hierarchical ensemble that first classifies into broad categories,
    then uses specialized models for each category.
    """
    
    def __init__(self, 
                 primary_classifier: Any,
                 specialized_classifiers: Dict[str, Any],
                 feature_extractor: Any,
                 category_threshold: float = 0.7,
                 random_state: int = 42):
        """
        Initialize the hierarchical ensemble.
        
        Args:
            primary_classifier: First-level classifier for broad categorization
            specialized_classifiers: Dict of specialized classifiers for each category
            feature_extractor: Feature extractor to use
            category_threshold: Threshold for category assignment
            random_state: Random state for reproducibility
        """
        self.primary_classifier = primary_classifier
        self.specialized_classifiers = specialized_classifiers
        self.feature_extractor = feature_extractor
        self.category_threshold = category_threshold
        self.random_state = random_state
        
        # Fitted components
        self.fitted_primary = None
        self.fitted_specialized = {}
        self.is_fitted = False
    
    def fit(self, X: List[str], y: np.ndarray, categories: Optional[np.ndarray] = None):
        """
        Fit the hierarchical ensemble.
        
        Args:
            X: List of text documents
            y: Target labels
            categories: Category labels (if None, will be inferred)
            
        Returns:
            self
        """
        # Fit feature extractor
        self.feature_extractor = clone(self.feature_extractor)
        X_features = self.feature_extractor.fit_transform(X)
        
        # If categories not provided, create simple binary categories
        if categories is None:
            categories = y.copy()  # Use target as categories for simplicity
        
        # Fit primary classifier
        self.fitted_primary = clone(self.primary_classifier)
        self.fitted_primary.fit(X_features, categories)
        
        # Fit specialized classifiers for each category
        unique_categories = np.unique(categories)
        
        for category in unique_categories:
            if str(category) in self.specialized_classifiers:
                # Get samples for this category
                category_mask = categories == category
                X_cat = X_features[category_mask]
                y_cat = y[category_mask]
                
                if len(np.unique(y_cat)) > 1:  # Only fit if there are multiple classes
                    clf = clone(self.specialized_classifiers[str(category)])
                    clf.fit(X_cat, y_cat)
                    self.fitted_specialized[str(category)] = clf
        
        self.is_fitted = True
        return self
    
    def predict(self, X: List[str]) -> np.ndarray:
        """
        Make predictions using the hierarchical ensemble.
        
        Args:
            X: List of text documents
            
        Returns:
            Predicted labels
        """
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] > 0.5).astype(int)
    
    def predict_proba(self, X: List[str]) -> np.ndarray:
        """
        Predict class probabilities using the hierarchical ensemble.
        
        Args:
            X: List of text documents
            
        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Transform features
        X_features = self.feature_extractor.transform(X)
        
        # Get primary predictions
        primary_proba = self.fitted_primary.predict_proba(X_features)
        primary_pred = self.fitted_primary.predict(X_features)
        
        # Initialize final probabilities
        final_probabilities = np.zeros((len(X), 2))
        
        # For each sample, use appropriate specialized classifier
        for i, (pred, proba) in enumerate(zip(primary_pred, primary_proba)):
            category = str(pred)
            
            # Check if we have a specialized classifier and confidence is high enough
            if (category in self.fitted_specialized and 
                np.max(proba) > self.category_threshold):
                
                # Use specialized classifier
                specialized_proba = self.fitted_specialized[category].predict_proba(
                    X_features[i:i+1]
                )[0]
                final_probabilities[i] = specialized_proba
            else:
                # Use primary classifier probabilities
                # Convert category probabilities to binary probabilities
                if len(proba) == 2:
                    final_probabilities[i] = proba
                else:
                    # For multiclass, use max probability as positive class
                    max_prob = np.max(proba)
                    final_probabilities[i] = [1 - max_prob, max_prob]
        
        return final_probabilities

class NovelEnsembleManager:
    """
    Manager class for creating and evaluating novel ensemble methods.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the novel ensemble manager.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.ensembles = {}
        self.evaluation_results = {}
    
    def create_diverse_feature_ensemble(self, 
                                      feature_model_pairs: List[Tuple[Any, Any]],
                                      name: str = 'diverse_features'):
        """
        Create a diverse feature ensemble.
        
        Args:
            feature_model_pairs: List of (feature_extractor, model) pairs
            name: Name for the ensemble
        """
        ensemble = DiverseFeatureEnsemble(
            feature_model_pairs=feature_model_pairs,
            random_state=self.random_state
        )
        self.ensembles[name] = ensemble
        return ensemble
    
    def create_adaptive_weighted_ensemble(self, 
                                        models: List[Tuple[str, Any]],
                                        feature_extractor: Any,
                                        name: str = 'adaptive_weighted'):
        """
        Create an adaptive weighted ensemble.
        
        Args:
            models: List of (name, model) pairs
            feature_extractor: Feature extractor to use
            name: Name for the ensemble
        """
        ensemble = AdaptiveWeightedEnsemble(
            models=models,
            feature_extractor=feature_extractor,
            random_state=self.random_state
        )
        self.ensembles[name] = ensemble
        return ensemble
    
    def create_hierarchical_ensemble(self, 
                                   primary_classifier: Any,
                                   specialized_classifiers: Dict[str, Any],
                                   feature_extractor: Any,
                                   name: str = 'hierarchical'):
        """
        Create a hierarchical ensemble.
        
        Args:
            primary_classifier: First-level classifier
            specialized_classifiers: Dict of specialized classifiers
            feature_extractor: Feature extractor to use
            name: Name for the ensemble
        """
        ensemble = HierarchicalEnsemble(
            primary_classifier=primary_classifier,
            specialized_classifiers=specialized_classifiers,
            feature_extractor=feature_extractor,
            random_state=self.random_state
        )
        self.ensembles[name] = ensemble
        return ensemble
    
    def evaluate_ensemble(self, 
                         ensemble_name: str,
                         X_train: List[str],
                         y_train: np.ndarray,
                         X_test: List[str],
                         y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate an ensemble and store results.
        
        Args:
            ensemble_name: Name of the ensemble to evaluate
            X_train: Training texts
            y_train: Training labels
            X_test: Test texts
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if ensemble_name not in self.ensembles:
            raise ValueError(f"Ensemble '{ensemble_name}' not found")
        
        ensemble = self.ensembles[ensemble_name]
        
        # Fit and predict
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)
        y_proba = ensemble.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        self.evaluation_results[ensemble_name] = results
        return results
    
    def get_best_ensemble(self, metric: str = 'f1_score') -> Tuple[str, Dict[str, float]]:
        """
        Get the best performing ensemble based on a metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (ensemble_name, results)
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available")
        
        best_name = max(self.evaluation_results.keys(), 
                       key=lambda x: self.evaluation_results[x][metric])
        
        return best_name, self.evaluation_results[best_name]
    
    def get_ensemble_comparison(self) -> pd.DataFrame:
        """
        Get a comparison of all evaluated ensembles.
        
        Returns:
            DataFrame with ensemble comparison
        """
        if not self.evaluation_results:
            return pd.DataFrame()
        
        return pd.DataFrame(self.evaluation_results).T