import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
try:
    from .advanced_feature_extractor import AdvancedHybridFeatureExtractor
    from .domain_specific_features import CommunityModerationFeatureExtractor
    from .topic_modeling_features import TopicModelingFeatureExtractor
    from .advanced_ensemble_methods import NovelEnsembleManager
    from .explainability_module import ModelExplainer
    from .advanced_imbalance_handler import AdvancedImbalanceHandler
    from .meta_learning_approaches import MetaLearningPipeline
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append('.')
    from advanced_feature_extractor import AdvancedHybridFeatureExtractor
    from domain_specific_features import CommunityModerationFeatureExtractor
    from topic_modeling_features import TopicModelingFeatureExtractor
    from advanced_ensemble_methods import NovelEnsembleManager
    from explainability_module import ModelExplainer
    from advanced_imbalance_handler import AdvancedImbalanceHandler
    from meta_learning_approaches import MetaLearningPipeline

class ResearchMLPipeline:
    """
    Comprehensive research-style ML pipeline that combines all novel approaches
    for community rule violation detection.
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 random_state: Optional[int] = 42):
        """
        Initialize the research ML pipeline.
        
        Args:
            config: Configuration dictionary for pipeline components
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.feature_extractors = {}
        self.ensemble_manager = None
        self.explainer = None
        self.imbalance_handler = None
        self.meta_learner = None
        
        # Results storage
        self.experiment_results = {}
        self.best_model = None
        self.best_score = 0
        self.feature_importance_analysis = {}
        
        self._initialize_components()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for the pipeline.
        
        Returns:
            Default configuration dictionary
        """
        return {
            'feature_extraction': {
                'use_hybrid_features': True,
                'use_domain_features': True,
                'use_topic_modeling': True,
                'tfidf_params': {
                    'max_features': 10000,
                    'ngram_range': (1, 2),
                    'min_df': 2,
                    'max_df': 0.95
                },
                'topic_modeling_params': {
                    'n_topics_lda': 10,
                    'n_topics_nmf': 8
                }
            },
            'ensemble_methods': {
                'use_diverse_ensemble': True,
                'use_adaptive_ensemble': True,
                'use_hierarchical_ensemble': True,
                'voting_weights': 'uniform'
            },
            'imbalance_handling': {
                'strategy': 'combined',
                'sampling_method': 'smote',
                'cost_sensitive': True,
                'threshold_optimization': True
            },
            'meta_learning': {
                'adversarial_validation': True,
                'pseudo_labeling': True,
                'unsupervised_features': True
            },
            'explainability': {
                'methods': ['rule_based', 'lime', 'shap'],
                'generate_reports': True
            },
            'evaluation': {
                'cv_folds': 5,
                'metrics': ['f1', 'precision', 'recall', 'roc_auc', 'accuracy'],
                'detailed_analysis': True
            }
        }
    
    def _initialize_components(self):
        """
        Initialize all pipeline components based on configuration.
        """
        # Feature extractors
        if self.config['feature_extraction']['use_hybrid_features']:
            tfidf_params = self.config['feature_extraction']['tfidf_params']
            # Map generic parameters to specific ones for AdvancedHybridFeatureExtractor
            hybrid_params = {
                'word_max_features': tfidf_params.get('max_features', 5000),
                'char_max_features': tfidf_params.get('max_features', 3000) // 2,
                'word_ngram_range': tfidf_params.get('ngram_range', (1, 2)),
                'char_ngram_range': (2, 4),
                'min_df': tfidf_params.get('min_df', 2),
                'max_df': tfidf_params.get('max_df', 0.95)
            }
            self.feature_extractors['hybrid'] = AdvancedHybridFeatureExtractor(**hybrid_params)
        
        if self.config['feature_extraction']['use_domain_features']:
            self.feature_extractors['domain'] = CommunityModerationFeatureExtractor()
        
        if self.config['feature_extraction']['use_topic_modeling']:
            self.feature_extractors['topic'] = TopicModelingFeatureExtractor(
                **self.config['feature_extraction']['topic_modeling_params']
            )
        
        # Ensemble manager
        if any(self.config['ensemble_methods'].values()):
            self.ensemble_manager = NovelEnsembleManager(
                random_state=self.random_state
            )
        
        # Imbalance handler
        self.imbalance_handler = AdvancedImbalanceHandler(
            **self.config['imbalance_handling'],
            random_state=self.random_state
        )
        
        # Meta-learning pipeline
        meta_config = self.config['meta_learning']
        meta_params = {
            'enable_adversarial_validation': meta_config.get('adversarial_validation', True),
            'enable_pseudo_labeling': meta_config.get('pseudo_labeling', True),
            'enable_unsupervised_features': meta_config.get('unsupervised_features', True),
            'random_state': self.random_state
        }
        self.meta_learner = MetaLearningPipeline(**meta_params)
    
    def extract_features(self, 
                        texts: List[str],
                        fit: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract features using all configured feature extractors.
        
        Args:
            texts: List of text samples
            fit: Whether to fit the extractors (True for training, False for inference)
            
        Returns:
            Combined feature matrix and feature information
        """
        print("Extracting features...")
        
        feature_matrices = []
        feature_info = {
            'feature_names': [],
            'feature_ranges': {},
            'extractor_info': {}
        }
        
        for name, extractor in self.feature_extractors.items():
            print(f"Applying {name} feature extraction...")
            
            try:
                if fit:
                    features = extractor.fit_transform(texts)
                else:
                    features = extractor.transform(texts)
                
                feature_matrices.append(features)
                
                # Store feature information
                start_idx = sum(f.shape[1] for f in feature_matrices[:-1])
                end_idx = start_idx + features.shape[1]
                
                feature_info['feature_ranges'][name] = (start_idx, end_idx)
                feature_info['extractor_info'][name] = {
                    'n_features': features.shape[1],
                    'feature_type': type(extractor).__name__
                }
                
                # Get feature names if available
                if hasattr(extractor, 'get_feature_names_out'):
                    names = extractor.get_feature_names_out()
                    feature_info['feature_names'].extend([f"{name}_{fn}" for fn in names])
                elif hasattr(extractor, 'feature_names_'):
                    names = extractor.feature_names_
                    feature_info['feature_names'].extend([f"{name}_{fn}" for fn in names])
                else:
                    feature_info['feature_names'].extend(
                        [f"{name}_feature_{i}" for i in range(features.shape[1])]
                    )
                
                print(f"{name} features: {features.shape[1]} dimensions")
                
            except Exception as e:
                print(f"Error in {name} feature extraction: {e}")
                continue
        
        if not feature_matrices:
            raise ValueError("No features were successfully extracted")
        
        # Combine all features
        if len(feature_matrices) == 1:
            combined_features = feature_matrices[0]
        else:
            # Handle sparse matrices
            from scipy import sparse
            if any(sparse.issparse(f) for f in feature_matrices):
                # Convert all to sparse
                sparse_matrices = []
                for f in feature_matrices:
                    if sparse.issparse(f):
                        sparse_matrices.append(f)
                    else:
                        sparse_matrices.append(sparse.csr_matrix(f))
                combined_features = sparse.hstack(sparse_matrices)
            else:
                combined_features = np.hstack(feature_matrices)
        
        print(f"Total combined features: {combined_features.shape[1]}")
        
        return combined_features, feature_info
    
    def run_comprehensive_experiment(self, 
                                   X_train: np.ndarray,
                                   y_train: np.ndarray,
                                   X_test: Optional[np.ndarray] = None,
                                   y_test: Optional[np.ndarray] = None,
                                   texts_train: Optional[List[str]] = None,
                                   texts_test: Optional[List[str]] = None,
                                   experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a comprehensive experiment with all novel approaches.
        
        Args:
            X_train: Training features (can be raw text features)
            y_train: Training labels
            X_test: Test features (optional)
            y_test: Test labels (optional)
            texts_train: Training texts for feature extraction (optional)
            texts_test: Test texts for feature extraction (optional)
            experiment_name: Name for this experiment
            
        Returns:
            Comprehensive experiment results
        """
        if experiment_name is None:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n=== Starting Comprehensive Experiment: {experiment_name} ===")
        
        results = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'feature_extraction': {},
            'imbalance_handling': {},
            'meta_learning': {},
            'model_evaluation': {},
            'ensemble_evaluation': {},
            'explainability': {},
            'best_model_info': {},
            'recommendations': []
        }
        
        # Step 1: Advanced Feature Extraction
        print("\n--- Step 1: Advanced Feature Extraction ---")
        
        if texts_train is not None:
            # Extract features from text
            X_enhanced, feature_info = self.extract_features(texts_train, fit=True)
            results['feature_extraction'] = feature_info
            
            if texts_test is not None:
                X_test_enhanced, _ = self.extract_features(texts_test, fit=False)
            else:
                X_test_enhanced = None
        else:
            # Use provided features
            X_enhanced = X_train
            X_test_enhanced = X_test
            results['feature_extraction'] = {'original_features_used': True}
        
        # Step 2: Meta-Learning Approaches
        print("\n--- Step 2: Meta-Learning Approaches ---")
        
        meta_results = self.meta_learner.run_pipeline(
            X_enhanced, y_train, 
            X_test=X_test_enhanced,
            X_unlabeled=None  # Could add unlabeled data here
        )
        results['meta_learning'] = meta_results
        
        # Use enhanced features from meta-learning if available
        if meta_results.get('enhanced_features') is not None:
            X_final = meta_results['enhanced_features']
            print(f"Using meta-learning enhanced features: {X_final.shape}")
        else:
            X_final = X_enhanced
        
        # Step 3: Advanced Imbalance Handling
        print("\n--- Step 3: Advanced Imbalance Handling ---")
        
        # Create base models for imbalance evaluation
        base_models = [
            LogisticRegression(random_state=self.random_state, max_iter=1000),
            RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            SVC(probability=True, random_state=self.random_state)
        ]
        model_names = ['LogisticRegression', 'RandomForest', 'SVM']
        
        # Evaluate imbalance handling strategies
        imbalance_results = self.imbalance_handler.evaluate_strategies(
            X_final, y_train, base_models, model_names
        )
        results['imbalance_handling'] = {
            'strategy_comparison': imbalance_results.to_dict(),
            'best_strategy': self.imbalance_handler.get_best_strategy(imbalance_results)
        }
        
        # Apply best imbalance handling strategy
        best_strategy = results['imbalance_handling']['best_strategy']
        processed_data = self.imbalance_handler.handle_imbalance(X_final, y_train)
        
        X_processed = processed_data['X_processed']
        y_processed = processed_data['y_processed']
        
        print(f"Applied imbalance handling: {best_strategy['strategy']}")
        print(f"Processed data shape: {X_processed.shape}")
        
        # Step 4: Model Evaluation with Cross-Validation
        print("\n--- Step 4: Model Evaluation ---")
        
        models_to_evaluate = {
            'LogisticRegression': LogisticRegression(
                random_state=self.random_state, 
                max_iter=1000,
                class_weight=processed_data.get('class_weights')
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_state,
                class_weight=processed_data.get('class_weights')
            ),
            'SVM': SVC(
                probability=True, 
                random_state=self.random_state,
                class_weight=processed_data.get('class_weights')
            ),
            'MultinomialNB': MultinomialNB()
        }
        
        model_results = self._evaluate_models(
            models_to_evaluate, X_processed, y_processed
        )
        results['model_evaluation'] = model_results
        
        # Step 5: Advanced Ensemble Methods
        print("\n--- Step 5: Advanced Ensemble Methods ---")
        
        if self.ensemble_manager:
            ensemble_results = self._evaluate_ensembles(
                X_processed, y_processed, processed_data.get('class_weights')
            )
            results['ensemble_evaluation'] = ensemble_results
        
        # Step 6: Model Explainability
        print("\n--- Step 6: Model Explainability ---")
        
        # Select best model for explainability
        best_model_name, best_model_info = self._select_best_model(
            model_results, results.get('ensemble_evaluation', {})
        )
        
        results['best_model_info'] = {
            'model_name': best_model_name,
            'model_info': best_model_info,
            'optimal_threshold': processed_data.get('optimal_threshold', 0.5)
        }
        
        # Initialize explainer with best model
        if texts_train is not None and 'feature_names' in results['feature_extraction']:
            feature_names = results['feature_extraction']['feature_names']
        else:
            feature_names = None
        
        explainability_results = self._analyze_explainability(
            best_model_info['model'], X_processed, y_processed,
            texts_train, feature_names
        )
        results['explainability'] = explainability_results
        
        # Step 7: Final Evaluation on Test Set
        if X_test_enhanced is not None and y_test is not None:
            print("\n--- Step 7: Final Test Evaluation ---")
            
            # Fit the best model on full training data
            best_model = best_model_info['model']
            print(f"Fitting best model ({best_model_name}) on full training data...")
            best_model.fit(X_processed, y_processed)
            
            test_results = self._evaluate_on_test_set(
                best_model, X_test_enhanced, y_test,
                processed_data.get('optimal_threshold', 0.5)
            )
            results['test_evaluation'] = test_results
        
        # Step 8: Generate Recommendations
        print("\n--- Step 8: Generating Recommendations ---")
        
        recommendations = self._generate_recommendations(results)
        results['recommendations'] = recommendations
        
        # Store results
        self.experiment_results[experiment_name] = results
        
        print(f"\n=== Experiment {experiment_name} Completed ===")
        
        return results
    
    def _evaluate_models(self, 
                        models: Dict[str, BaseEstimator],
                        X: np.ndarray,
                        y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate multiple models using cross-validation.
        
        Args:
            models: Dictionary of models to evaluate
            X: Feature matrix
            y: Target labels
            
        Returns:
            Model evaluation results
        """
        cv = StratifiedKFold(
            n_splits=self.config['evaluation']['cv_folds'],
            shuffle=True,
            random_state=self.random_state
        )
        
        scoring = self.config['evaluation']['metrics']
        results = {}
        
        for name, model in models.items():
            print(f"Evaluating {name}...")
            
            try:
                cv_results = cross_validate(
                    model, X, y, cv=cv, scoring=scoring,
                    return_train_score=True, n_jobs=-1
                )
                
                model_result = {
                    'model': model,
                    'cv_scores': {}
                }
                
                for metric in scoring:
                    test_scores = cv_results[f'test_{metric}']
                    train_scores = cv_results[f'train_{metric}']
                    
                    model_result['cv_scores'][metric] = {
                        'test_mean': np.mean(test_scores),
                        'test_std': np.std(test_scores),
                        'train_mean': np.mean(train_scores),
                        'train_std': np.std(train_scores),
                        'overfitting': np.mean(train_scores) - np.mean(test_scores)
                    }
                
                results[name] = model_result
                
                # Print summary
                f1_score = model_result['cv_scores']['f1']['test_mean']
                print(f"{name} - F1: {f1_score:.4f} (±{model_result['cv_scores']['f1']['test_std']:.4f})")
                
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    def _evaluate_ensembles(self, 
                           X: np.ndarray,
                           y: np.ndarray,
                           class_weights: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Evaluate advanced ensemble methods.
        
        Args:
            X: Feature matrix
            y: Target labels
            class_weights: Class weights for models
            
        Returns:
            Ensemble evaluation results
        """
        ensemble_results = {}
        
        # Create base models
        base_models = [
            ('lr', LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight=class_weights
            )),
            ('rf', RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight=class_weights
            )),
            ('svm', SVC(
                probability=True,
                random_state=self.random_state,
                class_weight=class_weights
            ))
        ]
        
        # Evaluate different ensemble methods
        ensemble_configs = {
            'diverse_features': self.config['ensemble_methods']['use_diverse_ensemble'],
            'adaptive_weights': self.config['ensemble_methods']['use_adaptive_ensemble'],
            'hierarchical': self.config['ensemble_methods']['use_hierarchical_ensemble']
        }
        
        for ensemble_type, enabled in ensemble_configs.items():
            if not enabled:
                continue
            
            print(f"Evaluating {ensemble_type} ensemble...")
            
            try:
                if ensemble_type == 'diverse_features':
                    ensemble = self.ensemble_manager.create_diverse_feature_ensemble(
                        base_models, X, y
                    )
                elif ensemble_type == 'adaptive_weights':
                    ensemble = self.ensemble_manager.create_adaptive_weighted_ensemble(
                        base_models
                    )
                elif ensemble_type == 'hierarchical':
                    ensemble = self.ensemble_manager.create_hierarchical_ensemble(
                        base_models
                    )
                
                # Cross-validation evaluation
                cv = StratifiedKFold(
                    n_splits=self.config['evaluation']['cv_folds'],
                    shuffle=True,
                    random_state=self.random_state
                )
                
                cv_results = cross_validate(
                    ensemble, X, y, cv=cv,
                    scoring=self.config['evaluation']['metrics'],
                    n_jobs=-1
                )
                
                ensemble_result = {
                    'ensemble': ensemble,
                    'cv_scores': {}
                }
                
                for metric in self.config['evaluation']['metrics']:
                    scores = cv_results[f'test_{metric}']
                    ensemble_result['cv_scores'][metric] = {
                        'mean': np.mean(scores),
                        'std': np.std(scores)
                    }
                
                ensemble_results[ensemble_type] = ensemble_result
                
                # Print summary
                f1_score = ensemble_result['cv_scores']['f1']['mean']
                print(f"{ensemble_type} ensemble - F1: {f1_score:.4f} (±{ensemble_result['cv_scores']['f1']['std']:.4f})")
                
            except Exception as e:
                print(f"Error evaluating {ensemble_type} ensemble: {e}")
                ensemble_results[ensemble_type] = {'error': str(e)}
        
        return ensemble_results
    
    def _select_best_model(self, 
                          model_results: Dict[str, Any],
                          ensemble_results: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Select the best performing model from all evaluations.
        
        Args:
            model_results: Individual model results
            ensemble_results: Ensemble model results
            
        Returns:
            Best model name and info
        """
        best_score = 0
        best_name = None
        best_info = None
        
        # Check individual models
        for name, result in model_results.items():
            if 'error' in result:
                continue
            
            f1_score = result['cv_scores']['f1']['test_mean']
            if f1_score > best_score:
                best_score = f1_score
                best_name = name
                best_info = result
        
        # Check ensemble models
        for name, result in ensemble_results.items():
            if 'error' in result:
                continue
            
            f1_score = result['cv_scores']['f1']['mean']
            if f1_score > best_score:
                best_score = f1_score
                best_name = f"ensemble_{name}"
                best_info = result
        
        return best_name, best_info
    
    def _analyze_explainability(self, 
                               model: BaseEstimator,
                               X: np.ndarray,
                               y: np.ndarray,
                               texts: Optional[List[str]] = None,
                               feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze model explainability using multiple methods.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target labels
            texts: Original texts (optional)
            feature_names: Feature names (optional)
            
        Returns:
            Explainability analysis results
        """
        explainability_results = {}
        
        try:
            # Initialize explainer
            # For this example, we'll use a simple feature extractor
            from sklearn.feature_extraction.text import TfidfVectorizer
            simple_extractor = TfidfVectorizer(max_features=1000)
            
            if texts is not None:
                simple_extractor.fit(texts)
            
            explainer = ModelExplainer(
                model, simple_extractor, feature_names
            )
            
            # Analyze a few sample predictions
            if texts is not None:
                sample_texts = texts[:5]  # Analyze first 5 samples
                
                explanations = []
                for text in sample_texts:
                    explanation = explainer.explain_prediction(
                        text, methods=self.config['explainability']['methods']
                    )
                    explanations.append(explanation)
                
                explainability_results['sample_explanations'] = explanations
            
            # Feature importance analysis
            if hasattr(model, 'feature_importances_'):
                importance_analysis = {
                    'feature_importances': model.feature_importances_,
                    'top_features': self._get_top_features(
                        model.feature_importances_, feature_names, top_k=20
                    )
                }
                explainability_results['feature_importance'] = importance_analysis
            
            elif hasattr(model, 'coef_'):
                coefficients = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                importance_analysis = {
                    'coefficients': coefficients,
                    'top_features': self._get_top_features(
                        np.abs(coefficients), feature_names, top_k=20
                    )
                }
                explainability_results['feature_importance'] = importance_analysis
            
        except Exception as e:
            print(f"Error in explainability analysis: {e}")
            explainability_results['error'] = str(e)
        
        return explainability_results
    
    def _get_top_features(self, 
                         importance_scores: np.ndarray,
                         feature_names: Optional[List[str]] = None,
                         top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Get top features by importance score.
        
        Args:
            importance_scores: Feature importance scores
            feature_names: Feature names (optional)
            top_k: Number of top features to return
            
        Returns:
            List of top features with scores
        """
        top_indices = np.argsort(importance_scores)[-top_k:][::-1]
        
        top_features = []
        for i, idx in enumerate(top_indices):
            feature_info = {
                'rank': i + 1,
                'index': int(idx),
                'score': float(importance_scores[idx])
            }
            
            if feature_names and idx < len(feature_names):
                feature_info['name'] = feature_names[idx]
            else:
                feature_info['name'] = f'feature_{idx}'
            
            top_features.append(feature_info)
        
        return top_features
    
    def _evaluate_on_test_set(self, 
                             model: BaseEstimator,
                             X_test: np.ndarray,
                             y_test: np.ndarray,
                             optimal_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Evaluate the best model on the test set.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            optimal_threshold: Optimal classification threshold
            
        Returns:
            Test evaluation results
        """
        # Make predictions
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= optimal_threshold).astype(int)
        else:
            y_pred = model.predict(X_test)
            y_proba = None
        
        # Calculate metrics
        test_results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        if y_proba is not None:
            test_results['roc_auc'] = roc_auc_score(y_test, y_proba)
        
        return test_results
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """
        Generate actionable recommendations based on experiment results.
        
        Args:
            results: Experiment results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Feature extraction recommendations
        if 'feature_extraction' in results:
            feature_info = results['feature_extraction']
            if 'extractor_info' in feature_info:
                total_features = sum(
                    info['n_features'] for info in feature_info['extractor_info'].values()
                )
                recommendations.append(
                    f"Successfully extracted {total_features} features using {len(feature_info['extractor_info'])} different methods"
                )
        
        # Imbalance handling recommendations
        if 'imbalance_handling' in results:
            best_strategy = results['imbalance_handling']['best_strategy']
            recommendations.append(
                f"Best imbalance handling strategy: {best_strategy['strategy']} with {best_strategy['model']}"
            )
        
        # Model performance recommendations
        if 'best_model_info' in results:
            best_model = results['best_model_info']
            recommendations.append(
                f"Best performing model: {best_model['model_name']}"
            )
        
        # Meta-learning recommendations
        if 'meta_learning' in results and 'recommendations' in results['meta_learning']:
            recommendations.extend(results['meta_learning']['recommendations'])
        
        # Test performance recommendations
        if 'test_evaluation' in results:
            test_f1 = results['test_evaluation']['f1']
            if test_f1 > 0.8:
                recommendations.append("Excellent test performance achieved - model ready for production")
            elif test_f1 > 0.7:
                recommendations.append("Good test performance - consider additional fine-tuning")
            else:
                recommendations.append("Test performance needs improvement - review feature engineering and model selection")
        
        return recommendations
    
    def save_experiment_results(self, 
                               experiment_name: str,
                               save_dir: str = "experiment_results") -> str:
        """
        Save experiment results to disk.
        
        Args:
            experiment_name: Name of the experiment
            save_dir: Directory to save results
            
        Returns:
            Path to saved results
        """
        if experiment_name not in self.experiment_results:
            raise ValueError(f"Experiment {experiment_name} not found")
        
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        results = self.experiment_results[experiment_name].copy()
        
        # Remove non-serializable objects
        self._clean_results_for_saving(results)
        
        # Save results
        results_file = save_path / f"{experiment_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Experiment results saved to: {results_file}")
        return str(results_file)
    
    def _clean_results_for_saving(self, results: Dict[str, Any]):
        """
        Clean results dictionary by removing non-serializable objects.
        
        Args:
            results: Results dictionary to clean
        """
        # Remove model objects and other non-serializable items
        if 'model_evaluation' in results:
            for model_name, model_result in results['model_evaluation'].items():
                if 'model' in model_result:
                    del model_result['model']
        
        if 'ensemble_evaluation' in results:
            for ensemble_name, ensemble_result in results['ensemble_evaluation'].items():
                if 'ensemble' in ensemble_result:
                    del ensemble_result['ensemble']
        
        if 'best_model_info' in results and 'model_info' in results['best_model_info']:
            if 'model' in results['best_model_info']['model_info']:
                del results['best_model_info']['model_info']['model']
    
    def generate_comprehensive_report(self, experiment_name: str) -> str:
        """
        Generate a comprehensive research report for an experiment.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Comprehensive report as string
        """
        if experiment_name not in self.experiment_results:
            raise ValueError(f"Experiment {experiment_name} not found")
        
        results = self.experiment_results[experiment_name]
        
        report = f"# Comprehensive ML Research Report: {experiment_name}\n\n"
        report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Executive Summary
        report += "## Executive Summary\n\n"
        if 'best_model_info' in results:
            best_model = results['best_model_info']['model_name']
            report += f"- **Best Model:** {best_model}\n"
        
        if 'test_evaluation' in results:
            test_f1 = results['test_evaluation']['f1']
            report += f"- **Test F1 Score:** {test_f1:.4f}\n"
        
        report += "\n"
        
        # Feature Engineering
        report += "## Feature Engineering\n\n"
        if 'feature_extraction' in results:
            feature_info = results['feature_extraction']
            if 'extractor_info' in feature_info:
                for name, info in feature_info['extractor_info'].items():
                    report += f"- **{name.title()} Features:** {info['n_features']} features using {info['feature_type']}\n"
        
        report += "\n"
        
        # Model Evaluation
        report += "## Model Evaluation Results\n\n"
        if 'model_evaluation' in results:
            report += "### Individual Models\n\n"
            for model_name, model_result in results['model_evaluation'].items():
                if 'error' not in model_result:
                    f1_mean = model_result['cv_scores']['f1']['test_mean']
                    f1_std = model_result['cv_scores']['f1']['test_std']
                    report += f"- **{model_name}:** F1 = {f1_mean:.4f} (±{f1_std:.4f})\n"
        
        if 'ensemble_evaluation' in results:
            report += "\n### Ensemble Models\n\n"
            for ensemble_name, ensemble_result in results['ensemble_evaluation'].items():
                if 'error' not in ensemble_result:
                    f1_mean = ensemble_result['cv_scores']['f1']['mean']
                    f1_std = ensemble_result['cv_scores']['f1']['std']
                    report += f"- **{ensemble_name.title()} Ensemble:** F1 = {f1_mean:.4f} (±{f1_std:.4f})\n"
        
        report += "\n"
        
        # Imbalance Handling
        report += "## Imbalance Handling\n\n"
        if 'imbalance_handling' in results:
            best_strategy = results['imbalance_handling']['best_strategy']
            report += f"- **Best Strategy:** {best_strategy['strategy']}\n"
            report += f"- **Best Model for Strategy:** {best_strategy['model']}\n"
            report += f"- **Performance Score:** {best_strategy['score']:.4f}\n\n"
        
        # Meta-Learning
        report += "## Meta-Learning Approaches\n\n"
        if 'meta_learning' in results:
            meta_results = results['meta_learning']
            
            if 'adversarial_validation' in meta_results and meta_results['adversarial_validation']:
                adv = meta_results['adversarial_validation']
                report += f"- **Adversarial Validation AUC:** {adv['auc_score']:.4f}\n"
                report += f"- **Distribution Shift Detected:** {adv['is_distribution_shift']}\n"
            
            if 'unsupervised_features' in meta_results and meta_results['unsupervised_features']:
                report += "- **Unsupervised Features:** Successfully extracted clustering and dimensionality reduction features\n"
        
        report += "\n"
        
        # Test Results
        if 'test_evaluation' in results:
            report += "## Final Test Results\n\n"
            test_results = results['test_evaluation']
            
            report += f"- **Accuracy:** {test_results['accuracy']:.4f}\n"
            report += f"- **F1 Score:** {test_results['f1']:.4f}\n"
            report += f"- **Precision:** {test_results['precision']:.4f}\n"
            report += f"- **Recall:** {test_results['recall']:.4f}\n"
            
            if 'roc_auc' in test_results:
                report += f"- **ROC AUC:** {test_results['roc_auc']:.4f}\n"
            
            report += "\n"
        
        # Recommendations
        report += "## Recommendations\n\n"
        if 'recommendations' in results:
            for i, rec in enumerate(results['recommendations'], 1):
                report += f"{i}. {rec}\n"
        
        report += "\n"
        
        # Technical Details
        report += "## Technical Configuration\n\n"
        report += "```json\n"
        report += json.dumps(results['config'], indent=2)
        report += "\n```\n"
        
        return report
    
    def plot_experiment_results(self, 
                               experiment_name: str,
                               save_path: Optional[str] = None):
        """
        Create comprehensive visualizations for experiment results.
        
        Args:
            experiment_name: Name of the experiment
            save_path: Path to save the plots (optional)
        """
        if experiment_name not in self.experiment_results:
            raise ValueError(f"Experiment {experiment_name} not found")
        
        results = self.experiment_results[experiment_name]
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Comprehensive Results: {experiment_name}', fontsize=16)
        
        # Model comparison
        if 'model_evaluation' in results:
            self._plot_model_comparison(results['model_evaluation'], axes[0, 0])
        
        # Ensemble comparison
        if 'ensemble_evaluation' in results:
            self._plot_ensemble_comparison(results['ensemble_evaluation'], axes[0, 1])
        
        # Feature importance
        if 'explainability' in results and 'feature_importance' in results['explainability']:
            self._plot_feature_importance(results['explainability']['feature_importance'], axes[0, 2])
        
        # Imbalance strategy comparison
        if 'imbalance_handling' in results:
            self._plot_imbalance_strategies(results['imbalance_handling'], axes[1, 0])
        
        # Test results
        if 'test_evaluation' in results:
            self._plot_test_results(results['test_evaluation'], axes[1, 1])
        
        # Configuration summary
        self._plot_config_summary(results['config'], axes[1, 2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _plot_model_comparison(self, model_results: Dict[str, Any], ax):
        """Plot model comparison."""
        models = []
        f1_scores = []
        f1_stds = []
        
        for name, result in model_results.items():
            if 'error' not in result:
                models.append(name)
                f1_scores.append(result['cv_scores']['f1']['test_mean'])
                f1_stds.append(result['cv_scores']['f1']['test_std'])
        
        if models:
            bars = ax.bar(models, f1_scores, yerr=f1_stds, capsize=5)
            ax.set_title('Model Performance Comparison')
            ax.set_ylabel('F1 Score')
            ax.tick_params(axis='x', rotation=45)
            
            # Color best model
            best_idx = np.argmax(f1_scores)
            bars[best_idx].set_color('gold')
    
    def _plot_ensemble_comparison(self, ensemble_results: Dict[str, Any], ax):
        """Plot ensemble comparison."""
        ensembles = []
        f1_scores = []
        f1_stds = []
        
        for name, result in ensemble_results.items():
            if 'error' not in result:
                ensembles.append(name.replace('_', ' ').title())
                f1_scores.append(result['cv_scores']['f1']['mean'])
                f1_stds.append(result['cv_scores']['f1']['std'])
        
        if ensembles:
            ax.bar(ensembles, f1_scores, yerr=f1_stds, capsize=5, color='lightblue')
            ax.set_title('Ensemble Performance Comparison')
            ax.set_ylabel('F1 Score')
            ax.tick_params(axis='x', rotation=45)
    
    def _plot_feature_importance(self, importance_data: Dict[str, Any], ax):
        """Plot feature importance."""
        if 'top_features' in importance_data:
            top_features = importance_data['top_features'][:10]  # Top 10
            names = [f['name'][:20] for f in top_features]  # Truncate names
            scores = [f['score'] for f in top_features]
            
            ax.barh(names, scores)
            ax.set_title('Top Feature Importance')
            ax.set_xlabel('Importance Score')
    
    def _plot_imbalance_strategies(self, imbalance_data: Dict[str, Any], ax):
        """Plot imbalance handling strategies."""
        if 'strategy_comparison' in imbalance_data:
            # This would need the actual DataFrame data
            ax.text(0.5, 0.5, 'Imbalance Strategy\nComparison', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Imbalance Handling Strategies')
    
    def _plot_test_results(self, test_results: Dict[str, Any], ax):
        """Plot test results."""
        metrics = ['accuracy', 'f1', 'precision', 'recall']
        values = [test_results.get(metric, 0) for metric in metrics]
        
        bars = ax.bar(metrics, values)
        ax.set_title('Final Test Performance')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        
        # Color bars by performance
        for bar, value in zip(bars, values):
            if value > 0.8:
                bar.set_color('green')
            elif value > 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')
    
    def _plot_config_summary(self, config: Dict[str, Any], ax):
        """Plot configuration summary."""
        # Count enabled features
        enabled_features = []
        
        if config['feature_extraction']['use_hybrid_features']:
            enabled_features.append('Hybrid Features')
        if config['feature_extraction']['use_domain_features']:
            enabled_features.append('Domain Features')
        if config['feature_extraction']['use_topic_modeling']:
            enabled_features.append('Topic Modeling')
        
        enabled_ensembles = sum(config['ensemble_methods'].values())
        enabled_meta = sum(config['meta_learning'].values())
        
        categories = ['Features', 'Ensembles', 'Meta-Learning']
        counts = [len(enabled_features), enabled_ensembles, enabled_meta]
        
        ax.bar(categories, counts, color=['skyblue', 'lightgreen', 'lightcoral'])
        ax.set_title('Configuration Summary')
        ax.set_ylabel('Enabled Components')

# Example usage and testing
if __name__ == "__main__":
    # This would be used for testing the pipeline
    print("Research ML Pipeline initialized successfully!")
    
    # Example configuration
    config = {
        'feature_extraction': {
            'use_hybrid_features': True,
            'use_domain_features': True,
            'use_topic_modeling': False,  # Disable for faster testing
        },
        'ensemble_methods': {
            'use_diverse_ensemble': True,
            'use_adaptive_ensemble': False,
            'use_hierarchical_ensemble': False,
        },
        'imbalance_handling': {
            'strategy': 'combined',
            'sampling_method': 'smote',
        },
        'meta_learning': {
            'adversarial_validation': False,  # Disable for faster testing
            'pseudo_labeling': False,
            'unsupervised_features': True,
        }
    }
    
    # Initialize pipeline
    pipeline = ResearchMLPipeline(config=config, random_state=42)
    print("Pipeline components initialized successfully!")