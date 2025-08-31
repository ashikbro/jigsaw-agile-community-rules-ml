import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class AdversarialValidator:
    """
    Adversarial validation to detect distribution shift between train and test data.
    """
    
    def __init__(self, 
                 model: Optional[BaseEstimator] = None,
                 cv_folds: int = 5,
                 random_state: Optional[int] = None):
        """
        Initialize adversarial validator.
        
        Args:
            model: Model to use for adversarial validation (default: RandomForest)
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
        """
        self.model = model or RandomForestClassifier(
            n_estimators=100, 
            random_state=random_state
        )
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.auc_score_ = None
        self.feature_importance_ = None
        self.is_distribution_shift_ = False
    
    def validate(self, 
                X_train: np.ndarray, 
                X_test: np.ndarray,
                threshold: float = 0.55) -> Dict[str, Any]:
        """
        Perform adversarial validation to detect distribution shift.
        
        Args:
            X_train: Training features
            X_test: Test features
            threshold: AUC threshold above which distribution shift is detected
            
        Returns:
            Dictionary with validation results
        """
        # Create adversarial dataset
        # Handle sparse matrices
        if sparse.issparse(X_train) or sparse.issparse(X_test):
            X_combined = sparse.vstack([X_train, X_test])
            train_size = X_train.shape[0]
            test_size = X_test.shape[0]
        else:
            X_combined = np.vstack([X_train, X_test])
            train_size = len(X_train)
            test_size = len(X_test)
        
        y_adversarial = np.hstack([
            np.zeros(train_size),  # Train = 0
            np.ones(test_size)     # Test = 1
        ])
        
        # Cross-validation to estimate AUC
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        auc_scores = []
        
        for train_idx, val_idx in cv.split(X_combined, y_adversarial):
            X_cv_train, X_cv_val = X_combined[train_idx], X_combined[val_idx]
            y_cv_train, y_cv_val = y_adversarial[train_idx], y_adversarial[val_idx]
            
            # Fit model
            self.model.fit(X_cv_train, y_cv_train)
            
            # Predict probabilities
            if hasattr(self.model, 'predict_proba'):
                y_proba = self.model.predict_proba(X_cv_val)[:, 1]
            else:
                y_proba = self.model.decision_function(X_cv_val)
            
            # Calculate AUC
            auc = roc_auc_score(y_cv_val, y_proba)
            auc_scores.append(auc)
        
        self.auc_score_ = np.mean(auc_scores)
        self.is_distribution_shift_ = self.auc_score_ > threshold
        
        # Fit on full data to get feature importance
        self.model.fit(X_combined, y_adversarial)
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            self.feature_importance_ = np.abs(self.model.coef_[0])
        
        result = {
            'auc_score': self.auc_score_,
            'auc_std': np.std(auc_scores),
            'is_distribution_shift': self.is_distribution_shift_,
            'threshold': threshold,
            'feature_importance': self.feature_importance_,
            'recommendation': self._get_recommendation()
        }
        
        return result
    
    def _get_recommendation(self) -> str:
        """
        Get recommendation based on adversarial validation results.
        
        Returns:
            Recommendation string
        """
        if self.auc_score_ is None:
            return "Run adversarial validation first"
        
        if self.auc_score_ < 0.52:
            return "No significant distribution shift detected. Proceed with standard validation."
        elif self.auc_score_ < 0.55:
            return "Minor distribution shift detected. Consider domain adaptation techniques."
        elif self.auc_score_ < 0.65:
            return "Moderate distribution shift detected. Use robust validation and feature engineering."
        else:
            return "Significant distribution shift detected. Consider data collection or domain adaptation."
    
    def plot_adversarial_results(self, 
                               feature_names: Optional[List[str]] = None,
                               top_features: int = 20,
                               save_path: Optional[str] = None):
        """
        Plot adversarial validation results.
        
        Args:
            feature_names: Names of features (optional)
            top_features: Number of top features to show
            save_path: Path to save the plot (optional)
        """
        if self.feature_importance_ is None:
            print("No feature importance available. Run validate() first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # AUC score visualization
        ax1.bar(['AUC Score'], [self.auc_score_], 
               color='red' if self.is_distribution_shift_ else 'green')
        ax1.axhline(y=0.55, color='orange', linestyle='--', label='Threshold')
        ax1.set_ylabel('AUC Score')
        ax1.set_title('Adversarial Validation AUC')
        ax1.legend()
        ax1.set_ylim(0.4, 1.0)
        
        # Feature importance
        top_indices = np.argsort(self.feature_importance_)[-top_features:]
        top_importance = self.feature_importance_[top_indices]
        
        if feature_names:
            top_names = [feature_names[i] if i < len(feature_names) else f'Feature_{i}' 
                        for i in top_indices]
        else:
            top_names = [f'Feature_{i}' for i in top_indices]
        
        ax2.barh(range(len(top_importance)), top_importance)
        ax2.set_yticks(range(len(top_importance)))
        ax2.set_yticklabels(top_names)
        ax2.set_xlabel('Feature Importance')
        ax2.set_title('Top Features Distinguishing Train/Test')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

class PseudoLabeler:
    """
    Pseudo-labeling for semi-supervised learning.
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.9,
                 max_iterations: int = 5,
                 min_improvement: float = 0.001):
        """
        Initialize pseudo-labeler.
        
        Args:
            confidence_threshold: Minimum confidence for pseudo-labels
            max_iterations: Maximum number of iterations
            min_improvement: Minimum improvement to continue iterations
        """
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.min_improvement = min_improvement
        self.iteration_history_ = []
    
    def fit_predict(self, 
                   model: BaseEstimator,
                   X_labeled: np.ndarray,
                   y_labeled: np.ndarray,
                   X_unlabeled: np.ndarray,
                   X_val: Optional[np.ndarray] = None,
                   y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform pseudo-labeling iteratively.
        
        Args:
            model: Base model to use
            X_labeled: Labeled training data
            y_labeled: Labels for training data
            X_unlabeled: Unlabeled data
            X_val: Validation data (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary with results and pseudo-labeled data
        """
        if not hasattr(model, 'predict_proba'):
            raise ValueError("Model must have predict_proba method for pseudo-labeling")
        
        # Initialize
        X_current = X_labeled.copy()
        y_current = y_labeled.copy()
        X_remaining = X_unlabeled.copy()
        
        best_score = 0
        best_model = None
        
        for iteration in range(self.max_iterations):
            print(f"Pseudo-labeling iteration {iteration + 1}/{self.max_iterations}")
            
            # Train model on current labeled data
            model.fit(X_current, y_current)
            
            # Predict on remaining unlabeled data
            probabilities = model.predict_proba(X_remaining)
            max_probs = np.max(probabilities, axis=1)
            predicted_labels = np.argmax(probabilities, axis=1)
            
            # Select high-confidence predictions
            confident_mask = max_probs >= self.confidence_threshold
            
            if np.sum(confident_mask) == 0:
                print(f"No confident predictions found in iteration {iteration + 1}")
                break
            
            # Add confident predictions to labeled data
            X_confident = X_remaining[confident_mask]
            y_confident = predicted_labels[confident_mask]
            
            X_current = np.vstack([X_current, X_confident])
            y_current = np.hstack([y_current, y_confident])
            
            # Remove confident samples from unlabeled data
            X_remaining = X_remaining[~confident_mask]
            
            # Evaluate on validation set if available
            val_score = 0
            if X_val is not None and y_val is not None:
                val_predictions = model.predict(X_val)
                val_score = f1_score(y_val, val_predictions)
            
            # Store iteration info
            iteration_info = {
                'iteration': iteration + 1,
                'labeled_samples': len(X_current),
                'confident_samples': np.sum(confident_mask),
                'remaining_samples': len(X_remaining),
                'validation_score': val_score,
                'avg_confidence': np.mean(max_probs[confident_mask]) if np.sum(confident_mask) > 0 else 0
            }
            self.iteration_history_.append(iteration_info)
            
            print(f"Added {np.sum(confident_mask)} confident samples")
            print(f"Total labeled samples: {len(X_current)}")
            print(f"Remaining unlabeled: {len(X_remaining)}")
            if X_val is not None:
                print(f"Validation F1: {val_score:.4f}")
            
            # Check for improvement
            if val_score > best_score + self.min_improvement:
                best_score = val_score
                best_model = model
            elif iteration > 0 and val_score < best_score - self.min_improvement:
                print("Performance degraded. Stopping early.")
                break
            
            if len(X_remaining) == 0:
                print("All unlabeled data has been pseudo-labeled.")
                break
        
        result = {
            'final_model': model,
            'best_model': best_model,
            'X_final': X_current,
            'y_final': y_current,
            'X_remaining': X_remaining,
            'iteration_history': self.iteration_history_,
            'total_pseudo_labeled': len(X_current) - len(X_labeled),
            'final_validation_score': val_score
        }
        
        return result
    
    def plot_pseudo_labeling_progress(self, save_path: Optional[str] = None):
        """
        Plot pseudo-labeling progress over iterations.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.iteration_history_:
            print("No iteration history available. Run fit_predict() first.")
            return
        
        df = pd.DataFrame(self.iteration_history_)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Pseudo-Labeling Progress')
        
        # Labeled samples over time
        axes[0, 0].plot(df['iteration'], df['labeled_samples'], 'b-o')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Total Labeled Samples')
        axes[0, 0].set_title('Growth of Labeled Dataset')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Confident samples per iteration
        axes[0, 1].bar(df['iteration'], df['confident_samples'])
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Confident Samples Added')
        axes[0, 1].set_title('Confident Samples per Iteration')
        
        # Validation score over time
        if 'validation_score' in df.columns and df['validation_score'].sum() > 0:
            axes[1, 0].plot(df['iteration'], df['validation_score'], 'g-o')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Validation F1 Score')
            axes[1, 0].set_title('Validation Performance')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Average confidence over time
        axes[1, 1].plot(df['iteration'], df['avg_confidence'], 'r-o')
        axes[1, 1].axhline(y=self.confidence_threshold, color='orange', 
                          linestyle='--', label='Threshold')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Average Confidence')
        axes[1, 1].set_title('Confidence of Pseudo-Labels')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

class UnsupervisedFeatureExtractor:
    """
    Extract features using unsupervised clustering and dimensionality reduction.
    """
    
    def __init__(self, 
                 clustering_methods: List[str] = ['kmeans', 'dbscan'],
                 n_clusters_range: Tuple[int, int] = (5, 20),
                 dimensionality_reduction: List[str] = ['pca', 'svd'],
                 n_components: int = 50,
                 random_state: Optional[int] = None):
        """
        Initialize unsupervised feature extractor.
        
        Args:
            clustering_methods: List of clustering methods to use
            n_clusters_range: Range of cluster numbers to try
            dimensionality_reduction: List of dimensionality reduction methods
            n_components: Number of components for dimensionality reduction
            random_state: Random state for reproducibility
        """
        self.clustering_methods = clustering_methods
        self.n_clusters_range = n_clusters_range
        self.dimensionality_reduction = dimensionality_reduction
        self.n_components = n_components
        self.random_state = random_state
        
        self.clusterers_ = {}
        self.reducers_ = {}
        self.cluster_features_ = None
        self.reduced_features_ = None
    
    def fit_transform(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Fit and transform data using unsupervised methods.
        
        Args:
            X: Input feature matrix
            
        Returns:
            Dictionary with different types of unsupervised features
        """
        features = {}
        
        # Dimensionality reduction features
        print("Applying dimensionality reduction...")
        for method in self.dimensionality_reduction:
            if method == 'pca':
                if sparse.issparse(X):
                    # Use TruncatedSVD for sparse matrices
                    reducer = TruncatedSVD(
                        n_components=min(self.n_components, X.shape[1] - 1),
                        random_state=self.random_state
                    )
                else:
                    reducer = PCA(
                        n_components=min(self.n_components, X.shape[1]),
                        random_state=self.random_state
                    )
            elif method == 'svd':
                reducer = TruncatedSVD(
                    n_components=min(self.n_components, X.shape[1] - 1),
                    random_state=self.random_state
                )
            else:
                continue
            
            reduced_X = reducer.fit_transform(X)
            features[f'{method}_features'] = reduced_X
            self.reducers_[method] = reducer
            print(f"{method.upper()} reduced dimensions: {X.shape[1]} -> {reduced_X.shape[1]}")
        
        # Clustering features
        print("Applying clustering methods...")
        
        # Use reduced features for clustering if available
        clustering_X = X
        if 'pca_features' in features:
            clustering_X = features['pca_features']
        elif 'svd_features' in features:
            clustering_X = features['svd_features']
        
        # Convert to dense if sparse (for some clustering algorithms)
        if sparse.issparse(clustering_X):
            if clustering_X.shape[1] > 1000:  # Only convert if not too large
                clustering_X = clustering_X.toarray()
        
        cluster_features_list = []
        
        for method in self.clustering_methods:
            if method == 'kmeans':
                # Try different numbers of clusters
                for n_clusters in range(self.n_clusters_range[0], self.n_clusters_range[1] + 1, 5):
                    clusterer = KMeans(
                        n_clusters=n_clusters,
                        random_state=self.random_state,
                        n_init=10
                    )
                    
                    try:
                        cluster_labels = clusterer.fit_predict(clustering_X)
                        
                        # Create one-hot encoded cluster features
                        cluster_onehot = np.zeros((len(cluster_labels), n_clusters))
                        cluster_onehot[np.arange(len(cluster_labels)), cluster_labels] = 1
                        
                        cluster_features_list.append(cluster_onehot)
                        self.clusterers_[f'kmeans_{n_clusters}'] = clusterer
                        
                        print(f"K-means with {n_clusters} clusters completed")
                    except Exception as e:
                        print(f"K-means with {n_clusters} clusters failed: {e}")
            
            elif method == 'dbscan':
                # Try different eps values
                eps_values = [0.1, 0.5, 1.0, 2.0]
                
                for eps in eps_values:
                    clusterer = DBSCAN(eps=eps, min_samples=5)
                    
                    try:
                        cluster_labels = clusterer.fit_predict(clustering_X)
                        
                        # Check if we got meaningful clusters
                        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                        
                        if n_clusters > 1 and n_clusters < len(cluster_labels) * 0.8:
                            # Create cluster distance features
                            unique_labels = sorted(set(cluster_labels))
                            cluster_distances = np.zeros((len(cluster_labels), len(unique_labels)))
                            
                            for i, label in enumerate(unique_labels):
                                mask = cluster_labels == label
                                if np.sum(mask) > 0:
                                    cluster_center = np.mean(clustering_X[mask], axis=0)
                                    distances = np.linalg.norm(
                                        clustering_X - cluster_center, axis=1
                                    )
                                    cluster_distances[:, i] = distances
                            
                            cluster_features_list.append(cluster_distances)
                            self.clusterers_[f'dbscan_{eps}'] = clusterer
                            
                            print(f"DBSCAN with eps={eps} found {n_clusters} clusters")
                    except Exception as e:
                        print(f"DBSCAN with eps={eps} failed: {e}")
        
        # Combine cluster features
        if cluster_features_list:
            self.cluster_features_ = np.hstack(cluster_features_list)
            features['cluster_features'] = self.cluster_features_
            print(f"Total cluster features: {self.cluster_features_.shape[1]}")
        
        # Statistical features from clusters
        if 'cluster_features' in features:
            cluster_stats = self._compute_cluster_statistics(features['cluster_features'])
            features['cluster_statistics'] = cluster_stats
        
        return features
    
    def _compute_cluster_statistics(self, cluster_features: np.ndarray) -> np.ndarray:
        """
        Compute statistical features from cluster assignments.
        
        Args:
            cluster_features: Cluster feature matrix
            
        Returns:
            Statistical features
        """
        stats = []
        
        # Basic statistics
        stats.append(np.mean(cluster_features, axis=1))  # Mean cluster assignment
        stats.append(np.std(cluster_features, axis=1))   # Std of cluster assignments
        stats.append(np.max(cluster_features, axis=1))   # Max cluster assignment
        stats.append(np.min(cluster_features, axis=1))   # Min cluster assignment
        
        # Entropy of cluster assignments (measure of uncertainty)
        cluster_probs = cluster_features / (np.sum(cluster_features, axis=1, keepdims=True) + 1e-8)
        entropy = -np.sum(cluster_probs * np.log(cluster_probs + 1e-8), axis=1)
        stats.append(entropy)
        
        return np.column_stack(stats)
    
    def transform(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Transform new data using fitted unsupervised methods.
        
        Args:
            X: Input feature matrix
            
        Returns:
            Dictionary with transformed features
        """
        features = {}
        
        # Apply dimensionality reduction
        for method, reducer in self.reducers_.items():
            reduced_X = reducer.transform(X)
            features[f'{method}_features'] = reduced_X
        
        # Apply clustering
        clustering_X = X
        if 'pca_features' in features:
            clustering_X = features['pca_features']
        elif 'svd_features' in features:
            clustering_X = features['svd_features']
        
        if sparse.issparse(clustering_X) and clustering_X.shape[1] <= 1000:
            clustering_X = clustering_X.toarray()
        
        cluster_features_list = []
        
        for name, clusterer in self.clusterers_.items():
            try:
                if 'kmeans' in name:
                    cluster_labels = clusterer.predict(clustering_X)
                    n_clusters = clusterer.n_clusters
                    
                    # Create one-hot encoded features
                    cluster_onehot = np.zeros((len(cluster_labels), n_clusters))
                    cluster_onehot[np.arange(len(cluster_labels)), cluster_labels] = 1
                    cluster_features_list.append(cluster_onehot)
                
                elif 'dbscan' in name:
                    # For DBSCAN, compute distances to cluster centers
                    cluster_labels = clusterer.fit_predict(clustering_X)
                    unique_labels = sorted(set(cluster_labels))
                    cluster_distances = np.zeros((len(cluster_labels), len(unique_labels)))
                    
                    for i, label in enumerate(unique_labels):
                        mask = cluster_labels == label
                        if np.sum(mask) > 0:
                            cluster_center = np.mean(clustering_X[mask], axis=0)
                            distances = np.linalg.norm(
                                clustering_X - cluster_center, axis=1
                            )
                            cluster_distances[:, i] = distances
                    
                    cluster_features_list.append(cluster_distances)
            
            except Exception as e:
                print(f"Error applying {name}: {e}")
        
        if cluster_features_list:
            cluster_features = np.hstack(cluster_features_list)
            features['cluster_features'] = cluster_features
            
            # Compute statistics
            cluster_stats = self._compute_cluster_statistics(cluster_features)
            features['cluster_statistics'] = cluster_stats
        
        return features
    
    def plot_clustering_results(self, 
                              X: np.ndarray,
                              method: str = 'kmeans_10',
                              save_path: Optional[str] = None):
        """
        Plot clustering results using 2D visualization.
        
        Args:
            X: Original feature matrix
            method: Clustering method to visualize
            save_path: Path to save the plot (optional)
        """
        if method not in self.clusterers_:
            print(f"Method {method} not found in fitted clusterers.")
            return
        
        clusterer = self.clusterers_[method]
        
        # Reduce to 2D for visualization
        if X.shape[1] > 2:
            if sparse.issparse(X):
                reducer = TruncatedSVD(n_components=2, random_state=self.random_state)
            else:
                reducer = PCA(n_components=2, random_state=self.random_state)
            
            X_2d = reducer.fit_transform(X)
        else:
            X_2d = X
        
        # Get cluster labels
        if hasattr(clusterer, 'predict'):
            labels = clusterer.predict(X if X.shape[1] <= 50 else X_2d)
        else:
            labels = clusterer.fit_predict(X if X.shape[1] <= 50 else X_2d)
        
        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title(f'Clustering Results: {method}')
        
        # Plot cluster centers if available
        if hasattr(clusterer, 'cluster_centers_'):
            centers_2d = reducer.transform(clusterer.cluster_centers_)
            plt.scatter(centers_2d[:, 0], centers_2d[:, 1], 
                       c='red', marker='x', s=200, linewidths=3, label='Centroids')
            plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

class MetaLearningPipeline:
    """
    Comprehensive meta-learning pipeline combining all approaches.
    """
    
    def __init__(self, 
                 enable_adversarial_validation: bool = True,
                 enable_pseudo_labeling: bool = True,
                 enable_unsupervised_features: bool = True,
                 random_state: Optional[int] = None):
        """
        Initialize meta-learning pipeline.
        
        Args:
            enable_adversarial_validation: Whether to use adversarial validation
            enable_pseudo_labeling: Whether to use pseudo-labeling
            enable_unsupervised_features: Whether to extract unsupervised features
            random_state: Random state for reproducibility
        """
        self.enable_adversarial_validation = enable_adversarial_validation
        self.enable_pseudo_labeling = enable_pseudo_labeling
        self.enable_unsupervised_features = enable_unsupervised_features
        self.random_state = random_state
        
        # Initialize components
        self.adversarial_validator = None
        self.pseudo_labeler = None
        self.unsupervised_extractor = None
        
        if enable_adversarial_validation:
            self.adversarial_validator = AdversarialValidator(random_state=random_state)
        
        if enable_pseudo_labeling:
            self.pseudo_labeler = PseudoLabeler()
        
        if enable_unsupervised_features:
            self.unsupervised_extractor = UnsupervisedFeatureExtractor(random_state=random_state)
    
    def run_pipeline(self, 
                    X_train: np.ndarray,
                    y_train: np.ndarray,
                    X_test: Optional[np.ndarray] = None,
                    X_unlabeled: Optional[np.ndarray] = None,
                    base_model: Optional[BaseEstimator] = None) -> Dict[str, Any]:
        """
        Run the complete meta-learning pipeline.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (optional)
            X_unlabeled: Unlabeled data for pseudo-labeling (optional)
            base_model: Base model to use (default: LogisticRegression)
            
        Returns:
            Dictionary with all results
        """
        if base_model is None:
            base_model = LogisticRegression(random_state=self.random_state)
        
        results = {
            'adversarial_validation': None,
            'pseudo_labeling': None,
            'unsupervised_features': None,
            'enhanced_features': None,
            'recommendations': []
        }
        
        # Adversarial validation
        if self.enable_adversarial_validation and X_test is not None:
            print("Running adversarial validation...")
            adv_results = self.adversarial_validator.validate(X_train, X_test)
            results['adversarial_validation'] = adv_results
            results['recommendations'].append(adv_results['recommendation'])
            
            print(f"Adversarial validation AUC: {adv_results['auc_score']:.4f}")
            print(f"Distribution shift detected: {adv_results['is_distribution_shift']}")
        
        # Unsupervised feature extraction
        if self.enable_unsupervised_features:
            print("Extracting unsupervised features...")
            unsupervised_features = self.unsupervised_extractor.fit_transform(X_train)
            results['unsupervised_features'] = unsupervised_features
            
            # Combine original and unsupervised features
            enhanced_X_train = X_train
            for feature_type, features in unsupervised_features.items():
                if sparse.issparse(enhanced_X_train) and not sparse.issparse(features):
                    features = sparse.csr_matrix(features)
                elif not sparse.issparse(enhanced_X_train) and sparse.issparse(features):
                    enhanced_X_train = sparse.csr_matrix(enhanced_X_train)
                
                enhanced_X_train = sparse.hstack([enhanced_X_train, features]) if sparse.issparse(enhanced_X_train) else np.hstack([enhanced_X_train, features])
            
            results['enhanced_features'] = enhanced_X_train
            print(f"Enhanced features shape: {enhanced_X_train.shape}")
        
        # Pseudo-labeling
        if self.enable_pseudo_labeling and X_unlabeled is not None:
            print("Running pseudo-labeling...")
            
            # Use enhanced features if available
            X_for_pseudo = results.get('enhanced_features', X_train)
            
            # Transform unlabeled data with same unsupervised features
            if self.enable_unsupervised_features:
                unlabeled_features = self.unsupervised_extractor.transform(X_unlabeled)
                enhanced_X_unlabeled = X_unlabeled
                
                for feature_type, features in unlabeled_features.items():
                    if sparse.issparse(enhanced_X_unlabeled) and not sparse.issparse(features):
                        features = sparse.csr_matrix(features)
                    elif not sparse.issparse(enhanced_X_unlabeled) and sparse.issparse(features):
                        enhanced_X_unlabeled = sparse.csr_matrix(enhanced_X_unlabeled)
                    
                    enhanced_X_unlabeled = sparse.hstack([enhanced_X_unlabeled, features]) if sparse.issparse(enhanced_X_unlabeled) else np.hstack([enhanced_X_unlabeled, features])
            else:
                enhanced_X_unlabeled = X_unlabeled
            
            pseudo_results = self.pseudo_labeler.fit_predict(
                base_model, X_for_pseudo, y_train, enhanced_X_unlabeled
            )
            results['pseudo_labeling'] = pseudo_results
            
            print(f"Pseudo-labeled {pseudo_results['total_pseudo_labeled']} samples")
            results['recommendations'].append(
                f"Added {pseudo_results['total_pseudo_labeled']} pseudo-labeled samples to training set"
            )
        
        return results
    
    def generate_meta_learning_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive meta-learning report.
        
        Args:
            results: Results from run_pipeline
            
        Returns:
            Report as string
        """
        report = "# Meta-Learning Pipeline Report\n\n"
        
        # Adversarial validation section
        if results['adversarial_validation']:
            adv = results['adversarial_validation']
            report += "## Adversarial Validation\n"
            report += f"- AUC Score: {adv['auc_score']:.4f}\n"
            report += f"- Distribution Shift Detected: {adv['is_distribution_shift']}\n"
            report += f"- Recommendation: {adv['recommendation']}\n\n"
        
        # Unsupervised features section
        if results['unsupervised_features']:
            report += "## Unsupervised Feature Extraction\n"
            for feature_type, features in results['unsupervised_features'].items():
                report += f"- {feature_type}: {features.shape[1]} features\n"
            
            if results['enhanced_features'] is not None:
                original_shape = results['enhanced_features'].shape
                report += f"- Total enhanced features: {original_shape[1]}\n\n"
        
        # Pseudo-labeling section
        if results['pseudo_labeling']:
            pseudo = results['pseudo_labeling']
            report += "## Pseudo-Labeling\n"
            report += f"- Total pseudo-labeled samples: {pseudo['total_pseudo_labeled']}\n"
            report += f"- Iterations completed: {len(pseudo['iteration_history'])}\n"
            
            if pseudo['final_validation_score'] > 0:
                report += f"- Final validation score: {pseudo['final_validation_score']:.4f}\n"
            
            report += "\n### Iteration History\n"
            for iteration in pseudo['iteration_history']:
                report += f"- Iteration {iteration['iteration']}: "
                report += f"{iteration['confident_samples']} samples added, "
                report += f"validation F1: {iteration['validation_score']:.4f}\n"
            
            report += "\n"
        
        # Recommendations section
        if results['recommendations']:
            report += "## Recommendations\n"
            for i, rec in enumerate(results['recommendations'], 1):
                report += f"{i}. {rec}\n"
        
        return report