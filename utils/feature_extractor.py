#!/usr/bin/env python3
"""
Feature Extraction Module for Text Classification
Implements TF-IDF (word-level, char-level), n-grams, and Bag of Words
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from scipy.sparse import hstack, csr_matrix
from typing import Tuple, List, Optional, Dict, Any
import joblib
import os

class FeatureExtractor:
    """
    Comprehensive feature extraction for text classification
    Combines multiple feature extraction methods
    """
    
    def __init__(self, 
                 max_features_word: int = 10000,
                 max_features_char: int = 5000,
                 word_ngram_range: Tuple[int, int] = (1, 3),
                 char_ngram_range: Tuple[int, int] = (3, 6),
                 min_df: int = 2,
                 max_df: float = 0.95,
                 use_idf: bool = True,
                 sublinear_tf: bool = True,
                 feature_selection_k: Optional[int] = None):
        """
        Initialize feature extractor
        
        Args:
            max_features_word: Maximum number of word-level features
            max_features_char: Maximum number of char-level features
            word_ngram_range: N-gram range for word-level features
            char_ngram_range: N-gram range for char-level features
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            use_idf: Whether to use IDF weighting
            sublinear_tf: Whether to use sublinear TF scaling
            feature_selection_k: Number of top features to select (None = no selection)
        """
        self.max_features_word = max_features_word
        self.max_features_char = max_features_char
        self.word_ngram_range = word_ngram_range
        self.char_ngram_range = char_ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.use_idf = use_idf
        self.sublinear_tf = sublinear_tf
        self.feature_selection_k = feature_selection_k
        
        # Initialize vectorizers
        self.word_tfidf = None
        self.char_tfidf = None
        self.bow_vectorizer = None
        self.feature_selector = None
        
        # Feature names for interpretability
        self.feature_names = []
        self.is_fitted = False
    
    def _create_word_tfidf(self) -> TfidfVectorizer:
        """Create word-level TF-IDF vectorizer"""
        return TfidfVectorizer(
            max_features=self.max_features_word,
            ngram_range=self.word_ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            use_idf=self.use_idf,
            sublinear_tf=self.sublinear_tf,
            analyzer='word',
            token_pattern=r'\b\w+\b',
            lowercase=False  # Already handled in preprocessing
        )
    
    def _create_char_tfidf(self) -> TfidfVectorizer:
        """Create character-level TF-IDF vectorizer"""
        return TfidfVectorizer(
            max_features=self.max_features_char,
            ngram_range=self.char_ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            use_idf=self.use_idf,
            sublinear_tf=self.sublinear_tf,
            analyzer='char',
            lowercase=False  # Already handled in preprocessing
        )
    
    def _create_bow_vectorizer(self) -> CountVectorizer:
        """Create Bag of Words vectorizer"""
        return CountVectorizer(
            max_features=self.max_features_word // 2,
            ngram_range=(1, 2),
            min_df=self.min_df,
            max_df=self.max_df,
            analyzer='word',
            token_pattern=r'\b\w+\b',
            lowercase=False  # Already handled in preprocessing
        )
    
    def fit(self, texts: List[str], labels: Optional[np.ndarray] = None) -> 'FeatureExtractor':
        """Fit all vectorizers on training data"""
        print("Fitting feature extractors...")
        
        # Initialize vectorizers
        self.word_tfidf = self._create_word_tfidf()
        self.char_tfidf = self._create_char_tfidf()
        self.bow_vectorizer = self._create_bow_vectorizer()
        
        # Fit vectorizers
        print("Fitting word-level TF-IDF...")
        word_features = self.word_tfidf.fit_transform(texts)
        
        print("Fitting character-level TF-IDF...")
        char_features = self.char_tfidf.fit_transform(texts)
        
        print("Fitting Bag of Words...")
        bow_features = self.bow_vectorizer.fit_transform(texts)
        
        # Combine features
        combined_features = hstack([word_features, char_features, bow_features])
        
        # Store feature names
        self.feature_names = (
            [f"word_tfidf_{name}" for name in self.word_tfidf.get_feature_names_out()] +
            [f"char_tfidf_{name}" for name in self.char_tfidf.get_feature_names_out()] +
            [f"bow_{name}" for name in self.bow_vectorizer.get_feature_names_out()]
        )
        
        # Apply feature selection if specified
        if self.feature_selection_k and labels is not None:
            print(f"Applying feature selection (k={self.feature_selection_k})...")
            self.feature_selector = SelectKBest(chi2, k=self.feature_selection_k)
            self.feature_selector.fit(combined_features, labels)
            
            # Update feature names
            selected_indices = self.feature_selector.get_support(indices=True)
            self.feature_names = [self.feature_names[i] for i in selected_indices]
        
        self.is_fitted = True
        print(f"Feature extraction fitted. Total features: {len(self.feature_names)}")
        return self
    
    def transform(self, texts: List[str]) -> csr_matrix:
        """Transform texts to feature matrix"""
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted before transform")
        
        # Extract features
        word_features = self.word_tfidf.transform(texts)
        char_features = self.char_tfidf.transform(texts)
        bow_features = self.bow_vectorizer.transform(texts)
        
        # Combine features
        combined_features = hstack([word_features, char_features, bow_features])
        
        # Apply feature selection if fitted
        if self.feature_selector is not None:
            combined_features = self.feature_selector.transform(combined_features)
        
        return combined_features
    
    def fit_transform(self, texts: List[str], labels: Optional[np.ndarray] = None) -> csr_matrix:
        """Fit and transform in one step"""
        return self.fit(texts, labels).transform(texts)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for interpretability"""
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted first")
        return self.feature_names
    
    def get_feature_importance(self, model, top_k: int = 20) -> Dict[str, float]:
        """Get top feature importances from a fitted model"""
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted first")
        
        # Get feature importances/coefficients
        if hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0] if model.coef_.ndim > 1 else model.coef_)
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            raise ValueError("Model doesn't have feature importances or coefficients")
        
        # Get top features
        top_indices = np.argsort(importances)[-top_k:]
        feature_importance_dict = {
            self.feature_names[i]: importances[i] for i in top_indices
        }
        
        return dict(sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def save(self, filepath: str) -> None:
        """Save the fitted feature extractor"""
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted before saving")
        
        save_dict = {
            'word_tfidf': self.word_tfidf,
            'char_tfidf': self.char_tfidf,
            'bow_vectorizer': self.bow_vectorizer,
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names,
            'config': {
                'max_features_word': self.max_features_word,
                'max_features_char': self.max_features_char,
                'word_ngram_range': self.word_ngram_range,
                'char_ngram_range': self.char_ngram_range,
                'min_df': self.min_df,
                'max_df': self.max_df,
                'use_idf': self.use_idf,
                'sublinear_tf': self.sublinear_tf,
                'feature_selection_k': self.feature_selection_k
            }
        }
        
        joblib.dump(save_dict, filepath)
        print(f"Feature extractor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FeatureExtractor':
        """Load a fitted feature extractor"""
        save_dict = joblib.load(filepath)
        
        # Create instance with saved config
        instance = cls(**save_dict['config'])
        
        # Restore fitted components
        instance.word_tfidf = save_dict['word_tfidf']
        instance.char_tfidf = save_dict['char_tfidf']
        instance.bow_vectorizer = save_dict['bow_vectorizer']
        instance.feature_selector = save_dict['feature_selector']
        instance.feature_names = save_dict['feature_names']
        instance.is_fitted = True
        
        return instance

# Example usage and testing
if __name__ == "__main__":
    # Test the feature extractor
    sample_texts = [
        "This is a sample text for testing",
        "Another text with different words",
        "Text classification requires good features",
        "Machine learning models need proper preprocessing"
    ]
    
    sample_labels = np.array([0, 1, 1, 0])
    
    # Create and fit feature extractor
    extractor = FeatureExtractor(
        max_features_word=1000,
        max_features_char=500,
        feature_selection_k=100
    )
    
    # Fit and transform
    features = extractor.fit_transform(sample_texts, sample_labels)
    
    print(f"Feature matrix shape: {features.shape}")
    print(f"Number of feature names: {len(extractor.get_feature_names())}")
    print(f"First 10 feature names: {extractor.get_feature_names()[:10]}")