#!/usr/bin/env python3
"""
Knowledge-Enhanced ML for Content Moderation

This module implements lexicon-augmented hybrid ML features for explainable content moderation,
combining statistical features (TF-IDF, BoW) with domain-specific knowledge (slur dictionaries,
community-specific toxic words, multilingual slang).

Novelty: Lexicon-augmented hybrid ML features for explainable content moderation
with multilingual support and domain-specific knowledge integration.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union, Set
from collections import defaultdict, Counter
import warnings
import logging
import json
import re
import string
from pathlib import Path
import pickle
from dataclasses import dataclass

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import scipy.sparse as sp

# Text processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LexiconEntry:
    """
    Represents a single lexicon entry with metadata.
    """
    term: str
    category: str  # e.g., 'slur', 'profanity', 'hate_speech', 'threat'
    severity: float  # 0.0 to 1.0
    language: str
    context: Optional[str] = None
    variants: Optional[List[str]] = None
    source: Optional[str] = None

class MultilingualLexicon:
    """
    Manages multilingual lexicons for toxic content detection.
    """
    
    def __init__(self):
        self.lexicons = defaultdict(lambda: defaultdict(list))  # {language: {category: [entries]}}
        self.term_to_entry = {}  # Fast lookup: {term: LexiconEntry}
        self.category_weights = {
            'slur': 1.0,
            'profanity': 0.7,
            'hate_speech': 0.9,
            'threat': 0.95,
            'harassment': 0.8,
            'discrimination': 0.85,
            'violence': 0.9,
            'sexual': 0.6,
            'spam': 0.3,
            'mild_profanity': 0.4
        }
        
        self._initialize_default_lexicons()
    
    def _initialize_default_lexicons(self):
        """
        Initialize with default lexicons for common toxic terms.
        """
        # English toxic terms (sample - in practice, use comprehensive datasets)
        english_slurs = [
            ('idiot', 'slur', 0.6), ('stupid', 'slur', 0.5), ('moron', 'slur', 0.7),
            ('retard', 'slur', 0.9), ('dumb', 'slur', 0.4), ('loser', 'slur', 0.5)
        ]
        
        english_profanity = [
            ('damn', 'profanity', 0.3), ('hell', 'profanity', 0.3), ('crap', 'profanity', 0.4),
            ('shit', 'profanity', 0.7), ('fuck', 'profanity', 0.8), ('bitch', 'profanity', 0.8)
        ]
        
        english_hate_speech = [
            ('kill yourself', 'threat', 0.95), ('die', 'threat', 0.8), ('murder', 'violence', 0.9),
            ('terrorist', 'hate_speech', 0.85), ('nazi', 'hate_speech', 0.9)
        ]
        
        # Add English terms
        for term, category, severity in english_slurs + english_profanity + english_hate_speech:
            self.add_term(term, category, severity, 'en')
        
        # Bengali toxic terms (sample)
        bengali_terms = [
            ('বোকা', 'slur', 0.5), ('গাধা', 'slur', 0.6), ('পাগল', 'slur', 0.4),
            ('মূর্খ', 'slur', 0.6), ('হারামি', 'profanity', 0.8), ('শালা', 'profanity', 0.7)
        ]
        
        for term, category, severity in bengali_terms:
            self.add_term(term, category, severity, 'bn')
        
        # Hindi toxic terms (sample)
        hindi_terms = [
            ('बेवकूफ', 'slur', 0.5), ('गधा', 'slur', 0.6), ('पागल', 'slur', 0.4),
            ('मूर्ख', 'slur', 0.6), ('हरामी', 'profanity', 0.8), ('साला', 'profanity', 0.7)
        ]
        
        for term, category, severity in hindi_terms:
            self.add_term(term, category, severity, 'hi')
        
        # Internet slang and leetspeak variants
        leetspeak_variants = [
            ('1d10t', 'slur', 0.6), ('st00p1d', 'slur', 0.5), ('n00b', 'slur', 0.4),
            ('f4gg0t', 'slur', 0.9), ('b1tch', 'profanity', 0.8), ('sh1t', 'profanity', 0.7)
        ]
        
        for term, category, severity in leetspeak_variants:
            self.add_term(term, category, severity, 'en', context='leetspeak')
        
        logger.info(f"Initialized default lexicons with {len(self.term_to_entry)} terms")
    
    def add_term(self, term: str, category: str, severity: float, language: str,
                 context: str = None, variants: List[str] = None, source: str = None):
        """
        Add a term to the lexicon.
        
        Args:
            term: The toxic term
            category: Category of toxicity
            severity: Severity score (0.0 to 1.0)
            language: Language code
            context: Optional context information
            variants: Optional list of variants
            source: Optional source information
        """
        entry = LexiconEntry(
            term=term.lower(),
            category=category,
            severity=severity,
            language=language,
            context=context,
            variants=variants or [],
            source=source
        )
        
        self.lexicons[language][category].append(entry)
        self.term_to_entry[term.lower()] = entry
        
        # Add variants
        if variants:
            for variant in variants:
                variant_entry = LexiconEntry(
                    term=variant.lower(),
                    category=category,
                    severity=severity * 0.9,  # Slightly lower severity for variants
                    language=language,
                    context=f"variant_of_{term}",
                    source=source
                )
                self.term_to_entry[variant.lower()] = variant_entry
    
    def load_lexicon_from_file(self, filepath: str, language: str, category: str):
        """
        Load lexicon from a file.
        
        Args:
            filepath: Path to lexicon file
            language: Language code
            category: Category of terms in the file
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split('\t')
                        term = parts[0]
                        severity = float(parts[1]) if len(parts) > 1 else 0.7
                        context = parts[2] if len(parts) > 2 else None
                        
                        self.add_term(term, category, severity, language, context=context, source=filepath)
            
            logger.info(f"Loaded lexicon from {filepath} for {language}/{category}")
        except Exception as e:
            logger.error(f"Error loading lexicon from {filepath}: {e}")
    
    def get_term_info(self, term: str) -> Optional[LexiconEntry]:
        """
        Get information about a specific term.
        
        Args:
            term: Term to look up
            
        Returns:
            LexiconEntry if found, None otherwise
        """
        return self.term_to_entry.get(term.lower())
    
    def get_category_terms(self, category: str, language: str = None) -> List[LexiconEntry]:
        """
        Get all terms in a specific category.
        
        Args:
            category: Category to retrieve
            language: Optional language filter
            
        Returns:
            List of lexicon entries
        """
        terms = []
        
        if language:
            terms.extend(self.lexicons[language].get(category, []))
        else:
            for lang_lexicons in self.lexicons.values():
                terms.extend(lang_lexicons.get(category, []))
        
        return terms
    
    def get_language_terms(self, language: str) -> Dict[str, List[LexiconEntry]]:
        """
        Get all terms for a specific language.
        
        Args:
            language: Language code
            
        Returns:
            Dictionary of {category: [entries]}
        """
        return dict(self.lexicons[language])
    
    def search_terms(self, query: str, fuzzy: bool = False) -> List[LexiconEntry]:
        """
        Search for terms matching a query.
        
        Args:
            query: Search query
            fuzzy: Whether to perform fuzzy matching
            
        Returns:
            List of matching entries
        """
        query = query.lower()
        matches = []
        
        for term, entry in self.term_to_entry.items():
            if fuzzy:
                if query in term or term in query:
                    matches.append(entry)
            else:
                if term == query:
                    matches.append(entry)
        
        return matches
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get lexicon statistics.
        
        Returns:
            Statistics dictionary
        """
        stats = {
            'total_terms': len(self.term_to_entry),
            'languages': list(self.lexicons.keys()),
            'categories': list(self.category_weights.keys()),
            'by_language': {},
            'by_category': defaultdict(int)
        }
        
        # Count by language
        for language, categories in self.lexicons.items():
            lang_count = sum(len(entries) for entries in categories.values())
            stats['by_language'][language] = lang_count
        
        # Count by category
        for entry in self.term_to_entry.values():
            stats['by_category'][entry.category] += 1
        
        stats['by_category'] = dict(stats['by_category'])
        
        return stats

class LexiconFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts lexicon-based features from text.
    """
    
    def __init__(self, lexicon: MultilingualLexicon, 
                 feature_types: List[str] = None,
                 normalize_scores: bool = True,
                 include_context: bool = True):
        self.lexicon = lexicon
        self.feature_types = feature_types or ['count', 'severity', 'category_count', 'max_severity']
        self.normalize_scores = normalize_scores
        self.include_context = include_context
        self.feature_names_ = []
        self.scaler = MinMaxScaler() if normalize_scores else None
    
    def fit(self, X, y=None):
        """
        Fit the feature extractor.
        
        Args:
            X: Training texts
            y: Training labels (unused)
            
        Returns:
            self
        """
        # Generate feature names
        self.feature_names_ = self._generate_feature_names()
        
        # If normalization is enabled, fit scaler
        if self.scaler:
            features = self.transform(X)
            self.scaler.fit(features)
        
        return self
    
    def transform(self, X):
        """
        Transform texts to lexicon-based features.
        
        Args:
            X: Input texts
            
        Returns:
            Feature matrix
        """
        features = []
        
        for text in X:
            text_features = self._extract_text_features(text)
            features.append(text_features)
        
        features = np.array(features)
        
        # Apply normalization if enabled
        if self.scaler and hasattr(self.scaler, 'transform'):
            features = self.scaler.transform(features)
        
        return features
    
    def _extract_text_features(self, text: str) -> List[float]:
        """
        Extract lexicon features from a single text.
        
        Args:
            text: Input text
            
        Returns:
            List of feature values
        """
        text_lower = text.lower()
        tokens = self._tokenize_text(text_lower)
        
        features = []
        
        # Basic lexicon matching
        matches = []
        for token in tokens:
            entry = self.lexicon.get_term_info(token)
            if entry:
                matches.append(entry)
        
        # Multi-word phrase matching
        for phrase_len in [2, 3, 4]:
            for i in range(len(tokens) - phrase_len + 1):
                phrase = ' '.join(tokens[i:i + phrase_len])
                entry = self.lexicon.get_term_info(phrase)
                if entry:
                    matches.append(entry)
        
        # Extract different types of features
        if 'count' in self.feature_types:
            features.append(len(matches))  # Total match count
        
        if 'severity' in self.feature_types:
            total_severity = sum(match.severity for match in matches)
            features.append(total_severity)  # Total severity score
        
        if 'max_severity' in self.feature_types:
            max_severity = max([match.severity for match in matches], default=0.0)
            features.append(max_severity)  # Maximum severity
        
        if 'avg_severity' in self.feature_types:
            avg_severity = np.mean([match.severity for match in matches]) if matches else 0.0
            features.append(avg_severity)  # Average severity
        
        if 'category_count' in self.feature_types:
            # Count matches by category
            category_counts = defaultdict(int)
            for match in matches:
                category_counts[match.category] += 1
            
            for category in self.lexicon.category_weights.keys():
                features.append(category_counts[category])
        
        if 'weighted_severity' in self.feature_types:
            # Weighted severity by category importance
            weighted_severity = sum(
                match.severity * self.lexicon.category_weights.get(match.category, 1.0)
                for match in matches
            )
            features.append(weighted_severity)
        
        if 'density' in self.feature_types:
            # Lexicon match density (matches per token)
            density = len(matches) / len(tokens) if tokens else 0.0
            features.append(density)
        
        if 'language_diversity' in self.feature_types:
            # Number of different languages in matches
            languages = set(match.language for match in matches)
            features.append(len(languages))
        
        if self.include_context:
            # Context-specific features
            leetspeak_matches = sum(1 for match in matches if match.context == 'leetspeak')
            features.append(leetspeak_matches)
            
            variant_matches = sum(1 for match in matches if match.context and 'variant' in match.context)
            features.append(variant_matches)
        
        return features
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text for lexicon matching.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Basic tokenization
        tokens = word_tokenize(text)
        
        # Remove punctuation and convert to lowercase
        tokens = [token.lower() for token in tokens if token not in string.punctuation]
        
        # Handle special characters and numbers in leetspeak
        processed_tokens = []
        for token in tokens:
            processed_tokens.append(token)
            
            # Add leetspeak normalization
            normalized = self._normalize_leetspeak(token)
            if normalized != token:
                processed_tokens.append(normalized)
        
        return processed_tokens
    
    def _normalize_leetspeak(self, token: str) -> str:
        """
        Normalize leetspeak to standard text.
        
        Args:
            token: Input token
            
        Returns:
            Normalized token
        """
        # Common leetspeak substitutions
        substitutions = {
            '0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's',
            '7': 't', '8': 'b', '@': 'a', '$': 's', '+': 't'
        }
        
        normalized = token
        for leet, normal in substitutions.items():
            normalized = normalized.replace(leet, normal)
        
        return normalized
    
    def _generate_feature_names(self) -> List[str]:
        """
        Generate feature names for the extracted features.
        
        Returns:
            List of feature names
        """
        names = []
        
        if 'count' in self.feature_types:
            names.append('lexicon_match_count')
        
        if 'severity' in self.feature_types:
            names.append('lexicon_total_severity')
        
        if 'max_severity' in self.feature_types:
            names.append('lexicon_max_severity')
        
        if 'avg_severity' in self.feature_types:
            names.append('lexicon_avg_severity')
        
        if 'category_count' in self.feature_types:
            for category in self.lexicon.category_weights.keys():
                names.append(f'lexicon_{category}_count')
        
        if 'weighted_severity' in self.feature_types:
            names.append('lexicon_weighted_severity')
        
        if 'density' in self.feature_types:
            names.append('lexicon_match_density')
        
        if 'language_diversity' in self.feature_types:
            names.append('lexicon_language_diversity')
        
        if self.include_context:
            names.extend(['lexicon_leetspeak_matches', 'lexicon_variant_matches'])
        
        return names
    
    def get_feature_names_out(self, input_features=None):
        """
        Get feature names for output features.
        
        Returns:
            Array of feature names
        """
        return np.array(self.feature_names_)

class HybridFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Combines statistical features (TF-IDF, BoW) with lexicon-based features.
    """
    
    def __init__(self, 
                 lexicon: MultilingualLexicon,
                 tfidf_params: Dict[str, Any] = None,
                 bow_params: Dict[str, Any] = None,
                 lexicon_params: Dict[str, Any] = None,
                 feature_weights: Dict[str, float] = None,
                 include_topic_features: bool = True):
        self.lexicon = lexicon
        
        # Initialize feature extractors
        self.tfidf_params = tfidf_params or {'max_features': 5000, 'ngram_range': (1, 3)}
        self.bow_params = bow_params or {'max_features': 2000, 'ngram_range': (1, 2)}
        self.lexicon_params = lexicon_params or {}
        
        self.tfidf_vectorizer = TfidfVectorizer(**self.tfidf_params)
        self.bow_vectorizer = CountVectorizer(**self.bow_params)
        self.lexicon_extractor = LexiconFeatureExtractor(lexicon, **self.lexicon_params)
        
        # Feature weighting
        self.feature_weights = feature_weights or {
            'tfidf': 0.5,
            'bow': 0.2,
            'lexicon': 0.3
        }
        
        self.include_topic_features = include_topic_features
        self.topic_extractor = None
        
        # Fitted flag
        self.is_fitted = False
    
    def fit(self, X, y=None):
        """
        Fit all feature extractors.
        
        Args:
            X: Training texts
            y: Training labels (unused)
            
        Returns:
            self
        """
        logger.info("Fitting hybrid feature extractor...")
        
        # Fit individual extractors
        self.tfidf_vectorizer.fit(X)
        self.bow_vectorizer.fit(X)
        self.lexicon_extractor.fit(X)
        
        # Fit topic extractor if enabled
        if self.include_topic_features:
            self.topic_extractor = self._create_topic_extractor()
            self.topic_extractor.fit(X)
        
        self.is_fitted = True
        logger.info("Hybrid feature extractor fitted successfully")
        
        return self
    
    def transform(self, X):
        """
        Transform texts to hybrid features.
        
        Args:
            X: Input texts
            
        Returns:
            Combined feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted before transform")
        
        # Extract individual feature types
        tfidf_features = self.tfidf_vectorizer.transform(X)
        bow_features = self.bow_vectorizer.transform(X)
        lexicon_features = self.lexicon_extractor.transform(X)
        
        # Apply feature weights
        tfidf_features = tfidf_features * self.feature_weights['tfidf']
        bow_features = bow_features * self.feature_weights['bow']
        lexicon_features = lexicon_features * self.feature_weights['lexicon']
        
        # Combine features
        feature_matrices = [tfidf_features, bow_features]
        
        # Add lexicon features (convert to sparse if needed)
        if sp.issparse(tfidf_features):
            lexicon_sparse = sp.csr_matrix(lexicon_features)
            feature_matrices.append(lexicon_sparse)
        else:
            feature_matrices.append(lexicon_features)
        
        # Add topic features if enabled
        if self.include_topic_features and self.topic_extractor:
            topic_features = self.topic_extractor.transform(X)
            if sp.issparse(tfidf_features):
                topic_sparse = sp.csr_matrix(topic_features)
                feature_matrices.append(topic_sparse)
            else:
                feature_matrices.append(topic_features)
        
        # Combine all features
        if sp.issparse(feature_matrices[0]):
            combined_features = sp.hstack(feature_matrices)
        else:
            combined_features = np.hstack(feature_matrices)
        
        return combined_features
    
    def _create_topic_extractor(self):
        """
        Create a simple topic-based feature extractor.
        
        Returns:
            Topic feature extractor
        """
        # Simple topic keywords for content moderation
        topic_keywords = {
            'violence': ['kill', 'murder', 'death', 'violence', 'attack', 'harm', 'hurt'],
            'harassment': ['bully', 'harass', 'stalk', 'threaten', 'intimidate'],
            'discrimination': ['racist', 'sexist', 'homophobic', 'discriminate', 'prejudice'],
            'sexual': ['sexual', 'rape', 'assault', 'inappropriate', 'explicit'],
            'substance': ['drug', 'alcohol', 'smoking', 'addiction', 'substance'],
            'spam': ['spam', 'advertisement', 'promotion', 'sale', 'buy', 'cheap']
        }
        
        return TopicFeatureExtractor(topic_keywords)
    
    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get information about the extracted features.
        
        Returns:
            Feature information dictionary
        """
        if not self.is_fitted:
            return {'status': 'not_fitted'}
        
        info = {
            'tfidf_features': self.tfidf_vectorizer.get_feature_names_out().shape[0],
            'bow_features': self.bow_vectorizer.get_feature_names_out().shape[0],
            'lexicon_features': len(self.lexicon_extractor.get_feature_names_out()),
            'feature_weights': self.feature_weights,
            'lexicon_stats': self.lexicon.get_statistics()
        }
        
        if self.include_topic_features and self.topic_extractor:
            info['topic_features'] = len(self.topic_extractor.get_feature_names_out())
        
        total_features = sum([
            info['tfidf_features'],
            info['bow_features'],
            info['lexicon_features']
        ])
        
        if 'topic_features' in info:
            total_features += info['topic_features']
        
        info['total_features'] = total_features
        
        return info
    
    def explain_features(self, text: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Explain the most important features for a given text.
        
        Args:
            text: Input text to explain
            top_k: Number of top features to return
            
        Returns:
            Feature explanation dictionary
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted before explanation")
        
        explanation = {
            'text': text,
            'lexicon_matches': [],
            'top_tfidf_terms': [],
            'top_bow_terms': [],
            'feature_summary': {}
        }
        
        # Lexicon matches
        tokens = word_tokenize(text.lower())
        for token in tokens:
            entry = self.lexicon.get_term_info(token)
            if entry:
                explanation['lexicon_matches'].append({
                    'term': entry.term,
                    'category': entry.category,
                    'severity': entry.severity,
                    'language': entry.language
                })
        
        # Top TF-IDF terms
        tfidf_features = self.tfidf_vectorizer.transform([text])
        tfidf_feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        if sp.issparse(tfidf_features):
            tfidf_scores = tfidf_features.toarray()[0]
        else:
            tfidf_scores = tfidf_features[0]
        
        top_tfidf_indices = np.argsort(tfidf_scores)[-top_k:][::-1]
        for idx in top_tfidf_indices:
            if tfidf_scores[idx] > 0:
                explanation['top_tfidf_terms'].append({
                    'term': tfidf_feature_names[idx],
                    'score': float(tfidf_scores[idx])
                })
        
        # Top BoW terms
        bow_features = self.bow_vectorizer.transform([text])
        bow_feature_names = self.bow_vectorizer.get_feature_names_out()
        
        if sp.issparse(bow_features):
            bow_scores = bow_features.toarray()[0]
        else:
            bow_scores = bow_features[0]
        
        top_bow_indices = np.argsort(bow_scores)[-top_k:][::-1]
        for idx in top_bow_indices:
            if bow_scores[idx] > 0:
                explanation['top_bow_terms'].append({
                    'term': bow_feature_names[idx],
                    'count': int(bow_scores[idx])
                })
        
        # Feature summary
        lexicon_features = self.lexicon_extractor.transform([text])[0]
        explanation['feature_summary'] = {
            'lexicon_match_count': int(lexicon_features[0]) if len(lexicon_features) > 0 else 0,
            'total_lexicon_severity': float(lexicon_features[1]) if len(lexicon_features) > 1 else 0.0,
            'tfidf_feature_count': np.sum(tfidf_scores > 0),
            'bow_feature_count': np.sum(bow_scores > 0)
        }
        
        return explanation

class TopicFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Simple topic-based feature extractor using keyword matching.
    """
    
    def __init__(self, topic_keywords: Dict[str, List[str]]):
        self.topic_keywords = topic_keywords
        self.feature_names_ = list(topic_keywords.keys())
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        
        for text in X:
            text_lower = text.lower()
            topic_scores = []
            
            for topic, keywords in self.topic_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                topic_scores.append(score)
            
            features.append(topic_scores)
        
        return np.array(features)
    
    def get_feature_names_out(self, input_features=None):
        return np.array([f'topic_{name}' for name in self.feature_names_])

class KnowledgeEnhancedClassifier:
    """
    A classifier that uses knowledge-enhanced features for content moderation.
    """
    
    def __init__(self, 
                 lexicon: MultilingualLexicon = None,
                 classifier_type: str = 'logistic_regression',
                 feature_extractor_params: Dict[str, Any] = None):
        self.lexicon = lexicon or MultilingualLexicon()
        self.classifier_type = classifier_type
        self.feature_extractor_params = feature_extractor_params or {}
        
        # Initialize components
        self.feature_extractor = HybridFeatureExtractor(
            lexicon=self.lexicon,
            **self.feature_extractor_params
        )
        
        self.classifier = self._create_classifier()
        self.is_fitted = False
    
    def _create_classifier(self):
        """
        Create the underlying classifier.
        
        Returns:
            Classifier instance
        """
        if self.classifier_type == 'logistic_regression':
            return LogisticRegression(random_state=42, max_iter=1000)
        elif self.classifier_type == 'random_forest':
            return RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unsupported classifier type: {self.classifier_type}")
    
    def fit(self, X: List[str], y: List[int]):
        """
        Fit the knowledge-enhanced classifier.
        
        Args:
            X: Training texts
            y: Training labels
        """
        logger.info("Training knowledge-enhanced classifier...")
        
        # Extract features
        self.feature_extractor.fit(X)
        X_features = self.feature_extractor.transform(X)
        
        # Train classifier
        self.classifier.fit(X_features, y)
        
        self.is_fitted = True
        logger.info("Knowledge-enhanced classifier trained successfully")
    
    def predict(self, X: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on texts.
        
        Args:
            X: Input texts
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")
        
        # Extract features
        X_features = self.feature_extractor.transform(X)
        
        # Make predictions
        predictions = self.classifier.predict(X_features)
        probabilities = self.classifier.predict_proba(X_features)
        
        return predictions, probabilities
    
    def explain_prediction(self, text: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Explain a prediction with feature importance.
        
        Args:
            text: Input text
            top_k: Number of top features to show
            
        Returns:
            Explanation dictionary
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before explanation")
        
        # Get prediction
        predictions, probabilities = self.predict([text])
        prediction = predictions[0]
        confidence = probabilities[0][1] if prediction == 1 else probabilities[0][0]
        
        # Get feature explanation
        feature_explanation = self.feature_extractor.explain_features(text, top_k)
        
        # Get feature importance (if available)
        feature_importance = None
        if hasattr(self.classifier, 'coef_'):
            # For linear models
            feature_importance = self.classifier.coef_[0]
        elif hasattr(self.classifier, 'feature_importances_'):
            # For tree-based models
            feature_importance = self.classifier.feature_importances_
        
        explanation = {
            'text': text,
            'prediction': int(prediction),
            'confidence': float(confidence),
            'feature_explanation': feature_explanation,
            'model_info': {
                'classifier_type': self.classifier_type,
                'feature_info': self.feature_extractor.get_feature_info()
            }
        }
        
        if feature_importance is not None:
            explanation['feature_importance_stats'] = {
                'mean_importance': float(np.mean(np.abs(feature_importance))),
                'max_importance': float(np.max(np.abs(feature_importance))),
                'min_importance': float(np.min(np.abs(feature_importance)))
            }
        
        return explanation
    
    def evaluate(self, X_test: List[str], y_test: List[int]) -> Dict[str, float]:
        """
        Evaluate the classifier on test data.
        
        Args:
            X_test: Test texts
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before evaluation")
        
        # Make predictions
        predictions, probabilities = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='binary')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def cross_validate(self, X: List[str], y: List[int], cv: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            X: Training texts
            y: Training labels
            cv: Number of cross-validation folds
            
        Returns:
            Cross-validation results
        """
        # Extract features
        self.feature_extractor.fit(X)
        X_features = self.feature_extractor.transform(X)
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.classifier, X_features, y, cv=cv, scoring='f1')
        
        return {
            'cv_scores': cv_scores.tolist(),
            'mean_score': float(np.mean(cv_scores)),
            'std_score': float(np.std(cv_scores)),
            'cv_folds': cv
        }

# Example usage and testing functions
def demo_knowledge_enhanced_ml():
    """
    Demonstrate the knowledge-enhanced ML system.
    """
    print("Knowledge-Enhanced ML for Content Moderation Demo")
    print("=" * 55)
    
    # Initialize lexicon
    lexicon = MultilingualLexicon()
    
    print("\nLexicon Statistics:")
    stats = lexicon.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test lexicon lookup
    print("\nLexicon Lookup Examples:")
    test_terms = ['idiot', 'বোকা', 'f4gg0t', 'kill yourself']
    for term in test_terms:
        entry = lexicon.get_term_info(term)
        if entry:
            print(f"  '{term}': {entry.category} (severity: {entry.severity}, lang: {entry.language})")
        else:
            print(f"  '{term}': Not found")
    
    # Test feature extraction
    print("\nFeature Extraction Demo:")
    sample_texts = [
        "You are such an idiot!",
        "This is a normal message.",
        "1d10t n00b go k1ll yourself",
        "তুমি একটা বোকা"
    ]
    
    # Initialize feature extractor
    feature_extractor = HybridFeatureExtractor(lexicon)
    
    print("\nFitting feature extractor...")
    feature_extractor.fit(sample_texts)
    
    print("\nFeature extraction results:")
    features = feature_extractor.transform(sample_texts)
    feature_info = feature_extractor.get_feature_info()
    
    print(f"Feature matrix shape: {features.shape}")
    print(f"Total features: {feature_info['total_features']}")
    print(f"  - TF-IDF: {feature_info['tfidf_features']}")
    print(f"  - BoW: {feature_info['bow_features']}")
    print(f"  - Lexicon: {feature_info['lexicon_features']}")
    if 'topic_features' in feature_info:
        print(f"  - Topic: {feature_info['topic_features']}")
    
    # Test feature explanation
    print("\nFeature Explanation Example:")
    explanation = feature_extractor.explain_features(sample_texts[0])
    print(f"Text: '{explanation['text']}'")
    print(f"Lexicon matches: {len(explanation['lexicon_matches'])}")
    for match in explanation['lexicon_matches']:
        print(f"  - {match['term']} ({match['category']}, severity: {match['severity']})")
    
    print("\nSystem components:")
    print("- MultilingualLexicon: Manages toxic terms across languages")
    print("- LexiconFeatureExtractor: Extracts lexicon-based features")
    print("- HybridFeatureExtractor: Combines TF-IDF, BoW, and lexicon features")
    print("- KnowledgeEnhancedClassifier: Complete classification pipeline")
    
    print("\nKey features:")
    print("- Multilingual lexicon support (English, Bengali, Hindi)")
    print("- Leetspeak and variant detection")
    print("- Category-based severity weighting")
    print("- Hybrid statistical + knowledge features")
    print("- Explainable predictions with feature importance")
    print("- Topic-based feature extraction")

if __name__ == "__main__":
    demo_knowledge_enhanced_ml()