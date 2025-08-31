import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from scipy.sparse import csr_matrix, hstack
import warnings
warnings.filterwarnings('ignore')

class TopicModelingFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Topic modeling feature extractor using LDA and NMF to discover latent themes
    in text data and use them as features for rule violation detection.
    """
    
    def __init__(self, 
                 n_topics_lda: int = 10,
                 n_topics_nmf: int = 8,
                 max_features: int = 1000,
                 min_df: int = 2,
                 max_df: float = 0.95,
                 ngram_range: Tuple[int, int] = (1, 2),
                 random_state: int = 42):
        """
        Initialize the topic modeling feature extractor.
        
        Args:
            n_topics_lda: Number of topics for LDA
            n_topics_nmf: Number of topics for NMF
            max_features: Maximum features for vectorization
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            ngram_range: N-gram range for feature extraction
            random_state: Random state for reproducibility
        """
        self.n_topics_lda = n_topics_lda
        self.n_topics_nmf = n_topics_nmf
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.random_state = random_state
        
        # Initialize models
        self.lda_vectorizer = None
        self.nmf_vectorizer = None
        self.lda_model = None
        self.nmf_model = None
        
        # Topic interpretations (will be populated after fitting)
        self.lda_topic_words = {}
        self.nmf_topic_words = {}
        self.topic_interpretations = {}
    
    def fit(self, X: List[str], y=None):
        """
        Fit the topic models on the training data.
        
        Args:
            X: List of text documents
            y: Target labels (unused)
            
        Returns:
            self
        """
        # Initialize vectorizers
        self.lda_vectorizer = CountVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            stop_words='english',
            lowercase=True
        )
        
        self.nmf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            stop_words='english',
            lowercase=True
        )
        
        # Fit vectorizers and transform data
        lda_features = self.lda_vectorizer.fit_transform(X)
        nmf_features = self.nmf_vectorizer.fit_transform(X)
        
        # Initialize and fit topic models
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics_lda,
            random_state=self.random_state,
            max_iter=10,
            learning_method='batch'
        )
        
        self.nmf_model = NMF(
            n_components=self.n_topics_nmf,
            random_state=self.random_state,
            max_iter=200,
            alpha_W=0.1,
            alpha_H=0.1
        )
        
        # Fit models
        self.lda_model.fit(lda_features)
        self.nmf_model.fit(nmf_features)
        
        # Extract topic words for interpretability
        self._extract_topic_words()
        self._interpret_topics()
        
        return self
    
    def transform(self, X: List[str]) -> csr_matrix:
        """
        Transform text documents into topic-based feature vectors.
        
        Args:
            X: List of text documents
            
        Returns:
            Sparse matrix of topic features
        """
        # Transform using vectorizers
        lda_features = self.lda_vectorizer.transform(X)
        nmf_features = self.nmf_vectorizer.transform(X)
        
        # Get topic distributions
        lda_topics = self.lda_model.transform(lda_features)
        nmf_topics = self.nmf_model.transform(nmf_features)
        
        # Combine topic features
        combined_topics = np.hstack([lda_topics, nmf_topics])
        
        # Add topic-based statistical features
        topic_stats = self._extract_topic_statistics(lda_topics, nmf_topics)
        
        # Combine all features
        all_features = np.hstack([combined_topics, topic_stats])
        
        return csr_matrix(all_features)
    
    def _extract_topic_words(self, n_words: int = 10):
        """
        Extract top words for each topic for interpretability.
        
        Args:
            n_words: Number of top words to extract per topic
        """
        # LDA topic words
        lda_feature_names = self.lda_vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words_idx = topic.argsort()[-n_words:][::-1]
            top_words = [lda_feature_names[i] for i in top_words_idx]
            self.lda_topic_words[f'lda_topic_{topic_idx}'] = top_words
        
        # NMF topic words
        nmf_feature_names = self.nmf_vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(self.nmf_model.components_):
            top_words_idx = topic.argsort()[-n_words:][::-1]
            top_words = [nmf_feature_names[i] for i in top_words_idx]
            self.nmf_topic_words[f'nmf_topic_{topic_idx}'] = top_words
    
    def _interpret_topics(self):
        """
        Interpret topics based on their top words and assign meaningful labels.
        """
        # Rule violation topic patterns
        violation_patterns = {
            'harassment': ['harass', 'bully', 'attack', 'abuse', 'threaten', 'intimidate'],
            'hate_speech': ['hate', 'racist', 'discrimination', 'bigot', 'supremacist'],
            'threats': ['kill', 'die', 'death', 'violence', 'harm', 'hurt'],
            'spam': ['buy', 'sell', 'promotion', 'advertisement', 'click', 'link'],
            'trolling': ['troll', 'bait', 'flame', 'provoke', 'annoy'],
            'off_topic': ['topic', 'relevant', 'belong', 'appropriate', 'related'],
            'personal_info': ['address', 'phone', 'email', 'personal', 'private'],
            'moderation': ['mod', 'admin', 'rule', 'ban', 'remove', 'delete'],
            'positive_community': ['help', 'support', 'thank', 'appreciate', 'good'],
            'discussion': ['think', 'opinion', 'believe', 'agree', 'disagree']
        }
        
        # Interpret LDA topics
        for topic_name, words in self.lda_topic_words.items():
            interpretation = self._match_topic_to_pattern(words, violation_patterns)
            self.topic_interpretations[topic_name] = interpretation
        
        # Interpret NMF topics
        for topic_name, words in self.nmf_topic_words.items():
            interpretation = self._match_topic_to_pattern(words, violation_patterns)
            self.topic_interpretations[topic_name] = interpretation
    
    def _match_topic_to_pattern(self, topic_words: List[str], patterns: Dict[str, List[str]]) -> str:
        """
        Match topic words to violation patterns to interpret the topic.
        
        Args:
            topic_words: List of top words in the topic
            patterns: Dictionary of violation patterns
            
        Returns:
            Interpreted topic label
        """
        topic_words_lower = [word.lower() for word in topic_words]
        
        best_match = 'general'
        best_score = 0
        
        for pattern_name, pattern_words in patterns.items():
            # Calculate overlap score
            overlap = len(set(topic_words_lower) & set(pattern_words))
            if overlap > best_score:
                best_score = overlap
                best_match = pattern_name
        
        return best_match if best_score > 0 else 'general'
    
    def _extract_topic_statistics(self, lda_topics: np.ndarray, nmf_topics: np.ndarray) -> np.ndarray:
        """
        Extract statistical features from topic distributions.
        
        Args:
            lda_topics: LDA topic distributions
            nmf_topics: NMF topic distributions
            
        Returns:
            Array of topic statistics
        """
        stats_features = []
        
        for i in range(len(lda_topics)):
            lda_dist = lda_topics[i]
            nmf_dist = nmf_topics[i]
            
            # LDA statistics
            lda_max = np.max(lda_dist)
            lda_entropy = -np.sum(lda_dist * np.log(lda_dist + 1e-10))
            lda_dominant_topics = np.sum(lda_dist > 0.1)  # Topics with >10% probability
            
            # NMF statistics
            nmf_max = np.max(nmf_dist)
            nmf_entropy = -np.sum(nmf_dist * np.log(nmf_dist + 1e-10)) if np.sum(nmf_dist) > 0 else 0
            nmf_dominant_topics = np.sum(nmf_dist > 0.1)
            
            # Combined statistics
            topic_diversity = lda_entropy + nmf_entropy
            topic_focus = lda_max + nmf_max
            
            stats = [
                lda_max, lda_entropy, lda_dominant_topics,
                nmf_max, nmf_entropy, nmf_dominant_topics,
                topic_diversity, topic_focus
            ]
            
            stats_features.append(stats)
        
        return np.array(stats_features)
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of all topic-based features.
        
        Returns:
            List of feature names
        """
        feature_names = []
        
        # LDA topic features
        for i in range(self.n_topics_lda):
            interpretation = self.topic_interpretations.get(f'lda_topic_{i}', 'general')
            feature_names.append(f'lda_topic_{i}_{interpretation}')
        
        # NMF topic features
        for i in range(self.n_topics_nmf):
            interpretation = self.topic_interpretations.get(f'nmf_topic_{i}', 'general')
            feature_names.append(f'nmf_topic_{i}_{interpretation}')
        
        # Statistical features
        stat_names = [
            'lda_max_prob', 'lda_entropy', 'lda_dominant_topics',
            'nmf_max_prob', 'nmf_entropy', 'nmf_dominant_topics',
            'topic_diversity', 'topic_focus'
        ]
        feature_names.extend(stat_names)
        
        return feature_names
    
    def get_topic_analysis(self) -> Dict[str, Any]:
        """
        Get detailed analysis of discovered topics.
        
        Returns:
            Dictionary with topic analysis
        """
        analysis = {
            'lda_topics': {},
            'nmf_topics': {},
            'interpretations': self.topic_interpretations,
            'violation_topics': [],
            'safe_topics': []
        }
        
        # Analyze LDA topics
        for topic_name, words in self.lda_topic_words.items():
            interpretation = self.topic_interpretations.get(topic_name, 'general')
            analysis['lda_topics'][topic_name] = {
                'words': words,
                'interpretation': interpretation
            }
            
            if interpretation in ['harassment', 'hate_speech', 'threats', 'trolling']:
                analysis['violation_topics'].append(topic_name)
            elif interpretation in ['positive_community', 'discussion']:
                analysis['safe_topics'].append(topic_name)
        
        # Analyze NMF topics
        for topic_name, words in self.nmf_topic_words.items():
            interpretation = self.topic_interpretations.get(topic_name, 'general')
            analysis['nmf_topics'][topic_name] = {
                'words': words,
                'interpretation': interpretation
            }
            
            if interpretation in ['harassment', 'hate_speech', 'threats', 'trolling']:
                analysis['violation_topics'].append(topic_name)
            elif interpretation in ['positive_community', 'discussion']:
                analysis['safe_topics'].append(topic_name)
        
        return analysis
    
    def analyze_document_topics(self, text: str) -> Dict[str, Any]:
        """
        Analyze topic distribution for a single document.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with document topic analysis
        """
        # Transform the document
        lda_features = self.lda_vectorizer.transform([text])
        nmf_features = self.nmf_vectorizer.transform([text])
        
        # Get topic distributions
        lda_topics = self.lda_model.transform(lda_features)[0]
        nmf_topics = self.nmf_model.transform(nmf_features)[0]
        
        analysis = {
            'text': text,
            'lda_distribution': {},
            'nmf_distribution': {},
            'dominant_topics': [],
            'violation_risk': 0.0
        }
        
        # Analyze LDA distribution
        for i, prob in enumerate(lda_topics):
            topic_name = f'lda_topic_{i}'
            interpretation = self.topic_interpretations.get(topic_name, 'general')
            analysis['lda_distribution'][f'{topic_name}_{interpretation}'] = prob
            
            if prob > 0.2:  # Dominant topic threshold
                analysis['dominant_topics'].append((topic_name, interpretation, prob))
        
        # Analyze NMF distribution
        for i, prob in enumerate(nmf_topics):
            topic_name = f'nmf_topic_{i}'
            interpretation = self.topic_interpretations.get(topic_name, 'general')
            analysis['nmf_distribution'][f'{topic_name}_{interpretation}'] = prob
            
            if prob > 0.2:  # Dominant topic threshold
                analysis['dominant_topics'].append((topic_name, interpretation, prob))
        
        # Calculate violation risk based on topic distributions
        violation_topics = ['harassment', 'hate_speech', 'threats', 'trolling', 'spam']
        violation_risk = 0.0
        
        for topic_name, interpretation, prob in analysis['dominant_topics']:
            if interpretation in violation_topics:
                violation_risk += prob
        
        analysis['violation_risk'] = min(violation_risk, 1.0)
        
        return analysis
    
    def fit_transform(self, X: List[str], y=None) -> csr_matrix:
        """
        Fit the topic models and transform the data.
        
        Args:
            X: List of text documents
            y: Target labels (unused)
            
        Returns:
            Transformed feature matrix
        """
        return self.fit(X, y).transform(X)