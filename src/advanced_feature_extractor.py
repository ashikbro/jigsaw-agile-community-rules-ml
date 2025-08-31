import numpy as np
import pandas as pd
import re
from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import hstack, csr_matrix
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class AdvancedHybridFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Advanced hybrid feature extractor combining multiple novel approaches:
    1. Word-level and character-level TF-IDF
    2. Statistical features (length, uppercase, links, etc.)
    3. Linguistic features (POS tags)
    4. Sentiment analysis scores
    5. Rule-specific keywords and patterns
    """
    
    def __init__(self, 
                 word_max_features: int = 5000,
                 char_max_features: int = 3000,
                 word_ngram_range: Tuple[int, int] = (1, 2),
                 char_ngram_range: Tuple[int, int] = (2, 4),
                 min_df: int = 2,
                 max_df: float = 0.95):
        """
        Initialize the advanced hybrid feature extractor.
        
        Args:
            word_max_features: Maximum features for word-level TF-IDF
            char_max_features: Maximum features for character-level TF-IDF
            word_ngram_range: N-gram range for word-level features
            char_ngram_range: N-gram range for character-level features
            min_df: Minimum document frequency
            max_df: Maximum document frequency
        """
        self.word_max_features = word_max_features
        self.char_max_features = char_max_features
        self.word_ngram_range = word_ngram_range
        self.char_ngram_range = char_ngram_range
        self.min_df = min_df
        self.max_df = max_df
        
        # Initialize vectorizers
        self.word_vectorizer = None
        self.char_vectorizer = None
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Rule-specific keywords for community moderation
        self.rule_keywords = {
            'moderation': ['mod', 'mods', 'moderator', 'moderators', 'admin', 'admins'],
            'rules': ['rule', 'rules', 'guideline', 'guidelines', 'policy', 'policies'],
            'violations': ['ban', 'banned', 'kick', 'kicked', 'remove', 'removed', 'delete', 'deleted'],
            'reporting': ['report', 'reported', 'flag', 'flagged', 'complaint'],
            'threats': ['kill', 'die', 'death', 'hurt', 'harm', 'violence', 'attack'],
            'harassment': ['harass', 'harassment', 'bully', 'bullying', 'stalk', 'stalking'],
            'hate_speech': ['hate', 'racist', 'sexist', 'homophobic', 'transphobic', 'bigot'],
            'spam': ['spam', 'advertisement', 'promotion', 'selling', 'buy now']
        }
        
        # Profanity patterns (basic list for demonstration)
        self.profanity_patterns = [
            r'\b(damn|hell|crap|stupid|idiot|moron|dumb)\b',
            r'\b(wtf|omg|lol|lmao|rofl)\b',
            r'[!@#$%^&*]{2,}',  # Multiple special characters
        ]
        
        # Politeness indicators
        self.politeness_positive = ['please', 'thank', 'thanks', 'sorry', 'excuse me', 'pardon']
        self.politeness_negative = ['shut up', 'whatever', 'don\'t care', 'stupid', 'dumb']
        
    def fit(self, X: List[str], y=None):
        """
        Fit the feature extractors on the training data.
        
        Args:
            X: List of text documents
            y: Target labels (unused)
            
        Returns:
            self
        """
        # Initialize and fit word-level TF-IDF
        self.word_vectorizer = TfidfVectorizer(
            max_features=self.word_max_features,
            ngram_range=self.word_ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            analyzer='word',
            lowercase=True,
            stop_words='english'
        )
        
        # Initialize and fit character-level TF-IDF
        self.char_vectorizer = TfidfVectorizer(
            max_features=self.char_max_features,
            ngram_range=self.char_ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            analyzer='char',
            lowercase=True
        )
        
        # Fit vectorizers
        self.word_vectorizer.fit(X)
        self.char_vectorizer.fit(X)
        
        return self
    
    def transform(self, X: List[str]) -> csr_matrix:
        """
        Transform text documents into hybrid feature vectors.
        
        Args:
            X: List of text documents
            
        Returns:
            Sparse matrix of combined features
        """
        # Extract different types of features
        word_features = self._extract_word_features(X)
        char_features = self._extract_char_features(X)
        statistical_features = self._extract_statistical_features(X)
        linguistic_features = self._extract_linguistic_features(X)
        sentiment_features = self._extract_sentiment_features(X)
        rule_features = self._extract_rule_specific_features(X)
        
        # Combine all features
        combined_features = hstack([
            word_features,
            char_features,
            statistical_features,
            linguistic_features,
            sentiment_features,
            rule_features
        ])
        
        return combined_features
    
    def _extract_word_features(self, X: List[str]) -> csr_matrix:
        """Extract word-level TF-IDF features."""
        return self.word_vectorizer.transform(X)
    
    def _extract_char_features(self, X: List[str]) -> csr_matrix:
        """Extract character-level TF-IDF features."""
        return self.char_vectorizer.transform(X)
    
    def _extract_statistical_features(self, X: List[str]) -> csr_matrix:
        """Extract statistical features from text."""
        features = []
        
        for text in X:
            text_features = []
            
            # Basic length features
            text_features.append(len(text))  # Character count
            text_features.append(len(text.split()))  # Word count
            text_features.append(len(text.split('\n')))  # Line count
            
            # Average word length
            words = text.split()
            avg_word_len = np.mean([len(word) for word in words]) if words else 0
            text_features.append(avg_word_len)
            
            # Uppercase features
            text_features.append(sum(1 for c in text if c.isupper()))  # Uppercase count
            text_features.append(sum(1 for c in text if c.isupper()) / len(text) if text else 0)  # Uppercase ratio
            
            # Special character features
            text_features.append(text.count('!'))  # Exclamation marks
            text_features.append(text.count('?'))  # Question marks
            text_features.append(len(re.findall(r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?]', text)))  # Special chars
            
            # URL and link detection
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            text_features.append(len(re.findall(url_pattern, text)))  # URL count
            
            # Email detection
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            text_features.append(len(re.findall(email_pattern, text)))  # Email count
            
            # Repetitive character patterns
            text_features.append(len(re.findall(r'(.)\1{2,}', text)))  # Repeated chars (3+)
            
            # Capitalized words
            text_features.append(sum(1 for word in words if word.isupper() and len(word) > 1))
            
            features.append(text_features)
        
        return csr_matrix(np.array(features))
    
    def _extract_linguistic_features(self, X: List[str]) -> csr_matrix:
        """Extract linguistic features using POS tagging."""
        features = []
        
        for text in X:
            try:
                # Tokenize and POS tag
                tokens = word_tokenize(text.lower())
                pos_tags = pos_tag(tokens)
                
                # Count different POS tags
                pos_counts = Counter([tag for _, tag in pos_tags])
                
                # Extract specific POS tag counts
                text_features = [
                    pos_counts.get('NN', 0) + pos_counts.get('NNS', 0) + pos_counts.get('NNP', 0) + pos_counts.get('NNPS', 0),  # Nouns
                    pos_counts.get('VB', 0) + pos_counts.get('VBD', 0) + pos_counts.get('VBG', 0) + pos_counts.get('VBN', 0) + pos_counts.get('VBP', 0) + pos_counts.get('VBZ', 0),  # Verbs
                    pos_counts.get('JJ', 0) + pos_counts.get('JJR', 0) + pos_counts.get('JJS', 0),  # Adjectives
                    pos_counts.get('RB', 0) + pos_counts.get('RBR', 0) + pos_counts.get('RBS', 0),  # Adverbs
                    pos_counts.get('PRP', 0) + pos_counts.get('PRP$', 0),  # Pronouns
                    pos_counts.get('DT', 0),  # Determiners
                    pos_counts.get('IN', 0),  # Prepositions
                    pos_counts.get('CC', 0),  # Conjunctions
                ]
                
                # Normalize by text length
                text_len = len(tokens) if tokens else 1
                text_features = [count / text_len for count in text_features]
                
            except Exception:
                # Fallback to zeros if POS tagging fails
                text_features = [0.0] * 8
            
            features.append(text_features)
        
        return csr_matrix(np.array(features))
    
    def _extract_sentiment_features(self, X: List[str]) -> csr_matrix:
        """Extract sentiment analysis features."""
        features = []
        
        for text in X:
            try:
                # Get sentiment scores
                scores = self.sentiment_analyzer.polarity_scores(text)
                
                text_features = [
                    scores['compound'],  # Overall sentiment
                    scores['pos'],       # Positive sentiment
                    scores['neu'],       # Neutral sentiment
                    scores['neg'],       # Negative sentiment
                ]
            except Exception:
                # Fallback to neutral sentiment
                text_features = [0.0, 0.0, 1.0, 0.0]
            
            features.append(text_features)
        
        return csr_matrix(np.array(features))
    
    def _extract_rule_specific_features(self, X: List[str]) -> csr_matrix:
        """Extract rule-specific and community moderation features."""
        features = []
        
        for text in X:
            text_lower = text.lower()
            text_features = []
            
            # Count rule-specific keywords
            for category, keywords in self.rule_keywords.items():
                count = sum(text_lower.count(keyword) for keyword in keywords)
                text_features.append(count)
            
            # Profanity detection
            profanity_count = 0
            for pattern in self.profanity_patterns:
                profanity_count += len(re.findall(pattern, text_lower))
            text_features.append(profanity_count)
            
            # Politeness scoring
            politeness_positive = sum(text_lower.count(word) for word in self.politeness_positive)
            politeness_negative = sum(text_lower.count(word) for word in self.politeness_negative)
            politeness_score = politeness_positive - politeness_negative
            text_features.append(politeness_score)
            
            # All caps words (shouting)
            words = text.split()
            caps_words = sum(1 for word in words if word.isupper() and len(word) > 2)
            text_features.append(caps_words)
            
            # Question vs statement ratio
            questions = text.count('?')
            statements = text.count('.') + text.count('!')
            question_ratio = questions / (questions + statements + 1)  # +1 to avoid division by zero
            text_features.append(question_ratio)
            
            features.append(text_features)
        
        return csr_matrix(np.array(features))
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features."""
        feature_names = []
        
        # Word TF-IDF features
        if self.word_vectorizer:
            word_features = [f"word_tfidf_{name}" for name in self.word_vectorizer.get_feature_names_out()]
            feature_names.extend(word_features)
        
        # Character TF-IDF features
        if self.char_vectorizer:
            char_features = [f"char_tfidf_{name}" for name in self.char_vectorizer.get_feature_names_out()]
            feature_names.extend(char_features)
        
        # Statistical features
        statistical_names = [
            'char_count', 'word_count', 'line_count', 'avg_word_length',
            'uppercase_count', 'uppercase_ratio', 'exclamation_count', 'question_count',
            'special_char_count', 'url_count', 'email_count', 'repeated_char_patterns',
            'capitalized_words'
        ]
        feature_names.extend(statistical_names)
        
        # Linguistic features
        linguistic_names = [
            'noun_ratio', 'verb_ratio', 'adjective_ratio', 'adverb_ratio',
            'pronoun_ratio', 'determiner_ratio', 'preposition_ratio', 'conjunction_ratio'
        ]
        feature_names.extend(linguistic_names)
        
        # Sentiment features
        sentiment_names = ['sentiment_compound', 'sentiment_positive', 'sentiment_neutral', 'sentiment_negative']
        feature_names.extend(sentiment_names)
        
        # Rule-specific features
        rule_names = list(self.rule_keywords.keys()) + [
            'profanity_count', 'politeness_score', 'caps_words', 'question_ratio'
        ]
        feature_names.extend(rule_names)
        
        return feature_names
    
    def fit_transform(self, X: List[str], y=None) -> csr_matrix:
        """Fit the extractor and transform the data."""
        return self.fit(X, y).transform(X)