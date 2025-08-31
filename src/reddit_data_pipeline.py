"""Reddit Comment Data Pipeline for Rule Violation Detection.

This module provides comprehensive data processing capabilities for Reddit comments,
including text preprocessing, feature extraction, and rule-based labeling for
the Kaggle competition on comment moderation.
"""

import pandas as pd
import numpy as np
import re
import string
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
from transformers import AutoTokenizer
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    logger.warning("Could not download NLTK data")

@dataclass
class RedditRule:
    """Represents a Reddit subreddit rule."""
    rule_id: str
    rule_text: str
    category: str
    severity: str
    keywords: List[str]
    patterns: List[str]

class RedditRuleEngine:
    """Engine for detecting potential rule violations in Reddit comments."""
    
    def __init__(self):
        self.rules = self._initialize_rules()
        self.profanity_words = self._load_profanity_list()
        
    def _initialize_rules(self) -> Dict[str, RedditRule]:
        """Initialize common Reddit rules based on typical subreddit guidelines."""
        rules = {
            'harassment': RedditRule(
                rule_id='harassment',
                rule_text='No harassment, bullying, or personal attacks',
                category='behavior',
                severity='high',
                keywords=['idiot', 'stupid', 'moron', 'loser', 'pathetic', 'worthless'],
                patterns=[r'you\s+are\s+(?:an?\s+)?(?:idiot|stupid|moron)', r'kill\s+yourself']
            ),
            'spam': RedditRule(
                rule_id='spam',
                rule_text='No spam or self-promotion',
                category='content',
                severity='medium',
                keywords=['buy now', 'click here', 'subscribe', 'follow me'],
                patterns=[r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                         r'www\.\w+\.\w+']
            ),
            'off_topic': RedditRule(
                rule_id='off_topic',
                rule_text='Stay on topic',
                category='relevance',
                severity='low',
                keywords=['unrelated', 'random', 'btw', 'anyway'],
                patterns=[r'^(?:btw|anyway|random)', r'completely\s+unrelated']
            ),
            'profanity': RedditRule(
                rule_id='profanity',
                rule_text='No excessive profanity or vulgar language',
                category='language',
                severity='medium',
                keywords=[],  # Will be populated from profanity list
                patterns=[]
            ),
            'misinformation': RedditRule(
                rule_id='misinformation',
                rule_text='No misinformation or conspiracy theories',
                category='content',
                severity='high',
                keywords=['fake news', 'conspiracy', 'hoax', 'lie', 'false'],
                patterns=[r'(?:this\s+is\s+)?(?:fake|false)\s+news', r'government\s+conspiracy']
            ),
            'low_effort': RedditRule(
                rule_id='low_effort',
                rule_text='No low-effort posts or comments',
                category='quality',
                severity='low',
                keywords=['lol', 'lmao', 'this', 'same', 'agree'],
                patterns=[r'^(?:lol|lmao|haha|this|same|agree)$', r'^.{1,5}$']
            )
        }
        return rules
    
    def _load_profanity_list(self) -> List[str]:
        """Load common profanity words."""
        # Basic profanity list - in production, use a comprehensive dataset
        return ['damn', 'hell', 'crap', 'shit', 'fuck', 'bitch', 'ass', 'bastard']
    
    def detect_violations(self, comment: str) -> Dict[str, float]:
        """Detect potential rule violations in a comment.
        
        Args:
            comment: The comment text to analyze
            
        Returns:
            Dictionary mapping rule IDs to violation probabilities (0-1)
        """
        violations = {}
        comment_lower = comment.lower()
        
        for rule_id, rule in self.rules.items():
            score = 0.0
            
            # Check keywords
            keyword_matches = sum(1 for keyword in rule.keywords if keyword in comment_lower)
            if rule.keywords:
                score += (keyword_matches / len(rule.keywords)) * 0.6
            
            # Check patterns
            pattern_matches = sum(1 for pattern in rule.patterns if re.search(pattern, comment_lower))
            if rule.patterns:
                score += (pattern_matches / len(rule.patterns)) * 0.8
            
            # Special handling for profanity
            if rule_id == 'profanity':
                profanity_count = sum(1 for word in self.profanity_words if word in comment_lower)
                score = min(profanity_count * 0.3, 1.0)
            
            # Normalize score
            violations[rule_id] = min(score, 1.0)
        
        return violations

class RedditTextPreprocessor:
    """Advanced text preprocessing for Reddit comments."""
    
    def __init__(self, use_spacy: bool = True):
        self.use_spacy = use_spacy
        self.lemmatizer = WordNetLemmatizer()
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
            
        if use_spacy:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except:
                logger.warning("SpaCy model not found, falling back to NLTK")
                self.use_spacy = False
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove Reddit-specific formatting
        text = re.sub(r'/u/\w+', '', text)  # Remove username mentions
        text = re.sub(r'/r/\w+', '', text)  # Remove subreddit mentions
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # Remove bold formatting
        text = re.sub(r'\*(.+?)\*', r'\1', text)  # Remove italic formatting
        
        # Remove excessive whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def preprocess(self, text: str, remove_stopwords: bool = True, 
                  lemmatize: bool = True) -> str:
        """Comprehensive text preprocessing."""
        # Clean text
        text = self.clean_text(text)
        
        if self.use_spacy:
            return self._preprocess_spacy(text, remove_stopwords, lemmatize)
        else:
            return self._preprocess_nltk(text, remove_stopwords, lemmatize)
    
    def _preprocess_spacy(self, text: str, remove_stopwords: bool, 
                         lemmatize: bool) -> str:
        """Preprocess using SpaCy."""
        doc = self.nlp(text)
        tokens = []
        
        for token in doc:
            # Skip punctuation and whitespace
            if token.is_punct or token.is_space:
                continue
            
            # Skip stop words if requested
            if remove_stopwords and token.is_stop:
                continue
            
            # Get lemma or original text
            word = token.lemma_ if lemmatize else token.text
            word = word.lower()
            
            # Skip very short words
            if len(word) > 1:
                tokens.append(word)
        
        return ' '.join(tokens)
    
    def _preprocess_nltk(self, text: str, remove_stopwords: bool, 
                        lemmatize: bool) -> str:
        """Preprocess using NLTK."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        # Remove stop words
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatize
        if lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Filter short words
        tokens = [token for token in tokens if len(token) > 1]
        
        return ' '.join(tokens)

class RedditDataPipeline:
    """Complete data pipeline for Reddit comment processing."""
    
    def __init__(self, model_name: str = 'bert-base-uncased'):
        self.preprocessor = RedditTextPreprocessor()
        self.rule_engine = RedditRuleEngine()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tfidf_vectorizer = None
        self.scaler = StandardScaler()
        
    def process_comments(self, df: pd.DataFrame, 
                        text_column: str = 'comment_text',
                        target_column: str = 'rule_violation') -> pd.DataFrame:
        """Process a DataFrame of Reddit comments.
        
        Args:
            df: DataFrame containing comments
            text_column: Name of the text column
            target_column: Name of the target column (if available)
            
        Returns:
            Processed DataFrame with features
        """
        processed_df = df.copy()
        
        # Text preprocessing
        logger.info("Preprocessing text...")
        processed_df['cleaned_text'] = processed_df[text_column].apply(
            self.preprocessor.preprocess
        )
        
        # Rule-based features
        logger.info("Extracting rule-based features...")
        rule_features = processed_df[text_column].apply(
            self.rule_engine.detect_violations
        )
        
        # Convert rule features to columns
        for rule_id in self.rule_engine.rules.keys():
            processed_df[f'rule_{rule_id}_score'] = rule_features.apply(
                lambda x: x.get(rule_id, 0.0)
            )
        
        # Text statistics
        logger.info("Computing text statistics...")
        processed_df['text_length'] = processed_df[text_column].str.len()
        processed_df['word_count'] = processed_df[text_column].str.split().str.len()
        processed_df['avg_word_length'] = processed_df[text_column].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0
        )
        processed_df['uppercase_ratio'] = processed_df[text_column].apply(
            lambda x: sum(1 for c in x if c.isupper()) / len(x) if x else 0
        )
        processed_df['punctuation_ratio'] = processed_df[text_column].apply(
            lambda x: sum(1 for c in x if c in string.punctuation) / len(x) if x else 0
        )
        
        return processed_df
    
    def create_features(self, df: pd.DataFrame, 
                       text_column: str = 'cleaned_text',
                       max_features: int = 5000) -> Tuple[np.ndarray, List[str]]:
        """Create feature matrix from processed comments.
        
        Args:
            df: Processed DataFrame
            text_column: Column containing cleaned text
            max_features: Maximum number of TF-IDF features
            
        Returns:
            Feature matrix and feature names
        """
        # TF-IDF features
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            tfidf_features = self.tfidf_vectorizer.fit_transform(df[text_column])
        else:
            tfidf_features = self.tfidf_vectorizer.transform(df[text_column])
        
        # Rule-based features
        rule_columns = [col for col in df.columns if col.startswith('rule_')]
        rule_features = df[rule_columns].values
        
        # Statistical features
        stat_columns = ['text_length', 'word_count', 'avg_word_length', 
                       'uppercase_ratio', 'punctuation_ratio']
        stat_features = df[stat_columns].values
        
        # Scale statistical features
        stat_features = self.scaler.fit_transform(stat_features)
        
        # Combine all features
        features = np.hstack([
            tfidf_features.toarray(),
            rule_features,
            stat_features
        ])
        
        # Feature names
        feature_names = (
            [f'tfidf_{i}' for i in range(tfidf_features.shape[1])] +
            rule_columns +
            stat_columns
        )
        
        return features, feature_names
    
    def tokenize_for_transformer(self, texts: List[str], 
                                max_length: int = 512) -> Dict[str, torch.Tensor]:
        """Tokenize texts for transformer models.
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with tokenized inputs
        """
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
    
    def prepare_kaggle_submission(self, predictions: np.ndarray, 
                                 row_ids: List[int],
                                 output_path: str = 'submission.csv') -> None:
        """Prepare submission file for Kaggle competition.
        
        Args:
            predictions: Array of prediction probabilities
            row_ids: List of row IDs
            output_path: Path to save submission file
        """
        submission_df = pd.DataFrame({
            'row_id': row_ids,
            'rule_violation': predictions
        })
        
        submission_df.to_csv(output_path, index=False)
        logger.info(f"Submission saved to {output_path}")

def load_reddit_data(data_path: str) -> pd.DataFrame:
    """Load Reddit comment data from various formats.
    
    Args:
        data_path: Path to the data file
        
    Returns:
        DataFrame with Reddit comments
    """
    path = Path(data_path)
    
    if path.suffix == '.csv':
        return pd.read_csv(data_path)
    elif path.suffix == '.json':
        return pd.read_json(data_path)
    elif path.suffix == '.parquet':
        return pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

def main():
    """Example usage of the Reddit data pipeline."""
    # Initialize pipeline
    pipeline = RedditDataPipeline()
    
    # Create sample data for demonstration
    sample_data = pd.DataFrame({
        'row_id': range(1000),
        'comment_text': [
            "This is a great post, thanks for sharing!",
            "You're an idiot if you believe this nonsense.",
            "Check out my website at www.example.com for more info!",
            "This is completely unrelated but I love pizza.",
            "What a load of crap, this is fake news.",
            "lol",
        ] * 167,  # Repeat to get 1000+ samples
        'rule_violation': np.random.randint(0, 2, 1000)
    })
    
    # Process comments
    processed_data = pipeline.process_comments(sample_data)
    
    # Create features
    features, feature_names = pipeline.create_features(processed_data)
    
    print(f"Processed {len(processed_data)} comments")
    print(f"Feature matrix shape: {features.shape}")
    print(f"Number of features: {len(feature_names)}")
    
    # Example predictions (random for demo)
    predictions = np.random.random(len(sample_data))
    
    # Prepare submission
    pipeline.prepare_kaggle_submission(
        predictions, 
        sample_data['row_id'].tolist(),
        'sample_submission.csv'
    )

if __name__ == '__main__':
    main()