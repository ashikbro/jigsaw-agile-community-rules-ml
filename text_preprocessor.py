#!/usr/bin/env python3
"""
Comprehensive Text Preprocessing Module for Binary Text Classification
Handles lowercasing, punctuation removal, stopwords, lemmatization, and special token replacement
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import emoji
from typing import List, Optional

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

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

class TextPreprocessor:
    """
    Comprehensive text preprocessor for Reddit comment classification
    """
    
    def __init__(self, 
                 remove_stopwords: bool = True,
                 use_lemmatization: bool = True,
                 min_word_length: int = 2):
        """
        Initialize the text preprocessor
        
        Args:
            remove_stopwords: Whether to remove stopwords
            use_lemmatization: Whether to use lemmatization (vs stemming)
            min_word_length: Minimum word length to keep
        """
        self.remove_stopwords = remove_stopwords
        self.use_lemmatization = use_lemmatization
        self.min_word_length = min_word_length
        
        # Initialize NLTK components
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.reddit_user_pattern = re.compile(r'u/[A-Za-z0-9_-]+')
        self.reddit_sub_pattern = re.compile(r'r/[A-Za-z0-9_-]+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.number_pattern = re.compile(r'\b\d+\b')
        self.whitespace_pattern = re.compile(r'\s+')
        
    def remove_html_tags(self, text: str) -> str:
        """Remove HTML tags using BeautifulSoup"""
        soup = BeautifulSoup(text, 'html.parser')
        return soup.get_text()
    
    def remove_emojis(self, text: str) -> str:
        """Remove emojis from text"""
        return emoji.replace_emoji(text, replace='')
    
    def replace_special_tokens(self, text: str) -> str:
        """Replace URLs, mentions, subreddit tags with special tokens"""
        # Replace URLs
        text = self.url_pattern.sub(' URL_TOKEN ', text)
        
        # Replace Reddit user mentions
        text = self.reddit_user_pattern.sub(' USER_TOKEN ', text)
        
        # Replace Reddit subreddit mentions
        text = self.reddit_sub_pattern.sub(' SUBREDDIT_TOKEN ', text)
        
        # Replace email addresses
        text = self.email_pattern.sub(' EMAIL_TOKEN ', text)
        
        return text
    
    def remove_punctuation_and_numbers(self, text: str) -> str:
        """Remove punctuation and numbers"""
        # Remove numbers
        text = self.number_pattern.sub(' ', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        return text
    
    def tokenize_and_clean(self, text: str) -> List[str]:
        """Tokenize text and apply word-level cleaning"""
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Filter tokens
        cleaned_tokens = []
        for token in tokens:
            # Skip if too short
            if len(token) < self.min_word_length:
                continue
                
            # Skip stopwords if enabled
            if self.remove_stopwords and token in self.stop_words:
                continue
                
            # Apply lemmatization if enabled
            if self.use_lemmatization and self.lemmatizer:
                token = self.lemmatizer.lemmatize(token)
                
            cleaned_tokens.append(token)
            
        return cleaned_tokens
    
    def preprocess_text(self, text: str) -> str:
        """Apply complete preprocessing pipeline"""
        if not isinstance(text, str):
            return ""
            
        # Step 1: Remove HTML tags
        text = self.remove_html_tags(text)
        
        # Step 2: Remove emojis
        text = self.remove_emojis(text)
        
        # Step 3: Replace special tokens (URLs, mentions, etc.)
        text = self.replace_special_tokens(text)
        
        # Step 4: Remove punctuation and numbers
        text = self.remove_punctuation_and_numbers(text)
        
        # Step 5: Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text)
        
        # Step 6: Tokenize and apply word-level cleaning
        tokens = self.tokenize_and_clean(text)
        
        # Step 7: Join tokens back to string
        return ' '.join(tokens)
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess a batch of texts"""
        return [self.preprocess_text(text) for text in texts]

# Example usage and testing
if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = TextPreprocessor()
    
    test_texts = [
        "Hey u/username, check out r/MachineLearning! Visit https://example.com for more info. 😀",
        "This is TERRIBLE!!! You're breaking the rules... Contact me at test@email.com",
        "<p>HTML content</p> with numbers 123 and punctuation!!!",
        "Normal text without special characters"
    ]
    
    print("Original vs Preprocessed:")
    for text in test_texts:
        processed = preprocessor.preprocess_text(text)
        print(f"Original: {text}")
        print(f"Processed: {processed}")
        print("-" * 50)