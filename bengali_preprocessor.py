#!/usr/bin/env python3
"""
Bengali Text Preprocessor for Content Moderation

This module provides comprehensive Bengali text preprocessing capabilities
including linguistic features, stemming, stop word removal, and normalization.

Features:
1. Bengali script normalization
2. Stop word removal (Bengali-specific)
3. Stemming and lemmatization
4. Punctuation and special character handling
5. Number normalization (Bengali to English digits)
6. Text cleaning and standardization
7. Cultural context preservation

Author: AI Assistant
Date: 2025-01-31
"""

import re
import string
import unicodedata
from typing import List, Dict, Set, Optional, Tuple
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BengaliPreprocessor:
    """Comprehensive Bengali text preprocessor for content moderation."""
    
    def __init__(self):
        """Initialize the Bengali preprocessor."""
        self._init_stop_words()
        self._init_normalization_maps()
        self._init_stemming_rules()
        self._init_regex_patterns()
    
    def _init_stop_words(self):
        """Initialize Bengali stop words."""
        self.bengali_stop_words = {
            # Common Bengali stop words
            'আমি', 'তুমি', 'তিনি', 'আমরা', 'তোমরা', 'তারা', 'আপনি', 'আপনারা',
            'এই', 'সেই', 'ওই', 'এটি', 'সেটি', 'ওটি', 'এটা', 'সেটা', 'ওটা',
            'যে', 'যা', 'যার', 'যাকে', 'যাদের', 'যেটি', 'যেটা',
            'কি', 'কী', 'কে', 'কার', 'কাকে', 'কাদের', 'কোন', 'কোনো',
            'এবং', 'বা', 'অথবা', 'কিন্তু', 'তবে', 'যদি', 'তাহলে',
            'না', 'নয়', 'নেই', 'নাই', 'হয়', 'হবে', 'হয়েছে', 'হয়েছিল',
            'আছে', 'ছিল', 'থাকে', 'থাকবে', 'থেকে', 'পর্যন্ত',
            'জন্য', 'কারণে', 'সাথে', 'দিয়ে', 'করে', 'হয়ে',
            'উপর', 'নিচে', 'ভিতর', 'বাহির', 'সামনে', 'পিছনে',
            'আগে', 'পরে', 'এখন', 'তখন', 'কখন', 'সবসময়',
            'অনেক', 'কম', 'বেশি', 'সব', 'সকল', 'কিছু', 'কিছুই',
            'একটি', 'একটা', 'দুটি', 'দুটো', 'তিনটি', 'তিনটে',
            'প্রথম', 'দ্বিতীয়', 'তৃতীয়', 'শেষ', 'মধ্যে',
            'ভাল', 'ভালো', 'মন্দ', 'খারাপ', 'ভাল', 'সুন্দর',
            'বড়', 'ছোট', 'নতুন', 'পুরাতন', 'পুরানো'
        }
        
        # Additional contextual stop words for content moderation
        self.contextual_stop_words = {
            'বলেছে', 'বলেছেন', 'বলেছিল', 'বলেছিলেন',
            'দেখেছে', 'দেখেছেন', 'দেখেছিল', 'দেখেছিলেন',
            'শুনেছে', 'শুনেছেন', 'শুনেছিল', 'শুনেছিলেন',
            'জানে', 'জানেন', 'জানত', 'জানতেন',
            'মনে', 'মনের', 'চোখে', 'চোখের', 'কানে', 'কানের'
        }
    
    def _init_normalization_maps(self):
        """Initialize character normalization mappings."""
        # Bengali to English digit mapping
        self.digit_map = {
            '০': '0', '১': '1', '২': '2', '৩': '3', '৪': '4',
            '৫': '5', '৬': '6', '৭': '7', '৮': '8', '৯': '9'
        }
        
        # Common Bengali character variations
        self.char_normalization = {
            'ৎ': 'ত্',  # Khanda ta
            'ং': 'ঙ্',  # Anusvara
            'ঃ': 'হ্',  # Visarga
            '।': '.',   # Dari (Bengali period)
            '॥': '.',   # Double dari
        }
        
        # Romanized Bengali to Bengali script (common patterns)
        self.romanized_map = {
            'ami': 'আমি',
            'tumi': 'তুমি',
            'apni': 'আপনি',
            'kemon': 'কেমন',
            'achen': 'আছেন',
            'bhalo': 'ভালো',
            'kharap': 'খারাপ',
            'dhonnobad': 'ধন্যবাদ'
        }
    
    def _init_stemming_rules(self):
        """Initialize Bengali stemming rules."""
        # Common Bengali suffixes for stemming
        self.suffixes = {
            # Verb suffixes
            'ছিলাম': '', 'ছিলে': '', 'ছিল': '', 'ছিলেন': '',
            'করেছি': 'কর', 'করেছে': 'কর', 'করেছেন': 'কর',
            'করব': 'কর', 'করবে': 'কর', 'করবেন': 'কর',
            'করি': 'কর', 'করে': 'কর', 'করেন': 'কর',
            'করত': 'কর', 'করতে': 'কর', 'করতেন': 'কর',
            
            # Noun suffixes
            'গুলো': '', 'গুলি': '', 'গুলা': '',
            'দের': '', 'রা': '', 'এরা': '',
            'টি': '', 'টা': '', 'খানা': '', 'খানি': '',
            
            # Adjective suffixes
            'ওয়ালা': '', 'ওয়ালী': '', 'দার': '',
            'কারী': '', 'কারি': '',
            
            # Case markers
            'কে': '', 'র': '', 'এর': '', 'তে': '', 'য়': ''
        }
        
        # Prefix patterns
        self.prefixes = {
            'অনু': '', 'প্রতি': '', 'সু': '', 'দুর': '',
            'অধি': '', 'উপ': '', 'নির': '', 'বি': ''
        }
    
    def _init_regex_patterns(self):
        """Initialize regex patterns for text processing."""
        # Bengali Unicode range
        self.bengali_pattern = re.compile(r'[\u0980-\u09FF]+')
        
        # URL pattern
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        # Email pattern
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        
        # Phone number pattern (Bengali context)
        self.phone_pattern = re.compile(r'[+]?[০-৯0-9]{10,15}')
        
        # Hashtag pattern
        self.hashtag_pattern = re.compile(r'#\w+')
        
        # Mention pattern
        self.mention_pattern = re.compile(r'@\w+')
        
        # Repeated character pattern
        self.repeated_char_pattern = re.compile(r'(.)\1{2,}')
        
        # Multiple punctuation pattern
        self.multiple_punct_pattern = re.compile(r'[।!?.,;:]{2,}')
        
        # Whitespace pattern
        self.whitespace_pattern = re.compile(r'\s+')
    
    def normalize_text(self, text: str) -> str:
        """Normalize Bengali text."""
        if not text:
            return ""
        
        # Convert to lowercase (preserves Bengali characters)
        text = text.lower()
        
        # Normalize Unicode characters
        text = unicodedata.normalize('NFC', text)
        
        # Convert Bengali digits to English digits
        for bengali_digit, english_digit in self.digit_map.items():
            text = text.replace(bengali_digit, english_digit)
        
        # Apply character normalization
        for old_char, new_char in self.char_normalization.items():
            text = text.replace(old_char, new_char)
        
        # Convert common romanized words to Bengali
        for romanized, bengali in self.romanized_map.items():
            text = re.sub(r'\b' + romanized + r'\b', bengali, text, flags=re.IGNORECASE)
        
        return text
    
    def clean_text(self, text: str, preserve_bengali: bool = True) -> str:
        """Clean and standardize text."""
        if not text:
            return ""
        
        # Remove URLs
        text = self.url_pattern.sub(' ', text)
        
        # Remove email addresses
        text = self.email_pattern.sub(' ', text)
        
        # Remove phone numbers
        text = self.phone_pattern.sub(' ', text)
        
        # Remove hashtags and mentions (optional)
        text = self.hashtag_pattern.sub(' ', text)
        text = self.mention_pattern.sub(' ', text)
        
        # Handle repeated characters (keep max 2 repetitions)
        text = self.repeated_char_pattern.sub(r'\1\1', text)
        
        # Normalize multiple punctuation
        text = self.multiple_punct_pattern.sub('.', text)
        
        # Remove extra whitespace
        text = self.whitespace_pattern.sub(' ', text)
        
        # Remove non-Bengali characters if preserve_bengali is True
        if preserve_bengali:
            # Keep Bengali characters, English letters, digits, and basic punctuation
            text = re.sub(r'[^\u0980-\u09FFa-zA-Z0-9\s।!?.,;:()\[\]{}"\'-]', ' ', text)
        
        return text.strip()
    
    def remove_stop_words(self, text: str, include_contextual: bool = False) -> str:
        """Remove Bengali stop words from text."""
        if not text:
            return ""
        
        words = text.split()
        stop_words = self.bengali_stop_words.copy()
        
        if include_contextual:
            stop_words.update(self.contextual_stop_words)
        
        filtered_words = [word for word in words if word not in stop_words]
        
        return ' '.join(filtered_words)
    
    def stem_text(self, text: str) -> str:
        """Apply Bengali stemming to text."""
        if not text:
            return ""
        
        words = text.split()
        stemmed_words = []
        
        for word in words:
            stemmed_word = self._stem_word(word)
            stemmed_words.append(stemmed_word)
        
        return ' '.join(stemmed_words)
    
    def _stem_word(self, word: str) -> str:
        """Stem a single Bengali word."""
        if len(word) < 3:
            return word
        
        original_word = word
        
        # Remove suffixes (longest match first)
        suffixes_sorted = sorted(self.suffixes.keys(), key=len, reverse=True)
        for suffix in suffixes_sorted:
            if word.endswith(suffix) and len(word) > len(suffix) + 1:
                stem = word[:-len(suffix)]
                if self.suffixes[suffix]:  # If there's a replacement
                    return self.suffixes[suffix]
                else:
                    return stem
        
        # Remove prefixes
        prefixes_sorted = sorted(self.prefixes.keys(), key=len, reverse=True)
        for prefix in prefixes_sorted:
            if word.startswith(prefix) and len(word) > len(prefix) + 1:
                return word[len(prefix):]
        
        return original_word
    
    def extract_features(self, text: str) -> Dict[str, any]:
        """Extract linguistic features from Bengali text."""
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'bengali_char_count': len(self.bengali_pattern.findall(text)),
            'has_bengali': bool(self.bengali_pattern.search(text)),
            'has_english': bool(re.search(r'[a-zA-Z]', text)),
            'has_digits': bool(re.search(r'[0-9০-৯]', text)),
            'has_punctuation': bool(re.search(r'[।!?.,;:]', text)),
            'has_urls': bool(self.url_pattern.search(text)),
            'has_emails': bool(self.email_pattern.search(text)),
            'has_phones': bool(self.phone_pattern.search(text)),
            'has_hashtags': bool(self.hashtag_pattern.search(text)),
            'has_mentions': bool(self.mention_pattern.search(text)),
            'repeated_chars': len(self.repeated_char_pattern.findall(text)),
            'multiple_punct': len(self.multiple_punct_pattern.findall(text))
        }
        
        # Calculate ratios
        if features['length'] > 0:
            features['bengali_ratio'] = features['bengali_char_count'] / features['length']
        else:
            features['bengali_ratio'] = 0.0
        
        # Detect formality level
        formal_indicators = ['আপনি', 'আপনার', 'আপনাকে', 'আপনাদের']
        informal_indicators = ['তুমি', 'তোমার', 'তোমাকে', 'তুই', 'তোর', 'তোকে']
        
        formal_count = sum(1 for indicator in formal_indicators if indicator in text)
        informal_count = sum(1 for indicator in informal_indicators if indicator in text)
        
        if formal_count > informal_count:
            features['formality'] = 'formal'
        elif informal_count > formal_count:
            features['formality'] = 'informal'
        else:
            features['formality'] = 'neutral'
        
        return features
    
    def preprocess_for_ml(self, text: str, 
                         normalize: bool = True,
                         clean: bool = True,
                         remove_stops: bool = True,
                         stem: bool = False,
                         preserve_bengali: bool = True) -> str:
        """Complete preprocessing pipeline for ML models."""
        if not text:
            return ""
        
        processed_text = text
        
        # Step 1: Normalize
        if normalize:
            processed_text = self.normalize_text(processed_text)
        
        # Step 2: Clean
        if clean:
            processed_text = self.clean_text(processed_text, preserve_bengali)
        
        # Step 3: Remove stop words
        if remove_stops:
            processed_text = self.remove_stop_words(processed_text)
        
        # Step 4: Stem (optional, can reduce accuracy for some models)
        if stem:
            processed_text = self.stem_text(processed_text)
        
        return processed_text.strip()
    
    def batch_preprocess(self, texts: List[str], **kwargs) -> List[str]:
        """Preprocess a batch of texts."""
        return [self.preprocess_for_ml(text, **kwargs) for text in texts]
    
    def detect_language_mix(self, text: str) -> Dict[str, float]:
        """Detect language mixing in text."""
        if not text:
            return {'bengali': 0.0, 'english': 0.0, 'mixed': 0.0}
        
        bengali_chars = len(self.bengali_pattern.findall(text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = len(re.sub(r'\s+', '', text))
        
        if total_chars == 0:
            return {'bengali': 0.0, 'english': 0.0, 'mixed': 0.0}
        
        bengali_ratio = bengali_chars / total_chars
        english_ratio = english_chars / total_chars
        
        # Consider mixed if both languages have significant presence
        mixed_threshold = 0.2
        is_mixed = bengali_ratio > mixed_threshold and english_ratio > mixed_threshold
        
        return {
            'bengali': bengali_ratio,
            'english': english_ratio,
            'mixed': 1.0 if is_mixed else 0.0
        }
    
    def get_preprocessing_stats(self, original_text: str, processed_text: str) -> Dict[str, any]:
        """Get statistics about preprocessing changes."""
        return {
            'original_length': len(original_text),
            'processed_length': len(processed_text),
            'length_reduction': len(original_text) - len(processed_text),
            'original_words': len(original_text.split()),
            'processed_words': len(processed_text.split()),
            'word_reduction': len(original_text.split()) - len(processed_text.split()),
            'compression_ratio': len(processed_text) / max(len(original_text), 1)
        }

def main():
    """Test the Bengali preprocessor."""
    preprocessor = BengaliPreprocessor()
    
    # Test texts
    test_texts = [
        "আমি তোমাকে খুব ভালোবাসি এবং তুমি আমার সবচেয়ে প্রিয় মানুষ।",
        "তুই একটা বোকা এবং আমি তোকে মারব।",
        "আজকের আবহাওয়া খুব ভালো, বৃষ্টি হতে পারে।",
        "ami tomar sathe kotha bolte chai।",
        "এই ওয়েবসাইট http://example.com দেখো।",
        "আমার ফোন নম্বর ০১৭১২৩৪৫৬৭৮।"
    ]
    
    print("Bengali Text Preprocessing Demo\n")
    print("=" * 50)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: {text}")
        print("-" * 30)
        
        # Normalize
        normalized = preprocessor.normalize_text(text)
        print(f"Normalized: {normalized}")
        
        # Clean
        cleaned = preprocessor.clean_text(normalized)
        print(f"Cleaned: {cleaned}")
        
        # Remove stop words
        no_stops = preprocessor.remove_stop_words(cleaned)
        print(f"No stops: {no_stops}")
        
        # Full preprocessing
        processed = preprocessor.preprocess_for_ml(text)
        print(f"Full preprocessing: {processed}")
        
        # Extract features
        features = preprocessor.extract_features(text)
        print(f"Features: {features}")
        
        # Language detection
        lang_mix = preprocessor.detect_language_mix(text)
        print(f"Language mix: {lang_mix}")
        
        # Stats
        stats = preprocessor.get_preprocessing_stats(text, processed)
        print(f"Stats: {stats}")

if __name__ == "__main__":
    main()