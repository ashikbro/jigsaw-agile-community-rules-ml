#!/usr/bin/env python3
"""
Multilingual Support Module for Content Moderation System
Provides language detection, translation, and cross-lingual model capabilities
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Language detection and translation
try:
    from langdetect import detect, detect_langs
    from langdetect.lang_detect_exception import LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logging.warning("langdetect not available. Install with: pip install langdetect")

try:
    from googletrans import Translator
    GOOGLETRANS_AVAILABLE = True
except ImportError:
    GOOGLETRANS_AVAILABLE = False
    logging.warning("googletrans not available. Install with: pip install googletrans==4.0.0rc1")

# Multilingual embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LanguageDetector:
    """
    Language detection utility with confidence scoring
    """
    
    def __init__(self):
        if not LANGDETECT_AVAILABLE:
            raise ImportError("langdetect is required. Install with: pip install langdetect")
    
    def detect_language(self, text: str, return_confidence: bool = False) -> Union[str, Tuple[str, float]]:
        """
        Detect the language of input text
        
        Args:
            text: Input text to analyze
            return_confidence: Whether to return confidence score
            
        Returns:
            Language code (e.g., 'en', 'es') or tuple of (language, confidence)
        """
        try:
            if return_confidence:
                langs = detect_langs(text)
                if langs:
                    return langs[0].lang, langs[0].prob
                return 'unknown', 0.0
            else:
                return detect(text)
        except LangDetectException:
            if return_confidence:
                return 'unknown', 0.0
            return 'unknown'
    
    def detect_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """
        Detect languages for a batch of texts
        
        Args:
            texts: List of input texts
            
        Returns:
            List of dictionaries with language and confidence information
        """
        results = []
        for text in texts:
            lang, confidence = self.detect_language(text, return_confidence=True)
            results.append({
                'text': text[:50] + '...' if len(text) > 50 else text,
                'language': lang,
                'confidence': confidence
            })
        return results

class MultilingualTranslator:
    """
    Translation utility for converting text to target language
    """
    
    def __init__(self, target_language: str = 'en'):
        if not GOOGLETRANS_AVAILABLE:
            raise ImportError("googletrans is required. Install with: pip install googletrans==4.0.0rc1")
        
        self.translator = Translator()
        self.target_language = target_language
        self.supported_languages = {
            'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
            'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese',
            'ko': 'Korean', 'zh': 'Chinese', 'ar': 'Arabic', 'hi': 'Hindi',
            'tr': 'Turkish', 'pl': 'Polish', 'nl': 'Dutch', 'sv': 'Swedish'
        }
    
    def translate_text(self, text: str, source_lang: Optional[str] = None) -> Dict[str, str]:
        """
        Translate text to target language
        
        Args:
            text: Input text to translate
            source_lang: Source language code (auto-detected if None)
            
        Returns:
            Dictionary with original text, translated text, and language info
        """
        try:
            if source_lang:
                result = self.translator.translate(text, src=source_lang, dest=self.target_language)
            else:
                result = self.translator.translate(text, dest=self.target_language)
            
            return {
                'original_text': text,
                'translated_text': result.text,
                'source_language': result.src,
                'target_language': self.target_language,
                'confidence': getattr(result, 'confidence', None)
            }
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return {
                'original_text': text,
                'translated_text': text,  # Return original if translation fails
                'source_language': 'unknown',
                'target_language': self.target_language,
                'confidence': None,
                'error': str(e)
            }
    
    def translate_batch(self, texts: List[str], source_langs: Optional[List[str]] = None) -> List[Dict[str, str]]:
        """
        Translate a batch of texts
        
        Args:
            texts: List of input texts
            source_langs: List of source language codes (optional)
            
        Returns:
            List of translation results
        """
        results = []
        for i, text in enumerate(texts):
            source_lang = source_langs[i] if source_langs else None
            result = self.translate_text(text, source_lang)
            results.append(result)
        return results

class MultilingualEmbeddings:
    """
    Multilingual text embeddings using sentence transformers
    """
    
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
        
        self.model_name = model_name
        self.model = None
        self.supported_models = {
            'paraphrase-multilingual-MiniLM-L12-v2': 'Fast multilingual model (50+ languages)',
            'paraphrase-multilingual-mpnet-base-v2': 'High-quality multilingual model',
            'distiluse-base-multilingual-cased': 'Distilled multilingual model',
            'stsb-xlm-r-multilingual': 'XLM-R based multilingual model'
        }
    
    def load_model(self):
        """Load the sentence transformer model"""
        if self.model is None:
            logger.info(f"Loading multilingual model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")
    
    def encode_texts(self, texts: List[str], normalize_embeddings: bool = True) -> np.ndarray:
        """
        Encode texts into multilingual embeddings
        
        Args:
            texts: List of input texts
            normalize_embeddings: Whether to normalize embeddings
            
        Returns:
            Numpy array of embeddings
        """
        if self.model is None:
            self.load_model()
        
        embeddings = self.model.encode(texts, normalize_embeddings=normalize_embeddings)
        return embeddings
    
    def similarity_search(self, query: str, corpus: List[str], top_k: int = 5) -> List[Dict[str, Union[str, float]]]:
        """
        Find most similar texts in corpus using multilingual embeddings
        
        Args:
            query: Query text
            corpus: List of texts to search
            top_k: Number of top results to return
            
        Returns:
            List of similar texts with similarity scores
        """
        if self.model is None:
            self.load_model()
        
        # Encode query and corpus
        query_embedding = self.model.encode([query])
        corpus_embeddings = self.model.encode(corpus)
        
        # Calculate similarities
        similarities = np.dot(query_embedding, corpus_embeddings.T)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'text': corpus[idx],
                'similarity': float(similarities[idx]),
                'index': int(idx)
            })
        
        return results

class MultilingualContentModerator:
    """
    Main class for multilingual content moderation
    """
    
    def __init__(self, target_language: str = 'en', auto_translate: bool = True):
        self.target_language = target_language
        self.auto_translate = auto_translate
        
        # Initialize components
        self.language_detector = LanguageDetector() if LANGDETECT_AVAILABLE else None
        self.translator = MultilingualTranslator(target_language) if GOOGLETRANS_AVAILABLE else None
        self.embeddings = MultilingualEmbeddings() if SENTENCE_TRANSFORMERS_AVAILABLE else None
        
        # Model storage
        self.models = {}
        self.language_stats = {}
    
    def analyze_text(self, text: str) -> Dict[str, Union[str, float, Dict]]:
        """
        Comprehensive analysis of multilingual text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with language detection, translation, and moderation results
        """
        result = {
            'original_text': text,
            'detected_language': None,
            'confidence': None,
            'translated_text': None,
            'moderation_result': None,
            'processing_steps': []
        }
        
        # Step 1: Language Detection
        if self.language_detector:
            lang, confidence = self.language_detector.detect_language(text, return_confidence=True)
            result['detected_language'] = lang
            result['confidence'] = confidence
            result['processing_steps'].append('language_detection')
        
        # Step 2: Translation (if needed)
        text_to_moderate = text
        if self.auto_translate and self.translator and result['detected_language'] != self.target_language:
            translation_result = self.translator.translate_text(text, result['detected_language'])
            result['translated_text'] = translation_result['translated_text']
            text_to_moderate = translation_result['translated_text']
            result['processing_steps'].append('translation')
        
        # Step 3: Content Moderation (placeholder - integrate with existing models)
        # This would integrate with your existing model prediction logic
        result['moderation_result'] = {
            'text_analyzed': text_to_moderate,
            'prediction': 'safe',  # Placeholder
            'confidence': 0.85,    # Placeholder
            'explanation': 'Multilingual analysis completed'  # Placeholder
        }
        result['processing_steps'].append('content_moderation')
        
        return result
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float, Dict]]]:
        """
        Analyze a batch of multilingual texts
        
        Args:
            texts: List of input texts
            
        Returns:
            List of analysis results
        """
        results = []
        for text in texts:
            result = self.analyze_text(text)
            results.append(result)
        
        # Update language statistics
        self._update_language_stats(results)
        
        return results
    
    def _update_language_stats(self, results: List[Dict]):
        """Update language usage statistics"""
        for result in results:
            lang = result.get('detected_language', 'unknown')
            if lang in self.language_stats:
                self.language_stats[lang] += 1
            else:
                self.language_stats[lang] = 1
    
    def get_language_statistics(self) -> Dict[str, Union[int, float]]:
        """
        Get language usage statistics
        
        Returns:
            Dictionary with language counts and percentages
        """
        total = sum(self.language_stats.values())
        if total == 0:
            return {}
        
        stats = {}
        for lang, count in self.language_stats.items():
            stats[lang] = {
                'count': count,
                'percentage': (count / total) * 100
            }
        
        return stats
    
    def save_configuration(self, filepath: str):
        """
        Save multilingual configuration to file
        
        Args:
            filepath: Path to save configuration
        """
        config = {
            'target_language': self.target_language,
            'auto_translate': self.auto_translate,
            'language_stats': self.language_stats,
            'available_components': {
                'language_detector': self.language_detector is not None,
                'translator': self.translator is not None,
                'embeddings': self.embeddings is not None
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Configuration saved to {filepath}")
    
    def load_configuration(self, filepath: str):
        """
        Load multilingual configuration from file
        
        Args:
            filepath: Path to load configuration from
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.target_language = config.get('target_language', 'en')
        self.auto_translate = config.get('auto_translate', True)
        self.language_stats = config.get('language_stats', {})
        
        logger.info(f"Configuration loaded from {filepath}")
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages
        
        Returns:
            List of supported language codes
        """
        # Return commonly supported languages by the models
        return [
            'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh-cn', 'zh-tw',
            'ar', 'hi', 'th', 'vi', 'tr', 'pl', 'nl', 'sv', 'da', 'no', 'fi',
            'cs', 'sk', 'hu', 'ro', 'bg', 'hr', 'sl', 'et', 'lv', 'lt', 'mt'
        ]

def create_multilingual_demo():
    """
    Create a demonstration of multilingual capabilities
    """
    print("🌍 Multilingual Content Moderation Demo")
    print("=" * 50)
    
    # Sample texts in different languages
    sample_texts = [
        "This is a great product! I love it.",  # English
        "Este es un producto increíble! Me encanta.",  # Spanish
        "C'est un produit formidable! Je l'adore.",  # French
        "Das ist ein tolles Produkt! Ich liebe es.",  # German
        "これは素晴らしい製品です！大好きです。",  # Japanese
        "这是一个很棒的产品！我喜欢它。",  # Chinese
    ]
    
    try:
        # Initialize multilingual moderator
        moderator = MultilingualContentModerator()
        
        print("\n📝 Analyzing multilingual texts...")
        results = moderator.analyze_batch(sample_texts)
        
        for i, result in enumerate(results, 1):
            print(f"\n--- Text {i} ---")
            print(f"Original: {result['original_text']}")
            print(f"Language: {result['detected_language']} (confidence: {result.get('confidence', 'N/A'):.2f})")
            if result['translated_text']:
                print(f"Translated: {result['translated_text']}")
            print(f"Processing: {' → '.join(result['processing_steps'])}")
        
        print("\n📊 Language Statistics:")
        stats = moderator.get_language_statistics()
        for lang, data in stats.items():
            print(f"{lang}: {data['count']} texts ({data['percentage']:.1f}%)")
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("\n📦 To install required packages:")
        print("pip install langdetect googletrans==4.0.0rc1 sentence-transformers")

if __name__ == "__main__":
    create_multilingual_demo()