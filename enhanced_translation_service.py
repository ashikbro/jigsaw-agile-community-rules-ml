#!/usr/bin/env python3
"""
Enhanced Translation Service with Multiple Fallbacks

This module provides a robust translation system with multiple fallback services
to ensure reliable translation for multilingual content moderation.

Features:
1. Multiple translation service providers
2. Automatic fallback on service failures
3. Translation quality validation
4. Caching for improved performance
5. Language detection with confidence scoring
6. Batch translation support

Author: AI Assistant
Date: 2025-01-31
"""

import os
import json
import logging
import time
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import requests
from urllib.parse import quote

# Language detection
try:
    import langdetect
    from langdetect import detect_langs
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logging.warning("langdetect not available")

# Google Translate
try:
    from googletrans import Translator as GoogleTranslator
    GOOGLE_TRANSLATE_AVAILABLE = True
except ImportError:
    GOOGLE_TRANSLATE_AVAILABLE = False
    logging.warning("googletrans not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TranslationResult:
    """Result of a translation operation."""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: float
    service_used: str
    processing_time: float
    success: bool
    error_message: Optional[str] = None

@dataclass
class LanguageDetectionResult:
    """Result of language detection."""
    text: str
    detected_language: str
    confidence: float
    all_detections: List[Tuple[str, float]]
    success: bool
    error_message: Optional[str] = None

class LibreTranslateService:
    """LibreTranslate service implementation (free, self-hostable)."""
    
    def __init__(self, base_url: str = "https://libretranslate.de", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'Enhanced-Translation-Service/1.0'
        })
    
    def translate(self, text: str, target_lang: str, source_lang: str = "auto") -> Optional[str]:
        """Translate text using LibreTranslate."""
        try:
            data = {
                'q': text,
                'source': source_lang,
                'target': target_lang,
                'format': 'text'
            }
            
            if self.api_key:
                data['api_key'] = self.api_key
            
            response = self.session.post(
                f"{self.base_url}/translate",
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('translatedText')
            else:
                logger.warning(f"LibreTranslate API error: {response.status_code}")
                return None
        
        except Exception as e:
            logger.warning(f"LibreTranslate translation failed: {e}")
            return None
    
    def detect_language(self, text: str) -> Optional[str]:
        """Detect language using LibreTranslate."""
        try:
            data = {'q': text}
            if self.api_key:
                data['api_key'] = self.api_key
            
            response = self.session.post(
                f"{self.base_url}/detect",
                json=data,
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                detections = result.get('detections', [])
                if detections:
                    return detections[0].get('language')
            
            return None
        
        except Exception as e:
            logger.warning(f"LibreTranslate language detection failed: {e}")
            return None

class MyMemoryTranslateService:
    """MyMemory translation service (free tier available)."""
    
    def __init__(self, email: Optional[str] = None):
        self.base_url = "https://api.mymemory.translated.net"
        self.email = email
        self.session = requests.Session()
    
    def translate(self, text: str, target_lang: str, source_lang: str = "auto") -> Optional[str]:
        """Translate text using MyMemory."""
        try:
            params = {
                'q': text,
                'langpair': f"{source_lang}|{target_lang}"
            }
            
            if self.email:
                params['de'] = self.email
            
            response = self.session.get(
                f"{self.base_url}/get",
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('responseStatus') == 200:
                    return result.get('responseData', {}).get('translatedText')
            
            return None
        
        except Exception as e:
            logger.warning(f"MyMemory translation failed: {e}")
            return None

class EnhancedTranslationService:
    """Enhanced translation service with multiple providers and fallbacks."""
    
    def __init__(self, cache_dir: str = "cache/translations"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize translation services
        self.services = []
        
        # Google Translate (primary)
        if GOOGLE_TRANSLATE_AVAILABLE:
            try:
                google_translator = GoogleTranslator()
                self.services.append(('google', google_translator))
                logger.info("Google Translate service initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Google Translate: {e}")
        
        # LibreTranslate (fallback 1)
        try:
            libre_translator = LibreTranslateService()
            self.services.append(('libretranslate', libre_translator))
            logger.info("LibreTranslate service initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize LibreTranslate: {e}")
        
        # MyMemory (fallback 2)
        try:
            mymemory_translator = MyMemoryTranslateService()
            self.services.append(('mymemory', mymemory_translator))
            logger.info("MyMemory service initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize MyMemory: {e}")
        
        if not self.services:
            logger.error("No translation services available!")
        
        # Language mappings for different services
        self.language_mappings = {
            'bn': 'bn',  # Bengali
            'hi': 'hi',  # Hindi
            'en': 'en',  # English
            'es': 'es',  # Spanish
            'fr': 'fr',  # French
            'de': 'de',  # German
            'ar': 'ar',  # Arabic
            'zh': 'zh',  # Chinese
            'ja': 'ja',  # Japanese
            'ko': 'ko',  # Korean
            'ru': 'ru',  # Russian
            'pt': 'pt',  # Portuguese
            'it': 'it',  # Italian
            'nl': 'nl',  # Dutch
            'tr': 'tr',  # Turkish
        }
    
    def _get_cache_key(self, text: str, source_lang: str, target_lang: str) -> str:
        """Generate cache key for translation."""
        content = f"{text}|{source_lang}|{target_lang}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[str]:
        """Load translation from cache."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('translated_text')
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, original_text: str, translated_text: str, 
                      source_lang: str, target_lang: str, service_used: str):
        """Save translation to cache."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            data = {
                'original_text': original_text,
                'translated_text': translated_text,
                'source_lang': source_lang,
                'target_lang': target_lang,
                'service_used': service_used,
                'timestamp': time.time()
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
    
    def detect_language(self, text: str) -> LanguageDetectionResult:
        """Detect language with confidence scoring."""
        start_time = time.time()
        
        if not text or len(text.strip()) == 0:
            return LanguageDetectionResult(
                text=text,
                detected_language="unknown",
                confidence=0.0,
                all_detections=[],
                success=False,
                error_message="Empty text"
            )
        
        # Try langdetect first (most reliable)
        if LANGDETECT_AVAILABLE:
            try:
                detections = detect_langs(text)
                if detections:
                    all_detections = [(d.lang, d.prob) for d in detections]
                    best_detection = detections[0]
                    
                    return LanguageDetectionResult(
                        text=text,
                        detected_language=best_detection.lang,
                        confidence=best_detection.prob,
                        all_detections=all_detections,
                        success=True
                    )
            except Exception as e:
                logger.warning(f"langdetect failed: {e}")
        
        # Fallback to service-based detection
        for service_name, service in self.services:
            try:
                if hasattr(service, 'detect_language'):
                    detected_lang = service.detect_language(text)
                    if detected_lang:
                        return LanguageDetectionResult(
                            text=text,
                            detected_language=detected_lang,
                            confidence=0.8,  # Assume reasonable confidence
                            all_detections=[(detected_lang, 0.8)],
                            success=True
                        )
            except Exception as e:
                logger.warning(f"Language detection failed with {service_name}: {e}")
        
        # Final fallback
        return LanguageDetectionResult(
            text=text,
            detected_language="unknown",
            confidence=0.0,
            all_detections=[],
            success=False,
            error_message="All language detection methods failed"
        )
    
    def validate_translation(self, original: str, translated: str, 
                           source_lang: str, target_lang: str) -> Tuple[bool, str]:
        """Validate translation quality."""
        if not translated or len(translated.strip()) == 0:
            return False, "Empty translation"
        
        # Check if translation is identical to original (likely failed)
        if original.strip().lower() == translated.strip().lower():
            return False, "Translation identical to original"
        
        # Check reasonable length ratio
        length_ratio = len(translated) / max(len(original), 1)
        if length_ratio < 0.1 or length_ratio > 10:
            return False, f"Unreasonable length ratio: {length_ratio:.2f}"
        
        # Check for common error patterns
        error_patterns = [
            "error", "failed", "unable", "cannot", "invalid",
            "ত্রুটি", "ব্যর্থ", "অক্ষম"  # Bengali error terms
        ]
        
        translated_lower = translated.lower()
        for pattern in error_patterns:
            if pattern in translated_lower:
                return False, f"Error pattern detected: {pattern}"
        
        # Check if target language characters are present (basic check)
        if target_lang == 'bn':  # Bengali
            bengali_chars = any('\u0980' <= char <= '\u09FF' for char in translated)
            if not bengali_chars and len(translated) > 10:
                return False, "No Bengali characters detected in translation"
        
        return True, "Translation appears valid"
    
    def translate_text(self, text: str, target_lang: str = "en", 
                      source_lang: str = "auto", use_cache: bool = True) -> TranslationResult:
        """Translate text with fallback services."""
        start_time = time.time()
        
        if not text or len(text.strip()) == 0:
            return TranslationResult(
                original_text=text,
                translated_text="",
                source_language="unknown",
                target_language=target_lang,
                confidence=0.0,
                service_used="none",
                processing_time=0.0,
                success=False,
                error_message="Empty input text"
            )
        
        # Auto-detect source language if needed
        if source_lang == "auto":
            detection_result = self.detect_language(text)
            if detection_result.success:
                source_lang = detection_result.detected_language
            else:
                source_lang = "unknown"
        
        # Check if translation is needed
        if source_lang == target_lang:
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=source_lang,
                target_language=target_lang,
                confidence=1.0,
                service_used="none",
                processing_time=time.time() - start_time,
                success=True
            )
        
        # Check cache
        cache_key = None
        if use_cache:
            cache_key = self._get_cache_key(text, source_lang, target_lang)
            cached_translation = self._load_from_cache(cache_key)
            if cached_translation:
                return TranslationResult(
                    original_text=text,
                    translated_text=cached_translation,
                    source_language=source_lang,
                    target_language=target_lang,
                    confidence=0.9,  # Assume cached translations are good
                    service_used="cache",
                    processing_time=time.time() - start_time,
                    success=True
                )
        
        # Try each translation service
        for service_name, service in self.services:
            try:
                logger.info(f"Attempting translation with {service_name}")
                
                translated_text = None
                
                if service_name == 'google' and hasattr(service, 'translate'):
                    result = service.translate(text, dest=target_lang, src=source_lang)
                    if result and hasattr(result, 'text'):
                        translated_text = result.text
                
                elif hasattr(service, 'translate'):
                    translated_text = service.translate(text, target_lang, source_lang)
                
                if translated_text:
                    # Validate translation
                    is_valid, validation_message = self.validate_translation(
                        text, translated_text, source_lang, target_lang
                    )
                    
                    if is_valid:
                        # Save to cache
                        if use_cache and cache_key:
                            self._save_to_cache(
                                cache_key, text, translated_text, 
                                source_lang, target_lang, service_name
                            )
                        
                        return TranslationResult(
                            original_text=text,
                            translated_text=translated_text,
                            source_language=source_lang,
                            target_language=target_lang,
                            confidence=0.8,  # Assume good quality for validated translations
                            service_used=service_name,
                            processing_time=time.time() - start_time,
                            success=True
                        )
                    else:
                        logger.warning(f"Translation validation failed for {service_name}: {validation_message}")
            
            except Exception as e:
                logger.warning(f"Translation failed with {service_name}: {e}")
                continue
        
        # All services failed
        return TranslationResult(
            original_text=text,
            translated_text="",
            source_language=source_lang,
            target_language=target_lang,
            confidence=0.0,
            service_used="none",
            processing_time=time.time() - start_time,
            success=False,
            error_message="All translation services failed"
        )
    
    def translate_batch(self, texts: List[str], target_lang: str = "en", 
                       source_lang: str = "auto") -> List[TranslationResult]:
        """Translate multiple texts efficiently."""
        results = []
        
        for text in texts:
            result = self.translate_text(text, target_lang, source_lang)
            results.append(result)
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
        
        return results
    
    def get_service_status(self) -> Dict[str, bool]:
        """Check status of all translation services."""
        status = {}
        
        for service_name, service in self.services:
            try:
                # Test with a simple translation
                test_result = self.translate_text("hello", "es", "en")
                status[service_name] = test_result.success
            except Exception:
                status[service_name] = False
        
        return status

def main():
    """Test the enhanced translation service."""
    # Initialize service
    translator = EnhancedTranslationService()
    
    # Test texts
    test_texts = [
        "Hello, how are you?",
        "তুমি কেমন আছো?",  # Bengali
        "This is a test message for content moderation.",
        "তুমি একটা বোকা এবং আমি তোমাকে ঘৃণা করি।"  # Bengali violation text
    ]
    
    print("Testing Enhanced Translation Service")
    print("=" * 50)
    
    # Test language detection
    for text in test_texts:
        detection = translator.detect_language(text)
        print(f"\nText: {text[:50]}...")
        print(f"Detected: {detection.detected_language} (confidence: {detection.confidence:.3f})")
        
        # Test translation to English
        if detection.detected_language != 'en':
            translation = translator.translate_text(text, "en")
            print(f"Translation: {translation.translated_text}")
            print(f"Service used: {translation.service_used}")
            print(f"Success: {translation.success}")
    
    # Test service status
    print("\nService Status:")
    status = translator.get_service_status()
    for service, is_working in status.items():
        print(f"{service}: {'✓' if is_working else '✗'}")

if __name__ == "__main__":
    main()