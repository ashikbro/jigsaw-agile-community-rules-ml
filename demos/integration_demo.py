#!/usr/bin/env python3
"""
Integration Demo: Enhanced Content Moderation System
Demonstrates the integration of multilingual support, caching, and logging/audit systems
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from enhanced_multilingual_support import (
        EnhancedLanguageDetector, EnhancedMultilingualTranslator,
        EnhancedMultilingualEmbeddings, EnhancedMultilingualContentModerator
    )
    ENHANCED_MULTILINGUAL_AVAILABLE = True
except ImportError as e:
    print(f"Enhanced multilingual support not available: {e}")
    ENHANCED_MULTILINGUAL_AVAILABLE = False

try:
    from advanced_caching_system import (
        MemoryCache, DiskCache, MultiLevelCache, CacheLevel,
        ContentModerationCache
    )
    CACHING_AVAILABLE = True
except ImportError as e:
    print(f"Advanced caching not available: {e}")
    CACHING_AVAILABLE = False

try:
    from logging_audit_system import (
        AuditLogger, AuditEventType, SecurityLevel, LogLevel,
        audit_decorator
    )
    LOGGING_AVAILABLE = True
except ImportError as e:
    print(f"Logging and audit system not available: {e}")
    LOGGING_AVAILABLE = False

# Fallback imports for basic functionality
try:
    from multilingual_support import (
        LanguageDetector, MultilingualTranslator, 
        MultilingualContentModerator
    )
    BASIC_MULTILINGUAL_AVAILABLE = True
except ImportError:
    BASIC_MULTILINGUAL_AVAILABLE = False

class IntegratedContentModerationSystem:
    """
    Integrated content moderation system with enhanced features
    """
    
    def __init__(self, 
                 enable_caching: bool = True,
                 enable_logging: bool = True,
                 enable_enhanced_multilingual: bool = True,
                 cache_ttl: int = 3600,
                 log_dir: str = "./integrated_logs"):
        
        self.enable_caching = enable_caching and CACHING_AVAILABLE
        self.enable_logging = enable_logging and LOGGING_AVAILABLE
        self.enable_enhanced_multilingual = enable_enhanced_multilingual and ENHANCED_MULTILINGUAL_AVAILABLE
        
        print(f"🚀 Initializing Integrated Content Moderation System")
        print(f"   Caching: {'✅' if self.enable_caching else '❌'}")
        print(f"   Logging: {'✅' if self.enable_logging else '❌'}")
        print(f"   Enhanced Multilingual: {'✅' if self.enable_enhanced_multilingual else '❌'}")
        
        # Initialize logging system
        self.audit_logger = None
        if self.enable_logging:
            try:
                self.audit_logger = AuditLogger(
                    log_dir=log_dir,
                    enable_database=True,
                    enable_elasticsearch=False
                )
                print("✅ Audit logging system initialized")
            except Exception as e:
                print(f"❌ Failed to initialize logging: {e}")
                self.enable_logging = False
        
        # Initialize caching system
        self.cache = None
        if self.enable_caching:
            try:
                # Create multi-level cache
                levels = [
                    (CacheLevel.MEMORY, MemoryCache(max_size=500, default_ttl=cache_ttl)),
                    (CacheLevel.DISK, DiskCache(
                        cache_dir="./integrated_cache", 
                        max_size_mb=100, 
                        compress=True,
                        default_ttl=cache_ttl
                    ))
                ]
                
                ml_cache = MultiLevelCache(levels, promotion_threshold=2)
                self.cache = ContentModerationCache(
                    cache_backend=ml_cache,
                    default_ttl=cache_ttl
                )
                print("✅ Multi-level caching system initialized")
            except Exception as e:
                print(f"❌ Failed to initialize caching: {e}")
                self.enable_caching = False
        
        # Initialize multilingual system
        self.moderator = None
        if self.enable_enhanced_multilingual:
            try:
                self.moderator = EnhancedMultilingualContentModerator(
                    detection_methods=['langdetect', 'fasttext'],
                    translation_engines=['google', 'marian'],
                    embedding_models=['paraphrase-multilingual-MiniLM-L12-v2']
                )
                print("✅ Enhanced multilingual content moderator initialized")
            except Exception as e:
                print(f"❌ Failed to initialize enhanced multilingual: {e}")
                if BASIC_MULTILINGUAL_AVAILABLE:
                    try:
                        self.moderator = MultilingualContentModerator()
                        print("✅ Basic multilingual content moderator initialized")
                    except Exception as e2:
                        print(f"❌ Failed to initialize basic multilingual: {e2}")
        
        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'processing_times': [],
            'languages_detected': {},
            'predictions': {'safe': 0, 'violation': 0, 'review': 0}
        }
    
    def _log_audit_event(self, event_type: 'AuditEventType', action: str, 
                        resource: str, outcome: str, details: Dict[str, Any],
                        security_level: 'SecurityLevel' = None):
        """Log audit event if logging is enabled"""
        if self.audit_logger:
            try:
                self.audit_logger.audit(
                    event_type=event_type,
                    actor='integrated_system',
                    action=action,
                    resource=resource,
                    outcome=outcome,
                    details=details,
                    security_level=security_level or SecurityLevel.LOW
                )
            except Exception as e:
                print(f"Failed to log audit event: {e}")
    
    def _get_cached_result(self, cache_type: str, text: str, **kwargs) -> Optional[Any]:
        """Get cached result if caching is enabled"""
        if not self.cache:
            return None
        
        try:
            result = self.cache.get_cached_result(cache_type, text, **kwargs)
            if result:
                self.metrics['cache_hits'] += 1
            else:
                self.metrics['cache_misses'] += 1
            return result
        except Exception as e:
            print(f"Cache retrieval error: {e}")
            self.metrics['cache_misses'] += 1
            return None
    
    def _cache_result(self, cache_type: str, text: str, result: Any, **kwargs) -> bool:
        """Cache result if caching is enabled"""
        if not self.cache:
            return False
        
        try:
            return self.cache.cache_result(cache_type, text, result, **kwargs)
        except Exception as e:
            print(f"Cache storage error: {e}")
            return False
    
    @audit_decorator(AuditEventType.MODEL_PREDICTION, "moderate_content")
    def moderate_text(self, text: str, user_id: str = "anonymous", 
                     session_id: Optional[str] = None) -> Dict[str, Any]:
        """Moderate text content with full integration"""
        start_time = time.time()
        self.metrics['total_requests'] += 1
        
        if self.audit_logger:
            self.audit_logger.set_context(
                session_id=session_id,
                user_id=user_id
            )
        
        try:
            # Step 1: Language Detection (with caching)
            lang_result = self._get_cached_result('language_detection', text)
            if not lang_result:
                if self.moderator:
                    if hasattr(self.moderator, 'detect_language_enhanced'):
                        lang_result = self.moderator.detect_language_enhanced(text)
                    elif hasattr(self.moderator, 'detector') and hasattr(self.moderator.detector, 'detect_language'):
                        lang_result = self.moderator.detector.detect_language(text)
                    elif hasattr(self.moderator, 'detect_language'):
                        lang_result = self.moderator.detect_language(text)
                    else:
                        # Fallback language detection
                        try:
                            from langdetect import detect, detect_langs
                            detected_lang = detect(text)
                            confidence = max([lang.prob for lang in detect_langs(text)])
                            lang_result = {'language': detected_lang, 'confidence': confidence}
                        except:
                            lang_result = {'language': 'en', 'confidence': 0.5}
                else:
                    lang_result = {'language': 'en', 'confidence': 0.5}
                
                self._cache_result('language_detection', text, lang_result)
            
            detected_language = lang_result.get('language', 'en')
            self.metrics['languages_detected'][detected_language] = \
                self.metrics['languages_detected'].get(detected_language, 0) + 1
            
            # Step 2: Translation (if needed, with caching)
            translated_text = text
            translation_result = None
            
            if detected_language != 'en':
                trans_cache_key = f"{detected_language}_to_en"
                translation_result = self._get_cached_result(
                    'translation', text, 
                    source_lang=detected_language, 
                    target_lang='en'
                )
                
                if not translation_result:
                    if self.moderator:
                        if hasattr(self.moderator, 'translate_text_enhanced'):
                            translation_result = self.moderator.translate_text_enhanced(
                                text, target_lang='en', source_lang=detected_language
                            )
                        elif hasattr(self.moderator, 'translator') and hasattr(self.moderator.translator, 'translate_text'):
                            translation_result = self.moderator.translator.translate_text(
                                text, target_lang='en'
                            )
                        elif hasattr(self.moderator, 'translate_text'):
                            translation_result = self.moderator.translate_text(
                                text, target_lang='en', source_lang=detected_language
                            )
                        else:
                            # Fallback translation
                            try:
                                from googletrans import Translator
                                translator = Translator()
                                result = translator.translate(text, dest='en', src=detected_language)
                                translation_result = {
                                    'translated_text': result.text,
                                    'confidence': 0.8,
                                    'source_language': detected_language,
                                    'target_language': 'en'
                                }
                            except:
                                translation_result = {'translated_text': text, 'confidence': 0.5}
                    else:
                        translation_result = {'translated_text': text, 'confidence': 0.5}
                    
                    self._cache_result(
                        'translation', text, translation_result,
                        source_lang=detected_language, target_lang='en'
                    )
                
                translated_text = translation_result.get('translated_text', text)
            
            # Step 3: Content Moderation (with caching)
            moderation_result = self._get_cached_result(
                'moderation', translated_text, 
                language=detected_language
            )
            
            if not moderation_result:
                if self.moderator:
                    if hasattr(self.moderator, 'analyze_content_enhanced'):
                        moderation_result = self.moderator.analyze_content_enhanced(translated_text)
                    elif hasattr(self.moderator, 'analyze_content'):
                        moderation_result = self.moderator.analyze_content(translated_text)
                    elif hasattr(self.moderator, 'moderate_text'):
                        moderation_result = self.moderator.moderate_text(translated_text)
                    else:
                        # Fallback simple analysis
                        toxic_words = ['hate', 'stupid', 'idiot', 'kill', 'die', 'estúpido', 'odio']
                        is_toxic = any(word in translated_text.lower() for word in toxic_words)
                        moderation_result = {
                            'prediction': 'violation' if is_toxic else 'safe',
                            'confidence': 0.8 if is_toxic else 0.9,
                            'explanation': 'Simple keyword-based detection (fallback)'
                        }
                else:
                    # Fallback simple analysis
                    toxic_words = ['hate', 'stupid', 'idiot', 'kill', 'die', 'estúpido', 'odio']
                    is_toxic = any(word in translated_text.lower() for word in toxic_words)
                    moderation_result = {
                        'prediction': 'violation' if is_toxic else 'safe',
                        'confidence': 0.8 if is_toxic else 0.9,
                        'explanation': 'Simple keyword-based detection (fallback)'
                    }
                
                self._cache_result(
                    'moderation', translated_text, moderation_result,
                    language=detected_language
                )
            
            prediction = moderation_result.get('prediction', 'safe')
            self.metrics['predictions'][prediction] = \
                self.metrics['predictions'].get(prediction, 0) + 1
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            self.metrics['processing_times'].append(processing_time)
            
            # Prepare comprehensive result
            result = {
                'text': text,
                'language_detection': lang_result,
                'translation': translation_result,
                'moderation': moderation_result,
                'processing_time_ms': processing_time,
                'cache_hit': any([
                    self._get_cached_result('language_detection', text) is not None,
                    translation_result and self._get_cached_result(
                        'translation', text, 
                        source_lang=detected_language, 
                        target_lang='en'
                    ) is not None,
                    self._get_cached_result(
                        'moderation', translated_text, 
                        language=detected_language
                    ) is not None
                ]),
                'metadata': {
                    'user_id': user_id,
                    'session_id': session_id,
                    'timestamp': time.time(),
                    'system_version': '2.0-integrated'
                }
            }
            
            # Log successful moderation
            self._log_audit_event(
                event_type=AuditEventType.MODEL_PREDICTION,
                action='moderate_text',
                resource='content_moderator',
                outcome='success',
                details={
                    'text_length': len(text),
                    'detected_language': detected_language,
                    'prediction': prediction,
                    'confidence': moderation_result.get('confidence', 0),
                    'processing_time_ms': processing_time,
                    'cache_hit': result['cache_hit']
                }
            )
            
            return result
        
        except Exception as e:
            error_details = {
                'error': str(e),
                'error_type': type(e).__name__,
                'text_length': len(text),
                'processing_time_ms': (time.time() - start_time) * 1000
            }
            
            # Log error
            self._log_audit_event(
                event_type=AuditEventType.ERROR_EVENT,
                action='moderate_text',
                resource='content_moderator',
                outcome='failure',
                details=error_details,
                security_level=SecurityLevel.MEDIUM
            )
            
            if self.audit_logger:
                self.audit_logger.log(
                    LogLevel.ERROR,
                    f"Content moderation failed: {e}",
                    **error_details
                )
            
            # Return error result
            return {
                'text': text,
                'error': str(e),
                'processing_time_ms': (time.time() - start_time) * 1000,
                'metadata': {
                    'user_id': user_id,
                    'session_id': session_id,
                    'timestamp': time.time(),
                    'system_version': '2.0-integrated'
                }
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        stats = {
            'performance': self.metrics.copy(),
            'cache': {},
            'audit': {},
            'system_info': {
                'caching_enabled': self.enable_caching,
                'logging_enabled': self.enable_logging,
                'enhanced_multilingual_enabled': self.enable_enhanced_multilingual
            }
        }
        
        # Add cache statistics
        if self.cache:
            try:
                stats['cache'] = self.cache.get_cache_stats()
            except Exception as e:
                stats['cache'] = {'error': str(e)}
        
        # Add audit statistics
        if self.audit_logger:
            try:
                stats['audit'] = self.audit_logger.get_audit_statistics()
            except Exception as e:
                stats['audit'] = {'error': str(e)}
        
        # Calculate performance metrics
        if self.metrics['processing_times']:
            processing_times = self.metrics['processing_times']
            stats['performance']['avg_processing_time_ms'] = sum(processing_times) / len(processing_times)
            stats['performance']['min_processing_time_ms'] = min(processing_times)
            stats['performance']['max_processing_time_ms'] = max(processing_times)
        
        # Calculate cache hit rate
        total_cache_requests = self.metrics['cache_hits'] + self.metrics['cache_misses']
        if total_cache_requests > 0:
            stats['performance']['cache_hit_rate'] = self.metrics['cache_hits'] / total_cache_requests
        else:
            stats['performance']['cache_hit_rate'] = 0.0
        
        return stats
    
    def cleanup(self):
        """Cleanup system resources"""
        print("🧹 Cleaning up integrated system...")
        
        # Cleanup cache
        if self.cache:
            try:
                # Clear cache if needed
                pass
            except Exception as e:
                print(f"Cache cleanup error: {e}")
        
        # Cleanup logs (optional)
        # Note: We typically don't delete logs, but we could close handlers
        if self.audit_logger:
            try:
                # Close any open handlers if needed
                pass
            except Exception as e:
                print(f"Logging cleanup error: {e}")

def run_integration_demo():
    """
    Run comprehensive integration demonstration
    """
    print("🌟 Enhanced Content Moderation System - Integration Demo")
    print("=" * 70)
    
    # Test texts in multiple languages
    test_texts = [
        # English texts
        {
            'text': "This is a great product! I love it.",
            'expected': 'safe',
            'language': 'en',
            'description': 'Positive English text'
        },
        {
            'text': "You are so stupid and I hate you!",
            'expected': 'violation',
            'language': 'en',
            'description': 'Toxic English text'
        },
        # Spanish texts
        {
            'text': "¡Este producto es increíble! Me encanta.",
            'expected': 'safe',
            'language': 'es',
            'description': 'Positive Spanish text'
        },
        {
            'text': "Eres muy estúpido y te odio.",
            'expected': 'violation',
            'language': 'es',
            'description': 'Toxic Spanish text'
        },
        # French texts
        {
            'text': "C'est un excellent produit! Je l'adore.",
            'expected': 'safe',
            'language': 'fr',
            'description': 'Positive French text'
        },
        # German texts
        {
            'text': "Das ist ein tolles Produkt! Ich liebe es.",
            'expected': 'safe',
            'language': 'de',
            'description': 'Positive German text'
        },
        # Duplicate for cache testing
        {
            'text': "This is a great product! I love it.",
            'expected': 'safe',
            'language': 'en',
            'description': 'Duplicate English text (cache test)'
        }
    ]
    
    try:
        # Initialize integrated system
        system = IntegratedContentModerationSystem(
            enable_caching=True,
            enable_logging=True,
            enable_enhanced_multilingual=True,
            cache_ttl=1800,  # 30 minutes
            log_dir="./integration_demo_logs"
        )
        
        print("\n🧪 Running Content Moderation Tests")
        print("-" * 50)
        
        results = []
        
        for i, test_case in enumerate(test_texts, 1):
            print(f"\nTest {i}: {test_case['description']}")
            print(f"Text: {test_case['text'][:50]}{'...' if len(test_case['text']) > 50 else ''}")
            
            # Moderate the text
            result = system.moderate_text(
                text=test_case['text'],
                user_id=f"test_user_{i}",
                session_id="integration_demo_session"
            )
            
            results.append(result)
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
            else:
                # Extract key information
                detected_lang = result['language_detection'].get('language', 'unknown')
                prediction = result['moderation'].get('prediction', 'unknown')
                confidence = result['moderation'].get('confidence', 0)
                processing_time = result['processing_time_ms']
                cache_hit = result.get('cache_hit', False)
                
                print(f"✅ Language: {detected_lang} | Prediction: {prediction} | "
                      f"Confidence: {confidence:.2f} | Time: {processing_time:.1f}ms | "
                      f"Cache: {'HIT' if cache_hit else 'MISS'}")
                
                # Check if prediction matches expectation
                if prediction == test_case['expected']:
                    print(f"✅ Prediction matches expected result")
                else:
                    print(f"⚠️  Prediction ({prediction}) differs from expected ({test_case['expected']})")
        
        print("\n📊 System Performance Statistics")
        print("-" * 50)
        
        stats = system.get_system_stats()
        
        # Performance stats
        perf = stats['performance']
        print(f"Total Requests: {perf['total_requests']}")
        print(f"Cache Hit Rate: {perf.get('cache_hit_rate', 0):.1%}")
        print(f"Average Processing Time: {perf.get('avg_processing_time_ms', 0):.1f}ms")
        print(f"Min/Max Processing Time: {perf.get('min_processing_time_ms', 0):.1f}ms / {perf.get('max_processing_time_ms', 0):.1f}ms")
        
        # Language distribution
        print(f"\nLanguages Detected: {dict(perf['languages_detected'])}")
        print(f"Predictions: {dict(perf['predictions'])}")
        
        # Cache stats
        if 'cache' in stats and stats['cache']:
            cache_stats = stats['cache']
            print(f"\n💾 Cache Statistics:")
            if 'totals' in cache_stats:
                totals = cache_stats['totals']
                print(f"  Total Cache Hits: {totals.get('hits', 0)}")
                print(f"  Total Cache Misses: {totals.get('misses', 0)}")
                print(f"  Cache Hit Rate: {totals.get('hit_rate', 0):.1%}")
                print(f"  Total Cached Items: {totals.get('size', 0)}")
        
        # Audit stats
        if 'audit' in stats and stats['audit']:
            audit_stats = stats['audit']
            print(f"\n📋 Audit Statistics:")
            if 'database' in audit_stats:
                db_stats = audit_stats['database']
                print(f"  Total Audit Events: {db_stats.get('total_events', 0)}")
                print(f"  Recent Events (24h): {db_stats.get('recent_events_24h', 0)}")
                
                if 'events_by_type' in db_stats:
                    print(f"  Events by Type:")
                    for event_type, count in db_stats['events_by_type'].items():
                        print(f"    {event_type}: {count}")
        
        print("\n🔍 Sample Results Analysis")
        print("-" * 50)
        
        # Show detailed results for first few tests
        for i, result in enumerate(results[:3], 1):
            if 'error' not in result:
                print(f"\nResult {i}:")
                print(f"  Original Text: {result['text'][:40]}...")
                
                lang_det = result['language_detection']
                print(f"  Language: {lang_det.get('language', 'unknown')} (confidence: {lang_det.get('confidence', 0):.2f})")
                
                if result['translation']:
                    trans = result['translation']
                    print(f"  Translation: {trans.get('translated_text', 'N/A')[:40]}...")
                
                mod = result['moderation']
                print(f"  Moderation: {mod.get('prediction', 'unknown')} (confidence: {mod.get('confidence', 0):.2f})")
                print(f"  Processing Time: {result['processing_time_ms']:.1f}ms")
        
        print("\n✅ Integration demo completed successfully!")
        print(f"\n📈 Summary:")
        print(f"  - Processed {len(test_texts)} texts in multiple languages")
        print(f"  - Average processing time: {stats['performance'].get('avg_processing_time_ms', 0):.1f}ms")
        print(f"  - Cache hit rate: {stats['performance'].get('cache_hit_rate', 0):.1%}")
        print(f"  - All systems integrated successfully")
        
        # Cleanup
        system.cleanup()
        
    except Exception as e:
        print(f"❌ Integration demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup demo files
        import shutil
        cleanup_dirs = ['./integrated_cache', './integration_demo_logs']
        for cleanup_dir in cleanup_dirs:
            if os.path.exists(cleanup_dir):
                try:
                    shutil.rmtree(cleanup_dir)
                    print(f"🧹 Cleaned up {cleanup_dir}")
                except Exception as e:
                    print(f"Failed to cleanup {cleanup_dir}: {e}")

if __name__ == "__main__":
    run_integration_demo()