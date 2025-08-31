#!/usr/bin/env python3
"""
Test script for multilingual content moderation API endpoints

This script demonstrates the multilingual capabilities of the content moderation API,
including language detection, translation, and content analysis in multiple languages.

Author: AI Assistant
Date: 2025
"""

import requests
import json
import time
from typing import Dict, Any

# API configuration
API_BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

def test_multilingual_prediction(text: str, target_language: str = "en", auto_translate: bool = True) -> Dict[str, Any]:
    """Test multilingual content moderation prediction."""
    url = f"{API_BASE_URL}/predict/multilingual"
    
    payload = {
        "text": text,
        "target_language": target_language,
        "auto_translate": auto_translate,
        "model_type": "auto",
        "include_explanation": True,
        "include_probabilities": True
    }
    
    try:
        response = requests.post(url, json=payload, headers=HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        return {"error": str(e)}

def test_multilingual_stats() -> Dict[str, Any]:
    """Test multilingual statistics endpoint."""
    url = f"{API_BASE_URL}/multilingual/stats"
    
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Stats request failed: {e}")
        return {"error": str(e)}

def test_api_health() -> Dict[str, Any]:
    """Test API health endpoint."""
    url = f"{API_BASE_URL}/health"
    
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return {"error": str(e)}

def print_prediction_result(result: Dict[str, Any], test_name: str):
    """Pretty print prediction results."""
    print(f"\n--- {test_name} ---")
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print(f"📝 Original: {result.get('original_text', 'N/A')}")
    print(f"🌍 Language: {result.get('detected_language', 'N/A')} (confidence: {result.get('language_confidence', 0):.2f})")
    
    if result.get('translated_text'):
        print(f"🔄 Translated: {result['translated_text']}")
    
    print(f"🎯 Prediction: {result.get('prediction', 'N/A')} (confidence: {result.get('confidence', 0):.2f})")
    
    if result.get('probabilities'):
        probs = result['probabilities']
        print(f"📊 Probabilities: Safe={probs.get('safe', 0):.3f}, Violation={probs.get('violation', 0):.3f}")
    
    print(f"⚙️ Model: {result.get('model_used', 'N/A')}")
    print(f"⏱️ Processing time: {result.get('processing_time_ms', 0):.1f}ms")

def main():
    """Main test function."""
    print("🌍 Testing Multilingual Content Moderation API")
    print("=" * 50)
    
    # Test API health first
    print("\n🏥 Testing API Health...")
    health = test_api_health()
    if "error" not in health:
        print(f"✅ API Status: {health.get('status', 'unknown')}")
        print(f"📊 Models loaded: {len(health.get('models_loaded', {}))}")
    else:
        print(f"❌ API Health check failed: {health['error']}")
        return
    
    # Test multilingual predictions
    test_cases = [
        {
            "text": "This is a great product! I love it.",
            "name": "English Text (Positive)",
            "target_lang": "en"
        },
        {
            "text": "Este producto es terrible, no lo recomiendo para nada.",
            "name": "Spanish Text (Negative)",
            "target_lang": "en"
        },
        {
            "text": "C'est un produit formidable! Je le recommande vivement.",
            "name": "French Text (Positive)",
            "target_lang": "en"
        },
        {
            "text": "Das ist ein schreckliches Produkt. Totaler Müll!",
            "name": "German Text (Negative)",
            "target_lang": "en"
        },
        {
            "text": "这个产品很棒！我非常喜欢它。",
            "name": "Chinese Text (Positive)",
            "target_lang": "en"
        },
        {
            "text": "You are stupid and worthless!",
            "name": "English Toxic Content",
            "target_lang": "en"
        }
    ]
    
    print("\n🧪 Testing Multilingual Predictions...")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] {test_case['name']}")
        
        result = test_multilingual_prediction(
            text=test_case['text'],
            target_language=test_case['target_lang'],
            auto_translate=True
        )
        
        print_prediction_result(result, test_case['name'])
        
        # Small delay between requests
        time.sleep(0.5)
    
    # Test multilingual statistics
    print("\n\n📈 Testing Multilingual Statistics...")
    stats = test_multilingual_stats()
    
    if "error" not in stats:
        print(f"✅ Total texts processed: {stats.get('total_texts_processed', 0)}")
        print(f"🌍 Supported languages: {len(stats.get('supported_languages', []))}")
        print(f"🔄 Translation enabled: {stats.get('translation_enabled', False)}")
        
        lang_dist = stats.get('language_distribution', {})
        if lang_dist:
            print("\n📊 Language Distribution:")
            for lang, data in lang_dist.items():
                count = data.get('count', 0) if isinstance(data, dict) else data
                print(f"   {lang}: {count} texts")
    else:
        print(f"❌ Stats request failed: {stats['error']}")
    
    print("\n" + "=" * 50)
    print("🎉 Multilingual API testing completed!")
    print("\n💡 Next steps:")
    print("   • Visit http://localhost:8000/docs for interactive API documentation")
    print("   • Test more languages and content types")
    print("   • Integrate with your applications")

if __name__ == "__main__":
    main()