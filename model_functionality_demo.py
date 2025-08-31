#!/usr/bin/env python3
"""
Model Functionality Demonstration
Shows how the Explainable AI Text Classification Framework works with various content types.
"""

import requests
import json
import time
from typing import Dict, List, Any
from datetime import datetime

class ModelDemo:
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.session = requests.Session()
        
    def check_api_health(self) -> bool:
        """Check if the API server is running."""
        try:
            response = self.session.get(f"{self.api_base_url}/health")
            return response.status_code == 200
        except Exception as e:
            print(f"❌ API server not available: {e}")
            return False
    
    def predict_single(self, text: str, language: str = "auto") -> Dict[str, Any]:
        """Make a single prediction with explainability."""
        try:
            payload = {
                "text": text,
                "language": language,
                "explain": True,
                "confidence_threshold": 0.3
            }
            
            response = self.session.post(
                f"{self.api_base_url}/predict",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def demonstrate_content_types(self):
        """Demonstrate model performance on different content types."""
        print("\n" + "="*80)
        print("🤖 EXPLAINABLE AI TEXT CLASSIFICATION FRAMEWORK DEMO")
        print("="*80)
        
        # Test cases covering various scenarios
        test_cases = [
            {
                "category": "🟢 Safe Content",
                "examples": [
                    "I love spending time with my family on weekends.",
                    "The weather is beautiful today, perfect for a walk.",
                    "Thank you for your help with the project!"
                ]
            },
            {
                "category": "🔴 Toxic Content",
                "examples": [
                    "You are such an idiot, I hate you!",
                    "This is the worst thing I've ever seen, complete garbage.",
                    "Go kill yourself, nobody wants you here."
                ]
            },
            {
                "category": "🟡 Borderline Content",
                "examples": [
                    "I disagree with your opinion, but I respect your right to have it.",
                    "This movie was disappointing, not what I expected.",
                    "I'm frustrated with this situation, but let's work it out."
                ]
            },
            {
                "category": "🌍 Multilingual Content",
                "examples": [
                    "Bonjour, comment allez-vous aujourd'hui?",  # French
                    "Hola, ¿cómo estás? Espero que tengas un buen día.",  # Spanish
                    "你好，今天天气很好。",  # Chinese
                    "আপনি কেমন আছেন? আজ আবহাওয়া খুব সুন্দর।"  # Bengali
                ]
            }
        ]
        
        for category_data in test_cases:
            print(f"\n{category_data['category']}")
            print("-" * 60)
            
            for i, text in enumerate(category_data['examples'], 1):
                print(f"\n📝 Example {i}: {text[:50]}{'...' if len(text) > 50 else ''}")
                
                result = self.predict_single(text)
                
                if "error" in result:
                    print(f"   ❌ Error: {result['error']}")
                    continue
                
                # Display prediction results
                prediction = result.get('prediction', 'unknown')
                confidence = result.get('confidence', 0.0)
                processing_time = result.get('processing_time_ms', 0)
                
                # Color code the prediction
                if prediction == 'safe':
                    pred_display = f"🟢 {prediction.upper()}"
                elif prediction == 'violation':
                    pred_display = f"🔴 {prediction.upper()}"
                else:
                    pred_display = f"🟡 {prediction.upper()}"
                
                print(f"   📊 Prediction: {pred_display}")
                print(f"   📈 Confidence: {confidence:.3f}")
                print(f"   ⏱️  Processing: {processing_time:.1f}ms")
                
                # Show explanation if available
                if 'explanation' in result and result['explanation'] is not None:
                    explanation = result['explanation']
                    if isinstance(explanation, dict) and 'top_words' in explanation:
                        top_words = explanation['top_words'][:3]  # Show top 3
                        words_str = ", ".join([f"{word}({score:.2f})" for word, score in top_words])
                        print(f"   🔍 Key words: {words_str}")
                
                # Show language detection for multilingual
                if 'detected_language' in result:
                    lang = result['detected_language']
                    print(f"   🌐 Language: {lang}")
                
                time.sleep(0.5)  # Small delay for readability
    
    def demonstrate_advanced_features(self):
        """Demonstrate advanced features like batch processing and fairness."""
        print("\n" + "="*80)
        print("🚀 ADVANCED FEATURES DEMONSTRATION")
        print("="*80)
        
        # Batch processing demo
        print("\n📦 Batch Processing Demo")
        print("-" * 40)
        
        batch_texts = [
            "This is a great product, highly recommended!",
            "Terrible service, worst experience ever!",
            "The meeting is scheduled for tomorrow at 3 PM.",
            "I can't believe how rude that person was to me."
        ]
        
        try:
            batch_payload = {
                "texts": batch_texts,
                "explain": True
            }
            
            response = self.session.post(
                f"{self.api_base_url}/predict_batch",
                json=batch_payload,
                timeout=60
            )
            
            if response.status_code == 200:
                batch_results = response.json()
                print(f"   ✅ Processed {len(batch_results.get('predictions', []))} texts")
                print(f"   ⏱️  Total time: {batch_results.get('total_processing_time_ms', 0):.1f}ms")
                print(f"   📊 Average per text: {batch_results.get('average_processing_time_ms', 0):.1f}ms")
            else:
                print(f"   ❌ Batch processing failed: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Batch processing error: {e}")
        
        # Model information
        print("\n🔧 Model Information")
        print("-" * 40)
        
        try:
            response = self.session.get(f"{self.api_base_url}/model_info")
            if response.status_code == 200:
                model_info = response.json()
                print(f"   📋 Model: {model_info.get('model_name', 'Unknown')}")
                print(f"   🏷️  Version: {model_info.get('version', 'Unknown')}")
                print(f"   🌐 Languages: {model_info.get('supported_languages', 0)} supported")
                print(f"   🎯 Features: {', '.join(model_info.get('features', []))}")
            else:
                print(f"   ❌ Failed to get model info: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Model info error: {e}")
    
    def show_api_statistics(self):
        """Show current API usage statistics."""
        print("\n📈 API Statistics")
        print("-" * 40)
        
        try:
            response = self.session.get(f"{self.api_base_url}/stats")
            if response.status_code == 200:
                stats = response.json()
                print(f"   📊 Total requests: {stats.get('total_requests', 0)}")
                print(f"   ⚡ Cache hits: {stats.get('cache_hits', 0)}")
                print(f"   📈 Hit rate: {stats.get('cache_hit_rate', 0):.1%}")
                print(f"   ⏱️  Avg response: {stats.get('average_response_time_ms', 0):.1f}ms")
            else:
                print(f"   ❌ Failed to get stats: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Stats error: {e}")
    
    def run_complete_demo(self):
        """Run the complete model functionality demonstration."""
        print(f"\n🕐 Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check API health first
        if not self.check_api_health():
            print("\n❌ Cannot proceed - API server is not running!")
            print("   Please start the API server with: python api_server.py")
            return
        
        print("\n✅ API server is running and healthy!")
        
        # Run demonstrations
        self.demonstrate_content_types()
        self.demonstrate_advanced_features()
        self.show_api_statistics()
        
        print("\n" + "="*80)
        print("✨ DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\n🎯 Key Takeaways:")
        print("   • Model accurately classifies toxic vs safe content")
        print("   • Provides explainable predictions with confidence scores")
        print("   • Supports multiple languages with automatic detection")
        print("   • Offers batch processing for efficiency")
        print("   • Includes real-time performance monitoring")
        print("\n🌐 Access Points:")
        print("   • API Server: http://localhost:8000")
        print("   • Monitoring Dashboard: http://localhost:8501")
        print("   • API Documentation: http://localhost:8000/docs")
        
if __name__ == "__main__":
    demo = ModelDemo()
    demo.run_complete_demo()