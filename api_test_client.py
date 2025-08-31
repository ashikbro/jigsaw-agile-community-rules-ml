#!/usr/bin/env python3
"""
Content Moderation API Test Client

A simple test client to demonstrate the REST API functionality.
Shows how to:
- Make single predictions
- Process batch requests
- Get model information
- Check API health
- Use explainable AI features

Author: AI Assistant
Date: 2025
"""

import requests
import json
import time
from typing import List, Dict, Any

class ContentModerationClient:
    """Client for Content Moderation API."""
    
    def __init__(self, base_url: str = "http://localhost:8000", auth_token: str = None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        if auth_token:
            self.session.headers.update({"Authorization": f"Bearer {auth_token}"})
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_models_info(self) -> Dict[str, Any]:
        """Get information about available models."""
        response = self.session.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()
    
    def predict_text(self, 
                    text: str, 
                    model_type: str = "auto",
                    include_explanation: bool = False,
                    include_probabilities: bool = True) -> Dict[str, Any]:
        """Predict if text violates content rules."""
        payload = {
            "text": text,
            "model_type": model_type,
            "include_explanation": include_explanation,
            "include_probabilities": include_probabilities
        }
        
        response = self.session.post(f"{self.base_url}/predict", json=payload)
        response.raise_for_status()
        return response.json()
    
    def predict_batch(self, 
                     texts: List[str], 
                     model_type: str = "auto",
                     include_explanation: bool = False,
                     include_probabilities: bool = True) -> Dict[str, Any]:
        """Predict multiple texts in batch."""
        payload = {
            "texts": texts,
            "model_type": model_type,
            "include_explanation": include_explanation,
            "include_probabilities": include_probabilities
        }
        
        response = self.session.post(f"{self.base_url}/predict/batch", json=payload)
        response.raise_for_status()
        return response.json()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        response = self.session.get(f"{self.base_url}/stats")
        response.raise_for_status()
        return response.json()

def print_prediction_result(result: Dict[str, Any], title: str = "Prediction Result"):
    """Pretty print prediction results."""
    print(f"\n=== {title} ===")
    print(f"Text: {result['text'][:100]}{'...' if len(result['text']) > 100 else ''}")
    print(f"Prediction: {result['prediction'].upper()}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Model Used: {result['model_used']}")
    print(f"Processing Time: {result['processing_time_ms']:.1f}ms")
    
    if 'probabilities' in result:
        probs = result['probabilities']
        print(f"Probabilities: Safe={probs['safe']:.1%}, Violation={probs['violation']:.1%}")
    
    if 'explanation' in result and result['explanation']:
        exp = result['explanation']
        if 'error' not in exp:
            print("\n--- Explanation ---")
            if 'lime_explanation' in exp:
                lime = exp['lime_explanation']
                print(f"LIME Score: {lime.get('prediction_score', 'N/A')}")
                if 'top_features' in lime:
                    print("Top Features:")
                    for feature, score in lime['top_features'][:3]:
                        print(f"  {feature}: {score:.3f}")
            
            if 'text_analysis' in exp:
                analysis = exp['text_analysis']
                print(f"Word Count: {analysis.get('word_count', 'N/A')}")
                print(f"Threat Words: {analysis.get('threat_words', 'N/A')}")
                print(f"Person References: {analysis.get('person_references', 'N/A')}")
        else:
            print(f"Explanation Error: {exp['error']}")

def main():
    """Main test function."""
    print("Content Moderation API Test Client")
    print("=" * 40)
    
    # Initialize client
    client = ContentModerationClient()
    
    try:
        # Test 1: Health Check
        print("\n1. Testing Health Check...")
        health = client.health_check()
        print(f"Status: {health['status']}")
        print(f"Models Loaded: {list(health['models_loaded'].keys())}")
        print(f"Uptime: {health['uptime_seconds']:.1f}s")
        
        # Test 2: Model Information
        print("\n2. Getting Model Information...")
        models_info = client.get_models_info()
        print(f"Default Model: {models_info['default_model']}")
        print(f"Explainable AI Available: {models_info['explainable_ai_available']}")
        print(f"Available Models: {list(models_info['available_models'].keys())}")
        
        # Test 3: Single Predictions
        print("\n3. Testing Single Predictions...")
        
        test_texts = [
            "This is a great product, I love it!",
            "I want to kill the person who made this food",
            "You are stupid and should die",
            "The weather is nice today",
            "I hate this stupid bug in the code"
        ]
        
        for i, text in enumerate(test_texts[:3], 1):
            result = client.predict_text(
                text=text,
                include_explanation=True,
                include_probabilities=True
            )
            print_prediction_result(result, f"Test {i}")
        
        # Test 4: Batch Prediction
        print("\n4. Testing Batch Prediction...")
        batch_result = client.predict_batch(
            texts=test_texts,
            include_probabilities=True
        )
        
        print(f"\n=== Batch Results ===")
        print(f"Total Processed: {batch_result['total_processed']}")
        print(f"Total Time: {batch_result['total_time_ms']:.1f}ms")
        print(f"Average Time per Text: {batch_result['total_time_ms']/len(test_texts):.1f}ms")
        
        print("\nBatch Results Summary:")
        for i, result in enumerate(batch_result['results'], 1):
            prediction = result['prediction']
            confidence = result['confidence']
            text_preview = result['text'][:50] + "..." if len(result['text']) > 50 else result['text']
            print(f"  {i}. {prediction.upper()} ({confidence:.1%}) - {text_preview}")
        
        # Test 5: Performance Test
        print("\n5. Performance Test...")
        start_time = time.time()
        
        for _ in range(10):
            client.predict_text("This is a test message for performance testing")
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10 * 1000
        print(f"Average response time (10 requests): {avg_time:.1f}ms")
        
        # Test 6: API Statistics
        print("\n6. Getting API Statistics...")
        stats = client.get_stats()
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Active Clients: {stats['active_clients']}")
        print(f"Models Loaded: {stats['models_loaded']}")
        print(f"Explainers Available: {stats['explainers_available']}")
        
        print("\n✅ All tests completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API server.")
        print("Make sure the API server is running on http://localhost:8000")
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        print(f"Response: {e.response.text if e.response else 'No response'}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()