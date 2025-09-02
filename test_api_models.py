#!/usr/bin/env python3
"""
Test script to verify which models the API is actually using
"""

import requests
import json

API_BASE_URL = "http://localhost:8000"

def test_model_endpoints():
    """Test different model endpoints to see their behavior"""
    
    test_texts = [
        "This is a great discussion!",
        "You're an idiot!",
        "Thank you for your help",
        "I hate you so much"
    ]
    
    print("Testing Regular Endpoint:")
    print("=" * 50)
    
    for text in test_texts:
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json={"text": text},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Text: '{text}'")
                print(f"Prediction: {result['prediction']} (Prob: {result['confidence']:.3f})")
                print(f"Model: {result.get('model_used', 'unknown')}")
                print()
            else:
                print(f"Error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"Request failed: {e}")
    
    print("\nTesting Multilingual Endpoint:")
    print("=" * 50)
    
    for text in test_texts:
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict/multilingual",
                json={"text": text},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Text: '{text}'")
                print(f"Prediction: {result['prediction']} (Prob: {result['confidence']:.3f})")
                print(f"Model: {result.get('model_used', 'unknown')}")
                print(f"Language: {result.get('detected_language', 'unknown')}")
                print()
            else:
                print(f"Error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"Request failed: {e}")

def test_model_info():
    """Get information about available models"""
    try:
        response = requests.get(f"{API_BASE_URL}/models", timeout=10)
        if response.status_code == 200:
            models = response.json()
            print("\nAvailable Models:")
            print("=" * 50)
            for model_name, info in models.items():
                print(f"Model: {model_name}")
                print(f"  Type: {info.get('type', 'unknown')}")
                print(f"  Algorithm: {info.get('algorithm', 'unknown')}")
                print(f"  Path: {info.get('path', 'unknown')}")
                print()
        else:
            print(f"Error getting model info: {response.status_code}")
    except Exception as e:
        print(f"Failed to get model info: {e}")

if __name__ == "__main__":
    test_model_info()
    test_model_endpoints()