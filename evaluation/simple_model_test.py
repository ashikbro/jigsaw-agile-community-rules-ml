#!/usr/bin/env python3
"""
Simple test to verify real dataset models are working
"""

import requests
import json
from datetime import datetime

def test_api_health():
    """Test if API is responding"""
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def test_model_prediction(text, model_type="simple_model"):
    """Test model prediction"""
    try:
        payload = {
            "text": text,
            "model_type": model_type
        }
        response = requests.post("http://127.0.0.1:8000/predict", 
                               json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def main():
    print("🚀 Testing Real Dataset Models")
    print("=" * 40)
    
    # Test API health
    print("\n📡 Testing API health...")
    if test_api_health():
        print("✅ API is responding")
    else:
        print("❌ API is not responding")
        return
    
    # Test samples
    test_samples = [
        "This movie is absolutely fantastic!",
        "I hate this stupid product",
        "The weather is nice today"
    ]
    
    # Test models
    models_to_test = ["simple_model", "logistic_regression_real", "random_forest_real"]
    
    results = []
    
    for model in models_to_test:
        print(f"\n🔍 Testing {model}")
        print("-" * 30)
        
        model_results = []
        for i, text in enumerate(test_samples, 1):
            result = test_model_prediction(text, model)
            if "error" not in result:
                prediction = result.get("prediction", "unknown")
                confidence = result.get("confidence", 0)
                print(f"  {i}. '{text[:30]}...' → {prediction} ({confidence:.3f})")
                model_results.append({"text": text, "prediction": prediction, "confidence": confidence})
            else:
                print(f"  {i}. Error: {result['error']}")
                model_results.append({"text": text, "error": result['error']})
        
        results.append({"model": model, "results": model_results})
    
    # Summary
    print("\n📊 SUMMARY")
    print("=" * 20)
    
    for model_result in results:
        model = model_result["model"]
        successful = len([r for r in model_result["results"] if "error" not in r])
        total = len(model_result["results"])
        print(f"  {model}: {successful}/{total} successful predictions")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f"simple_test_report_{timestamp}.json"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": timestamp,
            "api_health": True,
            "results": results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {report_file}")
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    main()