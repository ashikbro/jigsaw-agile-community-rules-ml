#!/usr/bin/env python3
"""
Comprehensive Model Performance Test
Tests the trained models with various types of content to evaluate their accuracy and behavior.
"""

import requests
import json
import time
from typing import List, Dict, Any

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Test cases with expected outcomes
TEST_CASES = [
    # Clear toxic content
    {
        "text": "You are such an idiot and I hate you!",
        "expected_toxic": True,
        "category": "clearly_toxic"
    },
    {
        "text": "Go kill yourself, nobody likes you anyway",
        "expected_toxic": True,
        "category": "clearly_toxic"
    },
    {
        "text": "This is complete garbage and you're stupid",
        "expected_toxic": True,
        "category": "clearly_toxic"
    },
    
    # Clear non-toxic content
    {
        "text": "I really enjoyed this movie, it was fantastic!",
        "expected_toxic": False,
        "category": "clearly_non_toxic"
    },
    {
        "text": "Thank you for your help, I appreciate it very much.",
        "expected_toxic": False,
        "category": "clearly_non_toxic"
    },
    {
        "text": "The weather is nice today, perfect for a walk.",
        "expected_toxic": False,
        "category": "clearly_non_toxic"
    },
    
    # Borderline cases (more challenging)
    {
        "text": "I disagree with your opinion strongly",
        "expected_toxic": False,
        "category": "borderline"
    },
    {
        "text": "This is annoying but I understand",
        "expected_toxic": False,
        "category": "borderline"
    },
    {
        "text": "You're wrong about this issue",
        "expected_toxic": False,
        "category": "borderline"
    },
    
    # Multilingual content
    {
        "text": "Eres un idiota",  # Spanish: "You are an idiot"
        "expected_toxic": True,
        "category": "multilingual_toxic"
    },
    {
        "text": "Muchas gracias por tu ayuda",  # Spanish: "Thank you very much for your help"
        "expected_toxic": False,
        "category": "multilingual_non_toxic"
    },
    {
        "text": "তুমি একটা বোকা",  # Bengali: "You are a fool"
        "expected_toxic": True,
        "category": "bengali_toxic"
    },
    {
        "text": "আপনাকে ধন্যবাদ",  # Bengali: "Thank you"
        "expected_toxic": False,
        "category": "bengali_non_toxic"
    }
]

def test_single_prediction(text: str) -> Dict[str, Any]:
    """Test a single text prediction"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"text": text},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def test_multilingual_prediction(text: str) -> Dict[str, Any]:
    """Test multilingual prediction"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/multilingual",
            json={"text": text},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def evaluate_prediction(result: Dict[str, Any], expected_toxic: bool, threshold: float = 0.5) -> bool:
    """Evaluate if prediction matches expected outcome"""
    if "error" in result:
        return False
    
    prediction_label = result.get("prediction", "safe")
    predicted_toxic = prediction_label == "violation"
    return predicted_toxic == expected_toxic

def run_performance_test():
    """Run comprehensive performance test"""
    print("=" * 60)
    print("COMPREHENSIVE MODEL PERFORMANCE TEST")
    print("=" * 60)
    
    # Test API health first
    try:
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        health_response.raise_for_status()
        print("✓ API Health Check: PASSED")
    except Exception as e:
        print(f"✗ API Health Check: FAILED - {e}")
        return
    
    results = {
        "total_tests": 0,
        "correct_predictions": 0,
        "by_category": {},
        "detailed_results": []
    }
    
    print("\nTesting individual predictions...")
    print("-" * 60)
    
    for i, test_case in enumerate(TEST_CASES, 1):
        text = test_case["text"]
        expected_toxic = test_case["expected_toxic"]
        category = test_case["category"]
        
        print(f"\nTest {i}/{len(TEST_CASES)} - Category: {category}")
        print(f"Text: {text[:50]}{'...' if len(text) > 50 else ''}")
        print(f"Expected: {'TOXIC' if expected_toxic else 'NON-TOXIC'}")
        
        # Test with regular endpoint
        result = test_single_prediction(text)
        
        if "error" in result:
            print(f"✗ ERROR: {result['error']}")
            continue
        
        prediction_label = result.get("prediction", "safe")
        confidence = float(result.get("confidence", 0.0))
        probabilities = result.get("probabilities", {})
        violation_prob = probabilities.get("violation", 0.0)
        predicted_toxic = prediction_label == "violation"
        
        is_correct = evaluate_prediction(result, expected_toxic)
        
        print(f"Prediction: {prediction_label} (Violation Prob: {violation_prob:.3f})")
        print(f"Confidence: {confidence:.3f}")
        print(f"Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
        
        # Update statistics
        results["total_tests"] += 1
        if is_correct:
            results["correct_predictions"] += 1
        
        if category not in results["by_category"]:
            results["by_category"][category] = {"total": 0, "correct": 0}
        
        results["by_category"][category]["total"] += 1
        if is_correct:
            results["by_category"][category]["correct"] += 1
        
        results["detailed_results"].append({
            "text": text,
            "category": category,
            "expected_toxic": expected_toxic,
            "prediction_label": prediction_label,
            "violation_probability": violation_prob,
            "confidence": confidence,
            "predicted_toxic": predicted_toxic,
            "correct": is_correct
        })
        
        # Small delay to avoid overwhelming the API
        time.sleep(0.1)
    
    # Print summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    overall_accuracy = (results["correct_predictions"] / results["total_tests"]) * 100
    print(f"Overall Accuracy: {overall_accuracy:.1f}% ({results['correct_predictions']}/{results['total_tests']})")
    
    print("\nAccuracy by Category:")
    for category, stats in results["by_category"].items():
        accuracy = (stats["correct"] / stats["total"]) * 100
        print(f"  {category}: {accuracy:.1f}% ({stats['correct']}/{stats['total']})")
    
    # Test multilingual endpoint with a few samples
    print("\n" + "-" * 60)
    print("MULTILINGUAL ENDPOINT TEST")
    print("-" * 60)
    
    multilingual_tests = [
        test for test in TEST_CASES 
        if test["category"] in ["multilingual_toxic", "multilingual_non_toxic", "bengali_toxic", "bengali_non_toxic"]
    ]
    
    for test_case in multilingual_tests[:3]:  # Test first 3 multilingual cases
        text = test_case["text"]
        print(f"\nTesting: {text}")
        
        ml_result = test_multilingual_prediction(text)
        if "error" not in ml_result:
            prediction_label = ml_result.get('prediction', 'safe')
            probabilities = ml_result.get('probabilities', {})
            violation_prob = probabilities.get('violation', 0.0)
            print(f"Prediction: {prediction_label} (Violation Prob: {violation_prob:.3f})")
            print(f"Detected Language: {ml_result.get('detected_language', 'unknown')}")
            print(f"Processing Time: {ml_result.get('processing_time_ms', 0):.1f}ms")
        else:
            print(f"Error: {ml_result['error']}")
    
    # Performance recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if overall_accuracy >= 85:
        print("✓ Model performance is EXCELLENT (≥85% accuracy)")
    elif overall_accuracy >= 75:
        print("⚠ Model performance is GOOD (75-84% accuracy)")
        print("  Consider fine-tuning for better results")
    elif overall_accuracy >= 65:
        print("⚠ Model performance is FAIR (65-74% accuracy)")
        print("  Recommend retraining with more data")
    else:
        print("✗ Model performance is POOR (<65% accuracy)")
        print("  Urgent: Review training data and model architecture")
    
    # Category-specific recommendations
    for category, stats in results["by_category"].items():
        accuracy = (stats["correct"] / stats["total"]) * 100
        if accuracy < 70:
            print(f"⚠ Low accuracy in {category}: Consider targeted improvements")
    
    return results

if __name__ == "__main__":
    results = run_performance_test()
    
    # Save detailed results
    with open("model_performance_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: model_performance_results.json")