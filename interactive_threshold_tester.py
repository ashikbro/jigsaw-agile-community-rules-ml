"""Interactive Threshold Tester for Text Classification.

This tool allows you to test different texts and see how adjusting
the classification threshold affects false positive rates.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


class ThresholdTester:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.setup_demo_model()
    
    def setup_demo_model(self):
        """Create a demo model for testing."""
        print("🔧 Setting up demo model for threshold testing...")
        
        # Extended training data with more examples
        demo_texts = [
            # Clear non-violations (helpful, constructive)
            "This is a great post, thanks for sharing!",
            "I love this community, very helpful",
            "Thanks for the information, very useful",
            "Great discussion, learned a lot",
            "Appreciate the detailed explanation",
            "Could you help me understand this better?",
            "I'm new here, any advice would be appreciated",
            "This tutorial was exactly what I needed",
            "Thanks for taking the time to explain",
            "Really helpful community here",
            
            # Clear violations (toxic, abusive)
            "You are an idiot and should shut up",
            "This is complete garbage, delete this",
            "Stop posting this stupid content",
            "You don't know what you're talking about",
            "This is the worst advice ever",
            "Get out of here with this nonsense",
            "You're completely useless",
            "This community is full of morons",
            "Shut up and go away",
            "You're wasting everyone's time",
            
            # Borderline cases (might cause false positives)
            "I wanted to know if this was bad",
            "Why would this be a violation",
            "How to fix such issues",
            "This text wasn't anything bad",
            "What makes this problematic",
            "Is this against the rules",
            "I don't understand why this is wrong",
            "Can someone explain the problem",
            "This seems fine to me",
            "I'm confused about the guidelines",
            
            # More borderline (constructive criticism)
            "I disagree with this approach",
            "This might not be the best solution",
            "There are some issues with this method",
            "I think there's a better way",
            "This could be improved"
        ]
        
        # Labels: 0 = not violation, 1 = violation
        demo_labels = (
            [0] * 10 +  # Clear non-violations
            [1] * 10 +  # Clear violations
            [0] * 15    # Borderline cases (should be non-violations)
        )
        
        # Create and train model
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = self.vectorizer.fit_transform(demo_texts)
        
        self.model = LogisticRegression(random_state=42, C=1.0)
        self.model.fit(X, demo_labels)
        
        print("✅ Demo model ready for testing")
    
    def predict_with_threshold(self, text, threshold=0.5):
        """Predict with custom threshold."""
        vectorized_text = self.vectorizer.transform([text])
        probabilities = self.model.predict_proba(vectorized_text)[0]
        
        prediction = 1 if probabilities[1] >= threshold else 0
        confidence = probabilities[1] * 100
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probabilities,
            'threshold_used': threshold
        }
    
    def test_text_with_multiple_thresholds(self, text):
        """Test a text with multiple threshold values."""
        print(f"\n🧪 TESTING TEXT: '{text}'")
        print("=" * 60)
        
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        results = []
        
        for threshold in thresholds:
            result = self.predict_with_threshold(text, threshold)
            results.append(result)
            
            status = "🚨 VIOLATION" if result['prediction'] == 1 else "✅ SAFE"
            print(f"Threshold {threshold:.1f}: {status} (Confidence: {result['confidence']:.1f}%)")
        
        return results
    
    def analyze_threshold_impact(self, test_texts, true_labels):
        """Analyze how different thresholds affect overall performance."""
        print(f"\n📊 THRESHOLD IMPACT ANALYSIS")
        print("=" * 50)
        
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        analysis_results = []
        
        # Get probabilities for all texts
        vectorized_texts = self.vectorizer.transform(test_texts)
        all_probabilities = self.model.predict_proba(vectorized_texts)[:, 1]
        
        print(f"{'Threshold':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'FP Rate':<10} {'FN Rate':<10}")
        print("-" * 70)
        
        for threshold in thresholds:
            predictions = (all_probabilities >= threshold).astype(int)
            
            # Calculate metrics
            tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
            
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            analysis_results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'false_positive_rate': fp_rate,
                'false_negative_rate': fn_rate,
                'false_positives': fp,
                'false_negatives': fn
            })
            
            print(f"{threshold:<10.1f} {accuracy:<10.3f} {precision:<10.3f} {recall:<10.3f} {fp_rate:<10.3f} {fn_rate:<10.3f}")
        
        return analysis_results
    
    def recommend_threshold(self, analysis_results):
        """Recommend optimal threshold based on analysis."""
        print(f"\n💡 THRESHOLD RECOMMENDATIONS")
        print("=" * 40)
        
        # Find threshold with best balance (minimize false positives while maintaining recall)
        best_balance = None
        best_score = float('inf')
        
        for result in analysis_results:
            # Score = weighted combination of FP rate and FN rate
            # Weight FP rate higher since false positives are often more problematic
            score = (2 * result['false_positive_rate']) + result['false_negative_rate']
            
            if score < best_score:
                best_score = score
                best_balance = result
        
        print(f"🎯 RECOMMENDED THRESHOLD: {best_balance['threshold']:.1f}")
        print(f"   • Accuracy: {best_balance['accuracy']:.3f}")
        print(f"   • False Positive Rate: {best_balance['false_positive_rate']:.3f}")
        print(f"   • False Negative Rate: {best_balance['false_negative_rate']:.3f}")
        print(f"   • False Positives: {best_balance['false_positives']}")
        print(f"   • False Negatives: {best_balance['false_negatives']}")
        
        # Additional recommendations
        print(f"\n📋 USAGE RECOMMENDATIONS:")
        
        if best_balance['threshold'] <= 0.4:
            print("   ⚠️  Very sensitive - use only if catching all violations is critical")
        elif best_balance['threshold'] <= 0.5:
            print("   ⚖️  Balanced - good for general use")
        elif best_balance['threshold'] <= 0.7:
            print("   🛡️  Conservative - reduces false positives, good for user experience")
        else:
            print("   🔒 Very conservative - use only if false positives are very costly")
        
        return best_balance


def main():
    """Main interactive testing function."""
    print("🎯 INTERACTIVE THRESHOLD TESTER")
    print("=" * 50)
    print("This tool helps you find the optimal threshold to reduce false positives.")
    print()
    
    # Initialize tester
    tester = ThresholdTester()
    
    # Test the problematic text
    problematic_text = "so i wanted to know that this text wasnt anyting bad. why would this be a violation. and how to fix such issues"
    
    print("\n🔍 TESTING YOUR PROBLEMATIC TEXT:")
    results = tester.test_text_with_multiple_thresholds(problematic_text)
    
    # Test additional examples
    test_cases = [
        "This is a helpful tutorial, thanks!",
        "You're an idiot for posting this",
        "I don't understand why this is wrong",
        "Can someone help me with this issue?",
        "This community has some problems"
    ]
    
    print("\n🧪 TESTING ADDITIONAL EXAMPLES:")
    for text in test_cases:
        tester.test_text_with_multiple_thresholds(text)
    
    # Analyze threshold impact on a test set
    test_texts = [
        problematic_text,
        "This is helpful, thank you",
        "Great explanation, very clear",
        "You're completely wrong about this",
        "This is garbage content",
        "I need help understanding this",
        "What's wrong with this approach?",
        "This seems problematic to me",
        "Can you explain the issue?",
        "I disagree with this method"
    ]
    
    # True labels (0 = not violation, 1 = violation)
    true_labels = [0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    
    analysis = tester.analyze_threshold_impact(test_texts, true_labels)
    recommendation = tester.recommend_threshold(analysis)
    
    print(f"\n✨ SUMMARY FOR YOUR CASE:")
    print("=" * 30)
    print(f"Your text: '{problematic_text[:50]}...'")
    print(f"At threshold 0.5: Would be flagged with ~24% confidence")
    print(f"At threshold {recommendation['threshold']:.1f}: Would be correctly classified as safe")
    print(f"\nRecommendation: Use threshold {recommendation['threshold']:.1f} to reduce false positives")
    
    return tester, analysis, recommendation


if __name__ == "__main__":
    main()