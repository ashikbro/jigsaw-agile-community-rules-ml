#!/usr/bin/env python3
"""
Test basic feature extraction and model training.
"""

import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Import our custom feature extractors
from advanced_feature_extractor import AdvancedHybridFeatureExtractor
from domain_specific_features import CommunityModerationFeatureExtractor
from topic_modeling_features import TopicModelingFeatureExtractor

def create_test_data():
    """
    Create test data for feature extraction validation.
    """
    texts = [
        "This is a great post, thanks for sharing!",
        "You're such an idiot, go kill yourself!",
        "I disagree with your opinion but respect your right to have it.",
        "This is completely stupid and worthless trash!",
        "Nice work on this project, very helpful.",
        "You should be banned from this community.",
        "I think we need better moderation rules here.",
        "This violates community guidelines and should be removed.",
        "Great discussion, learned a lot from everyone.",
        "Stop posting this garbage, it's against the rules.",
        "Thanks for the helpful explanation.",
        "This kind of behavior is not acceptable here.",
        "I appreciate the constructive feedback.",
        "You're breaking the community rules with this post.",
        "This is a well-reasoned argument.",
        "This content should be reported to moderators.",
        "What a fantastic contribution to the discussion!",
        "Delete this post immediately, it's offensive.",
        "I found this very informative and useful.",
        "This user should be permanently banned."
    ]
    
    # Binary labels (0 = acceptable, 1 = violation)
    labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    
    return texts, labels

def test_hybrid_features():
    """
    Test hybrid feature extraction.
    """
    print("\n🔧 Testing Hybrid Feature Extraction")
    print("-" * 40)
    
    texts, labels = create_test_data()
    
    # Initialize hybrid feature extractor
    hybrid_extractor = AdvancedHybridFeatureExtractor(
        word_max_features=100,
        char_max_features=50,
        word_ngram_range=(1, 2),
        char_ngram_range=(2, 4),
        min_df=1,
        max_df=0.95
    )
    
    # Split data
    texts_train, texts_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    # Extract features
    X_train = hybrid_extractor.fit_transform(texts_train)
    X_test = hybrid_extractor.transform(texts_test)
    
    print(f"  Training features shape: {X_train.shape}")
    print(f"  Test features shape: {X_test.shape}")
    
    # Train a simple model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  F1-Score: {f1:.3f}")
    
    return True

def test_domain_features():
    """
    Test domain-specific feature extraction.
    """
    print("\n🏘️ Testing Domain-Specific Features")
    print("-" * 40)
    
    texts, _ = create_test_data()
    
    # Initialize domain feature extractor
    domain_extractor = CommunityModerationFeatureExtractor()
    
    # Extract features
    features = domain_extractor.fit_transform(texts[:5])  # Test with first 5 texts
    
    print(f"  Domain features shape: {features.shape}")
    if hasattr(domain_extractor, 'get_feature_names'):
        print(f"  Feature names: {domain_extractor.get_feature_names()[:10]}...")  # Show first 10
    else:
        print(f"  Domain features extracted successfully")
    
    return True

def test_topic_modeling():
    """
    Test topic modeling feature extraction.
    """
    print("\n📊 Testing Topic Modeling Features")
    print("-" * 40)
    
    texts, labels = create_test_data()
    
    try:
        # Initialize topic modeling extractor
        topic_extractor = TopicModelingFeatureExtractor(
            n_topics_lda=3,
            n_topics_nmf=3,
            max_features=50
        )
        
        # Split data
        texts_train, texts_test, _, _ = train_test_split(
            texts, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        # Extract features
        X_train = topic_extractor.fit_transform(texts_train)
        X_test = topic_extractor.transform(texts_test)
        
        print(f"  Training topic features shape: {X_train.shape}")
        print(f"  Test topic features shape: {X_test.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ⚠️ Topic modeling test failed: {e}")
        return False

def test_combined_features():
    """
    Test combining all feature types.
    """
    print("\n🔗 Testing Combined Features")
    print("-" * 40)
    
    texts, labels = create_test_data()
    
    # Split data
    texts_train, texts_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    # Initialize extractors
    hybrid_extractor = AdvancedHybridFeatureExtractor(
        word_max_features=50,
        char_max_features=25,
        min_df=1
    )
    
    domain_extractor = CommunityModerationFeatureExtractor()
    
    # Extract features
    hybrid_train = hybrid_extractor.fit_transform(texts_train)
    hybrid_test = hybrid_extractor.transform(texts_test)
    
    domain_train = domain_extractor.fit_transform(texts_train)
    domain_test = domain_extractor.transform(texts_test)
    
    # Combine features
    from scipy.sparse import hstack
    
    X_train_combined = hstack([hybrid_train, domain_train])
    X_test_combined = hstack([hybrid_test, domain_test])
    
    print(f"  Combined training features shape: {X_train_combined.shape}")
    print(f"  Combined test features shape: {X_test_combined.shape}")
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_combined, y_train)
    
    # Predict
    y_pred = model.predict(X_test_combined)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"  Combined model accuracy: {accuracy:.3f}")
    print(f"  Combined model F1-score: {f1:.3f}")
    
    return True

def main():
    """
    Run all feature tests.
    """
    print("🧪 Testing Advanced ML Features")
    print("=" * 50)
    
    tests = [
        ("Hybrid Features", test_hybrid_features),
        ("Domain Features", test_domain_features),
        ("Topic Modeling", test_topic_modeling),
        ("Combined Features", test_combined_features)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"  ✅ {test_name} test passed")
            else:
                print(f"  ❌ {test_name} test failed")
        except Exception as e:
            print(f"  ❌ {test_name} test error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 50)
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All advanced ML features are working correctly!")
        return True
    else:
        print(f"\n⚠️ {total - passed} tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)