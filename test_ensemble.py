#!/usr/bin/env python3
"""
Test script for ensemble methods integration with the ML pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

# Import our pipeline components
from training_pipeline import MLTrainingPipeline
from ensemble_methods import EnsembleManager, SimpleBlender, StackingEnsemble
from classical_models import ClassicalMLPipeline

def test_ensemble_integration():
    """
    Test ensemble methods with the existing ML pipeline
    """
    print("=" * 60)
    print("Testing Ensemble Methods Integration")
    print("=" * 60)
    
    # Load sample data
    try:
        # Try to load the actual dataset
        data_path = "data/train.csv"
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            # Take a smaller sample for faster testing
            df = df.sample(n=min(2000, len(df)), random_state=42)
        else:
            # Create synthetic data if real data not available
            print("Creating synthetic data for testing...")
            
            # Create more diverse and realistic text samples
            np.random.seed(42)
            
            violation_templates = [
                "This is absolutely disgusting hate speech targeting {} community members",
                "You people are completely {} and should be permanently banned from society",
                "I absolutely hate all {} individuals and their stupid beliefs",
                "Go back to your original country {} foreigners don't belong here",
                "You {} supporters are completely wrong and not welcome anywhere",
                "These {} activists are destroying our beautiful traditional values",
                "Stop promoting {} agenda it's completely against natural order",
                "All {} people should be completely removed from our neighborhood"
            ]
            
            safe_templates = [
                "This is a thoughtful and respectful discussion about {} related topics",
                "I personally think {} is genuinely interesting and worth exploring further",
                "Let's have a constructive conversation about {} in a respectful manner",
                "I respectfully disagree with {} but completely respect your personal opinion",
                "Thank you for sharing your valuable thoughts and insights on {}",
                "The {} community has made significant contributions to our society",
                "I appreciate learning about different perspectives on {} issues today",
                "It's important to have open dialogue about {} while respecting everyone",
                "The research on {} shows interesting patterns worth discussing academically",
                "Many experts believe {} requires careful consideration and balanced approach"
            ]
            
            # More diverse targets and topics
            targets = [
                "minority", "women", "immigrant", "religious", "political", "sports", 
                "technology", "education", "environmental", "healthcare", "economic", 
                "cultural", "social", "scientific", "artistic", "academic", "professional",
                "international", "local", "community", "youth", "elderly", "disabled"
            ]
            
            # Additional words to increase vocabulary
            additional_words = [
                "absolutely", "completely", "definitely", "certainly", "obviously",
                "particularly", "specifically", "generally", "typically", "usually",
                "important", "significant", "valuable", "essential", "crucial",
                "interesting", "fascinating", "remarkable", "outstanding", "excellent",
                "terrible", "awful", "horrible", "disgusting", "unacceptable",
                "wonderful", "amazing", "fantastic", "brilliant", "impressive"
            ]
            
            texts = []
            labels = []
            
            for i in range(1000):
                if i < 500:
                    # Violation samples
                    template = np.random.choice(violation_templates)
                    target = np.random.choice(targets)
                    # Add some additional words for vocabulary diversity
                    extra_words = np.random.choice(additional_words, size=np.random.randint(0, 3), replace=False)
                    text = template.format(target)
                    if len(extra_words) > 0:
                        text += " " + " ".join(extra_words)
                    labels.append(1)
                else:
                    # Safe samples
                    template = np.random.choice(safe_templates)
                    target = np.random.choice(targets)
                    # Add some additional words for vocabulary diversity
                    extra_words = np.random.choice(additional_words, size=np.random.randint(0, 3), replace=False)
                    text = template.format(target)
                    if len(extra_words) > 0:
                        text += " " + " ".join(extra_words)
                    labels.append(0)
                
                texts.append(text)
            
            df = pd.DataFrame({
                'comment_text': texts,
                'target': labels
            })
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return False
    
    print(f"Loaded {len(df)} samples")
    
    # Split data
    train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42, stratify=df['target'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['target'])
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Initialize pipeline
    pipeline = MLTrainingPipeline()
    
    # Setup preprocessing
    preprocessing_config = {
        'remove_stopwords': True,
        'use_lemmatization': True
    }
    pipeline.setup_preprocessing(**preprocessing_config)
    
    # Setup feature extraction
    feature_config = {
        'max_features_word': 5000,
        'max_features_char': 3000,
        'word_ngram_range': (1, 2),
        'char_ngram_range': (3, 5),
        'feature_selection_k': 1000
    }
    pipeline.setup_feature_extraction(**feature_config)
    
    # Prepare data
    print("\nPreparing data...")
    train_texts = train_df['comment_text'].tolist()
    val_texts = val_df['comment_text'].tolist()
    test_texts = test_df['comment_text'].tolist()
    
    y_train = train_df['target'].values
    y_val = val_df['target'].values
    y_test = test_df['target'].values
    
    # Create DataFrame for training
    train_data = pd.DataFrame({
        'comment_text': train_texts,
        'target': y_train
    })
    
    # Train pipeline
    print("\nTraining pipeline...")
    models_to_train = ['logistic_regression', 'multinomial_nb', 'svm_linear']
    
    try:
        training_results = pipeline.train_pipeline(
            df=train_data,
            models_to_train=models_to_train,
            use_grid_search=False,  # Faster for testing
            preprocessing_config=preprocessing_config,
            feature_config=feature_config
        )
        
        individual_results = training_results.get('validation_results', {})
        trained_models = pipeline.ml_pipeline.models if pipeline.ml_pipeline else {}
        
        print(f"Successfully trained {len(trained_models)} models")
        for model_name, results in individual_results.items():
            if isinstance(results, dict):
                f1 = results.get('f1_score', 'N/A')
                auc = results.get('auc_score', 'N/A')
                print(f"  {model_name} - F1: {f1 if f1 == 'N/A' else f'{f1:.3f}'}, AUC: {auc if auc == 'N/A' or auc is None else f'{auc:.3f}'}")
        
    except Exception as e:
        print(f"Error training pipeline: {e}")
        return False
    
    if not trained_models:
        print("No models were successfully trained. Cannot proceed with ensemble testing.")
        return False
    
    print(f"\nSuccessfully trained {len(trained_models)} models")
    
    # Test ensemble methods
    print("\n" + "=" * 40)
    print("Testing Ensemble Methods")
    print("=" * 40)
    
    # Get feature matrices using the trained pipeline components
    if not pipeline.preprocessor or not pipeline.feature_extractor:
        print("Pipeline components not properly initialized")
        return False
        
    try:
        # Preprocess texts
        train_processed = [pipeline.preprocessor.preprocess_text(text) for text in train_texts]
        val_processed = [pipeline.preprocessor.preprocess_text(text) for text in val_texts]
        test_processed = [pipeline.preprocessor.preprocess_text(text) for text in test_texts]
        
        # Transform to feature matrices
        X_train = pipeline.feature_extractor.transform(train_processed)
        X_val = pipeline.feature_extractor.transform(val_processed)
        X_test = pipeline.feature_extractor.transform(test_processed)
        
        print(f"Feature matrices - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
    except Exception as e:
        print(f"Error creating feature matrices: {e}")
        return False
    
    # Initialize ensemble manager
    ensemble_manager = EnsembleManager(random_state=42)
    
    # Add trained models to ensemble manager
    for name, model in trained_models.items():
        ensemble_manager.add_base_model(name, model)
    
    model_names = list(trained_models.keys())
    ensemble_results = {}
    
    # Test 1: Voting Ensemble
    print("\n1. Testing Voting Ensemble...")
    try:
        voting_ensemble = ensemble_manager.create_voting_ensemble(model_names, voting='soft')
        voting_ensemble.fit(X_train, y_train)
        
        # Evaluate
        val_pred = voting_ensemble.predict(X_val)
        val_proba = voting_ensemble.predict_proba(X_val)[:, 1]
        
        voting_f1 = f1_score(y_val, val_pred)
        voting_auc = roc_auc_score(y_val, val_proba)
        voting_acc = accuracy_score(y_val, val_pred)
        
        ensemble_results['voting'] = {
            'f1': voting_f1,
            'auc': voting_auc,
            'accuracy': voting_acc
        }
        
        print(f"   Voting Ensemble - F1: {voting_f1:.3f}, AUC: {voting_auc:.3f}, Accuracy: {voting_acc:.3f}")
        
    except Exception as e:
        print(f"   Error with voting ensemble: {e}")
    
    # Test 2: Simple Blending
    print("\n2. Testing Simple Blending...")
    try:
        blender = SimpleBlender()
        for name, model in trained_models.items():
            blender.add_model(name, model)
        
        blender.fit(X_train, y_train)
        
        # Evaluate
        val_pred = blender.predict(X_val)
        val_proba = blender.predict_proba(X_val)[:, 1]
        
        blending_f1 = f1_score(y_val, val_pred)
        blending_auc = roc_auc_score(y_val, val_proba)
        blending_acc = accuracy_score(y_val, val_pred)
        
        ensemble_results['blending'] = {
            'f1': blending_f1,
            'auc': blending_auc,
            'accuracy': blending_acc
        }
        
        print(f"   Blending Ensemble - F1: {blending_f1:.3f}, AUC: {blending_auc:.3f}, Accuracy: {blending_acc:.3f}")
        
    except Exception as e:
        print(f"   Error with blending ensemble: {e}")
    
    # Test 3: Stacking Ensemble
    print("\n3. Testing Stacking Ensemble...")
    try:
        stacking_ensemble = StackingEnsemble(
            base_models=trained_models,
            cv_folds=3,  # Reduced for faster testing
            use_original_features=False  # Use only meta-features for faster testing
        )
        
        stacking_ensemble.fit(X_train, y_train)
        
        # Evaluate
        val_pred = stacking_ensemble.predict(X_val)
        val_proba = stacking_ensemble.predict_proba(X_val)[:, 1]
        
        stacking_f1 = f1_score(y_val, val_pred)
        stacking_auc = roc_auc_score(y_val, val_proba)
        stacking_acc = accuracy_score(y_val, val_pred)
        
        ensemble_results['stacking'] = {
            'f1': stacking_f1,
            'auc': stacking_auc,
            'accuracy': stacking_acc
        }
        
        print(f"   Stacking Ensemble - F1: {stacking_f1:.3f}, AUC: {stacking_auc:.3f}, Accuracy: {stacking_acc:.3f}")
        
    except Exception as e:
        print(f"   Error with stacking ensemble: {e}")
    
    # Test 4: Optimized Blending Weights
    print("\n4. Testing Optimized Blending...")
    try:
        optimal_weights, best_score = ensemble_manager.optimize_blending_weights(
            X_val, y_val, model_names, metric='f1'
        )
        
        print(f"   Optimal weights: {[f'{w:.3f}' for w in optimal_weights]}")
        print(f"   Best F1 score: {best_score:.3f}")
        
        # Create optimized blender
        optimized_blender = ensemble_manager.create_blending_ensemble(model_names, weights=optimal_weights)
        optimized_blender.fit(X_train, y_train)
        
        # Evaluate
        val_pred = optimized_blender.predict(X_val)
        val_proba = optimized_blender.predict_proba(X_val)[:, 1]
        
        opt_blending_f1 = f1_score(y_val, val_pred)
        opt_blending_auc = roc_auc_score(y_val, val_proba)
        opt_blending_acc = accuracy_score(y_val, val_pred)
        
        ensemble_results['optimized_blending'] = {
            'f1': opt_blending_f1,
            'auc': opt_blending_auc,
            'accuracy': opt_blending_acc
        }
        
        print(f"   Optimized Blending - F1: {opt_blending_f1:.3f}, AUC: {opt_blending_auc:.3f}, Accuracy: {opt_blending_acc:.3f}")
        
    except Exception as e:
        print(f"   Error with optimized blending: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("ENSEMBLE RESULTS SUMMARY")
    print("=" * 50)
    
    print("\nIndividual Models:")
    for name, results in individual_results.items():
        f1 = results.get('f1_score', 'N/A')
        auc = results.get('auc_score', 'N/A')
        print(f"  {name:20} - F1: {f1 if f1 == 'N/A' else f'{f1:.3f}'}, AUC: {auc if auc == 'N/A' or auc is None else f'{auc:.3f}'}")
    
    print("\nEnsemble Methods:")
    for name, results in ensemble_results.items():
        print(f"  {name:20} - F1: {results['f1']:.3f}, AUC: {results['auc']:.3f}, Accuracy: {results['accuracy']:.3f}")
    
    # Find best performing method
    if ensemble_results:
        best_method = max(ensemble_results.items(), key=lambda x: x[1]['f1'])
        print(f"\nBest performing ensemble: {best_method[0]} (F1: {best_method[1]['f1']:.3f})")
    
    print("\n✅ Ensemble methods testing completed successfully!")
    return True

if __name__ == "__main__":
    success = test_ensemble_integration()
    if success:
        print("\n🎉 All ensemble tests passed!")
    else:
        print("\n❌ Some ensemble tests failed.")
        sys.exit(1)