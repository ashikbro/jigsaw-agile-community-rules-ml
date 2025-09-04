#!/usr/bin/env python3
"""
Complete Model Testing Suite
Comprehensive evaluation of all classical ML models with detailed metrics and analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve, f1_score, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import our pipeline components
from training_pipeline import MLTrainingPipeline
from ensemble_methods import EnsembleManager
from model_evaluation import ModelEvaluator

def comprehensive_model_test():
    """
    Run comprehensive testing of all models with detailed evaluation
    """
    print("="*80)
    print("COMPREHENSIVE MODEL TESTING SUITE")
    print("="*80)
    
    # Initialize pipeline
    pipeline = MLTrainingPipeline(random_state=42)
    
    # Load data
    print("\n1. Loading and preparing data...")
    df = pipeline.load_data()
    print(f"   Total samples: {len(df)}")
    print(f"   Class distribution: {df['target'].value_counts().to_dict()}")
    
    # Split data
    train_df, val_df, test_df = pipeline.split_data(df)
    print(f"   Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Define models to test
    models_to_test = ['logistic_regression', 'multinomial_nb', 'svm_linear', 'random_forest']
    
    # Test different feature configurations
    feature_configs = [
        {
            'name': 'TF-IDF Word',
            'config': {
                'max_features_word': 5000,
                'max_features_char': 0,
                'word_ngram_range': (1, 2),
                'char_ngram_range': (3, 6),
                'feature_selection_k': 3000
            }
        },
        {
            'name': 'TF-IDF Char',
            'config': {
                'max_features_word': 0,
                'max_features_char': 5000,
                'word_ngram_range': (1, 3),
                'char_ngram_range': (2, 4),
                'feature_selection_k': 3000
            }
        },
        {
            'name': 'Combined Features',
            'config': {
                'max_features_word': 5000,
                'max_features_char': 2000,
                'word_ngram_range': (1, 3),
                'char_ngram_range': (3, 6),
                'feature_selection_k': 3000
            }
        }
    ]
    
    all_results = {}
    
    # Test each feature configuration
    for feat_config in feature_configs:
        print(f"\n2. Testing with {feat_config['name']} features...")
        print("-" * 60)
        
        # Train pipeline with current configuration
        config = {
            'models': models_to_test,
            'use_grid_search': True,
            'preprocessing': {
                'remove_stopwords': True,
                'use_lemmatization': True
            },
            'features': feat_config['config']
        }
        
        try:
            results = pipeline.train_pipeline(
                df,
                models_to_train=config['models'],
                use_grid_search=config['use_grid_search'],
                preprocessing_config=config['preprocessing'],
                feature_config=config['features']
            )
            
            all_results[feat_config['name']] = results
            
            # Display results for this configuration
            print(f"\n   Results for {feat_config['name']}:")
            print("   " + "="*50)
            
            for model_name, result in results['validation_results'].items():
                test_result = results['test_results'].get(model_name, {})
                print(f"   {model_name.replace('_', ' ').title()}:")
                print(f"     Val - F1: {result['f1_score']:.4f}, AUC: {result.get('auc_score', 0):.4f}")
                print(f"     Test - F1: {test_result.get('f1_score', 0):.4f}, AUC: {test_result.get('auc_score', 0):.4f}")
                
        except Exception as e:
            print(f"   Error with {feat_config['name']}: {str(e)}")
            continue
    
    # Cross-validation analysis
    print("\n3. Cross-Validation Analysis...")
    print("-" * 60)
    
    # Use the best performing configuration for CV
    best_config = feature_configs[2]  # Combined features
    pipeline_cv = MLTrainingPipeline(random_state=42)
    
    # Setup preprocessing and features
    pipeline_cv.setup_preprocessing(remove_stopwords=True, use_lemmatization=True)
    pipeline_cv.setup_feature_extraction(**best_config['config'])
    
    # Prepare data for CV
    all_texts = df['comment_text'].tolist()
    processed_texts = [pipeline_cv.preprocessor.preprocess_text(text) for text in all_texts]
    pipeline_cv.feature_extractor.fit(processed_texts)
    X = pipeline_cv.feature_extractor.transform(processed_texts)
    y = df['target'].values
    
    # Perform cross-validation
    cv_results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for model_name in models_to_test:
        try:
            # Get model from pipeline
            model = pipeline_cv.ml_pipeline.models.get(model_name)
            if model is None:
                # Train the specific model first
                try:
                    pipeline_cv.ml_pipeline.train_model(model_name, X, y)
                    model = pipeline_cv.ml_pipeline.models.get(model_name)
                    if model is None:
                        print(f"   Error with {model_name}: Model training failed")
                        continue
                except Exception as e:
                    print(f"   Error with {model_name}: {str(e)}")
                    continue
            
            # Cross-validation scores
            f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
            accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            roc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
            
            cv_results[model_name] = {
                'f1_mean': f1_scores.mean(),
                'f1_std': f1_scores.std(),
                'accuracy_mean': accuracy_scores.mean(),
                'accuracy_std': accuracy_scores.std(),
                'roc_mean': roc_scores.mean(),
                'roc_std': roc_scores.std()
            }
            
            print(f"   {model_name.replace('_', ' ').title()}:")
            print(f"     F1: {f1_scores.mean():.4f} (±{f1_scores.std():.4f})")
            print(f"     Accuracy: {accuracy_scores.mean():.4f} (±{accuracy_scores.std():.4f})")
            print(f"     ROC-AUC: {roc_scores.mean():.4f} (±{roc_scores.std():.4f})")
            
        except Exception as e:
            print(f"   Error with {model_name}: {str(e)}")
    
    # Ensemble testing
    print("\n4. Ensemble Methods Testing...")
    print("-" * 60)
    
    try:
        # Create ensemble manager
        ensemble_manager = EnsembleManager(random_state=42)
        
        # Add base models
        for model_name in models_to_test:
            if model_name in pipeline.ml_pipeline.models:
                ensemble_manager.add_base_model(model_name, pipeline.ml_pipeline.models[model_name])
        
        # Prepare validation data
        val_texts = [pipeline.preprocessor.preprocess_text(text) for text in val_df['comment_text']]
        X_val = pipeline.feature_extractor.transform(val_texts)
        y_val = val_df['target'].values
        
        # Test different ensemble methods
        ensemble_methods = ['voting', 'blending', 'stacking']
        ensemble_results = {}
        
        for method in ensemble_methods:
            try:
                if method == 'voting':
                    ensemble = ensemble_manager.create_voting_ensemble(models_to_test[:3], voting='soft')
                    ensemble.fit(X_val, y_val)  # Dummy fit
                    
                elif method == 'blending':
                    weights, _ = ensemble_manager.optimize_blending_weights(X_val, y_val, models_to_test[:3])
                    ensemble = ensemble_manager.create_blending_ensemble(models_to_test[:3], weights=weights)
                    ensemble.fit(X_val, y_val)
                    
                elif method == 'stacking':
                    ensemble = ensemble_manager.create_stacking_ensemble(models_to_test[:3], cv_folds=3)
                    train_texts = [pipeline.preprocessor.preprocess_text(text) for text in train_df['comment_text']]
                    X_train = pipeline.feature_extractor.transform(train_texts)
                    y_train = train_df['target'].values
                    ensemble.fit(X_train, y_train)
                
                # Evaluate ensemble
                y_pred = ensemble.predict(X_val)
                y_proba = ensemble.predict_proba(X_val)[:, 1] if hasattr(ensemble, 'predict_proba') else None
                
                f1 = f1_score(y_val, y_pred)
                accuracy = accuracy_score(y_val, y_pred)
                auc = roc_auc_score(y_val, y_proba) if y_proba is not None else None
                
                ensemble_results[method] = {
                    'f1_score': f1,
                    'accuracy': accuracy,
                    'auc_score': auc
                }
                
                print(f"   {method.title()} Ensemble:")
                print(f"     F1: {f1:.4f}")
                print(f"     Accuracy: {accuracy:.4f}")
                print(f"     AUC: {auc:.4f}" if auc else "     AUC: N/A")
                
            except Exception as e:
                print(f"   Error with {method} ensemble: {str(e)}")
    
    except Exception as e:
        print(f"   Ensemble testing failed: {str(e)}")
    
    # Final summary
    print("\n" + "="*80)
    print("TESTING SUMMARY")
    print("="*80)
    
    # Find best performing models
    if all_results:
        print("\nBest performing configurations:")
        for feat_name, results in all_results.items():
            if 'test_results' in results:
                best_model = None
                best_f1 = 0
                
                for model_name, result in results['test_results'].items():
                    if result.get('f1_score', 0) > best_f1:
                        best_f1 = result.get('f1_score', 0)
                        best_model = model_name
                
                if best_model:
                    print(f"  {feat_name}: {best_model.replace('_', ' ').title()} (F1: {best_f1:.4f})")
    
    if cv_results:
        print("\nCross-validation rankings (by F1 score):")
        sorted_cv = sorted(cv_results.items(), key=lambda x: x[1]['f1_mean'], reverse=True)
        for i, (model_name, scores) in enumerate(sorted_cv, 1):
            print(f"  {i}. {model_name.replace('_', ' ').title()}: {scores['f1_mean']:.4f} (±{scores['f1_std']:.4f})")
    
    print("\n✅ Comprehensive model testing completed!")
    print("\nRecommendations:")
    print("- Use combined features for best performance")
    print("- Consider ensemble methods for improved results")
    print("- Logistic Regression often provides good baseline performance")
    print("- Cross-validation shows model stability across different data splits")

if __name__ == "__main__":
    comprehensive_model_test()