#!/usr/bin/env python3
"""
Test Ensemble Models for Content Moderation
Demonstrates ensemble methods using existing trained models
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from ensemble_methods import EnsembleManager, SimpleBlender, StackingEnsemble
import warnings
warnings.filterwarnings('ignore')

def load_available_models():
    """Load all available trained models from the models directory."""
    models_dir = Path("models")
    available_models = {}
    vectorizers = {}
    
    if not models_dir.exists():
        print("❌ Models directory not found!")
        return {}, {}
    
    # Look for different model types
    model_patterns = [
        "human_aware_large_dataset_*",
        "human_aware_medium_dataset_*", 
        "context_aware_large_dataset_*",
        "context_aware_medium_dataset_*",
        "large_dataset_*",
        "medium_dataset_*"
    ]
    
    for pattern in model_patterns:
        model_dirs = list(models_dir.glob(pattern))
        for model_dir in model_dirs:
            if model_dir.is_dir():
                print(f"📁 Checking directory: {model_dir.name}")
                
                # Load vectorizer
                vectorizer_path = model_dir / "vectorizer.joblib"
                if not vectorizer_path.exists():
                    print(f"  ⚠️  No vectorizer found in {model_dir.name}")
                    continue
                
                try:
                    vectorizer = joblib.load(vectorizer_path)
                    
                    # Look for different model types
                    model_files = {
                        'logistic_regression': 'logistic_regression.joblib',
                        'random_forest': 'random_forest.joblib',
                        'svm': 'svm.joblib',
                        'naive_bayes': 'naive_bayes.joblib',
                        'xgboost': 'xgboost.joblib'
                    }
                    
                    for model_name, filename in model_files.items():
                        model_path = model_dir / filename
                        if model_path.exists():
                            try:
                                model = joblib.load(model_path)
                                model_key = f"{model_dir.name}_{model_name}"
                                available_models[model_key] = model
                                vectorizers[model_key] = vectorizer
                                print(f"  ✅ Loaded: {model_key}")
                            except Exception as e:
                                print(f"  ❌ Failed to load {filename}: {e}")
                
                except Exception as e:
                    print(f"  ❌ Failed to load vectorizer from {model_dir.name}: {e}")
    
    print(f"\n📊 Total models loaded: {len(available_models)}")
    return available_models, vectorizers

def load_test_data():
    """Load test data for ensemble evaluation."""
    # Try to load existing test data
    test_files = [
        "data/test_data.csv",
        "data/validation_data.csv",
        "outputs/test_results.csv"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            try:
                df = pd.read_csv(test_file)
                if 'comment_text' in df.columns and 'toxic' in df.columns:
                    print(f"📄 Loaded test data from {test_file}")
                    return df['comment_text'].tolist(), df['toxic'].tolist()
            except Exception as e:
                print(f"❌ Failed to load {test_file}: {e}")
    
    # If no test data found, create sample data
    print("📝 Creating sample test data...")
    sample_texts = [
        "This is a great post, thanks for sharing!",
        "I completely disagree with your opinion but respect your right to have it.",
        "You are such an idiot, go kill yourself!",
        "This content is inappropriate and should be removed.",
        "I love this community, everyone is so helpful.",
        "What a stupid waste of time this is.",
        "Thanks for the detailed explanation, very informative.",
        "I hate people like you, you make me sick.",
        "Could you please provide more information about this topic?",
        "This is the dumbest thing I've ever read.",
        "Great job on this project, well done!",
        "You people are all morons and don't deserve to live.",
        "I'm not sure I understand, could you clarify?",
        "This is toxic garbage that shouldn't exist.",
        "Excellent work, keep it up!",
        "I want to punch everyone who thinks like this.",
        "This is a thoughtful and well-reasoned argument.",
        "Die in a fire, you worthless piece of trash.",
        "I appreciate your perspective on this matter.",
        "This makes me want to hurt someone."
    ]
    
    sample_labels = [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    
    return sample_texts, sample_labels

def evaluate_individual_models(models, vectorizers, test_texts, test_labels):
    """Evaluate individual models to understand their performance."""
    print("\n" + "="*60)
    print("INDIVIDUAL MODEL EVALUATION")
    print("="*60)
    
    individual_results = []
    
    for model_name, model in models.items():
        try:
            vectorizer = vectorizers[model_name]
            
            # Vectorize test texts
            X_test = vectorizer.transform(test_texts)
            
            # Make predictions
            predictions = model.predict(X_test)
            
            # Get probabilities if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_test)[:, 1]
            else:
                probabilities = predictions.astype(float)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(test_labels, probabilities)
            except:
                auc = None
            
            accuracy = accuracy_score(test_labels, predictions)
            f1 = f1_score(test_labels, predictions)
            precision = precision_score(test_labels, predictions, zero_division=0)
            recall = recall_score(test_labels, predictions, zero_division=0)
            
            result = {
                'model_name': model_name,
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'auc_score': auc
            }
            
            individual_results.append(result)
            
            print(f"\n📊 {model_name}:")
            print(f"   Accuracy:  {accuracy:.4f}")
            print(f"   F1 Score:  {f1:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall:    {recall:.4f}")
            if auc is not None:
                print(f"   AUC Score: {auc:.4f}")
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
    
    # Sort by F1 score
    individual_results.sort(key=lambda x: x['f1_score'], reverse=True)
    
    print("\n🏆 Model Rankings (by F1 Score):")
    for i, result in enumerate(individual_results, 1):
        print(f"   {i}. {result['model_name']}: {result['f1_score']:.4f}")
    
    return individual_results

def create_and_test_ensembles(models, vectorizers, test_texts, test_labels, top_n=3):
    """Create and test different ensemble methods."""
    print("\n" + "="*60)
    print("ENSEMBLE MODEL TESTING")
    print("="*60)
    
    if len(models) < 2:
        print("❌ Need at least 2 models to create ensembles")
        return
    
    # Select top performing models for ensemble
    individual_results = evaluate_individual_models(models, vectorizers, test_texts, test_labels)
    top_models = [result['model_name'] for result in individual_results[:min(top_n, len(individual_results))]]
    
    print(f"\n🎯 Creating ensembles with top {len(top_models)} models: {top_models}")
    
    # Prepare data for ensemble training
    # We'll use a simple approach where we vectorize with the first model's vectorizer
    # In practice, you might want to use a more sophisticated approach
    primary_vectorizer = vectorizers[top_models[0]]
    X_test = primary_vectorizer.transform(test_texts)
    
    # Split test data for ensemble validation
    X_val, X_test_final, y_val, y_test_final = train_test_split(
        X_test, test_labels, test_size=0.5, random_state=42, stratify=test_labels
    )
    
    # Create ensemble manager
    manager = EnsembleManager(random_state=42)
    
    # Add models that use the same vectorizer (for simplicity)
    compatible_models = {}
    for model_name in top_models:
        # For this demo, we'll use models that can work with the primary vectorizer
        # In practice, you'd need to handle different vectorizers properly
        try:
            model = models[model_name]
            # Test if model can work with primary vectorizer
            test_pred = model.predict(X_val[:1])
            compatible_models[model_name] = model
            manager.add_base_model(model_name, model)
        except Exception as e:
            print(f"⚠️  Skipping {model_name} due to vectorizer incompatibility: {e}")
    
    if len(compatible_models) < 2:
        print("❌ Not enough compatible models for ensemble creation")
        return
    
    model_names = list(compatible_models.keys())
    print(f"\n✅ Using compatible models: {model_names}")
    
    ensemble_results = []
    
    # 1. Voting Ensemble (Hard)
    try:
        print("\n🗳️  Testing Hard Voting Ensemble...")
        voting_hard = manager.create_voting_ensemble(model_names, voting='hard')
        voting_hard.fit(X_val, y_val)
        
        pred_hard = voting_hard.predict(X_test_final)
        f1_hard = f1_score(y_test_final, pred_hard)
        acc_hard = accuracy_score(y_test_final, pred_hard)
        
        ensemble_results.append({
            'name': 'Hard Voting',
            'f1_score': f1_hard,
            'accuracy': acc_hard
        })
        
        print(f"   F1 Score: {f1_hard:.4f}")
        print(f"   Accuracy: {acc_hard:.4f}")
        
    except Exception as e:
        print(f"❌ Hard voting failed: {e}")
    
    # 2. Voting Ensemble (Soft)
    try:
        print("\n🗳️  Testing Soft Voting Ensemble...")
        voting_soft = manager.create_voting_ensemble(model_names, voting='soft')
        voting_soft.fit(X_val, y_val)
        
        pred_soft = voting_soft.predict(X_test_final)
        f1_soft = f1_score(y_test_final, pred_soft)
        acc_soft = accuracy_score(y_test_final, pred_soft)
        
        ensemble_results.append({
            'name': 'Soft Voting',
            'f1_score': f1_soft,
            'accuracy': acc_soft
        })
        
        print(f"   F1 Score: {f1_soft:.4f}")
        print(f"   Accuracy: {acc_soft:.4f}")
        
    except Exception as e:
        print(f"❌ Soft voting failed: {e}")
    
    # 3. Optimized Blending
    try:
        print("\n🔄 Testing Optimized Blending Ensemble...")
        optimal_weights, best_score = manager.optimize_blending_weights(
            X_val, y_val, model_names, metric='f1'
        )
        
        blending = manager.create_blending_ensemble(model_names, weights=optimal_weights)
        blending.fit(X_val, y_val)
        
        pred_blend = blending.predict(X_test_final)
        f1_blend = f1_score(y_test_final, pred_blend)
        acc_blend = accuracy_score(y_test_final, pred_blend)
        
        ensemble_results.append({
            'name': 'Optimized Blending',
            'f1_score': f1_blend,
            'accuracy': acc_blend
        })
        
        print(f"   F1 Score: {f1_blend:.4f}")
        print(f"   Accuracy: {acc_blend:.4f}")
        print(f"   Optimal Weights: {dict(zip(model_names, optimal_weights))}")
        
    except Exception as e:
        print(f"❌ Blending failed: {e}")
    
    # 4. Stacking Ensemble
    try:
        print("\n📚 Testing Stacking Ensemble...")
        stacking = manager.create_stacking_ensemble(model_names, cv_folds=3)
        stacking.fit(X_val, y_val)
        
        pred_stack = stacking.predict(X_test_final)
        f1_stack = f1_score(y_test_final, pred_stack)
        acc_stack = accuracy_score(y_test_final, pred_stack)
        
        ensemble_results.append({
            'name': 'Stacking',
            'f1_score': f1_stack,
            'accuracy': acc_stack
        })
        
        print(f"   F1 Score: {f1_stack:.4f}")
        print(f"   Accuracy: {acc_stack:.4f}")
        
    except Exception as e:
        print(f"❌ Stacking failed: {e}")
    
    # Compare ensemble results
    if ensemble_results:
        print("\n" + "="*60)
        print("ENSEMBLE COMPARISON")
        print("="*60)
        
        ensemble_results.sort(key=lambda x: x['f1_score'], reverse=True)
        
        print(f"{'Rank':<4} {'Method':<20} {'F1 Score':<10} {'Accuracy':<10}")
        print("-" * 50)
        
        for i, result in enumerate(ensemble_results, 1):
            print(f"{i:<4} {result['name']:<20} {result['f1_score']:<10.4f} {result['accuracy']:<10.4f}")
        
        best_ensemble = ensemble_results[0]
        print(f"\n🏆 Best Ensemble: {best_ensemble['name']} (F1: {best_ensemble['f1_score']:.4f})")
        
        # Compare with best individual model
        best_individual = individual_results[0]
        improvement = best_ensemble['f1_score'] - best_individual['f1_score']
        
        print(f"\n📈 Improvement over best individual model:")
        print(f"   Best Individual: {best_individual['model_name']} (F1: {best_individual['f1_score']:.4f})")
        print(f"   Best Ensemble: {best_ensemble['name']} (F1: {best_ensemble['f1_score']:.4f})")
        print(f"   Improvement: {improvement:+.4f} ({improvement/best_individual['f1_score']*100:+.2f}%)")
    
    return ensemble_results

def main():
    """Main function to test ensemble methods."""
    print("🚀 Content Moderation Ensemble Testing")
    print("=" * 60)
    
    # Load available models
    models, vectorizers = load_available_models()
    
    if not models:
        print("❌ No models found! Please train some models first.")
        print("   Run: python classical_models.py")
        print("   Or:  python train_improved_model.py")
        return
    
    # Load test data
    test_texts, test_labels = load_test_data()
    print(f"📊 Test data: {len(test_texts)} samples")
    print(f"   Toxic: {sum(test_labels)} ({sum(test_labels)/len(test_labels)*100:.1f}%)")
    print(f"   Safe:  {len(test_labels) - sum(test_labels)} ({(len(test_labels) - sum(test_labels))/len(test_labels)*100:.1f}%)")
    
    # Test ensembles
    ensemble_results = create_and_test_ensembles(models, vectorizers, test_texts, test_labels)
    
    print("\n" + "="*60)
    print("TESTING COMPLETED")
    print("="*60)
    
    if ensemble_results:
        print("✅ Ensemble methods successfully tested!")
        print("\n💡 Key Findings:")
        print("   - Ensemble methods can improve model performance")
        print("   - Different ensemble methods work better for different scenarios")
        print("   - Stacking often provides the best performance but is more complex")
        print("   - Voting ensembles are simpler and often provide good results")
    else:
        print("⚠️  No ensemble results available")
    
    print("\n🔧 Next Steps:")
    print("   - Train more diverse models for better ensemble performance")
    print("   - Experiment with different meta-classifiers for stacking")
    print("   - Use larger validation sets for better weight optimization")
    print("   - Consider feature-level ensembles for even better results")

if __name__ == "__main__":
    main()