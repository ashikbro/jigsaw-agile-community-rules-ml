#!/usr/bin/env python3
"""
Advanced Model Training Pipeline
Enhanced training with hyperparameter optimization, ensemble methods, and advanced features
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    RandomizedSearchCV, StratifiedKFold
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, VotingClassifier, BaggingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, f1_score
)
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import randint, uniform
import time
import warnings
warnings.filterwarnings('ignore')

class AdvancedModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_models = {}
        self.results = {}
        self.feature_importance = {}
        
    def create_enhanced_dataset(self, size=8000):
        """Create a more challenging and realistic dataset"""
        np.random.seed(self.random_state)
        
        # Enhanced toxic patterns
        toxic_patterns = [
            "you are stupid and worthless", "go kill yourself", "i hate you so much",
            "you're a complete idiot", "shut up moron", "you suck at everything",
            "nobody likes you", "you're pathetic", "get lost loser",
            "you're disgusting", "i wish you were dead", "you're a waste of space",
            "you're so annoying", "you make me sick", "you're a failure",
            "you're absolutely terrible", "i can't stand you", "you're worthless trash",
            "you should be ashamed", "you're completely useless", "you're so dumb",
            "i hope you suffer", "you're a horrible person", "you deserve nothing",
            "you're beyond help", "you're a complete disaster", "you're so ugly",
            "you're mentally ill", "you're a psychopath", "you're insane"
        ]
        
        # Enhanced normal patterns
        normal_patterns = [
            "thank you for sharing your thoughts", "i appreciate your perspective",
            "that's an interesting point of view", "i respectfully disagree",
            "could you explain more about this", "i think we should consider",
            "this is a great discussion", "i learned something new today",
            "thanks for the helpful information", "i agree with your analysis",
            "this makes a lot of sense", "good point about the topic",
            "i understand your concern", "let's work together on this",
            "that's a valid observation", "i see what you mean",
            "thanks for clarifying", "this is very informative",
            "i appreciate the feedback", "great job on the research",
            "this helps me understand better", "excellent explanation",
            "i'm glad we could discuss this", "thanks for being patient",
            "this is a thoughtful response", "i value your input",
            "let's continue this conversation", "i'm interested to hear more",
            "this adds valuable context", "i appreciate your time"
        ]
        
        # Generate samples with more complexity
        texts = []
        labels = []
        
        # Generate toxic samples (25% of dataset)
        toxic_count = int(size * 0.25)
        for i in range(toxic_count):
            # Mix patterns and add noise
            base_text = np.random.choice(toxic_patterns)
            
            # Add variations
            if np.random.random() < 0.3:
                base_text = base_text.upper()  # Shouting
            if np.random.random() < 0.4:
                base_text += "!!!" * np.random.randint(1, 4)  # Multiple exclamations
            if np.random.random() < 0.2:
                base_text = base_text.replace(' ', '  ')  # Extra spaces
            if np.random.random() < 0.3:
                # Add random words
                noise_words = ['really', 'totally', 'absolutely', 'completely', 'seriously']
                base_text = np.random.choice(noise_words) + ' ' + base_text
            
            texts.append(base_text)
            labels.append(1)
        
        # Generate normal samples (75% of dataset)
        normal_count = size - toxic_count
        for i in range(normal_count):
            base_text = np.random.choice(normal_patterns)
            
            # Add variations to normal text too
            if np.random.random() < 0.2:
                base_text += "."  # Proper punctuation
            if np.random.random() < 0.3:
                # Add polite words
                polite_words = ['please', 'kindly', 'perhaps', 'maybe', 'possibly']
                base_text = np.random.choice(polite_words) + ' ' + base_text
            
            texts.append(base_text)
            labels.append(0)
        
        # Create DataFrame
        df = pd.DataFrame({
            'text': texts,
            'target': labels
        })
        
        # Shuffle the dataset
        df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        return df
    
    def extract_advanced_features(self, df):
        """Extract comprehensive text features"""
        # Basic text features
        df['text_length'] = df['text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
        df['char_count'] = df['text'].str.replace(' ', '').str.len()
        df['sentence_count'] = df['text'].str.count(r'[.!?]') + 1
        df['avg_word_length'] = df['char_count'] / df['word_count']
        
        # Advanced features
        df['punctuation_count'] = df['text'].str.count(r'[^\w\s]')
        df['uppercase_count'] = df['text'].str.count(r'[A-Z]')
        df['digit_count'] = df['text'].str.count(r'\d')
        df['exclamation_count'] = df['text'].str.count('!')
        df['question_count'] = df['text'].str.count('\\?')
        
        # Ratio features
        df['uppercase_ratio'] = df['uppercase_count'] / df['text_length']
        df['punctuation_ratio'] = df['punctuation_count'] / df['text_length']
        df['digit_ratio'] = df['digit_count'] / df['text_length']
        
        # Sentiment-like features
        df['repeated_chars'] = df['text'].str.count(r'(.)\1{2,}')
        df['caps_words'] = df['text'].str.count(r'\b[A-Z]{2,}\b')
        df['multiple_exclamations'] = df['text'].str.count(r'!{2,}')
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for training"""
        # Extract advanced features
        df = self.extract_advanced_features(df)
        
        # TF-IDF features with advanced parameters
        tfidf = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 3),  # Include trigrams
            min_df=2,
            max_df=0.95,
            stop_words='english',
            sublinear_tf=True,
            analyzer='word'
        )
        
        tfidf_features = tfidf.fit_transform(df['text']).toarray()
        
        # Numerical features
        numerical_cols = [
            'text_length', 'word_count', 'char_count', 'sentence_count',
            'avg_word_length', 'punctuation_count', 'uppercase_count',
            'digit_count', 'exclamation_count', 'question_count',
            'uppercase_ratio', 'punctuation_ratio', 'digit_ratio',
            'repeated_chars', 'caps_words', 'multiple_exclamations'
        ]
        
        # Scale numerical features
        scaler = StandardScaler()
        numerical_features = scaler.fit_transform(df[numerical_cols])
        
        # Combine features
        X = np.hstack([tfidf_features, numerical_features])
        y = df['target'].values
        
        return X, y, tfidf, scaler
    
    def define_advanced_models(self):
        """Define models with hyperparameter search spaces"""
        self.models = {
            'random_forest': {
                'model': RandomForestClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            },
            'lightgbm': {
                'model': lgb.LGBMClassifier(random_state=self.random_state, verbose=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'num_leaves': [31, 50, 100],
                    'subsample': [0.8, 0.9]
                }
            },
            'extra_trees': {
                'model': ExtraTreesClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            'svm': {
                'model': SVC(random_state=self.random_state, probability=True),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            }
        }
    
    def train_with_optimization(self, X_train, y_train, cv_folds=5):
        """Train models with hyperparameter optimization"""
        print("🚀 Starting Advanced Model Training with Hyperparameter Optimization...\n")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for name, model_config in self.models.items():
            print(f"🔧 Optimizing {name.replace('_', ' ').title()}...")
            start_time = time.time()
            
            # Use RandomizedSearchCV for faster optimization
            search = RandomizedSearchCV(
                model_config['model'],
                model_config['params'],
                n_iter=20,  # Limit iterations for speed
                cv=cv,
                scoring='f1',
                n_jobs=-1,
                random_state=self.random_state
            )
            
            search.fit(X_train, y_train)
            
            # Store best model and results
            self.best_models[name] = search.best_estimator_
            
            training_time = time.time() - start_time
            
            # Cross-validation scores
            cv_scores = cross_val_score(
                search.best_estimator_, X_train, y_train, 
                cv=cv, scoring='f1', n_jobs=-1
            )
            
            self.results[name] = {
                'best_params': search.best_params_,
                'best_cv_score': search.best_score_,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_time': training_time
            }
            
            print(f"   ✅ Best F1: {search.best_score_:.4f} (±{cv_scores.std():.4f})")
            print(f"   ⏱️  Time: {training_time:.2f}s")
            print(f"   🎯 Best params: {search.best_params_}\n")
    
    def create_ensemble_models(self, X_train, y_train):
        """Create ensemble models from best performers"""
        print("🎭 Creating Ensemble Models...\n")
        
        # Get top 3 models by CV score
        sorted_models = sorted(
            self.results.items(), 
            key=lambda x: x[1]['cv_mean'], 
            reverse=True
        )[:3]
        
        top_models = [(name, self.best_models[name]) for name, _ in sorted_models]
        
        # Voting Classifier (Hard and Soft)
        voting_hard = VotingClassifier(
            estimators=top_models,
            voting='hard'
        )
        
        voting_soft = VotingClassifier(
            estimators=top_models,
            voting='soft'
        )
        
        # Bagging ensemble
        best_single_model = top_models[0][1]
        bagging = BaggingClassifier(
            base_estimator=best_single_model,
            n_estimators=10,
            random_state=self.random_state
        )
        
        # Train ensemble models
        ensemble_models = {
            'voting_hard': voting_hard,
            'voting_soft': voting_soft,
            'bagging': bagging
        }
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        for name, model in ensemble_models.items():
            print(f"🎯 Training {name.replace('_', ' ').title()}...")
            start_time = time.time()
            
            model.fit(X_train, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1
            )
            
            training_time = time.time() - start_time
            
            self.best_models[name] = model
            self.results[name] = {
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_time': training_time
            }
            
            print(f"   ✅ F1: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            print(f"   ⏱️  Time: {training_time:.2f}s\n")
    
    def evaluate_models(self, X_test, y_test):
        """Comprehensive model evaluation"""
        print("📊 Evaluating All Models on Test Set...\n")
        
        evaluation_results = {}
        
        for name, model in self.best_models.items():
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Metrics
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            
            evaluation_results[name] = {
                'f1_score': f1,
                'roc_auc': roc_auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"🎯 {name.replace('_', ' ').title()}:")
            print(f"   F1 Score: {f1:.4f}")
            if roc_auc:
                print(f"   ROC-AUC: {roc_auc:.4f}")
            print()
        
        return evaluation_results
    
    def generate_advanced_visualizations(self, X_test, y_test, evaluation_results):
        """Generate comprehensive visualizations"""
        print("📈 Generating Advanced Visualizations...\n")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Model Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Advanced Model Performance Analysis', fontsize=16, fontweight='bold')
        
        # F1 Scores
        models = list(evaluation_results.keys())
        f1_scores = [evaluation_results[m]['f1_score'] for m in models]
        
        axes[0, 0].barh(models, f1_scores, color='skyblue')
        axes[0, 0].set_xlabel('F1 Score')
        axes[0, 0].set_title('F1 Score Comparison')
        axes[0, 0].set_xlim(0, 1)
        
        # Add value labels
        for i, v in enumerate(f1_scores):
            axes[0, 0].text(v + 0.01, i, f'{v:.3f}', va='center')
        
        # ROC-AUC Scores
        roc_scores = [evaluation_results[m]['roc_auc'] for m in models if evaluation_results[m]['roc_auc']]
        roc_models = [m for m in models if evaluation_results[m]['roc_auc']]
        
        axes[0, 1].barh(roc_models, roc_scores, color='lightcoral')
        axes[0, 1].set_xlabel('ROC-AUC Score')
        axes[0, 1].set_title('ROC-AUC Comparison')
        axes[0, 1].set_xlim(0, 1)
        
        for i, v in enumerate(roc_scores):
            axes[0, 1].text(v + 0.01, i, f'{v:.3f}', va='center')
        
        # Training Time Comparison
        training_times = [self.results[m]['training_time'] for m in models]
        
        axes[1, 0].barh(models, training_times, color='lightgreen')
        axes[1, 0].set_xlabel('Training Time (seconds)')
        axes[1, 0].set_title('Training Time Comparison')
        
        for i, v in enumerate(training_times):
            axes[1, 0].text(v + max(training_times)*0.01, i, f'{v:.1f}s', va='center')
        
        # CV Score Distribution
        cv_means = [self.results[m]['cv_mean'] for m in models]
        cv_stds = [self.results[m]['cv_std'] for m in models]
        
        axes[1, 1].barh(models, cv_means, xerr=cv_stds, color='gold', capsize=5)
        axes[1, 1].set_xlabel('Cross-Validation F1 Score')
        axes[1, 1].set_title('CV Performance with Error Bars')
        axes[1, 1].set_xlim(0, 1)
        
        plt.tight_layout()
        plt.savefig('eda/advanced_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ROC Curves
        plt.figure(figsize=(12, 8))
        
        for name in models:
            if evaluation_results[name]['roc_auc']:
                fpr, tpr, _ = roc_curve(y_test, evaluation_results[name]['probabilities'])
                auc_score = evaluation_results[name]['roc_auc']
                plt.plot(fpr, tpr, label=f"{name.replace('_', ' ').title()} (AUC = {auc_score:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Advanced Models')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('eda/advanced_roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Confusion Matrices for Top 3 Models
        top_3_models = sorted(models, key=lambda x: evaluation_results[x]['f1_score'], reverse=True)[:3]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Confusion Matrices - Top 3 Models', fontsize=16, fontweight='bold')
        
        for i, model_name in enumerate(top_3_models):
            cm = confusion_matrix(y_test, evaluation_results[model_name]['predictions'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Normal', 'Toxic'], 
                       yticklabels=['Normal', 'Toxic'],
                       ax=axes[i])
            
            f1 = evaluation_results[model_name]['f1_score']
            axes[i].set_title(f"{model_name.replace('_', ' ').title()}\nF1: {f1:.3f}")
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('eda/advanced_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Advanced visualizations saved to eda/ directory")
    
    def run_complete_pipeline(self):
        """Run the complete advanced training pipeline"""
        print("🚀 Starting Advanced Model Training Pipeline\n")
        print("=" * 60)
        
        # 1. Create enhanced dataset
        print("📊 Creating Enhanced Dataset...")
        df = self.create_enhanced_dataset(size=8000)
        print(f"   Dataset size: {len(df)} samples")
        print(f"   Toxic ratio: {df['target'].mean():.1%}\n")
        
        # 2. Prepare features
        print("🔧 Preparing Advanced Features...")
        X, y, tfidf, scaler = self.prepare_features(df)
        print(f"   Feature dimensions: {X.shape}")
        print(f"   TF-IDF features: {X.shape[1] - 16}")
        print(f"   Numerical features: 16\n")
        
        # 3. Handle class imbalance
        print("⚖️ Handling Class Imbalance with SMOTEENN...")
        smoteenn = SMOTEENN(random_state=self.random_state)
        X_balanced, y_balanced = smoteenn.fit_resample(X, y)
        print(f"   Original shape: {X.shape}")
        print(f"   Balanced shape: {X_balanced.shape}")
        print(f"   New toxic ratio: {y_balanced.mean():.1%}\n")
        
        # 4. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=0.2, 
            random_state=self.random_state, stratify=y_balanced
        )
        
        # 5. Define models
        self.define_advanced_models()
        
        # 6. Train with optimization
        self.train_with_optimization(X_train, y_train)
        
        # 7. Create ensemble models
        self.create_ensemble_models(X_train, y_train)
        
        # 8. Evaluate models
        evaluation_results = self.evaluate_models(X_test, y_test)
        
        # 9. Generate visualizations
        self.generate_advanced_visualizations(X_test, y_test, evaluation_results)
        
        # 10. Print final summary
        self.print_final_summary(evaluation_results)
        
        return evaluation_results
    
    def print_final_summary(self, evaluation_results):
        """Print comprehensive results summary"""
        print("\n" + "=" * 60)
        print("🏆 FINAL RESULTS SUMMARY")
        print("=" * 60)
        
        # Sort by F1 score
        sorted_results = sorted(
            evaluation_results.items(), 
            key=lambda x: x[1]['f1_score'], 
            reverse=True
        )
        
        print("\n📊 Model Rankings (by F1 Score):")
        print("-" * 50)
        
        for i, (name, results) in enumerate(sorted_results, 1):
            f1 = results['f1_score']
            roc_auc = results['roc_auc']
            training_time = self.results[name]['training_time']
            
            print(f"{i}. {name.replace('_', ' ').title()}")
            print(f"   F1 Score: {f1:.4f}")
            if roc_auc:
                print(f"   ROC-AUC: {roc_auc:.4f}")
            print(f"   Training Time: {training_time:.2f}s")
            print()
        
        # Best model details
        best_model_name, best_results = sorted_results[0]
        print(f"🥇 Best Model: {best_model_name.replace('_', ' ').title()}")
        print(f"   F1 Score: {best_results['f1_score']:.4f}")
        if best_results['roc_auc']:
            print(f"   ROC-AUC: {best_results['roc_auc']:.4f}")
        
        if best_model_name in self.results:
            print(f"   CV Mean: {self.results[best_model_name]['cv_mean']:.4f}")
            print(f"   CV Std: {self.results[best_model_name]['cv_std']:.4f}")
            if 'best_params' in self.results[best_model_name]:
                print(f"   Best Params: {self.results[best_model_name]['best_params']}")
        
        print("\n🎯 Key Improvements Achieved:")
        print("   ✅ Hyperparameter optimization with RandomizedSearchCV")
        print("   ✅ Advanced feature engineering (TF-IDF + numerical)")
        print("   ✅ Class imbalance handling with SMOTEENN")
        print("   ✅ Ensemble methods (Voting, Bagging)")
        print("   ✅ Comprehensive evaluation metrics")
        print("   ✅ Advanced gradient boosting (XGBoost, LightGBM)")
        
        print("\n📁 Generated Files:")
        print("   📊 eda/advanced_model_comparison.png")
        print("   📈 eda/advanced_roc_curves.png")
        print("   🎯 eda/advanced_confusion_matrices.png")

def main():
    """Main execution function"""
    trainer = AdvancedModelTrainer(random_state=42)
    results = trainer.run_complete_pipeline()
    return trainer, results

if __name__ == "__main__":
    trainer, results = main()