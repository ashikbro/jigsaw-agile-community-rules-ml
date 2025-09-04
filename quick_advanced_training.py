#!/usr/bin/env python3
"""
Quick Advanced Model Training Pipeline
Faster version with optimized parameters for demonstration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    StratifiedKFold
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, f1_score, accuracy_score, precision_score, recall_score
)
from imblearn.over_sampling import SMOTE
import time
import warnings
warnings.filterwarnings('ignore')

class QuickAdvancedTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_models = {}
        self.results = {}
        
    def create_enhanced_dataset(self, size=6000):
        """Create a challenging and realistic dataset"""
        np.random.seed(self.random_state)
        
        # Enhanced toxic patterns with more variety
        toxic_patterns = [
            "you are absolutely worthless and stupid", "go kill yourself now", 
            "i hate you so much it hurts", "you're a complete moron idiot", 
            "shut up you pathetic loser", "you suck at everything badly",
            "nobody likes you at all", "you're totally pathetic", 
            "get lost you stupid loser", "you're disgusting and ugly",
            "i wish you were dead seriously", "you're a waste of space",
            "you're so annoying and dumb", "you make me sick badly", 
            "you're a complete failure", "you're absolutely terrible",
            "i can't stand you anymore", "you're worthless trash",
            "you should be ashamed deeply", "you're completely useless",
            "you're so incredibly dumb", "i hope you suffer badly",
            "you're a horrible person", "you deserve absolutely nothing",
            "you're beyond any help", "you're a complete disaster",
            "you're so ugly inside", "you're mentally disturbed",
            "you're a total psychopath", "you're completely insane"
        ]
        
        # Enhanced normal patterns
        normal_patterns = [
            "thank you for sharing your thoughtful perspective", 
            "i really appreciate your detailed viewpoint",
            "that's a very interesting point of view", 
            "i respectfully disagree with your opinion",
            "could you please explain more about this topic", 
            "i think we should carefully consider this",
            "this is a really great discussion", 
            "i learned something very new today",
            "thanks for the helpful and useful information", 
            "i completely agree with your analysis",
            "this makes a lot of sense to me", 
            "good point about the important topic",
            "i understand your valid concern here", 
            "let's work together on this problem",
            "that's a very valid observation", 
            "i see exactly what you mean",
            "thanks for clarifying that point", 
            "this is very informative content",
            "i appreciate the constructive feedback", 
            "great job on the thorough research",
            "this helps me understand much better", 
            "excellent and clear explanation",
            "i'm glad we could discuss this", 
            "thanks for being so patient",
            "this is a thoughtful response", 
            "i value your input greatly",
            "let's continue this conversation", 
            "i'm interested to hear more",
            "this adds valuable context", 
            "i appreciate your time"
        ]
        
        texts = []
        labels = []
        
        # Generate toxic samples (30% of dataset for more challenge)
        toxic_count = int(size * 0.30)
        for i in range(toxic_count):
            base_text = np.random.choice(toxic_patterns)
            
            # Add realistic variations
            if np.random.random() < 0.4:
                base_text = base_text.upper()  # Shouting
            if np.random.random() < 0.5:
                base_text += "!!!" * np.random.randint(1, 5)  # Multiple exclamations
            if np.random.random() < 0.3:
                base_text = base_text.replace(' ', '  ')  # Extra spaces
            if np.random.random() < 0.4:
                noise_words = ['really', 'totally', 'absolutely', 'completely', 'seriously', 'extremely']
                base_text = np.random.choice(noise_words) + ' ' + base_text
            if np.random.random() < 0.2:
                base_text += " " + np.random.choice(["damn", "hell", "crap", "stupid"])
            
            texts.append(base_text)
            labels.append(1)
        
        # Generate normal samples (70% of dataset)
        normal_count = size - toxic_count
        for i in range(normal_count):
            base_text = np.random.choice(normal_patterns)
            
            # Add variations to normal text
            if np.random.random() < 0.3:
                base_text += "."  # Proper punctuation
            if np.random.random() < 0.4:
                polite_words = ['please', 'kindly', 'perhaps', 'maybe', 'possibly', 'hopefully']
                base_text = np.random.choice(polite_words) + ' ' + base_text
            if np.random.random() < 0.2:
                base_text = "I think " + base_text
            
            texts.append(base_text)
            labels.append(0)
        
        # Create DataFrame and shuffle
        df = pd.DataFrame({'text': texts, 'target': labels})
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
        df['question_count'] = df['text'].str.count(r'\?')
        
        # Ratio features
        df['uppercase_ratio'] = df['uppercase_count'] / df['text_length']
        df['punctuation_ratio'] = df['punctuation_count'] / df['text_length']
        df['digit_ratio'] = df['digit_count'] / df['text_length']
        
        # Sentiment-like features
        df['repeated_chars'] = df['text'].str.count(r'(.)\1{2,}')
        df['caps_words'] = df['text'].str.count(r'\b[A-Z]{2,}\b')
        df['multiple_exclamations'] = df['text'].str.count(r'!{2,}')
        
        # Toxicity indicators
        df['negative_words'] = df['text'].str.lower().str.count(
            r'\b(hate|stupid|idiot|kill|die|ugly|worthless|pathetic)\b'
        )
        df['positive_words'] = df['text'].str.lower().str.count(
            r'\b(thank|appreciate|great|good|excellent|helpful|thoughtful)\b'
        )
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for training"""
        # Extract advanced features
        df = self.extract_advanced_features(df)
        
        # TF-IDF features with optimized parameters
        tfidf = TfidfVectorizer(
            max_features=1500,
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=2,
            max_df=0.95,
            stop_words='english',
            sublinear_tf=True
        )
        
        tfidf_features = tfidf.fit_transform(df['text']).toarray()
        
        # Numerical features
        numerical_cols = [
            'text_length', 'word_count', 'char_count', 'sentence_count',
            'avg_word_length', 'punctuation_count', 'uppercase_count',
            'digit_count', 'exclamation_count', 'question_count',
            'uppercase_ratio', 'punctuation_ratio', 'digit_ratio',
            'repeated_chars', 'caps_words', 'multiple_exclamations',
            'negative_words', 'positive_words'
        ]
        
        # Scale numerical features
        scaler = StandardScaler()
        numerical_features = scaler.fit_transform(df[numerical_cols])
        
        # Combine features
        X = np.hstack([tfidf_features, numerical_features])
        y = df['target'].values
        
        return X, y, tfidf, scaler
    
    def define_optimized_models(self):
        """Define models with focused parameter grids"""
        self.models = {
            'random_forest_optimized': {
                'model': RandomForestClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [150, 200],
                    'max_depth': [15, 20],
                    'min_samples_split': [2, 5],
                    'max_features': ['sqrt', 'log2']
                }
            },
            'gradient_boosting_optimized': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 150],
                    'learning_rate': [0.1, 0.15],
                    'max_depth': [5, 7],
                    'subsample': [0.8, 0.9]
                }
            },
            'extra_trees_optimized': {
                'model': ExtraTreesClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [150, 200],
                    'max_depth': [15, 20],
                    'min_samples_split': [2, 5]
                }
            },
            'logistic_regression_optimized': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'params': {
                    'C': [1, 10, 50],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            },
            'svm_optimized': {
                'model': SVC(random_state=self.random_state, probability=True),
                'params': {
                    'C': [1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale']
                }
            }
        }
    
    def train_with_optimization(self, X_train, y_train, cv_folds=3):
        """Train models with hyperparameter optimization"""
        print("🚀 Starting Quick Advanced Model Training...\n")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for name, model_config in self.models.items():
            print(f"🔧 Optimizing {name.replace('_', ' ').title()}...")
            start_time = time.time()
            
            # Use GridSearchCV for thorough but fast optimization
            search = GridSearchCV(
                model_config['model'],
                model_config['params'],
                cv=cv,
                scoring='f1',
                n_jobs=-1
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
    
    def create_ensemble_model(self, X_train, y_train):
        """Create ensemble from top 3 models"""
        print("🎭 Creating Ensemble Model...\n")
        
        # Get top 3 models by CV score
        sorted_models = sorted(
            self.results.items(), 
            key=lambda x: x[1]['cv_mean'], 
            reverse=True
        )[:3]
        
        top_models = [(name, self.best_models[name]) for name, _ in sorted_models]
        
        # Voting Classifier (Soft voting for probabilities)
        voting_ensemble = VotingClassifier(
            estimators=top_models,
            voting='soft'
        )
        
        print(f"🎯 Training Ensemble (Top 3 Models)...")
        start_time = time.time()
        
        voting_ensemble.fit(X_train, y_train)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(
            voting_ensemble, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1
        )
        
        training_time = time.time() - start_time
        
        self.best_models['ensemble_voting'] = voting_ensemble
        self.results['ensemble_voting'] = {
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
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Comprehensive metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            evaluation_results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"🎯 {name.replace('_', ' ').title()}:")
            print(f"   Accuracy:  {accuracy:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall:    {recall:.4f}")
            print(f"   F1 Score:  {f1:.4f}")
            print(f"   ROC-AUC:   {roc_auc:.4f}")
            print()
        
        return evaluation_results
    
    def generate_visualizations(self, X_test, y_test, evaluation_results):
        """Generate comprehensive visualizations"""
        print("📈 Generating Advanced Visualizations...\n")
        
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Comprehensive Model Comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Quick Advanced Model Performance Analysis', fontsize=16, fontweight='bold')
        
        models = list(evaluation_results.keys())
        
        # Metrics comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            if i < 5:  # First 5 subplots
                row, col = i // 3, i % 3
                values = [evaluation_results[m][metric] for m in models]
                
                bars = axes[row, col].barh(models, values, color=sns.color_palette("husl", len(models)))
                axes[row, col].set_xlabel(metric_name)
                axes[row, col].set_title(f'{metric_name} Comparison')
                axes[row, col].set_xlim(0, 1)
                
                # Add value labels
                for j, v in enumerate(values):
                    axes[row, col].text(v + 0.01, j, f'{v:.3f}', va='center')
        
        # Training time comparison
        training_times = [self.results[m]['training_time'] for m in models]
        axes[1, 2].barh(models, training_times, color='lightgreen')
        axes[1, 2].set_xlabel('Training Time (seconds)')
        axes[1, 2].set_title('Training Time Comparison')
        
        for i, v in enumerate(training_times):
            axes[1, 2].text(v + max(training_times)*0.01, i, f'{v:.1f}s', va='center')
        
        plt.tight_layout()
        plt.savefig('eda/quick_advanced_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ROC Curves
        plt.figure(figsize=(12, 8))
        
        for name in models:
            fpr, tpr, _ = roc_curve(y_test, evaluation_results[name]['probabilities'])
            auc_score = evaluation_results[name]['roc_auc']
            plt.plot(fpr, tpr, label=f"{name.replace('_', ' ').title()} (AUC = {auc_score:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Quick Advanced Models')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('eda/quick_advanced_roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Confusion Matrices for Top 3 Models
        top_3_models = sorted(models, key=lambda x: evaluation_results[x]['f1_score'], reverse=True)[:3]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Confusion Matrices - Top 3 Quick Advanced Models', fontsize=16, fontweight='bold')
        
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
        plt.savefig('eda/quick_advanced_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Quick advanced visualizations saved to eda/ directory")
    
    def run_complete_pipeline(self):
        """Run the complete quick advanced training pipeline"""
        print("🚀 Starting Quick Advanced Model Training Pipeline\n")
        print("=" * 60)
        
        # 1. Create enhanced dataset
        print("📊 Creating Enhanced Dataset...")
        df = self.create_enhanced_dataset(size=6000)
        print(f"   Dataset size: {len(df)} samples")
        print(f"   Toxic ratio: {df['target'].mean():.1%}\n")
        
        # 2. Prepare features
        print("🔧 Preparing Advanced Features...")
        X, y, tfidf, scaler = self.prepare_features(df)
        print(f"   Feature dimensions: {X.shape}")
        print(f"   TF-IDF features: {X.shape[1] - 18}")
        print(f"   Numerical features: 18\n")
        
        # 3. Handle class imbalance
        print("⚖️ Handling Class Imbalance with SMOTE...")
        smote = SMOTE(random_state=self.random_state)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        print(f"   Original shape: {X.shape}")
        print(f"   Balanced shape: {X_balanced.shape}")
        print(f"   New toxic ratio: {y_balanced.mean():.1%}\n")
        
        # 4. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=0.2, 
            random_state=self.random_state, stratify=y_balanced
        )
        
        # 5. Define models
        self.define_optimized_models()
        
        # 6. Train with optimization
        self.train_with_optimization(X_train, y_train)
        
        # 7. Create ensemble model
        self.create_ensemble_model(X_train, y_train)
        
        # 8. Evaluate models
        evaluation_results = self.evaluate_models(X_test, y_test)
        
        # 9. Generate visualizations
        self.generate_visualizations(X_test, y_test, evaluation_results)
        
        # 10. Print final summary
        self.print_final_summary(evaluation_results)
        
        return evaluation_results
    
    def print_final_summary(self, evaluation_results):
        """Print comprehensive results summary"""
        print("\n" + "=" * 60)
        print("🏆 QUICK ADVANCED TRAINING RESULTS")
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
            print(f"{i}. {name.replace('_', ' ').title()}")
            print(f"   F1 Score:  {results['f1_score']:.4f}")
            print(f"   Accuracy:  {results['accuracy']:.4f}")
            print(f"   Precision: {results['precision']:.4f}")
            print(f"   Recall:    {results['recall']:.4f}")
            print(f"   ROC-AUC:   {results['roc_auc']:.4f}")
            print(f"   Training:  {self.results[name]['training_time']:.2f}s")
            print()
        
        # Best model details
        best_model_name, best_results = sorted_results[0]
        print(f"🥇 Best Model: {best_model_name.replace('_', ' ').title()}")
        print(f"   F1 Score: {best_results['f1_score']:.4f}")
        print(f"   ROC-AUC:  {best_results['roc_auc']:.4f}")
        print(f"   CV Mean:  {self.results[best_model_name]['cv_mean']:.4f}")
        print(f"   CV Std:   {self.results[best_model_name]['cv_std']:.4f}")
        print(f"   Best Params: {self.results[best_model_name]['best_params']}")
        
        print("\n🎯 Key Improvements Achieved:")
        print("   ✅ Hyperparameter optimization with GridSearchCV")
        print("   ✅ Advanced feature engineering (1500 TF-IDF + 18 numerical)")
        print("   ✅ Enhanced dataset with realistic noise and patterns")
        print("   ✅ Class imbalance handling with SMOTE")
        print("   ✅ Ensemble method (Soft Voting)")
        print("   ✅ Comprehensive evaluation metrics")
        print("   ✅ Optimized model parameters for better performance")
        
        print("\n📁 Generated Files:")
        print("   📊 eda/quick_advanced_comparison.png")
        print("   📈 eda/quick_advanced_roc_curves.png")
        print("   🎯 eda/quick_advanced_confusion_matrices.png")

def main():
    """Main execution function"""
    trainer = QuickAdvancedTrainer(random_state=42)
    results = trainer.run_complete_pipeline()
    return trainer, results

if __name__ == "__main__":
    trainer, results = main()