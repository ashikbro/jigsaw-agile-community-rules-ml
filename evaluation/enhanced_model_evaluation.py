#!/usr/bin/env python3
"""
Enhanced Model Evaluation with Realistic Performance Metrics
Creates more challenging dataset and advanced model comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
import time
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (15, 10)

def create_realistic_dataset(n_samples=10000):
    """
    Create a more realistic and challenging toxic comment dataset
    """
    np.random.seed(42)
    
    # More sophisticated word lists
    toxic_patterns = [
        ['hate', 'stupid', 'idiot', 'kill', 'die'],
        ['moron', 'dumb', 'loser', 'pathetic', 'worthless'],
        ['shut', 'up', 'annoying', 'irritating', 'disgusting'],
        ['racist', 'sexist', 'bigot', 'nazi', 'fascist'],
        ['trash', 'garbage', 'scum', 'filth', 'waste']
    ]
    
    normal_patterns = [
        ['good', 'nice', 'thanks', 'help', 'please'],
        ['great', 'awesome', 'wonderful', 'excellent', 'amazing'],
        ['interesting', 'informative', 'useful', 'helpful', 'valuable'],
        ['agree', 'disagree', 'opinion', 'think', 'believe'],
        ['question', 'answer', 'discussion', 'topic', 'subject']
    ]
    
    comments = []
    labels = []
    
    for i in range(n_samples):
        # Create more realistic class imbalance (15% toxic)
        if np.random.random() < 0.15:
            # Create toxic comment with varying complexity
            pattern = np.random.choice(len(toxic_patterns))
            words = list(np.random.choice(toxic_patterns[pattern], size=np.random.randint(2, 5)))
            
            # Add some normal words to make it more realistic (mixed content)
            if np.random.random() < 0.3:
                normal_words = np.random.choice(normal_patterns[0], size=np.random.randint(1, 3))
                words.extend(normal_words)
            
            # Add noise and variations
            if np.random.random() < 0.2:
                words.append('really')
            if np.random.random() < 0.1:
                words.append('very')
                
            comment = ' '.join(words) + f' comment {i}'
            labels.append(1)
        else:
            # Create normal comment
            pattern = np.random.choice(len(normal_patterns))
            words = list(np.random.choice(normal_patterns[pattern], size=np.random.randint(3, 7)))
            
            # Add some ambiguous words that might confuse models
            if np.random.random() < 0.1:
                ambiguous = ['bad', 'wrong', 'problem', 'issue', 'concern']
                words.append(np.random.choice(ambiguous))
            
            comment = ' '.join(words) + f' discussion {i}'
            labels.append(0)
        
        comments.append(comment)
    
    # Create DataFrame
    df = pd.DataFrame({
        'id': range(n_samples),
        'comment_text': comments,
        'toxic': labels
    })
    
    # Add more sophisticated derived features
    df['text_length'] = df['comment_text'].str.len()
    df['word_count'] = df['comment_text'].str.split().str.len()
    df['char_count'] = df['comment_text'].str.replace(' ', '').str.len()
    df['uppercase_count'] = df['comment_text'].str.count(r'[A-Z]')
    df['punctuation_count'] = df['comment_text'].str.count(r'[.,!?;:]')
    df['exclamation_count'] = df['comment_text'].str.count(r'!')
    df['question_count'] = df['comment_text'].str.count(r'\?')
    df['avg_word_length'] = df['char_count'] / df['word_count'].replace(0, 1)
    df['uppercase_ratio'] = df['uppercase_count'] / df['text_length'].replace(0, 1)
    
    # Add some noise to make classification more challenging
    noise_indices = np.random.choice(len(df), size=int(0.05 * len(df)), replace=False)
    df.loc[noise_indices, 'toxic'] = 1 - df.loc[noise_indices, 'toxic']  # Flip labels for 5% of data
    
    return df

def advanced_model_training(df):
    """
    Train advanced models with more sophisticated evaluation
    """
    print("🚀 Advanced Model Training and Evaluation")
    print("=" * 60)
    
    # Prepare features
    # Text features using TF-IDF with more parameters
    vectorizer = TfidfVectorizer(
        max_features=2000, 
        stop_words='english',
        ngram_range=(1, 2),  # Include bigrams
        min_df=2,
        max_df=0.95
    )
    X_text = vectorizer.fit_transform(df['comment_text'])
    
    # Numerical features
    numerical_features = [
        'text_length', 'word_count', 'char_count', 'uppercase_count', 
        'punctuation_count', 'exclamation_count', 'question_count',
        'avg_word_length', 'uppercase_ratio'
    ]
    
    # Scale numerical features
    scaler = StandardScaler()
    X_numerical = scaler.fit_transform(df[numerical_features])
    
    # Combine features
    X_text_dense = X_text.toarray()
    X = np.hstack([X_text_dense, X_numerical])
    y = df['toxic'].values
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"Original training set: {len(X_train)} samples")
    print(f"Balanced training set: {len(X_train_balanced)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Define advanced models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, 
            max_depth=10, 
            min_samples_split=5,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42, 
            max_iter=2000,
            C=0.1,
            class_weight='balanced'
        ),
        'SVM (RBF)': SVC(
            random_state=42, 
            probability=True,
            kernel='rbf',
            C=1.0,
            class_weight='balanced'
        ),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42,
            alpha=0.01
        ),
        'Ensemble Voting': VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
                ('lr', LogisticRegression(random_state=42, max_iter=1000))
            ],
            voting='soft'
        )
    }
    
    # Results storage
    results = []
    model_objects = {}
    
    print("\n🔄 Training models with cross-validation...")
    
    for name, model in models.items():
        print(f"\n📊 Training {name}...")
        
        # Cross-validation on balanced data
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=cv, scoring='f1')
        
        # Train model
        start_time = time.time()
        model.fit(X_train_balanced, y_train_balanced)
        training_time = time.time() - start_time
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Comprehensive metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # ROC AUC
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else roc_auc_score(y_test, y_pred)
        except:
            roc_auc = np.nan
        
        # Store results
        results.append({
            'Model': name,
            'CV_F1_Mean': cv_scores.mean(),
            'CV_F1_Std': cv_scores.std(),
            'Test_Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'ROC_AUC': roc_auc,
            'Training_Time': training_time
        })
        
        model_objects[name] = model
        
        print(f"  ✅ {name} completed")
        print(f"     Accuracy: {accuracy:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
    
    return results, model_objects, X_test, y_test, y_pred_proba

def generate_enhanced_visualizations(results, model_objects, X_test, y_test):
    """
    Generate comprehensive visualizations
    """
    print("\n📊 Generating Enhanced Visualizations...")
    
    results_df = pd.DataFrame(results)
    
    # 1. Comprehensive Model Comparison
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Accuracy comparison
    axes[0,0].bar(results_df['Model'], results_df['Test_Accuracy'], color='skyblue', alpha=0.8)
    axes[0,0].set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].set_ylim(0.7, 1.0)
    
    # F1 Score comparison
    axes[0,1].bar(results_df['Model'], results_df['F1_Score'], color='lightgreen', alpha=0.8)
    axes[0,1].set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
    axes[0,1].set_ylabel('F1 Score')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].set_ylim(0.7, 1.0)
    
    # Precision vs Recall
    axes[0,2].scatter(results_df['Recall'], results_df['Precision'], s=100, alpha=0.7)
    for i, model in enumerate(results_df['Model']):
        axes[0,2].annotate(model, (results_df['Recall'].iloc[i], results_df['Precision'].iloc[i]), 
                          xytext=(5, 5), textcoords='offset points', fontsize=9)
    axes[0,2].set_xlabel('Recall')
    axes[0,2].set_ylabel('Precision')
    axes[0,2].set_title('Precision vs Recall', fontsize=14, fontweight='bold')
    axes[0,2].grid(True, alpha=0.3)
    
    # ROC AUC comparison
    valid_roc = results_df.dropna(subset=['ROC_AUC'])
    axes[1,0].bar(valid_roc['Model'], valid_roc['ROC_AUC'], color='gold', alpha=0.8)
    axes[1,0].set_title('ROC AUC Comparison', fontsize=14, fontweight='bold')
    axes[1,0].set_ylabel('ROC AUC')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].set_ylim(0.8, 1.0)
    
    # Training time comparison
    axes[1,1].bar(results_df['Model'], results_df['Training_Time'], color='lightcoral', alpha=0.8)
    axes[1,1].set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    axes[1,1].set_ylabel('Time (seconds)')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    # Cross-validation F1 scores with error bars
    axes[1,2].bar(results_df['Model'], results_df['CV_F1_Mean'], 
                  yerr=results_df['CV_F1_Std'], capsize=5, color='purple', alpha=0.7)
    axes[1,2].set_title('Cross-Validation F1 Scores', fontsize=14, fontweight='bold')
    axes[1,2].set_ylabel('CV F1 Score')
    axes[1,2].tick_params(axis='x', rotation=45)
    axes[1,2].set_ylim(0.7, 1.0)
    
    plt.suptitle('🚀 Enhanced Model Performance Analysis', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('eda/enhanced_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Enhanced Confusion Matrices
    top_3_models = results_df.nlargest(3, 'F1_Score')['Model'].tolist()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, model_name in enumerate(top_3_models):
        model = model_objects[model_name]
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotations with both counts and percentages
        annotations = np.empty_like(cm).astype(str)
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                annotations[row, col] = f'{cm[row, col]}\n({cm_percent[row, col]:.1f}%)'
        
        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', ax=axes[i],
                   xticklabels=['Normal', 'Toxic'], yticklabels=['Normal', 'Toxic'])
        
        # Get metrics for title
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        axes[i].set_title(f'{model_name}\nAcc: {accuracy:.3f} | F1: {f1:.3f}', 
                         fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
    
    plt.suptitle('🎯 Enhanced Confusion Matrices (Top 3 Models)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('eda/enhanced_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. ROC Curves
    plt.figure(figsize=(12, 8))
    
    for model_name in top_3_models:
        model = model_objects[model_name]
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc_score:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('📈 ROC Curves Comparison', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig('eda/enhanced_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results_df

def main():
    """
    Main function for enhanced model evaluation
    """
    print("🚀 Enhanced Model Evaluation with Realistic Performance")
    print("=" * 70)
    
    # Create realistic dataset
    print("📝 Creating realistic and challenging dataset...")
    df = create_realistic_dataset(10000)
    
    print(f"✅ Dataset created with {len(df)} samples")
    print(f"   - Normal comments: {len(df[df['toxic'] == 0])} ({len(df[df['toxic'] == 0])/len(df)*100:.1f}%)")
    print(f"   - Toxic comments: {len(df[df['toxic'] == 1])} ({len(df[df['toxic'] == 1])/len(df)*100:.1f}%)")
    
    # Train advanced models
    results, model_objects, X_test, y_test, y_pred_proba = advanced_model_training(df)
    
    # Generate enhanced visualizations
    results_df = generate_enhanced_visualizations(results, model_objects, X_test, y_test)
    
    # Print comprehensive results
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE MODEL PERFORMANCE RESULTS")
    print("=" * 80)
    print(results_df.round(4).to_string(index=False))
    
    # Save results
    results_df.to_csv('eda/enhanced_model_results.csv', index=False)
    
    # Top performers
    print("\n🏆 TOP 3 MODELS BY F1 SCORE:")
    top_models = results_df.nlargest(3, 'F1_Score')
    for i, (_, row) in enumerate(top_models.iterrows(), 1):
        print(f"{i}. {row['Model']}: F1={row['F1_Score']:.4f}, Accuracy={row['Test_Accuracy']:.4f}, ROC-AUC={row['ROC_AUC']:.4f}")
    
    print("\n🎉 Enhanced evaluation completed successfully!")
    print("📁 New files generated:")
    print("   - enhanced_model_comparison.png")
    print("   - enhanced_confusion_matrices.png")
    print("   - enhanced_roc_curves.png")
    print("   - enhanced_model_results.csv")
    
    return df, results_df

if __name__ == "__main__":
    import os
    os.makedirs('eda', exist_ok=True)
    
    df, results = main()