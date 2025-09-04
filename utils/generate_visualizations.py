#!/usr/bin/env python3
"""
Generate Histograms and Confusion Matrices for Jigsaw Toxic Comment Classification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (12, 8)

def create_sample_data(n_samples=5000):
    """
    Create sample toxic comment data for demonstration
    """
    np.random.seed(42)
    
    # Sample comments (simplified for demo)
    toxic_words = ['hate', 'stupid', 'idiot', 'kill', 'die', 'moron', 'dumb']
    normal_words = ['good', 'nice', 'thanks', 'help', 'please', 'great', 'awesome']
    
    comments = []
    labels = []
    
    for i in range(n_samples):
        if np.random.random() < 0.1:  # 10% toxic
            # Create toxic comment
            words = np.random.choice(toxic_words, size=np.random.randint(3, 8))
            comment = ' '.join(words) + ' comment ' + str(i)
            labels.append(1)
        else:
            # Create normal comment
            words = np.random.choice(normal_words, size=np.random.randint(3, 8))
            comment = ' '.join(words) + ' comment ' + str(i)
            labels.append(0)
        
        comments.append(comment)
    
    # Create DataFrame
    df = pd.DataFrame({
        'id': range(n_samples),
        'comment_text': comments,
        'toxic': labels
    })
    
    # Add derived features
    df['text_length'] = df['comment_text'].str.len()
    df['word_count'] = df['comment_text'].str.split().str.len()
    df['char_count'] = df['comment_text'].str.replace(' ', '').str.len()
    df['uppercase_count'] = df['comment_text'].str.count(r'[A-Z]')
    df['punctuation_count'] = df['comment_text'].str.count(r'[.,!?;:]')
    
    return df

def generate_histograms(df):
    """
    Generate histograms for numerical features
    """
    print("📊 Generating Histograms...")
    
    # Numerical columns for histograms
    numerical_cols = ['text_length', 'word_count', 'char_count', 'uppercase_count', 'punctuation_count']
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(numerical_cols):
        # Overall histogram
        axes[i].hist(df[col], bins=30, alpha=0.7, color='skyblue', edgecolor='black', label='All')
        
        # Separate by toxicity
        toxic_data = df[df['toxic'] == 1][col]
        normal_data = df[df['toxic'] == 0][col]
        
        axes[i].hist(toxic_data, bins=30, alpha=0.5, color='red', label='Toxic')
        axes[i].hist(normal_data, bins=30, alpha=0.5, color='green', label='Normal')
        
        axes[i].set_title(f'Distribution of {col.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel(col.replace('_', ' ').title())
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = df[col].mean()
        axes[i].axvline(mean_val, color='black', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.1f}')
    
    # Hide the last subplot if not needed
    axes[-1].set_visible(False)
    
    plt.suptitle('📊 Feature Distributions by Toxicity Label', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('eda/histograms_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional histogram: Target variable distribution
    plt.figure(figsize=(10, 6))
    toxicity_counts = df['toxic'].value_counts()
    plt.bar(['Normal (0)', 'Toxic (1)'], toxicity_counts.values, color=['green', 'red'], alpha=0.7)
    plt.title('🎯 Target Variable Distribution', fontsize=14, fontweight='bold')
    plt.ylabel('Count')
    plt.xlabel('Toxicity Label')
    
    # Add percentage labels
    total = len(df)
    for i, count in enumerate(toxicity_counts.values):
        percentage = (count / total) * 100
        plt.text(i, count + 50, f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.savefig('eda/target_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Histograms generated and saved!")

def train_models_and_generate_confusion_matrices(df):
    """
    Train models and generate confusion matrices
    """
    print("🤖 Training Models and Generating Confusion Matrices...")
    
    # Prepare features using TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_text = vectorizer.fit_transform(df['comment_text'])
    
    # Combine with numerical features
    numerical_features = df[['text_length', 'word_count', 'char_count', 'uppercase_count', 'punctuation_count']]
    
    # Convert sparse matrix to dense for concatenation
    X_text_dense = X_text.toarray()
    X = np.hstack([X_text_dense, numerical_features.values])
    y = df['toxic'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, kernel='linear')
    }
    
    # Train models and generate confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    results = []
    
    for i, (name, model) in enumerate(models.items()):
        print(f"Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                   xticklabels=['Normal', 'Toxic'], yticklabels=['Normal', 'Toxic'])
        axes[i].set_title(f'{name}\nAccuracy: {accuracy:.3f}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
        
        # Store results
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'True_Negatives': cm[0, 0],
            'False_Positives': cm[0, 1],
            'False_Negatives': cm[1, 0],
            'True_Positives': cm[1, 1]
        })
        
        print(f"  ✅ {name} - Accuracy: {accuracy:.3f}")
    
    plt.suptitle('🔍 Confusion Matrices for Different Models', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('eda/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create results summary
    results_df = pd.DataFrame(results)
    print("\n📋 Model Performance Summary:")
    print("=" * 50)
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv('eda/model_results.csv', index=False)
    
    # Generate detailed classification reports
    print("\n📊 Detailed Classification Reports:")
    print("=" * 50)
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        print(f"\n{name}:")
        print("-" * 30)
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Toxic']))
    
    print("✅ Confusion matrices and model evaluation completed!")
    
    return results_df

def generate_additional_visualizations(df):
    """
    Generate additional useful visualizations
    """
    print("📈 Generating Additional Visualizations...")
    
    # 1. Text length by toxicity
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    df.boxplot(column='text_length', by='toxic', ax=plt.gca())
    plt.title('Text Length Distribution by Toxicity')
    plt.xlabel('Toxicity Label (0=Normal, 1=Toxic)')
    plt.ylabel('Text Length')
    
    plt.subplot(1, 2, 2)
    df.boxplot(column='word_count', by='toxic', ax=plt.gca())
    plt.title('Word Count Distribution by Toxicity')
    plt.xlabel('Toxicity Label (0=Normal, 1=Toxic)')
    plt.ylabel('Word Count')
    
    plt.suptitle('📝 Text Characteristics by Toxicity', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('eda/text_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Correlation heatmap
    plt.figure(figsize=(10, 8))
    numerical_cols = ['text_length', 'word_count', 'char_count', 'uppercase_count', 'punctuation_count', 'toxic']
    correlation_matrix = df[numerical_cols].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f')
    plt.title('🔗 Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('eda/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Additional visualizations completed!")

def main():
    """
    Main function to generate all visualizations
    """
    print("🚀 Starting Visualization Generation for Jigsaw Toxic Comment Classification")
    print("=" * 80)
    
    # Create sample data
    print("📝 Creating sample dataset...")
    df = create_sample_data(5000)
    print(f"✅ Dataset created with {len(df)} samples")
    print(f"   - Normal comments: {len(df[df['toxic'] == 0])}")
    print(f"   - Toxic comments: {len(df[df['toxic'] == 1])}")
    
    # Generate histograms
    generate_histograms(df)
    
    # Train models and generate confusion matrices
    results_df = train_models_and_generate_confusion_matrices(df)
    
    # Generate additional visualizations
    generate_additional_visualizations(df)
    
    print("\n🎉 All visualizations completed successfully!")
    print("📁 Files saved in 'eda/' directory:")
    print("   - histograms_analysis.png")
    print("   - target_distribution.png")
    print("   - confusion_matrices.png")
    print("   - text_analysis.png")
    print("   - correlation_matrix.png")
    print("   - model_results.csv")
    
    return df, results_df

if __name__ == "__main__":
    # Create eda directory if it doesn't exist
    import os
    os.makedirs('eda', exist_ok=True)
    
    # Run main function
    df, results = main()