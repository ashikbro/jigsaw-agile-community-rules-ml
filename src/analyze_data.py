"""
Analyze the Jigsaw Agile Community Rules dataset
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Set up paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

# Set plot style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

def load_data():
    """Load the dataset."""
    train_path = DATA_DIR / 'train.csv'
    test_path = DATA_DIR / 'test.csv'
    
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, test_df

def basic_eda(df, df_name):
    """Perform basic exploratory data analysis."""
    print(f"\n=== {df_name} Dataset Analysis ===")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    print("\nBasic statistics:")
    print(df.describe(include='all'))

def analyze_text_length(df, text_col, label_col=None):
    """Analyze text length distribution."""
    df['text_length'] = df[text_col].astype(str).apply(len)
    
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='text_length', bins=50, hue=label_col if label_col else None, 
                element='step', fill=True, alpha=0.5)
    plt.title(f'Distribution of Text Lengths in {text_col}')
    plt.xlabel('Text Length')
    plt.ylabel('Count')
    
    # Save the plot
    plot_path = OUTPUT_DIR / f'text_length_distribution_{text_col}.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved text length distribution plot to {plot_path}")
    
    return df

def analyze_label_distribution(df, label_col):
    """Analyze the distribution of labels."""
    if label_col not in df.columns:
        print(f"Label column '{label_col}' not found in the dataset.")
        return
    
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=df, x=label_col)
    plt.title(f'Distribution of {label_col}')
    
    # Add count labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.0f}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', xytext=(0, 10), 
                   textcoords='offset points')
    
    # Save the plot
    plot_path = OUTPUT_DIR / f'label_distribution.png'
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved label distribution plot to {plot_path}")

def analyze_word_frequencies(df, text_col, label_col=None, n_most_common=20):
    """Analyze most common words in the text."""
    # Initialize NLTK components
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    def preprocess_text(text):
        # Tokenize and lemmatize
        tokens = word_tokenize(str(text).lower())
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        # Remove stopwords and non-alphabetic tokens
        tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
        return tokens
    
    # Get all words
    all_words = []
    for text in df[text_col]:
        all_words.extend(preprocess_text(text))
    
    # Get most common words
    word_freq = Counter(all_words)
    most_common = word_freq.most_common(n_most_common)
    
    # Plot most common words
    plt.figure(figsize=(12, 8))
    words, counts = zip(*most_common)
    sns.barplot(x=list(counts), y=list(words))
    plt.title(f'Top {n_most_common} Most Common Words')
    plt.xlabel('Frequency')
    plt.ylabel('Word')
    
    # Save the plot
    plot_path = OUTPUT_DIR / f'most_common_words_{text_col}.png'
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved most common words plot to {plot_path}")
    
    # If label column is provided, analyze word frequencies by label
    if label_col and label_col in df.columns:
        for label in sorted(df[label_col].unique()):
            label_texts = df[df[label_col] == label][text_col]
            label_words = []
            
            for text in label_texts:
                label_words.extend(preprocess_text(text))
            
            word_freq = Counter(label_words)
            most_common = word_freq.most_common(n_most_common)
            
            plt.figure(figsize=(12, 6))
            words, counts = zip(*most_common)
            sns.barplot(x=list(counts), y=list(words))
            plt.title(f'Top {n_most_common} Most Common Words (Label: {label})')
            plt.xlabel('Frequency')
            plt.ylabel('Word')
            
            # Save the plot
            plot_path = OUTPUT_DIR / f'most_common_words_{text_col}_label_{label}.png'
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved most common words plot for label {label} to {plot_path}")

def analyze_subreddit_distribution(df, subreddit_col='subreddit', label_col=None):
    """Analyze the distribution of subreddits."""
    if subreddit_col not in df.columns:
        print(f"Subreddit column '{subreddit_col}' not found in the dataset.")
        return
    
    # Get top N subreddits
    top_subreddits = df[subreddit_col].value_counts().nlargest(20)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_subreddits.values, y=top_subreddits.index)
    plt.title('Top 20 Subreddits by Number of Posts')
    plt.xlabel('Number of Posts')
    plt.ylabel('Subreddit')
    
    # Save the plot
    plot_path = OUTPUT_DIR / 'top_subreddits.png'
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved top subreddits plot to {plot_path}")
    
    # If label column is provided, analyze subreddit distribution by label
    if label_col and label_col in df.columns:
        for label in sorted(df[label_col].unique()):
            label_subreddits = df[df[label_col] == label][subreddit_col].value_counts().nlargest(10)
            
            plt.figure(figsize=(12, 6))
            sns.barplot(x=label_subreddits.values, y=label_subreddits.index)
            plt.title(f'Top 10 Subreddits for Label {label}')
            plt.xlabel('Number of Posts')
            plt.ylabel('Subreddit')
            
            # Save the plot
            plot_path = OUTPUT_DIR / f'top_subreddits_label_{label}.png'
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved top subreddits plot for label {label} to {plot_path}")

def main():
    """Main function to run the analysis."""
    print("Starting data analysis...")
    
    # Load data
    train_df, test_df = load_data()
    
    # Basic EDA
    basic_eda(train_df, 'Training')
    basic_eda(test_df, 'Test')
    
    # Analyze text length
    print("\nAnalyzing text length...")
    train_df = analyze_text_length(train_df, 'body', 'rule_violation' if 'rule_violation' in train_df.columns else None)
    
    # Analyze label distribution (if labels exist)
    if 'rule_violation' in train_df.columns:
        print("\nAnalyzing label distribution...")
        analyze_label_distribution(train_df, 'rule_violation')
    
    # Analyze word frequencies
    print("\nAnalyzing word frequencies...")
    analyze_word_frequencies(train_df, 'body', 'rule_violation' if 'rule_violation' in train_df.columns else None)
    
    # Analyze subreddit distribution
    if 'subreddit' in train_df.columns:
        print("\nAnalyzing subreddit distribution...")
        analyze_subreddit_distribution(
            train_df, 
            label_col='rule_violation' if 'rule_violation' in train_df.columns else None
        )
    
    print("\nData analysis completed! Check the 'outputs' directory for visualizations.")

if __name__ == "__main__":
    main()
