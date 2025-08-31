"""Dataset Integration Module for Multiple Text Classification Datasets.

This module provides unified loaders for various text classification datasets
to support comprehensive academic analysis and cross-dataset evaluation.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
import requests
import zipfile
import tarfile
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetLoader:
    """Base class for dataset loaders."""
    
    def __init__(self, data_dir: str = "data/external"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url: str, filename: str) -> Path:
        """Download a file from URL if it doesn't exist."""
        filepath = self.data_dir / filename
        if not filepath.exists():
            logger.info(f"Downloading {filename}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Downloaded {filename}")
        return filepath
    
    def extract_archive(self, filepath: Path, extract_to: Optional[Path] = None) -> Path:
        """Extract archive file."""
        if extract_to is None:
            extract_to = filepath.parent / filepath.stem
        
        if not extract_to.exists():
            logger.info(f"Extracting {filepath.name}...")
            if filepath.suffix == '.zip':
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif filepath.suffix in ['.tar', '.gz']:
                with tarfile.open(filepath, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_to)
            logger.info(f"Extracted to {extract_to}")
        
        return extract_to


class JigsawToxicLoader(DatasetLoader):
    """Loader for Jigsaw Toxic Comment Classification dataset."""
    
    def load_data(self, sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load Jigsaw Toxic Comment dataset.
        
        Note: This is a placeholder implementation. In practice, you would:
        1. Download from Kaggle API or provide manual download instructions
        2. Load the actual CSV files
        
        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info("Loading Jigsaw Toxic Comment dataset...")
        
        # Placeholder: Generate synthetic data for demonstration
        # In practice, replace with actual data loading
        np.random.seed(42)
        
        toxic_comments = [
            "You are such an idiot and waste of space",
            "This is completely stupid and wrong",
            "I hate this so much, it's terrible",
            "What a moron, can't believe this",
            "This is absolutely disgusting behavior"
        ]
        
        clean_comments = [
            "This is a great article, thanks for sharing",
            "I appreciate your perspective on this topic",
            "Interesting point, I hadn't considered that",
            "Thank you for the helpful information",
            "This is a well-written and informative post"
        ]
        
        # Generate synthetic dataset
        n_samples = sample_size or 1000
        n_toxic = n_samples // 2
        n_clean = n_samples - n_toxic
        
        data = []
        for i in range(n_toxic):
            comment = np.random.choice(toxic_comments) + f" {i}"
            data.append({'comment_text': comment, 'toxic': 1})
        
        for i in range(n_clean):
            comment = np.random.choice(clean_comments) + f" {i}"
            data.append({'comment_text': comment, 'toxic': 0})
        
        df = pd.DataFrame(data)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['toxic'])
        
        logger.info(f"Loaded {len(train_df)} training and {len(test_df)} test samples")
        return train_df, test_df


class AGNewsLoader(DatasetLoader):
    """Loader for AG News dataset."""
    
    def load_data(self, sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load AG News dataset.
        
        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info("Loading AG News dataset...")
        
        # Placeholder: Generate synthetic news data
        np.random.seed(42)
        
        categories = ['World', 'Sports', 'Business', 'Sci/Tech']
        
        world_news = [
            "International summit discusses climate change policies",
            "New trade agreement signed between major countries",
            "Political tensions rise in the region"
        ]
        
        sports_news = [
            "Championship game ends with surprising victory",
            "New record set in Olympic trials",
            "Star player announces retirement from professional sports"
        ]
        
        business_news = [
            "Stock market reaches new highs amid economic growth",
            "Major corporation announces merger plans",
            "Startup receives significant funding round"
        ]
        
        tech_news = [
            "New AI breakthrough announced by research team",
            "Latest smartphone features revolutionary technology",
            "Cybersecurity concerns grow with new threats"
        ]
        
        news_by_category = {
            'World': world_news,
            'Sports': sports_news,
            'Business': business_news,
            'Sci/Tech': tech_news
        }
        
        # Generate synthetic dataset
        n_samples = sample_size or 1000
        n_per_category = n_samples // 4
        
        data = []
        for category in categories:
            for i in range(n_per_category):
                text = np.random.choice(news_by_category[category]) + f" {i}"
                data.append({'text': text, 'label': category})
        
        df = pd.DataFrame(data)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        
        logger.info(f"Loaded {len(train_df)} training and {len(test_df)} test samples")
        return train_df, test_df


class IMDBLoader(DatasetLoader):
    """Loader for IMDB Movie Reviews dataset."""
    
    def load_data(self, sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load IMDB Movie Reviews dataset.
        
        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info("Loading IMDB Movie Reviews dataset...")
        
        # Placeholder: Generate synthetic movie reviews
        np.random.seed(42)
        
        positive_reviews = [
            "This movie was absolutely fantastic with great acting and plot",
            "Amazing cinematography and brilliant performances throughout",
            "One of the best films I've ever seen, highly recommended",
            "Excellent storytelling and character development",
            "A masterpiece that deserves all the praise it gets"
        ]
        
        negative_reviews = [
            "Terrible movie with poor acting and confusing plot",
            "Waste of time, boring and predictable storyline",
            "Disappointing film that failed to meet expectations",
            "Poor direction and weak character development",
            "One of the worst movies I've ever watched"
        ]
        
        # Generate synthetic dataset
        n_samples = sample_size or 1000
        n_positive = n_samples // 2
        n_negative = n_samples - n_positive
        
        data = []
        for i in range(n_positive):
            review = np.random.choice(positive_reviews) + f" {i}"
            data.append({'review': review, 'sentiment': 'positive'})
        
        for i in range(n_negative):
            review = np.random.choice(negative_reviews) + f" {i}"
            data.append({'review': review, 'sentiment': 'negative'})
        
        df = pd.DataFrame(data)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['sentiment'])
        
        logger.info(f"Loaded {len(train_df)} training and {len(test_df)} test samples")
        return train_df, test_df


class TwitterHateSpeechLoader(DatasetLoader):
    """Loader for Twitter Hate Speech dataset."""
    
    def load_data(self, sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load Twitter Hate Speech dataset.
        
        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info("Loading Twitter Hate Speech dataset...")
        
        # Placeholder: Generate synthetic tweets
        np.random.seed(42)
        
        hate_tweets = [
            "This group of people are terrible and should not exist",
            "I hate everyone from that country, they are all bad",
            "These people are inferior and don't deserve rights"
        ]
        
        normal_tweets = [
            "Having a great day at the park with friends",
            "Just finished reading an interesting book about history",
            "Looking forward to the weekend and some relaxation",
            "Great weather today, perfect for outdoor activities",
            "Excited about the new project I'm working on"
        ]
        
        # Generate synthetic dataset
        n_samples = sample_size or 1000
        n_hate = n_samples // 4  # Less hate speech samples (realistic distribution)
        n_normal = n_samples - n_hate
        
        data = []
        for i in range(n_hate):
            tweet = np.random.choice(hate_tweets) + f" {i}"
            data.append({'tweet': tweet, 'hate_speech': 1})
        
        for i in range(n_normal):
            tweet = np.random.choice(normal_tweets) + f" {i}"
            data.append({'tweet': tweet, 'hate_speech': 0})
        
        df = pd.DataFrame(data)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['hate_speech'])
        
        logger.info(f"Loaded {len(train_df)} training and {len(test_df)} test samples")
        return train_df, test_df


class FakeNewsLoader(DatasetLoader):
    """Loader for Fake News dataset."""
    
    def load_data(self, sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load Fake News dataset.
        
        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info("Loading Fake News dataset...")
        
        # Placeholder: Generate synthetic news articles
        np.random.seed(42)
        
        fake_news = [
            "BREAKING: Scientists discover aliens living among us, government covers up evidence",
            "SHOCKING: Celebrity secretly controls world economy through hidden organization",
            "EXCLUSIVE: New study proves that common food causes instant death"
        ]
        
        real_news = [
            "Local government announces new infrastructure development project",
            "Research team publishes findings on climate change effects",
            "Economic indicators show steady growth in manufacturing sector",
            "University receives grant for renewable energy research",
            "New healthcare policy aims to improve patient outcomes"
        ]
        
        # Generate synthetic dataset
        n_samples = sample_size or 1000
        n_fake = n_samples // 2
        n_real = n_samples - n_fake
        
        data = []
        for i in range(n_fake):
            article = np.random.choice(fake_news) + f" {i}"
            data.append({'text': article, 'label': 'fake'})
        
        for i in range(n_real):
            article = np.random.choice(real_news) + f" {i}"
            data.append({'text': article, 'label': 'real'})
        
        df = pd.DataFrame(data)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        
        logger.info(f"Loaded {len(train_df)} training and {len(test_df)} test samples")
        return train_df, test_df


class DatasetManager:
    """Manager class for handling multiple datasets."""
    
    def __init__(self, data_dir: str = "data/external"):
        self.data_dir = data_dir
        self.loaders = {
            'jigsaw_toxic': JigsawToxicLoader(data_dir),
            'ag_news': AGNewsLoader(data_dir),
            'imdb': IMDBLoader(data_dir),
            'twitter_hate': TwitterHateSpeechLoader(data_dir),
            'fake_news': FakeNewsLoader(data_dir)
        }
    
    def load_dataset(self, dataset_name: str, sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load a specific dataset.
        
        Args:
            dataset_name: Name of the dataset to load
            sample_size: Optional sample size limit
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if dataset_name not in self.loaders:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.loaders.keys())}")
        
        return self.loaders[dataset_name].load_data(sample_size)
    
    def load_all_datasets(self, sample_size: Optional[int] = None) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Load all available datasets.
        
        Args:
            sample_size: Optional sample size limit for each dataset
            
        Returns:
            Dictionary mapping dataset names to (train_df, test_df) tuples
        """
        datasets = {}
        for name in self.loaders.keys():
            try:
                datasets[name] = self.load_dataset(name, sample_size)
                logger.info(f"Successfully loaded {name}")
            except Exception as e:
                logger.error(f"Failed to load {name}: {e}")
        
        return datasets
    
    def get_dataset_info(self) -> Dict[str, Dict[str, Union[str, int]]]:
        """Get information about all datasets.
        
        Returns:
            Dictionary with dataset information
        """
        info = {
            'jigsaw_toxic': {
                'description': 'Toxic comment classification',
                'task': 'binary_classification',
                'text_column': 'comment_text',
                'label_column': 'toxic',
                'classes': ['non-toxic', 'toxic']
            },
            'ag_news': {
                'description': 'News categorization',
                'task': 'multi_class_classification',
                'text_column': 'text',
                'label_column': 'label',
                'classes': ['World', 'Sports', 'Business', 'Sci/Tech']
            },
            'imdb': {
                'description': 'Movie review sentiment analysis',
                'task': 'binary_classification',
                'text_column': 'review',
                'label_column': 'sentiment',
                'classes': ['negative', 'positive']
            },
            'twitter_hate': {
                'description': 'Hate speech detection in tweets',
                'task': 'binary_classification',
                'text_column': 'tweet',
                'label_column': 'hate_speech',
                'classes': ['normal', 'hate_speech']
            },
            'fake_news': {
                'description': 'Fake news detection',
                'task': 'binary_classification',
                'text_column': 'text',
                'label_column': 'label',
                'classes': ['real', 'fake']
            }
        }
        return info
    
    def standardize_dataset(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Standardize dataset format for unified processing.
        
        Args:
            df: DataFrame to standardize
            dataset_name: Name of the dataset
            
        Returns:
            Standardized DataFrame with 'text' and 'label' columns
        """
        info = self.get_dataset_info()[dataset_name]
        text_col = info['text_column']
        label_col = info['label_column']
        
        standardized_df = df.copy()
        standardized_df = standardized_df.rename(columns={
            text_col: 'text',
            label_col: 'label'
        })
        
        # Ensure we only keep text and label columns
        standardized_df = standardized_df[['text', 'label']]
        
        return standardized_df


def main():
    """Example usage of the dataset integration module."""
    # Initialize dataset manager
    manager = DatasetManager()
    
    # Load a single dataset
    print("Loading IMDB dataset...")
    train_df, test_df = manager.load_dataset('imdb', sample_size=100)
    print(f"IMDB - Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Load all datasets
    print("\nLoading all datasets...")
    all_datasets = manager.load_all_datasets(sample_size=50)
    
    # Print dataset information
    print("\nDataset Information:")
    info = manager.get_dataset_info()
    for name, details in info.items():
        print(f"{name}: {details['description']} ({details['task']})")
    
    # Demonstrate standardization
    print("\nStandardizing datasets...")
    for name, (train_df, test_df) in all_datasets.items():
        std_train = manager.standardize_dataset(train_df, name)
        std_test = manager.standardize_dataset(test_df, name)
        print(f"{name} - Standardized columns: {list(std_train.columns)}")


if __name__ == "__main__":
    main()