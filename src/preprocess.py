import os
import re
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from config.config import (
    DATA_DIR, OUTPUT_DIR, MODEL_DIR,
    MODEL_CONFIG, PROCESSING_CONFIG, TRAIN_CONFIG
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Class for preprocessing the Jigsaw Agile Community Rules dataset."""
    
    def __init__(self, tokenizer_name: str = None, max_length: int = None):
        """Initialize the data preprocessor.
        
        Args:
            tokenizer_name: Name of the tokenizer to use
            max_length: Maximum sequence length
        """
        self.tokenizer_name = tokenizer_name or MODEL_CONFIG['model_name']
        self.max_length = max_length or MODEL_CONFIG['max_length']
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        
        # Text cleaning patterns
        self.url_pattern = re.compile(r'https?\S+|www\.\S+')
        self.user_mention_pattern = re.compile(r'@\w+')
        self.html_pattern = re.compile(r'<.*?>')
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text data.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = self.url_pattern.sub(' [URL] ', text)
        
        # Remove user mentions
        text = self.user_mention_pattern.sub(' [USER] ', text)
        
        # Remove HTML tags
        text = self.html_pattern.sub(' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the input dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Preprocessed dataframe
        """
        df = df.copy()
        
        # Clean text columns
        text_columns = [col for col in PROCESSING_CONFIG['text_columns'] if col in df.columns]
        for col in text_columns:
            df[col] = df[col].astype(str).apply(self.clean_text)
        
        return df
    
    def tokenize_data(self, df: pd.DataFrame) -> Dict:
        """Tokenize the input data.
        
        Args:
            df: Input dataframe with text columns
            
        Returns:
            Dictionary containing tokenized inputs
        """
        text_columns = [col for col in PROCESSING_CONFIG['text_columns'] if col in df.columns]
        
        # Combine text columns with a separator
        if len(text_columns) > 1:
            texts = df[text_columns[0]].str.cat(df[text_columns[1:]], sep=' [SEP] ')
        else:
            texts = df[text_columns[0]]
        
        # Tokenize the text
        tokenized = self.tokenizer(
            texts.tolist(),
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='np',
            return_attention_mask=True,
            return_token_type_ids=True
        )
        
        return tokenized
    
    def create_datasets(self, df: pd.DataFrame) -> Tuple[Dict, Dict, Dict]:
        """Create train, validation, and test datasets.
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple containing train, validation, and test datasets
        """
        # Split the data
        train_df, temp_df = train_test_split(
            df,
            test_size=1 - TRAIN_CONFIG['train_frac'],
            random_state=TRAIN_CONFIG['random_state']
        )
        
        val_size = TRAIN_CONFIG['val_frac'] / (TRAIN_CONFIG['val_frac'] + TRAIN_CONFIG['test_frac'])
        val_df, test_df = train_test_split(
            temp_df,
            test_size=1 - val_size,
            random_state=TRAIN_CONFIG['random_state']
        )
        
        logger.info(f"Train size: {len(train_df)}")
        logger.info(f"Validation size: {len(val_df)}")
        logger.info(f"Test size: {len(test_df)}")
        
        # Tokenize the data
        train_encodings = self.tokenize_data(train_df)
        val_encodings = self.tokenize_data(val_df)
        test_encodings = self.tokenize_data(test_df)
        
        # Get labels if they exist
        label_col = PROCESSING_CONFIG.get('label_column')
        if label_col and label_col in df.columns:
            train_encodings['labels'] = train_df[label_col].values
            val_encodings['labels'] = val_df[label_col].values
            test_encodings['labels'] = test_df[label_col].values
        
        return train_encodings, val_encodings, test_encodings


def load_and_preprocess_data(data_path: str = None) -> Tuple[Dict, Dict, Dict]:
    """Load and preprocess the dataset.
    
    Args:
        data_path: Path to the dataset file
        
    Returns:
        Tuple containing train, validation, and test datasets
    """
    if data_path is None:
        data_path = DATA_DIR / 'train.csv'
    
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Preprocess the data
    logger.info("Preprocessing data...")
    df_processed = preprocessor.preprocess_dataframe(df)
    
    # Create datasets
    logger.info("Creating datasets...")
    train_data, val_data, test_data = preprocessor.create_datasets(df_processed)
    
    return train_data, val_data, test_data


if __name__ == "__main__":
    # Example usage
    train_data, val_data, test_data = load_and_preprocess_data()
    logger.info("Data preprocessing completed successfully!")
    
    # Save processed data
    # np.savez(OUTPUT_DIR / 'train_data.npz', **train_data)
    # np.savez(OUTPUT_DIR / 'val_data.npz', **val_data)
    # np.savez(OUTPUT_DIR / 'test_data.npz', **test_data)
