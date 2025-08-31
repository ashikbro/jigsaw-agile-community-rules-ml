import unittest
import sys
import os
import pandas as pd
from pathlib import Path

# Add the src directory to the path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import DataPreprocessor

class TestPreprocess(unittest.TestCase):
    """Test cases for preprocessing functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = DataPreprocessor(tokenizer_name="bert-base-uncased", max_length=128)
        
        # Sample text for testing
        self.sample_texts = [
            "This is a normal text.",
            "This text has a URL: https://example.com",
            "This text has @user mentions and <html> tags.",
            ""  # Empty text
        ]
    
    def test_clean_text(self):
        """Test the clean_text method."""
        cleaned_texts = [self.preprocessor.clean_text(text) for text in self.sample_texts]
        
        # Check that URLs are removed
        self.assertNotIn("https://example.com", cleaned_texts[1])
        
        # Check that user mentions are removed
        self.assertNotIn("@user", cleaned_texts[2])
        
        # Check that HTML tags are removed
        self.assertNotIn("<html>", cleaned_texts[2])
        
        # Check that empty text is handled properly
        self.assertEqual(cleaned_texts[3], "")

if __name__ == '__main__':
    unittest.main()