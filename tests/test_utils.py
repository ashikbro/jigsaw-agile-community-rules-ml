import unittest
import sys
import os
import numpy as np
from pathlib import Path

# Add the src directory to the path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import calculate_metrics, load_json

class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_calculate_metrics(self):
        """Test the calculate_metrics function."""
        # Test binary classification case
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.4, 0.7])
        
        metrics = calculate_metrics(y_true, y_pred, y_prob)
        
        # Check that all expected metrics are present
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        self.assertIn('roc_auc', metrics)
        
        # Check specific values
        self.assertEqual(metrics['accuracy'], 0.6)  # 3 out of 5 correct
        
    def test_load_json(self):
        """Test the load_json function."""
        # Create a temporary JSON file
        import tempfile
        import json
        
        test_data = {"test": "data", "number": 42}
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(test_data, f)
            temp_file = f.name
        
        try:
            # Test loading the JSON file
            loaded_data = load_json(temp_file)
            self.assertEqual(loaded_data, test_data)
        finally:
            # Clean up
            os.unlink(temp_file)

if __name__ == '__main__':
    unittest.main()