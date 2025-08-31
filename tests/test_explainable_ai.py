"""Tests for the Explainable AI module."""

import sys
import os
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import the module to test
from src.explainable_ai import (
    extract_feature_importance,
    visualize_feature_importance,
    predict_with_explanation,
    has_xai_libs
)


class TestExplainableAI(unittest.TestCase):
    """Test cases for the Explainable AI module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock vectorizer
        self.mock_vectorizer = MagicMock()
        self.mock_vectorizer.get_feature_names_out.return_value = [
            'feature1', 'feature2', 'feature3', 'feature4', 'feature5'
        ]
        
        # Create mock linear model (e.g., LogisticRegression)
        self.mock_linear_model = MagicMock()
        self.mock_linear_model.coef_ = np.array([[0.5, -0.3, 0.8, -0.6, 0.2]])
        self.mock_linear_model.predict.return_value = np.array([1])
        self.mock_linear_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        
        # Create mock tree-based model (e.g., RandomForest)
        self.mock_tree_model = MagicMock()
        self.mock_tree_model.feature_importances_ = np.array([0.2, 0.3, 0.1, 0.25, 0.15])
        self.mock_tree_model.predict.return_value = np.array([0])
        self.mock_tree_model.predict_proba.return_value = np.array([[0.8, 0.2]])
        # Ensure tree model doesn't have coef_ attribute
        del self.mock_tree_model.coef_
        
        # Sample text
        self.sample_text = "This is a sample text for testing."
    
    def test_extract_feature_importance_linear(self):
        """Test feature importance extraction for linear models."""
        # Call the function
        result = extract_feature_importance(self.mock_linear_model, self.mock_vectorizer)
        
        # Check the result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)  # All features should be included
        self.assertTrue('Feature' in result.columns)
        self.assertTrue('Importance' in result.columns)
        self.assertTrue('Abs_Importance' in result.columns)
        
        # Check sorting by absolute importance
        self.assertEqual(result.iloc[0]['Feature'], 'feature3')  # 0.8 has highest abs value
        self.assertEqual(result.iloc[1]['Feature'], 'feature4')  # -0.6 has second highest abs value
    
    def test_extract_feature_importance_tree(self):
        """Test feature importance extraction for tree-based models."""
        # Call the function
        result = extract_feature_importance(self.mock_tree_model, self.mock_vectorizer)
        
        # Check the result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)  # All features should be included
        self.assertTrue('Feature' in result.columns)
        self.assertTrue('Importance' in result.columns)
        self.assertFalse('Abs_Importance' in result.columns)  # Tree models don't have this
        
        # Check sorting by importance
        self.assertEqual(result.iloc[0]['Feature'], 'feature2')  # 0.3 has highest value
        self.assertEqual(result.iloc[1]['Feature'], 'feature4')  # 0.25 has second highest value
    
    @patch('src.explainable_ai.plt')
    def test_visualize_feature_importance(self, mock_plt):
        """Test feature importance visualization."""
        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'Feature': ['feature1', 'feature2', 'feature3', 'feature4', 'feature5'],
            'Importance': [0.5, -0.3, 0.8, -0.6, 0.2],
            'Abs_Importance': [0.5, 0.3, 0.8, 0.6, 0.2]
        })
        
        # Call the function
        visualize_feature_importance(
            feature_importance, 
            'TestModel', 
            is_linear=True,
            output_path='test_output.png'
        )
        
        # Check that figure was created and saved
        mock_plt.figure.assert_called_once()
        mock_plt.savefig.assert_called_once_with('test_output.png')
    
    @unittest.skipIf(not has_xai_libs, "XAI libraries not available")
    def test_predict_with_explanation(self):
        """Test prediction with explanation."""
        # This test only runs if XAI libraries are available
        with patch('src.explainable_ai.explain_prediction_with_lime') as mock_lime, \
             patch('src.explainable_ai.explain_prediction_with_shap') as mock_shap:
            
            # Set up mocks
            mock_lime.return_value = MagicMock()
            mock_shap.return_value = MagicMock()
            
            # Call the function
            result = predict_with_explanation(
                self.sample_text, 
                self.mock_linear_model, 
                self.mock_vectorizer
            )
            
            # Check the result
            self.assertIsInstance(result, dict)
            self.assertEqual(result['text'], self.sample_text)
            self.assertEqual(result['prediction'], 1)
            self.assertAlmostEqual(result['probability'], 0.7)
            self.assertIn('lime_explanation', result)
            self.assertIn('shap_explanation', result)


if __name__ == '__main__':
    unittest.main()