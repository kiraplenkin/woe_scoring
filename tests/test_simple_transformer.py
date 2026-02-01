import unittest
import numpy as np
import pandas as pd
import os
import tempfile
from woe_scoring.core.main import WOETransformer


class TestSimpleTransformer(unittest.TestCase):
    def setUp(self):
        # Create a simpler test dataset with only numeric features
        np.random.seed(42)
        n_samples = 100

        # Create only numeric features to avoid type issues
        self.X = pd.DataFrame({
            'numeric1': np.random.normal(0, 1, n_samples),
            'numeric2': np.random.uniform(-5, 5, n_samples)
        })

        # Create a target variable with binary outcome (0 or 1)
        self.y = np.random.binomial(1, 0.3, n_samples)

        # Default parameters for transformer
        self.transformer = WOETransformer(
            max_bins=5,
            min_pct_group=0.05,
            n_jobs=1,
            prefix="WOE_",
            merge_type="chi2",
            cat_features=[],  # No categorical features
            special_cols=None,
            cat_features_threshold=0,
            diff_woe_threshold=0.05,
            safe_original_data=False
        )

    def test_initialization(self):
        """Test that the WOETransformer initializes correctly"""
        self.assertEqual(self.transformer.max_bins, 5)
        self.assertEqual(self.transformer.min_pct_group, 0.05)
        self.assertEqual(self.transformer.n_jobs, 1)
        self.assertEqual(self.transformer.prefix, "WOE_")
        self.assertEqual(self.transformer.merge_type, "chi2")
        self.assertEqual(self.transformer.cat_features, [])
        self.assertIsNone(self.transformer.special_cols)
        self.assertEqual(self.transformer.cat_features_threshold, 0)
        self.assertEqual(self.transformer.diff_woe_threshold, 0.05)
        self.assertEqual(self.transformer.safe_original_data, False)

    def test_fit_transform(self):
        """Test a simple fit and transform flow"""
        try:
            # Try to fit the transformer
            self.transformer.fit(self.X, self.y)

            # Print available attributes for debugging
            print(f"Available attributes: {dir(self.transformer)}")

            # Check that woe_iv_dict was created
            self.assertTrue(hasattr(self.transformer, 'woe_iv_dict'))

            # Try to transform data
            X_transformed = self.transformer.transform(self.X)

            # Check that transformed data has the same number of rows
            self.assertEqual(len(X_transformed), len(self.X))

            # Print transformed data info
            print(f"Transformed data shape: {X_transformed.shape}")
            print(f"Transformed data columns: {X_transformed.columns}")

            # Test passed if we got here without errors
            self.assertTrue(True)
        except Exception as e:
            # If an error occurs, the test will be marked as failed
            # Print the error for debugging
            print(f"Error: {str(e)}")
            self.fail(f"fit_transform test failed with error: {str(e)}")


if __name__ == '__main__':
    unittest.main()
