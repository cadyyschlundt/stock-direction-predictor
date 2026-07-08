"""
Model predictor for stock direction.
Loads trained HGB model and makes predictions.
Extracted from notebooks/02_baseline.ipynb cells 14-16
"""

import joblib
from pathlib import Path


class StockDirectionPredictor:
    """
    Wrapper around trained HistGradientBoostingClassifier.
    HGB is tree-based, does not require scaling.
    """

    def __init__(self, model_path: str = None):
        """
        Load trained model from joblib.

        Args:
            model_path: path to hgb_v2.joblib. If None, uses ../models/hgb_v2.joblib
        """
        if model_path is None:
            model_path = Path(__file__).resolve().parent.parent / 'models' / 'hgb_v2.joblib'

        self.model = joblib.load(model_path)

    def predict(self, X):
        """
        Predict direction (0=down, 1=up) for features X.

        Args:
            X: feature dataframe or array with 15 columns

        Returns:
            predictions: array of 0 or 1
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict probability of each class for features X.

        Args:
            X: feature dataframe or array with 15 columns

        Returns:
            probabilities: array of shape (n_samples, 2), columns are P(down), P(up)
        """
        return self.model.predict_proba(X)