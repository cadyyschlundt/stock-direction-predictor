"""
Simple tests for predictor.py
Tests that model loads and makes predictions with correct shape and values.
"""

import numpy as np
from src.predictor import StockDirectionPredictor
from src.data import load_and_prepare_data


def test_model_loads():
    """Model loads without error."""
    predictor = StockDirectionPredictor()
    assert predictor.model is not None


def test_predict_shape():
    """Predictions have correct shape (n_samples,)."""
    predictor = StockDirectionPredictor()
    X, y, df = load_and_prepare_data()

    predictions = predictor.predict(X[:10])
    assert predictions.shape == (10,), f"Expected shape (10,), got {predictions.shape}"


def test_predict_values_binary():
    """Predictions are 0 or 1 only."""
    predictor = StockDirectionPredictor()
    X, y, df = load_and_prepare_data()

    predictions = predictor.predict(X)
    assert set(predictions) == {0, 1}, f"Expected 0 and 1, got {set(predictions)}"


def test_predict_proba_shape():
    """Probabilities have correct shape (n_samples, 2)."""
    predictor = StockDirectionPredictor()
    X, y, df = load_and_prepare_data()

    probabilities = predictor.predict_proba(X[:10])
    assert probabilities.shape == (10, 2), f"Expected shape (10, 2), got {probabilities.shape}"


def test_predict_proba_values():
    """Probabilities are between 0 and 1."""
    predictor = StockDirectionPredictor()
    X, y, df = load_and_prepare_data()

    probabilities = predictor.predict_proba(X)
    assert (probabilities >= 0).all() and (probabilities <= 1).all(), "Probabilities outside [0, 1]"


def test_predict_proba_sum_to_one():
    """Probabilities sum to 1 for each sample."""
    predictor = StockDirectionPredictor()
    X, y, df = load_and_prepare_data()

    probabilities = predictor.predict_proba(X[:10])
    sums = probabilities.sum(axis=1)
    assert np.allclose(sums, 1.0), f"Probabilities don't sum to 1: {sums}"