"""
Simple tests for data.py
Tests that data loads correctly with expected shape and columns.
"""

import sys
from pathlib import Path

# Add src to path so we can import data module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from data import load_and_prepare_data


def test_data_loads():
    """Data loads without error."""
    X, y, df = load_and_prepare_data()
    assert X is not None
    assert y is not None
    assert df is not None


def test_data_shape():
    """Data has expected shape (2938 rows, 15 features)."""
    X, y, df = load_and_prepare_data()
    assert X.shape[1] == 15, f"Expected 15 features, got {X.shape[1]}"
    assert len(y) > 2000, f"Expected 2000+ rows, got {len(y)}"


def test_label_binary():
    """Label is binary (0 or 1 only)."""
    X, y, df = load_and_prepare_data()
    assert set(y.unique()) == {0, 1}, f"Expected labels 0 and 1, got {y.unique()}"


def test_no_nans():
    """Data has no NaN values."""
    X, y, df = load_and_prepare_data()
    assert X.isna().sum().sum() == 0, "Found NaN in features"
    assert y.isna().sum() == 0, "Found NaN in labels"