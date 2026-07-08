"""
Data loading and preprocessing for stock direction predictor.
Extracted from notebooks/02_baseline.ipynb
"""

import pandas as pd
from pathlib import Path


def load_and_prepare_data(data_path: str = None):
    """
    Load SPY features CSV, drop redundant columns, drop NaNs.

    Args:
        data_path: path to spy_features.csv. If None, uses ../data/spy_features.csv

    Returns:
        X: feature dataframe (15 columns)
        y: label series (binary, 0/1 for down/up)
        df: full dataframe with all columns for reference
    """
    if data_path is None:
        data_path = Path(__file__).resolve().parent.parent / 'data' / 'spy_features.csv'

    # Load data
    df = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')
    df = df.sort_index()

    # Drop redundant price-level features
    cols_to_drop = ['High', 'Low', 'Open', 'BB_Middle', 'BB_Lower', 'SMA_20', 'SMA_200', 'EMA_12', 'EMA_26']
    df = df.drop(columns=cols_to_drop)

    # Drop rows with NaN (indicator warmup period)
    df = df.dropna()

    # Split features and label
    X = df.drop(columns=['Label'])
    y = df['Label']

    return X, y, df