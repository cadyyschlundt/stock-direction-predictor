"""
Model metadata and metrics.
Loads results and configuration from metrics.json.
Extracted from notebooks/02_baseline.ipynb cell 29 (metrics saved to JSON)
"""

import json
from pathlib import Path


def load_metrics(metrics_path: str = None):
    """
    Load model metrics, results, and metadata from JSON.

    Args:
        metrics_path: path to metrics.json. If None, uses ../results/metrics.json

    Returns:
        metrics: dict with keys:
            - selected_model: name and hyperparameters of chosen model
            - baseline: baseline accuracy (always predict up)
            - model_comparison: all 4 models' accuracy/precision/recall
            - equity_curve_metrics: strategy vs buy-and-hold Sharpe/return/drawdown
            - key_finding: honest caveat about predictive skill
            - metadata: training date, features list, etc.
    """
    if metrics_path is None:
        metrics_path = Path(__file__).resolve().parent.parent / 'results' / 'metrics.json'

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    return metrics


def get_model_info():
    """
    Convenience function to get key model info for display/API.

    Returns:
        dict with model name, hyperparameters, key_finding caveat
    """
    metrics = load_metrics()

    return {
        'model_name': metrics.get('selected_model', {}).get('name', 'HistGradientBoosting'),
        'hyperparameters': metrics.get('selected_model', {}).get('hyperparameters', {}),
        'key_finding': metrics.get('key_finding', 'No predictive skill detected'),
        'training_date': metrics.get('metadata', {}).get('training_date', 'unknown'),
        'features': metrics.get('metadata', {}).get('features', [])
    }


def get_equity_curve_metrics():
    """
    Convenience function to get backtest results.

    Returns:
        dict with strategy and buy-and-hold Sharpe/return/drawdown
    """
    metrics = load_metrics()
    return metrics.get('equity_curve_metrics', {})