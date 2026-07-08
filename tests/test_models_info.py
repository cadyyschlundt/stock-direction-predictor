"""
Simple tests for models_info.py
Tests that metrics.json loads and contains expected data.
"""

import sys
from pathlib import Path

# Add src to path so we can import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from models_info import load_metrics, get_model_info, get_equity_curve_metrics


def test_metrics_loads():
    """Metrics.json loads without error."""
    metrics = load_metrics()
    assert metrics is not None
    assert isinstance(metrics, dict)


def test_metrics_has_key_fields():
    """Metrics has expected top-level keys."""
    metrics = load_metrics()
    expected_keys = ['selected_model', 'baseline', 'key_finding']
    for key in expected_keys:
        assert key in metrics, f"Missing key: {key}"


def test_get_model_info():
    """get_model_info returns dict with expected keys."""
    info = get_model_info()
    assert isinstance(info, dict)
    expected_keys = ['model_name', 'hyperparameters', 'key_finding', 'training_date', 'features']
    for key in expected_keys:
        assert key in info, f"Missing key in model_info: {key}"


def test_get_equity_curve_metrics():
    """get_equity_curve_metrics returns dict with strategy and buy_and_hold."""
    metrics = get_equity_curve_metrics()
    assert isinstance(metrics, dict)
    # Should have at least strategy or buy_and_hold
    assert len(metrics) > 0, "No equity curve metrics found"