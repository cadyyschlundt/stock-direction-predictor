"""Feature engineering for attempt_2.

Builds three feature sets from data/SPY_features.csv (read-only):
  - "original15": the exact 15-feature set used in attempt 1 (control)
  - "engineered": new backward-looking features attempt 1 deferred
    (lagged returns, calendar, volatility regime, normalized price ratios)
  - "combined":   original15 + engineered

All engineered features use only information available at day t to predict
the day t+1 direction label, matching attempt 1's label construction.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "SPY_features.csv"

ORIGINAL_15 = [
    "Close", "Volume", "RSI", "MACD", "MACD_Signal", "MACD_Hist", "BB_Upper",
    "SMA_50", "ATR", "OBV", "Stoch_K", "Stoch_D", "Williams_R", "ROC", "Daily_Return",
]


def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col="Date")


def build_engineered(df: pd.DataFrame) -> pd.DataFrame:
    eng = pd.DataFrame(index=df.index)
    ret = df["Daily_Return"]

    # lagged returns
    for lag in range(1, 6):
        eng[f"ret_lag_{lag}"] = ret.shift(lag)

    # calendar effects
    eng["day_of_week"] = df.index.dayofweek
    eng["month"] = df.index.month

    # volatility regime
    eng["vol_5"] = ret.rolling(5).std()
    eng["vol_21"] = ret.rolling(21).std()
    eng["vol_ratio"] = eng["vol_5"] / eng["vol_21"]
    eng["atr_norm"] = df["ATR"] / df["Close"]

    # mean-reversion / momentum state (price ratios, not raw levels)
    eng["close_sma50"] = df["Close"] / df["SMA_50"] - 1
    eng["close_bb_upper"] = df["Close"] / df["BB_Upper"] - 1
    eng["ret_sum_5"] = ret.rolling(5).sum()
    eng["ret_sum_10"] = ret.rolling(10).sum()

    # consecutive up-day streak ending at t
    up = (ret > 0).astype(int)
    blocks = (up != up.shift()).cumsum()
    eng["up_streak"] = up * (up.groupby(blocks).cumcount() + 1)

    # volume state
    vol_mean = df["Volume"].rolling(21).mean()
    vol_std = df["Volume"].rolling(21).std()
    eng["volume_z_21"] = (df["Volume"] - vol_mean) / vol_std
    eng["obv_chg_5"] = df["OBV"].pct_change(5)

    # bounded oscillators carried over in change form
    eng["rsi_chg_3"] = df["RSI"].diff(3)
    eng["stoch_k_chg_3"] = df["Stoch_K"].diff(3)

    return eng


def build_feature_sets():
    """Return ({name: X DataFrame}, y Series) aligned on a common index."""
    df = load_raw()
    engineered = build_engineered(df)

    combined = pd.concat([df[ORIGINAL_15], engineered], axis=1)
    # single common dropna so every feature set sees identical rows/folds
    valid = combined.dropna().index.intersection(df["Label"].dropna().index)

    y = df.loc[valid, "Label"].astype(int)
    feature_sets = {
        "original15": df.loc[valid, ORIGINAL_15],
        "engineered": engineered.loc[valid],
        "combined": combined.loc[valid],
    }
    return feature_sets, y
