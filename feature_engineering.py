"""
Feature engineering for 1-day mean reversion classifiers
Input file: cleaned_data.csv

Expected columns (minimum):
- symbol
- date (or timestamp)
- open, high, low, close, volume  (volume optional but strongly recommended)

Output:
- same df with engineered features added (computed per symbol, leakage-safe)
"""

import pandas as pd
import numpy as np


# -------------------------
# IO + basic checks
# -------------------------
def read_df(path: str = "cleaned_data.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    # Try to ensure we have a datetime column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        raise ValueError("Expected a 'date' or 'timestamp' column.")

    return df


def validate_columns(df: pd.DataFrame) -> None:
    required = {"symbol", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Helpful warnings (not fatal)
    recommended = {"open", "high", "low", "volume"}
    missing_rec = recommended - set(df.columns)
    if missing_rec:
        print(f"Warning: missing recommended columns: {missing_rec}. "
              f"Some features (ATR/range/volume) will be skipped.")


def sort_panel(df: pd.DataFrame) -> pd.DataFrame:
    # Use date if present, else timestamp
    time_col = "date" if "date" in df.columns else "timestamp"
    return df.sort_values(["symbol", time_col]).reset_index(drop=True)


# -------------------------
# Core: mean reversion features
# -------------------------
def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("symbol", group_keys=False)

    # Simple daily returns
    df["ret_1"] = g["close"].pct_change(1)
    df["ret_2"] = g["close"].pct_change(2)
    df["ret_3"] = g["close"].pct_change(3)

    # (Optional) log return (often nicer behaved)
    df["logret_1"] = g["close"].transform(lambda x: np.log(x).diff(1))

    return df


def add_moving_averages_and_distances(df: pd.DataFrame, windows=(3, 5, 10, 20)) -> pd.DataFrame:
    """
    For 1-day mean reversion, these are the most useful windows.
    (50/200 can be added later as regime context; not core entry signals.)
    """
    g = df.groupby("symbol", group_keys=False)

    for w in windows:
        ma_col = f"ma_{w}"
        std_col = f"std_close_{w}"

        df[ma_col] = g["close"].transform(lambda x: x.rolling(w).mean())
        df[std_col] = g["close"].transform(lambda x: x.rolling(w).std(ddof=1))

        # Percent distance from mean
        df[f"pct_dist_ma_{w}"] = (df["close"] - df[ma_col]) / df[ma_col]

        # Z-score from mean (use std of CLOSE, not std of MA)
        df[f"zscore_{w}"] = (df["close"] - df[ma_col]) / df[std_col]

    return df


def add_rsi(df: pd.DataFrame, periods=(2, 5, 14)) -> pd.DataFrame:
    """
    RSI(2) is a classic mean-reversion feature.
    Computed per symbol on close.
    """
    g = df.groupby("symbol", group_keys=False)

    def rsi_series(close: pd.Series, n: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        # Wilder's smoothing via EMA with alpha = 1/n
        avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
        avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    for n in periods:
        df[f"rsi_{n}"] = g["close"].transform(lambda x: rsi_series(x, n))

    return df


def add_volatility_features(df: pd.DataFrame, windows=(5, 10, 20)) -> pd.DataFrame:
    """
    Mean reversion depends heavily on volatility regime.
    Uses return volatility + range expansion when OHLC available.
    """
    g = df.groupby("symbol", group_keys=False)

    # Realized vol of returns (rolling std)
    for w in windows:
        df[f"rv_{w}"] = g["ret_1"].transform(lambda x: x.rolling(w).std(ddof=1))

    # True range / ATR if OHLC exist
    if {"high", "low", "close"}.issubset(df.columns):
        # True range uses prior close (per symbol)
        prev_close = g["close"].shift(1)
        tr = pd.concat([
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)

        df["true_range"] = tr

        # ATR (simple rolling mean of TR) and percent ATR
        for w in windows:
            df[f"atr_{w}"] = g["true_range"].transform(lambda x: x.rolling(w).mean())
            df[f"atr_pct_{w}"] = df[f"atr_{w}"] / df["close"]

        # Range expansion: today's TR relative to rolling mean TR
        df["tr_rel_10"] = df["true_range"] / g["true_range"].transform(lambda x: x.rolling(10).mean())

    return df


def add_volume_features(df: pd.DataFrame, windows=(5, 20)) -> pd.DataFrame:
    """
    Volume spikes often indicate exhaustion.
    Skips gracefully if volume missing.
    """
    if "volume" not in df.columns:
        return df

    g = df.groupby("symbol", group_keys=False)

    # Dollar volume (liquidity)
    df["dollar_vol"] = df["close"] * df["volume"]

    for w in windows:
        vol_ma = g["volume"].transform(lambda x: x.rolling(w).mean())
        vol_std = g["volume"].transform(lambda x: x.rolling(w).std(ddof=1))

        df[f"rel_vol_{w}"] = df["volume"] / vol_ma
        df[f"vol_z_{w}"] = (df["volume"] - vol_ma) / vol_std

    return df


def add_gap_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Useful if you trade close->next close (or next open).
    """
    if "open" not in df.columns:
        return df

    g = df.groupby("symbol", group_keys=False)
    prev_close = g["close"].shift(1)

    df["gap_open"] = (df["open"] - prev_close) / prev_close  # today's open vs yesterday close

    return df


# -------------------------
# Target (example)
# -------------------------
def add_1d_forward_target(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """
    Mean reversion target example: will it be up tomorrow by more than threshold?
    threshold = 0.0 -> simply green day tomorrow
    threshold = 0.005 -> > +0.5% tomorrow
    """
    g = df.groupby("symbol", group_keys=False)
    df["fwd_ret_1"] = g["close"].shift(-1) / df["close"] - 1
    df["target_1d"] = (df["fwd_ret_1"] > threshold).astype("int")
    return df


# -------------------------
# Pipeline
# -------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    validate_columns(df)
    df = sort_panel(df)

    df = add_returns(df)
    df = add_moving_averages_and_distances(df, windows=(3, 5, 10, 20))
    df = add_rsi(df, periods=(2, 5, 14))
    df = add_volatility_features(df, windows=(5, 10, 20))
    df = add_volume_features(df, windows=(5, 20))
    df = add_gap_features(df)

    # Optional: add target for training
    df = add_1d_forward_target(df, threshold=0.0)

    return df


def main():
    df = read_df("cleaned_data.csv")
    df = build_features(df)

    # Keep your dataset clean: drop rows where key features are NaN due to rolling windows
    # (Alternatively, keep and let your model handle it after proper train/test splitting.)
    # df = df.dropna(subset=["zscore_5", "rsi_2", "rv_5", "target_1d"])

    df.to_csv("featured_data.csv", index=False)
    print("Saved: featured_data.csv")


if __name__ == "__main__":
    main()
   