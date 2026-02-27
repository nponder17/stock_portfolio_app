import pandas as pd
import numpy as np

DATA_PATH = "featured_data.csv"

ZSCORE_COL = "zscore_5"
DATE_COL = "date"
SYMBOL_COL = "symbol"
CLOSE_COL = "close"

BOTTOM_N = 10                 # long bottom 10
MIN_NAMES_PER_DAY = 200       # skip days with too few symbols
USE_EQUAL_WEIGHT = True       # equal weight across names

# Optional: only trade when zscore is sufficiently extreme (helps)
USE_ZSCORE_CUTOFF = False
ZSCORE_CUTOFF = -1.0          # only include names with zscore_5 <= -1


def load_data():
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, SYMBOL_COL]).reset_index(drop=True)
    return df


def add_forward_returns(df: pd.DataFrame):
    g = df.groupby(SYMBOL_COL, group_keys=False)

    # Already have fwd_ret_1? If not, compute it.
    if "fwd_ret_1" not in df.columns:
        df["fwd_ret_1"] = g[CLOSE_COL].shift(-1) / df[CLOSE_COL] - 1

    # Compute 3-day forward return
    df["fwd_ret_3"] = g[CLOSE_COL].shift(-3) / df[CLOSE_COL] - 1

    return df


def sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return np.nan
    return (r.mean() / r.std()) * np.sqrt(252)


def backtest_cs(df: pd.DataFrame, hold_days: int = 1):
    """
    Each day:
    - rank by zscore_5 (ascending)
    - take bottom N (most negative)
    - equal-weight
    - realized return uses fwd_ret_{hold_days}
    """
    fwd_col = f"fwd_ret_{hold_days}"
    if fwd_col not in df.columns:
        raise ValueError(f"Missing {fwd_col}. Make sure forward returns are computed.")

    # Keep necessary columns + drop rows missing key fields
    sub = df[[DATE_COL, SYMBOL_COL, ZSCORE_COL, fwd_col]].dropna(subset=[ZSCORE_COL, fwd_col]).copy()

    # Optional: trade only when sufficiently oversold
    if USE_ZSCORE_CUTOFF:
        sub = sub[sub[ZSCORE_COL] <= ZSCORE_CUTOFF]

    # Ensure enough names per day
    counts = sub.groupby(DATE_COL)[SYMBOL_COL].nunique()
    valid_days = counts[counts >= MIN_NAMES_PER_DAY].index
    sub = sub[sub[DATE_COL].isin(valid_days)]

    # Select bottom N per day
    sub["rank"] = sub.groupby(DATE_COL)[ZSCORE_COL].rank(method="first", ascending=True)
    picks = sub[sub["rank"] <= BOTTOM_N].copy()

    # Portfolio daily return = mean of forward returns across picks
    daily = picks.groupby(DATE_COL)[fwd_col].mean().rename("port_ret").to_frame()

    # Diagnostics
    daily["n_names"] = picks.groupby(DATE_COL)[SYMBOL_COL].count()
    daily["win"] = (daily["port_ret"] > 0).astype(int)

    # Summary stats
    out = {
        "hold_days": hold_days,
        "bottom_n": BOTTOM_N,
        "zscore_cutoff": ZSCORE_CUTOFF if USE_ZSCORE_CUTOFF else None,
        "days_traded": int(daily.shape[0]),
        "avg_daily_ret": float(daily["port_ret"].mean()),
        "ann_ret_simple": float(daily["port_ret"].mean() * 252),
        "ann_vol": float(daily["port_ret"].std() * np.sqrt(252)),
        "sharpe": float(sharpe(daily["port_ret"])),
        "win_rate_days": float(daily["win"].mean()),
        "avg_names": float(daily["n_names"].mean()),
    }

    return daily, picks, out


def main():
    df = load_data()
    df = add_forward_returns(df)

    # 1-day CS
    daily_1, picks_1, stats_1 = backtest_cs(df, hold_days=1)

    # 3-day CS
    daily_3, picks_3, stats_3 = backtest_cs(df, hold_days=3)

    print("\n====================")
    print("Cross-Sectional MR Results (Long Bottom N by zscore_5)")
    print("====================\n")

    for s in [stats_1, stats_3]:
        print(f"HOLD: {s['hold_days']} day(s)")
        print(f"  bottom_n: {s['bottom_n']}")
        print(f"  zscore_cutoff: {s['zscore_cutoff']}")
        print(f"  days_traded: {s['days_traded']}")
        print(f"  avg_names/day: {s['avg_names']:.2f}")
        print(f"  avg_daily_ret: {s['avg_daily_ret']:.6f}")
        print(f"  ann_ret_simple: {s['ann_ret_simple']:.3f}")
        print(f"  ann_vol: {s['ann_vol']:.3f}")
        print(f"  Sharpe: {s['sharpe']:.3f}")
        print(f"  win_rate_days: {s['win_rate_days']:.3f}")
        print("")

    # Save outputs for your UI
    daily_1.to_csv("cs_mr_daily_1d.csv", index=True)
    daily_3.to_csv("cs_mr_daily_3d.csv", index=True)
    picks_1.to_csv("cs_mr_picks_1d.csv", index=False)
    picks_3.to_csv("cs_mr_picks_3d.csv", index=False)

    print("Saved:")
    print("  cs_mr_daily_1d.csv")
    print("  cs_mr_daily_3d.csv")
    print("  cs_mr_picks_1d.csv")
    print("  cs_mr_picks_3d.csv")


if __name__ == "__main__":
    main()