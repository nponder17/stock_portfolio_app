import pandas as pd
import numpy as np

DATA_PATH = "featured_data.csv"

DATE_COL = "date"
SYMBOL_COL = "symbol"
ZSCORE_COL = "zscore_5"
CLOSE_COL = "close"

BOTTOM_N = 10
TOP_N = 10

# Optional: only trade extreme tails
USE_ZSCORE_CUTOFF = False
ZSCORE_LONG_CUTOFF = -1.0   # only allow longs with z <= this
ZSCORE_SHORT_CUTOFF = 1.0   # only allow shorts with z >= this


def load_data():
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, SYMBOL_COL]).reset_index(drop=True)
    return df


def add_forward_returns(df: pd.DataFrame):
    g = df.groupby(SYMBOL_COL, group_keys=False)
    if "fwd_ret_1" not in df.columns:
        df["fwd_ret_1"] = g[CLOSE_COL].shift(-1) / df[CLOSE_COL] - 1
    df["fwd_ret_3"] = g[CLOSE_COL].shift(-3) / df[CLOSE_COL] - 1
    return df


def sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return np.nan
    return (r.mean() / r.std()) * np.sqrt(252)


def build_daily_cohort_returns(df: pd.DataFrame, hold_days: int):
    """
    Cohort returns formed each day t:
      cohort_long_ret(t)  = mean fwd_ret_hold among long picks
      cohort_short_ret(t) = mean fwd_ret_hold among short picks (raw underlying fwd ret)
      cohort_ls_ret(t)    = cohort_long_ret(t) - cohort_short_ret(t)
    """
    fwd_col = f"fwd_ret_{hold_days}"
    sub = df[[DATE_COL, SYMBOL_COL, ZSCORE_COL, fwd_col]].dropna(subset=[ZSCORE_COL, fwd_col]).copy()

    if USE_ZSCORE_CUTOFF:
        long_pool = sub[sub[ZSCORE_COL] <= ZSCORE_LONG_CUTOFF].copy()
        short_pool = sub[sub[ZSCORE_COL] >= ZSCORE_SHORT_CUTOFF].copy()
    else:
        long_pool = sub
        short_pool = sub

    long_pool["rank_low"] = long_pool.groupby(DATE_COL)[ZSCORE_COL].rank(method="first", ascending=True)
    short_pool["rank_high"] = short_pool.groupby(DATE_COL)[ZSCORE_COL].rank(method="first", ascending=False)

    longs = long_pool[long_pool["rank_low"] <= BOTTOM_N].copy()
    shorts = short_pool[short_pool["rank_high"] <= TOP_N].copy()

    long_daily = longs.groupby(DATE_COL)[fwd_col].mean().rename("cohort_long_ret").to_frame()
    short_daily = shorts.groupby(DATE_COL)[fwd_col].mean().rename("cohort_short_ret").to_frame()

    cohort = long_daily.join(short_daily, how="inner")
    cohort["cohort_ls_ret"] = cohort["cohort_long_ret"] - cohort["cohort_short_ret"]
    cohort["n_longs"] = longs.groupby(DATE_COL)[SYMBOL_COL].count()
    cohort["n_shorts"] = shorts.groupby(DATE_COL)[SYMBOL_COL].count()
    return cohort, longs, shorts


def overlap_to_daily_portfolio(cohort: pd.DataFrame, hold_days: int):
    """
    Overlapping-hold book approximation:
      daily return at t = mean of cohort returns formed over last hold_days days
    """
    daily = pd.DataFrame(index=cohort.index)

    daily["port_long_ret"] = cohort["cohort_long_ret"].rolling(window=hold_days).mean()
    daily["port_short_underlying_ret"] = cohort["cohort_short_ret"].rolling(window=hold_days).mean()

    # Short PnL is NEGATIVE of underlying return
    daily["port_short_pnl"] = -daily["port_short_underlying_ret"]

    # Market-neutral long-short PnL
    daily["port_ls_ret"] = daily["port_long_ret"] + daily["port_short_pnl"]

    daily = daily.dropna()
    daily["win_ls"] = (daily["port_ls_ret"] > 0).astype(int)
    daily["win_long"] = (daily["port_long_ret"] > 0).astype(int)
    daily["win_short"] = (daily["port_short_pnl"] > 0).astype(int)

    stats = {
        "hold_days": hold_days,
        "bottom_n": BOTTOM_N,
        "top_n": TOP_N,
        "zscore_cutoff": (ZSCORE_LONG_CUTOFF, ZSCORE_SHORT_CUTOFF) if USE_ZSCORE_CUTOFF else None,
        "days": int(daily.shape[0]),
        "avg_daily_ret_long": float(daily["port_long_ret"].mean()),
        "avg_daily_ret_short_pnl": float(daily["port_short_pnl"].mean()),
        "avg_daily_ret_ls": float(daily["port_ls_ret"].mean()),
        "ann_ret_long_simple": float(daily["port_long_ret"].mean() * 252),
        "ann_ret_short_simple": float(daily["port_short_pnl"].mean() * 252),
        "ann_ret_ls_simple": float(daily["port_ls_ret"].mean() * 252),
        "ann_vol_long": float(daily["port_long_ret"].std() * np.sqrt(252)),
        "ann_vol_short": float(daily["port_short_pnl"].std() * np.sqrt(252)),
        "ann_vol_ls": float(daily["port_ls_ret"].std() * np.sqrt(252)),
        "sharpe_long": float(sharpe(daily["port_long_ret"])),
        "sharpe_short": float(sharpe(daily["port_short_pnl"])),
        "sharpe_ls": float(sharpe(daily["port_ls_ret"])),
        "win_rate_long": float(daily["win_long"].mean()),
        "win_rate_short": float(daily["win_short"].mean()),
        "win_rate_ls": float(daily["win_ls"].mean()),
    }
    return daily, stats


def run(hold_days: int):
    df = load_data()
    df = add_forward_returns(df)

    cohort, longs, shorts = build_daily_cohort_returns(df, hold_days=hold_days)
    daily_port, stats = overlap_to_daily_portfolio(cohort, hold_days=hold_days)

    return cohort, daily_port, longs, shorts, stats


def main():
    print("\n====================")
    print("Cross-Sectional MR Diagnostics (rank by zscore_5)")
    print("====================\n")
    print(f"Universe file: {DATA_PATH}")
    print(f"Params: bottom_n={BOTTOM_N}, top_n={TOP_N}, cutoff={USE_ZSCORE_CUTOFF}")
    if USE_ZSCORE_CUTOFF:
        print(f"  long z <= {ZSCORE_LONG_CUTOFF}, short z >= {ZSCORE_SHORT_CUTOFF}")
    print("")

    for hold_days in [1, 3]:
        cohort, daily, longs, shorts, s = run(hold_days=hold_days)

        print(f"HOLD: {hold_days} day(s)")
        print(f"  days: {s['days']}")
        print(f"  avg_daily_ret_long:       {s['avg_daily_ret_long']:.6f}")
        print(f"  avg_daily_ret_short_pnl:  {s['avg_daily_ret_short_pnl']:.6f}")
        print(f"  avg_daily_ret_longshort:  {s['avg_daily_ret_ls']:.6f}\n")

        print(f"  Sharpe long:   {s['sharpe_long']:.3f}")
        print(f"  Sharpe short:  {s['sharpe_short']:.3f}")
        print(f"  Sharpe L/S:    {s['sharpe_ls']:.3f}\n")

        print(f"  win_rate long:  {s['win_rate_long']:.3f}")
        print(f"  win_rate short: {s['win_rate_short']:.3f}")
        print(f"  win_rate L/S:   {s['win_rate_ls']:.3f}\n")

        # Save outputs
        cohort.to_csv(f"cs_diag_cohort_{hold_days}d.csv")
        daily.to_csv(f"cs_diag_port_{hold_days}d.csv")
        longs.to_csv(f"cs_diag_longs_{hold_days}d.csv", index=False)
        shorts.to_csv(f"cs_diag_shorts_{hold_days}d.csv", index=False)

        print("  Saved:")
        print(f"    cs_diag_cohort_{hold_days}d.csv")
        print(f"    cs_diag_port_{hold_days}d.csv")
        print(f"    cs_diag_longs_{hold_days}d.csv")
        print(f"    cs_diag_shorts_{hold_days}d.csv\n")

    print("Done.\n")


if __name__ == "__main__":
    main()