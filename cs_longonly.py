import pandas as pd
import numpy as np

# =========================
# Config
# =========================
DATA_PATH = "featured_data.csv"

DATE_COL = "date"
SYMBOL_COL = "symbol"
ZSCORE_COL = "zscore_5"
CLOSE_COL = "close"

HOLD_DAYS_LIST = [3]                 # you can add [1,3,5] later
BOTTOM_N_LIST = [5, 10, 20, 30, 50]  # concentration sweep

# Transaction cost in **bps per one-way trade**
# Example: 10 bps one-way ~= 20 bps round-trip for names that fully churn in/out
COST_BPS_LIST = [0, 5, 10, 20]

# Optional: only trade extreme oversold
USE_ZSCORE_CUTOFF = False
ZSCORE_LONG_CUTOFF = -1.0            # only allow longs with z <= -1

# Subperiods for regime checks (edit as desired)
SUBPERIODS = [
    ("full", "2015-11-19", "2025-11-14"),
    ("2015-2019", "2015-11-19", "2019-12-31"),
    ("2020-2021", "2020-01-01", "2021-12-31"),
    ("2022-2025", "2022-01-01", "2025-11-14"),
]


# =========================
# Helpers
# =========================
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, SYMBOL_COL]).reset_index(drop=True)
    return df


def add_forward_returns(df: pd.DataFrame, hold_days: int) -> pd.DataFrame:
    g = df.groupby(SYMBOL_COL, group_keys=False)
    fwd_col = f"fwd_ret_{hold_days}"
    if fwd_col not in df.columns:
        df[fwd_col] = g[CLOSE_COL].shift(-hold_days) / df[CLOSE_COL] - 1
    return df


def sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return np.nan
    s = r.std()
    if s == 0 or np.isnan(s):
        return np.nan
    return (r.mean() / s) * np.sqrt(252)


def annualized_vol(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return np.nan
    return r.std() * np.sqrt(252)


def apply_subperiod(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    return df[(df[DATE_COL] >= start_dt) & (df[DATE_COL] <= end_dt)].copy()


# =========================
# Core long-only engine
# =========================
def build_long_cohort(df: pd.DataFrame, hold_days: int, bottom_n: int):
    """
    For each formation date t:
      - rank by zscore_5 (ascending)
      - pick bottom_n
      - compute cohort return = mean fwd_ret_hold
      - also store the set of symbols for turnover estimation
    """
    fwd_col = f"fwd_ret_{hold_days}"
    cols = [DATE_COL, SYMBOL_COL, ZSCORE_COL, fwd_col]
    sub = df[cols].dropna(subset=[ZSCORE_COL, fwd_col]).copy()

    if USE_ZSCORE_CUTOFF:
        sub = sub[sub[ZSCORE_COL] <= ZSCORE_LONG_CUTOFF].copy()

    # rank within date
    sub["rank_low"] = sub.groupby(DATE_COL)[ZSCORE_COL].rank(method="first", ascending=True)
    longs = sub[sub["rank_low"] <= bottom_n].copy()

    # cohort return per formation day
    cohort = (
        longs.groupby(DATE_COL)[fwd_col]
        .mean()
        .rename("cohort_ret")
        .to_frame()
    )
    cohort["n_longs"] = longs.groupby(DATE_COL)[SYMBOL_COL].count()

    # symbol sets per day for turnover proxy
    symsets = longs.groupby(DATE_COL)[SYMBOL_COL].apply(lambda x: set(x)).rename("symbols")
    cohort = cohort.join(symsets, how="inner")

    return cohort, longs


def overlap_cohorts_to_daily(cohort: pd.DataFrame, hold_days: int):
    """
    Overlapping portfolio approximation:
      daily_return(t) = mean of cohort_ret formed over last hold_days days
    """
    daily = pd.DataFrame(index=cohort.index)
    daily["gross_ret"] = cohort["cohort_ret"].rolling(window=hold_days).mean()
    daily = daily.dropna()
    daily["win"] = (daily["gross_ret"] > 0).astype(int)
    return daily


def estimate_turnover_from_sets(cohort: pd.DataFrame, hold_days: int) -> float:
    """
    Rough turnover proxy for the *daily* overlapping book.

    We estimate the average fraction of names that are 'new' when the new cohort enters.
    For long-only overlapping:
      - Each day, a 1/hold_days slice of the book is replaced by the new cohort.
      - If the new cohort overlaps with prior cohorts, effective turnover is lower.

    We compute:
      new_frac_t = 1 - |S_t ∩ S_{t-1}| / |S_t|
    and then scale by 1/hold_days to approximate portfolio-level daily turnover.

    Returns: avg_daily_turnover (0..1)
    """
    sets = cohort["symbols"].dropna()
    if len(sets) < 2:
        return np.nan

    dates = sets.index.to_list()
    new_fracs = []
    for i in range(1, len(dates)):
        s_prev = sets.iloc[i - 1]
        s_now = sets.iloc[i]
        if not s_now:
            continue
        overlap = len(s_now.intersection(s_prev))
        new_frac = 1.0 - (overlap / max(len(s_now), 1))
        new_fracs.append(new_frac)

    if not new_fracs:
        return np.nan

    avg_new_frac = float(np.mean(new_fracs))
    avg_daily_turnover = avg_new_frac * (1.0 / hold_days)
    return avg_daily_turnover


def apply_costs(daily: pd.DataFrame, cost_bps_oneway: float, avg_daily_turnover: float):
    """
    Cost model (simple, transparent):

    - avg_daily_turnover is the fraction of the portfolio traded per day (0..1).
    - cost_bps_oneway is cost per one-way trade.
    - daily cost drag (in return units) = turnover * (cost_bps_oneway / 10_000) * 2
      (multiply by 2 to approximate round-trip for the churned slice of the book)

    This is a rough but useful sensitivity test.
    """
    cost_per_day = avg_daily_turnover * (cost_bps_oneway / 10_000.0) * 2.0
    out = daily.copy()
    out["net_ret"] = out["gross_ret"] - cost_per_day
    out["cost_per_day"] = cost_per_day
    return out


# =========================
# Runner
# =========================
def compute_stats(returns: pd.Series):
    r = returns.dropna()
    if len(r) == 0:
        return {
            "days": 0,
            "avg_daily_ret": np.nan,
            "ann_ret_simple": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "win_rate": np.nan,
        }
    return {
        "days": int(len(r)),
        "avg_daily_ret": float(r.mean()),
        "ann_ret_simple": float(r.mean() * 252),
        "ann_vol": float(annualized_vol(r)),
        "sharpe": float(sharpe(r)),
        "win_rate": float((r > 0).mean()),
    }


def main():
    print("\n====================")
    print("Long-Only Cross-Sectional MR Robustness (rank by zscore_5)")
    print("====================\n")
    print(f"Universe file: {DATA_PATH}")
    print(f"Hold days: {HOLD_DAYS_LIST}")
    print(f"Bottom N sweep: {BOTTOM_N_LIST}")
    print(f"Cost bps (one-way): {COST_BPS_LIST}")
    print(f"Cutoff enabled: {USE_ZSCORE_CUTOFF}")
    if USE_ZSCORE_CUTOFF:
        print(f"  z <= {ZSCORE_LONG_CUTOFF}")
    print("")

    base = load_data()

    rows = []

    # Pre-add forward returns for max hold day to avoid repeated groupbys (safe & fast)
    for hold_days in sorted(set(HOLD_DAYS_LIST)):
        base = add_forward_returns(base, hold_days=hold_days)

    for period_name, start, end in SUBPERIODS:
        dfp = apply_subperiod(base, start, end)

        for hold_days in HOLD_DAYS_LIST:
            for bottom_n in BOTTOM_N_LIST:
                cohort, longs = build_long_cohort(dfp, hold_days=hold_days, bottom_n=bottom_n)
                if cohort.empty:
                    for cost_bps in COST_BPS_LIST:
                        rows.append({
                            "period": period_name,
                            "start": start,
                            "end": end,
                            "hold_days": hold_days,
                            "bottom_n": bottom_n,
                            "cost_bps_oneway": cost_bps,
                            "avg_daily_turnover": np.nan,
                            "days": 0,
                            "avg_daily_ret_gross": np.nan,
                            "sharpe_gross": np.nan,
                            "ann_ret_gross": np.nan,
                            "ann_vol_gross": np.nan,
                            "win_rate_gross": np.nan,
                            "avg_daily_ret_net": np.nan,
                            "sharpe_net": np.nan,
                            "ann_ret_net": np.nan,
                            "ann_vol_net": np.nan,
                            "win_rate_net": np.nan,
                        })
                    continue

                daily = overlap_cohorts_to_daily(cohort, hold_days=hold_days)
                avg_daily_turnover = estimate_turnover_from_sets(cohort, hold_days=hold_days)

                gross_stats = compute_stats(daily["gross_ret"])

                for cost_bps in COST_BPS_LIST:
                    daily_net = apply_costs(daily, cost_bps_oneway=cost_bps, avg_daily_turnover=avg_daily_turnover)
                    net_stats = compute_stats(daily_net["net_ret"])

                    rows.append({
                        "period": period_name,
                        "start": start,
                        "end": end,
                        "hold_days": hold_days,
                        "bottom_n": bottom_n,
                        "cost_bps_oneway": cost_bps,
                        "avg_daily_turnover": avg_daily_turnover,
                        "days": gross_stats["days"],
                        "avg_daily_ret_gross": gross_stats["avg_daily_ret"],
                        "sharpe_gross": gross_stats["sharpe"],
                        "ann_ret_gross": gross_stats["ann_ret_simple"],
                        "ann_vol_gross": gross_stats["ann_vol"],
                        "win_rate_gross": gross_stats["win_rate"],
                        "avg_daily_ret_net": net_stats["avg_daily_ret"],
                        "sharpe_net": net_stats["sharpe"],
                        "ann_ret_net": net_stats["ann_ret_simple"],
                        "ann_vol_net": net_stats["ann_vol"],
                        "win_rate_net": net_stats["win_rate"],
                    })

    res = pd.DataFrame(rows)

    # Pretty print the most relevant slice: FULL period, hold=3, cost=0 and cost=10
    def show_slice(period="full", hold=3, costs=(0, 10)):
        sl = res[(res["period"] == period) & (res["hold_days"] == hold) & (res["cost_bps_oneway"].isin(costs))].copy()
        sl = sl.sort_values(["cost_bps_oneway", "bottom_n"])
        cols = [
            "period", "hold_days", "bottom_n", "cost_bps_oneway",
            "avg_daily_turnover", "days",
            "sharpe_gross", "sharpe_net",
            "ann_ret_gross", "ann_ret_net",
            "ann_vol_gross", "ann_vol_net",
            "win_rate_gross", "win_rate_net",
        ]
        print("----- Snapshot -----")
        print(sl[cols].to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
        print("")

    show_slice(period="full", hold=3, costs=(0, 10))

    # Save full results
    out_path = "cs_longonly_robustness_results.csv"
    res.to_csv(out_path, index=False)
    print(f"Saved full grid results to: {out_path}\n")


if __name__ == "__main__":
    main()