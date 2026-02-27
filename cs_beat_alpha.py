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

HOLD_DAYS = 3
BOTTOM_N = 50

# Optional oversold cutoff
USE_ZSCORE_CUTOFF = False
ZSCORE_LONG_CUTOFF = -1.0

# Transaction costs (one-way bps). Set to 0 to analyze gross only.
COST_BPS_ONEWAY = 10

# Regime splits
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


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 2:
        return np.nan
    s = r.std()
    if s == 0 or np.isnan(s):
        return np.nan
    return (r.mean() / s) * np.sqrt(252)


def annualized_vol(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 2:
        return np.nan
    return r.std() * np.sqrt(252)


def apply_subperiod(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    return df[(df[DATE_COL] >= start_dt) & (df[DATE_COL] <= end_dt)].copy()


def ols_alpha_beta(y: pd.Series, x: pd.Series):
    """
    Regress y ~ alpha + beta*x
    Returns alpha (daily), beta, r2, n
    """
    d = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    n = len(d)
    if n < 30:
        return {"alpha_daily": np.nan, "beta": np.nan, "r2": np.nan, "n": n}

    yv = d["y"].values
    xv = d["x"].values
    X = np.column_stack([np.ones(n), xv])
    # OLS: b = (X'X)^-1 X'y
    try:
        b = np.linalg.lstsq(X, yv, rcond=None)[0]
    except Exception:
        return {"alpha_daily": np.nan, "beta": np.nan, "r2": np.nan, "n": n}

    alpha = float(b[0])
    beta = float(b[1])

    yhat = X @ b
    ss_res = float(np.sum((yv - yhat) ** 2))
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {"alpha_daily": alpha, "beta": beta, "r2": r2, "n": n}


# =========================
# Build portfolio + benchmark
# =========================
def build_long_cohort(df: pd.DataFrame, hold_days: int, bottom_n: int) -> pd.DataFrame:
    fwd_col = f"fwd_ret_{hold_days}"
    sub = df[[DATE_COL, SYMBOL_COL, ZSCORE_COL, fwd_col]].dropna(subset=[ZSCORE_COL, fwd_col]).copy()

    if USE_ZSCORE_CUTOFF:
        sub = sub[sub[ZSCORE_COL] <= ZSCORE_LONG_CUTOFF].copy()

    sub["rank_low"] = sub.groupby(DATE_COL)[ZSCORE_COL].rank(method="first", ascending=True)
    longs = sub[sub["rank_low"] <= bottom_n].copy()

    cohort = longs.groupby(DATE_COL)[fwd_col].mean().rename("cohort_ret").to_frame()
    symsets = longs.groupby(DATE_COL)[SYMBOL_COL].apply(lambda x: set(x)).rename("symbols")
    cohort = cohort.join(symsets, how="inner")
    return cohort


def overlap_cohorts_to_daily(cohort: pd.DataFrame, hold_days: int) -> pd.DataFrame:
    daily = pd.DataFrame(index=cohort.index)
    daily["port_gross"] = cohort["cohort_ret"].rolling(window=hold_days).mean()
    daily = daily.dropna()
    return daily


def estimate_turnover_from_sets(cohort: pd.DataFrame, hold_days: int) -> float:
    sets = cohort["symbols"].dropna()
    if len(sets) < 2:
        return np.nan

    new_fracs = []
    for i in range(1, len(sets)):
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
    return avg_new_frac * (1.0 / hold_days)


def apply_costs(port: pd.Series, avg_daily_turnover: float, cost_bps_oneway: float) -> pd.Series:
    # daily drag = turnover * one-way cost * 2 (round-trip) in return units
    cost_per_day = avg_daily_turnover * (cost_bps_oneway / 10_000.0) * 2.0
    return port - cost_per_day


def build_equal_weight_benchmark(df: pd.DataFrame) -> pd.Series:
    """
    Equal-weight universe daily return:
      ret_1 = close(t+1)/close(t)-1 per symbol
      ew_ret(t) = mean over symbols available that day
    """
    g = df.groupby(SYMBOL_COL, group_keys=False)
    df = df.copy()
    df["ret_1"] = g[CLOSE_COL].shift(-1) / df[CLOSE_COL] - 1
    ew = df.dropna(subset=["ret_1"]).groupby(DATE_COL)["ret_1"].mean().rename("ew_universe_ret")
    return ew


def summarize_series(name: str, r: pd.Series) -> dict:
    r = r.dropna()
    return {
        "name": name,
        "days": int(len(r)),
        "avg_daily_ret": float(r.mean()) if len(r) else np.nan,
        "ann_ret_simple": float(r.mean() * 252) if len(r) else np.nan,
        "ann_vol": float(annualized_vol(r)),
        "sharpe": float(sharpe(r)),
        "win_rate": float((r > 0).mean()) if len(r) else np.nan,
    }


# =========================
# Main
# =========================
def main():
    print("\n====================")
    print("Long-Only MR: Beta / Alpha Diagnostics")
    print("====================\n")
    print(f"Universe file: {DATA_PATH}")
    print(f"Strategy: long bottom_n={BOTTOM_N} by {ZSCORE_COL}, hold_days={HOLD_DAYS} (overlapping)")
    print(f"Costs: {COST_BPS_ONEWAY} bps one-way (applied via turnover proxy)")
    print(f"Cutoff enabled: {USE_ZSCORE_CUTOFF}")
    if USE_ZSCORE_CUTOFF:
        print(f"  z <= {ZSCORE_LONG_CUTOFF}")
    print("Benchmark: equal-weight universe 1-day return\n")

    df = load_data()
    df = add_forward_returns(df, HOLD_DAYS)

    bench = build_equal_weight_benchmark(df)

    rows = []
    for period, start, end in SUBPERIODS:
        dfp = apply_subperiod(df, start, end)
        bench_p = apply_subperiod(bench.reset_index().rename(columns={0: "x"}), start, end).set_index(DATE_COL)["ew_universe_ret"]

        cohort = build_long_cohort(dfp, HOLD_DAYS, BOTTOM_N)
        port_daily = overlap_cohorts_to_daily(cohort, HOLD_DAYS)

        # Align benchmark to portfolio dates
        aligned = port_daily.join(bench_p, how="inner")
        if aligned.empty:
            print(f"{period}: no overlapping dates; skipping")
            continue

        turnover = estimate_turnover_from_sets(cohort.loc[aligned.index], HOLD_DAYS)

        # Gross / Net portfolio
        port_gross = aligned["port_gross"]
        port_net = apply_costs(port_gross, turnover, COST_BPS_ONEWAY)

        # Summary stats
        s_port_g = summarize_series("port_gross", port_gross)
        s_port_n = summarize_series("port_net", port_net)
        s_bench = summarize_series("bench_ew", aligned["ew_universe_ret"])

        # Regressions
        reg_g = ols_alpha_beta(port_gross, aligned["ew_universe_ret"])
        reg_n = ols_alpha_beta(port_net, aligned["ew_universe_ret"])

        # Annualize alpha (simple)
        alpha_ann_g = reg_g["alpha_daily"] * 252 if not np.isnan(reg_g["alpha_daily"]) else np.nan
        alpha_ann_n = reg_n["alpha_daily"] * 252 if not np.isnan(reg_n["alpha_daily"]) else np.nan

        rows.append({
            "period": period,
            "start": start,
            "end": end,
            "days": s_port_g["days"],
            "turnover_proxy": turnover,

            "sharpe_gross": s_port_g["sharpe"],
            "ann_ret_gross": s_port_g["ann_ret_simple"],
            "ann_vol_gross": s_port_g["ann_vol"],

            "sharpe_net": s_port_n["sharpe"],
            "ann_ret_net": s_port_n["ann_ret_simple"],
            "ann_vol_net": s_port_n["ann_vol"],

            "bench_sharpe": s_bench["sharpe"],
            "bench_ann_ret": s_bench["ann_ret_simple"],
            "bench_ann_vol": s_bench["ann_vol"],

            "beta_gross": reg_g["beta"],
            "alpha_ann_gross": alpha_ann_g,
            "r2_gross": reg_g["r2"],

            "beta_net": reg_n["beta"],
            "alpha_ann_net": alpha_ann_n,
            "r2_net": reg_n["r2"],
        })

    out = pd.DataFrame(rows)

    # Pretty print
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 200)
    fmt_cols = [
        "period", "days", "turnover_proxy",
        "sharpe_gross", "ann_ret_gross", "ann_vol_gross",
        "sharpe_net", "ann_ret_net", "ann_vol_net",
        "bench_sharpe", "bench_ann_ret", "bench_ann_vol",
        "beta_gross", "alpha_ann_gross", "r2_gross",
        "beta_net", "alpha_ann_net", "r2_net",
    ]
    print(out[fmt_cols].to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
    print("")

    # Save
    out_path = "cs_longonly_beta_alpha_diagnostics.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}\n")


if __name__ == "__main__":
    main()