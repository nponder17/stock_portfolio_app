import pandas as pd
import numpy as np

DATA_PATH = "featured_data.csv"

DATE_COL = "date"
SYMBOL_COL = "symbol"
ZSCORE_COL = "zscore_5"
CLOSE_COL = "close"

HOLD_DAYS = 3
BOTTOM_N_GRID = [5, 10, 20, 30, 50]
COST_BPS_ONEWAY = 10  # apply during selection + testing (set 0 for gross walkforward)

# Walk-forward config
TRAIN_YEARS = 5
TEST_YEARS = 1

USE_ZSCORE_CUTOFF = False
ZSCORE_LONG_CUTOFF = -1.0


def load_data():
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, SYMBOL_COL]).reset_index(drop=True)
    return df


def add_forward_returns(df: pd.DataFrame, hold_days: int):
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


def build_long_cohort(df: pd.DataFrame, hold_days: int, bottom_n: int):
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


def overlap_to_daily(cohort: pd.DataFrame, hold_days: int):
    daily = pd.DataFrame(index=cohort.index)
    daily["gross"] = cohort["cohort_ret"].rolling(window=hold_days).mean()
    return daily.dropna()


def turnover_proxy_from_sets(cohort: pd.DataFrame, hold_days: int) -> float:
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
        new_fracs.append(1.0 - overlap / max(len(s_now), 1))
    if not new_fracs:
        return np.nan
    return float(np.mean(new_fracs)) * (1.0 / hold_days)


def apply_costs(daily: pd.DataFrame, turn: float, bps_oneway: float) -> pd.Series:
    cost_per_day = turn * (bps_oneway / 10_000.0) * 2.0
    return daily["gross"] - cost_per_day


def slice_dates(df, start, end):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    return df[(df[DATE_COL] >= start) & (df[DATE_COL] <= end)].copy()


def main():
    print("\n====================")
    print("Walk-forward OOS (select bottom_n on train, test forward)")
    print("====================\n")
    print(f"HOLD_DAYS={HOLD_DAYS}, GRID={BOTTOM_N_GRID}, COST_BPS_ONEWAY={COST_BPS_ONEWAY}")
    print(f"TRAIN_YEARS={TRAIN_YEARS}, TEST_YEARS={TEST_YEARS}\n")

    df = load_data()
    df = add_forward_returns(df, HOLD_DAYS)

    start_date = df[DATE_COL].min()
    end_date = df[DATE_COL].max()

    # Build year boundaries
    years = sorted(df[DATE_COL].dt.year.unique())
    results = []

    # Walk-forward by calendar years
    for i in range(len(years)):
        train_start_year = years[i]
        train_end_year = train_start_year + TRAIN_YEARS - 1
        test_start_year = train_end_year + 1
        test_end_year = test_start_year + TEST_YEARS - 1

        if test_end_year > years[-1]:
            break

        train_start = f"{train_start_year}-01-01"
        train_end = f"{train_end_year}-12-31"
        test_start = f"{test_start_year}-01-01"
        test_end = f"{test_end_year}-12-31"

        df_train = slice_dates(df, train_start, train_end)
        df_test = slice_dates(df, test_start, test_end)

        if df_train.empty or df_test.empty:
            continue

        # Select best bottom_n on training
        best = {"bottom_n": None, "sharpe": -np.inf, "turn": np.nan}
        for n in BOTTOM_N_GRID:
            cohort = build_long_cohort(df_train, HOLD_DAYS, n)
            daily = overlap_to_daily(cohort, HOLD_DAYS)
            if daily.empty:
                continue
            turn = turnover_proxy_from_sets(cohort.loc[daily.index], HOLD_DAYS)
            net = apply_costs(daily, turn, COST_BPS_ONEWAY)
            s = sharpe(net)
            if np.isnan(s):
                continue
            if s > best["sharpe"]:
                best = {"bottom_n": n, "sharpe": s, "turn": turn}

        # Evaluate best on test
        n_star = best["bottom_n"]
        cohort_t = build_long_cohort(df_test, HOLD_DAYS, n_star)
        daily_t = overlap_to_daily(cohort_t, HOLD_DAYS)
        turn_t = turnover_proxy_from_sets(cohort_t.loc[daily_t.index], HOLD_DAYS)
        net_t = apply_costs(daily_t, turn_t, COST_BPS_ONEWAY)

        results.append({
            "train": f"{train_start_year}-{train_end_year}",
            "test": f"{test_start_year}-{test_end_year}",
            "chosen_bottom_n": n_star,
            "train_sharpe_net": best["sharpe"],
            "test_sharpe_net": sharpe(net_t),
            "test_ann_ret_net": float(net_t.mean() * 252),
            "test_ann_vol": float(net_t.std() * np.sqrt(252)),
            "test_days": int(len(net_t.dropna())),
            "test_turnover": turn_t
        })

    out = pd.DataFrame(results)
    print(out.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
    out.to_csv("cs_walkforward_oos.csv", index=False)
    print("\nSaved: cs_walkforward_oos.csv\n")


if __name__ == "__main__":
    main()