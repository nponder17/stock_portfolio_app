import pandas as pd
import numpy as np

DATA_PATH = "featured_data.csv"

DATE_COL = "date"
SYMBOL_COL = "symbol"
ZSCORE_COL = "zscore_5"
CLOSE_COL = "close"

HOLD_DAYS = 3
BOTTOM_N = 50

# stress one-way bps
COST_BPS_LIST = [0, 5, 10, 20, 40, 75]

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
    daily = daily.dropna()
    return daily


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
        new_frac = 1.0 - overlap / max(len(s_now), 1)
        new_fracs.append(new_frac)

    if not new_fracs:
        return np.nan

    avg_new = float(np.mean(new_fracs))
    return avg_new * (1.0 / hold_days)


def main():
    print("\n====================")
    print("Slippage Stress (exact fwd_ret overlap)")
    print("====================\n")

    df = load_data()
    df = add_forward_returns(df, HOLD_DAYS)

    cohort = build_long_cohort(df, HOLD_DAYS, BOTTOM_N)
    daily = overlap_to_daily(cohort, HOLD_DAYS)
    turn = turnover_proxy_from_sets(cohort.loc[daily.index], HOLD_DAYS)

    print(f"Params: hold_days={HOLD_DAYS}, bottom_n={BOTTOM_N}, turnover_proxy={turn:.4f}")
    print(f"Gross: Sharpe={sharpe(daily['gross']):.3f}, ann_ret={daily['gross'].mean()*252:.3f}, ann_vol={daily['gross'].std()*np.sqrt(252):.3f}\n")

    for bps in COST_BPS_LIST:
        cost_per_day = turn * (bps / 10_000.0) * 2.0
        net = daily["gross"] - cost_per_day
        print(f"Cost {bps:>3} bps one-way -> Sharpe={sharpe(net):.3f}, ann_ret={net.mean()*252:.3f}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()