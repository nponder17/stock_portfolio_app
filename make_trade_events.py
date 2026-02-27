import pandas as pd
import numpy as np
from pathlib import Path
from collections import deque, defaultdict

DATA_PATH = Path(__file__).resolve().parent / "featured_data.csv"
OUT_PATH = Path(__file__).resolve().parent / "cs_trade_events_hold3_n50.csv"

DATE_COL = "date"
SYMBOL_COL = "symbol"
ZSCORE_COL = "zscore_5"

HOLD_DAYS = 3
BOTTOM_N = 50

USE_ZSCORE_CUTOFF = False
ZSCORE_LONG_CUTOFF = -1.0


def load_data():
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, SYMBOL_COL]).reset_index(drop=True)
    return df


def build_daily_cohort_picks(df: pd.DataFrame) -> dict:
    sub = df[[DATE_COL, SYMBOL_COL, ZSCORE_COL]].dropna(subset=[ZSCORE_COL]).copy()
    if USE_ZSCORE_CUTOFF:
        sub = sub[sub[ZSCORE_COL] <= ZSCORE_LONG_CUTOFF].copy()

    sub["rank_low"] = sub.groupby(DATE_COL)[ZSCORE_COL].rank(method="first", ascending=True)
    picks = sub[sub["rank_low"] <= BOTTOM_N]
    return picks.groupby(DATE_COL)[SYMBOL_COL].apply(list).to_dict()


def build_overlapping_target_weights(all_dates, date_to_syms):
    """
    Overlapping book:
      each day holds HOLD_DAYS cohorts equally weighted
      within cohort: equal weight across names
    returns: dict date -> dict(symbol->weight)
    """
    window = deque(maxlen=HOLD_DAYS)
    targets = {}

    for dt in all_dates:
        window.append(date_to_syms.get(dt, []))

        if len(window) < HOLD_DAYS:
            continue

        w = defaultdict(float)
        per_cohort = 1.0 / HOLD_DAYS
        per_name = per_cohort / max(BOTTOM_N, 1)

        for cohort_syms in window:
            for s in cohort_syms:
                w[s] += per_name

        # normalize to sum to 1 (safe)
        tot = sum(w.values())
        if tot > 0:
            for s in list(w.keys()):
                w[s] /= tot

        targets[dt] = dict(w)

    return targets


def weight_events(targets: dict, eps: float = 1e-12) -> pd.DataFrame:
    dates = sorted(targets.keys())
    prev = {}
    rows = []

    for dt in dates:
        now = targets[dt]
        syms = set(prev.keys()) | set(now.keys())

        for s in syms:
            w_prev = prev.get(s, 0.0)
            w_now = now.get(s, 0.0)
            if abs(w_now - w_prev) < eps:
                continue

            # Entry/Exit definition
            if w_prev <= eps and w_now > eps:
                event = "ENTRY"
            elif w_prev > eps and w_now <= eps:
                event = "EXIT"
            else:
                event = "REBAL"  # still held but weight changed

            rows.append(
                {
                    "date": dt,
                    "symbol": s,
                    "event": event,
                    "w_prev": w_prev,
                    "w_now": w_now,
                    "dw": w_now - w_prev,
                }
            )

        prev = now

    ev = pd.DataFrame(rows).sort_values(["date", "symbol"])
    return ev


def main():
    df = load_data()
    all_dates = sorted(df[DATE_COL].unique())
    date_to_syms = build_daily_cohort_picks(df)
    targets = build_overlapping_target_weights(all_dates, date_to_syms)
    ev = weight_events(targets)

    ev.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()