import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# WINNER STRATEGY CONFIG
# =========================
DATA_PATH = Path("featured_data.csv")

DATE_COL = "date"
SYMBOL_COL = "symbol"
CLOSE_COL = "close"
ZSCORE_COL = "zscore_5"
RSI_COL = "rsi_2"

HOLD_DAYS = 3
BOTTOM_N = 10

# Hybrid filter AFTER ranking bottom_n by RSI:
ZSCORE_FILTER = -1.0   # keep only if zscore_5 <= -1.0

# Trading cost model
COST_BPS_ONEWAY = 10   # 10 bps = 0.10% one-way

# Output files
OUT_WEIGHTS = Path("winner_weights_hold3_rsi_zle-1_n10.csv")
OUT_TURNOVER = Path("winner_turnover_hold3_rsi_zle-1_n10.csv")
OUT_PORT = Path("winner_portfolio_hold3_rsi_zle-1_n10.csv")
OUT_EVENTS = Path("cs_trade_events_hold3_rsi_zle-1_n10.csv")
OUT_PAIRS = Path("cs_trade_pairs_hold3_rsi_zle-1_n10.csv")


# =========================
# Helpers
# =========================
def add_forward_returns(df: pd.DataFrame, hold_days: int) -> pd.DataFrame:
    g = df.groupby(SYMBOL_COL, group_keys=False)
    fwd_col = f"fwd_ret_{hold_days}"
    if fwd_col not in df.columns:
        df[fwd_col] = g[CLOSE_COL].shift(-hold_days) / df[CLOSE_COL] - 1
    return df


def build_picks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily:
      1) rank by RSI ascending
      2) take bottom N
      3) filter: zscore_5 <= ZSCORE_FILTER  (AFTER ranking)
    """
    df = df.copy()

    need = {DATE_COL, SYMBOL_COL, RSI_COL, ZSCORE_COL, CLOSE_COL}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"featured_data.csv missing required columns: {missing}")

    df = add_forward_returns(df, HOLD_DAYS)
    fwd_col = f"fwd_ret_{HOLD_DAYS}"

    sub = df[[DATE_COL, SYMBOL_COL, RSI_COL, ZSCORE_COL, CLOSE_COL, fwd_col]].dropna().copy()

    # Rank by RSI per day
    sub["score"] = sub[RSI_COL]
    sub["rank"] = sub.groupby(DATE_COL)["score"].rank(method="first", ascending=True)

    picks = sub[sub["rank"] <= BOTTOM_N].copy()

    # Hybrid filter AFTER bottom_n selection
    picks = picks[picks[ZSCORE_COL] <= ZSCORE_FILTER].copy()

    return picks


def build_overlapping_weights(picks: pd.DataFrame, hold_days: int) -> pd.DataFrame:
    """
    Cohort weights on day d: equal-weight among that day's picks.
    Portfolio weights on day t: average of last hold_days cohort weights.
    """
    if picks.empty:
        return pd.DataFrame()

    # cohort equal weight by day
    k = picks.groupby(DATE_COL)[SYMBOL_COL].count().rename("k")
    picks = picks.join(k, on=DATE_COL)
    picks["w_cohort"] = 1.0 / picks["k"].astype(float)

    cohort_w = picks.pivot_table(
        index=DATE_COL,
        columns=SYMBOL_COL,
        values="w_cohort",
        aggfunc="sum",
        fill_value=0.0
    ).sort_index()

    # overlapping book
    w_port = cohort_w.rolling(window=hold_days).mean().dropna()
    return w_port


def build_portfolio_returns_from_cohorts(picks: pd.DataFrame, hold_days: int) -> pd.Series:
    """
    Cohort return on day d = mean fwd_ret_hold among picks formed on d.
    Portfolio daily return = rolling mean of last hold_days cohort returns.
    """
    if picks.empty:
        return pd.Series(dtype=float)

    fwd_col = f"fwd_ret_{hold_days}"
    cohort_ret = picks.groupby(DATE_COL)[fwd_col].mean().rename("cohort_ret").sort_index()
    port_gross = cohort_ret.rolling(window=hold_days).mean().dropna().rename("port_gross")
    return port_gross


def turnover_from_weights(w_port: pd.DataFrame) -> pd.Series:
    """
    turnover_t = 0.5 * sum_i |w_t,i - w_{t-1,i}|
    """
    if w_port.empty:
        return pd.Series(dtype=float)
    dw = w_port.diff()
    turnover = 0.5 * dw.abs().sum(axis=1)
    return turnover.rename("turnover")


def build_trade_events(w_port: pd.DataFrame) -> pd.DataFrame:
    """
    Create per-date per-symbol events where weight changes.
    Columns:
      date, symbol, event, w_prev, w_now, dw
    """
    if w_port.empty:
        return pd.DataFrame(columns=["date", "symbol", "event", "w_prev", "w_now", "dw"])

    w_prev = w_port.shift(1).fillna(0.0)
    dw = (w_port - w_prev)

    # melt to long format only where changed
    events = dw.stack().rename("dw").reset_index()
    events.columns = ["date", "symbol", "dw"]

    # keep meaningful changes
    eps = 1e-12
    events = events[events["dw"].abs() > eps].copy()

    # attach prev/now
    prev_long = w_prev.stack().rename("w_prev").reset_index()
    prev_long.columns = ["date", "symbol", "w_prev"]

    now_long = w_port.stack().rename("w_now").reset_index()
    now_long.columns = ["date", "symbol", "w_now"]

    events = events.merge(prev_long, on=["date", "symbol"], how="left")
    events = events.merge(now_long, on=["date", "symbol"], how="left")

    # classify event
    def classify(row):
        wp = row["w_prev"]
        wn = row["w_now"]
        if wp == 0 and wn > 0:
            return "ENTRY"
        if wp > 0 and wn == 0:
            return "EXIT"
        return "REBAL"

    events["event"] = events.apply(classify, axis=1)
    events = events.sort_values(["symbol", "date"]).reset_index(drop=True)
    return events[["date", "symbol", "event", "w_prev", "w_now", "dw"]]


def build_trade_pairs(events: pd.DataFrame) -> pd.DataFrame:
    """
    For each symbol, pair ENTRY -> EXIT sequentially and assign trade_id.
    Output:
      symbol, trade_id, entry_date, exit_date, holding_days
    """
    if events.empty:
        return pd.DataFrame(columns=["symbol", "trade_id", "entry_date", "exit_date", "holding_days"])

    out_rows = []

    for sym, g in events.sort_values(["symbol", "date"]).groupby("symbol"):
        trade_id = 0
        open_entry = None

        for _, row in g.iterrows():
            if row["event"] == "ENTRY":
                # If we somehow have overlapping entries without exit, keep the earliest open one.
                if open_entry is None:
                    open_entry = row["date"]
                    trade_id += 1
            elif row["event"] == "EXIT":
                if open_entry is not None:
                    entry_date = open_entry
                    exit_date = row["date"]
                    holding = (exit_date - entry_date).days
                    out_rows.append({
                        "symbol": sym,
                        "trade_id": trade_id,
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "holding_days": holding
                    })
                    open_entry = None

        # If a position is still open at end, we leave it unpaired (no exit yet)

    pairs = pd.DataFrame(out_rows)
    if not pairs.empty:
        pairs = pairs.sort_values(["symbol", "trade_id"]).reset_index(drop=True)
    return pairs


def main():
    print("\n====================")
    print("Build Trade Events — WINNER Strategy")
    print("====================\n")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find {DATA_PATH.resolve()}")

    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, SYMBOL_COL]).reset_index(drop=True)

    # 1) Picks
    picks = build_picks(df)
    if picks.empty:
        print("No picks after filters. Try loosening thresholds.")
        return

    # 2) Weights + turnover
    w_port = build_overlapping_weights(picks, HOLD_DAYS)
    turnover = turnover_from_weights(w_port)

    # 3) Portfolio returns
    port_gross = build_portfolio_returns_from_cohorts(picks, HOLD_DAYS)

    # align indices
    common_idx = port_gross.index.intersection(w_port.index).intersection(turnover.index)
    w_port = w_port.loc[common_idx]
    turnover = turnover.loc[common_idx]
    port_gross = port_gross.loc[common_idx]

    # 4) Net returns using correct cost model
    cost_oneway = COST_BPS_ONEWAY / 10000.0
    daily_cost = (2.0 * cost_oneway * turnover).rename("daily_cost")
    port_net = (port_gross - daily_cost).rename("port_net")

    # 5) Trade events + pairs
    events = build_trade_events(w_port)
    pairs = build_trade_pairs(events)

    # Save outputs
    w_port.to_csv(OUT_WEIGHTS)
    turnover.to_frame().to_csv(OUT_TURNOVER)

    port_df = pd.concat([port_gross, daily_cost, port_net], axis=1)
    port_df.to_csv(OUT_PORT)

    events.to_csv(OUT_EVENTS, index=False)
    pairs.to_csv(OUT_PAIRS, index=False)

    # Print quick summary
    print("Strategy params:")
    print(f"  HOLD_DAYS={HOLD_DAYS}, BOTTOM_N={BOTTOM_N}, RSI rank ascending, filter zscore<= {ZSCORE_FILTER}")
    print(f"  COST_BPS_ONEWAY={COST_BPS_ONEWAY} (net uses 2*cost*turnover)")

    print("\nSaved:")
    print(f"  {OUT_WEIGHTS}")
    print(f"  {OUT_TURNOVER}")
    print(f"  {OUT_PORT}")
    print(f"  {OUT_EVENTS}")
    print(f"  {OUT_PAIRS}")

    print("\nDiagnostics:")
    print(f"  days: {len(port_df)}")
    print(f"  avg_turnover: {turnover.mean():.4f}")
    print(f"  avg_breadth (names held): {(w_port > 0).sum(axis=1).mean():.2f}")
    print(f"  avg_daily_ret_net: {port_net.mean():.6f}")
    if port_net.std() > 0:
        sh = (port_net.mean() / port_net.std()) * np.sqrt(252)
        print(f"  Sharpe_net: {sh:.3f}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()