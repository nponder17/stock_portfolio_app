import pandas as pd
import numpy as np
from collections import deque, Counter, defaultdict

# =========================
# Config
# =========================
DATA_PATH = "featured_data.csv"

DATE_COL = "date"
SYMBOL_COL = "symbol"
ZSCORE_COL = "zscore_5"
CLOSE_COL = "close"

# Liquidity columns (script will auto-detect)
VOLUME_COL_CANDIDATES = ["volume", "vol", "Volume", "VOL"]
DOLLAR_VOL_COL_CANDIDATES = ["dollar_volume", "dv", "DollarVolume"]

HOLD_DAYS = 3
BOTTOM_N = 50
ADV_WINDOW = 20  # rolling days for ADV($)

# AUM levels to test
AUM_LIST = [1e6, 5e6, 10e6, 25e6, 50e6, 100e6]

# Fixed cost stress (one-way bps)
FIXED_COST_BPS_LIST = [0, 5, 10, 20, 40]

# Impact model (very simple, tunable)
# slip_bps = BASE_BPS + K_SQRT * sqrt(participation)
# participation is fraction of ADV ($ traded / ADV$)
IMPACT_BASE_BPS = 1.0
IMPACT_K_SQRT = 30.0

# Optional oversold cutoff
USE_ZSCORE_CUTOFF = False
ZSCORE_LONG_CUTOFF = -1.0


# =========================
# Helpers
# =========================
def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 2:
        return np.nan
    s = r.std()
    if s == 0 or np.isnan(s):
        return np.nan
    return (r.mean() / s) * np.sqrt(252)


def load_data():
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, SYMBOL_COL]).reset_index(drop=True)
    return df


def detect_liquidity_columns(df: pd.DataFrame):
    vol_col = next((c for c in VOLUME_COL_CANDIDATES if c in df.columns), None)
    dv_col = next((c for c in DOLLAR_VOL_COL_CANDIDATES if c in df.columns), None)
    return vol_col, dv_col


def add_returns(df: pd.DataFrame):
    g = df.groupby(SYMBOL_COL, group_keys=False)
    df = df.copy()
    df["ret_1"] = g[CLOSE_COL].shift(-1) / df[CLOSE_COL] - 1
    return df


def build_cohort_sets(df: pd.DataFrame):
    """
    For each date t: pick bottom_n symbols by zscore_5.
    Returns: dict date -> list(symbols)
    """
    sub = df[[DATE_COL, SYMBOL_COL, ZSCORE_COL]].dropna(subset=[ZSCORE_COL]).copy()
    if USE_ZSCORE_CUTOFF:
        sub = sub[sub[ZSCORE_COL] <= ZSCORE_LONG_CUTOFF].copy()

    sub["rank_low"] = sub.groupby(DATE_COL)[ZSCORE_COL].rank(method="first", ascending=True)
    picks = sub[sub["rank_low"] <= BOTTOM_N].copy()

    # date -> list of symbols
    date_to_syms = picks.groupby(DATE_COL)[SYMBOL_COL].apply(list).to_dict()
    return date_to_syms


def build_daily_target_weights(dates, date_to_syms):
    """
    Overlapping cohort weights:
      each day t holds cohorts formed at t, t-1, ..., t-(HOLD_DAYS-1)
      each cohort has weight 1/HOLD_DAYS and equal-weight within cohort: 1/BOTTOM_N
    We compute per-date dict: symbol -> target_weight
    """
    window = deque(maxlen=HOLD_DAYS)
    targets = {}

    for dt in dates:
        syms_today = date_to_syms.get(dt, [])
        window.append(syms_today)

        # Need full window to be "invested"
        if len(window) < HOLD_DAYS:
            continue

        w = defaultdict(float)
        cohort_slice = 1.0 / HOLD_DAYS
        per_name = cohort_slice / max(BOTTOM_N, 1)

        for cohort_syms in window:
            for s in cohort_syms:
                w[s] += per_name

        # Normalize safety (should already sum ~1 if cohorts full size)
        total_w = sum(w.values())
        if total_w > 0:
            for k in list(w.keys()):
                w[k] /= total_w

        targets[dt] = dict(w)

    return targets


def weights_to_trades(targets: dict):
    """
    Compute daily trades from target weights:
      delta_w(s) = w_t(s) - w_{t-1}(s)
      one-way turnover fraction = 0.5 * sum(abs(delta_w))
    Return: DataFrame indexed by date with turnover fraction
            and dict date -> dict(symbol -> abs(delta_w))
    """
    dates = sorted(targets.keys())
    prev = {}
    turnover = []
    abs_deltas_by_date = {}

    for dt in dates:
        w = targets[dt]
        syms = set(prev.keys()) | set(w.keys())
        abs_d = {}
        total_abs = 0.0
        for s in syms:
            dw = w.get(s, 0.0) - prev.get(s, 0.0)
            adw = abs(dw)
            if adw > 0:
                abs_d[s] = adw
            total_abs += adw

        one_way = 0.5 * total_abs
        turnover.append((dt, one_way))
        abs_deltas_by_date[dt] = abs_d
        prev = w

    turn_df = pd.DataFrame(turnover, columns=[DATE_COL, "turnover_oneway"]).set_index(DATE_COL)
    return turn_df, abs_deltas_by_date


def compute_adv_dollars(df: pd.DataFrame, vol_col: str, dv_col: str):
    """
    ADV$ per symbol per day (rolling mean).
    Uses:
      - dollar_volume column if present
      - else close * volume
    Returns a DataFrame with columns: date, symbol, adv_dollars
    """
    d = df[[DATE_COL, SYMBOL_COL, CLOSE_COL]].copy()

    if dv_col is not None:
        d["dollar_volume"] = df[dv_col].astype(float)
    else:
        if vol_col is None:
            raise ValueError("No volume/dollar_volume column found. Add volume or dollar_volume to featured_data.csv.")
        d["dollar_volume"] = df[vol_col].astype(float) * df[CLOSE_COL].astype(float)

    d = d.sort_values([SYMBOL_COL, DATE_COL])
    g = d.groupby(SYMBOL_COL, group_keys=False)
    d["adv_dollars"] = g["dollar_volume"].rolling(ADV_WINDOW).mean().reset_index(level=0, drop=True)
    return d[[DATE_COL, SYMBOL_COL, "adv_dollars"]]


def impact_slippage_bps(participation: float) -> float:
    # Simple concave impact: base + k * sqrt(participation)
    if np.isnan(participation) or participation <= 0:
        return IMPACT_BASE_BPS
    return IMPACT_BASE_BPS + IMPACT_K_SQRT * np.sqrt(participation)


# =========================
# Main
# =========================
def main():
    print("\n====================")
    print("Capacity + Slippage Diagnostics (Long-only overlapping cohorts)")
    print("====================\n")
    print(f"Params: hold_days={HOLD_DAYS}, bottom_n={BOTTOM_N}, ADV_WINDOW={ADV_WINDOW}")
    print(f"Fixed cost bps (one-way): {FIXED_COST_BPS_LIST}")
    print(f"Impact model: slip_bps = {IMPACT_BASE_BPS} + {IMPACT_K_SQRT}*sqrt(participation)\n")

    df = load_data()
    df = add_returns(df)

    vol_col, dv_col = detect_liquidity_columns(df)
    if vol_col is None and dv_col is None:
        raise SystemExit(
            "ERROR: Could not find volume or dollar_volume column in featured_data.csv.\n"
            "Add one of these columns (volume or dollar_volume) and rerun."
        )
    print(f"Detected liquidity columns: volume={vol_col}, dollar_volume={dv_col}\n")

    # Build cohorts and targets
    dates = sorted(df[DATE_COL].unique())
    date_to_syms = build_cohort_sets(df)
    targets = build_daily_target_weights(dates, date_to_syms)

    # Trades & turnover
    turn_df, abs_deltas_by_date = weights_to_trades(targets)

    # ADV$
    adv = compute_adv_dollars(df, vol_col, dv_col)
    adv = adv.dropna(subset=["adv_dollars"])

    # Merge benchmark and portfolio gross returns (portfolio return = sum weights * next-day ret_1)
    # NOTE: Your MR research uses fwd_ret_3 for cohort returns.
    # For capacity & participation, we care about trade sizes; for PnL we’ll still estimate daily return
    # using the overlapping structure on ret_1 as a proxy of daily PnL path.
    # If you want PnL exactly matching fwd_ret_3 smoothing, tell me and I’ll align it.
    ret_panel = df[[DATE_COL, SYMBOL_COL, "ret_1"]].dropna()
    ret_map = ret_panel.set_index([DATE_COL, SYMBOL_COL])["ret_1"]

    port_gross = []
    for dt, w in targets.items():
        # daily return from next-day ret_1 (proxy for PnL path)
        rs = 0.0
        denom = 0.0
        for s, ws in w.items():
            r = ret_map.get((dt, s), np.nan)
            if not np.isnan(r):
                rs += ws * r
                denom += ws
        port_gross.append((dt, rs / denom if denom > 0 else np.nan))
    port_gross = pd.DataFrame(port_gross, columns=[DATE_COL, "port_gross_ret"]).set_index(DATE_COL).dropna()

    # Prepare ADV lookup
    adv_map = adv.set_index([DATE_COL, SYMBOL_COL])["adv_dollars"]

    # Capacity & slippage for each AUM
    for aum in AUM_LIST:
        # participation stats across all traded names
        parts = []
        worst_rows = []

        for dt, abs_d in abs_deltas_by_date.items():
            # If we don’t have ADV for that date, skip
            # (early days before ADV_WINDOW)
            for s, adw in abs_d.items():
                trade_dollars = adw * aum
                adv_d = adv_map.get((dt, s), np.nan)
                if np.isnan(adv_d) or adv_d <= 0:
                    continue
                participation = trade_dollars / adv_d
                parts.append(participation)

        parts = np.array(parts) if len(parts) else np.array([np.nan])
        p50 = np.nanpercentile(parts, 50)
        p90 = np.nanpercentile(parts, 90)
        p95 = np.nanpercentile(parts, 95)
        p99 = np.nanpercentile(parts, 99)
        pmax = np.nanmax(parts)

        print(f"\n--- AUM = ${aum:,.0f} ---")
        print(f"Participation (trade$/ADV$) percentiles:")
        print(f"  p50={p50:.4f}  p90={p90:.4f}  p95={p95:.4f}  p99={p99:.4f}  max={pmax:.4f}")

        # Apply fixed-cost stress to portfolio returns using turnover fraction
        joined = port_gross.join(turn_df, how="inner").dropna()
        if joined.empty:
            print("  (No overlapping dates after joins)")
            continue

        for bps in FIXED_COST_BPS_LIST:
            cost = joined["turnover_oneway"] * (bps / 10_000.0) * 2.0
            net = joined["port_gross_ret"] - cost
            print(f"  Fixed cost {bps:>2} bps one-way -> Sharpe {sharpe(net):.3f}, ann_ret {net.mean()*252:.3f}")

        # Impact-model slippage: convert participation percentiles into an implied bps “typical”
        # (This is coarse; true impact should be computed per-symbol per-day and aggregated.)
        slip_p50 = impact_slippage_bps(p50)
        slip_p95 = impact_slippage_bps(p95)
        slip_p99 = impact_slippage_bps(p99)
        print("  Impact model implied slippage (bps one-way):")
        print(f"    p50_part -> {slip_p50:.2f} bps, p95_part -> {slip_p95:.2f} bps, p99_part -> {slip_p99:.2f} bps")

    print("\nDone.\n")


if __name__ == "__main__":
    main()