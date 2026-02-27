import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple
from pathlib import Path

# =========================
# Config
# =========================
DATA_PATH = Path("featured_data.csv")

DATE_COL = "date"
SYMBOL_COL = "symbol"
CLOSE_COL = "close"
ZSCORE_COL = "zscore_5"
RSI_COL = "rsi_2"

HOLD_DAYS = 3
BOTTOM_N_GRID = [5, 10, 20, 30, 50]

# Costs (one-way bps, per dollar traded)
COST_BPS_ONEWAY = 10

# Walk-forward
TRAIN_YEARS = 5
TEST_YEARS = 1

# Output
OUT_RESULTS = Path("cs_experiment_matrix_results.csv")
OUT_DIR = Path("cs_experiments_out")
OUT_DIR.mkdir(exist_ok=True)

# Threshold grids
ZSCORE_THRESHOLDS = [None, -1.0, -1.5, -2.0]
RSI_THRESHOLDS = [None, 20.0, 15.0, 10.0]


# =========================
# Stats helpers
# =========================
def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 2 or r.std() == 0:
        return np.nan
    return float((r.mean() / r.std()) * np.sqrt(252))


def ann_stats(r: pd.Series) -> Tuple[float, float, float]:
    r = r.dropna()
    if r.empty:
        return (np.nan, np.nan, np.nan)
    ann_ret = float(r.mean() * 252)
    ann_vol = float(r.std() * np.sqrt(252))
    sh = sharpe(r)
    return ann_ret, ann_vol, sh


def add_forward_returns(df: pd.DataFrame, hold_days: int) -> pd.DataFrame:
    g = df.groupby(SYMBOL_COL, group_keys=False)
    fwd_col = f"fwd_ret_{hold_days}"
    if fwd_col not in df.columns:
        df[fwd_col] = g[CLOSE_COL].shift(-hold_days) / df[CLOSE_COL] - 1
    return df


def equal_weight_benchmark_1d(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    df = add_forward_returns(df, hold_days=1)
    return df.groupby(DATE_COL)["fwd_ret_1"].mean().rename("bench_ret")


def beta_alpha(strategy_ret: pd.Series, bench_ret: pd.Series) -> Tuple[float, float, float]:
    tmp = pd.concat([strategy_ret.rename("s"), bench_ret.rename("b")], axis=1).dropna()
    if tmp.shape[0] < 60:
        return (np.nan, np.nan, np.nan)

    x = tmp["b"].values
    y = tmp["s"].values
    x_mean = x.mean()
    y_mean = y.mean()

    var_x = np.sum((x - x_mean) ** 2)
    if var_x == 0:
        return (np.nan, np.nan, np.nan)

    cov_xy = np.sum((x - x_mean) * (y - y_mean))
    beta = cov_xy / var_x
    alpha_daily = y_mean - beta * x_mean

    y_hat = alpha_daily + beta * x
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return float(beta), float(alpha_daily * 252), float(r2)


def year_windows(df: pd.DataFrame) -> List[int]:
    return sorted(df[DATE_COL].dt.year.unique().tolist())


# =========================
# Strategy definition
# =========================
@dataclass
class StrategySpec:
    name: str
    rank_mode: str               # "zscore" | "rsi" | "combo"
    bottom_n: int
    hold_days: int
    cost_bps_oneway: float
    zscore_filter: Optional[float] = None   # keep only if zscore <= threshold (AFTER ranking bottom_n)
    rsi_filter: Optional[float] = None      # keep only if rsi <= threshold (AFTER ranking bottom_n)


def compute_daily_score(df_day: pd.DataFrame, rank_mode: str) -> pd.Series:
    """
    Lower score = more "oversold".
    - zscore: score = zscore_5
    - rsi:    score = rsi_2
    - combo:  score = rank(zscore) + rank(rsi)
    """
    if rank_mode == "zscore":
        return df_day[ZSCORE_COL]
    if rank_mode == "rsi":
        return df_day[RSI_COL]
    if rank_mode == "combo":
        rz = df_day[ZSCORE_COL].rank(method="first", ascending=True)
        rr = df_day[RSI_COL].rank(method="first", ascending=True)
        return rz + rr
    raise ValueError(f"Unknown rank_mode: {rank_mode}")


def build_picks(df: pd.DataFrame, spec: StrategySpec) -> pd.DataFrame:
    """
    Returns per-day picks after:
      rank by score -> take bottom_n -> apply optional filters (AND behavior)
    """
    df = df.copy()
    df = add_forward_returns(df, spec.hold_days)
    fwd_col = f"fwd_ret_{spec.hold_days}"

    need = {DATE_COL, SYMBOL_COL, ZSCORE_COL, RSI_COL, fwd_col}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    sub = df[[DATE_COL, SYMBOL_COL, ZSCORE_COL, RSI_COL, fwd_col]].dropna().copy()

    # score and rank per day
    sub["score"] = sub.groupby(DATE_COL, group_keys=False).apply(
        lambda g: compute_daily_score(g, spec.rank_mode)
    ).values
    sub["rank"] = sub.groupby(DATE_COL)["score"].rank(method="first", ascending=True)

    picks = sub[sub["rank"] <= spec.bottom_n].copy()

    # Apply filters AFTER bottom_n selection (hybrid “AND”)
    if spec.zscore_filter is not None:
        picks = picks[picks[ZSCORE_COL] <= spec.zscore_filter]
    if spec.rsi_filter is not None:
        picks = picks[picks[RSI_COL] <= spec.rsi_filter]

    return picks


def build_overlapping_weights(picks: pd.DataFrame, hold_days: int) -> pd.DataFrame:
    """
    Build daily portfolio target weights for overlapping cohorts.

    cohort weights on day d:
      w_cohort[d, sym] = 1 / k_d  if sym in picks(d) else 0

    portfolio weights on day t:
      w_port[t] = mean( w_cohort[t], w_cohort[t-1], ..., w_cohort[t-h+1] )
    """
    if picks.empty:
        return pd.DataFrame()

    # cohort weights
    counts = picks.groupby(DATE_COL)[SYMBOL_COL].count().rename("k")
    picks = picks.join(counts, on=DATE_COL)
    picks["w_cohort"] = 1.0 / picks["k"].astype(float)

    cohort_w = picks.pivot_table(
        index=DATE_COL,
        columns=SYMBOL_COL,
        values="w_cohort",
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()

    # overlapping portfolio weights
    w_port = cohort_w.rolling(window=hold_days).mean().dropna()
    return w_port


def compute_turnover_from_weights(w_port: pd.DataFrame) -> pd.Series:
    """
    turnover_t = 0.5 * sum_i |w_t,i - w_{t-1,i}|
    """
    if w_port is None or w_port.empty:
        return pd.Series(dtype=float)

    dw = w_port.diff()
    turnover = 0.5 * dw.abs().sum(axis=1)
    return turnover.rename("turnover")


def build_portfolio_returns_from_cohorts(df: pd.DataFrame, picks: pd.DataFrame, hold_days: int) -> pd.Series:
    """
    Cohort return on formation date d = mean fwd_ret_hold among picks formed on d
    Overlapping portfolio return at t = mean(cohort_ret[t], ..., cohort_ret[t-h+1])
    """
    if picks.empty:
        return pd.Series(dtype=float)

    fwd_col = f"fwd_ret_{hold_days}"
    cohort_ret = picks.groupby(DATE_COL)[fwd_col].mean().rename("cohort_ret").sort_index()
    port_gross = cohort_ret.rolling(window=hold_days).mean().dropna().rename("port_gross")
    return port_gross


def run_strategy(df: pd.DataFrame, spec: StrategySpec) -> Dict:
    """
    Main runner with correct weight-based turnover and cost.
    """
    df = df.copy()
    df = add_forward_returns(df, spec.hold_days)

    picks = build_picks(df, spec)

    # returns series
    port_gross = build_portfolio_returns_from_cohorts(df, picks, spec.hold_days)

    # weights + turnover
    w_port = build_overlapping_weights(picks, spec.hold_days)
    turnover = compute_turnover_from_weights(w_port)

    # Align indices
    common_idx = port_gross.index.intersection(turnover.index)
    port_gross = port_gross.loc[common_idx]
    turnover = turnover.loc[common_idx]
    w_port = w_port.loc[common_idx]

    # costs
    cost_oneway = spec.cost_bps_oneway / 10000.0
    # daily cost = cost_oneway * sum|Δw| = 2*cost_oneway*turnover
    daily_cost = (2.0 * cost_oneway * turnover).rename("daily_cost")

    port_net = (port_gross - daily_cost).rename("port_net")

    # summary stats
    ann_ret_g, ann_vol_g, sh_g = ann_stats(port_gross)
    ann_ret_n, ann_vol_n, sh_n = ann_stats(port_net)

    avg_breadth = float((w_port > 0).sum(axis=1).mean()) if not w_port.empty else 0.0
    avg_turnover = float(turnover.mean()) if len(turnover) else np.nan

    out = {
        **asdict(spec),
        "days": int(port_gross.shape[0]),
        "avg_breadth": avg_breadth,
        "turnover": avg_turnover,
        "sharpe_gross": float(sh_g),
        "ann_ret_gross": float(ann_ret_g),
        "ann_vol_gross": float(ann_vol_g),
        "win_rate_gross": float((port_gross > 0).mean()) if len(port_gross) else np.nan,
        "sharpe_net": float(sh_n),
        "ann_ret_net": float(ann_ret_n),
        "ann_vol_net": float(ann_vol_n),
        "win_rate_net": float((port_net > 0).mean()) if len(port_net) else np.nan,
    }

    # Keep picks + weights for optional downstream visualization
    return {
        "summary": out,
        "port_gross": port_gross,
        "port_net": port_net,
        "turnover": turnover,
        "weights": w_port,
        "picks": picks,
    }


# =========================
# Walk-forward (select bottom_n on train, test forward)
# =========================
def walk_forward_select_bottom_n(
    df: pd.DataFrame,
    rank_mode: str,
    hold_days: int,
    cost_bps_oneway: float,
    bottom_n_grid: List[int],
    zscore_filter: Optional[float] = None,
    rsi_filter: Optional[float] = None,
) -> pd.DataFrame:
    years = year_windows(df)
    rows = []

    for i in range(0, len(years) - (TRAIN_YEARS + TEST_YEARS) + 1):
        train_years = years[i : i + TRAIN_YEARS]
        test_years = years[i + TRAIN_YEARS : i + TRAIN_YEARS + TEST_YEARS]

        train_df = df[df[DATE_COL].dt.year.isin(train_years)].copy()
        test_df = df[df[DATE_COL].dt.year.isin(test_years)].copy()

        best_n = None
        best_sh = -1e9

        for n in bottom_n_grid:
            spec = StrategySpec(
                name="WF_train",
                rank_mode=rank_mode,
                bottom_n=n,
                hold_days=hold_days,
                cost_bps_oneway=cost_bps_oneway,
                zscore_filter=zscore_filter,
                rsi_filter=rsi_filter,
            )
            sh = run_strategy(train_df, spec)["summary"]["sharpe_net"]
            if np.isfinite(sh) and sh > best_sh:
                best_sh = sh
                best_n = n

        spec_test = StrategySpec(
            name="WF_test",
            rank_mode=rank_mode,
            bottom_n=int(best_n),
            hold_days=hold_days,
            cost_bps_oneway=cost_bps_oneway,
            zscore_filter=zscore_filter,
            rsi_filter=rsi_filter,
        )
        test_run = run_strategy(test_df, spec_test)
        s = test_run["summary"]

        rows.append({
            "train": f"{train_years[0]}-{train_years[-1]}",
            "test": f"{test_years[0]}-{test_years[-1]}",
            "rank_mode": rank_mode,
            "zscore_filter": zscore_filter,
            "rsi_filter": rsi_filter,
            "chosen_bottom_n": int(best_n),
            "train_sharpe_net": float(best_sh),
            "test_sharpe_net": float(s["sharpe_net"]),
            "test_ann_ret_net": float(s["ann_ret_net"]),
            "test_ann_vol_net": float(s["ann_vol_net"]),
            "test_days": int(s["days"]),
            "test_turnover": float(s["turnover"]) if np.isfinite(s["turnover"]) else np.nan,
        })

    return pd.DataFrame(rows)


# =========================
# Experiment matrix
# =========================
def build_specs() -> List[StrategySpec]:
    specs: List[StrategySpec] = []

    # A) Baselines
    for n in BOTTOM_N_GRID:
        specs.append(StrategySpec("z_rank", "zscore", n, HOLD_DAYS, COST_BPS_ONEWAY))
        specs.append(StrategySpec("rsi_rank", "rsi", n, HOLD_DAYS, COST_BPS_ONEWAY))
        specs.append(StrategySpec("combo_rank (rank_z + rank_rsi)", "combo", n, HOLD_DAYS, COST_BPS_ONEWAY))

    # B) z-rank + z thresholds (hybrid)
    for thr in ZSCORE_THRESHOLDS:
        if thr is None:
            continue
        for n in BOTTOM_N_GRID:
            specs.append(StrategySpec(f"z_rank AND z<= {thr}", "zscore", n, HOLD_DAYS, COST_BPS_ONEWAY, zscore_filter=thr))

    # C) rsi-rank + rsi thresholds (hybrid)
    for thr in RSI_THRESHOLDS:
        if thr is None:
            continue
        for n in BOTTOM_N_GRID:
            specs.append(StrategySpec(f"rsi_rank AND rsi<= {thr}", "rsi", n, HOLD_DAYS, COST_BPS_ONEWAY, rsi_filter=thr))

    # D) Cross filters
    for thr in RSI_THRESHOLDS:
        if thr is None:
            continue
        for n in BOTTOM_N_GRID:
            specs.append(StrategySpec(f"z_rank AND rsi<= {thr}", "zscore", n, HOLD_DAYS, COST_BPS_ONEWAY, rsi_filter=thr))

    for thr in ZSCORE_THRESHOLDS:
        if thr is None:
            continue
        for n in BOTTOM_N_GRID:
            specs.append(StrategySpec(f"rsi_rank AND z<= {thr}", "rsi", n, HOLD_DAYS, COST_BPS_ONEWAY, zscore_filter=thr))

    # E) combo + thresholds (optional)
    for zthr in [-1.0, -1.5, -2.0]:
        for n in BOTTOM_N_GRID:
            specs.append(StrategySpec(f"combo_rank AND z<= {zthr}", "combo", n, HOLD_DAYS, COST_BPS_ONEWAY, zscore_filter=zthr))
    for rthr in [20.0, 15.0, 10.0]:
        for n in BOTTOM_N_GRID:
            specs.append(StrategySpec(f"combo_rank AND rsi<= {rthr}", "combo", n, HOLD_DAYS, COST_BPS_ONEWAY, rsi_filter=rthr))

    return specs


def main():
    print("\n====================")
    print("Cross-Sectional MR Experiment Matrix (weight-based turnover)")
    print("====================\n")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find {DATA_PATH.resolve()}")

    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, SYMBOL_COL]).reset_index(drop=True)

    for col in [ZSCORE_COL, RSI_COL, CLOSE_COL]:
        if col not in df.columns:
            raise ValueError(f"Missing column in featured_data.csv: {col}")

    bench = equal_weight_benchmark_1d(df)

    specs = build_specs()
    summaries = []

    for k, spec in enumerate(specs, start=1):
        res = run_strategy(df, spec)
        s = res["summary"]

        beta, alpha_ann, r2 = beta_alpha(res["port_net"], bench)
        s["beta_net_vs_eqw"] = beta
        s["alpha_ann_net_vs_eqw"] = alpha_ann
        s["r2_net_vs_eqw"] = r2

        summaries.append(s)

        # Save per-strategy outputs (returns + turnover + weights + picks)
        tag = f"{spec.rank_mode}_n{spec.bottom_n}_h{spec.hold_days}_z{spec.zscore_filter}_r{spec.rsi_filter}"
        tag = tag.replace(" ", "").replace("None", "NA").replace("<=", "le")
        res["port_net"].to_frame().to_csv(OUT_DIR / f"port_net_{tag}.csv")
        res["turnover"].to_frame().to_csv(OUT_DIR / f"turnover_{tag}.csv")
        res["weights"].to_csv(OUT_DIR / f"weights_{tag}.csv")
        res["picks"].to_csv(OUT_DIR / f"picks_{tag}.csv", index=False)

        if k % 10 == 0:
            print(f"Ran {k}/{len(specs)}...")

    out = pd.DataFrame(summaries).sort_values(["sharpe_net", "ann_ret_net"], ascending=False)
    out.to_csv(OUT_RESULTS, index=False)

    print("\nSaved results:")
    print(f"  {OUT_RESULTS}")
    print(f"  Per-strategy files in: {OUT_DIR}/")

    print("\nTop 15 by Sharpe (net):")
    cols_show = [
        "name", "rank_mode", "bottom_n", "hold_days",
        "zscore_filter", "rsi_filter",
        "avg_breadth", "turnover",
        "sharpe_net", "ann_ret_net", "ann_vol_net",
        "beta_net_vs_eqw", "alpha_ann_net_vs_eqw", "r2_net_vs_eqw",
    ]
    print(out[cols_show].head(15).to_string(index=False))

    # Walk-forward sanity on a few families
    print("\n====================")
    print("Walk-forward OOS (examples)")
    print("====================\n")

    wf_cases = [
        ("zscore", None, None),
        ("zscore", None, 15.0),
        ("rsi", None, None),
        ("rsi", -1.0, None),     # rsi rank + z gate
        ("combo", None, None),
    ]

    for rank_mode, zf, rf in wf_cases:
        wf = walk_forward_select_bottom_n(
            df=df,
            rank_mode=rank_mode,
            hold_days=HOLD_DAYS,
            cost_bps_oneway=COST_BPS_ONEWAY,
            bottom_n_grid=BOTTOM_N_GRID,
            zscore_filter=zf,
            rsi_filter=rf,
        )
        tag = f"wf_{rank_mode}_z{zf}_r{rf}".replace("None", "NA")
        wf.to_csv(OUT_DIR / f"{tag}.csv", index=False)
        print(f"\nCase: rank_mode={rank_mode}, z_filter={zf}, rsi_filter={rf}")
        print(wf.to_string(index=False))


if __name__ == "__main__":
    main()