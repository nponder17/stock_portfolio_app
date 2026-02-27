"""
Stock dashboard – featured_data.csv + Winner strategy trade overlays.
Fix: prevent x-axis "condensing" when trade overlays are toggled on by
locking x-axis range to the candle date window and filtering overlays
to the visible date range.

Run: streamlit run app.py
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "featured_data.csv"

# Winner strategy output files (created by build_winning_strat_trades.py)
TRADES_PATH = BASE_DIR / "cs_trade_events_hold3_rsi_zle-1_n10.csv"
PAIRS_PATH = BASE_DIR / "cs_trade_pairs_hold3_rsi_zle-1_n10.csv"

# =========================
# Chart layout
# =========================
ROW_HEIGHTS = [0.30, 0.13, 0.13, 0.10, 0.10, 0.14, 0.10]
N_ROWS = 7


# =========================
# Loaders
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    if "date" not in df.columns:
        raise ValueError("featured_data.csv must include a 'date' column.")
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_trade_events():
    if not TRADES_PATH.exists():
        return pd.DataFrame(columns=["date", "symbol", "event", "w_prev", "w_now", "dw"])
    t = pd.read_csv(TRADES_PATH)
    t["date"] = pd.to_datetime(t["date"])
    return t


@st.cache_data
def load_trade_pairs():
    if not PAIRS_PATH.exists():
        return pd.DataFrame(columns=["symbol", "trade_id", "entry_date", "exit_date", "holding_days"])
    p = pd.read_csv(PAIRS_PATH)
    p["entry_date"] = pd.to_datetime(p["entry_date"])
    p["exit_date"] = pd.to_datetime(p["exit_date"])
    return p


def _core_ready_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    For clean visuals, only plot rows where core indicators exist.
    This avoids the early-window NaN region from rolling indicators.
    """
    required = [
        "open", "high", "low", "close", "volume",
        "ma_5", "ma_10", "zscore_5", "rsi_2",
        "ret_1", "tr_rel_10", "rel_vol_20",
    ]
    present = [c for c in required if c in df.columns]
    if not present:
        return df.copy()
    return df.dropna(subset=present).copy()


# =========================
# Winner trade overlay helpers
# =========================
def add_trade_ids_to_events(events: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Assign trade_id to ENTRY/EXIT events using pairs file:
      - ENTRY event gets trade_id where date == entry_date
      - EXIT event gets trade_id where date == exit_date
    REBAL events keep trade_id NaN.
    """
    events = events.copy()
    events["trade_id"] = np.nan

    if events.empty or pairs.empty:
        return events

    # Map (symbol, date) -> trade_id for entry and exit
    entry_map = pairs.set_index(["symbol", "entry_date"])["trade_id"].to_dict()
    exit_map = pairs.set_index(["symbol", "exit_date"])["trade_id"].to_dict()

    def assign_tid(row):
        key = (row["symbol"], row["date"])
        if row["event"] == "ENTRY":
            return entry_map.get(key, np.nan)
        if row["event"] == "EXIT":
            return exit_map.get(key, np.nan)
        return np.nan

    events["trade_id"] = events.apply(assign_tid, axis=1)
    return events


def build_main_chart(
    df: pd.DataFrame,
    symbol: str,
    show_trades: bool,
    connect_trades: bool,
    trades_sym: pd.DataFrame,
    pairs_sym: pd.DataFrame,
) -> go.Figure:
    df = df.sort_values("date").reset_index(drop=True)
    plot_df = _core_ready_df(df)

    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough data to plot (rolling features may be NaN for early rows).",
            x=0.5, y=0.5, showarrow=False,
        )
        return fig

    x = plot_df["date"]

    fig = make_subplots(
        rows=N_ROWS,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=ROW_HEIGHTS,
        subplot_titles=(
            f"{symbol} – Price (MA5, MA10) + Winner trades",
            "zscore_5 (±2)",
            "rsi_2 (10/20/80/90)",
            "ret_1",
            "tr_rel_10",
            "Volume",
            "rel_vol_20",
        ),
    )

    # --- Row 1: Candlestick + MAs ---
    fig.add_trace(
        go.Candlestick(
            x=x,
            open=plot_df["open"],
            high=plot_df["high"],
            low=plot_df["low"],
            close=plot_df["close"],
            name="OHLC",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1, col=1,
    )

    if "ma_5" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=x, y=plot_df["ma_5"],
                name="MA(5)",
                line=dict(color="#ff9800", width=1.5),
                mode="lines",
            ),
            row=1, col=1,
        )
    if "ma_10" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=x, y=plot_df["ma_10"],
                name="MA(10)",
                line=dict(color="#2196f3", width=1.5),
                mode="lines",
            ),
            row=1, col=1,
        )

    # Stable offset for markers
    y_range = (plot_df["high"].max() - plot_df["low"].min())
    offset = y_range * 0.012 if y_range and np.isfinite(y_range) else 0.0

    # =========================
    # Winner trades overlay (ENTRY/EXIT)
    # =========================
    if show_trades and not trades_sym.empty:
        trades_sym = trades_sym.sort_values("date").copy()

        # IMPORTANT: Filter overlays to the visible candle window.
        # This prevents Plotly from expanding the x-axis range based on out-of-window overlay points.
        dmin = plot_df["date"].min()
        dmax = plot_df["date"].max()
        trades_sym = trades_sym[(trades_sym["date"] >= dmin) & (trades_sym["date"] <= dmax)].copy()

        if pairs_sym is not None and not pairs_sym.empty:
            pairs_sym = pairs_sym[(pairs_sym["entry_date"] >= dmin) & (pairs_sym["exit_date"] <= dmax)].copy()

        entries = trades_sym[trades_sym["event"] == "ENTRY"].copy()
        exits = trades_sym[trades_sym["event"] == "EXIT"].copy()

        # For plotting y positions, place entries slightly below low, exits slightly above high
        day_low = plot_df.set_index("date")["low"]
        day_high = plot_df.set_index("date")["high"]

        if not entries.empty:
            entries["y"] = entries["date"].map(day_low) - offset
            fig.add_trace(
                go.Scatter(
                    x=entries["date"],
                    y=entries["y"],
                    mode="markers+text",
                    text=entries["trade_id"].fillna("").astype(str),
                    textposition="bottom center",
                    marker=dict(
                        symbol="circle",
                        size=18,
                        color="rgba(38,166,154,0.15)",
                        line=dict(color="#26a69a", width=3),
                    ),
                    name="ENTRY (winner)",
                    customdata=np.column_stack([
                        entries["trade_id"].values,
                        entries.get("w_prev", np.nan).values,
                        entries.get("w_now", np.nan).values,
                        entries.get("dw", np.nan).values,
                    ]),
                    hovertemplate=(
                        "Date: %{x|%Y-%m-%d}<br>"
                        "<b>ENTRY</b><br>"
                        "trade_id: %{customdata[0]}<br>"
                        "w_prev: %{customdata[1]:.4f}<br>"
                        "w_now: %{customdata[2]:.4f}<br>"
                        "dw: %{customdata[3]:.4f}<extra></extra>"
                    ),
                ),
                row=1, col=1,
            )

        if not exits.empty:
            exits["y"] = exits["date"].map(day_high) + offset
            fig.add_trace(
                go.Scatter(
                    x=exits["date"],
                    y=exits["y"],
                    mode="markers+text",
                    text=exits["trade_id"].fillna("").astype(str),
                    textposition="top center",
                    marker=dict(
                        symbol="x",
                        size=18,
                        color="rgba(239,83,80,0.10)",
                        line=dict(color="#ef5350", width=3),
                    ),
                    name="EXIT (winner)",
                    customdata=np.column_stack([
                        exits["trade_id"].values,
                        exits.get("w_prev", np.nan).values,
                        exits.get("w_now", np.nan).values,
                        exits.get("dw", np.nan).values,
                    ]),
                    hovertemplate=(
                        "Date: %{x|%Y-%m-%d}<br>"
                        "<b>EXIT</b><br>"
                        "trade_id: %{customdata[0]}<br>"
                        "w_prev: %{customdata[1]:.4f}<br>"
                        "w_now: %{customdata[2]:.4f}<br>"
                        "dw: %{customdata[3]:.4f}<extra></extra>"
                    ),
                ),
                row=1, col=1,
            )

        # Optional: connect entry->exit with a dashed line per trade_id
        if connect_trades and pairs_sym is not None and not pairs_sym.empty:
            for _, r in pairs_sym.iterrows():
                e = r["entry_date"]
                x_ = r["exit_date"]
                y0 = float(day_low.get(e, np.nan) - offset)
                y1 = float(day_high.get(x_, np.nan) + offset)
                if not (np.isfinite(y0) and np.isfinite(y1)):
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=[e, x_],
                        y=[y0, y1],
                        mode="lines",
                        line=dict(color="rgba(0,0,0,0.35)", width=2, dash="dash"),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1, col=1,
                )

    # --- Row 2: zscore_5 with ±2 bands ---
    if "zscore_5" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=plot_df["zscore_5"],
                name="zscore_5",
                line=dict(color="#9c27b0", width=1.5),
                mode="lines",
            ),
            row=2, col=1,
        )
        fig.add_hline(y=2, line_dash="dash", line_color="rgba(239,83,80,0.7)", row=2, col=1)
        fig.add_hline(y=-2, line_dash="dash", line_color="rgba(38,166,154,0.7)", row=2, col=1)

    # --- Row 3: rsi_2 with 10/20/80/90 ---
    if "rsi_2" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=plot_df["rsi_2"],
                name="rsi_2",
                line=dict(color="#00bcd4", width=1.5),
                mode="lines",
            ),
            row=3, col=1,
        )
        fig.add_hline(y=10, line_dash="dash", line_color="rgba(38,166,154,0.8)", row=3, col=1)
        fig.add_hline(y=20, line_dash="dot", line_color="rgba(0,0,0,0.35)", row=3, col=1)
        fig.add_hline(y=80, line_dash="dot", line_color="rgba(0,0,0,0.35)", row=3, col=1)
        fig.add_hline(y=90, line_dash="dash", line_color="rgba(239,83,80,0.8)", row=3, col=1)

    # --- Row 4: ret_1 bars ---
    if "ret_1" in plot_df.columns:
        ret = plot_df["ret_1"].fillna(0.0)
        colors = ["#26a69a" if r >= 0 else "#ef5350" for r in ret]
        fig.add_trace(go.Bar(x=x, y=ret, name="ret_1", marker_color=colors, showlegend=False), row=4, col=1)
        fig.add_hline(y=0, line_dash="solid", line_color="rgba(0,0,0,0.3)", row=4, col=1)

    # --- Row 5: tr_rel_10 ---
    if "tr_rel_10" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=plot_df["tr_rel_10"],
                name="tr_rel_10",
                line=dict(color="#795548", width=1.5),
                mode="lines",
            ),
            row=5, col=1,
        )

    # --- Row 6: Volume ---
    if {"open", "close", "volume"}.issubset(plot_df.columns):
        vol_colors = ["#26a69a" if c >= o else "#ef5350" for o, c in zip(plot_df["open"], plot_df["close"])]
        fig.add_trace(go.Bar(x=x, y=plot_df["volume"], name="Volume", marker_color=vol_colors, showlegend=False), row=6, col=1)

    # --- Row 7: rel_vol_20 ---
    if "rel_vol_20" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=plot_df["rel_vol_20"],
                name="rel_vol_20",
                line=dict(color="#ff9800", width=1.5),
                mode="lines",
                showlegend=False,
            ),
            row=7, col=1,
        )
        fig.add_hline(y=1.0, line_dash="dot", line_color="rgba(0,0,0,0.35)", row=7, col=1)

    fig.update_layout(
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        height=980,
        margin=dict(t=30, b=30, l=50, r=50),
        font=dict(size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # ===== FIX: lock x-axis to candle window and force date type =====
    xmin = plot_df["date"].min()
    xmax = plot_df["date"].max()
    # apply to all shared x-axes
    for r in range(1, N_ROWS + 1):
        fig.update_xaxes(type="date", range=[xmin, xmax], row=r, col=1)

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="zscore_5", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="ret_1", row=4, col=1)
    fig.update_yaxes(title_text="tr_rel_10", row=5, col=1)
    fig.update_yaxes(title_text="Volume", row=6, col=1)
    fig.update_yaxes(title_text="rel_vol_20", row=7, col=1)

    return fig


def main():
    st.set_page_config(page_title="Stock Dashboard", page_icon="📈", layout="wide")
    st.title("Stock Dashboard")
    st.caption("Featured data • Core panels • Winner strategy trades (ENTRY/EXIT with trade_id)")

    df = load_data()
    if df.empty:
        st.error("No data. Ensure featured_data.csv exists at the expected path.")
        return

    trades = load_trade_events()
    pairs = load_trade_pairs()

    # Attach trade_id to events so we can label markers
    trades = add_trade_ids_to_events(trades, pairs)

    symbols = sorted(df["symbol"].unique().tolist())
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()

    with st.sidebar:
        st.header("Filters")
        default_ix = symbols.index("AAPL") if "AAPL" in symbols else 0
        symbol = st.selectbox("Symbol", options=symbols, index=default_ix)

        date_range = st.date_input(
            "Date range (for chart + tables)",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = min_date, max_date

        show_trades = st.toggle("Show winner strategy trades", value=True)
        connect_trades = st.toggle("Connect ENTRY→EXIT lines", value=True)

        st.divider()
        st.caption("Winner strategy:")
        st.code("Rank by lowest RSI_2 (bottom 10) then filter zscore_5 <= -1.0\nHold 3 days (overlapping)")

    subset = df[(df["symbol"] == symbol) & (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)].copy()
    if subset.empty:
        st.warning("No data for the selected symbol and date range.")
        return

    subset = subset.sort_values("date").reset_index(drop=True)

    # trades for this symbol
    trades_sym = trades[trades["symbol"] == symbol].copy()
    pairs_sym = pairs[pairs["symbol"] == symbol].copy()

    # --- Top metrics (use latest row in selected range) ---
    latest = subset.iloc[-1]
    first = subset.iloc[0]
    pct = ((latest["close"] - first["close"]) / first["close"] * 100) if float(first["close"]) else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Close", f"${latest['close']:.2f}", f"{pct:+.2f}%")
    if "open" in subset.columns:
        c2.metric("Open", f"${latest['open']:.2f}", None)
    if "high" in subset.columns:
        c3.metric("High", f"${latest['high']:.2f}", None)
    if "low" in subset.columns:
        c4.metric("Low", f"${latest['low']:.2f}", None)
    if "volume" in subset.columns:
        c5.metric("Volume", f"{latest['volume']:,.0f}", None)

    st.plotly_chart(
        build_main_chart(
            subset,
            symbol,
            show_trades=show_trades,
            connect_trades=connect_trades,
            trades_sym=trades_sym,
            pairs_sym=pairs_sym,
        ),
        use_container_width=True
    )

    # --- Trades table in selected date range ---
    if show_trades:
        st.subheader("Winner trades (selected symbol)")
        dmin = pd.to_datetime(start_date)
        dmax = pd.to_datetime(end_date)

        in_range = trades_sym[(trades_sym["date"] >= dmin) & (trades_sym["date"] <= dmax)].copy()
        if in_range.empty:
            st.info("No winner strategy trade events in this date range for the selected symbol.")
        else:
            show_cols = ["date", "event", "trade_id", "w_prev", "w_now", "dw"]
            st.dataframe(
                in_range[show_cols].sort_values("date", ascending=False).style.format(
                    {"w_prev": "{:.4f}", "w_now": "{:.4f}", "dw": "{:.4f}"}
                ),
                use_container_width=True,
                height=260,
            )

        if not pairs_sym.empty:
            in_pairs = pairs_sym[(pairs_sym["entry_date"] >= dmin) & (pairs_sym["exit_date"] <= dmax)].copy()
            if not in_pairs.empty:
                st.subheader("Trade pairs (ENTRY ↔ EXIT)")
                st.dataframe(in_pairs.sort_values("entry_date", ascending=False), use_container_width=True, height=260)

    # --- Data table (selected columns) ---
    with st.expander("Data table (selected columns)"):
        cols = [
            "date", "open", "high", "low", "close", "volume",
            "ma_5", "ma_10", "zscore_5", "rsi_2", "ret_1",
            "tr_rel_10", "rel_vol_20", "fwd_ret_1",
        ]
        cols = [c for c in cols if c in subset.columns]
        st.dataframe(
            subset[cols].style.format(
                {
                    "open": "${:.2f}", "high": "${:.2f}", "low": "${:.2f}", "close": "${:.2f}",
                    "volume": "{:,.0f}", "ma_5": "${:.2f}", "ma_10": "${:.2f}",
                    "zscore_5": "{:.2f}", "rsi_2": "{:.1f}", "ret_1": "{:.4f}", "tr_rel_10": "{:.4f}",
                    "rel_vol_20": "{:.2f}", "fwd_ret_1": "{:.2%}",
                }
            ),
            use_container_width=True,
            height=320,
        )


if __name__ == "__main__":
    main()