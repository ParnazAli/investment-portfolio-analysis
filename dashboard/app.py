"""
app.py
------
Interactive Streamlit dashboard for the Investment Portfolio Analysis
project.

Presents the full analytics pipeline (EDA, risk metrics, portfolio
optimization, backtesting) as an explorable web app, reading exclusively
from the pre-computed CSV outputs in ``data/processed/`` so the dashboard
loads instantly without recomputation.

Run locally with:
    streamlit run dashboard/app.py

Author: [Your Name]
Course: Investment Management — Portfolio Analytics Project
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Allow importing from src/ when running via `streamlit run dashboard/app.py`
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    ASSET_INFO,
    DATA_PROCESSED_DIR,
    RISK_FREE_RATE_ANNUAL,
    TRADING_DAYS_PER_YEAR,
)

st.set_page_config(
    page_title="Investment Portfolio Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------
# Data loading (cached so the app stays fast on interaction)
# ----------------------------------------------------------------------
@st.cache_data
def load_data() -> dict[str, pd.DataFrame]:
    """Load all processed CSV outputs produced by the src/ pipeline.

    Returns:
        Dict mapping a short name to its corresponding DataFrame.
    """
    d = DATA_PROCESSED_DIR
    return {
        "close_prices": pd.read_csv(d / "close_prices.csv", index_col=0, parse_dates=True),
        "simple_returns": pd.read_csv(d / "daily_simple_returns.csv", index_col=0, parse_dates=True),
        "correlation": pd.read_csv(d / "correlation_matrix.csv", index_col=0),
        "risk_metrics": pd.read_csv(d / "risk_return_metrics.csv", index_col=0),
        "optimal_weights": pd.read_csv(d / "optimal_portfolio_weights.csv", index_col=0),
        "optimal_summary": pd.read_csv(d / "optimal_portfolio_summary.csv", index_col=0),
        "backtest_summary": pd.read_csv(d / "backtest_summary.csv", index_col=0),
    }


try:
    data = load_data()
except FileNotFoundError as e:
    st.error(
        f"Processed data not found ({e}). "
        "Run the pipeline first: `python -m src.data_cleaning`, then "
        "`src.eda`, `src.risk_metrics`, `src.portfolio_optimization`, `src.backtesting`."
    )
    st.stop()


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
st.sidebar.title("📈 Portfolio Analytics")
st.sidebar.markdown(
    "**Investment Management — Portfolio Analysis Project**\n\n"
    "9-asset universe (8 holdings + SPY benchmark), 5-year daily history."
)

all_tickers = list(data["close_prices"].columns)
selected_tickers = st.sidebar.multiselect(
    "Assets to display", options=all_tickers, default=all_tickers
)

date_min, date_max = data["close_prices"].index.min(), data["close_prices"].index.max()
date_range = st.sidebar.date_input(
    "Date range", value=(date_min, date_max), min_value=date_min, max_value=date_max
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Risk-free rate assumption: {RISK_FREE_RATE_ANNUAL:.1%} (annualized)")
st.sidebar.caption(f"Trading days/year: {TRADING_DAYS_PER_YEAR}")

if len(date_range) == 2:
    mask = (data["close_prices"].index >= pd.Timestamp(date_range[0])) & (
        data["close_prices"].index <= pd.Timestamp(date_range[1])
    )
else:
    mask = slice(None)


# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab_overview, tab_prices, tab_risk, tab_opt, tab_backtest = st.tabs(
    ["🏠 Overview", "💹 Price & Returns", "⚠️ Risk Analysis", "🎯 Portfolio Optimization", "🔁 Backtest"]
)

# ---------------- Overview ----------------
with tab_overview:
    st.header("Investment Universe Overview")
    meta = pd.DataFrame(ASSET_INFO).T
    meta.index.name = "Ticker"
    st.dataframe(meta, use_container_width=True)

    st.subheader("Sample Period")
    col1, col2, col3 = st.columns(3)
    col1.metric("Start Date", str(date_min.date()))
    col2.metric("End Date", str(date_max.date()))
    col3.metric("Trading Days", f"{len(data['close_prices']):,}")

    st.subheader("Quick Performance Snapshot")
    snapshot = data["risk_metrics"][["Annualized Return (%)", "Annualized Volatility (%)", "Sharpe Ratio"]]
    st.dataframe(snapshot.style.background_gradient(cmap="RdYlGn", subset=["Sharpe Ratio"]),
                 use_container_width=True)

# ---------------- Price & Returns ----------------
with tab_prices:
    st.header("Price Trends & Cumulative Returns")

    close = data["close_prices"].loc[mask, selected_tickers]
    normalized = close / close.iloc[0] * 100

    fig1 = px.line(normalized, title="Growth of $100 Invested (Normalized)")
    fig1.update_layout(yaxis_title="Value of $100", xaxis_title="Date", legend_title="Asset")
    st.plotly_chart(fig1, use_container_width=True)

    returns = data["simple_returns"].loc[mask, selected_tickers] if isinstance(mask, pd.Series) else data["simple_returns"][selected_tickers]
    cumulative = (1 + returns).cumprod() - 1
    fig2 = px.line(cumulative * 100, title="Cumulative Return (%)")
    fig2.update_layout(yaxis_title="Cumulative Return (%)", xaxis_title="Date")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Daily Return Distribution")
    selected_for_hist = st.selectbox("Select an asset", options=selected_tickers)
    fig3 = px.histogram(returns[selected_for_hist] * 100, nbins=80,
                         title=f"{selected_for_hist} — Daily Return Distribution (%)")
    fig3.update_layout(showlegend=False, xaxis_title="Daily Return (%)")
    st.plotly_chart(fig3, use_container_width=True)

# ---------------- Risk Analysis ----------------
with tab_risk:
    st.header("Risk & Return Metrics")

    st.dataframe(data["risk_metrics"].style.format("{:.2f}"), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Correlation Matrix")
        fig4 = px.imshow(
            data["correlation"], text_auto=".2f", color_continuous_scale="RdYlGn",
            zmin=-1, zmax=1, aspect="auto",
        )
        st.plotly_chart(fig4, use_container_width=True)

    with col2:
        st.subheader("Risk-Return Map")
        rm = data["risk_metrics"]
        fig5 = px.scatter(
            rm, x="Annualized Volatility (%)", y="Annualized Return (%)",
            color="Sharpe Ratio", text=rm.index, color_continuous_scale="RdYlGn",
            size=[20] * len(rm),
        )
        fig5.update_traces(textposition="top center")
        st.plotly_chart(fig5, use_container_width=True)

# ---------------- Portfolio Optimization ----------------
with tab_opt:
    st.header("Markowitz Portfolio Optimization")

    st.subheader("Optimal Portfolio Performance")
    st.dataframe(data["optimal_summary"], use_container_width=True)

    st.subheader("Optimal Allocations")
    weights = data["optimal_weights"]
    weights_long = weights.reset_index().melt(id_vars="index", var_name="Strategy", value_name="Weight")
    weights_long.columns = ["Ticker", "Strategy", "Weight"]
    weights_long = weights_long[weights_long["Weight"] > 0.001]

    fig6 = px.bar(
        weights_long, x="Ticker", y="Weight", color="Strategy", barmode="group",
        title="Max Sharpe vs. Minimum Variance Portfolio Weights",
    )
    fig6.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig6, use_container_width=True)

    st.image(str(Path(__file__).resolve().parent.parent / "figures" / "08_efficient_frontier.png"),
              caption="Efficient Frontier with Monte Carlo Simulation", use_container_width=True)

# ---------------- Backtest ----------------
with tab_backtest:
    st.header("Backtest Results")

    st.dataframe(data["backtest_summary"], use_container_width=True)

    best_strategy = data["backtest_summary"]["Sharpe Ratio"].idxmax()
    st.success(f"🏆 Best risk-adjusted performer: **{best_strategy}** "
               f"(Sharpe Ratio: {data['backtest_summary'].loc[best_strategy, 'Sharpe Ratio']:.2f})")

    st.image(str(Path(__file__).resolve().parent.parent / "figures" / "10_backtest_comparison.png"),
              caption="Growth of $1 — Strategies vs. Benchmark", use_container_width=True)
    st.image(str(Path(__file__).resolve().parent.parent / "figures" / "11_backtest_drawdowns.png"),
              caption="Drawdown Comparison", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Built with Python, pandas, SciPy, Plotly & Streamlit")
