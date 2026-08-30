"""
backtesting.py
---------------
Backtesting module for the Investment Portfolio Analysis project.

Simulates how the two optimal portfolios identified in Step 5 (Maximum
Sharpe and Global Minimum Variance) would have actually performed over
the historical sample, under two portfolio-management approaches:

    1. Buy & Hold      — weights set once at t=0 and left to drift as
                         asset prices diverge (no trading thereafter).
    2. Periodic Rebalancing — weights are reset back to their target
                         allocation at fixed intervals (quarterly by
                         default), which is standard practice for real
                         portfolio management and controls risk drift.

Each strategy's simulated equity curve is compared against a Buy & Hold
position in the benchmark (SPY) using the same risk/return metric suite
from Step 4.

Usage:
    python -m src.backtesting

Author: [Your Name]
Course: Investment Management — Portfolio Analytics Project
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    BENCHMARK_TICKER,
    DATA_PROCESSED_DIR,
    FIGURE_DPI,
    FIGURES_DIR,
)
from src.risk_metrics import (
    annualized_return,
    annualized_volatility,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def backtest_portfolio(
    target_weights: dict[str, float],
    simple_returns: pd.DataFrame,
    rebalance_freq: str | None = None,
) -> pd.Series:
    """Simulate a portfolio's daily value path given target weights.

    Args:
        target_weights: Mapping of ticker -> target portfolio weight
            (must sum to ~1).
        simple_returns: Wide DataFrame of daily simple returns for the
            assets in ``target_weights`` (columns must match its keys).
        rebalance_freq: If None, weights are set once at t=0 and drift
            freely (Buy & Hold). If a pandas offset alias (e.g. "QE" for
            quarter-end, "ME" for month-end) is given, weights are reset
            to target at the start of each new period.

    Returns:
        A Series of portfolio value over time, indexed by date, starting
        at 1.0 (i.e., normalized to a $1 initial investment).
    """
    tickers = list(target_weights.keys())
    returns = simple_returns[tickers].copy()
    target = np.array([target_weights[t] for t in tickers])

    portfolio_value = [1.0]
    values_per_asset = target.copy()  # dollar value allocated to each asset

    rebalance_dates = set()
    if rebalance_freq is not None:
        period_starts = returns.index.to_series().dt.to_period(
            "Q" if rebalance_freq.upper().startswith("Q") else "M"
        )
        rebalance_dates = set(
            returns.index[period_starts.ne(period_starts.shift(1))]
        )

    for date, row in returns.iterrows():
        if date in rebalance_dates and date != returns.index[0]:
            total = values_per_asset.sum()
            values_per_asset = target * total  # reset to target weights
        values_per_asset = values_per_asset * (1 + row.values)
        total_value = values_per_asset.sum()
        portfolio_value.append(total_value)

    portfolio_value = portfolio_value[1:]  # drop the initial seed value
    return pd.Series(portfolio_value, index=returns.index, name="Portfolio Value")


def compute_strategy_metrics(value_series: pd.Series) -> dict:
    """Compute the standard risk/return metric suite from a value series.

    Args:
        value_series: Portfolio (or benchmark) value over time.

    Returns:
        Dict of summary metrics (annualized return, volatility, Sharpe,
        Sortino, max drawdown, final cumulative return).
    """
    rets = value_series.pct_change().dropna()
    return {
        "Annualized Return (%)": annualized_return(rets) * 100,
        "Annualized Volatility (%)": annualized_volatility(rets) * 100,
        "Sharpe Ratio": sharpe_ratio(rets),
        "Sortino Ratio": sortino_ratio(rets),
        "Max Drawdown (%)": max_drawdown(rets) * 100,
        "Total Return (%)": (value_series.iloc[-1] / value_series.iloc[0] - 1) * 100,
    }


def plot_backtest_comparison(strategies: dict[str, pd.Series]) -> None:
    """Plot cumulative growth of $1 for all backtested strategies vs. benchmark.

    Args:
        strategies: Mapping of strategy label -> value Series (all
            sharing the same date index).
    """
    fig, ax = plt.subplots(figsize=(12, 6.5))
    styles = {
        BENCHMARK_TICKER + " (Benchmark, Buy & Hold)": dict(color="black", linewidth=2.5, linestyle="--"),
    }

    for label, series in strategies.items():
        style = styles.get(label, {})
        ax.plot(series.index, series.values, label=label, linewidth=style.get("linewidth", 1.8),
                color=style.get("color"), linestyle=style.get("linestyle", "-"))

    ax.set_title("Backtest: Growth of $1 — Optimal Portfolios vs. Benchmark")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="upper left", frameon=True)
    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")
    fig.tight_layout()

    outpath = FIGURES_DIR / "10_backtest_comparison.png"
    fig.savefig(outpath, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved %s", outpath.name)


def plot_backtest_drawdowns(strategies: dict[str, pd.Series]) -> None:
    """Plot the underwater (drawdown) curves for all backtested strategies.

    Args:
        strategies: Mapping of strategy label -> value Series.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    for label, series in strategies.items():
        running_max = series.cummax()
        dd = (series - running_max) / running_max * 100
        linewidth = 2.5 if "Benchmark" in label else 1.5
        ax.plot(series.index, dd, label=label, linewidth=linewidth)

    ax.set_title("Backtest Drawdowns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()

    outpath = FIGURES_DIR / "11_backtest_drawdowns.png"
    fig.savefig(outpath, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved %s", outpath.name)


def run_backtest() -> pd.DataFrame:
    """Execute the full backtesting pipeline and persist results.

    Returns:
        A summary DataFrame of performance metrics for every strategy
        (rows) tested, including the benchmark.
    """
    logger.info("=" * 60)
    logger.info("STEP 6: BACKTESTING — START")
    logger.info("=" * 60)

    simple_returns = pd.read_csv(
        DATA_PROCESSED_DIR / "daily_simple_returns.csv", index_col=0, parse_dates=True
    )
    weights_df = pd.read_csv(DATA_PROCESSED_DIR / "optimal_portfolio_weights.csv", index_col=0)

    max_sharpe_weights = weights_df["Max Sharpe Weight"].to_dict()
    min_var_weights = weights_df["Min Variance Weight"].to_dict()

    logger.info("Simulating Max Sharpe portfolio (Buy & Hold)...")
    ms_bh = backtest_portfolio(max_sharpe_weights, simple_returns, rebalance_freq=None)

    logger.info("Simulating Max Sharpe portfolio (Quarterly Rebalanced)...")
    ms_qr = backtest_portfolio(max_sharpe_weights, simple_returns, rebalance_freq="Q")

    logger.info("Simulating Min Variance portfolio (Buy & Hold)...")
    mv_bh = backtest_portfolio(min_var_weights, simple_returns, rebalance_freq=None)

    logger.info("Simulating Min Variance portfolio (Quarterly Rebalanced)...")
    mv_qr = backtest_portfolio(min_var_weights, simple_returns, rebalance_freq="Q")

    benchmark_value = (1 + simple_returns[BENCHMARK_TICKER]).cumprod()

    strategies = {
        "Max Sharpe — Buy & Hold": ms_bh,
        "Max Sharpe — Quarterly Rebalanced": ms_qr,
        "Min Variance — Buy & Hold": mv_bh,
        "Min Variance — Quarterly Rebalanced": mv_qr,
        f"{BENCHMARK_TICKER} (Benchmark, Buy & Hold)": benchmark_value,
    }

    plot_backtest_comparison(strategies)
    plot_backtest_drawdowns(strategies)

    summary = pd.DataFrame({label: compute_strategy_metrics(s) for label, s in strategies.items()}).T
    summary = summary.round(3)
    summary.to_csv(DATA_PROCESSED_DIR / "backtest_summary.csv")

    logger.info("-" * 60)
    logger.info("Backtest Summary:\n%s", summary.to_string())
    logger.info("Saved backtest_summary.csv")
    logger.info("=" * 60)
    logger.info("STEP 6: BACKTESTING — COMPLETE")
    logger.info("=" * 60)

    return summary


if __name__ == "__main__":
    result = run_backtest()
