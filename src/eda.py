"""
eda.py
------
Exploratory Data Analysis module for the Investment Portfolio Analysis
project.

Generates the core visual diagnostics used to understand asset behavior
before any risk metrics or optimization are computed:

    1. Normalized price trends (growth of $100 invested in each asset)
    2. Cumulative returns vs. the benchmark
    3. Cross-asset correlation matrix (heatmap)
    4. Daily return distributions (histogram + KDE)
    5. 30-day rolling annualized volatility

All figures are saved as high-resolution PNGs to ``FIGURES_DIR`` so they
can be embedded directly in the analysis notebook and the GitHub README.

Usage:
    python -m src.eda

Author: [Your Name]
Course: Investment Management — Portfolio Analytics Project
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import (
    BENCHMARK_TICKER,
    DATA_PROCESSED_DIR,
    FIGURE_DPI,
    FIGURES_DIR,
    TRADING_DAYS_PER_YEAR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = FIGURE_DPI
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.titlesize"] = 13


def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the cleaned close-price panel and daily simple returns.

    Returns:
        A tuple of (close_prices, simple_returns), both indexed by Date.

    Raises:
        FileNotFoundError: If Step 2 (data_cleaning) has not been run yet.
    """
    price_path = DATA_PROCESSED_DIR / "close_prices.csv"
    returns_path = DATA_PROCESSED_DIR / "daily_simple_returns.csv"

    if not price_path.exists() or not returns_path.exists():
        raise FileNotFoundError(
            "Processed data not found. Run `python -m src.data_cleaning` first."
        )

    close_prices = pd.read_csv(price_path, index_col=0, parse_dates=True)
    simple_returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    return close_prices, simple_returns


def plot_normalized_price_trends(close_prices: pd.DataFrame) -> None:
    """Plot growth of a $100 investment in each asset over the sample period.

    Normalizing to a common base (100) makes assets with very different
    price levels (e.g., TLT ~$80 vs. SPY ~$770) visually comparable.

    Args:
        close_prices: Wide DataFrame of daily close prices, one column
            per ticker.
    """
    normalized = close_prices / close_prices.iloc[0] * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    for col in normalized.columns:
        linewidth = 2.5 if col == BENCHMARK_TICKER else 1.4
        linestyle = "--" if col == BENCHMARK_TICKER else "-"
        ax.plot(normalized.index, normalized[col], label=col,
                linewidth=linewidth, linestyle=linestyle)

    ax.set_title("Growth of $100 Invested — Aug 2021 to Aug 2026")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value of $100 Investment")
    ax.legend(ncol=3, loc="upper left", frameon=True)
    ax.axhline(100, color="grey", linewidth=0.8, linestyle=":")
    fig.tight_layout()

    outpath = FIGURES_DIR / "01_normalized_price_trends.png"
    fig.savefig(outpath, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved %s", outpath.name)


def plot_cumulative_returns(simple_returns: pd.DataFrame) -> None:
    """Plot cumulative compounded returns for each asset vs. the benchmark.

    Args:
        simple_returns: Wide DataFrame of daily simple returns.
    """
    cumulative = (1 + simple_returns).cumprod() - 1

    fig, ax = plt.subplots(figsize=(12, 6))
    for col in cumulative.columns:
        if col == BENCHMARK_TICKER:
            continue
        ax.plot(cumulative.index, cumulative[col] * 100, alpha=0.85, linewidth=1.3, label=col)

    ax.plot(cumulative.index, cumulative[BENCHMARK_TICKER] * 100,
            color="black", linewidth=2.5, linestyle="--", label=f"{BENCHMARK_TICKER} (Benchmark)")

    ax.set_title("Cumulative Returns vs. Benchmark")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend(ncol=3, loc="upper left", frameon=True)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    fig.tight_layout()

    outpath = FIGURES_DIR / "02_cumulative_returns.png"
    fig.savefig(outpath, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved %s", outpath.name)


def plot_correlation_heatmap(simple_returns: pd.DataFrame) -> pd.DataFrame:
    """Plot and save a heatmap of pairwise return correlations.

    This matrix is the direct input to the covariance matrix used later
    in Markowitz portfolio optimization — low or negative correlations
    (e.g., equities vs. TLT/GLD) are what create diversification benefit.

    Args:
        simple_returns: Wide DataFrame of daily simple returns.

    Returns:
        The correlation matrix as a DataFrame (also saved to CSV).
    """
    corr = simple_returns.corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
        vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax,
        cbar_kws={"label": "Correlation Coefficient"},
    )
    ax.set_title("Daily Return Correlation Matrix")
    fig.tight_layout()

    outpath = FIGURES_DIR / "03_correlation_heatmap.png"
    fig.savefig(outpath, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved %s", outpath.name)

    corr.to_csv(DATA_PROCESSED_DIR / "correlation_matrix.csv")
    return corr


def plot_return_distributions(simple_returns: pd.DataFrame) -> None:
    """Plot histogram + KDE of daily returns for each asset in a grid.

    Visually flags fat tails / skewness that summary statistics alone
    can hide, motivating the use of VaR/CVaR (not just standard
    deviation) in the risk-metrics stage.

    Args:
        simple_returns: Wide DataFrame of daily simple returns.
    """
    tickers = simple_returns.columns.tolist()
    n = len(tickers)
    ncols = 3
    nrows = -(-n // ncols)  # ceiling division

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows))
    axes = axes.flatten()

    for i, ticker in enumerate(tickers):
        sns.histplot(simple_returns[ticker] * 100, bins=60, kde=True,
                     ax=axes[i], color=sns.color_palette("deep")[i % 10])
        axes[i].set_title(ticker)
        axes[i].set_xlabel("Daily Return (%)")
        axes[i].axvline(0, color="black", linewidth=0.8, linestyle=":")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Distribution of Daily Returns by Asset", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()

    outpath = FIGURES_DIR / "04_return_distributions.png"
    fig.savefig(outpath, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", outpath.name)


def plot_rolling_volatility(simple_returns: pd.DataFrame, window: int = 30) -> None:
    """Plot rolling annualized volatility for each asset.

    Args:
        simple_returns: Wide DataFrame of daily simple returns.
        window: Rolling window size in trading days (default 30).
    """
    rolling_vol = simple_returns.rolling(window).std() * (TRADING_DAYS_PER_YEAR ** 0.5) * 100
    rolling_vol = rolling_vol.dropna()

    fig, ax = plt.subplots(figsize=(12, 6))
    for col in rolling_vol.columns:
        linewidth = 2.5 if col == BENCHMARK_TICKER else 1.2
        ax.plot(rolling_vol.index, rolling_vol[col], label=col, linewidth=linewidth)

    ax.set_title(f"{window}-Day Rolling Annualized Volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized Volatility (%)")
    ax.legend(ncol=3, loc="upper left", frameon=True)
    fig.tight_layout()

    outpath = FIGURES_DIR / "05_rolling_volatility.png"
    fig.savefig(outpath, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved %s", outpath.name)


def run_eda() -> None:
    """Execute the full EDA pipeline: load data, generate, and save all plots."""
    logger.info("=" * 60)
    logger.info("STEP 3: EXPLORATORY DATA ANALYSIS — START")
    logger.info("=" * 60)

    close_prices, simple_returns = load_processed_data()
    logger.info("Loaded %d trading days x %d assets", *close_prices.shape)

    plot_normalized_price_trends(close_prices)
    plot_cumulative_returns(simple_returns)
    corr = plot_correlation_heatmap(simple_returns)
    plot_return_distributions(simple_returns)
    plot_rolling_volatility(simple_returns)

    logger.info("-" * 60)
    logger.info("Lowest pairwise correlation: %.2f (%s)",
                 corr.where(~corr.isna()).values[corr.values != 1].min(),
                 "see correlation_matrix.csv for full pair detail")
    logger.info("All figures saved to %s", FIGURES_DIR)
    logger.info("=" * 60)
    logger.info("STEP 3: EXPLORATORY DATA ANALYSIS — COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_eda()
