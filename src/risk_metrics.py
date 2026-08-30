"""
risk_metrics.py
----------------
Risk and return analytics module for the Investment Portfolio Analysis
project.

Computes the standard suite of metrics used in investment management to
evaluate individual assets (and, later, portfolios):

    Return:            Annualized (CAGR-style) return
    Risk:              Annualized volatility, Maximum Drawdown
    Risk-adjusted:     Sharpe Ratio, Sortino Ratio, Calmar Ratio
    Tail risk:         Historical & Parametric VaR, CVaR (Expected Shortfall)
    Systematic risk:   Beta and Jensen's Alpha vs. the benchmark

All metrics are computed from daily simple returns and annualized using
252 trading days/year, consistent with standard industry convention.

Usage:
    python -m src.risk_metrics

Author: [Your Name]
Course: Investment Management — Portfolio Analytics Project
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.config import (
    BENCHMARK_TICKER,
    DATA_PROCESSED_DIR,
    FIGURE_DPI,
    FIGURES_DIR,
    RISK_FREE_RATE_ANNUAL,
    TRADING_DAYS_PER_YEAR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DAILY_RF = (1 + RISK_FREE_RATE_ANNUAL) ** (1 / TRADING_DAYS_PER_YEAR) - 1


def annualized_return(returns: pd.Series) -> float:
    """Compute the annualized (geometric) return from daily simple returns.

    Args:
        returns: Series of daily simple returns.

    Returns:
        Annualized compounded return, expressed as a decimal (e.g. 0.15 = 15%).
    """
    cumulative = (1 + returns).prod()
    n_years = len(returns) / TRADING_DAYS_PER_YEAR
    return cumulative ** (1 / n_years) - 1


def annualized_volatility(returns: pd.Series) -> float:
    """Compute annualized volatility (standard deviation) of daily returns."""
    return returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown(returns: pd.Series) -> float:
    """Compute the maximum peak-to-trough drawdown over the sample period.

    Args:
        returns: Series of daily simple returns.

    Returns:
        Maximum drawdown as a negative decimal (e.g., -0.35 = -35%).
    """
    wealth_index = (1 + returns).cumprod()
    running_max = wealth_index.cummax()
    drawdown = (wealth_index - running_max) / running_max
    return drawdown.min()


def sharpe_ratio(returns: pd.Series, risk_free_annual: float = RISK_FREE_RATE_ANNUAL) -> float:
    """Compute the annualized Sharpe Ratio.

    Sharpe = (annualized excess return) / (annualized volatility)

    Args:
        returns: Series of daily simple returns.
        risk_free_annual: Annualized risk-free rate assumption.

    Returns:
        Annualized Sharpe Ratio.
    """
    excess_daily = returns - DAILY_RF
    ann_excess_return = excess_daily.mean() * TRADING_DAYS_PER_YEAR
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    return ann_excess_return / ann_vol if ann_vol > 0 else np.nan


def sortino_ratio(returns: pd.Series, risk_free_annual: float = RISK_FREE_RATE_ANNUAL) -> float:
    """Compute the annualized Sortino Ratio (penalizes only downside volatility).

    Args:
        returns: Series of daily simple returns.
        risk_free_annual: Annualized risk-free rate assumption.

    Returns:
        Annualized Sortino Ratio.
    """
    excess_daily = returns - DAILY_RF
    downside = excess_daily[excess_daily < 0]
    downside_std = downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    ann_excess_return = excess_daily.mean() * TRADING_DAYS_PER_YEAR
    return ann_excess_return / downside_std if downside_std > 0 else np.nan


def calmar_ratio(returns: pd.Series) -> float:
    """Compute the Calmar Ratio: annualized return / |max drawdown|.

    A ratio combining long-run growth with worst-case capital loss;
    widely used to evaluate strategies where drawdown magnitude matters
    as much as average return (e.g., for investors with liquidity needs).
    """
    mdd = max_drawdown(returns)
    ann_ret = annualized_return(returns)
    return ann_ret / abs(mdd) if mdd != 0 else np.nan


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Compute historical (non-parametric) Value at Risk.

    Args:
        returns: Series of daily simple returns.
        confidence: Confidence level (e.g., 0.95 for 95% VaR).

    Returns:
        Daily VaR as a negative decimal — the loss threshold not expected
        to be exceeded on (1 - confidence) of days, based on the empirical
        distribution of historical returns.
    """
    return np.percentile(returns, (1 - confidence) * 100)


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Compute parametric (variance-covariance) VaR assuming normality.

    Args:
        returns: Series of daily simple returns.
        confidence: Confidence level (e.g., 0.95 for 95% VaR).

    Returns:
        Daily VaR as a negative decimal, assuming returns are normally
        distributed with the sample's mean and standard deviation.
    """
    mu, sigma = returns.mean(), returns.std()
    z = stats.norm.ppf(1 - confidence)
    return mu + z * sigma


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Compute historical Conditional VaR (Expected Shortfall).

    CVaR answers: "given that we are in the worst (1 - confidence) tail
    of outcomes, what is the average loss?" It is a coherent risk measure
    (unlike VaR) and is increasingly preferred by regulators (e.g. Basel)
    for tail-risk assessment.

    Args:
        returns: Series of daily simple returns.
        confidence: Confidence level (e.g., 0.95 for 95%).

    Returns:
        Expected shortfall as a negative decimal.
    """
    var_threshold = historical_var(returns, confidence)
    tail_losses = returns[returns <= var_threshold]
    return tail_losses.mean() if len(tail_losses) > 0 else np.nan


def beta_alpha(asset_returns: pd.Series, benchmark_returns: pd.Series) -> tuple[float, float]:
    """Compute CAPM Beta and annualized Jensen's Alpha vs. a benchmark.

    Beta is estimated via OLS regression of asset excess returns on
    benchmark excess returns: (R_a - Rf) = alpha + beta * (R_b - Rf) + e.

    Args:
        asset_returns: Daily simple returns of the asset.
        benchmark_returns: Daily simple returns of the benchmark (aligned index).

    Returns:
        A tuple (beta, annualized_alpha).
    """
    asset_excess = asset_returns - DAILY_RF
    bench_excess = benchmark_returns - DAILY_RF

    slope, intercept, _, _, _ = stats.linregress(bench_excess, asset_excess)
    beta = slope
    alpha_annualized = intercept * TRADING_DAYS_PER_YEAR
    return beta, alpha_annualized


def compute_all_metrics(simple_returns: pd.DataFrame, benchmark: str = BENCHMARK_TICKER) -> pd.DataFrame:
    """Compute the full risk/return metric suite for every asset.

    Args:
        simple_returns: Wide DataFrame of daily simple returns, one column
            per ticker (must include the benchmark column).
        benchmark: Ticker to use as the market proxy for Beta/Alpha.

    Returns:
        A DataFrame indexed by ticker with one column per metric.
    """
    bench_returns = simple_returns[benchmark]
    rows = {}

    for ticker in simple_returns.columns:
        r = simple_returns[ticker]
        beta, alpha = beta_alpha(r, bench_returns)

        rows[ticker] = {
            "Annualized Return (%)": annualized_return(r) * 100,
            "Annualized Volatility (%)": annualized_volatility(r) * 100,
            "Sharpe Ratio": sharpe_ratio(r),
            "Sortino Ratio": sortino_ratio(r),
            "Calmar Ratio": calmar_ratio(r),
            "Max Drawdown (%)": max_drawdown(r) * 100,
            "Historical VaR 95% (Daily, %)": historical_var(r, 0.95) * 100,
            "Parametric VaR 95% (Daily, %)": parametric_var(r, 0.95) * 100,
            "CVaR 95% (Daily, %)": conditional_var(r, 0.95) * 100,
            "Beta (vs. " + benchmark + ")": beta,
            "Jensen's Alpha (Annualized, %)": alpha * 100,
        }

    return pd.DataFrame(rows).T


def plot_risk_return_scatter(metrics: pd.DataFrame) -> None:
    """Plot annualized return vs. volatility, colored by Sharpe Ratio.

    This is the canonical "risk-return map" used throughout investment
    management to visually compare assets on a single chart: assets in
    the upper-left (high return, low risk) are most attractive.

    Args:
        metrics: Output of ``compute_all_metrics`` (index = ticker).
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        metrics["Annualized Volatility (%)"],
        metrics["Annualized Return (%)"],
        c=metrics["Sharpe Ratio"],
        cmap="RdYlGn",
        s=220,
        edgecolor="black",
        linewidth=0.8,
    )

    for ticker, row in metrics.iterrows():
        ax.annotate(
            ticker,
            (row["Annualized Volatility (%)"], row["Annualized Return (%)"]),
            xytext=(6, 6), textcoords="offset points", fontsize=10, fontweight="bold",
        )

    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Annualized Volatility (%)  —  Risk")
    ax.set_ylabel("Annualized Return (%)")
    ax.set_title("Risk-Return Profile by Asset (color = Sharpe Ratio)")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Sharpe Ratio")
    fig.tight_layout()

    outpath = FIGURES_DIR / "06_risk_return_scatter.png"
    fig.savefig(outpath, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved %s", outpath.name)


def plot_drawdown_chart(simple_returns: pd.DataFrame) -> None:
    """Plot the drawdown time series for each asset (underwater equity curve).

    Args:
        simple_returns: Wide DataFrame of daily simple returns.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    for ticker in simple_returns.columns:
        wealth = (1 + simple_returns[ticker]).cumprod()
        running_max = wealth.cummax()
        dd = (wealth - running_max) / running_max * 100
        linewidth = 2.5 if ticker == BENCHMARK_TICKER else 1.2
        ax.plot(dd.index, dd, label=ticker, linewidth=linewidth)

    ax.set_title("Drawdown from Peak (\"Underwater\" Chart)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(ncol=3, loc="lower left", frameon=True)
    fig.tight_layout()

    outpath = FIGURES_DIR / "07_drawdown_chart.png"
    fig.savefig(outpath, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved %s", outpath.name)


def run_risk_analysis() -> pd.DataFrame:
    """Load processed returns, compute all metrics, and persist results."""
    logger.info("=" * 60)
    logger.info("STEP 4: RISK & RETURN METRICS — START")
    logger.info("=" * 60)

    returns_path = DATA_PROCESSED_DIR / "daily_simple_returns.csv"
    if not returns_path.exists():
        raise FileNotFoundError("Run `python -m src.data_cleaning` first.")

    simple_returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    logger.info("Loaded returns for %d assets over %d trading days",
                simple_returns.shape[1], simple_returns.shape[0])

    metrics = compute_all_metrics(simple_returns)
    metrics = metrics.round(3)

    outpath = DATA_PROCESSED_DIR / "risk_return_metrics.csv"
    metrics.to_csv(outpath)
    logger.info("Saved metrics table -> %s", outpath.name)

    plot_risk_return_scatter(metrics)
    plot_drawdown_chart(simple_returns)

    logger.info("-" * 60)
    best_sharpe = metrics["Sharpe Ratio"].idxmax()
    worst_dd = metrics["Max Drawdown (%)"].idxmin()
    logger.info("Highest Sharpe Ratio: %s (%.2f)", best_sharpe, metrics.loc[best_sharpe, "Sharpe Ratio"])
    logger.info("Worst Max Drawdown:   %s (%.1f%%)", worst_dd, metrics.loc[worst_dd, "Max Drawdown (%)"])
    logger.info("=" * 60)
    logger.info("STEP 4: RISK & RETURN METRICS — COMPLETE")
    logger.info("=" * 60)

    return metrics


if __name__ == "__main__":
    result = run_risk_analysis()
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print(result)
