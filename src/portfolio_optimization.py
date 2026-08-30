"""
portfolio_optimization.py
--------------------------
Modern Portfolio Theory (Markowitz) optimization module.

Given the daily return history of the investable universe (all assets
except the benchmark), this module:

    1. Estimates expected returns and the covariance matrix.
    2. Solves for the Global Minimum Variance (GMV) portfolio.
    3. Solves for the Maximum Sharpe Ratio ("tangency") portfolio.
    4. Traces the full Efficient Frontier via constrained optimization
       (minimize variance for a grid of target returns).
    5. Runs a Monte Carlo simulation of random long-only portfolios to
       visually confirm the frontier is indeed the efficient boundary.
    6. Plots the Capital Market Line (CML) from the risk-free rate
       through the tangency portfolio.

Long-only, fully-invested portfolios are assumed throughout
(weights >= 0, sum(weights) == 1), consistent with a typical retail /
academic portfolio construction exercise.

Usage:
    python -m src.portfolio_optimization

Author: [Your Name]
Course: Investment Management — Portfolio Analytics Project
"""

from __future__ import annotations

import logging
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.config import (
    DATA_PROCESSED_DIR,
    FIGURE_DPI,
    FIGURES_DIR,
    PORTFOLIO_TICKERS,
    RANDOM_SEED,
    RISK_FREE_RATE_ANNUAL,
    TRADING_DAYS_PER_YEAR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

np.random.seed(RANDOM_SEED)


def portfolio_performance(
    weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame
) -> tuple[float, float, float]:
    """Compute annualized return, volatility, and Sharpe Ratio for a portfolio.

    Args:
        weights: Array of portfolio weights (must sum to 1).
        mean_returns: Annualized expected returns per asset.
        cov_matrix: Annualized covariance matrix of asset returns.

    Returns:
        Tuple of (annualized_return, annualized_volatility, sharpe_ratio).
    """
    ret = float(np.dot(weights, mean_returns))
    vol = float(np.sqrt(weights.T @ cov_matrix @ weights))
    sharpe = (ret - RISK_FREE_RATE_ANNUAL) / vol if vol > 0 else np.nan
    return ret, vol, sharpe


def _negative_sharpe(weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame) -> float:
    """Objective function for maximizing Sharpe Ratio (minimize its negative)."""
    _, _, sharpe = portfolio_performance(weights, mean_returns, cov_matrix)
    return -sharpe


def _portfolio_volatility(weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame) -> float:
    """Objective function for minimizing portfolio volatility."""
    _, vol, _ = portfolio_performance(weights, mean_returns, cov_matrix)
    return vol


def _optimize(
    objective: Callable, mean_returns: pd.Series, cov_matrix: pd.DataFrame,
    extra_constraints: list | None = None,
) -> np.ndarray:
    """Run a bounded, constrained SLSQP optimization over portfolio weights.

    Args:
        objective: Function of (weights, mean_returns, cov_matrix) to minimize.
        mean_returns: Annualized expected returns per asset.
        cov_matrix: Annualized covariance matrix.
        extra_constraints: Additional scipy constraint dicts (e.g., a
            target-return constraint for frontier tracing).

    Returns:
        Optimal weights array.
    """
    n = len(mean_returns)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    if extra_constraints:
        constraints.extend(extra_constraints)

    bounds = tuple((0.0, 1.0) for _ in range(n))
    init_guess = np.repeat(1 / n, n)

    result = minimize(
        objective, init_guess, args=(mean_returns, cov_matrix),
        method="SLSQP", bounds=bounds, constraints=constraints,
    )
    if not result.success:
        logger.warning("Optimization did not converge: %s", result.message)
    return result.x


def max_sharpe_portfolio(mean_returns: pd.Series, cov_matrix: pd.DataFrame) -> np.ndarray:
    """Solve for the portfolio that maximizes the Sharpe Ratio (tangency portfolio)."""
    return _optimize(_negative_sharpe, mean_returns, cov_matrix)


def min_variance_portfolio(mean_returns: pd.Series, cov_matrix: pd.DataFrame) -> np.ndarray:
    """Solve for the Global Minimum Variance (GMV) portfolio."""
    return _optimize(_portfolio_volatility, mean_returns, cov_matrix)


def efficient_frontier(
    mean_returns: pd.Series, cov_matrix: pd.DataFrame, n_points: int = 60
) -> pd.DataFrame:
    """Trace the efficient frontier by minimizing volatility for a grid of target returns.

    For each target return between the GMV return and the max single-asset
    return, solves: minimize portfolio volatility subject to
    sum(weights) == 1 and portfolio_return == target.

    Args:
        mean_returns: Annualized expected returns per asset.
        cov_matrix: Annualized covariance matrix.
        n_points: Number of points to trace along the frontier.

    Returns:
        DataFrame with columns [Return, Volatility] describing the frontier,
        sorted by ascending return.
    """
    gmv_weights = min_variance_portfolio(mean_returns, cov_matrix)
    gmv_return, _, _ = portfolio_performance(gmv_weights, mean_returns, cov_matrix)

    target_returns = np.linspace(gmv_return, mean_returns.max(), n_points)
    frontier_vols = []

    for target in target_returns:
        return_constraint = {
            "type": "eq",
            "fun": lambda w, target=target: np.dot(w, mean_returns) - target,
        }
        weights = _optimize(_portfolio_volatility, mean_returns, cov_matrix, [return_constraint])
        _, vol, _ = portfolio_performance(weights, mean_returns, cov_matrix)
        frontier_vols.append(vol)

    return pd.DataFrame({"Return": target_returns, "Volatility": frontier_vols})


def monte_carlo_simulation(
    mean_returns: pd.Series, cov_matrix: pd.DataFrame, n_portfolios: int = 15_000
) -> pd.DataFrame:
    """Simulate random long-only portfolios to visualize the feasible set.

    Args:
        mean_returns: Annualized expected returns per asset.
        cov_matrix: Annualized covariance matrix.
        n_portfolios: Number of random portfolios to generate.

    Returns:
        DataFrame with columns [Return, Volatility, Sharpe] plus one
        weight column per asset.
    """
    n_assets = len(mean_returns)
    records = np.zeros((n_portfolios, 3 + n_assets))

    for i in range(n_portfolios):
        weights = np.random.dirichlet(np.ones(n_assets))
        ret, vol, sharpe = portfolio_performance(weights, mean_returns, cov_matrix)
        records[i, 0] = ret
        records[i, 1] = vol
        records[i, 2] = sharpe
        records[i, 3:] = weights

    columns = ["Return", "Volatility", "Sharpe"] + list(mean_returns.index)
    return pd.DataFrame(records, columns=columns)


def plot_efficient_frontier(
    frontier: pd.DataFrame,
    simulations: pd.DataFrame,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    max_sharpe_w: np.ndarray,
    min_var_w: np.ndarray,
) -> None:
    """Plot the Monte Carlo cloud, efficient frontier, key portfolios, and CML.

    Args:
        frontier: Output of ``efficient_frontier``.
        simulations: Output of ``monte_carlo_simulation``.
        mean_returns: Annualized expected returns per asset.
        cov_matrix: Annualized covariance matrix.
        max_sharpe_w: Weights of the max-Sharpe (tangency) portfolio.
        min_var_w: Weights of the GMV portfolio.
    """
    fig, ax = plt.subplots(figsize=(11, 7))

    scatter = ax.scatter(
        simulations["Volatility"] * 100, simulations["Return"] * 100,
        c=simulations["Sharpe"], cmap="viridis", s=6, alpha=0.35,
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Sharpe Ratio")

    ax.plot(frontier["Volatility"] * 100, frontier["Return"] * 100,
            color="black", linewidth=2.5, label="Efficient Frontier")

    ms_ret, ms_vol, _ = portfolio_performance(max_sharpe_w, mean_returns, cov_matrix)
    mv_ret, mv_vol, _ = portfolio_performance(min_var_w, mean_returns, cov_matrix)

    ax.scatter(ms_vol * 100, ms_ret * 100, marker="*", color="red", s=500,
               edgecolor="black", linewidth=1, zorder=5, label="Max Sharpe Portfolio")
    ax.scatter(mv_vol * 100, mv_ret * 100, marker="D", color="gold", s=200,
               edgecolor="black", linewidth=1, zorder=5, label="Min Variance Portfolio")

    # Capital Market Line: from risk-free rate through the tangency portfolio
    cml_x = np.linspace(0, simulations["Volatility"].max() * 100, 50)
    cml_slope = (ms_ret - RISK_FREE_RATE_ANNUAL) / ms_vol
    cml_y = (RISK_FREE_RATE_ANNUAL + cml_slope * cml_x / 100) * 100
    ax.plot(cml_x, cml_y, color="crimson", linestyle="--", linewidth=1.5, label="Capital Market Line")

    for ticker in mean_returns.index:
        ax.annotate(ticker, (np.sqrt(cov_matrix.loc[ticker, ticker]) * 100, mean_returns[ticker] * 100),
                    fontsize=9, fontweight="bold", xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Annualized Volatility (%)")
    ax.set_ylabel("Annualized Return (%)")
    ax.set_title(f"Efficient Frontier — Monte Carlo Simulation ({len(simulations):,} portfolios)")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()

    outpath = FIGURES_DIR / "08_efficient_frontier.png"
    fig.savefig(outpath, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved %s", outpath.name)


def plot_optimal_weights(max_sharpe_w: np.ndarray, min_var_w: np.ndarray, tickers: list[str]) -> None:
    """Plot a side-by-side bar chart comparing GMV and Max Sharpe portfolio weights.

    Args:
        max_sharpe_w: Weights of the max-Sharpe portfolio.
        min_var_w: Weights of the GMV portfolio.
        tickers: Asset tickers corresponding to the weight arrays.
    """
    df = pd.DataFrame({"Max Sharpe": max_sharpe_w, "Min Variance": min_var_w}, index=tickers)
    df = df[df.sum(axis=1) > 0.001]  # drop negligible allocations for readability

    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(kind="bar", ax=ax, color=["#2E86AB", "#F6C453"], edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Portfolio Weight")
    ax.set_title("Optimal Portfolio Allocations")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(title="")
    plt.xticks(rotation=0)
    fig.tight_layout()

    outpath = FIGURES_DIR / "09_optimal_weights.png"
    fig.savefig(outpath, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved %s", outpath.name)


def run_portfolio_optimization() -> dict:
    """Execute the full portfolio optimization pipeline end-to-end.

    Returns:
        A dict summarizing the GMV and Max Sharpe portfolios (weights and
        performance), for downstream use in backtesting.
    """
    logger.info("=" * 60)
    logger.info("STEP 5: PORTFOLIO OPTIMIZATION — START")
    logger.info("=" * 60)

    log_returns = pd.read_csv(
        DATA_PROCESSED_DIR / "daily_log_returns.csv", index_col=0, parse_dates=True
    )
    log_returns = log_returns[PORTFOLIO_TICKERS]  # exclude benchmark from optimization universe
    logger.info("Optimizing over %d assets: %s", len(PORTFOLIO_TICKERS), PORTFOLIO_TICKERS)

    mean_returns = log_returns.mean() * TRADING_DAYS_PER_YEAR
    cov_matrix = log_returns.cov() * TRADING_DAYS_PER_YEAR

    logger.info("Solving for Maximum Sharpe Ratio portfolio...")
    max_sharpe_w = max_sharpe_portfolio(mean_returns, cov_matrix)
    ms_ret, ms_vol, ms_sharpe = portfolio_performance(max_sharpe_w, mean_returns, cov_matrix)
    logger.info("  -> Return: %.2f%% | Vol: %.2f%% | Sharpe: %.2f", ms_ret * 100, ms_vol * 100, ms_sharpe)

    logger.info("Solving for Global Minimum Variance portfolio...")
    min_var_w = min_variance_portfolio(mean_returns, cov_matrix)
    mv_ret, mv_vol, mv_sharpe = portfolio_performance(min_var_w, mean_returns, cov_matrix)
    logger.info("  -> Return: %.2f%% | Vol: %.2f%% | Sharpe: %.2f", mv_ret * 100, mv_vol * 100, mv_sharpe)

    logger.info("Tracing efficient frontier...")
    frontier = efficient_frontier(mean_returns, cov_matrix)

    logger.info("Running Monte Carlo simulation (15,000 random portfolios)...")
    simulations = monte_carlo_simulation(mean_returns, cov_matrix)

    plot_efficient_frontier(frontier, simulations, mean_returns, cov_matrix, max_sharpe_w, min_var_w)
    plot_optimal_weights(max_sharpe_w, min_var_w, PORTFOLIO_TICKERS)

    # Persist weight tables
    weights_df = pd.DataFrame(
        {"Max Sharpe Weight": max_sharpe_w, "Min Variance Weight": min_var_w},
        index=PORTFOLIO_TICKERS,
    ).round(4)
    weights_df.to_csv(DATA_PROCESSED_DIR / "optimal_portfolio_weights.csv")

    summary = pd.DataFrame({
        "Max Sharpe Portfolio": [ms_ret * 100, ms_vol * 100, ms_sharpe],
        "Min Variance Portfolio": [mv_ret * 100, mv_vol * 100, mv_sharpe],
    }, index=["Annualized Return (%)", "Annualized Volatility (%)", "Sharpe Ratio"]).round(3)
    summary.to_csv(DATA_PROCESSED_DIR / "optimal_portfolio_summary.csv")

    logger.info("Saved optimal_portfolio_weights.csv and optimal_portfolio_summary.csv")
    logger.info("=" * 60)
    logger.info("STEP 5: PORTFOLIO OPTIMIZATION — COMPLETE")
    logger.info("=" * 60)

    return {
        "max_sharpe_weights": dict(zip(PORTFOLIO_TICKERS, max_sharpe_w)),
        "min_var_weights": dict(zip(PORTFOLIO_TICKERS, min_var_w)),
        "max_sharpe_performance": (ms_ret, ms_vol, ms_sharpe),
        "min_var_performance": (mv_ret, mv_vol, mv_sharpe),
    }


if __name__ == "__main__":
    results = run_portfolio_optimization()
    print("\nMax Sharpe Portfolio Weights:")
    for ticker, w in sorted(results["max_sharpe_weights"].items(), key=lambda x: -x[1]):
        if w > 0.001:
            print(f"  {ticker}: {w:.1%}")
    print("\nMin Variance Portfolio Weights:")
    for ticker, w in sorted(results["min_var_weights"].items(), key=lambda x: -x[1]):
        if w > 0.001:
            print(f"  {ticker}: {w:.1%}")
