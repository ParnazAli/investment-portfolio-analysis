"""
config.py
---------
Centralized configuration for the Investment Portfolio Analysis project.

Keeping all paths, tickers, and shared constants in one place avoids
"magic strings" scattered across scripts and makes the pipeline easy
to reconfigure (e.g., swapping tickers, changing the risk-free rate)
without touching business logic.
"""

from pathlib import Path
from typing import Dict

# ----------------------------------------------------------------------
# Project paths
# ----------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

for _dir in (DATA_PROCESSED_DIR, FIGURES_DIR, REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Universe definition
# ----------------------------------------------------------------------
BENCHMARK_TICKER: str = "SPY"

TICKERS: list[str] = ["AAPL", "MSFT", "JPM", "JNJ", "AMZN", "XOM", "GLD", "TLT", "SPY"]

# Tickers actually eligible for portfolio optimization (benchmark excluded)
PORTFOLIO_TICKERS: list[str] = [t for t in TICKERS if t != BENCHMARK_TICKER]

ASSET_INFO: Dict[str, Dict[str, str]] = {
    "AAPL": {"name": "Apple Inc.",                        "sector": "Technology"},
    "MSFT": {"name": "Microsoft Corp.",                   "sector": "Technology"},
    "JPM":  {"name": "JPMorgan Chase & Co.",              "sector": "Financials"},
    "JNJ":  {"name": "Johnson & Johnson",                 "sector": "Healthcare"},
    "AMZN": {"name": "Amazon.com Inc.",                   "sector": "Consumer Discretionary"},
    "XOM":  {"name": "Exxon Mobil Corp.",                 "sector": "Energy"},
    "GLD":  {"name": "SPDR Gold Shares",                  "sector": "Commodities (Gold)"},
    "TLT":  {"name": "iShares 20+ Yr Treasury Bond ETF",  "sector": "Fixed Income"},
    "SPY":  {"name": "SPDR S&P 500 ETF Trust",            "sector": "Benchmark (Equity Index)"},
}

# ----------------------------------------------------------------------
# Analysis assumptions
# ----------------------------------------------------------------------
TRADING_DAYS_PER_YEAR: int = 252

# Approximate risk-free rate proxy (annualized), used for Sharpe/Sortino.
# In a real report this should be sourced from the current 3M/1Y T-Bill
# yield; documented here explicitly so it's easy to update and defend.
RISK_FREE_RATE_ANNUAL: float = 0.045

RANDOM_SEED: int = 42

# ----------------------------------------------------------------------
# Plotting style
# ----------------------------------------------------------------------
FIGURE_DPI: int = 150
COLOR_PALETTE: str = "viridis"
