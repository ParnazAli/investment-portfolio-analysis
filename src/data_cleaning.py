"""
data_cleaning.py
------------------
Ingestion and cleaning pipeline for the Investment Portfolio Analysis
project.

This module loads raw historical daily OHLCV price data (exported from
WSJ Market Data) for each asset in the investment universe, validates
and standardizes it, aligns all assets onto a common trading calendar,
and derives the daily return series used throughout the rest of the
analysis (EDA, risk metrics, portfolio optimization, backtesting).

Usage:
    python -m src.data_cleaning
    # or
    from src.data_cleaning import build_clean_dataset
    close_prices, simple_returns, log_returns = build_clean_dataset()

Author: [Your Name]
Course: Investment Management — Portfolio Analytics Project
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.config import (
    ASSET_INFO,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    TICKERS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REQUIRED_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


class DataQualityError(Exception):
    """Raised when a raw data file fails a critical validation check."""


def load_and_clean_single_asset(ticker: str) -> pd.DataFrame:
    """Load and clean the raw OHLCV CSV for a single ticker.

    Performs the following standardization steps:
        1. Strip whitespace from column names (WSJ exports use ' Open', etc.)
        2. Parse the Date column into a proper datetime dtype.
        3. Coerce price/volume columns to numeric, catching silent
           string-formatting issues.
        4. Sort chronologically (WSJ exports newest-first).
        5. Drop duplicate trading dates, keeping the first occurrence.
        6. Drop or flag rows with non-positive prices or missing values.

    Args:
        ticker: The asset's ticker symbol; expects a file named
            ``{ticker}.csv`` inside ``DATA_RAW_DIR``.

    Returns:
        A cleaned DataFrame with columns
        ``[Date, Open, High, Low, Close, Volume, Ticker]``, sorted by
        Date ascending.

    Raises:
        DataQualityError: If the raw file is missing required columns
            or contains no valid rows after cleaning.
    """
    filepath = DATA_RAW_DIR / f"{ticker}.csv"
    if not filepath.exists():
        raise DataQualityError(f"Raw data file not found: {filepath}")

    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]

    missing_cols = set(["Date"] + REQUIRED_PRICE_COLUMNS) - set(df.columns)
    if missing_cols:
        raise DataQualityError(f"{ticker}: missing expected columns {missing_cols}")

    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")

    for col in REQUIRED_PRICE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("Date").reset_index(drop=True)
    df = df.drop_duplicates(subset="Date", keep="first")

    price_cols = ["Open", "High", "Low", "Close"]
    invalid_price_mask = (df[price_cols] <= 0).any(axis=1)
    if invalid_price_mask.any():
        logger.warning(
            "%s: dropping %d rows with non-positive prices",
            ticker, invalid_price_mask.sum(),
        )
        df = df.loc[~invalid_price_mask].reset_index(drop=True)

    n_before = len(df)
    df = df.dropna(subset=price_cols + ["Volume"])
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.info("%s: dropped %d rows with missing values", ticker, n_dropped)

    if df.empty:
        raise DataQualityError(f"{ticker}: no valid rows remain after cleaning")

    df["Ticker"] = ticker
    return df


def _align_close_prices(cleaned: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge per-asset Close price series onto a common trading calendar.

    An inner join is used deliberately: different assets can have
    slightly different trading calendars (e.g., ETF vs. equity holiday
    conventions), and portfolio-level math (covariance, optimization)
    requires every asset to be observed on exactly the same dates.

    Args:
        cleaned: Mapping of ticker -> cleaned OHLCV DataFrame.

    Returns:
        A wide DataFrame indexed by Date, one column per ticker,
        containing only dates common to all assets.
    """
    series_list = [
        df.set_index("Date")["Close"].rename(ticker)
        for ticker, df in cleaned.items()
    ]
    close_prices = pd.concat(series_list, axis=1, join="inner").sort_index()
    return close_prices


def build_clean_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full cleaning pipeline and persist all processed outputs.

    Returns:
        A 3-tuple of:
            - close_prices: wide DataFrame of aligned daily close prices.
            - simple_returns: daily simple (percentage) returns.
            - log_returns: daily logarithmic returns.

    Side Effects:
        Writes the following files to ``DATA_PROCESSED_DIR``:
            - close_prices.csv
            - daily_simple_returns.csv
            - daily_log_returns.csv
            - ``{TICKER}_clean.csv`` for each asset
            - asset_metadata.csv
    """
    logger.info("=" * 60)
    logger.info("DATA CLEANING PIPELINE — START")
    logger.info("=" * 60)

    cleaned: Dict[str, pd.DataFrame] = {}
    for ticker in TICKERS:
        logger.info("Processing %s (%s)", ticker, ASSET_INFO[ticker]["name"])
        df = load_and_clean_single_asset(ticker)
        logger.info(
            "  -> %d rows | %s to %s",
            len(df), df["Date"].min().date(), df["Date"].max().date(),
        )
        cleaned[ticker] = df

    close_prices = _align_close_prices(cleaned)
    logger.info("-" * 60)
    logger.info(
        "Aligned panel: %d trading days x %d assets (%s to %s)",
        close_prices.shape[0], close_prices.shape[1],
        close_prices.index.min().date(), close_prices.index.max().date(),
    )

    n_missing = int(close_prices.isnull().sum().sum())
    if n_missing:
        logger.warning("%d missing values remain after alignment", n_missing)
    else:
        logger.info("No missing values after alignment.")

    simple_returns = close_prices.pct_change().dropna()
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()

    # Persist outputs
    close_prices.to_csv(DATA_PROCESSED_DIR / "close_prices.csv")
    simple_returns.to_csv(DATA_PROCESSED_DIR / "daily_simple_returns.csv")
    log_returns.to_csv(DATA_PROCESSED_DIR / "daily_log_returns.csv")

    for ticker, df in cleaned.items():
        df.to_csv(DATA_PROCESSED_DIR / f"{ticker}_clean.csv", index=False)

    meta_df = pd.DataFrame(ASSET_INFO).T
    meta_df.index.name = "Ticker"
    meta_df.to_csv(DATA_PROCESSED_DIR / "asset_metadata.csv")

    logger.info("Saved processed outputs to %s", DATA_PROCESSED_DIR)
    logger.info("=" * 60)
    logger.info("DATA CLEANING PIPELINE — COMPLETE")
    logger.info("=" * 60)

    return close_prices, simple_returns, log_returns


if __name__ == "__main__":
    build_clean_dataset()
