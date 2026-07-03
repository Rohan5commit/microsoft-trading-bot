"""Stock universe management - top US stocks by market cap using Twelve Data."""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

try:
    from twelve_data import TwelveDataClient
except ImportError:
    TwelveDataClient = None

logger = logging.getLogger(__name__)


# Default top 100 US stocks by market cap (updated periodically)
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK.B", "LLY", "AVGO", "TSLA",
    "WMT", "JPM", "V", "UNH", "XOM", "MA", "COST", "HD", "PG", "JNJ",
    "ABBV", "NFLX", "CRM", "MRK", "BAC", "AMD", "ORCL", "CVX", "TMO", "ADBE",
    "KO", "WFC", "ACN", "MCD", "CSCO", "ABT", "LIN", "DHR", "NEE", "TXN",
    "PM", "UNP", "UPS", "RTX", "AMGN", "HON", "LOW", "INTC", "IBM", "QCOM",
    "SPGI", "CAT", "BA", "GE", "BLK", "AXP", "SYK", "MDT", "GILD", "ADI",
    "VRTX", "MMC", "CB", "PLD", "ISRG", "SCHW", "ZTS", "CI", "SO", "CME",
    "DUK", "CL", "REGN", "BSX", "BDX", "ICE", "WM", "PNC", "TGT", "USB",
    "ADP", "NSC", "FIS", "FDX", "FCX", "GD", "PFE", "TFC", "EMR", "AON",
    "SHW", "SLB", "OXY", "MPC", "PSX", "VLO", "EOG", "KMI", "WMB", "OKE",
]


def get_universe(
    config: dict,
    universe_file: Optional[str] = None,
    force_refresh: bool = False,
) -> list[str]:
    """Get the stock universe, refreshing if needed.

    Args:
        config: Bot configuration dict
        universe_file: Path to cached universe file
        force_refresh: Force refresh even if cache is valid

    Returns:
        List of ticker symbols
    """
    if universe_file is None:
        universe_file = config.get("trading", {}).get("universe_file", "universe.json")

    cache_path = Path(__file__).parent / universe_file
    max_stocks = config.get("universe", {}).get("max_stocks", 100)
    refresh_days = config.get("universe", {}).get("refresh_days", 30)
    exclude = set(config.get("universe", {}).get("exclude", []))

    # Check if cache is valid
    if not force_refresh and cache_path.exists():
        try:
            with open(cache_path) as f:
                cache = json.load(f)
            cached_date = datetime.fromisoformat(cache["updated"])
            if datetime.now() - cached_date < timedelta(days=refresh_days):
                tickers = [t for t in cache["tickers"] if t not in exclude]
                return tickers[:max_stocks]
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    # Fetch fresh universe using Twelve Data
    logger.info("Fetching fresh stock universe via Twelve Data...")
    try:
        tickers = _fetch_from_twelve_data(max_stocks, exclude)
    except Exception as e:
        logger.warning(f"Twelve Data fetch failed: {e}, using default universe")
        tickers = _filter_default_universe(max_stocks, exclude)

    # Cache the result
    cache = {
        "updated": datetime.now().isoformat(),
        "tickers": tickers,
    }
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)

    return tickers


def _fetch_from_twelve_data(
    max_stocks: int = 100,
    exclude: set[str] = None,
) -> list[str]:
    """Fetch top stocks from Twelve Data."""
    if exclude is None:
        exclude = set()

    client = TwelveDataClient()

    # Filter by default universe (known large caps)
    valid_tickers = []
    for ticker in DEFAULT_UNIVERSE:
        if ticker in exclude:
            continue

        try:
            # Verify the ticker exists in Twelve Data
            quote = client.get_quote(ticker)
            if quote and quote.get("close", 0) > 0:
                valid_tickers.append(ticker)
        except Exception:
            continue

        if len(valid_tickers) >= max_stocks:
            break

    return valid_tickers


def _filter_default_universe(
    max_stocks: int = 100,
    exclude: set[str] = None,
) -> list[str]:
    """Filter default universe (fallback)."""
    if exclude is None:
        exclude = set()

    return [t for t in DEFAULT_UNIVERSE if t not in exclude][:max_stocks]


def get_universe_with_info(config: dict) -> list[dict]:
    """Get universe with additional info (name, price, change).

    Returns:
        List of dicts with keys: ticker, name, price, change
    """
    tickers = get_universe(config)
    result = []

    client = TwelveDataClient()

    for ticker in tickers:
        try:
            quote = client.get_quote(ticker)
            result.append({
                "ticker": ticker,
                "name": quote.get("name", ticker),
                "price": quote.get("close", 0),
                "change_pct": quote.get("percent_change", 0),
                "volume": quote.get("volume", 0),
            })
        except Exception:
            result.append({
                "ticker": ticker,
                "name": ticker,
                "price": 0,
                "change_pct": 0,
                "volume": 0,
            })

    return result


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")

    config = {
        "universe": {
            "max_stocks": 20,
            "min_market_cap_billion": 50,
            "exclude": ["SPY", "QQQ"],
            "refresh_days": 30,
        }
    }
    tickers = get_universe(config, force_refresh=True)
    print(f"Universe: {len(tickers)} stocks")
    for t in tickers:
        print(f"  {t}")
