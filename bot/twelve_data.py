"""Twelve Data API client with rotating API keys."""

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class TwelveDataClient:
    """Twelve Data API client with automatic key rotation."""

    BASE_URL = "https://api.twelvedata.com"

    def __init__(self, api_keys: Optional[list[str]] = None):
        """Initialize with rotating API keys.

        Args:
            api_keys: List of API keys (or loaded from TWELVE_DATA_KEYS env)
        """
        if api_keys is None:
            keys_str = os.getenv("TWELVE_DATA_KEYS", "")
            api_keys = [k.strip() for k in keys_str.split(",") if k.strip()]

        if not api_keys:
            raise ValueError(
                "Twelve Data API keys required. Set TWELVE_DATA_KEYS in .env "
                "or pass keys directly."
            )

        self._api_keys = list(api_keys)
        self._key_index = 0
        self._lock = threading.Lock()
        self.key_usage = {k: 0 for k in self._api_keys}
        self._key_last_used = {k: 0.0 for k in self._api_keys}
        self._min_interval = 60.0 / 8  # 7.5s between calls per key (8 req/min limit)
        self.cache_dir = Path(__file__).parent / "cache" / "twelvedata"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_key(self) -> str:
        """Get next available API key, respecting per-key rate limits."""
        with self._lock:
            now = time.time()
            # Find a key that has cooled down
            for _ in range(len(self._api_keys)):
                key = self._api_keys[self._key_index % len(self._api_keys)]
                self._key_index += 1
                elapsed = now - self._key_last_used[key]
                if elapsed >= self._min_interval:
                    self._key_last_used[key] = now
                    return key
            # All keys are rate-limited; wait for the soonest one
            soonest_key = min(self._key_last_used, key=self._key_last_used.get)
            wait = self._min_interval - (now - self._key_last_used[soonest_key])
            if wait > 0:
                time.sleep(wait)
            self._key_last_used[soonest_key] = time.time()
            return soonest_key

    def _request(self, endpoint: str, params: dict) -> dict:
        """Make API request with key rotation on rate limit."""
        # Copy params to avoid mutating the caller's dict
        params = dict(params)
        max_retries = 3  # Try all keys up to 3 times

        for attempt in range(max_retries * len(self._api_keys)):
            key = self._get_key()
            params["apikey"] = key

            try:
                response = requests.get(
                    f"{self.BASE_URL}/{endpoint}",
                    params=params,
                    timeout=10,
                )
                data = response.json()

                # Check for rate limit
                if "code" in data and data["code"] == 429:
                    retry_after = float(data.get("retry_after", 1.0))
                    # Exponential backoff: wait longer on each retry round
                    backoff = min(retry_after * (1 + attempt // len(self._api_keys)), 15.0)
                    time.sleep(backoff)
                    continue

                self.key_usage[key] += 1
                return data

            except requests.exceptions.Timeout:
                logger.warning(f"Twelve Data request timed out for endpoint={endpoint}")
                continue
            except requests.exceptions.ConnectionError:
                logger.warning(f"Twelve Data connection error for endpoint={endpoint}")
                continue
            except Exception as e:
                logger.warning(f"Twelve Data request failed for endpoint={endpoint}: {type(e).__name__}")
                continue

        raise Exception(f"All Twelve Data API keys exhausted or rate limited for endpoint={endpoint}")

    def get_quote(self, symbol: str) -> dict:
        """Get real-time quote for a symbol."""
        cache_file = self.cache_dir / f"quote_{symbol}.json"

        # Check cache (valid for 1 minute)
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    cached = json.load(f)
                cache_time = datetime.fromisoformat(cached["timestamp"])
                if datetime.now() - cache_time < timedelta(minutes=1):
                    return cached["data"]
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        data = self._request("quote", {"symbol": symbol})

        if "symbol" in data:
            result = {
                "symbol": data["symbol"],
                "name": data.get("name", symbol),
                "exchange": data.get("exchange", ""),
                "close": float(data.get("close", 0)),
                "open": float(data.get("open", 0)),
                "high": float(data.get("high", 0)),
                "low": float(data.get("low", 0)),
                "volume": int(data.get("volume", 0)),
                "percent_change": float(data.get("percent_change", 0)),
                "previous_close": float(data.get("previous_close", 0)),
            }

            # Atomic cache write
            try:
                cache_file_tmp = cache_file.with_suffix(".tmp")
                with open(cache_file_tmp, "w") as f:
                    json.dump({"timestamp": datetime.now().isoformat(), "data": result}, f)
                os.replace(cache_file_tmp, cache_file)
            except Exception as e:
                logger.warning(f"Failed to cache quote for {symbol}: {e}")

            return result

        error_msg = data.get("message", "unknown error")
        raise Exception(f"Failed to get quote for {symbol}: {error_msg}")

    def get_time_series(
        self,
        symbol: str,
        interval: str = "1day",
        outputsize: int = 30,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> list[dict]:
        """Get time series data for a symbol."""
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
        }

        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        data = self._request("time_series", params)

        if "values" in data:
            return [
                {
                    "datetime": bar["datetime"],
                    "open": float(bar["open"]),
                    "high": float(bar["high"]),
                    "low": float(bar["low"]),
                    "close": float(bar["close"]),
                    "volume": int(bar.get("volume", 0)),
                }
                for bar in data["values"]
            ]

        error_msg = data.get("message", "unknown error")
        raise Exception(f"Failed to get time series for {symbol}: {error_msg}")

    def get_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        quote = self.get_quote(symbol)
        return quote["close"]

    def get_ema(self, symbol: str, period: int = 20, interval: str = "1day") -> Optional[float]:
        """Get EMA (Exponential Moving Average). Returns None on failure."""
        try:
            data = self._request("ema", {
                "symbol": symbol,
                "interval": interval,
                "time_period": period,
            })
            if "values" in data and data["values"]:
                return float(data["values"][0].get("ema", 0))
        except Exception as e:
            logger.warning(f"Failed to get EMA for {symbol}: {e}")
        return None

    def get_rsi(self, symbol: str, period: int = 14, interval: str = "1day") -> Optional[float]:
        """Get RSI (Relative Strength Index). Returns None on failure."""
        try:
            data = self._request("rsi", {
                "symbol": symbol,
                "interval": interval,
                "time_period": period,
            })
            if "values" in data and data["values"]:
                return float(data["values"][0].get("rsi", 0))
        except Exception as e:
            logger.warning(f"Failed to get RSI for {symbol}: {e}")
        return None

    def get_macd(self, symbol: str, interval: str = "1day") -> Optional[dict]:
        """Get MACD indicator. Returns None on failure."""
        try:
            data = self._request("macd", {
                "symbol": symbol,
                "interval": interval,
            })
            if "values" in data and data["values"]:
                val = data["values"][0]
                return {
                    "macd": float(val.get("macd", 0)),
                    "signal": float(val.get("signal", 0)),
                    "histogram": float(val.get("histogram", 0)),
                }
        except Exception as e:
            logger.warning(f"Failed to get MACD for {symbol}: {e}")
        return None

    def get_bollinger_bands(self, symbol: str, period: int = 20, interval: str = "1day") -> Optional[dict]:
        """Get Bollinger Bands. Returns None on failure."""
        try:
            data = self._request("bbands", {
                "symbol": symbol,
                "interval": interval,
                "time_period": period,
            })
            if "values" in data and data["values"]:
                val = data["values"][0]
                return {
                    "upper": float(val.get("upper_band", 0)),
                    "middle": float(val.get("middle_band", 0)),
                    "lower": float(val.get("lower_band", 0)),
                }
        except Exception as e:
            logger.warning(f"Failed to get Bollinger Bands for {symbol}: {e}")
        return None

    def get_stock_list(self, exchange: str = "NASDAQ") -> list[dict]:
        """Get list of stocks on an exchange."""
        data = self._request("stocks", {"exchange": exchange})

        if "data" in data:
            return [
                {
                    "symbol": s["symbol"],
                    "name": s.get("name", ""),
                    "exchange": s.get("exchange", ""),
                    "type": s.get("type", ""),
                }
                for s in data["data"]
            ]

        return []

    def get_market_status(self) -> dict:
        """Get market status."""
        data = self._request("market_status", {})
        return data

    def clear_cache(self):
        """Clear the API key cache."""
        for f in self.cache_dir.glob("*.json"):
            f.unlink()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")

    client = TwelveDataClient()

    # Test quote
    quote = client.get_quote("AAPL")
    print(f"AAPL: ${quote['close']:.2f}")

    # Test time series
    bars = client.get_time_series("AAPL", outputsize=5)
    print(f"\nLast 5 bars:")
    for bar in bars:
        print(f"  {bar['datetime']}: O={bar['open']:.2f} H={bar['high']:.2f} L={bar['low']:.2f} C={bar['close']:.2f}")

    # Test indicators
    rsi = client.get_rsi("AAPL")
    print(f"\nRSI: {rsi:.2f}" if rsi is not None else "\nRSI: N/A")

    macd = client.get_macd("AAPL")
    if macd:
        print(f"MACD: {macd['macd']:.4f}, Signal: {macd['signal']:.4f}")
    else:
        print("MACD: N/A")
