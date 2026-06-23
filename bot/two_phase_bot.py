"""Two-phase trading bot: Quick scan all stocks, then deep analysis on candidates.

Phase 1: Fast scan (5 min) - Analyze all 1000 stocks with single LLM call
Phase 2: Deep analysis (1-2 hrs) - Full TradingAgents on top 20-30 candidates

This gives full quality on stocks that matter without wasting time on rejects.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

from risk_manager import RiskManager
from universe import get_universe
from twelve_data import TwelveDataClient

try:
    from alpaca_client import AlpacaClient
    ALPACA_AVAILABLE = True
except (ImportError, ValueError):
    ALPACA_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(Path(__file__).parent / "bot.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class TwoPhaseBot:
    """Two-phase bot: quick scan + deep analysis on candidates."""

    def __init__(self, config_path: Optional[str] = None, max_concurrent: int = 20):
        if config_path is None:
            config_path = Path(__file__).parent / "config.json"

        with open(config_path) as f:
            self.config = json.load(f)

        self.max_concurrent = max_concurrent
        self.twelve_data = TwelveDataClient()
        self.risk_manager = RiskManager(self.config)

        # Alpaca optional
        self.alpaca = None
        if ALPACA_AVAILABLE and os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"):
            try:
                self.alpaca = AlpacaClient(paper=True)
            except Exception:
                pass

        # TradingAgents config for deep analysis
        self.ta_config = DEFAULT_CONFIG.copy()
        self.ta_config["llm_provider"] = self.config["llm"]["provider"]
        self.ta_config["deep_think_llm"] = self.config["llm"]["deep_think_model"]
        self.ta_config["quick_think_llm"] = self.config["llm"]["quick_think_model"]
        self.ta_config["temperature"] = self.config["llm"]["temperature"]

        # Deep analysis settings
        self.deep_count = self.config.get("deep_analysis", {}).get("count", 20)
        self.min_conviction = self.config.get("deep_analysis", {}).get("min_conviction", 0.6)

        logger.info(f"Two-phase bot initialized (deep_count={self.deep_count})")

    def batch_get_prices(self, tickers: list[str]) -> dict[str, float]:
        """Fetch prices for multiple tickers concurrently."""
        prices = {}

        def fetch_one(ticker):
            try:
                return ticker, self.twelve_data.get_price(ticker)
            except Exception:
                return ticker, 0

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(fetch_one, t) for t in tickers]
            for f in futures:
                ticker, price = f.result()
                if price > 0:
                    prices[ticker] = price

        return prices

    def batch_get_indicators(self, tickers: list[str]) -> dict[str, dict]:
        """Fetch indicators for multiple tickers concurrently."""
        indicators = {}

        def fetch_one(ticker):
            try:
                rsi = self.twelve_data.get_rsi(ticker)
                macd = self.twelve_data.get_macd(ticker)
                return ticker, {"rsi": rsi, "macd": macd}
            except Exception:
                return ticker, {}

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(fetch_one, t) for t in tickers]
            for f in futures:
                ticker, ind = f.result()
                indicators[ticker] = ind

        return indicators

    def quick_scan(self, ticker: str, price: float, indicators: dict) -> dict:
        """Fast single-call scan to identify candidates.

        Returns score 0-1 indicating how interesting this stock is.
        """
        from langchain_core.messages import HumanMessage

        rsi = indicators.get("rsi", 50)
        macd = indicators.get("macd", {})
        macd_hist = macd.get("histogram", 0)

        prompt = f"""You are a stock scanner. Rate how interesting {ticker} is for trading RIGHT NOW.

Data:
- Price: ${price:.2f}
- RSI: {rsi:.1f}
- MACD histogram: {macd_hist:.4f}

Scoring rules:
- RSI 25-35 (oversold bounce): HIGH score (0.7-0.9)
- RSI 65-75 (overbought reversal): HIGH score (0.7-0.9)
- MACD histogram crossing zero: HIGH score (0.7-0.9)
- RSI 40-60 (neutral): LOW score (0.2-0.4)
- RSI < 20 or > 80: MEDIUM (extreme, risky)

Respond with ONLY a number 0.0-1.0 (the score). Nothing else."""

        try:
            from tradingagents.llm_clients import create_llm_client
            client = create_llm_client(
                provider=self.ta_config["llm_provider"],
                model=self.ta_config["quick_think_llm"],
            )
            llm = client.get_llm()
            response = llm.invoke([HumanMessage(content=prompt)])
            text = response.content.strip()

            # Extract number
            import re
            match = re.search(r'(\d+\.?\d*)', text)
            score = float(match.group(1)) if match else 0.5
            score = max(0.0, min(1.0, score))

            return {"ticker": ticker, "score": score, "price": price, "indicators": indicators}

        except Exception as e:
            logger.error(f"Quick scan failed for {ticker}: {e}")
            return {"ticker": ticker, "score": 0.0, "price": price, "indicators": indicators}

    async def phase1_scan(self, tickers: list[str], prices: dict, indicators: dict) -> list[dict]:
        """Phase 1: Quick scan all stocks to find candidates."""
        logger.info(f"Phase 1: Scanning {len(tickers)} stocks...")
        start = time.time()

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def scan_one(ticker):
            async with semaphore:
                price = prices.get(ticker, 0)
                ind = indicators.get(ticker, {})
                if price <= 0:
                    return {"ticker": ticker, "score": 0, "price": 0, "indicators": {}}
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self.quick_scan, ticker, price, ind)

        tasks = [scan_one(t) for t in tickers]
        results = await asyncio.gather(*tasks)

        elapsed = time.time() - start
        logger.info(f"Phase 1 complete in {elapsed:.1f}s")

        # Sort by score and return top candidates
        scored = sorted(results, key=lambda x: x["score"], reverse=True)
        candidates = [s for s in scored if s["score"] >= self.min_conviction][:self.deep_count]

        logger.info(f"Found {len(candidates)} candidates (score >= {self.min_conviction})")
        for c in candidates:
            logger.info(f"  {c['ticker']}: score={c['score']:.2f}, price=${c['price']:.2f}")

        return candidates

    def deep_analysis(self, ticker: str, date: str) -> dict:
        """Phase 2: Full TradingAgents multi-agent analysis."""
        try:
            ta = TradingAgentsGraph(debug=False, config=self.ta_config.copy())
            state, decision = ta.propagate(ticker, date)

            decision_lower = decision.lower() if decision else ""
            if any(w in decision_lower for w in ["buy", "strong buy", "overweight"]):
                action = "buy"
            elif any(w in decision_lower for w in ["sell", "strong sell", "underweight"]):
                action = "sell"
            else:
                action = "hold"

            return {
                "ticker": ticker,
                "action": action,
                "conviction": 0.8 if action != "hold" else 0.5,
                "reasoning": decision,
                "mode": "deep",
            }

        except Exception as e:
            logger.error(f"Deep analysis failed for {ticker}: {e}")
            return {
                "ticker": ticker,
                "action": "hold",
                "conviction": 0,
                "reasoning": f"Error: {str(e)}",
                "mode": "deep",
            }

    async def phase2_deep(self, candidates: list[dict]) -> list[dict]:
        """Phase 2: Deep analysis on top candidates."""
        if not candidates:
            return []

        logger.info(f"Phase 2: Deep analysis on {len(candidates)} candidates...")
        start = time.time()

        # Deep analysis is sequential (each takes ~5 min)
        results = []
        for c in candidates:
            logger.info(f"  Analyzing {c['ticker']}...")
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self.deep_analysis, c["ticker"], datetime.now().strftime("%Y-%m-%d")
            )
            result["scan_score"] = c["score"]
            result["price"] = c["price"]
            result["indicators"] = c["indicators"]
            results.append(result)

        elapsed = time.time() - start
        logger.info(f"Phase 2 complete in {elapsed:.1f}s")

        return results

    async def run_two_phase(
        self, tickers: Optional[list[str]] = None, deep_count: int = None
    ) -> dict:
        """Run the full two-phase analysis.

        Phase 1: Quick scan all stocks (~5 min for 1000)
        Phase 2: Deep analysis on top candidates (~5 min each)
        """
        if deep_count:
            self.deep_count = deep_count

        total_start = time.time()
        logger.info("=" * 60)
        logger.info("Starting two-phase analysis")

        # Get universe
        if tickers is None:
            tickers = get_universe(self.config)

        logger.info(f"Universe: {len(tickers)} stocks")

        # Batch fetch data
        logger.info("Fetching prices...")
        prices = self.batch_get_prices(tickers)
        logger.info(f"Got prices for {len(prices)}/{len(tickers)}")

        logger.info("Fetching indicators...")
        indicators = self.batch_get_indicators(tickers)
        logger.info(f"Got indicators for {len(indicators)}/{len(tickers)}")

        # Phase 1: Quick scan
        candidates = await self.phase1_scan(tickers, prices, indicators)

        # Phase 2: Deep analysis
        deep_results = await self.phase2_deep(candidates)

        # Compile final results
        total_elapsed = time.time() - total_start

        buy_signals = [r for r in deep_results if r["action"] == "buy"]
        sell_signals = [r for r in deep_results if r["action"] == "sell"]
        hold_signals = [r for r in deep_results if r["action"] == "hold"]

        summary = {
            "phase1": {
                "total_scanned": len(tickers),
                "candidates_found": len(candidates),
                "elapsed_seconds": 0,  # filled below
            },
            "phase2": {
                "analyzed": len(deep_results),
                "buy": len(buy_signals),
                "sell": len(sell_signals),
                "hold": len(hold_signals),
            },
            "total_elapsed_seconds": round(total_elapsed, 1),
            "deep_results": deep_results,
            "candidates": candidates,
        }

        logger.info("=" * 60)
        logger.info(f"TWO-PHASE ANALYSIS COMPLETE")
        logger.info(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
        logger.info(f"Phase 1: {len(tickers)} scanned -> {len(candidates)} candidates")
        logger.info(f"Phase 2: {len(deep_results)} analyzed")
        logger.info(f"  Buy: {len(buy_signals)}, Sell: {len(sell_signals)}, Hold: {len(hold_signals)}")

        # Show buy signals
        if buy_signals:
            logger.info("\nBUY SIGNALS:")
            for b in sorted(buy_signals, key=lambda x: x["conviction"], reverse=True):
                logger.info(f"  {b['ticker']}: ${b['price']:.2f} (conviction: {b['conviction']:.2f})")
                logger.info(f"    {b['reasoning'][:100]}...")

        # Save results
        results_path = Path(__file__).parent / "results" / f"two_phase_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results_path.parent.mkdir(exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        return summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Two-Phase Trading Bot")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers")
    parser.add_argument("--deep-count", type=int, default=20, help="Number of stocks for deep analysis")
    parser.add_argument("--concurrent", type=int, default=20)

    args = parser.parse_args()

    bot = TwoPhaseBot(max_concurrent=args.concurrent)
    results = asyncio.run(bot.run_two_phase(tickers=args.tickers, deep_count=args.deep_count))

    print(f"\n{'='*60}")
    print(f"TWO-PHASE RESULTS")
    print(f"{'='*60}")
    print(f"Phase 1: {results['phase1']['total_scanned']} scanned -> {results['phase1']['candidates_found']} candidates")
    print(f"Phase 2: {results['phase2']['analyzed']} analyzed")
    print(f"Total time: {results['total_elapsed_seconds']:.1f}s ({results['total_elapsed_seconds']/60:.1f} min)")
    print(f"\nBuy: {results['phase2']['buy']}")
    print(f"Sell: {results['phase2']['sell']}")
    print(f"Hold: {results['phase2']['hold']}")

    if results["deep_results"]:
        print(f"\nTop Buy Signals:")
        buys = [r for r in results["deep_results"] if r["action"] == "buy"]
        for b in sorted(buys, key=lambda x: x["conviction"], reverse=True)[:5]:
            print(f"  {b['ticker']}: ${b['price']:.2f} (conviction: {b['conviction']:.2f})")


if __name__ == "__main__":
    main()
