"""Fast async trading bot - processes 1000+ stocks in minutes.

Key optimizations:
1. Async parallel analysis (20+ stocks concurrently)
2. Quick analysis mode (single LLM call instead of multi-agent debate)
3. Pre-filtering by technical indicators
4. Batch Twelve Data API calls
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


class FastTradingBot:
    """Optimized bot that processes many stocks concurrently."""

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

        # TradingAgents config
        self.ta_config = DEFAULT_CONFIG.copy()
        self.ta_config["llm_provider"] = self.config["llm"]["provider"]
        self.ta_config["deep_think_llm"] = self.config["llm"]["deep_think_model"]
        self.ta_config["quick_think_llm"] = self.config["llm"]["quick_think_model"]
        self.ta_config["temperature"] = self.config["llm"]["temperature"]

        logger.info(f"Fast bot initialized (max_concurrent={max_concurrent})")

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
                return ticker, {"rsi": rsi}
            except Exception:
                return ticker, {}

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(fetch_one, t) for t in tickers]
            for f in futures:
                ticker, ind = f.result()
                indicators[ticker] = ind

        return indicators

    def pre_filter(self, ticker: str, price: float, indicators: dict) -> dict:
        """Quick pre-filter based on technical indicators.

        Returns dict with should_analyze (bool) and reason.
        """
        if price <= 0:
            return {"should_analyze": False, "reason": "No price data"}

        rsi = indicators.get("rsi", 50)

        # Filter out stocks with extreme RSI (overbought/oversold)
        # But still analyze if RSI is in interesting zone (30-70)
        if rsi > 75:
            return {"should_analyze": True, "reason": f"RSI {rsi:.0f} - potential reversal"}
        if rsi < 25:
            return {"should_analyze": True, "reason": f"RSI {rsi:.0f} - potential bounce"}

        return {"should_analyze": True, "reason": "Passes filter"}

    def quick_analysis(self, ticker: str, price: float, indicators: dict) -> dict:
        """Fast single-call analysis instead of full multi-agent debate.

        Uses one LLM call with market data to make a quick decision.
        Much faster than the full TradingAgents pipeline.
        """
        from langchain_core.messages import HumanMessage

        rsi = indicators.get("rsi", 50)

        prompt = f"""You are a stock analyst. Analyze {ticker} and give a trading decision.

Current Data:
- Ticker: {ticker}
- Price: ${price:.2f}
- RSI: {rsi:.1f}

Rules:
- RSI > 70: Consider sell (overbought)
- RSI < 30: Consider buy (oversold)
- RSI 30-70: Hold unless strong reason

Respond with EXACTLY one of: BUY, SELL, HOLD
Then a brief 1-sentence reason.

Format: ACTION | reason"""

        try:
            # Create a fresh LLM for quick analysis
            from tradingagents.llm_clients import create_llm_client
            client = create_llm_client(
                provider=self.ta_config["llm_provider"],
                model=self.ta_config["quick_think_llm"],
            )
            llm = client.get_llm()

            response = llm.invoke([HumanMessage(content=prompt)])
            text = response.content.strip()

            # Parse response
            lines = text.split("\n")
            first_line = lines[0].strip()

            if "BUY" in first_line.upper():
                action = "buy"
                conviction = 0.7
            elif "SELL" in first_line.upper():
                action = "sell"
                conviction = 0.7
            else:
                action = "hold"
                conviction = 0.5

            reason = lines[1].strip() if len(lines) > 1 else first_line

            return {
                "ticker": ticker,
                "action": action,
                "conviction": conviction,
                "reasoning": reason,
                "price": price,
                "indicators": indicators,
                "mode": "quick",
            }

        except Exception as e:
            logger.error(f"Quick analysis failed for {ticker}: {e}")
            return {
                "ticker": ticker,
                "action": "hold",
                "conviction": 0,
                "reasoning": f"Error: {str(e)}",
                "price": price,
                "indicators": indicators,
                "mode": "quick",
            }

    def full_analysis(self, ticker: str, date: str) -> dict:
        """Full TradingAgents multi-agent analysis (slower but deeper)."""
        try:
            ta = TradingAgentsGraph(debug=False, config=self.ta_config.copy())
            state, decision = ta.propagate(ticker, date)

            decision_lower = decision.lower() if decision else ""
            if any(w in decision_lower for w in ["buy", "strong buy"]):
                action = "buy"
            elif any(w in decision_lower for w in ["sell", "strong sell"]):
                action = "sell"
            else:
                action = "hold"

            return {
                "ticker": ticker,
                "action": action,
                "conviction": 0.7 if action != "hold" else 0.5,
                "reasoning": decision,
                "price": 0,
                "indicators": {},
                "mode": "full",
            }
        except Exception as e:
            return {
                "ticker": ticker,
                "action": "hold",
                "conviction": 0,
                "reasoning": f"Error: {str(e)}",
                "price": 0,
                "indicators": {},
                "mode": "full",
            }

    async def analyze_batch(
        self, tickers: list[str], prices: dict, indicators: dict, mode: str = "quick"
    ) -> list[dict]:
        """Analyze a batch of tickers concurrently."""
        results = []
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def analyze_one(ticker):
            async with semaphore:
                price = prices.get(ticker, 0)
                ind = indicators.get(ticker, {})

                # Pre-filter
                filt = self.pre_filter(ticker, price, ind)
                if not filt["should_analyze"]:
                    return {
                        "ticker": ticker,
                        "action": "skip",
                        "conviction": 0,
                        "reasoning": filt["reason"],
                        "price": price,
                        "indicators": ind,
                        "mode": "filtered",
                    }

                # Run analysis in thread pool (LLM calls are blocking)
                loop = asyncio.get_event_loop()
                if mode == "quick":
                    result = await loop.run_in_executor(
                        None, self.quick_analysis, ticker, price, ind
                    )
                else:
                    date = datetime.now().strftime("%Y-%m-%d")
                    result = await loop.run_in_executor(
                        None, self.full_analysis, ticker, date
                    )
                return result

        tasks = [analyze_one(t) for t in tickers]
        results = await asyncio.gather(*tasks)
        return list(results)

    async def run_fast(
        self, tickers: Optional[list[str]] = None, mode: str = "quick"
    ) -> dict:
        """Run fast analysis on all tickers.

        Args:
            tickers: List of tickers (default: from universe config)
            mode: "quick" (single LLM call) or "full" (multi-agent)

        Returns:
            Results dict
        """
        start_time = time.time()
        logger.info(f"Starting fast analysis (mode={mode})")

        # Get universe
        if tickers is None:
            tickers = get_universe(self.config)

        logger.info(f"Analyzing {len(tickers)} tickers with {self.max_concurrent} concurrent workers")

        # Batch fetch prices and indicators
        logger.info("Fetching prices...")
        prices = self.batch_get_prices(tickers)
        logger.info(f"Got prices for {len(prices)}/{len(tickers)} tickers")

        logger.info("Fetching indicators...")
        indicators = self.batch_get_indicators(tickers)
        logger.info(f"Got indicators for {len(indicators)}/{len(tickers)} tickers")

        # Run analysis
        results = await self.analyze_batch(tickers, prices, indicators, mode)

        # Count results
        buy_count = sum(1 for r in results if r["action"] == "buy")
        sell_count = sum(1 for r in results if r["action"] == "sell")
        hold_count = sum(1 for r in results if r["action"] == "hold")
        skip_count = sum(1 for r in results if r["action"] == "skip")
        error_count = sum(1 for r in results if "Error" in r.get("reasoning", ""))

        elapsed = time.time() - start_time

        summary = {
            "total": len(tickers),
            "analyzed": len(tickers) - skip_count,
            "buy": buy_count,
            "sell": sell_count,
            "hold": hold_count,
            "skipped": skip_count,
            "errors": error_count,
            "elapsed_seconds": round(elapsed, 1),
            "stocks_per_second": round(len(tickers) / elapsed, 2),
            "mode": mode,
            "decisions": results,
        }

        logger.info(f"Fast analysis complete in {elapsed:.1f}s ({summary['stocks_per_second']} stocks/sec)")
        logger.info(f"  Buy: {buy_count}, Sell: {sell_count}, Hold: {hold_count}, Skip: {skip_count}")

        # Save results
        results_path = Path(__file__).parent / "results" / f"fast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results_path.parent.mkdir(exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        return summary

    def execute_decisions(self, results: dict) -> dict:
        """Execute trading decisions (requires Alpaca)."""
        if not self.alpaca:
            return {"executed": 0, "reason": "Alpaca not configured"}

        executed = 0
        for decision in results.get("decisions", []):
            if decision["action"] == "buy" and decision["conviction"] >= 0.6:
                try:
                    price = decision.get("price", 0)
                    if price <= 0:
                        continue

                    account = self.alpaca.get_account()
                    portfolio_value = account["portfolio_value"]
                    position_value = portfolio_value * 0.05  # 5% per position
                    qty = int(position_value / price)

                    if qty > 0:
                        self.alpaca.market_buy(decision["ticker"], qty=qty)
                        executed += 1
                        logger.info(f"Bought {qty} x {decision['ticker']} @ ${price:.2f}")
                except Exception as e:
                    logger.error(f"Failed to buy {decision['ticker']}: {e}")

            elif decision["action"] == "sell":
                try:
                    position = self.alpaca.get_position(decision["ticker"])
                    if position:
                        self.alpaca.market_sell(decision["ticker"], qty=int(position["qty"]))
                        executed += 1
                        logger.info(f"Sold {decision['ticker']}")
                except Exception as e:
                    logger.error(f"Failed to sell {decision['ticker']}: {e}")

        return {"executed": executed}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fast Trading Bot")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--concurrent", type=int, default=20)
    parser.add_argument("--execute", action="store_true", help="Execute trades")

    args = parser.parse_args()

    bot = FastTradingBot(max_concurrent=args.concurrent)

    results = asyncio.run(bot.run_fast(tickers=args.tickers, mode=args.mode))

    print(f"\n{'='*50}")
    print(f"FAST ANALYSIS RESULTS")
    print(f"{'='*50}")
    print(f"Total: {results['total']} stocks")
    print(f"Time: {results['elapsed_seconds']}s ({results['stocks_per_second']} stocks/sec)")
    print(f"Buy: {results['buy']}")
    print(f"Sell: {results['sell']}")
    print(f"Hold: {results['hold']}")
    print(f"Skipped: {results['skipped']}")

    # Show top buy signals
    buys = [d for d in results["decisions"] if d["action"] == "buy"]
    if buys:
        print(f"\nTop Buy Signals:")
        for b in sorted(buys, key=lambda x: x["conviction"], reverse=True)[:10]:
            print(f"  {b['ticker']}: ${b['price']:.2f} - {b['reasoning'][:60]}")

    if args.execute:
        exec_result = bot.execute_decisions(results)
        print(f"\nExecuted: {exec_result['executed']} trades")


if __name__ == "__main__":
    main()
