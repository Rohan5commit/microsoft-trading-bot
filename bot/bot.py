"""Main trading bot orchestrator.

Ties together TradingAgents analysis, Twelve Data, Alpaca execution, and risk management.
Alpaca is optional - bot runs analysis-only without it.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load env from tradingbot root
load_dotenv(Path(__file__).parent.parent / ".env")

# Add tradingbot root to path for TradingAgents import
sys.path.insert(0, str(Path(__file__).parent.parent))

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

from risk_manager import RiskManager
from universe import get_universe
from twelve_data import TwelveDataClient

# Alpaca is optional
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


class TradingBot:
    """Main trading bot that orchestrates analysis and execution."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the bot.

        Args:
            config_path: Path to config.json (default: bot/config.json)
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.json"

        with open(config_path) as f:
            self.config = json.load(f)

        # Initialize components
        self.twelve_data = TwelveDataClient()
        self.risk_manager = RiskManager(self.config)

        # Alpaca is optional
        self.alpaca = None
        if ALPACA_AVAILABLE and os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"):
            try:
                self.alpaca = AlpacaClient(paper=True)
                logger.info("Alpaca connected (paper trading)")
            except Exception as e:
                logger.warning(f"Alpaca not available: {e}")
        else:
            logger.info("Running in analysis-only mode (no Alpaca keys)")

        # TradingAgents config
        self.ta_config = DEFAULT_CONFIG.copy()
        self.ta_config["llm_provider"] = self.config["llm"]["provider"]
        self.ta_config["deep_think_llm"] = self.config["llm"]["deep_think_model"]
        self.ta_config["quick_think_llm"] = self.config["llm"]["quick_think_model"]
        self.ta_config["temperature"] = self.config["llm"]["temperature"]

        # Initialize TradingAgents
        self.ta = TradingAgentsGraph(debug=False, config=self.ta_config)

        mode = "analysis + trading" if self.alpaca else "analysis-only"
        logger.info(f"Trading bot initialized ({mode}) - Twelve Data + NVIDIA NIM")

    def get_price(self, ticker: str) -> float:
        """Get current stock price from Twelve Data (primary) or Alpaca (fallback)."""
        try:
            return self.twelve_data.get_price(ticker)
        except Exception:
            if self.alpaca:
                try:
                    return self.alpaca.get_stock_price(ticker)
                except Exception:
                    pass
            logger.error(f"Failed to get price for {ticker}")
            return 0

    def get_technical_indicators(self, ticker: str) -> dict:
        """Get technical indicators from Twelve Data."""
        try:
            rsi = self.twelve_data.get_rsi(ticker)
            macd = self.twelve_data.get_macd(ticker)
            ema_20 = self.twelve_data.get_ema(ticker, period=20)
            bb = self.twelve_data.get_bollinger_bands(ticker)

            return {
                "rsi": rsi,
                "macd": macd,
                "ema_20": ema_20,
                "bollinger": bb,
            }
        except Exception as e:
            logger.warning(f"Failed to get indicators for {ticker}: {e}")
            return {}

    def run_analysis(self, ticker: str, date: Optional[str] = None) -> dict:
        """Run TradingAgents analysis on a single ticker.

        Args:
            ticker: Stock symbol
            date: Analysis date (YYYY-MM-DD), default: today

        Returns:
            Decision dict with action, conviction, reasoning
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Running analysis for {ticker} on {date}")

        # Get current market data from Twelve Data
        price = 0
        indicators = {}
        try:
            price = self.get_price(ticker)
            indicators = self.get_technical_indicators(ticker)
            logger.info(f"{ticker}: ${price:.2f}, RSI={indicators.get('rsi', 'N/A')}")
        except Exception as e:
            logger.warning(f"Failed to fetch Twelve Data for {ticker}: {e}")

        try:
            state, decision = self.ta.propagate(ticker, date)

            # Parse the decision
            parsed = self._parse_decision(decision, state)
            parsed["ticker"] = ticker
            parsed["date"] = date
            parsed["price"] = price
            parsed["indicators"] = indicators

            logger.info(f"Decision for {ticker}: {parsed['action']} (conviction: {parsed['conviction']:.2f})")
            return parsed

        except Exception as e:
            logger.error(f"Analysis failed for {ticker}: {e}")
            return {
                "ticker": ticker,
                "date": date,
                "action": "hold",
                "conviction": 0,
                "reasoning": f"Analysis error: {str(e)}",
                "state": {},
                "price": 0,
                "indicators": {},
            }

    def _parse_decision(self, decision: str, state: dict) -> dict:
        """Parse TradingAgents decision into structured format."""
        decision_lower = decision.lower() if decision else ""

        # Determine action
        if any(word in decision_lower for word in ["buy", "strong buy", "overweight"]):
            action = "buy"
        elif any(word in decision_lower for word in ["sell", "strong sell", "underweight"]):
            action = "sell"
        else:
            action = "hold"

        # Extract conviction from decision text
        conviction = self._extract_conviction(decision)

        return {
            "action": action,
            "conviction": conviction,
            "reasoning": decision,
            "raw_decision": decision,
        }

    def _extract_conviction(self, decision: str) -> float:
        """Extract conviction score from decision text."""
        if not decision:
            return 0.5

        decision_lower = decision.lower()

        if any(w in decision_lower for w in ["strong buy", "strongly recommend buy"]):
            return 0.9
        if any(w in decision_lower for w in ["strong sell", "strongly recommend sell"]):
            return 0.9
        if "buy" in decision_lower and "overweight" in decision_lower:
            return 0.8
        if "sell" in decision_lower and "underweight" in decision_lower:
            return 0.8
        if "buy" in decision_lower:
            return 0.7
        if "sell" in decision_lower:
            return 0.7
        if "hold" in decision_lower or "neutral" in decision_lower:
            return 0.5

        return 0.5

    def execute_decision(self, decision: dict) -> dict:
        """Execute a trading decision (requires Alpaca)."""
        if not self.alpaca:
            return {"executed": False, "reason": "Alpaca not configured - analysis only"}

        ticker = decision["ticker"]
        action = decision["action"]
        conviction = decision["conviction"]

        logger.info(f"Executing decision for {ticker}: {action}")

        account = self.alpaca.get_account()
        portfolio_value = account["portfolio_value"]
        positions = self.alpaca.get_positions()

        risk_check = self.risk_manager.should_open_position(
            ticker, conviction, portfolio_value, positions
        )

        if action == "buy":
            if not risk_check["should_open"]:
                return {"executed": False, "reason": risk_check["reasoning"]}

            price = self.get_price(ticker)
            sizing = self.risk_manager.calculate_position_size(
                ticker, price, conviction, portfolio_value, positions
            )

            if sizing["action"] != "buy" or sizing["qty"] <= 0:
                return {"executed": False, "reason": sizing["reasoning"]}

            try:
                order = self.alpaca.market_buy(ticker, qty=sizing["qty"])
                self.risk_manager.record_trade(ticker, "buy", sizing["qty"], price, decision["reasoning"])
                return {"executed": True, "order": order, "sizing": sizing}
            except Exception as e:
                return {"executed": False, "reason": str(e)}

        elif action == "sell":
            position = self.alpaca.get_position(ticker)
            if not position:
                return {"executed": False, "reason": "No position to sell"}

            try:
                order = self.alpaca.market_sell(ticker, qty=int(position["qty"]))
                price = self.get_price(ticker)
                pnl = (price - position["avg_entry_price"]) * position["qty"]
                self.risk_manager.update_daily_pnl(pnl)
                self.risk_manager.record_trade(ticker, "sell", int(position["qty"]), price, decision["reasoning"])
                return {"executed": True, "order": order, "pnl": pnl}
            except Exception as e:
                return {"executed": False, "reason": str(e)}

        else:
            return {"executed": False, "reason": "Hold - no action"}

    def check_stop_losses(self) -> list[dict]:
        """Check all positions for stop-losses (requires Alpaca)."""
        if not self.alpaca:
            return []

        positions = self.alpaca.get_positions()
        results = []

        for pos in positions:
            ticker = pos["ticker"]
            entry_price = pos["avg_entry_price"]
            current_price = self.get_price(ticker)

            stop_check = self.risk_manager.check_stop_loss(ticker, entry_price, current_price)
            if stop_check["should_stop"]:
                self.alpaca.close_position(ticker)
                pnl = (current_price - entry_price) * pos["qty"]
                self.risk_manager.update_daily_pnl(pnl)
                self.risk_manager.record_trade(ticker, "sell", int(pos["qty"]), current_price, "stop-loss")
                results.append({"ticker": ticker, "action": "stop-loss", "pnl": pnl})
                continue

            tp_check = self.risk_manager.check_take_profit(ticker, entry_price, current_price)
            if tp_check["should_take_profit"]:
                self.alpaca.close_position(ticker)
                pnl = (current_price - entry_price) * pos["qty"]
                self.risk_manager.update_daily_pnl(pnl)
                self.risk_manager.record_trade(ticker, "sell", int(pos["qty"]), current_price, "take-profit")
                results.append({"ticker": ticker, "action": "take-profit", "pnl": pnl})

        return results

    def run_daily(self, tickers: Optional[list[str]] = None) -> dict:
        """Run the daily trading cycle."""
        logger.info("=" * 60)
        logger.info("Starting daily trading cycle")

        # Check if market is open (skip check if no Alpaca)
        if self.alpaca and not self.alpaca.is_market_open():
            logger.info("Market is closed. Skipping.")
            return {"status": "market_closed"}

        # Get universe
        if tickers is None:
            tickers = get_universe(self.config)

        logger.info(f"Analyzing {len(tickers)} tickers")

        # Check stop-losses first (only if Alpaca available)
        if self.alpaca:
            stop_results = self.check_stop_losses()
            if stop_results:
                logger.info(f"Stop-loss/take-profit actions: {len(stop_results)}")

        # Run analysis
        results = {
            "analyzed": 0,
            "bought": 0,
            "sold": 0,
            "held": 0,
            "errors": 0,
            "decisions": [],
            "mode": "analysis-only" if not self.alpaca else "trading",
        }

        for ticker in tickers:
            try:
                # Run analysis
                decision = self.run_analysis(ticker)
                results["analyzed"] += 1

                # Execute only if Alpaca is available
                if self.alpaca:
                    execution = self.execute_decision(decision)
                    decision["execution"] = execution

                    if execution.get("executed"):
                        if decision["action"] == "buy":
                            results["bought"] += 1
                        elif decision["action"] == "sell":
                            results["sold"] += 1
                    else:
                        results["held"] += 1
                else:
                    results["held"] += 1

                results["decisions"].append(decision)

                # Rate limit
                time.sleep(2)

            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                results["errors"] += 1

        # Summary
        logger.info(f"Daily cycle complete: {results['analyzed']} analyzed, "
                     f"{results['bought']} bought, {results['sold']} sold, "
                     f"{results['held']} held, {results['errors']} errors")

        if self.alpaca:
            account = self.alpaca.get_account()
            results["portfolio_value"] = account["portfolio_value"]
            results["positions"] = len(self.alpaca.get_positions())
            logger.info(f"Portfolio value: ${account['portfolio_value']:,.2f}")

        # Save results
        results_path = Path(__file__).parent / "results" / f"{datetime.now().strftime('%Y%m%d')}.json"
        results_path.parent.mkdir(exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        return results

    def get_status(self) -> dict:
        """Get current bot status."""
        status = {
            "mode": "analysis-only" if not self.alpaca else "trading",
            "alpaca_connected": self.alpaca is not None,
        }

        if self.alpaca:
            account = self.alpaca.get_account()
            positions = self.alpaca.get_positions()
            status.update({
                "portfolio_value": account["portfolio_value"],
                "cash": account["cash"],
                "buying_power": account["buying_power"],
                "position_count": len(positions),
                "positions": positions,
                "market_open": self.alpaca.is_market_open(),
                "market_clock": self.alpaca.get_market_clock(),
            })

        return status


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Trading Bot")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--analyze", help="Analyze a single ticker")
    parser.add_argument("--date", help="Analysis date (YYYY-MM-DD)")
    parser.add_argument("--daily", action="store_true", help="Run daily cycle")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to analyze")

    args = parser.parse_args()

    bot = TradingBot(config_path=args.config)

    if args.analyze:
        result = bot.run_analysis(args.analyze, args.date)
        print(json.dumps(result, indent=2, default=str))
    elif args.daily:
        result = bot.run_daily(tickers=args.tickers)
        print(json.dumps(result, indent=2, default=str))
    elif args.status:
        status = bot.get_status()
        print(json.dumps(status, indent=2, default=str))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
