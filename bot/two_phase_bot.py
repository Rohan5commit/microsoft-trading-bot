"""Two-phase trading bot: Quick scan all stocks, then deep analysis on candidates.

Phase 1: Fast scan (5 min) - Analyze all 1000 stocks with single LLM call
Phase 2: Deep analysis (1-2 hrs) - Full TradingAgents on top 20-30 candidates

Model decides everything: allocation, conviction, entry/exit. No fixed constraints.
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    TRADINGAGENTS_AVAILABLE = True
except ImportError:
    TRADINGAGENTS_AVAILABLE = False
    DEFAULT_CONFIG = {}
    TradingAgentsGraph = None

from risk_manager import RiskManager
from universe import get_universe
from twelve_data import TwelveDataClient

try:
    from alpaca_client import AlpacaClient
    ALPACA_AVAILABLE = True
except (ImportError, ValueError):
    ALPACA_AVAILABLE = False

try:
    from email_sender import EmailSender
    EMAIL_AVAILABLE = True
except (ImportError, ValueError):
    EMAIL_AVAILABLE = False

try:
    from portfolio import PortfolioTracker
    PORTFOLIO_AVAILABLE = True
except (ImportError, ValueError):
    PORTFOLIO_AVAILABLE = False

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

        # Email sender optional
        self.email_sender = None
        if EMAIL_AVAILABLE and os.getenv("GMAIL_APP_PASSWORD"):
            try:
                self.email_sender = EmailSender()
            except Exception:
                pass

        # Portfolio tracker optional
        self.portfolio = None
        if PORTFOLIO_AVAILABLE:
            try:
                self.portfolio = PortfolioTracker()
                portfolio_config = self.config.get("portfolio", {})
                if "initial_capital" in portfolio_config:
                    self.portfolio.initial_capital = portfolio_config["initial_capital"]
                if "leverage" in portfolio_config:
                    self.portfolio.leverage = portfolio_config["leverage"]
            except Exception:
                pass

        # TradingAgents config for deep analysis
        if TRADINGAGENTS_AVAILABLE:
            self.ta_config = DEFAULT_CONFIG.copy()
            self.ta_config["llm_provider"] = self.config["llm"]["provider"]
            self.ta_config["deep_think_llm"] = self.config["llm"]["deep_think_model"]
            self.ta_config["quick_think_llm"] = self.config["llm"]["quick_think_model"]
            self.ta_config["temperature"] = self.config["llm"]["temperature"]
        else:
            self.ta_config = {}
            logger.warning("TradingAgents not available - deep analysis disabled")

        # Deep analysis settings
        self.deep_count = self.config.get("deep_analysis", {}).get("count", 20)
        self.min_conviction = self.config.get("deep_analysis", {}).get("min_conviction", 0.3)

        # Execution settings
        self.execution_enabled = self.config.get("execution", {}).get("enabled", False)
        self.execution_mode = self.config.get("execution", {}).get("mode", "market")

        logger.info(f"Two-phase bot initialized (deep_count={self.deep_count}, execution={self.execution_enabled})")

    def get_alpaca_status(self) -> Optional[dict]:
        """Get Alpaca account status with positions."""
        if not self.alpaca:
            return None

        try:
            account = self.alpaca.get_account()
            positions = self.alpaca.get_positions()

            return {
                "portfolio_value": account["portfolio_value"],
                "cash": account["cash"],
                "buying_power": account["buying_power"],
                "market_open": self.alpaca.is_market_open(),
                "positions": positions,
            }
        except Exception as e:
            logger.error(f"Failed to get Alpaca status: {e}")
            return None

    def send_email(self, results: dict, alpaca_status: Optional[dict] = None) -> bool:
        """Send daily update email."""
        if not self.email_sender:
            logger.info("Email sender not configured, skipping email")
            return False

        try:
            success = self.email_sender.send_daily_update(results, alpaca_status)
            if success:
                logger.info("Daily update email sent successfully")
            else:
                logger.error("Failed to send daily update email")
            return success
        except Exception as e:
            logger.error(f"Email error: {e}")
            return False

    def send_error_email(self, error_message: str, traceback_str: str = "") -> bool:
        """Send error notification email."""
        if not self.email_sender:
            return False

        try:
            return self.email_sender.send_error_notification(error_message, traceback_str)
        except Exception as e:
            logger.error(f"Error email failed: {e}")
            return False

    def get_portfolio_metrics(self, alpaca_status: Optional[dict] = None) -> Optional[dict]:
        """Get portfolio return metrics if tracker is available."""
        if not self.portfolio:
            return None

        try:
            if alpaca_status:
                current_equity = alpaca_status.get("portfolio_value", 0)
            else:
                current_equity = self.portfolio.initial_capital

            return self.portfolio.get_status(current_equity)
        except Exception as e:
            logger.error(f"Portfolio metrics error: {e}")
            return None

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
        """Fast single-call scan to identify candidates."""
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

            match = re.search(r'(\d+\.?\d*)', text)
            score = float(match.group(1)) if match else 0.5
            # Detect percentage vs decimal: if > 1.0, assume percentage
            if score > 1.0:
                score = score / 100.0
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
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, self.quick_scan, ticker, price, ind)

        tasks = [scan_one(t) for t in tickers]
        results = await asyncio.gather(*tasks)

        elapsed = time.time() - start
        logger.info(f"Phase 1 complete in {elapsed:.1f}s")

        scored = sorted(results, key=lambda x: x["score"], reverse=True)
        candidates = [s for s in scored if s["score"] >= self.min_conviction][:self.deep_count]

        logger.info(f"Found {len(candidates)} candidates (score >= {self.min_conviction})")
        for c in candidates:
            logger.info(f"  {c['ticker']}: score={c['score']:.2f}, price=${c['price']:.2f}")

        return candidates

    def _extract_conviction_and_allocation(self, decision: str, action: str) -> tuple[float, Optional[float]]:
        """Extract conviction and allocation % from model decision text.

        Model MUST specify allocation. If not found, returns None and trade is skipped.

        Looks for patterns like:
        - "conviction: 0.8" or "confidence: 75%"
        - "allocation: 5%" or "allocate 5%" or "position size: 3%"
        - "sizing: 2% of portfolio"

        Returns (conviction, allocation_pct or None)
        """
        conviction = 0.7  # default
        allocation = None

        # Try to extract conviction
        conv_match = re.search(r'conviction[:\s]+(\d+\.?\d*)', decision, re.IGNORECASE)
        if conv_match:
            val = float(conv_match.group(1))
            if val > 1.0:
                val = val / 100.0
            conviction = min(1.0, max(0.0, val))
        else:
            conf_match = re.search(r'confidence[:\s]+(\d+\.?\d*)', decision, re.IGNORECASE)
            if conf_match:
                val = float(conf_match.group(1))
                if val > 1.0:
                    val = val / 100.0
                conviction = min(1.0, max(0.0, val))
            else:
                lower = decision.lower()
                if any(w in lower for w in ["strong buy", "strong buy signal", "high conviction"]):
                    conviction = 0.9
                elif any(w in lower for w in ["buy", "overweight", "accumulate"]):
                    conviction = 0.75
                elif any(w in lower for w in ["weak buy", "slight buy", "marginal"]):
                    conviction = 0.55
                elif any(w in lower for w in ["strong sell", "strong sell signal"]):
                    conviction = 0.9
                elif any(w in lower for w in ["sell", "underweight", "reduce"]):
                    conviction = 0.75
                elif any(w in lower for w in ["weak sell", "slight sell"]):
                    conviction = 0.55

        # Extract allocation - model MUST provide this
        # Pattern 1: "allocation: 5%" or "allocation 5%"
        alloc_match = re.search(r'allocat(?:ion|e)[:\s]+(\d+\.?\d*)\s*%', decision, re.IGNORECASE)
        if not alloc_match:
            # Pattern 1b: "allocation of 5%"
            alloc_match = re.search(r'allocat(?:ion|e)\s+of\s+(\d+\.?\d*)\s*%', decision, re.IGNORECASE)
        if alloc_match:
            allocation = float(alloc_match.group(1))
        else:
            # Pattern 2: "position size: 3%" or "position: 3%"
            pos_match = re.search(r'position\s*(?:size)?[:\s]+(\d+\.?\d*)\s*%', decision, re.IGNORECASE)
            if pos_match:
                allocation = float(pos_match.group(1))
            else:
                # Pattern 3: "sizing: 2% of portfolio"
                sizing_match = re.search(r'sizing[:\s]+(\d+\.?\d*)\s*%', decision, re.IGNORECASE)
                if sizing_match:
                    allocation = float(sizing_match.group(1))
                else:
                    # Pattern 4: "recommend X% portfolio" or "X% of portfolio"
                    portfolio_match = re.search(r'(\d+\.?\d*)\s*%\s*(?:of\s+)?(?:portfolio|capital|equity)', decision, re.IGNORECASE)
                    if portfolio_match:
                        allocation = float(portfolio_match.group(1))
                    else:
                        # Pattern 5: "put/invest X%" or "stake of X%"
                        stake_match = re.search(r'(?:put|invest|stake|size)[:\s]+(\d+\.?\d*)\s*%', decision, re.IGNORECASE)
                        if stake_match:
                            allocation = float(stake_match.group(1))

        return conviction, allocation

    def _force_allocation(self, ticker: str, decision: str, action: str) -> tuple[float, float]:
        """Force model to output conviction + allocation via follow-up LLM call.

        Returns (conviction, allocation_pct). allocation is always provided.
        """
        from langchain_core.messages import HumanMessage

        prompt = f"""You are a portfolio manager. Based on this analysis for {ticker}, give your final conviction and portfolio allocation.

ANALYSIS:
{decision[:2000]}

RECOMMENDATION: {action.upper()}

You MUST respond in EXACTLY this format (nothing else):

CONVICTION: <number between 0 and 1>
ALLOCATION: <number between 0.1 and 25, representing percent of portfolio>

Example:
CONVICTION: 0.85
ALLOCATION: 8.5

Do NOT include any other text. Only the two lines above."""

        try:
            from tradingagents.llm_clients import create_llm_client
            client = create_llm_client(
                provider=self.ta_config["llm_provider"],
                model=self.ta_config["deep_think_llm"],
            )
            llm = client.get_llm()
            response = llm.invoke([HumanMessage(content=prompt)])
            text = response.content.strip()

            # Parse strict format
            conv_match = re.search(r'CONVICTION:\s*(\d+\.?\d*)', text, re.IGNORECASE)
            alloc_match = re.search(r'ALLOCATION:\s*(\d+\.?\d*)', text, re.IGNORECASE)

            conviction = 0.7
            allocation = 5.0  # default 5% if parsing fails

            if conv_match:
                conviction = float(conv_match.group(1))
                conviction = max(0.0, min(1.0, conviction))

            if alloc_match:
                allocation = float(alloc_match.group(1))
                allocation = max(0.1, min(25.0, allocation))

            logger.info(f"Forced allocation for {ticker}: conviction={conviction:.2f}, allocation={allocation:.1f}%")
            return conviction, allocation

        except Exception as e:
            logger.error(f"Force allocation failed for {ticker}: {e}")
            return 0.7, 5.0

    def _detect_action(self, decision: str) -> str:
        """Detect action from LLM decision text, handling negation.

        Returns "buy", "sell", or "hold".
        """
        lower = decision.lower()

        # Check for negation patterns first
        negation_patterns = [
            r"don'?t\s+buy",
            r"do\s+not\s+buy",
            r"avoid\s+buy",
            r"shouldn'?t\s+buy",
            r"should\s+not\s+buy",
            r"not\s+a\s+buy",
            r"no\s+buy\s+signal",
            r"not\s+recommend.*buy",
            r"wouldn'?t\s+buy",
            r"would\s+not\s+buy",
            r"don'?t\s+sell",
            r"do\s+not\s+sell",
            r"avoid\s+sell",
            r"shouldn'?t\s+sell",
            r"should\s+not\s+sell",
            r"not\s+a\s+sell",
            r"no\s+sell\s+signal",
            r"not\s+recommend.*sell",
            r"wouldn'?t\s+sell",
            r"would\s+not\s+sell",
        ]

        has_buy_negation = any(re.search(p, lower) for p in negation_patterns[:10])
        has_sell_negation = any(re.search(p, lower) for p in negation_patterns[10:])

        # Strong signals (look for emphatic/declarative patterns)
        strong_buy = any(w in lower for w in ["strong buy", "strong buy signal", "high conviction buy", "overweight", "accumulate"])
        strong_sell = any(w in lower for w in ["strong sell", "strong sell signal", "high conviction sell", "underweight"])

        # Check if the final recommendation (last 200 chars) is buy/sell
        # The conclusion/recommendation section is usually at the end
        conclusion = lower[-200:] if len(lower) > 200 else lower
        conclusion_has_buy = any(w in conclusion for w in ["buy", "buy signal", "go long", "long position"])
        conclusion_has_sell = any(w in conclusion for w in ["sell", "sell signal", "go short", "short position"])

        # General mentions (anywhere in text)
        mention_buy = any(w in lower for w in ["buy", "accumulate"])
        mention_sell = any(w in lower for w in ["sell", "reduce"])

        # Decision logic: conclusion > strong signals > mentions, with negation override
        if has_buy_negation and not mention_buy:
            return "hold"
        if has_sell_negation and not mention_sell:
            return "hold"

        if strong_buy and not has_buy_negation:
            return "buy"
        if strong_sell and not has_sell_negation:
            return "sell"

        if conclusion_has_buy and not has_buy_negation:
            return "buy"
        if conclusion_has_sell and not has_sell_negation:
            return "sell"

        if mention_buy and not has_buy_negation:
            return "buy"
        if mention_sell and not has_sell_negation:
            return "sell"

        return "hold"

    def deep_analysis(self, ticker: str, date: str) -> dict:
        """Phase 2: Full TradingAgents multi-agent analysis + forced allocation."""
        if not TRADINGAGENTS_AVAILABLE or TradingAgentsGraph is None:
            return {
                "ticker": ticker,
                "action": "hold",
                "conviction": 0,
                "suggested_allocation_pct": None,
                "reasoning": "TradingAgents not installed - deep analysis skipped",
                "mode": "deep",
            }

        try:
            ta = TradingAgentsGraph(debug=False, config=self.ta_config.copy())
            state, decision = ta.propagate(ticker, date)

            action = self._detect_action(decision or "")

            # First try to extract conviction + allocation from existing decision
            # Only make extra LLM call if extraction fails
            if action != "hold":
                conviction, allocation = self._extract_conviction_and_allocation(decision or "", action)
                if allocation is None:
                    conviction, allocation = self._force_allocation(ticker, decision or "", action)
            else:
                conviction = 0.5
                allocation = 0.0

            return {
                "ticker": ticker,
                "action": action,
                "conviction": conviction,
                "suggested_allocation_pct": allocation if action != "hold" else None,
                "reasoning": decision,
                "mode": "deep",
            }

        except Exception as e:
            logger.error(f"Deep analysis failed for {ticker}: {e}")
            return {
                "ticker": ticker,
                "action": "hold",
                "conviction": 0,
                "suggested_allocation_pct": None,
                "reasoning": f"Error: {str(e)}",
                "mode": "deep",
            }

    async def phase2_deep(self, candidates: list[dict]) -> list[dict]:
        """Phase 2: Deep analysis on top candidates."""
        if not candidates:
            return []

        logger.info(f"Phase 2: Deep analysis on {len(candidates)} candidates...")
        start = time.time()

        results = []
        for c in candidates:
            logger.info(f"  Analyzing {c['ticker']}...")
            loop = asyncio.get_running_loop()
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

    async def execute_trades(self, deep_results: list[dict]) -> list[dict]:
        """Execute trades based on model decisions.

        Model decides everything: what, how much, when.
        We only enforce circuit breaker and buying power limits.

        Returns list of execution results.
        """
        if not self.execution_enabled:
            logger.info("Execution disabled, skipping trade execution")
            return []

        if not self.alpaca:
            logger.warning("Alpaca not configured, cannot execute trades")
            return []

        if not self.alpaca.is_market_open():
            logger.warning("Market is closed, skipping trade execution")
            return []

        logger.info("=" * 60)
        logger.info("EXECUTING TRADES")
        logger.info("=" * 60)

        # Get current state
        account = self.alpaca.get_account()
        portfolio_value = account["portfolio_value"]
        buying_power = account["buying_power"]
        current_positions = self.alpaca.get_positions()

        logger.info(f"Portfolio: ${portfolio_value:,.2f} | Positions: {len(current_positions)}")

        # Circuit breaker check
        cb = self.risk_manager.check_circuit_breaker(portfolio_value)
        if cb["tripped"]:
            logger.warning(f"CIRCUIT BREAKER TRIPPED: {cb['reasoning']}")
            return [{"status": "circuit_breaker", "reasoning": cb["reasoning"]}]

        execution_results = []

        # Build position lookup
        pos_by_ticker = {p["ticker"]: p for p in current_positions}

        # Process each signal
        for signal in deep_results:
            ticker = signal["ticker"]
            action = signal["action"]
            conviction = signal["conviction"]
            price = signal["price"]
            allocation_pct = signal.get("suggested_allocation_pct")

            if action == "hold":
                continue

            existing_pos = pos_by_ticker.get(ticker)
            existing_side = existing_pos["side"] if existing_pos else None

            result = {
                "ticker": ticker,
                "signal": action,
                "conviction": conviction,
                "price": price,
                "status": "pending",
                "reasoning": "",
            }

            try:
                # CASE 1: Buy signal + no position -> Open long
                if action == "buy" and not existing_pos:
                    sizing = self.risk_manager.calculate_position_size(
                        ticker, price, conviction, portfolio_value,
                        current_positions, allocation_pct, buying_power
                    )
                    if sizing["action"] == "skip":
                        result["status"] = "skipped"
                        result["reasoning"] = sizing["reasoning"]
                    else:
                        order = self.alpaca.market_buy(ticker, qty=sizing["qty"])
                        result["status"] = "filled"
                        result["qty"] = sizing["qty"]
                        result["order_id"] = order.get("id")
                        result["reasoning"] = sizing["reasoning"]
                        self.risk_manager.record_trade(ticker, "buy", sizing["qty"], price, sizing["reasoning"])
                        logger.info(f"BUY {ticker}: {sizing['qty']} shares @ ${price:.2f} = ${sizing['notional']:,.2f}")

                # CASE 2: Buy signal + short position -> Close short, open long
                elif action == "buy" and existing_side == "short":
                    # Close short first
                    qty_to_close = abs(int(existing_pos["qty"]))
                    entry_price = existing_pos["avg_entry_price"]
                    self.alpaca.market_buy(ticker, qty=qty_to_close)
                    # P&L on short close: (entry - exit) * qty
                    pnl = (entry_price - price) * qty_to_close
                    self.risk_manager.update_daily_pnl(pnl)
                    self.risk_manager.record_trade(ticker, "close_short", qty_to_close, price, "Closing short before long")
                    logger.info(f"CLOSE SHORT {ticker}: {qty_to_close} shares, P&L: ${pnl:+,.2f}")

                    # Open long
                    sizing = self.risk_manager.calculate_position_size(
                        ticker, price, conviction, portfolio_value,
                        current_positions, allocation_pct, buying_power
                    )
                    if sizing["action"] != "skip":
                        order = self.alpaca.market_buy(ticker, qty=sizing["qty"])
                        result["status"] = "filled"
                        result["qty"] = sizing["qty"]
                        result["order_id"] = order.get("id")
                        result["reasoning"] = f"Closed short (P&L: ${pnl:+,.2f}) + {sizing['reasoning']}"
                        self.risk_manager.record_trade(ticker, "buy", sizing["qty"], price, sizing["reasoning"])
                        logger.info(f"BUY {ticker}: {sizing['qty']} shares @ ${price:.2f}")
                    else:
                        result["status"] = "filled"
                        result["reasoning"] = f"Closed short (P&L: ${pnl:+,.2f}), model did not open replacement long"

                # CASE 3: Sell signal + long position -> Close long
                elif action == "sell" and existing_side == "long":
                    qty_to_close = abs(int(existing_pos["qty"]))
                    entry_price = existing_pos["avg_entry_price"]
                    self.alpaca.market_sell(ticker, qty=qty_to_close)
                    # P&L on long close: (exit - entry) * qty
                    pnl = (price - entry_price) * qty_to_close
                    self.risk_manager.update_daily_pnl(pnl)
                    self.risk_manager.record_trade(ticker, "sell", qty_to_close, price, "Closing long on sell signal")
                    result["status"] = "filled"
                    result["qty"] = qty_to_close
                    result["reasoning"] = f"Closed long: {qty_to_close} shares, P&L: ${pnl:+,.2f}"
                    logger.info(f"SELL (close long) {ticker}: {qty_to_close} shares @ ${price:.2f}, P&L: ${pnl:+,.2f}")

                # CASE 4: Sell signal + no position -> Open short
                elif action == "sell" and not existing_pos:
                    sizing = self.risk_manager.calculate_position_size(
                        ticker, price, conviction, portfolio_value,
                        current_positions, allocation_pct, buying_power
                    )
                    if sizing["action"] == "skip":
                        result["status"] = "skipped"
                        result["reasoning"] = sizing["reasoning"]
                    else:
                        order = self.alpaca.market_sell(ticker, qty=sizing["qty"])
                        result["status"] = "filled"
                        result["qty"] = sizing["qty"]
                        result["order_id"] = order.get("id")
                        result["reasoning"] = sizing["reasoning"]
                        self.risk_manager.record_trade(ticker, "short", sizing["qty"], price, sizing["reasoning"])
                        logger.info(f"SHORT {ticker}: {sizing['qty']} shares @ ${price:.2f}")

                # CASE 5: Sell signal + short position -> Hold short (model already short)
                elif action == "sell" and existing_side == "short":
                    result["status"] = "no_action"
                    result["reasoning"] = "Already short, model confirms hold"

                # CASE 6: Buy signal + long position -> Hold long (model already long)
                elif action == "buy" and existing_side == "long":
                    result["status"] = "no_action"
                    result["reasoning"] = "Already long, model confirms hold"

                else:
                    result["status"] = "no_action"
                    result["reasoning"] = f"Signal={action}, position={existing_side or 'none'}"

            except Exception as e:
                result["status"] = "error"
                result["reasoning"] = f"Execution error: {str(e)}"
                logger.error(f"Execution error for {ticker}: {e}")

            execution_results.append(result)

            # Re-fetch state after a filled trade to prevent stale data
            if result["status"] == "filled":
                try:
                    account = self.alpaca.get_account()
                    portfolio_value = account["portfolio_value"]
                    buying_power = account["buying_power"]
                    current_positions = self.alpaca.get_positions()
                    pos_by_ticker = {p["ticker"]: p for p in current_positions}
                    logger.info(f"State refreshed: portfolio=${portfolio_value:,.2f}, positions={len(current_positions)}")

                    # Check circuit breaker after each trade
                    cb = self.risk_manager.check_circuit_breaker(portfolio_value)
                    if cb["tripped"]:
                        logger.warning(f"CIRCUIT BREAKER TRIPPED mid-execution: {cb['reasoning']}")
                        execution_results.append({"status": "circuit_breaker", "reasoning": cb["reasoning"]})
                        break
                except Exception as e:
                    logger.warning(f"Failed to refresh state: {e}")

        # Summary
        filled = [r for r in execution_results if r["status"] == "filled"]
        skipped = [r for r in execution_results if r["status"] == "skipped"]
        errors = [r for r in execution_results if r["status"] == "error"]

        logger.info(f"Execution complete: {len(filled)} filled, {len(skipped)} skipped, {len(errors)} errors")

        return execution_results

    async def run_two_phase(
        self, tickers: Optional[list[str]] = None, deep_count: int = None
    ) -> dict:
        """Run the full two-phase analysis + execution."""
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

        # Execute trades
        execution_results = await self.execute_trades(deep_results)

        # Get Alpaca account status (after execution)
        alpaca_status = self.get_alpaca_status()

        # Get portfolio metrics
        portfolio_metrics = self.get_portfolio_metrics(alpaca_status)

        # Compile final results
        total_elapsed = time.time() - total_start

        buy_signals = [r for r in deep_results if r["action"] == "buy"]
        sell_signals = [r for r in deep_results if r["action"] == "sell"]
        hold_signals = [r for r in deep_results if r["action"] == "hold"]

        summary = {
            "phase1": {
                "total_scanned": len(tickers),
                "candidates_found": len(candidates),
                "elapsed_seconds": 0,
            },
            "phase2": {
                "analyzed": len(deep_results),
                "buy": len(buy_signals),
                "sell": len(sell_signals),
                "hold": len(hold_signals),
            },
            "execution": {
                "enabled": self.execution_enabled,
                "results": execution_results,
                "filled": len([r for r in execution_results if r.get("status") == "filled"]),
                "skipped": len([r for r in execution_results if r.get("status") == "skipped"]),
                "errors": len([r for r in execution_results if r.get("status") == "error"]),
            },
            "total_elapsed_seconds": round(total_elapsed, 1),
            "deep_results": deep_results,
            "candidates": candidates,
            "alpaca_status": alpaca_status,
            "portfolio_metrics": portfolio_metrics,
        }

        # Send email with results and Alpaca status
        self.send_email(summary, alpaca_status)

        logger.info("=" * 60)
        logger.info(f"TWO-PHASE ANALYSIS COMPLETE")
        logger.info(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
        logger.info(f"Phase 1: {len(tickers)} scanned -> {len(candidates)} candidates")
        logger.info(f"Phase 2: {len(deep_results)} analyzed")
        logger.info(f"  Buy: {len(buy_signals)}, Sell: {len(sell_signals)}, Hold: {len(hold_signals)}")
        if execution_results:
            logger.info(f"  Execution: {summary['execution']['filled']} filled, {summary['execution']['skipped']} skipped, {summary['execution']['errors']} errors")

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

    if results["execution"]["results"]:
        print(f"\nExecution: {results['execution']['filled']} filled, {results['execution']['skipped']} skipped, {results['execution']['errors']} errors")

    if results["deep_results"]:
        print(f"\nTop Buy Signals:")
        buys = [r for r in results["deep_results"] if r["action"] == "buy"]
        for b in sorted(buys, key=lambda x: x["conviction"], reverse=True)[:5]:
            print(f"  {b['ticker']}: ${b['price']:.2f} (conviction: {b['conviction']:.2f})")


if __name__ == "__main__":
    main()
