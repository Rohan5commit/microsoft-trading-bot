"""Two-phase trading bot: Quick scan all stocks, then deep analysis on candidates.

Phase 1: Fast scan (5 min) - Analyze all stocks with single LLM call
Phase 2: Deep analysis (1-2 hrs) - Full TradingAgents on top candidates

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

logger = logging.getLogger(__name__)

# NVIDIA NIM: patch analysts to use sequential tool-calls (fixes 500 errors)
# Also patch debate agents with citation requirements for better signal quality
if TRADINGAGENTS_AVAILABLE:
    try:
        import tradingagents.agents.analysts.market_analyst as _mkt
        import tradingagents.agents.analysts.news_analyst as _news
        import tradingagents.agents.analysts.fundamentals_analyst as _fund
        import tradingagents.agents.researchers.bull_researcher as _bull
        import tradingagents.agents.researchers.bear_researcher as _bear
        import tradingagents.agents.risk_mgmt.aggressive_debator as _aggr
        import tradingagents.agents.risk_mgmt.conservative_debator as _cons
        import tradingagents.agents.risk_mgmt.neutral_debator as _neut
        from bot.nvidia_nim_compat import (
            create_market_analyst_nim,
            create_news_analyst_nim,
            create_fundamentals_analyst_nim,
            create_bull_researcher_nim,
            create_bear_researcher_nim,
            create_aggressive_debator_nim,
            create_conservative_debator_nim,
            create_neutral_debator_nim,
        )
        _mkt.create_market_analyst = create_market_analyst_nim
        _news.create_news_analyst = create_news_analyst_nim
        _fund.create_fundamentals_analyst = create_fundamentals_analyst_nim
        _bull.create_bull_researcher = create_bull_researcher_nim
        _bear.create_bear_researcher = create_bear_researcher_nim
        _aggr.create_aggressive_debator = create_aggressive_debator_nim
        _cons.create_conservative_debator = create_conservative_debator_nim
        _neut.create_neutral_debator = create_neutral_debator_nim

        # CRITICAL: Also patch tradingagents.graph.setup module directly.
        # The lambdas in setup_graph() resolve function names from this
        # module's globals at call time. Without this, the original
        # functions are used even after the module-level patches above.
        import tradingagents.graph.setup as _setup
        _setup.create_market_analyst = create_market_analyst_nim
        _setup.create_news_analyst = create_news_analyst_nim
        _setup.create_fundamentals_analyst = create_fundamentals_analyst_nim
        _setup.create_bull_researcher = create_bull_researcher_nim
        _setup.create_bear_researcher = create_bear_researcher_nim
        _setup.create_aggressive_debator = create_aggressive_debator_nim
        _setup.create_conservative_debator = create_conservative_debator_nim
        _setup.create_neutral_debator = create_neutral_debator_nim

        logger.info("NVIDIA NIM compat: analysts + debate agents patched (setup module targeted)")
    except ImportError as e:
        logger.warning(f"NVIDIA NIM compat: patch skipped ({e})")

try:
    from langchain_core.messages import HumanMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    HumanMessage = None

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

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.json"

        with open(config_path) as f:
            self.config = json.load(f)

        self.twelve_data = TwelveDataClient()
        self.risk_manager = RiskManager(self.config)

        # Alpaca optional
        self.alpaca = None
        if ALPACA_AVAILABLE and os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"):
            try:
                self.alpaca = AlpacaClient(paper=True)
            except Exception as e:
                logger.warning(f"Alpaca init failed: {type(e).__name__}: {e}")

        # Email sender optional
        self.email_sender = None
        if EMAIL_AVAILABLE and os.getenv("GMAIL_APP_PASSWORD"):
            try:
                self.email_sender = EmailSender()
            except Exception as e:
                logger.warning(f"Email sender init failed: {type(e).__name__}: {e}")

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
            except Exception as e:
                logger.warning(f"Portfolio tracker init failed: {type(e).__name__}: {e}")

        # TradingAgents config for deep analysis
        if TRADINGAGENTS_AVAILABLE:
            self.ta_config = DEFAULT_CONFIG.copy()
            llm_cfg = self.config.get("llm", {})
            deep_cfg = self.config.get("deep_analysis", {})
            self.ta_config["llm_provider"] = llm_cfg.get("provider", "nvidia")
            self.ta_config["deep_think_llm"] = llm_cfg.get("deep_think_model", "meta/llama-3.1-70b-instruct")
            self.ta_config["quick_think_llm"] = llm_cfg.get("quick_think_model", "meta/llama-3.1-8b-instruct")
            self.ta_config["temperature"] = llm_cfg.get("temperature", 0)
            # Deeper debates for better signal quality
            self.ta_config["max_debate_rounds"] = deep_cfg.get("max_debate_rounds", 3)
            self.ta_config["max_risk_discuss_rounds"] = deep_cfg.get("max_risk_discuss_rounds", 3)
            self.ta_config["memory_log_max_entries"] = deep_cfg.get("memory_log_max_entries", 50)
        else:
            self.ta_config = {}
            logger.warning("TradingAgents not available - deep analysis disabled")

        # Deep analysis settings
        self.deep_count = self.config.get("deep_analysis", {}).get("count", 20)
        self.min_conviction = self.config.get("deep_analysis", {}).get("min_conviction", 0.3)

        # Execution settings
        self.execution_enabled = self.config.get("execution", {}).get("enabled", False)

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
        failed_count = 0

        def fetch_one(ticker):
            try:
                return ticker, self.twelve_data.get_price(ticker)
            except Exception:
                return ticker, 0

        num_keys = len(self.twelve_data._api_keys)
        max_workers = min(num_keys, 8)  # match parallelism to available keys

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(fetch_one, t) for t in tickers]
            for f in futures:
                try:
                    ticker, price = f.result(timeout=30)
                except (TimeoutError, Exception):
                    failed_count += 1
                    continue
                if price > 0:
                    prices[ticker] = price
                else:
                    failed_count += 1

        if failed_count > 0:
            logger.warning(f"Failed to get prices for {failed_count}/{len(tickers)} tickers")
        return prices

    def batch_get_indicators(self, tickers: list[str]) -> dict[str, dict]:
        """Fetch indicators for multiple tickers concurrently."""
        indicators = {}
        failed_count = 0

        def fetch_one(ticker):
            try:
                rsi = self.twelve_data.get_rsi(ticker)
                macd = self.twelve_data.get_macd(ticker)
                result = {}
                if rsi is not None:
                    result["rsi"] = rsi
                if macd is not None:
                    result["macd"] = macd
                return ticker, result
            except Exception:
                return ticker, {}

        num_keys = len(self.twelve_data._api_keys)
        max_workers = min(num_keys, 8)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(fetch_one, t) for t in tickers]
            for f in futures:
                try:
                    ticker, ind = f.result(timeout=30)
                except (TimeoutError, Exception):
                    failed_count += 1
                    continue
                indicators[ticker] = ind
                if not ind:
                    failed_count += 1

        if failed_count > 0:
            logger.warning(f"Failed to get indicators for {failed_count}/{len(tickers)} tickers")
        return indicators

    def quick_scan(self, ticker: str, price: float, indicators: dict) -> dict:
        """Fast single-call scan to identify candidates."""
        rsi = indicators.get("rsi", 50)
        macd = indicators.get("macd", {})
        macd_hist = macd.get("histogram", 0) if macd else 0

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
            if not TRADINGAGENTS_AVAILABLE or not LANGCHAIN_AVAILABLE:
                raise ImportError("TradingAgents or langchain_core not available")

            from tradingagents.llm_clients import create_llm_client
            client = create_llm_client(
                provider=self.ta_config["llm_provider"],
                model=self.ta_config["quick_think_llm"],
            )
            llm = client.get_llm()
            response = llm.invoke([HumanMessage(content=prompt)])
            text = response.content.strip()

            match = re.search(r'(\d+\.?\d*)', text)
            score = float(match.group(1)) if match else 0.0
            # Detect percentage vs decimal: if > 1.0, assume percentage
            if score > 1.0:
                score = score / 100.0
            score = max(0.0, min(1.0, score))

            return {"ticker": ticker, "score": score, "price": price, "indicators": indicators}

        except Exception as e:
            logger.error(f"Quick scan failed for {ticker}: {e}")
            return {"ticker": ticker, "score": 0.0, "price": price, "indicators": indicators}

    async def phase1_scan(self, tickers: list[str], prices: dict, indicators: dict, max_candidates: int = 20) -> tuple[list[dict], float]:
        """Phase 1: Quick scan all stocks to find candidates. Returns (candidates, elapsed)."""
        logger.info(f"Phase 1: Scanning {len(tickers)} stocks...")
        start = time.time()

        # RPM limit for NVIDIA NIM is 40. Use concurrency=1 + 2s delay for Phase 1 too.
        semaphore = asyncio.Semaphore(1)

        async def scan_one(ticker):
            async with semaphore:
                # Delay BEFORE request to respect RPM limit
                await asyncio.sleep(2.0)
                price = prices.get(ticker, 0)
                ind = indicators.get(ticker, {})
                if price <= 0:
                    return {"ticker": ticker, "score": 0, "price": 0, "indicators": {}}
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, self.quick_scan, ticker, price, ind)

        tasks = [scan_one(t) for t in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        results = [r for r in results if not isinstance(r, Exception)]

        elapsed = time.time() - start
        logger.info(f"Phase 1 complete in {elapsed:.1f}s")

        scored = sorted(results, key=lambda x: x["score"], reverse=True)
        candidates = [s for s in scored if s["score"] >= self.min_conviction][:max_candidates]

        logger.info(f"Found {len(candidates)} candidates (score >= {self.min_conviction})")
        for c in candidates:
            logger.info(f"  {c['ticker']}: score={c['score']:.2f}, price=${c['price']:.2f}")

        return candidates, elapsed

    def _extract_conviction_and_allocation(self, decision: str, action: str) -> tuple[float, Optional[float]]:
        """Extract conviction and allocation % from model decision text.

        Model MUST specify allocation. If not found, returns None and trade is skipped.

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
                if any(w in lower for w in ["strong buy", "strong buy signal", "high conviction buy"]):
                    conviction = 0.9
                elif any(w in lower for w in ["buy", "overweight", "accumulate"]):
                    conviction = 0.75
                elif any(w in lower for w in ["weak buy", "slight buy", "marginal"]):
                    conviction = 0.55
                elif any(w in lower for w in ["strong sell", "strong sell signal", "high conviction sell"]):
                    conviction = 0.9
                elif any(w in lower for w in ["sell", "underweight", "reduce"]):
                    conviction = 0.75
                elif any(w in lower for w in ["weak sell", "slight sell"]):
                    conviction = 0.55

        # Extract allocation - model MUST provide this
        alloc_match = re.search(r'allocat(?:ion|e)[:\s]+(\d+\.?\d*)\s*%', decision, re.IGNORECASE)
        if not alloc_match:
            alloc_match = re.search(r'allocat(?:ion|e)\s+of\s+(\d+\.?\d*)\s*%', decision, re.IGNORECASE)
        if alloc_match:
            allocation = float(alloc_match.group(1))
        else:
            pos_match = re.search(r'position\s*(?:size)?[:\s]+(\d+\.?\d*)\s*%', decision, re.IGNORECASE)
            if pos_match:
                allocation = float(pos_match.group(1))
            else:
                sizing_match = re.search(r'sizing[:\s]+(\d+\.?\d*)\s*%', decision, re.IGNORECASE)
                if sizing_match:
                    allocation = float(sizing_match.group(1))
                else:
                    portfolio_match = re.search(r'(\d+\.?\d*)\s*%\s*(?:of\s+)?(?:portfolio|capital|equity)', decision, re.IGNORECASE)
                    if portfolio_match:
                        allocation = float(portfolio_match.group(1))
                    else:
                        stake_match = re.search(r'(?:put|invest|stake|size)[:\s]+(\d+\.?\d*)\s*%', decision, re.IGNORECASE)
                        if stake_match:
                            allocation = float(stake_match.group(1))

        return conviction, allocation

    def _force_allocation(self, ticker: str, decision: str, action: str) -> tuple[float, Optional[float]]:
        """Force model to output conviction + allocation via follow-up LLM call.

        Returns (conviction, allocation_pct). Returns None for allocation on failure
        so the trade is skipped rather than risking with defaults.
        """
        if not LANGCHAIN_AVAILABLE:
            logger.error("langchain_core not available for force_allocation")
            return 0.0, None

        if not TRADINGAGENTS_AVAILABLE:
            logger.error("tradingagents not available for force_allocation")
            return 0.0, None

        # Keep the conclusion section (last 500 chars) intact
        truncated = decision[-2000:] if len(decision) > 2000 else decision

        prompt = f"""You are a portfolio manager. Based on this analysis for {ticker}, give your final conviction and portfolio allocation.

ANALYSIS:
{truncated}

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

            if not conv_match or not alloc_match:
                logger.warning(f"Force allocation parse failed for {ticker}: {text[:200]}")
                return 0.0, None

            conviction = float(conv_match.group(1))
            conviction = max(0.0, min(1.0, conviction))

            allocation = float(alloc_match.group(1))
            allocation = max(0.0, min(25.0, allocation))

            logger.info(f"Forced allocation for {ticker}: conviction={conviction:.2f}, allocation={allocation:.1f}%")
            return conviction, allocation

        except Exception as e:
            logger.error(f"Force allocation failed for {ticker}: {e}")
            return 0.0, None

    def _detect_action(self, decision: str) -> str:
        """Detect action from LLM decision text, handling negation.

        Returns "buy", "sell", or "hold".
        """
        lower = decision.lower()

        # Check for negation patterns first
        negation_patterns = [
            r"don['']?t\s+buy",
            r"do\s+not\s+buy",
            r"avoid\s+buy",
            r"shouldn['']?t\s+buy",
            r"should\s+not\s+buy",
            r"not\s+a\s+buy",
            r"not\s+a\s+strong\s+buy",
            r"no\s+buy\s+signal",
            r"not\s+recommend.*buy",
            r"wouldn['']?t\s+buy",
            r"would\s+not\s+buy",
            r"don['']?t\s+sell",
            r"do\s+not\s+sell",
            r"avoid\s+sell",
            r"shouldn['']?t\s+sell",
            r"should\s+not\s+sell",
            r"not\s+a\s+sell",
            r"not\s+a\s+strong\s+sell",
            r"no\s+sell\s+signal",
            r"not\s+recommend.*sell",
            r"wouldn['']?t\s+sell",
            r"would\s+not\s+sell",
        ]

        has_buy_negation = any(re.search(p, lower) for p in negation_patterns[:11])
        has_sell_negation = any(re.search(p, lower) for p in negation_patterns[11:])

        # Strong signals
        strong_buy = any(w in lower for w in ["strong buy", "strong buy signal", "high conviction buy", "overweight", "accumulate"])
        strong_sell = any(w in lower for w in ["strong sell", "strong sell signal", "high conviction sell", "underweight"])

        # Check conclusion (last 200 chars)
        conclusion = lower[-200:] if len(lower) > 200 else lower
        conclusion_has_buy = any(w in conclusion for w in ["buy", "buy signal", "go long", "long position"])
        conclusion_has_sell = any(w in conclusion for w in ["sell", "sell signal", "go short", "short position"])

        # General mentions (anywhere in text) — use specific phrases, not bare "buy"
        mention_buy = any(w in lower for w in ["buy signal", "go long", "long position", "accumulate", "buy recommendation"])
        mention_sell = any(w in lower for w in ["sell signal", "go short", "short position", "reduce position", "sell recommendation"])

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
                    forced_conviction, forced_allocation = self._force_allocation(ticker, decision or "", action)
                    if forced_allocation is not None:
                        conviction = forced_conviction
                        allocation = forced_allocation
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
                "reasoning": f"Deep analysis failed: {type(e).__name__}",
                "mode": "deep",
            }

    async def phase2_deep(self, candidates: list[dict]) -> list[dict]:
        """Phase 2: Deep analysis on top candidates (concurrent with throttling)."""
        if not candidates:
            return []

        logger.info(f"Phase 2: Deep analysis on {len(candidates)} candidates...")
        start = time.time()

        # RPM limit for NVIDIA NIM is 40. Use concurrency=1 + 2s delay to stay well under.
        semaphore = asyncio.Semaphore(1)

        async def analyze_one(c):
            async with semaphore:
                # Delay BEFORE request to respect RPM limit
                await asyncio.sleep(2.0)
                logger.info(f"  Analyzing {c['ticker']}...")
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None, self.deep_analysis, c["ticker"], datetime.now().strftime("%Y-%m-%d")
                )
                result["scan_score"] = c["score"]
                result["price"] = c["price"]
                result["indicators"] = c["indicators"]
                return result

        tasks = [analyze_one(c) for c in candidates]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        results = [r for r in results if not isinstance(r, Exception)]

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

        # Check market and get account state - wrap to avoid destroying analysis results on API failure
        try:
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
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca API, skipping execution: {e}")
            return []

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
                        order_id = order.get("id")
                        order_status = str(order.get("status", "")).lower()
                        if order_status in ("filled", "accepted", "new", "pending_new", "partially_filled"):
                            # Wait for fill verification
                            time.sleep(2.0)
                            verified = self.alpaca.get_order_by_id(order_id) if order_id else None
                            if verified:
                                verified_status = str(verified.get("status", "")).lower()
                                fill_price = verified.get("filled_avg_price")
                                if verified_status == "filled" and fill_price:
                                    result["status"] = "filled"
                                    result["fill_price"] = float(fill_price)
                                    result["qty"] = int(float(verified.get("qty", sizing["qty"])))
                                    logger.info(f"BUY {ticker}: {result['qty']} shares @ ${float(fill_price):.2f} (FILLED)")
                                else:
                                    result["status"] = "submitted"
                                    result["qty"] = sizing["qty"]
                                    logger.info(f"BUY {ticker}: {sizing['qty']} shares @ ${price:.2f} (status={verified_status})")
                            else:
                                result["status"] = "submitted"
                                result["qty"] = sizing["qty"]
                                logger.info(f"BUY {ticker}: {sizing['qty']} shares @ ${price:.2f} (status={order_status})")
                            result["order_id"] = order_id
                            result["reasoning"] = sizing["reasoning"]
                            self.risk_manager.record_trade(ticker, "buy", sizing["qty"], price, sizing["reasoning"])
                        else:
                            result["status"] = "error"
                            result["reasoning"] = f"Order rejected: status={order_status}"

                # CASE 2: Buy signal + short position -> Close short, open long
                elif action == "buy" and existing_side == "short":
                    # Close short first
                    qty_to_close = max(1, int(abs(float(existing_pos["qty"]))))
                    entry_price = existing_pos["avg_entry_price"]
                    close_order = self.alpaca.market_buy(ticker, qty=qty_to_close)
                    close_status = close_order.get("status", "")
                    if close_status in ("filled", "accepted", "new", "partially_filled"):
                        fill_price = float(close_order.get("filled_avg_price") or price)
                        pnl = (entry_price - fill_price) * qty_to_close
                        self.risk_manager.update_daily_pnl(pnl)
                        self.risk_manager.record_trade(ticker, "close_short", qty_to_close, fill_price, "Closing short before long")
                        logger.info(f"CLOSE SHORT {ticker}: {qty_to_close} shares, P&L: ${pnl:+,.2f}")
                    else:
                        logger.error(f"Failed to close short {ticker}: status={close_status}")
                        result["status"] = "error"
                        result["reasoning"] = f"Short close failed: {close_status}"
                        continue

                    # Re-fetch portfolio state after closing short
                    try:
                        account = self.alpaca.get_account()
                        portfolio_value = account["portfolio_value"]
                        buying_power = account["buying_power"]
                        current_positions = self.alpaca.get_positions()
                        pos_by_ticker = {p["ticker"]: p for p in current_positions}
                    except Exception as e:
                        logger.warning(f"Failed to refresh state after short close: {e}")

                    # Open long
                    sizing = self.risk_manager.calculate_position_size(
                        ticker, price, conviction, portfolio_value,
                        current_positions, allocation_pct, buying_power
                    )
                    if sizing["action"] != "skip":
                        order = self.alpaca.market_buy(ticker, qty=sizing["qty"])
                        order_status = str(order.get("status", "")).lower()
                        if order_status in ("filled", "accepted", "new", "pending_new", "partially_filled"):
                            result["status"] = "filled" if order_status == "filled" else "submitted"
                            result["qty"] = sizing["qty"]
                            result["order_id"] = order.get("id")
                            result["reasoning"] = f"Closed short (P&L: ${pnl:+,.2f}) + {sizing['reasoning']}"
                            self.risk_manager.record_trade(ticker, "buy", sizing["qty"], price, sizing["reasoning"])
                            logger.info(f"BUY {ticker}: {sizing['qty']} shares @ ${price:.2f} (status={order_status})")
                        else:
                            result["status"] = "partial"
                            result["reasoning"] = f"Closed short (P&L: ${pnl:+,.2f}), long order rejected: {order_status}"
                    else:
                        result["status"] = "partial"
                        result["reasoning"] = f"Closed short (P&L: ${pnl:+,.2f}), model did not open replacement long"

                # CASE 3: Sell signal + long position -> Close long
                elif action == "sell" and existing_side == "long":
                    qty_to_close = max(1, int(abs(float(existing_pos["qty"]))))
                    entry_price = existing_pos["avg_entry_price"]
                    close_order = self.alpaca.market_sell(ticker, qty=qty_to_close)
                    close_status = close_order.get("status", "")
                    if close_status in ("filled", "accepted", "new", "partially_filled"):
                        fill_price = float(close_order.get("filled_avg_price") or price)
                        pnl = (fill_price - entry_price) * qty_to_close
                        self.risk_manager.update_daily_pnl(pnl)
                        self.risk_manager.record_trade(ticker, "sell", qty_to_close, fill_price, "Closing long on sell signal")
                        result["status"] = "filled"
                        result["qty"] = qty_to_close
                        result["reasoning"] = f"Closed long: {qty_to_close} shares, P&L: ${pnl:+,.2f}"
                        logger.info(f"SELL (close long) {ticker}: {qty_to_close} shares @ ${fill_price:.2f}, P&L: ${pnl:+,.2f}")
                    else:
                        logger.error(f"Failed to close long {ticker}: status={close_status}")
                        result["status"] = "error"
                        result["reasoning"] = f"Long close failed: {close_status}"

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
                        order_id = order.get("id")
                        order_status = str(order.get("status", "")).lower()
                        if order_status in ("filled", "accepted", "new", "pending_new", "partially_filled"):
                            # Wait for fill verification
                            time.sleep(2.0)
                            verified = self.alpaca.get_order_by_id(order_id) if order_id else None
                            if verified:
                                verified_status = str(verified.get("status", "")).lower()
                                fill_price = verified.get("filled_avg_price")
                                if verified_status == "filled" and fill_price:
                                    result["status"] = "filled"
                                    result["fill_price"] = float(fill_price)
                                    result["qty"] = int(float(verified.get("qty", sizing["qty"])))
                                    logger.info(f"SHORT {ticker}: {result['qty']} shares @ ${float(fill_price):.2f} (FILLED)")
                                else:
                                    result["status"] = "submitted"
                                    result["qty"] = sizing["qty"]
                                    logger.info(f"SHORT {ticker}: {sizing['qty']} shares @ ${price:.2f} (status={verified_status})")
                            else:
                                result["status"] = "submitted"
                                result["qty"] = sizing["qty"]
                                logger.info(f"SHORT {ticker}: {sizing['qty']} shares @ ${price:.2f} (status={order_status})")
                            result["order_id"] = order_id
                            result["reasoning"] = sizing["reasoning"]
                            self.risk_manager.record_trade(ticker, "short", sizing["qty"], price, sizing["reasoning"])
                        else:
                            result["status"] = "error"
                            result["reasoning"] = f"Order rejected: status={order_status}"

                # CASE 5: Sell signal + short position -> Hold short
                elif action == "sell" and existing_side == "short":
                    result["status"] = "no_action"
                    result["reasoning"] = "Already short, model confirms hold"

                # CASE 6: Buy signal + long position -> Hold long
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

            # Re-fetch state after a filled or submitted trade to prevent stale data
            if result["status"] in ("filled", "submitted"):
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
        submitted = [r for r in execution_results if r["status"] == "submitted"]
        placed = filled + submitted  # Both are successful placements
        skipped = [r for r in execution_results if r["status"] == "skipped"]
        errors = [r for r in execution_results if r["status"] == "error"]
        circuit_breakers = [r for r in execution_results if r["status"] == "circuit_breaker"]

        logger.info(f"Execution complete: {len(placed)} placed ({len(filled)} filled, {len(submitted)} pending), "
                     f"{len(skipped)} skipped, {len(errors)} errors"
                     + (f", {len(circuit_breakers)} circuit breaker" if circuit_breakers else ""))

        return execution_results

    async def run_two_phase(
        self, tickers: Optional[list[str]] = None, deep_count: int = None
    ) -> dict:
        """Run the full two-phase analysis + execution."""
        # Use local variable instead of mutating instance state
        effective_deep_count = deep_count if deep_count is not None else self.deep_count

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
        candidates, phase1_elapsed = await self.phase1_scan(tickers, prices, indicators, max_candidates=effective_deep_count)

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
                "elapsed_seconds": round(phase1_elapsed, 1),
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
                "submitted": len([r for r in execution_results if r.get("status") == "submitted"]),
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
        logger.info(f"Phase 1: {len(tickers)} scanned -> {len(candidates)} candidates ({phase1_elapsed:.1f}s)")
        logger.info(f"Phase 2: {len(deep_results)} analyzed")
        logger.info(f"  Buy: {len(buy_signals)}, Sell: {len(sell_signals)}, Hold: {len(hold_signals)}")
        if execution_results:
            logger.info(f"  Execution: {summary['execution']['filled']} filled, {summary['execution']['skipped']} skipped, {summary['execution']['errors']} errors")

        # Show buy signals
        if buy_signals:
            logger.info("\nBUY SIGNALS:")
            for b in sorted(buy_signals, key=lambda x: x["conviction"], reverse=True):
                reasoning = (b.get("reasoning") or "")[:100]
                logger.info(f"  {b['ticker']}: ${b['price']:.2f} (conviction: {b['conviction']:.2f})")
                logger.info(f"    {reasoning}...")

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

    args = parser.parse_args()

    bot = TwoPhaseBot()
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
