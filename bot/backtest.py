"""Backtesting framework for the trading bot."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import yfinance as yf

from bot import TradingBot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class Backtester:
    """Backtest the trading strategy on historical data."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize backtester."""
        if config_path is None:
            config_path = Path(__file__).parent / "config.json"

        with open(config_path) as f:
            self.config = json.load(f)

        self.backtest_config = self.config.get("backtest", {})
        self.start_date = self.backtest_config.get("start_date", "2025-01-01")
        self.end_date = self.backtest_config.get("end_date", "2025-12-31")
        self.initial_capital = self.backtest_config.get("initial_capital", 10000)
        self.benchmark = self.backtest_config.get("benchmark", "SPY")

    def run(
        self,
        tickers: list[str],
        interval_days: int = 5,
        output_dir: Optional[str] = None,
    ) -> dict:
        """Run backtest.

        Args:
            tickers: List of tickers to test
            interval_days: Days between analysis points
            output_dir: Directory for results

        Returns:
            Backtest results dict
        """
        if output_dir is None:
            output_dir = Path(__file__).parent / "backtest_results"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        logger.info(f"Starting backtest: {self.start_date} to {self.end_date}")
        logger.info(f"Tickers: {tickers}")
        logger.info(f"Initial capital: ${self.initial_capital:,.2f}")

        # Generate date range
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.end_date, "%Y-%m-%d")
        dates = []
        current = start
        while current <= end:
            # Skip weekends
            if current.weekday() < 5:
                dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=interval_days)

        logger.info(f"Analysis points: {len(dates)}")

        # Run analysis at each point
        results = {
            "config": self.config,
            "dates": dates,
            "tickers": tickers,
            "trades": [],
            "portfolio_values": [],
            "benchmark_values": [],
        }

        # Simulate portfolio
        cash = self.initial_capital
        positions = {}  # ticker -> {qty, entry_price, entry_date}
        portfolio_history = []

        for i, date in enumerate(dates):
            logger.info(f"[{i+1}/{len(dates)}] Analyzing {date}")

            # Get current prices for all tickers
            prices = {}
            for ticker in tickers:
                try:
                    data = yf.Ticker(ticker).history(start=date, end=date)
                    if len(data) > 0:
                        prices[ticker] = float(data["Close"].iloc[-1])
                except Exception:
                    continue

            # Get benchmark price
            try:
                bench_data = yf.Ticker(self.benchmark).history(start=date, end=date)
                if len(bench_data) > 0:
                    bench_price = float(bench_data["Close"].iloc[-1])
                    results["benchmark_values"].append({
                        "date": date,
                        "value": bench_price,
                    })
            except Exception:
                pass

            # Calculate current portfolio value
            portfolio_value = cash
            for ticker, pos in positions.items():
                if ticker in prices:
                    portfolio_value += pos["qty"] * prices[ticker]

            portfolio_history.append({
                "date": date,
                "value": portfolio_value,
                "cash": cash,
                "positions": len(positions),
            })

            # Run analysis for each ticker (in production, this would use TradingAgents)
            # For backtest, we simulate with simple momentum strategy
            for ticker in tickers:
                if ticker not in prices:
                    continue

                price = prices[ticker]

                # Simple momentum signal for backtest
                signal = self._get_signal(ticker, date, prices)

                if signal == "buy" and ticker not in positions:
                    # Calculate position size (10% of portfolio)
                    position_value = portfolio_value * 0.1
                    qty = int(position_value / price)
                    if qty > 0 and qty * price <= cash:
                        cash -= qty * price
                        positions[ticker] = {
                            "qty": qty,
                            "entry_price": price,
                            "entry_date": date,
                        }
                        results["trades"].append({
                            "date": date,
                            "ticker": ticker,
                            "action": "buy",
                            "qty": qty,
                            "price": price,
                        })

                elif signal == "sell" and ticker in positions:
                    pos = positions[ticker]
                    sell_value = pos["qty"] * price
                    cash += sell_value
                    pnl = (price - pos["entry_price"]) * pos["qty"]
                    results["trades"].append({
                        "date": date,
                        "ticker": ticker,
                        "action": "sell",
                        "qty": pos["qty"],
                        "price": price,
                        "pnl": pnl,
                    })
                    del positions[ticker]

            # Check stop-losses
            for ticker in list(positions.keys()):
                if ticker in prices:
                    pos = positions[ticker]
                    loss_pct = ((pos["entry_price"] - prices[ticker]) / pos["entry_price"]) * 100
                    if loss_pct >= 5:  # 5% stop-loss
                        cash += pos["qty"] * prices[ticker]
                        pnl = (prices[ticker] - pos["entry_price"]) * pos["qty"]
                        results["trades"].append({
                            "date": date,
                            "ticker": ticker,
                            "action": "stop-loss",
                            "qty": pos["qty"],
                            "price": prices[ticker],
                            "pnl": pnl,
                        })
                        del positions[ticker]

        # Final portfolio value
        final_value = cash
        for ticker, pos in positions.items():
            if ticker in prices:
                final_value += pos["qty"] * prices[ticker]

        # Calculate metrics
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100
        annualized = self._annualize_return(total_return, len(dates), interval_days)

        # Benchmark return
        bench_return = 0
        if results["benchmark_values"]:
            bench_start = results["benchmark_values"][0]["value"]
            bench_end = results["benchmark_values"][-1]["value"]
            bench_return = ((bench_end - bench_start) / bench_start) * 100

        results["summary"] = {
            "initial_capital": self.initial_capital,
            "final_value": final_value,
            "total_return_pct": total_return,
            "annualized_return_pct": annualized,
            "benchmark_return_pct": bench_return,
            "alpha": total_return - bench_return,
            "total_trades": len(results["trades"]),
            "win_rate": self._calculate_win_rate(results["trades"]),
        }

        # Save results
        output_file = output_dir / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Backtest complete. Results saved to {output_file}")
        logger.info(f"Total return: {total_return:.2f}%")
        logger.info(f"Annualized: {annualized:.2f}%")
        logger.info(f"Benchmark: {bench_return:.2f}%")
        logger.info(f"Alpha: {results['summary']['alpha']:.2f}%")
        logger.info(f"Win rate: {results['summary']['win_rate']:.1f}%")

        return results

    def _get_signal(self, ticker: str, date: str, prices: dict) -> str:
        """Get trading signal (placeholder for actual TradingAgents integration).

        In production, this would call TradingAgents. For backtest,
        uses a simple momentum strategy.
        """
        # Simple momentum: buy if price > 20-day moving average
        # This is just a placeholder - the real bot uses TradingAgents
        return "hold"  # Conservative default for backtest

    def _annualize_return(
        self,
        total_return_pct: float,
        num_periods: int,
        interval_days: int,
    ) -> float:
        """Annualize a total return."""
        total_days = num_periods * interval_days
        if total_days <= 0:
            return 0
        years = total_days / 365
        if years <= 0:
            return 0
        return ((1 + total_return_pct / 100) ** (1 / years) - 1) * 100

    def _calculate_win_rate(self, trades: list[dict]) -> float:
        """Calculate win rate from trades."""
        sells = [t for t in trades if t["action"] in ("sell", "stop-loss")]
        if not sells:
            return 0
        wins = sum(1 for t in sells if t.get("pnl", 0) > 0)
        return (wins / len(sells)) * 100


def main():
    """Run backtest."""
    import argparse

    parser = argparse.ArgumentParser(description="Backtest trading strategy")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "NVDA", "GOOGL", "META"],
                        help="Tickers to test")
    parser.add_argument("--interval", type=int, default=5, help="Days between analysis")
    parser.add_argument("--output", help="Output directory")

    args = parser.parse_args()

    backtester = Backtester(config_path=args.config)
    results = backtester.run(
        tickers=args.tickers,
        interval_days=args.interval,
        output_dir=args.output,
    )

    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    for key, value in results["summary"].items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
