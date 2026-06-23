"""Scheduler for daily trading bot execution."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from bot import TradingBot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(Path(__file__).parent / "scheduler.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def run_scheduled():
    """Run the scheduled trading cycle."""
    logger.info("=" * 60)
    logger.info(f"Scheduled run started at {datetime.now()}")

    try:
        bot = TradingBot()
        result = bot.run_daily()

        logger.info(f"Scheduled run completed: {result}")
        return result

    except Exception as e:
        logger.error(f"Scheduled run failed: {e}")
        raise


def run_single(ticker: str, date: str = None):
    """Run analysis for a single ticker."""
    logger.info(f"Single analysis: {ticker}")

    try:
        bot = TradingBot()
        result = bot.run_analysis(ticker, date)

        logger.info(f"Analysis complete: {result}")
        return result

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


def check_status():
    """Check bot status."""
    try:
        bot = TradingBot()
        status = bot.get_status()

        logger.info("Bot Status:")
        logger.info(f"  Portfolio Value: ${status['portfolio_value']:,.2f}")
        logger.info(f"  Cash: ${status['cash']:,.2f}")
        logger.info(f"  Positions: {status['position_count']}")
        logger.info(f"  Market Open: {status['market_open']}")

        return status

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "daily":
            run_scheduled()
        elif command == "analyze" and len(sys.argv) > 2:
            ticker = sys.argv[2]
            date = sys.argv[3] if len(sys.argv) > 3 else None
            run_single(ticker, date)
        elif command == "status":
            check_status()
        else:
            print("Usage:")
            print("  python scheduler.py daily        - Run daily cycle")
            print("  python scheduler.py analyze AAPL - Analyze single ticker")
            print("  python scheduler.py status       - Check status")
    else:
        print("Usage:")
        print("  python scheduler.py daily        - Run daily cycle")
        print("  python scheduler.py analyze AAPL - Analyze single ticker")
        print("  python scheduler.py status       - Check status")
