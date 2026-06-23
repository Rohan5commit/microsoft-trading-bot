"""Fast scheduler - uses async parallel analysis."""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from fast_bot import FastTradingBot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(Path(__file__).parent / "scheduler.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def run_fast_daily(tickers=None, mode="quick", concurrent=20, execute=False):
    """Run fast daily analysis."""
    logger.info("=" * 60)
    logger.info(f"Fast daily run started at {datetime.now()}")

    bot = FastTradingBot(max_concurrent=concurrent)
    results = asyncio.run(bot.run_fast(tickers=tickers, mode=mode))

    if execute:
        exec_result = bot.execute_decisions(results)
        logger.info(f"Executed {exec_result['executed']} trades")

    logger.info(f"Fast daily run completed: {results['elapsed_seconds']}s")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fast Scheduler")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--concurrent", type=int, default=20)
    parser.add_argument("--execute", action="store_true")

    args = parser.parse_args()

    run_fast_daily(
        tickers=args.tickers,
        mode=args.mode,
        concurrent=args.concurrent,
        execute=args.execute,
    )
