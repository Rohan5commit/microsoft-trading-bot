"""Two-phase scheduler."""

import asyncio
import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent))

from two_phase_bot import TwoPhaseBot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(Path(__file__).parent / "scheduler.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def run_two_phase(tickers=None, deep_count=20, concurrent=20):
    """Run two-phase analysis."""
    logger.info("=" * 60)
    logger.info(f"Two-phase run started at {datetime.now()}")

    bot = None
    try:
        bot = TwoPhaseBot(max_concurrent=concurrent)
        results = asyncio.run(bot.run_two_phase(tickers=tickers, deep_count=deep_count))

        logger.info(f"Two-phase run completed: {results['total_elapsed_seconds']:.1f}s")
        return results

    except Exception as e:
        error_msg = f"Bot crashed: {str(e)}"
        tb_str = traceback.format_exc()
        logger.error(error_msg)
        logger.error(tb_str)

        # Send error notification email
        if bot and bot.email_sender:
            try:
                bot.send_error_email(error_msg, tb_str)
            except Exception as email_err:
                logger.error(f"Failed to send error email: {email_err}")

        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Two-Phase Scheduler")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers")
    parser.add_argument("--deep-count", type=int, default=20)
    parser.add_argument("--concurrent", type=int, default=20)

    args = parser.parse_args()

    run_two_phase(
        tickers=args.tickers,
        deep_count=args.deep_count,
        concurrent=args.concurrent,
    )
