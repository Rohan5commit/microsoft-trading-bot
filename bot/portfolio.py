"""Portfolio return tracking with leverage-adjusted metrics.

Tracks cumulative returns against a leveraged exposure base,
with support for carry-forward returns across resets.
"""

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class PortfolioTracker:
    """Track portfolio returns with leverage-adjusted calculations.

    Core parameters:
        initial_capital (C0): Starting cash balance (e.g., $100,000)
        leverage (L): Leverage multiplier (e.g., 2.0 for 2x)
        previous_return_pct (R_prev): Cumulative return from prior periods (decimal)
        current_equity (Et): Live total equity at time t

    Mathematical formulas:
        B = C0 * L (Exposure Divisor / Leveraged Notional Base)
        E_reset = C0 + (R_prev * B) (Reset Baseline Equity)
        current_return_pct = (Et - E_reset) / B (Period Return)
        cumulative_return_pct = R_prev + current_return_pct (Total Return)
    """

    def __init__(self, state_file: Optional[str] = None):
        if state_file is None:
            state_file = str(Path(__file__).parent / "portfolio_state.json")
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self) -> dict:
        """Load persisted portfolio state with corruption recovery."""
        defaults = {
            "initial_capital": 100000.0,
            "leverage": 2.0,
            "previous_return_pct": 0.0,
            "reset_count": 0,
            "last_reset_date": None,
        }
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file) as f:
                    loaded = json.load(f)
                    defaults.update(loaded)
                    return defaults
        except (json.JSONDecodeError, ValueError):
            # Corrupted file - back up and start fresh
            logger.warning(f"Corrupted portfolio state, backing up: {self.state_file}")
            backup = self.state_file + ".bak"
            try:
                os.rename(self.state_file, backup)
            except OSError:
                pass
        return defaults

    def _save_state(self):
        """Atomic write: write to temp file then rename."""
        dir_path = os.path.dirname(self.state_file) or "."
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(self.state, f, indent=2)
            os.replace(tmp_path, self.state_file)
        except Exception:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            raise

    @property
    def initial_capital(self) -> float:
        return self.state["initial_capital"]

    @initial_capital.setter
    def initial_capital(self, value: float):
        self.state["initial_capital"] = value
        self._save_state()

    @property
    def leverage(self) -> float:
        return self.state["leverage"]

    @leverage.setter
    def leverage(self, value: float):
        self.state["leverage"] = value
        self._save_state()

    @property
    def previous_return_pct(self) -> float:
        return self.state["previous_return_pct"]

    def get_exposure_divisor(self) -> float:
        """B = C0 * L (Leveraged Notional Base)."""
        return self.initial_capital * self.leverage

    def get_reset_baseline(self) -> float:
        """E_reset = C0 + (R_prev * B)."""
        B = self.get_exposure_divisor()
        return self.initial_capital + (self.previous_return_pct * B)

    def calculate_period_return(self, current_equity: float) -> dict:
        """Calculate return metrics for current period.

        Args:
            current_equity: Live total equity (cash + positions market value)

        Returns:
            dict with all return metrics
        """
        C0 = self.initial_capital
        L = self.leverage
        B = self.get_exposure_divisor()
        E_reset = self.get_reset_baseline()

        # Period return: (Et - E_reset) / B
        period_return = (current_equity - E_reset) / B if B != 0 else 0.0

        # Cumulative return: R_prev + period_return
        cumulative_return = self.previous_return_pct + period_return

        # Dollar P&L from reset baseline
        dollar_pnl = current_equity - E_reset

        # Dollar P&L from initial capital
        total_dollar_pnl = current_equity - C0

        return {
            "initial_capital": C0,
            "leverage": L,
            "exposure_divisor": B,
            "reset_baseline": E_reset,
            "current_equity": current_equity,
            "period_return_pct": round(period_return * 100, 4),
            "cumulative_return_pct": round(cumulative_return * 100, 4),
            "period_return_decimal": round(period_return, 6),
            "cumulative_return_decimal": round(cumulative_return, 6),
            "dollar_pnl": round(dollar_pnl, 2),
            "total_dollar_pnl": round(total_dollar_pnl, 2),
            "previous_return_pct": round(self.previous_return_pct * 100, 4),
        }

    def reset(self, current_equity: float):
        """Reset the baseline, carrying forward the cumulative return.

        Called periodically (e.g., weekly/monthly) to lock in gains
        and start fresh from the current equity level.
        """
        metrics = self.calculate_period_return(current_equity)

        self.state["previous_return_pct"] = metrics["cumulative_return_decimal"]
        self.state["reset_count"] = self.state.get("reset_count", 0) + 1
        self.state["last_reset_date"] = datetime.now().isoformat()
        self._save_state()

        logger.info(f"Portfolio reset #{self.state['reset_count']}: "
                    f"equity=${current_equity:,.2f}, cumulative={metrics['cumulative_return_pct']:+.2f}%")

        return metrics

    def get_status(self, current_equity: float) -> dict:
        """Get full portfolio status for email reporting."""
        metrics = self.calculate_period_return(current_equity)
        metrics["reset_count"] = self.state.get("reset_count", 0)
        metrics["last_reset_date"] = self.state.get("last_reset_date")
        return metrics
