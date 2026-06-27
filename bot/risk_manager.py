"""Risk management - minimal circuit breaker. Model decides everything."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


class RiskManager:
    """Minimal risk manager. Only circuit breaker. Model drives all decisions."""

    def __init__(self, config: dict):
        risk_cfg = config.get("risk", {})
        self.circuit_breaker_daily_loss_pct = risk_cfg.get("circuit_breaker_daily_loss_pct", 10.0)
        self.min_conviction = risk_cfg.get("min_conviction", 0.3)

        self.trades_file = Path(__file__).parent / "trades.json"
        self._load_trades()

    def _load_trades(self):
        if self.trades_file.exists():
            with open(self.trades_file) as f:
                self.trades = json.load(f)
        else:
            self.trades = {"positions": {}, "closed": [], "daily_pnl": {}}

    def _save_trades(self):
        with open(self.trades_file, "w") as f:
            json.dump(self.trades, f, indent=2, default=str)

    def check_circuit_breaker(self, portfolio_value: float) -> dict:
        """Check if daily loss circuit breaker is tripped. That's it."""
        today = datetime.now().strftime("%Y-%m-%d")
        daily_loss = self.trades.get("daily_pnl", {}).get(today, 0)

        if daily_loss < 0:
            loss_pct = abs(daily_loss) / portfolio_value * 100
            if loss_pct >= self.circuit_breaker_daily_loss_pct:
                return {
                    "tripped": True,
                    "reasoning": f"Circuit breaker: {loss_pct:.1f}% daily loss >= {self.circuit_breaker_daily_loss_pct}% threshold",
                }

        return {"tripped": False, "reasoning": "Circuit breaker clear"}

    def calculate_position_size(
        self,
        ticker: str,
        price: float,
        conviction: float,
        portfolio_value: float,
        current_positions: list[dict],
        suggested_allocation_pct: Optional[float] = None,
    ) -> dict:
        """Model-driven position sizing.

        The model provides conviction and suggested allocation %.
        We only enforce buying power limits and circuit breaker.

        Args:
            ticker: Stock symbol
            price: Current price
            conviction: Model conviction (0-1)
            portfolio_value: Total portfolio value
            current_positions: List of current positions
            suggested_allocation_pct: Model's suggested % of portfolio (0-100).
                If None, derives from conviction (conviction * 10% of portfolio).

        Returns:
            Dict with qty, notional, action, reasoning
        """
        # Circuit breaker check
        cb = self.check_circuit_breaker(portfolio_value)
        if cb["tripped"]:
            return {
                "qty": 0,
                "notional": 0,
                "action": "skip",
                "reasoning": cb["reasoning"],
            }

        # Conviction threshold
        if conviction < self.min_conviction:
            return {
                "qty": 0,
                "notional": 0,
                "action": "skip",
                "reasoning": f"Conviction {conviction:.2f} below minimum {self.min_conviction}",
            }

        # Model MUST specify allocation. No fallback.
        if suggested_allocation_pct is None:
            return {
                "qty": 0,
                "notional": 0,
                "action": "skip",
                "reasoning": "Model did not specify allocation % — trade skipped",
            }

        allocation_pct = suggested_allocation_pct

        # Safety bounds on allocation
        allocation_pct = max(0.1, min(25.0, allocation_pct))

        target_value = portfolio_value * (allocation_pct / 100.0)

        # Hard ceiling: never more than 25% in one position (safety)
        max_single = portfolio_value * 0.25
        if target_value > max_single:
            target_value = max_single

        # Don't exceed available buying power
        if target_value > portfolio_value * 0.95:
            target_value = portfolio_value * 0.95

        qty = int(target_value / price)
        if qty <= 0:
            return {
                "qty": 0,
                "notional": 0,
                "action": "skip",
                "reasoning": "Calculated qty is 0",
            }

        actual_notional = qty * price

        return {
            "qty": qty,
            "notional": actual_notional,
            "action": "execute",
            "reasoning": (
                f"Model allocation: {allocation_pct:.1f}% of portfolio = ${actual_notional:,.2f} "
                f"({qty} shares @ ${price:.2f}, conviction: {conviction:.2f})"
            ),
        }

    def record_trade(
        self,
        ticker: str,
        action: str,
        qty: int,
        price: float,
        reason: str = "",
    ):
        """Record a trade."""
        now = datetime.now()
        trade = {
            "ticker": ticker,
            "action": action,
            "qty": qty,
            "price": price,
            "timestamp": now.isoformat(),
            "reason": reason,
        }

        if action in ("buy", "short"):
            self.trades["positions"][ticker] = trade
        elif action in ("sell", "close_long", "close_short"):
            self.trades["closed"].append(trade)
            if ticker in self.trades["positions"]:
                del self.trades["positions"][ticker]

        self._save_trades()

    def update_daily_pnl(self, pnl: float):
        """Update daily P&L tracking."""
        today = datetime.now().strftime("%Y-%m-%d")
        if today not in self.trades.get("daily_pnl", {}):
            self.trades.setdefault("daily_pnl", {})[today] = 0
        self.trades["daily_pnl"][today] += pnl
        self._save_trades()

    def get_portfolio_risk(
        self,
        current_positions: list[dict],
        portfolio_value: float,
    ) -> dict:
        """Get portfolio-level risk metrics."""
        if not current_positions:
            return {
                "total_exposure": 0,
                "position_count": 0,
                "max_single_exposure": 0,
            }

        exposures = []
        for pos in current_positions:
            exposure = (pos["market_value"] / portfolio_value) * 100
            exposures.append(exposure)

        return {
            "total_exposure": sum(exposures),
            "position_count": len(current_positions),
            "max_single_exposure": max(exposures) if exposures else 0,
        }
