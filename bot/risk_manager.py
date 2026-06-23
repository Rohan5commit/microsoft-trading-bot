"""Risk management - position sizing, stop-losses, and portfolio limits."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


class RiskManager:
    """Manages portfolio risk, position sizing, and stop-losses."""

    def __init__(self, config: dict):
        """Initialize risk manager.

        Args:
            config: Bot configuration dict
        """
        risk_cfg = config.get("risk", {})
        self.max_position_pct = risk_cfg.get("max_position_pct", 10.0)
        self.max_positions = risk_cfg.get("max_positions", 10)
        self.stop_loss_pct = risk_cfg.get("stop_loss_pct", 5.0)
        self.take_profit_pct = risk_cfg.get("take_profit_pct", 20.0)
        self.max_daily_loss_pct = risk_cfg.get("max_daily_loss_pct", 3.0)
        self.min_conviction = risk_cfg.get("min_conviction", 0.6)
        self.position_size_method = risk_cfg.get("position_size_method", "equal_weight")

        self.trades_file = Path(__file__).parent / "trades.json"
        self._load_trades()

    def _load_trades(self):
        """Load trade history from file."""
        if self.trades_file.exists():
            with open(self.trades_file) as f:
                self.trades = json.load(f)
        else:
            self.trades = {"positions": {}, "closed": [], "daily_pnl": {}}

    def _save_trades(self):
        """Save trade history to file."""
        with open(self.trades_file, "w") as f:
            json.dump(self.trades, f, indent=2, default=str)

    def calculate_position_size(
        self,
        ticker: str,
        price: float,
        conviction: float,
        portfolio_value: float,
        current_positions: list[dict],
    ) -> dict:
        """Calculate position size based on risk parameters.

        Args:
            ticker: Stock symbol
            price: Current stock price
            conviction: Agent conviction score (0-1)
            portfolio_value: Total portfolio value
            current_positions: List of current positions

        Returns:
            Dict with qty, notional, and reasoning
        """
        # Check if we already have a position in this ticker
        for pos in current_positions:
            if pos["ticker"] == ticker:
                return {
                    "qty": 0,
                    "notional": 0,
                    "action": "hold",
                    "reasoning": f"Already holding {ticker}",
                }

        # Check max positions
        if len(current_positions) >= self.max_positions:
            return {
                "qty": 0,
                "notional": 0,
                "action": "skip",
                "reasoning": f"Max positions ({self.max_positions}) reached",
            }

        # Check conviction threshold
        if conviction < self.min_conviction:
            return {
                "qty": 0,
                "notional": 0,
                "action": "skip",
                "reasoning": f"Conviction {conviction:.2f} below threshold {self.min_conviction}",
            }

        # Calculate max position value
        max_position_value = portfolio_value * (self.max_position_pct / 100)

        # Adjust by conviction (higher conviction = larger position)
        conviction_multiplier = min(1.0, conviction / 0.8)  # Scale to 0.8 as full conviction
        target_value = max_position_value * conviction_multiplier

        # Check daily loss limit
        today = datetime.now().strftime("%Y-%m-%d")
        daily_loss = self.trades.get("daily_pnl", {}).get(today, 0)
        if daily_loss < 0:
            loss_pct = abs(daily_loss) / portfolio_value * 100
            if loss_pct >= self.max_daily_loss_pct:
                return {
                    "qty": 0,
                    "notional": 0,
                    "action": "skip",
                    "reasoning": f"Daily loss limit reached ({loss_pct:.1f}%)",
                }
            # Reduce position size proportionally
            remaining_budget = 1 - (loss_pct / self.max_daily_loss_pct)
            target_value *= remaining_budget

        # Calculate shares
        qty = int(target_value / price)

        # Ensure we don't exceed buying power
        if qty * price > portfolio_value * 0.95:  # 5% buffer
            qty = int((portfolio_value * 0.95) / price)

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
            "action": "buy",
            "reasoning": (
                f"Position size: {qty} shares @ ${price:.2f} = ${actual_notional:.2f} "
                f"({actual_notional/portfolio_value*100:.1f}% of portfolio, "
                f"conviction: {conviction:.2f})"
            ),
        }

    def check_stop_loss(
        self,
        ticker: str,
        entry_price: float,
        current_price: float,
    ) -> dict:
        """Check if a position should be stopped out.

        Returns:
            Dict with should_stop, loss_pct, and reasoning
        """
        if entry_price <= 0:
            return {"should_stop": False, "loss_pct": 0, "reasoning": "Invalid entry price"}

        loss_pct = ((entry_price - current_price) / entry_price) * 100

        if loss_pct >= self.stop_loss_pct:
            return {
                "should_stop": True,
                "loss_pct": loss_pct,
                "reasoning": f"Stop-loss triggered: {loss_pct:.1f}% loss (threshold: {self.stop_loss_pct}%)",
            }

        return {
            "should_stop": False,
            "loss_pct": loss_pct,
            "reasoning": f"Loss {loss_pct:.1f}% within tolerance",
        }

    def check_take_profit(
        self,
        ticker: str,
        entry_price: float,
        current_price: float,
    ) -> dict:
        """Check if a position should take profit.

        Returns:
            Dict with should_take_profit, gain_pct, and reasoning
        """
        if entry_price <= 0:
            return {"should_take_profit": False, "gain_pct": 0, "reasoning": "Invalid entry price"}

        gain_pct = ((current_price - entry_price) / entry_price) * 100

        if gain_pct >= self.take_profit_pct:
            return {
                "should_take_profit": True,
                "gain_pct": gain_pct,
                "reasoning": f"Take-profit triggered: {gain_pct:.1f}% gain (threshold: {self.take_profit_pct}%)",
            }

        return {
            "should_take_profit": False,
            "gain_pct": gain_pct,
            "reasoning": f"Gain {gain_pct:.1f}% below threshold",
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

        if action == "buy":
            self.trades["positions"][ticker] = trade
        elif action == "sell":
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
                "concentration_risk": "low",
            }

        exposures = []
        for pos in current_positions:
            exposure = (pos["market_value"] / portfolio_value) * 100
            exposures.append(exposure)

        total_exposure = sum(exposures)
        max_single = max(exposures) if exposures else 0

        # Determine concentration risk
        if max_single > 20:
            concentration = "high"
        elif max_single > 15:
            concentration = "medium"
        else:
            concentration = "low"

        return {
            "total_exposure": total_exposure,
            "position_count": len(current_positions),
            "max_single_exposure": max_single,
            "concentration_risk": concentration,
        }

    def should_open_position(
        self,
        ticker: str,
        conviction: float,
        portfolio_value: float,
        current_positions: list[dict],
    ) -> dict:
        """Comprehensive check if we should open a new position.

        Returns:
            Dict with should_open, sizing info, and reasoning
        """
        reasons = []

        # Check if already holding
        if any(p["ticker"] == ticker for p in current_positions):
            return {
                "should_open": False,
                "reasoning": f"Already holding {ticker}",
            }

        # Check max positions
        if len(current_positions) >= self.max_positions:
            reasons.append(f"Max positions ({self.max_positions}) reached")

        # Check conviction
        if conviction < self.min_conviction:
            reasons.append(f"Conviction {conviction:.2f} below {self.min_conviction}")

        # Check daily loss
        today = datetime.now().strftime("%Y-%m-%d")
        daily_loss = self.trades.get("daily_pnl", {}).get(today, 0)
        if daily_loss < 0:
            loss_pct = abs(daily_loss) / portfolio_value * 100
            if loss_pct >= self.max_daily_loss_pct:
                reasons.append(f"Daily loss limit {loss_pct:.1f}% >= {self.max_daily_loss_pct}%")

        if reasons:
            return {
                "should_open": False,
                "reasoning": "; ".join(reasons),
            }

        return {
            "should_open": True,
            "reasoning": "All risk checks passed",
        }
