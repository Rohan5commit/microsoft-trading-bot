"""Alpaca paper/live trading client wrapper."""

import os
import time
import threading
from datetime import datetime, timedelta
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopOrderRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


class AlpacaClient:
    """Wrapper around Alpaca trading and data APIs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        paper: bool = True,
    ):
        """Initialize Alpaca client.

        Args:
            api_key: Alpaca API key (or env ALPACA_API_KEY)
            secret_key: Alpaca secret key (or env ALPACA_SECRET_KEY)
            base_url: Base URL (or env ALPACA_BASE_URL)
            paper: Whether to use paper trading
        """
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        self.base_url = base_url or os.getenv("ALPACA_BASE_URL")

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca API keys required. Set ALPACA_API_KEY and ALPACA_SECRET_KEY "
                "in .env or pass them directly."
            )

        self.trading_client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=paper,
        )

        self.data_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
        )

        # Rate limiting: 200 RPM = 0.3s minimum between calls
        self._last_call_time = 0.0
        self._rate_lock = threading.Lock()
        self._min_interval = 0.35  # 0.35s for safety margin

    def _rate_limit(self):
        """Enforce minimum interval between API calls."""
        with self._rate_lock:
            now = time.time()
            elapsed = now - self._last_call_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call_time = time.time()

    def get_account(self) -> dict:
        """Get account information."""
        self._rate_limit()
        account = self.trading_client.get_account()
        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
            "day_trade_count": account.daytrade_count,
            "pattern_day_trader": account.pattern_day_trader,
        }

    def get_positions(self) -> list[dict]:
        """Get all open positions."""
        self._rate_limit()
        positions = self.trading_client.get_all_positions()
        return [
            {
                "ticker": p.symbol,
                "qty": float(p.qty),
                "market_value": float(p.market_value),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
                "side": p.side.value,
            }
            for p in positions
        ]

    def get_position(self, ticker: str) -> Optional[dict]:
        """Get position for a specific ticker."""
        try:
            self._rate_limit()
            p = self.trading_client.get_open_position(ticker)
            return {
                "ticker": p.symbol,
                "qty": float(p.qty),
                "market_value": float(p.market_value),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
                "side": p.side.value,
            }
        except Exception:
            return None

    def get_buying_power(self) -> float:
        """Get available buying power."""
        self._rate_limit()
        account = self.trading_client.get_account()
        return float(account.buying_power)

    def get_portfolio_value(self) -> float:
        """Get total portfolio value."""
        self._rate_limit()
        account = self.trading_client.get_account()
        return float(account.portfolio_value)

    def market_buy(
        self,
        ticker: str,
        qty: Optional[int] = None,
        notional: Optional[float] = None,
    ) -> dict:
        """Place a market buy order.

        Args:
            ticker: Stock symbol
            qty: Number of shares (mutually exclusive with notional)
            notional: Dollar amount to buy (mutually exclusive with qty)

        Returns:
            Order details
        """
        if qty is None and notional is None:
            raise ValueError("Must specify either qty or notional for market_buy")
        request = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            notional=notional,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        order = self.trading_client.submit_order(request)
        return self._order_to_dict(order)

    def market_sell(
        self,
        ticker: str,
        qty: Optional[int] = None,
        notional: Optional[float] = None,
    ) -> dict:
        """Place a market sell order."""
        if qty is None and notional is None:
            raise ValueError("Must specify either qty or notional for market_sell")
        request = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            notional=notional,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order = self.trading_client.submit_order(request)
        return self._order_to_dict(order)

    def limit_buy(self, ticker: str, qty: int, limit_price: float) -> dict:
        """Place a limit buy order."""
        request = LimitOrderRequest(
            symbol=ticker,
            qty=qty,
            limit_price=limit_price,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        order = self.trading_client.submit_order(request)
        return self._order_to_dict(order)

    def limit_sell(self, ticker: str, qty: int, limit_price: float) -> dict:
        """Place a limit sell order."""
        request = LimitOrderRequest(
            symbol=ticker,
            qty=qty,
            limit_price=limit_price,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order = self.trading_client.submit_order(request)
        return self._order_to_dict(order)

    def stop_loss(self, ticker: str, qty: int, stop_price: float) -> dict:
        """Place a stop-loss sell order."""
        request = StopOrderRequest(
            symbol=ticker,
            qty=qty,
            stop_price=stop_price,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
        )
        order = self.trading_client.submit_order(request)
        return self._order_to_dict(order)

    def get_orders(self, status: Optional[str] = "open") -> list[dict]:
        """Get orders, optionally filtered by status."""
        request = GetOrdersRequest(status=status)
        orders = self.trading_client.get_orders(request)
        return [self._order_to_dict(o) for o in orders]

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID."""
        try:
            self.trading_client.cancel_order_by_id(order_id)
            return True
        except Exception:
            return False

    def cancel_all_orders(self) -> bool:
        """Cancel all open orders."""
        try:
            self.trading_client.cancel_orders()
            return True
        except Exception:
            return False

    def close_position(self, ticker: str) -> dict:
        """Close all positions for a ticker."""
        try:
            order = self.trading_client.close_position(ticker)
            return self._order_to_dict(order)
        except Exception:
            return {}

    def get_stock_price(self, ticker: str) -> float:
        """Get current stock price."""
        self._rate_limit()
        from alpaca.data.requests import StockLatestQuoteRequest
        request = StockLatestQuoteRequest(symbol_or_symbols=ticker)
        quote = self.data_client.get_stock_latest_quote(request)
        ask = quote[ticker].ask_price
        if ask is None:
            bid = quote[ticker].bid_price
            if bid is not None:
                return float(bid)
            raise ValueError(f"No price data available for {ticker} (market may be closed)")
        return float(ask)

    def get_stock_bars(
        self,
        ticker: str,
        days: int = 30,
        timeframe: TimeFrame = TimeFrame.Hour,
    ):
        """Get historical stock bars."""
        self._rate_limit()
        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=timeframe,
            start=datetime.now() - timedelta(days=days),
        )
        return self.data_client.get_stock_bars(request)

    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        clock = self.trading_client.get_clock()
        return clock.is_open

    def get_market_clock(self) -> dict:
        """Get market clock info."""
        clock = self.trading_client.get_clock()
        return {
            "is_open": clock.is_open,
            "next_open": str(clock.next_open),
            "next_close": str(clock.next_close),
            "timestamp": str(clock.timestamp),
        }

    def _order_to_dict(self, order) -> dict:
        """Convert an order object to a dictionary."""
        return {
            "id": str(order.id),
            "symbol": order.symbol,
            "side": order.side.value if hasattr(order.side, "value") else str(order.side),
            "qty": str(order.qty) if order.qty else None,
            "notional": str(order.notional) if order.notional else None,
            "type": order.type.value if hasattr(order.type, "value") else str(order.type),
            "status": order.status.value if hasattr(order.status, "value") else str(order.status),
            "filled_avg_price": str(order.filled_avg_price) if order.filled_avg_price and str(order.filled_avg_price) != "0" else None,
            "submitted_at": str(order.submitted_at),
            "filled_at": str(order.filled_at) if order.filled_at else None,
        }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    client = AlpacaClient(paper=True)
    account = client.get_account()
    print(f"Account equity: ${account['equity']:,.2f}")
    print(f"Buying power: ${account['buying_power']:,.2f}")
    print(f"Market open: {client.is_market_open()}")
