"""Email sender for daily trading bot updates.

Uses Gmail SMTP with App Password authentication.
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional


class EmailSender:
    """Send trading bot updates via Gmail SMTP."""

    def __init__(
        self,
        sender_email: Optional[str] = None,
        app_password: Optional[str] = None,
        receiver_email: Optional[str] = None,
    ):
        self.sender_email = sender_email or os.getenv("SENDER_EMAIL")
        self.app_password = app_password or os.getenv("GMAIL_APP_PASSWORD")
        self.receiver_email = receiver_email or os.getenv("RECEIVER_EMAIL")

        if not all([self.sender_email, self.app_password, self.receiver_email]):
            raise ValueError(
                "Email credentials required. Set SENDER_EMAIL, GMAIL_APP_PASSWORD, "
                "and RECEIVER_EMAIL environment variables."
            )

    def send_daily_update(self, analysis_results: dict, alpaca_status: Optional[dict] = None) -> bool:
        """Send daily trading bot update email.

        Args:
            analysis_results: Results from two-phase analysis
            alpaca_status: Alpaca account status (optional)

        Returns:
            True if sent successfully
        """
        subject = f"Microsoft Trading Bot - Daily Update {datetime.now().strftime('%Y-%m-%d')}"
        html_body = self._build_html(analysis_results, alpaca_status)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"Microsoft Trading Bot <{self.sender_email}>"
        msg["To"] = self.receiver_email

        msg.attach(MIMEText(html_body, "html"))

        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(self.sender_email, self.app_password)
                server.sendmail(self.sender_email, self.receiver_email, msg.as_string())
            return True
        except Exception as e:
            print(f"Failed to send email: {e}")
            return False

    def _build_html(self, results: dict, alpaca_status: Optional[dict] = None) -> str:
        """Build HTML email body."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Extract analysis data
        phase1 = results.get("phase1", {})
        phase2 = results.get("phase2", {})
        deep_results = results.get("deep_results", [])
        total_time = results.get("total_elapsed_seconds", 0)

        # Build buy/sell signals
        buy_signals = [r for r in deep_results if r.get("action") == "buy"]
        sell_signals = [r for r in deep_results if r.get("action") == "sell"]
        hold_signals = [r for r in deep_results if r.get("action") == "hold"]

        # Alpaca account section
        alpaca_html = ""
        if alpaca_status:
            equity = alpaca_status.get("portfolio_value", 0)
            cash = alpaca_status.get("cash", 0)
            buying_power = alpaca_status.get("buying_power", 0)
            positions = alpaca_status.get("positions", [])
            market_open = alpaca_status.get("market_open", False)

            positions_html = ""
            for pos in positions:
                pnl = pos.get("unrealized_pl", 0)
                pnl_color = "#00c853" if pnl >= 0 else "#ff1744"
                positions_html += f"""
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">{pos.get('ticker', 'N/A')}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">{pos.get('qty', 0)}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">${pos.get('avg_entry_price', 0):.2f}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">${pos.get('current_price', 0):.2f}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #333; color: {pnl_color};">${pnl:.2f}</td>
                </tr>
                """

            alpaca_html = f"""
            <div style="background: #1a1a2e; border-radius: 10px; padding: 20px; margin: 20px 0; border: 1px solid #333;">
                <h2 style="color: #00d4ff; margin-top: 0;">Alpaca Account Status</h2>
                <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                    <div style="background: #16213e; padding: 15px; border-radius: 8px; flex: 1; min-width: 150px;">
                        <div style="color: #888; font-size: 12px;">Portfolio Value</div>
                        <div style="color: #00c853; font-size: 24px; font-weight: bold;">${equity:,.2f}</div>
                    </div>
                    <div style="background: #16213e; padding: 15px; border-radius: 8px; flex: 1; min-width: 150px;">
                        <div style="color: #888; font-size: 12px;">Cash</div>
                        <div style="color: #fff; font-size: 24px; font-weight: bold;">${cash:,.2f}</div>
                    </div>
                    <div style="background: #16213e; padding: 15px; border-radius: 8px; flex: 1; min-width: 150px;">
                        <div style="color: #888; font-size: 12px;">Buying Power</div>
                        <div style="color: #00d4ff; font-size: 24px; font-weight: bold;">${buying_power:,.2f}</div>
                    </div>
                    <div style="background: #16213e; padding: 15px; border-radius: 8px; flex: 1; min-width: 150px;">
                        <div style="color: #888; font-size: 12px;">Market</div>
                        <div style="color: {'#00c853' if market_open else '#ff1744'}; font-size: 24px; font-weight: bold;">{'OPEN' if market_open else 'CLOSED'}</div>
                    </div>
                </div>
                {"<h3 style="color: #fff; margin-top: 20px;">Open Positions</h3>" if positions else ""}
                {f'''
                <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
                    <thead>
                        <tr style="background: #16213e;">
                            <th style="padding: 10px; text-align: left; color: #888;">Ticker</th>
                            <th style="padding: 10px; text-align: left; color: #888;">Qty</th>
                            <th style="padding: 10px; text-align: left; color: #888;">Entry</th>
                            <th style="padding: 10px; text-align: left; color: #888;">Current</th>
                            <th style="padding: 10px; text-align: left; color: #888;">P&L</th>
                        </tr>
                    </thead>
                    <tbody>
                        {positions_html}
                    </tbody>
                </table>
                ''' if positions else ""}
            </div>
            """

        # Buy signals section
        buy_html = ""
        if buy_signals:
            buy_rows = ""
            for b in sorted(buy_signals, key=lambda x: x.get("conviction", 0), reverse=True):
                reasoning = b.get("reasoning", "")[:150]
                buy_rows += f"""
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #333; color: #00c853; font-weight: bold;">{b.get('ticker', 'N/A')}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">${b.get('price', 0):.2f}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">{b.get('conviction', 0):.0%}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #333; font-size: 12px;">{reasoning}...</td>
                </tr>
                """
            buy_html = f"""
            <div style="background: #1a1a2e; border-radius: 10px; padding: 20px; margin: 20px 0; border: 1px solid #00c853;">
                <h2 style="color: #00c853; margin-top: 0;">BUY Signals</h2>
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="background: #16213e;">
                            <th style="padding: 10px; text-align: left; color: #888;">Ticker</th>
                            <th style="padding: 10px; text-align: left; color: #888;">Price</th>
                            <th style="padding: 10px; text-align: left; color: #888;">Conviction</th>
                            <th style="padding: 10px; text-align: left; color: #888;">Reasoning</th>
                        </tr>
                    </thead>
                    <tbody>
                        {buy_rows}
                    </tbody>
                </table>
            </div>
            """

        # Sell signals section
        sell_html = ""
        if sell_signals:
            sell_rows = ""
            for s in sorted(sell_signals, key=lambda x: x.get("conviction", 0), reverse=True):
                reasoning = s.get("reasoning", "")[:150]
                sell_rows += f"""
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #333; color: #ff1744; font-weight: bold;">{s.get('ticker', 'N/A')}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">${s.get('price', 0):.2f}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">{s.get('conviction', 0):.0%}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #333; font-size: 12px;">{reasoning}...</td>
                </tr>
                """
            sell_html = f"""
            <div style="background: #1a1a2e; border-radius: 10px; padding: 20px; margin: 20px 0; border: 1px solid #ff1744;">
                <h2 style="color: #ff1744; margin-top: 0;">SELL Signals</h2>
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="background: #16213e;">
                            <th style="padding: 10px; text-align: left; color: #888;">Ticker</th>
                            <th style="padding: 10px; text-align: left; color: #888;">Price</th>
                            <th style="padding: 10px; text-align: left; color: #888;">Conviction</th>
                            <th style="padding: 10px; text-align: left; color: #888;">Reasoning</th>
                        </tr>
                    </thead>
                    <tbody>
                        {sell_rows}
                    </tbody>
                </table>
            </div>
            """

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
        </head>
        <body style="margin: 0; padding: 0; background-color: #0f0f23; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
            <div style="max-width: 800px; margin: 0 auto; padding: 20px;">
                <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 15px; padding: 30px; margin-bottom: 20px; border: 1px solid #333;">
                    <h1 style="color: #00d4ff; margin: 0; font-size: 28px;">Microsoft Trading Bot</h1>
                    <p style="color: #888; margin: 5px 0 0 0;">Daily Analysis Report</p>
                    <p style="color: #666; margin: 5px 0 0 0; font-size: 12px;">{now}</p>
                </div>

                <div style="background: #1a1a2e; border-radius: 10px; padding: 20px; margin: 20px 0; border: 1px solid #333;">
                    <h2 style="color: #00d4ff; margin-top: 0;">Analysis Summary</h2>
                    <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                        <div style="background: #16213e; padding: 15px; border-radius: 8px; flex: 1; min-width: 120px; text-align: center;">
                            <div style="color: #888; font-size: 12px;">Scanned</div>
                            <div style="color: #fff; font-size: 24px; font-weight: bold;">{phase1.get('total_scanned', 0)}</div>
                        </div>
                        <div style="background: #16213e; padding: 15px; border-radius: 8px; flex: 1; min-width: 120px; text-align: center;">
                            <div style="color: #888; font-size: 12px;">Candidates</div>
                            <div style="color: #00d4ff; font-size: 24px; font-weight: bold;">{phase1.get('candidates_found', 0)}</div>
                        </div>
                        <div style="background: #16213e; padding: 15px; border-radius: 8px; flex: 1; min-width: 120px; text-align: center;">
                            <div style="color: #888; font-size: 12px;">Buy</div>
                            <div style="color: #00c853; font-size: 24px; font-weight: bold;">{phase2.get('buy', 0)}</div>
                        </div>
                        <div style="background: #16213e; padding: 15px; border-radius: 8px; flex: 1; min-width: 120px; text-align: center;">
                            <div style="color: #888; font-size: 12px;">Sell</div>
                            <div style="color: #ff1744; font-size: 24px; font-weight: bold;">{phase2.get('sell', 0)}</div>
                        </div>
                        <div style="background: #16213e; padding: 15px; border-radius: 8px; flex: 1; min-width: 120px; text-align: center;">
                            <div style="color: #888; font-size: 12px;">Hold</div>
                            <div style="color: #ffd600; font-size: 24px; font-weight: bold;">{phase2.get('hold', 0)}</div>
                        </div>
                        <div style="background: #16213e; padding: 15px; border-radius: 8px; flex: 1; min-width: 120px; text-align: center;">
                            <div style="color: #888; font-size: 12px;">Time</div>
                            <div style="color: #fff; font-size: 24px; font-weight: bold;">{total_time/60:.1f}m</div>
                        </div>
                    </div>
                </div>

                {alpaca_html}
                {buy_html}
                {sell_html}

                <div style="background: #1a1a2e; border-radius: 10px; padding: 20px; margin: 20px 0; border: 1px solid #333;">
                    <h2 style="color: #00d4ff; margin-top: 0;">Hold Positions</h2>
                    <p style="color: #888;">{len(hold_signals)} stocks held - no action recommended</p>
                    <div style="display: flex; flex-wrap: wrap; gap: 10px;">
                        {"".join([f'<span style="background: #16213e; padding: 5px 10px; border-radius: 5px; color: #ffd600;">{h.get("ticker", "N/A")}</span>' for h in hold_signals])}
                    </div>
                </div>

                <div style="text-align: center; padding: 20px; color: #666; font-size: 12px;">
                    <p>Microsoft Trading Bot | Powered by TradingAgents + NVIDIA NIM</p>
                    <p>This is for educational purposes only. Trading involves risk of loss.</p>
                </div>
            </div>
        </body>
        </html>
        """

        return html


if __name__ == "__main__":
    sender = EmailSender()
    test_results = {
        "phase1": {"total_scanned": 1000, "candidates_found": 20},
        "phase2": {"buy": 3, "sell": 1, "hold": 16},
        "total_elapsed_seconds": 300,
        "deep_results": [
            {"ticker": "NVDA", "action": "buy", "conviction": 0.85, "price": 208.65, "reasoning": "Strong technical momentum with RSI oversold bounce"},
            {"ticker": "AAPL", "action": "buy", "conviction": 0.75, "price": 195.50, "reasoning": "Breaking out of consolidation pattern"},
            {"ticker": "TSLA", "action": "sell", "conviction": 0.70, "price": 245.30, "reasoning": "Overbought RSI with bearish divergence"},
        ],
    }
    test_alpaca = {
        "portfolio_value": 10000,
        "cash": 5000,
        "buying_power": 10000,
        "market_open": True,
        "positions": [
            {"ticker": "NVDA", "qty": 10, "avg_entry_price": 200.00, "current_price": 208.65, "unrealized_pl": 86.50},
            {"ticker": "AAPL", "qty": 15, "avg_entry_price": 190.00, "current_price": 195.50, "unrealized_pl": 82.50},
        ],
    }
    sender.send_daily_update(test_results, test_alpaca)
    print("Test email sent!")
