"""Email sender for daily trading bot updates.

Uses Gmail SMTP with App Password authentication.
"""

import html
import logging
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


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
        """Send daily trading bot update email."""
        subject = f"Trading Bot - Daily Update {datetime.now().strftime('%Y-%m-%d')}"
        html_body = self._build_html(analysis_results, alpaca_status)

        msg = MIMEMultipart("mixed")
        msg["Subject"] = subject
        msg["From"] = self.sender_email
        msg["To"] = self.receiver_email
        msg["Reply-To"] = self.sender_email

        msg.attach(MIMEText(html_body, "html"))

        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.ehlo()
                logger.info("SMTP connected to smtp.gmail.com:465")

                server.login(self.sender_email, self.app_password)
                logger.info("SMTP login successful")

                code, response = server.noop()
                logger.info(f"SMTP noop after login: code={code} response={response}")
                if code != 250:
                    logger.error(f"SMTP session invalid after login: {code} {response}")
                    return False

                rejects = server.sendmail(self.sender_email, self.receiver_email, msg.as_string())
                if rejects:
                    logger.error(f"Email rejected by Gmail: {rejects}")
                    return False

                code, response = server.noop()
                logger.info(f"SMTP noop after send: code={code} response={response}")

            logger.info(f"Email sent successfully to {self.receiver_email}")
            return True
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP authentication failed (bad credentials?): {e}")
            return False
        except smtplib.SMTPException as e:
            logger.error(f"SMTP error: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def _build_html(self, results: dict, alpaca_status: Optional[dict] = None) -> str:
        """Build HTML email body."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        phase1 = results.get("phase1", {})
        phase2 = results.get("phase2", {})
        deep_results = results.get("deep_results", [])
        total_time = results.get("total_elapsed_seconds", 0)
        portfolio_metrics = results.get("portfolio_metrics")
        execution = results.get("execution", {})

        buy_signals = [r for r in deep_results if r.get("action") == "buy"]
        sell_signals = [r for r in deep_results if r.get("action") == "sell"]
        hold_signals = [r for r in deep_results if r.get("action") == "hold"]

        # Portfolio metrics section
        portfolio_html = ""
        if portfolio_metrics:
            period_return = portfolio_metrics.get("period_return_pct", 0)
            cumulative_return = portfolio_metrics.get("cumulative_return_pct", 0)
            dollar_pnl = portfolio_metrics.get("dollar_pnl", 0)
            total_dollar_pnl = portfolio_metrics.get("total_dollar_pnl", 0)
            exposure_divisor = portfolio_metrics.get("exposure_divisor", 0)
            leverage = portfolio_metrics.get("leverage", 1)
            initial_capital = portfolio_metrics.get("initial_capital", 0)
            reset_baseline = portfolio_metrics.get("reset_baseline", 0)

            period_color = "#00c853" if period_return >= 0 else "#ff1744"
            cumulative_color = "#00c853" if cumulative_return >= 0 else "#ff1744"
            dollar_color = "#00c853" if dollar_pnl >= 0 else "#ff1744"
            total_color = "#00c853" if total_dollar_pnl >= 0 else "#ff1744"

            portfolio_html = f"""
            <div style="background: #1a1a2e; border-radius: 10px; padding: 20px; margin: 20px 0; border: 1px solid #00d4ff;">
                <h2 style="color: #00d4ff; margin-top: 0;">Portfolio Return Metrics</h2>
                <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                    <div style="background: #16213e; padding: 15px; border-radius: 8px; flex: 1; min-width: 150px;">
                        <div style="color: #888; font-size: 12px;">Initial Capital (C0)</div>
                        <div style="color: #fff; font-size: 20px; font-weight: bold;">${initial_capital:,.2f}</div>
                    </div>
                    <div style="background: #16213e; padding: 15px; border-radius: 8px; flex: 1; min-width: 150px;">
                        <div style="color: #888; font-size: 12px;">Leverage (L)</div>
                        <div style="color: #00d4ff; font-size: 20px; font-weight: bold;">{leverage:.1f}x</div>
                    </div>
                    <div style="background: #16213e; padding: 15px; border-radius: 8px; flex: 1; min-width: 150px;">
                        <div style="color: #888; font-size: 12px;">Exposure Divisor (B)</div>
                        <div style="color: #fff; font-size: 20px; font-weight: bold;">${exposure_divisor:,.2f}</div>
                    </div>
                    <div style="background: #16213e; padding: 15px; border-radius: 8px; flex: 1; min-width: 150px;">
                        <div style="color: #888; font-size: 12px;">Reset Baseline (E_reset)</div>
                        <div style="color: #fff; font-size: 20px; font-weight: bold;">${reset_baseline:,.2f}</div>
                    </div>
                </div>
                <div style="display: flex; gap: 20px; flex-wrap: wrap; margin-top: 15px;">
                    <div style="background: #16213e; padding: 15px; border-radius: 8px; flex: 1; min-width: 150px;">
                        <div style="color: #888; font-size: 12px;">Period Return</div>
                        <div style="color: {period_color}; font-size: 24px; font-weight: bold;">{period_return:+.2f}%</div>
                    </div>
                    <div style="background: #16213e; padding: 15px; border-radius: 8px; flex: 1; min-width: 150px;">
                        <div style="color: #888; font-size: 12px;">Cumulative Return</div>
                        <div style="color: {cumulative_color}; font-size: 24px; font-weight: bold;">{cumulative_return:+.2f}%</div>
                    </div>
                    <div style="background: #16213e; padding: 15px; border-radius: 8px; flex: 1; min-width: 150px;">
                        <div style="color: #888; font-size: 12px;">Period P&L</div>
                        <div style="color: {dollar_color}; font-size: 24px; font-weight: bold;">${dollar_pnl:+,.2f}</div>
                    </div>
                    <div style="background: #16213e; padding: 15px; border-radius: 8px; flex: 1; min-width: 150px;">
                        <div style="color: #888; font-size: 12px;">Total P&L</div>
                        <div style="color: {total_color}; font-size: 24px; font-weight: bold;">${total_dollar_pnl:+,.2f}</div>
                    </div>
                </div>
            </div>
            """

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
                side = pos.get("side", "long")
                side_color = "#00c853" if side == "long" else "#ff1744"
                positions_html += f"""
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">{pos.get('ticker', 'N/A')}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #333; color: {side_color};">{side.upper()}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">{pos.get('qty', 0)}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">${pos.get('avg_entry_price', 0):.2f}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">${pos.get('current_price', 0):.2f}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #333; color: {pnl_color};">${pnl:.2f}</td>
                </tr>
                """

            market_color = "#00c853" if market_open else "#ff1744"
            market_status = "OPEN" if market_open else "CLOSED"

            positions_header = '<h3 style="color: #fff; margin-top: 20px;">Open Positions</h3>' if positions else ""
            if positions:
                positions_table = f"""
                <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
                    <thead>
                        <tr style="background: #16213e;">
                            <th style="padding: 10px; text-align: left; color: #888;">Ticker</th>
                            <th style="padding: 10px; text-align: left; color: #888;">Side</th>
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
                """
            else:
                positions_table = ""

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
                        <div style="color: {market_color}; font-size: 24px; font-weight: bold;">{market_status}</div>
                    </div>
                </div>
                {positions_header}
                {positions_table}
            </div>
            """

        # Execution results section
        execution_html = ""
        exec_results = execution.get("results", [])
        if exec_results:
            exec_rows = ""
            for r in exec_results:
                status = r.get("status", "unknown")
                if status == "filled":
                    status_color = "#00c853"
                    status_text = "FILLED"
                elif status == "submitted":
                    status_color = "#29b6f6"
                    status_text = "SUBMITTED"
                elif status == "skipped":
                    status_color = "#ffd600"
                    status_text = "SKIPPED"
                elif status == "error":
                    status_color = "#ff1744"
                    status_text = "ERROR"
                elif status == "no_action":
                    status_color = "#888"
                    status_text = "NO ACTION"
                elif status == "circuit_breaker":
                    status_color = "#ff1744"
                    status_text = "CIRCUIT BREAKER"
                else:
                    status_color = "#888"
                    status_text = status.upper()

                signal = r.get("signal", "?")
                signal_color = "#00c853" if signal == "buy" else "#ff1744" if signal == "sell" else "#888"

                qty_str = str(r.get("qty", "-")) if r.get("qty") is not None else "-"
                price_str = f"${r.get('price', 0):.2f}" if r.get("price") else "-"
                reasoning = html.escape((r.get("reasoning") or "")[:120])

                exec_rows += f"""
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #333; font-weight: bold;">{r.get('ticker', 'N/A')}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #333; color: {signal_color};">{signal.upper()}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">{r.get('conviction', 0):.0%}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">{qty_str}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">{price_str}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #333; color: {status_color}; font-weight: bold;">{status_text}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #333; font-size: 11px;">{reasoning}</td>
                </tr>
                """

            filled_count = execution.get("filled", 0)
            skipped_count = execution.get("skipped", 0)
            error_count = execution.get("errors", 0)
            exec_enabled = execution.get("enabled", False)

            execution_html = f"""
            <div style="background: #1a1a2e; border-radius: 10px; padding: 20px; margin: 20px 0; border: 1px solid #00d4ff;">
                <h2 style="color: #00d4ff; margin-top: 0;">Trade Execution {'(LIVE)' if exec_enabled else '(DRY RUN)'}</h2>
                <div style="display: flex; gap: 15px; margin-bottom: 15px;">
                    <div style="background: #16213e; padding: 10px 15px; border-radius: 8px;">
                        <span style="color: #888; font-size: 12px;">Filled: </span>
                        <span style="color: #00c853; font-weight: bold;">{filled_count}</span>
                    </div>
                    <div style="background: #16213e; padding: 10px 15px; border-radius: 8px;">
                        <span style="color: #888; font-size: 12px;">Skipped: </span>
                        <span style="color: #ffd600; font-weight: bold;">{skipped_count}</span>
                    </div>
                    <div style="background: #16213e; padding: 10px 15px; border-radius: 8px;">
                        <span style="color: #888; font-size: 12px;">Errors: </span>
                        <span style="color: #ff1744; font-weight: bold;">{error_count}</span>
                    </div>
                </div>
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="background: #16213e;">
                            <th style="padding: 10px; text-align: left; color: #888;">Ticker</th>
                            <th style="padding: 10px; text-align: left; color: #888;">Signal</th>
                            <th style="padding: 10px; text-align: left; color: #888;">Conviction</th>
                            <th style="padding: 10px; text-align: left; color: #888;">Qty</th>
                            <th style="padding: 10px; text-align: left; color: #888;">Price</th>
                            <th style="padding: 10px; text-align: left; color: #888;">Status</th>
                            <th style="padding: 10px; text-align: left; color: #888;">Reasoning</th>
                        </tr>
                    </thead>
                    <tbody>
                        {exec_rows}
                    </tbody>
                </table>
            </div>
            """

        # Buy signals section
        buy_html = ""
        if buy_signals:
            buy_rows = ""
            for b in sorted(buy_signals, key=lambda x: x.get("conviction", 0), reverse=True):
                raw_reasoning = (b.get("reasoning") or "")
                reasoning = html.escape(raw_reasoning[:150])
                if len(raw_reasoning) > 150:
                    reasoning += "..."
                buy_rows += f"""
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #333; color: #00c853; font-weight: bold;">{b.get('ticker', 'N/A')}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">${b.get('price', 0):.2f}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">{b.get('conviction', 0):.0%}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #333; font-size: 12px;">{reasoning}</td>
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
                raw_reasoning = (s.get("reasoning") or "")
                reasoning = html.escape(raw_reasoning[:150])
                if len(raw_reasoning) > 150:
                    reasoning += "..."
                sell_rows += f"""
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #333; color: #ff1744; font-weight: bold;">{s.get('ticker', 'N/A')}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">${s.get('price', 0):.2f}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #333;">{s.get('conviction', 0):.0%}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #333; font-size: 12px;">{reasoning}</td>
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

        email_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
        </head>
        <body style="margin: 0; padding: 0; background-color: #0f0f23; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
            <div style="max-width: 800px; margin: 0 auto; padding: 20px;">
                <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 15px; padding: 30px; margin-bottom: 20px; border: 1px solid #333;">
                    <h1 style="color: #00d4ff; margin: 0; font-size: 28px;">Trading Bot</h1>
                    <p style="color: #888; margin: 5px 0 0 0;">Autonomous Daily Analysis Report</p>
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

                {portfolio_html}
                {alpaca_html}
                {execution_html}
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
                    <p>Trading Bot | Powered by TradingAgents + NVIDIA NIM</p>
                    <p>This is for educational purposes only. Trading involves risk of loss.</p>
                </div>
            </div>
        </body>
        </html>
        """

        return email_html


    def send_error_notification(self, error_message: str, traceback_str: str = "") -> bool:
        """Send error notification email."""
        subject = f"Trading Bot - ERROR {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        html_body = self._build_error_html(error_message, traceback_str)

        msg = MIMEMultipart("mixed")
        msg["Subject"] = subject
        msg["From"] = self.sender_email
        msg["To"] = self.receiver_email
        msg["Reply-To"] = self.sender_email

        msg.attach(MIMEText(html_body, "html"))

        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.ehlo()
                logger.info("SMTP connected to smtp.gmail.com:465 (error notification)")

                server.login(self.sender_email, self.app_password)
                logger.info("SMTP login successful (error notification)")

                code, response = server.noop()
                logger.info(f"SMTP noop after login: code={code} response={response}")
                if code != 250:
                    logger.error(f"SMTP session invalid after login: {code} {response}")
                    return False

                rejects = server.sendmail(self.sender_email, self.receiver_email, msg.as_string())
                if rejects:
                    logger.error(f"Error email rejected by Gmail: {rejects}")
                    return False

                code, response = server.noop()
                logger.info(f"SMTP noop after send: code={code} response={response}")

            logger.info(f"Error email sent successfully to {self.receiver_email}")
            return True
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP authentication failed (bad credentials?): {e}")
            return False
        except smtplib.SMTPException as e:
            logger.error(f"SMTP error: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to send error email: {e}")
            return False

    def _build_error_html(self, error_message: str, traceback_str: str = "") -> str:
        """Build HTML error notification body."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        safe_error = html.escape(str(error_message))
        safe_traceback = html.escape(str(traceback_str)) if traceback_str else ""

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
        </head>
        <body style="margin: 0; padding: 0; background-color: #0f0f23; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
            <div style="max-width: 800px; margin: 0 auto; padding: 20px;">
                <div style="background: linear-gradient(135deg, #2e1a1a 0%, #3e1616 100%); border-radius: 15px; padding: 30px; margin-bottom: 20px; border: 1px solid #ff1744;">
                    <h1 style="color: #ff1744; margin: 0; font-size: 28px;">Trading Bot</h1>
                    <p style="color: #ff8a80; margin: 5px 0 0 0;">ERROR REPORT</p>
                    <p style="color: #666; margin: 5px 0 0 0; font-size: 12px;">{now}</p>
                </div>

                <div style="background: #1a1a2e; border-radius: 10px; padding: 20px; margin: 20px 0; border: 1px solid #ff1744;">
                    <h2 style="color: #ff1744; margin-top: 0;">Error Details</h2>
                    <div style="background: #16213e; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                        <div style="color: #ff8a80; font-size: 12px; margin-bottom: 5px;">ERROR MESSAGE</div>
                        <div style="color: #fff; font-size: 16px; font-family: monospace;">{safe_error}</div>
                    </div>
                    {f"""
                    <div style="background: #16213e; padding: 15px; border-radius: 8px;">
                        <div style="color: #ff8a80; font-size: 12px; margin-bottom: 5px;">TRACEBACK</div>
                        <pre style="color: #aaa; font-size: 12px; white-space: pre-wrap; word-wrap: break-word;">{safe_traceback}</pre>
                    </div>
                    """ if safe_traceback else ""}
                </div>

                <div style="text-align: center; padding: 20px; color: #666; font-size: 12px;">
                    <p>Trading Bot | Powered by TradingAgents + NVIDIA NIM</p>
                </div>
            </div>
        </body>
        </html>
        """


if __name__ == "__main__":
    sender = EmailSender()
    sender.send_error_notification("Test error: Failed to connect to Alpaca API", "Traceback: ConnectionTimeout")
    print("Test error email sent!")
