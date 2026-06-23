# Microsoft Trading Bot

AI-powered trading bot using TradingAgents (multi-agent LLM analysis) + NVIDIA NIM + Twelve Data + Alpaca.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Cloud Server / Scheduler                       │
│                                                  │
│  ┌─────────────┐    ┌──────────────────────┐    │
│  │ Scheduler   │───>│ TradingAgents         │    │
│  │ (daily cron)│    │  - NVIDIA NIM (free)  │    │
│  └─────────────┘    │  - Twelve Data        │    │
│                     └──────────┬───────────┘    │
│                                │                 │
│                     ┌──────────▼───────────┐    │
│                     │ Decision Parser       │    │
│                     │  - Buy/Hold/Sell      │    │
│                     │  - Conviction level   │    │
│                     └──────────┬───────────┘    │
│                                │                 │
│                     ┌──────────▼───────────┐    │
│                     │ Alpaca API            │    │
│                     │  - Paper trading      │    │
│                     │  - Then live          │    │
│                     └──────────────────────┘    │
└─────────────────────────────────────────────────┘
```

## Features

- **TradingAgents**: Multi-agent LLM analysis framework (88k+ GitHub stars)
- **NVIDIA NIM**: Free LLM inference for trading decisions
- **Twelve Data**: Real-time market data with 8-key rotation
- **Alpaca**: Paper and live trading execution
- **Risk Management**: Position sizing, stop-loss, take-profit, daily loss limits

## Setup

### 1. Clone & Install

```bash
git clone https://github.com/Rohan5commit/microsoft-trading-bot.git
cd microsoft-trading-bot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env with your keys
```

Required:
- `NVIDIA_API_KEY` — Free from [build.nvidia.com](https://build.nvidia.com/)
- `TWELVE_DATA_KEYS` — Comma-separated API keys from [twelvedata.com](https://twelvedata.com/)

Optional:
- `ALPACA_API_KEY` + `ALPACA_SECRET_KEY` — From [app.alpaca.markets](https://app.alpaca.markets/paper/dashboard)

### 3. Verify Setup

```bash
python bot/setup.py
```

## Usage

### Analyze Single Stock

```bash
python bot/bot.py --analyze NVDA
python bot/bot.py --analyze AAPL --date 2025-01-15
```

### Run Daily Cycle

```bash
python bot/bot.py --daily
python bot/bot.py --daily --tickers AAPL MSFT NVDA
```

### Check Status

```bash
python bot/bot.py --status
```

### Backtest

```bash
python bot/backtest.py --tickers AAPL MSFT NVDA GOOGL META
```

## GitHub Actions

The bot runs daily via GitHub Actions. See `.github/workflows/daily-analysis.yml`.

### Secrets Required

Set these in GitHub repo Settings > Secrets:

- `NVIDIA_API_KEY`
- `TWELVE_DATA_KEYS`
- `ALPACA_API_KEY` (optional)
- `ALPACA_SECRET_KEY` (optional)

## Configuration

Edit `bot/config.json`:

```json
{
  "universe": {
    "max_stocks": 100,
    "min_market_cap_billion": 10
  },
  "risk": {
    "max_position_pct": 10.0,
    "max_positions": 10,
    "stop_loss_pct": 5.0,
    "take_profit_pct": 20.0
  },
  "llm": {
    "provider": "nvidia",
    "deep_think_model": "meta/llama-3.1-70b-instruct"
  }
}
```

## Risk Warning

This is for educational purposes. Trading involves risk of loss. Start with paper trading.
