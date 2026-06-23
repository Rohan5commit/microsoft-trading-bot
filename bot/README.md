# Trading Bot

Automated trading bot using TradingAgents (multi-agent LLM analysis) + Alpaca (execution).

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Cloud Server / Scheduler                       │
│                                                  │
│  ┌─────────────┐    ┌──────────────────────┐    │
│  │ Scheduler   │───>│ TradingAgents         │    │
│  │ (daily cron)│    │  - NVIDIA NIM (free)  │    │
│  └─────────────┘    │  - yfinance (free)    │    │
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

## Setup

### 1. Clone & Install

```bash
cd ~/tradingbot
python3 -m venv venv
source venv/bin/activate
pip install .
pip install alpaca-py
```

### 2. Configure API Keys

Edit `.env` in the tradingbot root:

```bash
NVIDIA_API_KEY=your_key_here
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_key_here
```

Get free API keys:
- NVIDIA NIM: https://build.nvidia.com/
- Alpaca Paper: https://app.alpaca.markets/paper/dashboard

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
python bot/backtest.py --interval 5 --output ./results
```

## Files

| File | Purpose |
|------|---------|
| `bot.py` | Main orchestrator |
| `alpaca_client.py` | Alpaca API wrapper |
| `risk_manager.py` | Position sizing & stop-loss |
| `universe.py` | Stock universe management |
| `backtest.py` | Historical testing |
| `scheduler.py` | Cron entry point |
| `config.json` | Configuration |
| `setup.py` | Environment verification |

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

## Cron Setup

Run daily at market open (9:30 AM ET):

```bash
crontab -e
```

Add:
```bash
30 9 * * 1-5 cd ~/tradingbot && source venv/bin/activate && python bot/scheduler.py daily
```

## Risk Warning

This is for educational purposes. Trading involves risk of loss. Start with paper trading.
