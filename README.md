# Microsoft Trading Bot

AI-powered trading bot using TradingAgents + NVIDIA NIM + Twelve Data.

## How It Works

Two-phase analysis for **full quality at speed**:

```
Phase 1 (5 min):     Quick scan 1000 stocks -> Top 20 candidates
Phase 2 (90 min):    Full TradingAgents deep analysis on candidates
```

| Phase | Method | Time | Quality |
|-------|--------|------|---------|
| 1 | Single LLM call per stock | ~5 min for 1000 | Quick filter |
| 2 | 8+ agents, bull/bear debate | ~5 min per stock | Full research |

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
```

Required:
- `NVIDIA_API_KEY` — Free from [build.nvidia.com](https://build.nvidia.com/)
- `TWELVE_DATA_KEYS` — From [twelvedata.com](https://twelvedata.com/)

### 3. Verify Setup

```bash
python bot/setup.py
```

## Usage

### Analyze Specific Stocks

```bash
python bot/two_phase_scheduler.py --tickers NVDA AAPL MSFT --deep-count 3
```

### Full Universe (1000 stocks)

```bash
python bot/two_phase_scheduler.py --deep-count 20
```

## GitHub Actions

Runs daily at 9:30 AM ET on weekdays.

### Manual Trigger

```bash
gh workflow run daily-analysis.yml

# With specific tickers
gh workflow run daily-analysis.yml -f tickers="NVDA,AAPL,MSFT" -f deep_count=5
```

### Secrets Required

Set in GitHub repo Settings > Secrets:

| Secret | Required |
|--------|----------|
| `NVIDIA_API_KEY` | Yes |
| `TWELVE_DATA_KEYS` | Yes |
| `ALPACA_API_KEY` | Optional (for trading) |
| `ALPACA_SECRET_KEY` | Optional (for trading) |

## Files

| File | Purpose |
|------|---------|
| `two_phase_bot.py` | Main two-phase analysis engine |
| `two_phase_scheduler.py` | Entry point for GitHub Actions |
| `twelve_data.py` | Market data with 8-key rotation |
| `risk_manager.py` | Position sizing & risk rules |
| `universe.py` | Stock universe management |
| `config.json` | Configuration |

## Configuration

Edit `bot/config.json`:

```json
{
  "universe": {
    "max_stocks": 1000,
    "min_market_cap_billion": 2
  },
  "deep_analysis": {
    "count": 20,
    "min_conviction": 0.6
  },
  "llm": {
    "provider": "nvidia",
    "deep_think_model": "meta/llama-3.1-70b-instruct",
    "quick_think_model": "meta/llama-3.1-8b-instruct"
  }
}
```

## Risk Warning

This is for educational purposes. Trading involves risk of loss. Start with paper trading.
