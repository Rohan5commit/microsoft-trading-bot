"""Setup script to verify environment and dependencies."""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def check_env():
    """Check environment variables."""
    print("Checking environment variables...")

    load_dotenv(Path(__file__).parent.parent / ".env")

    required = {
        "NVIDIA_API_KEY": "NVIDIA NIM API key",
        "TWELVE_DATA_KEYS": "Twelve Data API keys (comma-separated)",
    }

    optional = {
        "ALPACA_API_KEY": "Alpaca API key (needed for live trading)",
        "ALPACA_SECRET_KEY": "Alpaca secret key (needed for live trading)",
        "ALPACA_BASE_URL": "Alpaca base URL (default: paper trading)",
    }

    missing = []
    for var, desc in required.items():
        value = os.getenv(var)
        if not value or value.startswith("your_"):
            missing.append(f"  {var}: {desc}")
        else:
            # For TWELVE_DATA_KEYS, check it has at least one key
            if var == "TWELVE_DATA_KEYS":
                keys = [k.strip() for k in value.split(",") if k.strip()]
                if len(keys) > 0:
                    print(f"  ✓ {var} ({len(keys)} keys)")
                else:
                    missing.append(f"  {var}: {desc} (no valid keys found)")
            else:
                print(f"  ✓ {var}")

    for var, desc in optional.items():
        value = os.getenv(var)
        if value and not value.startswith("your_"):
            print(f"  ✓ {var}")

    if missing:
        print("\n✗ Missing required environment variables:")
        for m in missing:
            print(m)
        print("\nPlease set them in .env file")
        return False

    return True


def check_dependencies():
    """Check Python dependencies."""
    print("\nChecking dependencies...")

    deps = [
        ("requests", "requests"),
        ("alpaca.trading", "alpaca-py"),
        ("langchain_core", "langchain-core"),
        ("langgraph", "langgraph"),
        ("tradingagents", "tradingagents"),
    ]

    missing = []
    for module, package in deps:
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            missing.append(package)
            print(f"  ✗ {package}")

    if missing:
        print(f"\n✗ Missing dependencies: {', '.join(missing)}")
        print("Run: pip install .")
        return False

    return True


def check_config():
    """Check configuration file."""
    print("\nChecking configuration...")

    config_path = Path(__file__).parent / "config.json"
    if not config_path.exists():
        print(f"  ✗ Config file not found: {config_path}")
        return False

    try:
        with open(config_path) as f:
            config = json.load(f)
        print(f"  ✓ Config loaded")

        # Check required sections
        required_sections = ["universe", "risk", "trading", "llm"]
        for section in required_sections:
            if section in config:
                print(f"    ✓ {section}")
            else:
                print(f"    ✗ {section} missing")
                return False

        return True
    except json.JSONDecodeError as e:
        print(f"  ✗ Invalid JSON: {e}")
        return False


def check_twelve_data():
    """Test Twelve Data connection."""
    print("\nTesting Twelve Data connection...")

    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent / ".env")

        from twelve_data import TwelveDataClient
        client = TwelveDataClient()

        quote = client.get_quote("AAPL")
        if quote and quote.get("close", 0) > 0:
            print(f"  ✓ Twelve Data working (AAPL: ${quote['close']:.2f})")
            return True
        else:
            print(f"  ✗ Twelve Data returned invalid data")
            return False
    except Exception as e:
        print(f"  ✗ Twelve Data test failed: {e}")
        return False


def main():
    """Run setup checks."""
    print("=" * 50)
    print("TRADING BOT SETUP CHECK")
    print("=" * 50)

    results = {
        "env": check_env(),
        "deps": check_dependencies(),
        "config": check_config(),
        "twelve_data": check_twelve_data(),
    }

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    all_ok = all(results.values())
    for key, ok in results.items():
        status = "✓ OK" if ok else "✗ FAILED"
        print(f"  {key}: {status}")

    if all_ok:
        print("\n✓ All checks passed! Ready to run the bot.")
        print("\nNext steps:")
        print("  1. Test with: python bot/bot.py --analyze NVDA")
        print("  2. Run daily: python bot/bot.py --daily")
        print("  3. Check status: python bot/bot.py --status")
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
