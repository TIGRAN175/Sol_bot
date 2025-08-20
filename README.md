
# SOL Swing Bot (Binance Spot)

Automates SOL/USDT swing cycles:
- **Buys at thresholds of:** 40% @ 170, 40% @ 165, 20% @ 160
- **Sells at thershholds of:** 40% @ 180, 40% @ 190, 20% @ 200 with fail-safe to 191.5 if price < 192

## Quick start (local)
1. Install Python 3.11+
2. `python -m venv .venv` and activate it
3. `pip install -r requirements.txt`
4. Copy `.env.example` to `.env`, set keys (testnet first)
5. Run: `python bot.py`

## Testnet
Set `BINANCE_TESTNET=true` in `.env` and use testnet keys.

## Notes
- Uses LIMIT_MAKER where possible
- Respects min notional, tick size, and lot size
- Logs to `./logs/events.log` and `./logs/trades.csv`

## Docker
```
docker compose up --build -d
```
