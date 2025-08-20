# bot.py
# SOL/USDT swing bot for Binance Spot
# - Buys: 40% @ 170, 40% @ 165, 20% @ 160
# - Sells: 40% @ 180, 40% @ 190, 20% @ 200 (fail-safe: if price < 192, switch TP3 to 191.5)
# - Reads config from .env (BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_TESTNET, SYMBOL, BASE, QUOTE, etc.)

import os
import time
import csv
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN

from dotenv import load_dotenv, find_dotenv
from binance.spot import Spot as Binance

# =========================
# ENV + CLIENT INITIALIZE
# =========================

# Load .env explicitly from the same folder as this file; fall back to search upwards if needed
ENV_PATH = Path(__file__).resolve().parent / ".env"
env_loaded_from = ENV_PATH if ENV_PATH.exists() else find_dotenv()
load_dotenv(env_loaded_from)

API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# tolerate whitespace/comments after the value
_raw_flag = (os.getenv("BINANCE_TESTNET", "false") or "false").strip().split()[0].lower()
USE_TESTNET = _raw_flag in ("true", "1", "yes")

SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
BASE = os.getenv("BASE", "SOL").strip().upper()
QUOTE = os.getenv("QUOTE", "USDT").strip().upper()
MARKET = f"{BASE}/{QUOTE}"

# Buy levels (USDT -> BASE)
BUY_SPLITS = [0.40, 0.40, 0.20]  # 40% / 40% / 20%
BUY_LEVELS = [Decimal("170"), Decimal("165"), Decimal("160")]

# Sell levels (BASE -> QUOTE)
SELL_SPLITS = [0.40, 0.40, 0.20]  # 40% / 40% / 20%
SELL_LEVELS = [Decimal("180"), Decimal("190"), Decimal("200")]

# TP3 fail-safe: if price drops below TRIGGER and TP3 not filled, cancel 200 and place 191.5
TP3_FAILSAFE_TRIGGER = Decimal("192")
TP3_FAILSAFE_LIMIT   = Decimal("191.5")

POLL_SECONDS = int(os.getenv("POLL_SECONDS", "3"))
FEE_BUFFER = Decimal(os.getenv("FEE_BUFFER", "0.0015"))  # 0.15% buffer

# Directories
LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
STATE_DIR = Path(os.getenv("STATE_DIR", str(LOG_DIR)))  # default to LOG_DIR
LOG_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

# Files
logfile = str(LOG_DIR / "solbot.log")
trades_csv_path = LOG_DIR / "trades.csv"
last_trade_id_path = STATE_DIR / "last_trade_id.txt"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(logfile, mode="a"), logging.StreamHandler()]
)

def dt_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def log_json(payload: dict, level: int = logging.INFO):
    """Emit a single JSON line to stdout/file handlers."""
    try:
        logging.log(level, json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
    except Exception as e:
        logging.error(f"json_log_failed: {e} payload={payload}")

def log_trade_csv(side: str, price: Decimal, qty: Decimal, note: str, order_id: str = ""):
    """Append CSV line for trades/placements/cancels to trades.csv."""
    header = ["ts_iso","network","symbol","market","side","price","qty","quote_value","order_id","note"]
    write_header = not trades_csv_path.exists()
    with open(trades_csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow([dt_iso(), NETWORK, SYMBOL, MARKET, side.upper(), str(price), str(qty), str(price*qty), str(order_id), note])

# Sanity checks
if not API_KEY or not API_SECRET:
    raise SystemExit("Missing API keys. Put them in .env as BINANCE_API_KEY / BINANCE_API_SECRET")

# Client: ONLY pass base_url on testnet; never pass None on mainnet
if USE_TESTNET:
    client = Binance(api_key=API_KEY, api_secret=API_SECRET, base_url="https://testnet.binance.vision")
else:
    client = Binance(api_key=API_KEY, api_secret=API_SECRET)

NETWORK = "TESTNET" if USE_TESTNET else "MAINNET"
logging.info(f"Started bot | network={NETWORK} | symbol={SYMBOL} | env_loaded_from={env_loaded_from}")

# Verify symbol exists on the connected environment
ex_info = client.exchange_info()
SYMBOLS = {s["symbol"] for s in ex_info["symbols"]}
if SYMBOL not in SYMBOLS:
    raise SystemExit(f"Symbol {SYMBOL} not listed on this environment. Try BTCUSDT or ETHUSDT on testnet.")

# =========================
# EXCHANGE FILTERS & ROUNDS
# =========================

def exchange_filters(symbol: str):
    info = client.exchange_info(symbol=symbol)
    s = info["symbols"][0]
    filters = {flt["filterType"]: flt for flt in s["filters"]}
    # MIN_NOTIONAL vs NOTIONAL varies by deployment
    min_notional = Decimal(
        filters.get("NOTIONAL", {}).get("minNotional",
        filters.get("MIN_NOTIONAL", {}).get("minNotional", "0"))
    )
    return {
        "tickSize": Decimal(filters["PRICE_FILTER"]["tickSize"]),
        "stepSize": Decimal(filters["LOT_SIZE"]["stepSize"]),
        "minQty":    Decimal(filters["LOT_SIZE"]["minQty"]),
        "minNotional": min_notional
    }

FILTERS = exchange_filters(SYMBOL)

def round_price(p: Decimal) -> Decimal:
    ts = FILTERS["tickSize"]
    return (p / ts).to_integral_value(rounding=ROUND_DOWN) * ts  # floor to tick

def round_qty(q: Decimal) -> Decimal:
    ss = FILTERS["stepSize"]
    q = (q / ss).to_integral_value(rounding=ROUND_DOWN) * ss
    return q if q >= FILTERS["minQty"] else Decimal("0")

def notional_ok(price: Decimal, qty: Decimal) -> bool:
    return (price * qty) >= FILTERS["minNotional"]

# =========================
# EXCHANGE HELPERS
# =========================

def balances():
    acct = client.account()
    balmap = {b["asset"]: Decimal(b["free"]) for b in acct["balances"]}
    return balmap.get(BASE, Decimal("0")), balmap.get(QUOTE, Decimal("0"))

def open_orders():
    return client.get_open_orders(symbol=SYMBOL)

def get_price() -> Decimal:
    px = client.ticker_price(symbol=SYMBOL)
    return Decimal(px["price"])

def find_open_by_price(side: str, price: Decimal, tol_ticks: int = 0):
    """Return list of open orders of 'side' around 'price' (within tol_ticks)."""
    oo = open_orders()
    hits = []
    cmp = round_price(price)
    for od in oo:
        if od["side"].upper() != side.upper():
            continue
        op = Decimal(od["price"])
        if tol_ticks == 0:
            if op == cmp:
                hits.append(od)
        else:
            ts = FILTERS["tickSize"]
            if abs(op - cmp) <= (ts * tol_ticks):
                hits.append(od)
    return hits

# =========================
# ORDER ACTIONS (with JSON logs)
# =========================

def place_limit_maker(side: str, price: Decimal, qty: Decimal, tag: str = "") -> dict:
    """Post-only (LIMIT_MAKER) to seek maker fees where possible."""
    if qty <= 0:
        return {}
    price = round_price(price)
    qty = round_qty(qty)
    if qty <= 0 or not notional_ok(price, qty):
        return {}
    params = {
        "symbol": SYMBOL,
        "side": side.upper(),
        "type": "LIMIT_MAKER",
        "price": str(price),
        "quantity": str(qty),
        "newClientOrderId": f"{tag}-{int(time.time())}",
    }
    try:
        od = client.new_order(**params)
        msg = f"[{NETWORK}] Placed {side} LIMIT_MAKER {qty} @ {price} tag={tag} id={od.get('orderId')}"
        logging.info(msg)
        log_trade_csv(side, price, qty, f"placed {tag}", str(od.get("orderId","")))
        log_json({
            "ts": dt_iso(),
            "event": "order_placed",
            "network": NETWORK,
            "symbol": SYMBOL,
            "market": MARKET,
            "side": side.lower(),
            "type": "LIMIT_MAKER",
            "price": str(price),
            "qty": str(qty),
            "order_id": str(od.get("orderId","")),
            "client_order_id": od.get("clientOrderId",""),
            "tag": tag
        })
        return od
    except Exception as e:
        logging.error(f"[{NETWORK}] place_limit_maker failed: {e}")
        log_json({
            "ts": dt_iso(),
            "event": "error",
            "where": "place_limit_maker",
            "network": NETWORK,
            "symbol": SYMBOL,
            "market": MARKET,
            "side": side.lower(),
            "price": str(price),
            "qty": str(qty),
            "error": str(e),
            "tag": tag
        }, level=logging.ERROR)
        return {}

def place_limit(side: str, price: Decimal, qty: Decimal, tag: str = "") -> dict:
    """Standard GTC limit."""
    if qty <= 0:
        return {}
    price = round_price(price)
    qty = round_qty(qty)
    if qty <= 0 or not notional_ok(price, qty):
        return {}
    params = {
        "symbol": SYMBOL,
        "side": side.upper(),
        "type": "LIMIT",
        "timeInForce": "GTC",
        "price": str(price),
        "quantity": str(qty),
        "newClientOrderId": f"{tag}-{int(time.time())}",
    }
    try:
        od = client.new_order(**params)
        msg = f"[{NETWORK}] Placed {side} LIMIT {qty} @ {price} tag={tag} id={od.get('orderId')}"
        logging.info(msg)
        log_trade_csv(side, price, qty, f"placed {tag}", str(od.get("orderId","")))
        log_json({
            "ts": dt_iso(),
            "event": "order_placed",
            "network": NETWORK,
            "symbol": SYMBOL,
            "market": MARKET,
            "side": side.lower(),
            "type": "LIMIT",
            "price": str(price),
            "qty": str(qty),
            "order_id": str(od.get("orderId","")),
            "client_order_id": od.get("clientOrderId",""),
            "tag": tag
        })
        return od
    except Exception as e:
        logging.error(f"[{NETWORK}] place_limit failed: {e}")
        log_json({
            "ts": dt_iso(),
            "event": "error",
            "where": "place_limit",
            "network": NETWORK,
            "symbol": SYMBOL,
            "market": MARKET,
            "side": side.lower(),
            "price": str(price),
            "qty": str(qty),
            "error": str(e),
            "tag": tag
        }, level=logging.ERROR)
        return {}

def cancel_order(order_id: int):
    try:
        res = client.cancel_order(symbol=SYMBOL, orderId=order_id)
        logging.info(f"[{NETWORK}] Canceled order {order_id}")
        log_trade_csv("CANCEL", Decimal("0"), Decimal("0"), "canceled", str(order_id))
        log_json({
            "ts": dt_iso(),
            "event": "order_canceled",
            "network": NETWORK,
            "symbol": SYMBOL,
            "market": MARKET,
            "order_id": str(order_id)
        })
        return res
    except Exception as e:
        logging.error(f"[{NETWORK}] cancel_order failed: {e}")
        log_json({
            "ts": dt_iso(),
            "event": "error",
            "where": "cancel_order",
            "network": NETWORK,
            "symbol": SYMBOL,
            "market": MARKET,
            "order_id": str(order_id),
            "error": str(e)
        }, level=logging.ERROR)
        return {}

# =========================
# STRATEGY
# =========================

def ensure_buy_grid():
    """Place staggered BUY orders at 170/165/160 using available QUOTE (USDT)."""
    base_free, quote_free = balances()
    # Exclude QUOTE already locked in open BUY orders
    locked = Decimal("0")
    for od in open_orders():
        if od["side"].upper() == "BUY":
            locked += Decimal(od["origQty"]) * Decimal(od["price"])
    usable_usdt = max(Decimal("0"), quote_free - locked)

    targets = list(zip(BUY_SPLITS, BUY_LEVELS))
    for idx, (split, level) in enumerate(targets, start=1):
        tag = f"BUY{idx}"
        if find_open_by_price("BUY", level):
            continue
        alloc = usable_usdt * Decimal(str(split))
        if alloc <= 0:
            continue
        qty = alloc * (Decimal("1") - FEE_BUFFER) / level
        place_limit_maker("BUY", level, qty, tag)

def ensure_sell_grid():
    """Place staggered SELL orders at 180/190 and manage the TP3 200/192 logic."""
    base_free, _ = balances()
    # Exclude BASE already locked in open SELL orders
    locked = Decimal("0")
    for od in open_orders():
        if od["side"].upper() == "SELL":
            locked += Decimal(od["origQty"]) - Decimal(od["executedQty"])
    usable_sol = max(Decimal("0"), base_free - locked)

    # Two standard sell limits: 180 & 190 (40% each)
    for idx, (split, level) in enumerate(zip(SELL_SPLITS[:2], SELL_LEVELS[:2]), start=1):
        tag = f"SELL_TP{idx}"
        if find_open_by_price("SELL", level):
            continue
        qty = usable_sol * Decimal(str(split)) * (Decimal("1") - FEE_BUFFER)
        place_limit_maker("SELL", level, qty, tag)

    # TP3 with fail-safe: 20% at 200; if price < 192 and 200 not filled, switch to 191.5
    third_split = SELL_SPLITS[2]
    qty_tp3 = usable_sol * Decimal(str(third_split)) * (Decimal("1") - FEE_BUFFER)
    tp3_exists = bool(find_open_by_price("SELL", SELL_LEVELS[2]))
    fs_exists  = bool(find_open_by_price("SELL", TP3_FAILSAFE_LIMIT))

    last = get_price()

    if not tp3_exists and not fs_exists:
        if last < TP3_FAILSAFE_TRIGGER:
            place_limit("SELL", TP3_FAILSAFE_LIMIT, qty_tp3, "SELL_TP3_FS")
        else:
            place_limit("SELL", SELL_LEVELS[2], qty_tp3, "SELL_TP3")
    else:
        if tp3_exists and last < TP3_FAILSAFE_TRIGGER:
            # Cancel 200s and replace with 191.5
            for od in find_open_by_price("SELL", SELL_LEVELS[2]):
                cancel_order(od["orderId"])
            time.sleep(0.5)
            base_free, _ = balances()
            qty_tp3 = base_free * Decimal(str(third_split)) * (Decimal("1") - FEE_BUFFER)
            place_limit("SELL", TP3_FAILSAFE_LIMIT, qty_tp3, "SELL_TP3_FS")

# =========================
# FILLS (myTrades polling)
# =========================

def _load_last_trade_id() -> int:
    try:
        if last_trade_id_path.exists():
            return int(last_trade_id_path.read_text().strip())
    except Exception:
        pass
    return -1

def _save_last_trade_id(tid: int):
    try:
        last_trade_id_path.write_text(str(tid))
    except Exception as e:
        logging.error(f"save_last_trade_id failed: {e}")

def scan_new_fills():
    """
    Poll recent trades for this symbol and emit JSON/CSV lines for new fills.
    Uses 'id' increasing order. Stores last seen id in STATE_DIR/last_trade_id.txt.
    """
    last_seen = _load_last_trade_id()
    try:
        trades = client.my_trades(symbol=SYMBOL, limit=50)  # most recent first
    except Exception as e:
        logging.error(f"[{NETWORK}] my_trades failed: {e}")
        log_json({
            "ts": dt_iso(),
            "event": "error",
            "where": "my_trades",
            "network": NETWORK,
            "symbol": SYMBOL,
            "market": MARKET,
            "error": str(e)
        }, level=logging.ERROR)
        return

    # sort ascending by id so we process old -> new
    trades_sorted = sorted(trades, key=lambda t: int(t["id"]))
    max_id = last_seen
    for t in trades_sorted:
        tid = int(t["id"])
        if tid <= last_seen:
            continue
        # fields per Binance REST: price, qty, quoteQty, commission, commissionAsset, time, isBuyer, orderId
        price = Decimal(t.get("price", "0"))
        qty = Decimal(t.get("qty", "0"))
        side = "buy" if t.get("isBuyer") else "sell"
        order_id = str(t.get("orderId", ""))
        ts_ms = int(t.get("time", 0))
        ts_iso = datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc).isoformat()

        # CSV + JSON line for the fill
        log_trade_csv(side.upper(), price, qty, "fill", order_id)
        log_json({
            "ts": ts_iso,
            "event": "fill",
            "network": NETWORK,
            "symbol": SYMBOL,
            "market": MARKET,
            "side": side,
            "price": str(price),
            "qty": str(qty),
            "quote_qty": t.get("quoteQty"),
            "order_id": order_id,
            "trade_id": str(tid),
            "commission": t.get("commission"),
            "commission_asset": t.get("commissionAsset"),
            "is_maker": bool(t.get("isMaker")),
        })
        if tid > max_id:
            max_id = tid

    if max_id > last_seen:
        _save_last_trade_id(max_id)

# =========================
# MAIN LOOP
# =========================

def main():
    # Print balances once on start for visibility
    acct = client.account()
    logging.info(f"[{NETWORK}] Startup balances:")
    for b in acct["balances"]:
        if float(b["free"]) + float(b["locked"]) > 0:
            logging.info(f"[{NETWORK}]   {b['asset']}: free={b['free']} locked={b['locked']}")

    # Startup JSON heartbeat
    log_json({
        "ts": dt_iso(),
        "event": "bot_start",
        "network": NETWORK,
        "symbol": SYMBOL,
        "market": MARKET,
        "env_loaded_from": str(env_loaded_from)
    })

    logging.info(f"[{NETWORK}] Running… CTRL+C to stop")

    while True:
        try:
            price = get_price()

            # Strategy
            ensure_buy_grid()
            ensure_sell_grid()

            # Fills (actual executions)
            scan_new_fills()

            base_free, quote_free = balances()
            open_cnt = len(open_orders())

            # Text heartbeat
            logging.info(
                f"[{NETWORK}] Price={price:.2f} | {BASE}={base_free:.4f} | {QUOTE}={quote_free:.2f} | open_orders={open_cnt}"
            )

            # JSON heartbeat (lower frequency if needed)
            log_json({
                "ts": dt_iso(),
                "event": "heartbeat",
                "network": NETWORK,
                "symbol": SYMBOL,
                "market": MARKET,
                "price": str(price),
                "balances": {BASE: str(base_free), QUOTE: str(quote_free)},
                "open_orders": open_cnt
            })

            time.sleep(POLL_SECONDS)

        except KeyboardInterrupt:
            logging.info(f"[{NETWORK}] Stopping…")
            log_json({
                "ts": dt_iso(),
                "event": "bot_stop",
                "network": NETWORK,
                "symbol": SYMBOL,
                "market": MARKET
            })
            break
        except Exception as e:
            logging.error(f"[{NETWORK}] main loop error: {e}")
            log_json({
                "ts": dt_iso(),
                "event": "error",
                "where": "main_loop",
                "network": NETWORK,
                "symbol": SYMBOL,
                "market": MARKET,
                "error": str(e)
            }, level=logging.ERROR)
            time.sleep(max(5, POLL_SECONDS))

if __name__ == "__main__":
    main()
