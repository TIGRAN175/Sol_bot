# bot.py
# SOL/USDT swing bot for Binance Spot
# Buys:  40% @ 170, 40% @ 165, 20% @ 160
# Sells: 40% @ 180, 40% @ 190, 20% @ 200 (fail-safe: if price < 192 -> 191.5)
# - One order per level (per arm)
# - Re-arm SELL legs on any BUY fill; re-arm BUY legs on any SELL fill
# - LIMIT_MAKER safety to avoid -2010 errors
# - JSON + CSV logging; state persisted between runs

import os, time, json, csv, logging
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from dotenv import load_dotenv, find_dotenv
from binance.spot import Spot as Binance
from twilio.rest import Client as TwilioClient

# =========================
# ENV + CONSTANTS
# =========================
ENV_PATH = Path(__file__).resolve().parent / ".env"
env_loaded_from = ENV_PATH if ENV_PATH.exists() else find_dotenv()
load_dotenv(env_loaded_from)

API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")
USE_TESTNET = (os.getenv("BINANCE_TESTNET", "false").strip().lower() in ("true","1","yes"))

SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
BASE   = os.getenv("BASE", "SOL").strip().upper()
QUOTE  = os.getenv("QUOTE", "USDT").strip().upper()
MARKET = f"{BASE}/{QUOTE}"

# Strategy
BUY_LEVELS  = [Decimal("175"), Decimal("165"), Decimal("160")]
BUY_SPLITS  = [Decimal("0.30"), Decimal("0.50"), Decimal("0.20")]
SELL_LEVELS = [Decimal("185"), Decimal("190"), Decimal("200")]
SELL_SPLITS = [Decimal("0.40"), Decimal("0.40"), Decimal("0.20")]
TP3_FAILSAFE_TRIGGER = Decimal("192")
TP3_FAILSAFE_LIMIT   = Decimal("191.5")

POLL_SECONDS = int(os.getenv("POLL_SECONDS", "3"))
FEE_BUFFER   = Decimal(os.getenv("FEE_BUFFER", "0.0015"))  # 0.15%

# Paths
LOG_DIR   = Path(os.getenv("LOG_DIR", "./logs"))
STATE_DIR = Path(os.getenv("STATE_DIR", str(LOG_DIR)))
LOG_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

logfile            = str(LOG_DIR / "solbot.log")
trades_csv_path    = LOG_DIR / "trades.csv"
state_path         = STATE_DIR / "state.json"
last_trade_id_path = STATE_DIR / "last_trade_id.txt"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(logfile, mode="a"), logging.StreamHandler()]
)

from twilio.rest import Client as TwilioClient


def notify_whatsapp(text: str):
    try:
        w_from = os.getenv("WHATSAPP_FROM")
        w_to   = os.getenv("WHATSAPP_TO")
        if not (w_from and w_to):
            return

        api_sid    = os.getenv("TWILIO_API_KEY_SID")
        api_secret = os.getenv("TWILIO_API_KEY_SECRET")
        acct_sid   = os.getenv("TWILIO_ACCOUNT_SID")
        auth_tok   = os.getenv("TWILIO_AUTH_TOKEN")

        # Prefer API Key auth if present (must include Account SID as 3rd arg)
        if api_sid and api_secret and acct_sid:
            client = TwilioClient(api_sid, api_secret, acct_sid)
        elif acct_sid and auth_tok:
            client = TwilioClient(acct_sid, auth_tok)
        else:
            # Not configured correctly; skip quietly
            return

        client.messages.create(from_=w_from, to=w_to, body=text[:1500])
    except Exception as e:
        logging.error(f"notify_whatsapp failed: {e}")
        log_json({"ts": dt_iso(), "event": "error", "where": "notify_whatsapp", "error": str(e)}, level=logging.ERROR)






def dt_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def log_json(payload: dict, level=logging.INFO):
    try:
        logging.log(level, json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
    except Exception as e:
        logging.error(f"json_log_failed: {e} payload={payload}")

def log_trade_csv(side, price, qty, note, order_id=""):
    header = ["ts_iso","network","symbol","market","side","price","qty","quote_value","order_id","note"]
    write_header = not trades_csv_path.exists()
    with open(trades_csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow([dt_iso(), NETWORK, SYMBOL, MARKET, side.upper(), str(price), str(qty), str(price*qty), str(order_id), note])

# =========================
# CLIENT + EXCHANGE INFO
# =========================
if not API_KEY or not API_SECRET:
    raise SystemExit("Missing API keys (.env: BINANCE_API_KEY / BINANCE_API_SECRET).")

client = Binance(
    api_key=API_KEY,
    api_secret=API_SECRET,
    base_url="https://testnet.binance.vision" if USE_TESTNET else None
)

NETWORK = "TESTNET" if USE_TESTNET else "MAINNET"
logging.info(f"Started bot | network={NETWORK} | symbol={SYMBOL} | env_loaded_from={env_loaded_from}")

ex_info = client.exchange_info()
SYMBOLS = {s["symbol"] for s in ex_info["symbols"]}
if SYMBOL not in SYMBOLS:
    raise SystemExit(f"Symbol {SYMBOL} not listed on this environment.")

def exchange_filters(symbol: str):
    info = client.exchange_info(symbol=symbol)
    s = info["symbols"][0]
    filters = {flt["filterType"]: flt for flt in s["filters"]}
    min_notional = Decimal(
        filters.get("NOTIONAL", {}).get("minNotional",
        filters.get("MIN_NOTIONAL", {}).get("minNotional", "0"))
    )
    return {
        "tickSize": Decimal(filters["PRICE_FILTER"]["tickSize"]),
        "stepSize": Decimal(filters["LOT_SIZE"]["stepSize"]),
        "minQty": Decimal(filters["LOT_SIZE"]["minQty"]),
        "minNotional": min_notional
    }

FILTERS = exchange_filters(SYMBOL)

def round_price(p: Decimal) -> Decimal:
    ts = FILTERS["tickSize"]
    return (p/ts).to_integral_value(rounding=ROUND_DOWN) * ts

def round_qty(q: Decimal) -> Decimal:
    ss = FILTERS["stepSize"]
    q = (q/ss).to_integral_value(rounding=ROUND_DOWN) * ss
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
    return Decimal(client.ticker_price(symbol=SYMBOL)["price"])

def find_open_by_exact_price(side: str, price: Decimal):
    """Return list of open orders of 'side' exactly at rounded price."""
    oo = open_orders()
    target = round_price(price)
    hits = []
    for od in oo:
        if od["side"].upper() != side.upper():
            continue
        if Decimal(od["price"]) == target:
            hits.append(od)
    return hits

# =========================
# MAKER SAFETY
# =========================
def maker_safe(side: str, level: Decimal, last: Decimal) -> bool:
    """LIMIT_MAKER rules: SELL must be above market; BUY must be below market."""
    return (level > last) if side.lower() == "sell" else (level < last)

# =========================
# STATE (arming + last_trade_id)
# =========================
DEFAULT_STATE = {
    "armed": {
        "buys":  [True, True, True],   # one order per level when armed
        "sells": [True, True, True]
    },
    "last_trade_id": -1
}

def load_state() -> dict:
    if state_path.exists():
        try:
            return json.loads(state_path.read_text())
        except Exception:
            pass
    return DEFAULT_STATE.copy()

def save_state(st: dict):
    tmp = json.dumps(st, indent=0, separators=(",", ":"))
    state_path.write_text(tmp)

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

# =========================
# PLACEMENT HELPERS (single order per level)
# =========================
def place_limit_maker(side: str, price: Decimal, qty: Decimal, tag: str = "") -> dict:
    if qty <= 0: return {}
    price = round_price(price)
    qty   = round_qty(qty)
    if qty <= 0 or not notional_ok(price, qty): return {}
    try:
        od = client.new_order(
            symbol=SYMBOL, side=side.upper(), type="LIMIT_MAKER",
            price=str(price), quantity=str(qty),
            newClientOrderId=f"{tag}-{int(time.time())}"
        )
        logging.info(f"[{NETWORK}] Placed {side} LIMIT_MAKER {qty} @ {price} tag={tag} id={od.get('orderId')}")
        log_trade_csv(side, price, qty, f"placed {tag}", str(od.get("orderId","")))
        log_json({
            "ts": dt_iso(), "event": "order_placed", "network": NETWORK,
            "symbol": SYMBOL, "market": MARKET, "side": side.lower(),
            "type": "LIMIT_MAKER", "price": str(price), "qty": str(qty),
            "order_id": str(od.get("orderId","")), "client_order_id": od.get("clientOrderId",""),
            "tag": tag
        })
        return od
    except Exception as e:
        logging.error(f"[{NETWORK}] place_limit_maker failed: {e}")
        log_json({
            "ts": dt_iso(), "event": "error", "where": "place_limit_maker",
            "network": NETWORK, "symbol": SYMBOL, "market": MARKET,
            "side": side.lower(), "price": str(price), "qty": str(qty), "error": str(e), "tag": tag
        }, level=logging.ERROR)
        return {}

def place_limit(side: str, price: Decimal, qty: Decimal, tag: str = "") -> dict:
    if qty <= 0: return {}
    price = round_price(price)
    qty   = round_qty(qty)
    if qty <= 0 or not notional_ok(price, qty): return {}
    try:
        od = client.new_order(
            symbol=SYMBOL, side=side.upper(), type="LIMIT", timeInForce="GTC",
            price=str(price), quantity=str(qty),
            newClientOrderId=f"{tag}-{int(time.time())}"
        )
        logging.info(f"[{NETWORK}] Placed {side} LIMIT {qty} @ {price} tag={tag} id={od.get('orderId')}")
        log_trade_csv(side, price, qty, f"placed {tag}", str(od.get("orderId","")))
        log_json({
            "ts": dt_iso(), "event": "order_placed", "network": NETWORK,
            "symbol": SYMBOL, "market": MARKET, "side": side.lower(),
            "type": "LIMIT", "price": str(price), "qty": str(qty),
            "order_id": str(od.get("orderId","")), "client_order_id": od.get("clientOrderId",""),
            "tag": tag
        })
        return od
    except Exception as e:
        logging.error(f"[{NETWORK}] place_limit failed: {e}")
        log_json({
            "ts": dt_iso(), "event": "error", "where": "place_limit",
            "network": NETWORK, "symbol": SYMBOL, "market": MARKET,
            "side": side.lower(), "price": str(price), "qty": str(qty), "error": str(e), "tag": tag
        }, level=logging.ERROR)
        return {}

def cancel_order(order_id: int):
    try:
        res = client.cancel_order(symbol=SYMBOL, orderId=order_id)
        logging.info(f"[{NETWORK}] Canceled order {order_id}")
        log_json({
            "ts": dt_iso(), "event": "order_canceled", "network": NETWORK,
            "symbol": SYMBOL, "market": MARKET, "order_id": str(order_id)
        })
        return res
    except Exception as e:
        logging.error(f"[{NETWORK}] cancel_order failed: {e}")
        log_json({
            "ts": dt_iso(), "event": "error", "where": "cancel_order",
            "network": NETWORK, "symbol": SYMBOL, "market": MARKET,
            "order_id": str(order_id), "error": str(e)
        }, level=logging.ERROR)
        return {}

# =========================
# STRATEGY (armed per level)
# =========================
def ensure_buy_grid(st: dict):
    base_free, quote_free = balances()

    # Amount of QUOTE already locked in open BUYs (exclude from allocation)
    locked_quote = Decimal("0")
    for od in open_orders():
        if od["side"].upper() == "BUY":
            locked_quote += Decimal(od["origQty"]) * Decimal(od["price"])

    usable_usdt = max(Decimal("0"), quote_free - locked_quote)
    last = get_price()

    for idx, (split, level) in enumerate(zip(BUY_SPLITS, BUY_LEVELS), start=1):
        if not st["armed"]["buys"][idx-1]:
            continue  # disarmed — wait for re-arm signal
        tag = f"BUY{idx}"
        # Enforce one order per level: if any open at that exact level, skip this is to prevent duplicates / fees 
        if find_open_by_exact_price("BUY", level):
            continue
        # LIMIT_MAKER safety
        if not maker_safe("buy", level, last):
            logging.info(f"[{NETWORK}] Skip BUY {level} (maker-unsafe at last={last})")
            continue
        # Allocation for this leg
        alloc = usable_usdt * split
        if alloc <= 0:
            continue
        qty = alloc * (Decimal("1") - FEE_BUFFER) / level
        place_limit_maker("BUY", level, qty, tag)

def ensure_sell_grid(st: dict):
    base_free, _ = balances()

    # Amount of BASE locked in open SELLs (exclude from allocation)
    locked_base = Decimal("0")
    for od in open_orders():
        if od["side"].upper() == "SELL":
            locked_base += Decimal(od["origQty"]) - Decimal(od["executedQty"])

    usable_sol = max(Decimal("0"), base_free - locked_base)
    last = get_price()

    # TP1 (180) and TP2 (190)
    for idx, (split, level) in enumerate(zip(SELL_SPLITS[:2], SELL_LEVELS[:2]), start=1):
        if not st["armed"]["sells"][idx-1]:
            continue  # disarmed — wait for re-arm signal
        tag = f"SELL_TP{idx}"
        if find_open_by_exact_price("SELL", level):
            continue
        if not maker_safe("sell", level, last):
            logging.info(f"[{NETWORK}] Skip SELL {level} (maker-unsafe at last={last})")
            continue
        qty = usable_sol * split * (Decimal("1") - FEE_BUFFER)
        place_limit_maker("SELL", level, qty, tag)

    # TP3 (200) with fail-safe (191.5) in case price starts dropping back down before we reach 200 
    idx = 3
    if st["armed"]["sells"][idx-1]:
        tp3_exists = bool(find_open_by_exact_price("SELL", SELL_LEVELS[2]))
        fs_exists  = bool(find_open_by_exact_price("SELL", TP3_FAILSAFE_LIMIT))

        if not tp3_exists and not fs_exists:
            split = SELL_SPLITS[2]
            qty_tp3 = usable_sol * split * (Decimal("1") - FEE_BUFFER)
            if get_price() < TP3_FAILSAFE_TRIGGER:
                # failsafe uses normal LIMIT; intended to be closer to market
                place_limit("SELL", TP3_FAILSAFE_LIMIT, qty_tp3, "SELL_TP3_FS")
            else:
                if maker_safe("sell", SELL_LEVELS[2], last):
                    place_limit_maker("SELL", SELL_LEVELS[2], qty_tp3, "SELL_TP3")
                else:
                    logging.info(f"[{NETWORK}] Skip SELL {SELL_LEVELS[2]} (maker-unsafe at last={last})")
        elif tp3_exists and get_price() < TP3_FAILSAFE_TRIGGER:
            # switch 200 -> 191.5
            for od in find_open_by_exact_price("SELL", SELL_LEVELS[2]):
                cancel_order(od["orderId"])
            time.sleep(0.5)
            base_free, _ = balances()
            qty_tp3 = base_free * SELL_SPLITS[2] * (Decimal("1") - FEE_BUFFER)
            place_limit("SELL", TP3_FAILSAFE_LIMIT, qty_tp3, "SELL_TP3_FS")

# =========================
# FILLS (myTrades) + ARMS
# =========================
def _closest_level(price: Decimal, levels: list[Decimal], max_ticks: int = 1) -> int:
    """Return index of the closest level within max_ticks; else -1."""
    rp = round_price(price)
    for i, lvl in enumerate(levels):
        if abs(rp - round_price(lvl)) <= FILTERS["tickSize"] * max_ticks:
            return i
    return -1

def scan_new_fills_and_update_state(st: dict):
    """Poll recent trades; mark per-level arms and re-arms based on fills."""
    last_seen = _load_last_trade_id() if st.get("last_trade_id", -1) == -1 else st["last_trade_id"]
    try:
        trades = client.my_trades(symbol=SYMBOL, limit=50)  # most recent first
    except Exception as e:
        logging.error(f"[{NETWORK}] my_trades failed: {e}")
        log_json({
            "ts": dt_iso(), "event": "error", "where": "my_trades",
            "network": NETWORK, "symbol": SYMBOL, "market": MARKET, "error": str(e)
        }, level=logging.ERROR)
        return

    trades_sorted = sorted(trades, key=lambda t: int(t["id"]))
    max_id = last_seen
    buy_filled = False
    sell_filled = False

    for t in trades_sorted:
        tid = int(t["id"])
        if tid <= last_seen:
            continue

        price = Decimal(t.get("price", "0"))
        qty   = Decimal(t.get("qty", "0"))
        side  = "buy" if t.get("isBuyer") else "sell"
        order_id = str(t.get("orderId", ""))
        ts_ms = int(t.get("time", 0))
        ts_iso = datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc).isoformat()

        # CSV + JSON log for the fill
        log_trade_csv(side.upper(), price, qty, "fill", order_id)
        log_json({
            "ts": ts_iso, "event": "fill", "network": NETWORK, "symbol": SYMBOL, "market": MARKET,
            "side": side, "price": str(price), "qty": str(qty), "quote_qty": t.get("quoteQty"),
            "order_id": order_id, "trade_id": str(tid), "commission": t.get("commission"),
            "commission_asset": t.get("commissionAsset"), "is_maker": bool(t.get("isMaker"))
        })

        # ✅ WhatsApp notify on each fill (now variables are defined)
        try:
            side_emoji = "🟢 BUY" if side == "buy" else "🔴 SELL"
            msg = (
                f"{side_emoji} {MARKET}\n"
                f"Price: {price}\n"
                f"Qty:   {qty}\n"
                f"Quote: {t.get('quoteQty')}\n"
                f"Maker: {bool(t.get('isMaker'))}\n"
                f"Order: {order_id}\n"
                f"Trade: {tid}\n"
                f"Net:   {NETWORK}"
                
            )
            notify_whatsapp(msg)
        except Exception:
            pass

        # Map fill to a level and disarm that level
        if side == "buy":
            idx = _closest_level(price, BUY_LEVELS, max_ticks=1)
            if idx != -1 and st["armed"]["buys"][idx]:
                st["armed"]["buys"][idx] = False
            buy_filled = True
        else:
            idx = _closest_level(price, SELL_LEVELS, max_ticks=1)
            if idx == -1 and abs(round_price(price) - round_price(TP3_FAILSAFE_LIMIT)) <= FILTERS["tickSize"]:
                idx = 2  # treat fail-safe as TP3
            if idx != -1 and st["armed"]["sells"][idx]:
                st["armed"]["sells"][idx] = False
            sell_filled = True

        if tid > max_id:
            max_id = tid

    # Re-arm logic
    if buy_filled:
        st["armed"]["sells"] = [True, True, True]
        log_json({"ts": dt_iso(), "event": "rearm", "side": "sells", "reason": "buy_fill"})
    if sell_filled:
        st["armed"]["buys"] = [True, True, True]
        log_json({"ts": dt_iso(), "event": "rearm", "side": "buys", "reason": "sell_fill"})

    if max_id > last_seen:
        st["last_trade_id"] = max_id
        _save_last_trade_id(max_id)


# =========================
# MAIN LOOP
# =========================
def main():
    st = load_state()

    # Startup visibility
    acct = client.account()
    logging.info(f"[{NETWORK}] Startup balances:")
    for b in acct["balances"]:
        if float(b["free"]) + float(b["locked"]) > 0:
            logging.info(f"   {b['asset']}: free={b['free']} locked={b['locked']}")

    log_json({
        "ts": dt_iso(), "event": "bot_start", "network": NETWORK,
        "symbol": SYMBOL, "market": MARKET, "env_loaded_from": str(env_loaded_from),
        "armed": st["armed"]
    })

    while True:
        try:
            # Observe current fills and update arming state
            scan_new_fills_and_update_state(st)
            save_state(st)

            # Make sure only one order gets placed per armed level so, 180, 190, 200 or 191.5 
            ensure_buy_grid(st)
            ensure_sell_grid(st)

            # Heartbeat 
            price = get_price()
            base_free, quote_free = balances()
            open_cnt = len(open_orders())

            logging.info(f"[{NETWORK}] Price={price:.2f} | {BASE}={base_free:.4f} | {QUOTE}={quote_free:.2f} | open_orders={open_cnt}")
            log_json({
                "ts": dt_iso(), "event": "heartbeat", "network": NETWORK, "symbol": SYMBOL,
                "market": MARKET, "price": str(price),
                "balances": {BASE: str(base_free), QUOTE: str(quote_free)},
                "open_orders": open_cnt,
                "armed": st["armed"]
            })

            time.sleep(POLL_SECONDS)

        except KeyboardInterrupt:
            logging.info(f"[{NETWORK}] Stopping…")
            log_json({"ts": dt_iso(), "event": "bot_stop", "network": NETWORK, "symbol": SYMBOL, "market": MARKET})
            break
        except Exception as e:
            logging.error(f"[{NETWORK}] main loop error: {e}")
            log_json({"ts": dt_iso(), "event": "error", "where": "main_loop",
            "network": NETWORK, "symbol": SYMBOL, "market": MARKET, "error": str(e)}, level=logging.ERROR)
            time.sleep(max(5, POLL_SECONDS))

if __name__ == "__main__":
    main()
