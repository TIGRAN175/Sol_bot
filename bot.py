# bot.py
# SOL/USDT adaptive swing bot for Binance Spot
# Features:
# - Adaptive grid (resets if drift >10%)
# - One order per level
# - WhatsApp notifications
# - Stop-loss (anchor -15%)
# - Circuit breaker (drop >10% in 15m)

import os, time, json, csv, logging
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from dotenv import load_dotenv, find_dotenv
from binance.spot import Spot as Binance
from twilio.rest import Client as TwilioClient

# =========================
# ENV + CONFIG
# =========================
ENV_PATH = Path(__file__).resolve().parent / ".env"
env_loaded_from = ENV_PATH if ENV_PATH.exists() else find_dotenv()
load_dotenv(env_loaded_from)

API_KEY    = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")
USE_TESTNET = (os.getenv("BINANCE_TESTNET", "false").strip().lower() in ("true","1","yes"))

SYMBOL = os.getenv("SYMBOL", "SOLUSDT").strip().upper()
BASE   = os.getenv("BASE", "SOL").strip().upper()
QUOTE  = os.getenv("QUOTE", "USDT").strip().upper()
MARKET = f"{BASE}/{QUOTE}"

POLL_SECONDS   = int(os.getenv("POLL_SECONDS", "3"))
FEE_BUFFER     = Decimal(os.getenv("FEE_BUFFER", "0.0015"))
ADAPTIVE_SHIFT = Decimal(os.getenv("ADAPTIVE_SHIFT", "0.10"))

# Emergency controls
STOPLOSS_PCT = Decimal("0.15")   # 15% below anchor
CB_DROP_PCT  = Decimal("0.10")   # 10% crash in window
CB_WINDOW    = 900               # 15 minutes in seconds
SLEEP_ON_CB  = 1800              # 30 minutes cooldown

# Grid offsets
BUY_OFFSETS  = [Decimal("-0.05"), Decimal("-0.075"), Decimal("-0.10")]
SELL_OFFSETS = [Decimal("+0.05"), Decimal("+0.10"), Decimal("+0.15")]
SPLITS       = [Decimal("0.40"), Decimal("0.40"), Decimal("0.20")]

# Paths
LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
logfile = str(LOG_DIR / "solbot.log")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(logfile, mode="a"), logging.StreamHandler()]
)

def dt_iso():
    return datetime.now(timezone.utc).isoformat()

def log_json(payload: dict, level=logging.INFO):
    logging.log(level, json.dumps(payload, separators=(",", ":"), ensure_ascii=False))

# WhatsApp notify
def notify_whatsapp(text: str):
    try:
        sid  = os.getenv("TWILIO_ACCOUNT_SID")
        tok  = os.getenv("TWILIO_AUTH_TOKEN")
        w_from = os.getenv("WHATSAPP_FROM")
        w_to   = os.getenv("WHATSAPP_TO")
        if not (sid and tok and w_from and w_to): return
        client = TwilioClient(sid, tok)
        client.messages.create(from_=w_from, to=w_to, body=text[:1500])
    except Exception as e:
        logging.error(f"notify_whatsapp failed: {e}")

# =========================
# CLIENT
# =========================
client = Binance(
    api_key=API_KEY,
    api_secret=API_SECRET,
    base_url="https://testnet.binance.vision" if USE_TESTNET else None
)
NETWORK = "TESTNET" if USE_TESTNET else "MAINNET"

def exchange_filters(symbol: str):
    info = client.exchange_info(symbol=symbol)
    s = info["symbols"][0]
    filters = {f["filterType"]: f for f in s["filters"]}
    return {
        "tickSize": Decimal(filters["PRICE_FILTER"]["tickSize"]),
        "stepSize": Decimal(filters["LOT_SIZE"]["stepSize"]),
        "minQty": Decimal(filters["LOT_SIZE"]["minQty"]),
        "minNotional": Decimal(filters.get("NOTIONAL", {}).get("minNotional", "0"))
    }

FILTERS = exchange_filters(SYMBOL)

def round_price(p: Decimal) -> Decimal:
    ts = FILTERS["tickSize"]
    return (p/ts).to_integral_value(rounding=ROUND_DOWN) * ts

def round_qty(q: Decimal) -> Decimal:
    ss = FILTERS["stepSize"]
    q = (q/ss).to_integral_value(rounding=ROUND_DOWN) * ss
    return q if q >= FILTERS["minQty"] else Decimal("0")

def balances():
    acct = client.account()
    balmap = {b["asset"]: Decimal(b["free"]) for b in acct["balances"]}
    return balmap.get(BASE, Decimal("0")), balmap.get(QUOTE, Decimal("0"))

def open_orders():
    return client.get_open_orders(symbol=SYMBOL)

def get_price() -> Decimal:
    return Decimal(client.ticker_price(symbol=SYMBOL)["price"])

# =========================
# GRID LOGIC
# =========================
def compute_levels(anchor: Decimal):
    buy_lvls  = [round_price(anchor*(Decimal("1")+off)) for off in BUY_OFFSETS]
    sell_lvls = [round_price(anchor*(Decimal("1")+off)) for off in SELL_OFFSETS]
    return buy_lvls, sell_lvls

def adaptive_reset_state(st: dict, current: Decimal):
    anchor = st.get("anchor")
    if anchor is None:
        st["anchor"] = current
        st["buy_lvls"], st["sell_lvls"] = compute_levels(current)
        st["armed"] = {"buys":[True]*3, "sells":[True]*3}
        return False
    drift = abs(current - anchor)/anchor
    if drift > ADAPTIVE_SHIFT:
        logging.info(f"[{NETWORK}] Adaptive reset: old_anchor={anchor} new_anchor={current} drift={drift:.2%}")
        st["anchor"] = current
        st["buy_lvls"], st["sell_lvls"] = compute_levels(current)
        st["armed"] = {"buys":[True]*3, "sells":[True]*3}
        notify_whatsapp(f"♻️ Adaptive reset around {current}")
        return True
    return False

def ensure_grid(st: dict):
    base_free, quote_free = balances()
    last = get_price()

    # Buys
    for idx, level in enumerate(st["buy_lvls"], start=1):
        if not st["armed"]["buys"][idx-1]: continue
        if any(Decimal(o["price"])==level and o["side"]=="BUY" for o in open_orders()):
            continue
        qty = (quote_free*SPLITS[idx-1]*(Decimal("1")-FEE_BUFFER))/level
        if qty > 0:
            try:
                client.new_order(symbol=SYMBOL, side="BUY", type="LIMIT_MAKER",
                                 price=str(level), quantity=str(round_qty(qty)))
                logging.info(f"[{NETWORK}] BUY{idx} placed @ {level}")
            except Exception as e:
                logging.error(f"BUY{idx} failed: {e}")

    # Sells
    for idx, level in enumerate(st["sell_lvls"], start=1):
        if not st["armed"]["sells"][idx-1]: continue
        if any(Decimal(o["price"])==level and o["side"]=="SELL" for o in open_orders()):
            continue
        qty = base_free*SPLITS[idx-1]*(Decimal("1")-FEE_BUFFER)
        if qty > 0:
            try:
                client.new_order(symbol=SYMBOL, side="SELL", type="LIMIT_MAKER",
                                 price=str(level), quantity=str(round_qty(qty)))
                logging.info(f"[{NETWORK}] SELL{idx} placed @ {level}")
            except Exception as e:
                logging.error(f"SELL{idx} failed: {e}")

# =========================
# EMERGENCY EXIT
# =========================
price_history = []

def exit_all_positions():
    for od in open_orders():
        try: client.cancel_order(symbol=SYMBOL, orderId=od["orderId"])
        except: pass
    base_free, _ = balances()
    if base_free > 0:
        try:
            client.new_order(symbol=SYMBOL, side="SELL", type="MARKET",
                             quantity=str(round_qty(base_free)))
            logging.info(f"[{NETWORK}] Emergency exit {base_free} {BASE} at market")
        except Exception as e:
            logging.error(f"Emergency sell failed: {e}")

def check_emergency(st, px):
    global price_history
    now = time.time()
    price_history.append((now, px))
    price_history = [(t,p) for (t,p) in price_history if now - t <= CB_WINDOW]

    anchor = st.get("anchor")
    if anchor and px < anchor * (1 - STOPLOSS_PCT):
        logging.warning(f"[{NETWORK}] STOPLOSS triggered at {px}, anchor={anchor}")
        exit_all_positions()
        notify_whatsapp(f"💥 STOPLOSS triggered: price={px}, anchor={anchor}")
        return "stoploss"

    if price_history:
        high = max(p for (_,p) in price_history)
        low  = min(p for (_,p) in price_history)
        drop = (high-low)/high
        if drop > CB_DROP_PCT:
            logging.warning(f"[{NETWORK}] CIRCUIT BREAKER: drop={drop:.2%} in {CB_WINDOW}s")
            exit_all_positions()
            notify_whatsapp(f"⛔ Circuit breaker triggered, drop={drop:.2%}")
            time.sleep(SLEEP_ON_CB)
            return "circuit"

    return None

# =========================
# MAIN LOOP
# =========================
def main():
    st = {}
    logging.info(f"Started bot | network={NETWORK} | symbol={SYMBOL}")
    while True:
        try:
            px = get_price()
            event = check_emergency(st, px)
            if event:
                time.sleep(POLL_SECONDS)
                continue
            adaptive_reset_state(st, px)
            ensure_grid(st)
            logging.info(f"[{NETWORK}] Heartbeat price={px} anchor={st.get('anchor')}")
            time.sleep(POLL_SECONDS)
        except Exception as e:
            logging.error(f"main loop error: {e}")
            time.sleep(max(5, POLL_SECONDS))

if __name__ == "__main__":
    main()
