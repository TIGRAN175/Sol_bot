# bot.py
# SOL/USDT swing bot for Binance Spot
# Buys: 40% @ 170, 40% @ 165, 20% @ 160
# Sells: 40% @ 180, 40% @ 190, 20% @ 200 (fail-safe at 191.5 if price < 192)
# Reads config from .env (BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_TESTNET, SYMBOL, BASE, QUOTE, etc.)

import os, time, csv, json, logging
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from dotenv import load_dotenv, find_dotenv
from binance.spot import Spot as Binance

# =========================
# ENV + CLIENT
# =========================
ENV_PATH = Path(__file__).resolve().parent / ".env"
env_loaded_from = ENV_PATH if ENV_PATH.exists() else find_dotenv()
load_dotenv(env_loaded_from)

API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")
USE_TESTNET = (os.getenv("BINANCE_TESTNET", "false").lower() in ("true","1","yes"))

SYMBOL = os.getenv("SYMBOL", "SOLUSDT").upper()
BASE   = os.getenv("BASE", "SOL").upper()
QUOTE  = os.getenv("QUOTE", "USDT").upper()
MARKET = f"{BASE}/{QUOTE}"

BUY_SPLITS = [0.40, 0.40, 0.20]
BUY_LEVELS = [Decimal("170"), Decimal("165"), Decimal("160")]
SELL_SPLITS = [0.40, 0.40, 0.20]
SELL_LEVELS = [Decimal("180"), Decimal("190"), Decimal("200")]
TP3_FAILSAFE_TRIGGER = Decimal("192")
TP3_FAILSAFE_LIMIT   = Decimal("191.5")

POLL_SECONDS = int(os.getenv("POLL_SECONDS", "3"))
FEE_BUFFER   = Decimal(os.getenv("FEE_BUFFER", "0.0015"))

LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
STATE_DIR = Path(os.getenv("STATE_DIR", str(LOG_DIR)))
LOG_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

logfile = str(LOG_DIR / "solbot.log")
trades_csv_path = LOG_DIR / "trades.csv"
last_trade_id_path = STATE_DIR / "last_trade_id.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(logfile, mode="a"), logging.StreamHandler()]
)

def dt_iso(): return datetime.now(timezone.utc).isoformat()

def log_json(payload: dict, level=logging.INFO):
    try:
        logging.log(level, json.dumps(payload, separators=(",",":"), ensure_ascii=False))
    except Exception as e:
        logging.error(f"json_log_failed: {e} payload={payload}")

def log_trade_csv(side, price, qty, note, order_id=""):
    header = ["ts_iso","network","symbol","market","side","price","qty","quote_value","order_id","note"]
    write_header = not trades_csv_path.exists()
    with open(trades_csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header: w.writerow(header)
        w.writerow([dt_iso(), NETWORK, SYMBOL, MARKET, side.upper(), str(price), str(qty), str(price*qty), str(order_id), note])

if not API_KEY or not API_SECRET:
    raise SystemExit("Missing API keys")

client = Binance(api_key=API_KEY, api_secret=API_SECRET,
    base_url="https://testnet.binance.vision" if USE_TESTNET else None)

NETWORK = "TESTNET" if USE_TESTNET else "MAINNET"
logging.info(f"Started bot | network={NETWORK} | symbol={SYMBOL}")

ex_info = client.exchange_info()
SYMBOLS = {s["symbol"] for s in ex_info["symbols"]}
if SYMBOL not in SYMBOLS:
    raise SystemExit(f"Symbol {SYMBOL} not listed here")

# =========================
# EXCHANGE FILTERS
# =========================
def exchange_filters(symbol):
    info = client.exchange_info(symbol=symbol)
    s = info["symbols"][0]
    filters = {f["filterType"]: f for f in s["filters"]}
    min_notional = Decimal(filters.get("NOTIONAL",{}).get("minNotional",
                    filters.get("MIN_NOTIONAL",{}).get("minNotional","0")))
    return {
        "tickSize": Decimal(filters["PRICE_FILTER"]["tickSize"]),
        "stepSize": Decimal(filters["LOT_SIZE"]["stepSize"]),
        "minQty": Decimal(filters["LOT_SIZE"]["minQty"]),
        "minNotional": min_notional
    }

FILTERS = exchange_filters(SYMBOL)

def round_price(p): ts=FILTERS["tickSize"]; return (p/ts).to_integral_value(rounding=ROUND_DOWN)*ts
def round_qty(q): ss=FILTERS["stepSize"]; q=(q/ss).to_integral_value(rounding=ROUND_DOWN)*ss; return q if q>=FILTERS["minQty"] else Decimal("0")
def notional_ok(price, qty): return (price*qty)>=FILTERS["minNotional"]

def balances():
    acct = client.account()
    balmap = {b["asset"]: Decimal(b["free"]) for b in acct["balances"]}
    return balmap.get(BASE,Decimal("0")), balmap.get(QUOTE,Decimal("0"))

def open_orders(): return client.get_open_orders(symbol=SYMBOL)
def get_price(): return Decimal(client.ticker_price(symbol=SYMBOL)["price"])

def find_open_by_price(side, price, tol_ticks=0):
    oo=open_orders(); hits=[]; cmp=round_price(price)
    for od in oo:
        if od["side"].upper()!=side.upper(): continue
        op=Decimal(od["price"])
        if tol_ticks==0:
            if op==cmp: hits.append(od)
        else:
            ts=FILTERS["tickSize"]
            if abs(op-cmp)<=ts*tol_ticks: hits.append(od)
    return hits

# =========================
# MAKER SAFETY
# =========================
def maker_safe(side, level: Decimal, last: Decimal) -> bool:
    """LIMIT_MAKER rules: SELL must be above market, BUY must be below market"""
    return (level > last) if side.lower()=="sell" else (level < last)

# =========================
# ORDER ACTIONS
# =========================
def place_limit_maker(side, price, qty, tag=""):
    if qty<=0: return {}
    price, qty = round_price(price), round_qty(qty)
    if qty<=0 or not notional_ok(price,qty): return {}
    params={"symbol":SYMBOL,"side":side.upper(),"type":"LIMIT_MAKER",
            "price":str(price),"quantity":str(qty),
            "newClientOrderId":f"{tag}-{int(time.time())}"}
    try:
        od=client.new_order(**params)
        logging.info(f"[{NETWORK}] Placed {side} LIMIT_MAKER {qty}@{price} tag={tag} id={od.get('orderId')}")
        log_trade_csv(side,price,qty,f"placed {tag}",str(od.get("orderId","")))
        log_json({"ts":dt_iso(),"event":"order_placed","network":NETWORK,"symbol":SYMBOL,
                "market":MARKET,"side":side,"type":"LIMIT_MAKER","price":str(price),
                "qty":str(qty),"order_id":str(od.get("orderId","")), "tag":tag})
        return od
    except Exception as e:
        logging.error(f"[{NETWORK}] place_limit_maker failed: {e}")
        log_json({"ts":dt_iso(),"event":"error","where":"place_limit_maker","network":NETWORK,
                "symbol":SYMBOL,"market":MARKET,"side":side,"price":str(price),
                "qty":str(qty),"error":str(e),"tag":tag}, level=logging.ERROR)
        return {}

def place_limit(side, price, qty, tag=""):
    if qty<=0: return {}
    price, qty=round_price(price), round_qty(qty)
    if qty<=0 or not notional_ok(price,qty): return {}
    params={"symbol":SYMBOL,"side":side.upper(),"type":"LIMIT","timeInForce":"GTC",
            "price":str(price),"quantity":str(qty),
            "newClientOrderId":f"{tag}-{int(time.time())}"}
    try:
        od=client.new_order(**params)
        logging.info(f"[{NETWORK}] Placed {side} LIMIT {qty}@{price} tag={tag} id={od.get('orderId')}")
        log_trade_csv(side,price,qty,f"placed {tag}",str(od.get("orderId","")))
        log_json({"ts":dt_iso(),"event":"order_placed","network":NETWORK,"symbol":SYMBOL,
                "market":MARKET,"side":side,"type":"LIMIT","price":str(price),
                "qty":str(qty),"order_id":str(od.get("orderId","")), "tag":tag})
        return od
    except Exception as e:
        logging.error(f"[{NETWORK}] place_limit failed: {e}")
        return {}

def cancel_order(order_id):
    try:
        res=client.cancel_order(symbol=SYMBOL,orderId=order_id)
        logging.info(f"[{NETWORK}] Canceled order {order_id}")
        return res
    except Exception as e:
        logging.error(f"[{NETWORK}] cancel_order failed: {e}")
        return {}

# =========================
# STRATEGY
# =========================
def ensure_buy_grid():
    base_free, quote_free = balances()
    locked=Decimal("0")
    for od in open_orders():
        if od["side"].upper()=="BUY":
            locked+=Decimal(od["origQty"])*Decimal(od["price"])
    usable= max(Decimal("0"), quote_free-locked)
    last=get_price()

    for idx,(split,level) in enumerate(zip(BUY_SPLITS,BUY_LEVELS),start=1):
        tag=f"BUY{idx}"
        if find_open_by_price("BUY",level): continue
        alloc=usable*Decimal(str(split))
        if alloc<=0: continue
        qty=alloc*(Decimal("1")-FEE_BUFFER)/level
        if maker_safe("buy", level, last):
            place_limit_maker("BUY",level,qty,tag)
        else:
            logging.info(f"[{NETWORK}] Skip BUY {level} (maker-unsafe at last={last})")

def ensure_sell_grid():
    base_free,_=balances()
    locked=Decimal("0")
    for od in open_orders():
        if od["side"].upper()=="SELL":
            locked+=Decimal(od["origQty"])-Decimal(od["executedQty"])
    usable=max(Decimal("0"), base_free-locked)
    last=get_price()

    # TP1 + TP2
    for idx,(split,level) in enumerate(zip(SELL_SPLITS[:2],SELL_LEVELS[:2]),start=1):
        tag=f"SELL_TP{idx}"
        if find_open_by_price("SELL",level): continue
        qty=usable*Decimal(str(split))*(Decimal("1")-FEE_BUFFER)
        if maker_safe("sell", level, last):
            place_limit_maker("SELL",level,qty,tag)
        else:
            logging.info(f"[{NETWORK}] Skip SELL {level} (maker-unsafe at last={last})")

    # TP3 w/failsafe
    third_split=SELL_SPLITS[2]
    qty_tp3=usable*Decimal(str(third_split))*(Decimal("1")-FEE_BUFFER)
    tp3_exists=bool(find_open_by_price("SELL",SELL_LEVELS[2]))
    fs_exists=bool(find_open_by_price("SELL",TP3_FAILSAFE_LIMIT))

    if not tp3_exists and not fs_exists:
        if last < TP3_FAILSAFE_TRIGGER:
            place_limit("SELL",TP3_FAILSAFE_LIMIT,qty_tp3,"SELL_TP3_FS")
        else:
            if maker_safe("sell", SELL_LEVELS[2], last):
                place_limit_maker("SELL",SELL_LEVELS[2],qty_tp3,"SELL_TP3")
            else:
                logging.info(f"[{NETWORK}] Skip SELL {SELL_LEVELS[2]} (maker-unsafe at last={last})")
    elif tp3_exists and last < TP3_FAILSAFE_TRIGGER:
        for od in find_open_by_price("SELL",SELL_LEVELS[2]):
            cancel_order(od["orderId"])
        time.sleep(0.5)
        base_free,_=balances()
        qty_tp3=base_free*Decimal(str(third_split))*(Decimal("1")-FEE_BUFFER)
        place_limit("SELL",TP3_FAILSAFE_LIMIT,qty_tp3,"SELL_TP3_FS")

# =========================
# MAIN LOOP
# =========================
def main():
    acct=client.account()
    logging.info(f"[{NETWORK}] Startup balances:")
    for b in acct["balances"]:
        if float(b["free"])+float(b["locked"])>0:
            logging.info(f"   {b['asset']}: free={b['free']} locked={b['locked']}")
    log_json({"ts":dt_iso(),"event":"bot_start","network":NETWORK,"symbol":SYMBOL,"market":MARKET})
    logging.info(f"[{NETWORK}] Running… CTRL+C to stop")

    while True:
        try:
            price=get_price()
            ensure_buy_grid()
            ensure_sell_grid()
            base_free,quote_free=balances()
            open_cnt=len(open_orders())
            logging.info(f"[{NETWORK}] Price={price:.2f} | {BASE}={base_free:.4f} | {QUOTE}={quote_free:.2f} | open_orders={open_cnt}")
            log_json({"ts":dt_iso(),"event":"heartbeat","network":NETWORK,"symbol":SYMBOL,
                      "market":MARKET,"price":str(price),"balances":{BASE:str(base_free),QUOTE:str(quote_free)},"open_orders":open_cnt})
            time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            logging.info(f"[{NETWORK}] Stopping…")
            log_json({"ts":dt_iso(),"event":"bot_stop","network":NETWORK,"symbol":SYMBOL,"market":MARKET})
            break
        except Exception as e:
            logging.error(f"[{NETWORK}] main loop error: {e}")
            log_json({"ts":dt_iso(),"event":"error","where":"main_loop","network":NETWORK,"symbol":SYMBOL,"market":MARKET,"error":str(e)}, level=logging.ERROR)
            time.sleep(max(5,POLL_SECONDS))

if __name__=="__main__":
    main()
