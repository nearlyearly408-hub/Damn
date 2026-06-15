"""
Bot Scalping Quant v23.0.0 — HIGH PROBABILITY SCALP
===================================================
STRATEGI IDE #5:
- SL lebih pendek (0.10%) → loss lebih kecil
- TP lebih pendek (0.40%) → profit lebih cepat
- Filter SUPER KETAT → cuma ambil setup terbaik
- Target win rate 55-65% (bukan 40-50%)
- Lebih sedikit trade tapi profit konsisten
"""

import os, time, math, threading, queue
import requests
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from binance.client import Client
import ta
import pandas as pd
import numpy as np

load_dotenv()
client = Client(os.getenv("API_KEY"), os.getenv("API_SECRET"))
client.FUTURES_URL = "https://fapi.binance.com/fapi"

# ═══════════════════════════════════════════════════════
#  CONFIG v23.0.0 — HIGH PROBABILITY SCALP
# ═══════════════════════════════════════════════════════
LEVERAGE       = 5           
ORDER_USDT     = 2.0
MAX_POSITIONS  = 1           
FUTURES_FEE_PCT = 0.0005     # 0.05% per side (0.10% round trip)

SCAN_INTERVAL  = 0.5     
MONITOR_INT    = 0.05    
SCAN_DELAY     = 0.005   
BATCH_SIZE     = 30      
MAX_WORKERS    = 15      
SLOT_FILL_INT  = 0.05    

# 🔥 PERUBAHAN IDE #5: Filter SUPER KETAT
MIN_SCORE      = 85          # Naik dari 75 (cuma ambil sinyal terbaik)
MIN_GAP        = 5
SLIPPAGE_GUARD = 0.0015  
TTL_5M         = 2       

DAILY_LOSS     = -10.0       
CONSEC_MAX     = 2           

# 🔥 PERUBAHAN IDE #5: TP/SL lebih pendek & efisien
# SL 0.10% | TP 0.40%
# Loss net = 0.10% + 0.10% = 0.20%
# Profit net = 0.40% - 0.10% = 0.30%
# RR = 0.30 / 0.20 = 1.5
# Break even win rate = 0.20 / (0.30 + 0.20) = 40%
# Target win rate 55% → EV positif!
FIXED_SL_PCT = 0.0010   # 0.10% (lebih pendek)
FIXED_TP_PCT = 0.0040   # 0.40% (lebih pendek)

# ═══════════════════════════════════════════════════════
#  SYMBOLS (fokus ke yang likuid)
# ═══════════════════════════════════════════════════════
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "TRXUSDT", "DOTUSDT",
    "LINKUSDT", "MATICUSDT", "LTCUSDT", "ATOMUSDT", "UNIUSDT",
]

# ═══════════════════════════════════════════════════════
#  STATE & MEMORY
# ═══════════════════════════════════════════════════════
live_positions  = {}
trade_log       = []
_ohlcv_cache    = {}
_ticker_cache   = {}
_ticker_ts      = 0
_lock           = threading.Lock()
_executor       = ThreadPoolExecutor(max_workers=MAX_WORKERS)
_rescan_q       = queue.Queue()
_hot_syms       = deque(maxlen=20)

_past_trades    = deque(maxlen=50)
_cooldowns      = {}
_macro = {"fng": 50, "btc": "UNKNOWN", "last_fng": 0, "last_btc": 0}

_prot = {
    "consec_loss": 0, 
    "consec_win": 0, 
    "active_min_score": MIN_SCORE, 
    "pause_until": 0,
    "size_multiplier": 1.0
}
_ks    = {"active": False, "reason": "", "resume": 0, "consec": 0, "daily": 0.0, "day_reset": 0}
_stats = {
    "trades": 0, "wins": 0, "losses": 0, "pnl": 0.0, "best": 0.0, "worst": 0.0,
    "extreme_tp": 0, "hard_sl": 0, "force": 0, "btc_block": 0,
    "hist": deque(maxlen=200), "start": time.time(),
}

# ═══════════════════════════════════════════════════════
#  BINANCE UTILS
# ═══════════════════════════════════════════════════════
_precision_cache = {}
def get_precision(symbol):
    if symbol in _precision_cache: return _precision_cache[symbol]
    try:
        info = client.futures_exchange_info()
        for s in info['symbols']:
            if s['symbol'] == symbol:
                prec = int(s['quantityPrecision'])
                _precision_cache[symbol] = prec
                return prec
    except: pass
    return 2

def qty(symbol, price):
    size_mult = _prot["size_multiplier"]
    raw_qty = (ORDER_USDT * LEVERAGE * size_mult) / price
    return round(raw_qty, get_precision(symbol))

def price_live(symbol):
    try: return float(client.futures_symbol_ticker(symbol=symbol)["price"])
    except: return 0.0

def tickers_all():
    global _ticker_cache, _ticker_ts
    now = time.time()
    if now - _ticker_ts < 2 and _ticker_cache: return _ticker_cache
    try:
        raw = client.futures_ticker()
        _ticker_cache = {
            t["symbol"]: {
                "pct": float(t["priceChangePercent"]),
                "vol": float(t["quoteVolume"]),
                "last": float(t["lastPrice"])
            } for t in raw
        }
        _ticker_ts = now
        return _ticker_cache
    except: return _ticker_cache

def ohlcv(symbol, interval, limit=100):
    key, now = (symbol, interval), time.time()
    if key in _ohlcv_cache and now - _ohlcv_cache[key][0] < TTL_5M:
        return _ohlcv_cache[key][1]
    try:
        kl = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(kl, columns=["time","open","high","low","close","volume",
                                        "ct","qv","trades","tbbase","tbquote","ignore"])
        for c in ["open","high","low","close","volume","tbbase","tbquote"]:
            df[c] = df[c].astype(float)
        _ohlcv_cache[key] = (now, df)
        return df
    except: return _ohlcv_cache.get(key, (None, None))[1]

def run_ta(df):
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    df["rsi"]  = ta.momentum.RSIIndicator(c, 7).rsi()
    df["mh"]   = ta.trend.MACD(c, 12, 26, 9).macd_diff()
    df["e5"]   = ta.trend.EMAIndicator(c, 5).ema_indicator()
    df["e9"]   = ta.trend.EMAIndicator(c, 9).ema_indicator()
    df["e21"]  = ta.trend.EMAIndicator(c, 21).ema_indicator()
    df["e50"]  = ta.trend.EMAIndicator(c, 50).ema_indicator()
    df["atr"]  = ta.volatility.AverageTrueRange(h, l, c, 7).average_true_range()
    df["adx"]  = ta.trend.ADXIndicator(h, l, c, 14).adx()
    df["vm"]   = v.rolling(20).mean()
    df["vr"]   = v / df["vm"].replace(0, 1)
    df["body"] = abs(c - df["open"])
    df["rng"]  = h - l
    df["br2"]  = df["body"] / df["rng"].replace(0, 1)
    df["m5"]   = (c - c.shift(5)) / c.shift(5)
    df["m1"]   = (c - c.shift(1)) / c.shift(1)
    
    # 🔥 TAMBAHAN IDE #5: Wick ratio (candle harus closed di ujung)
    df["upper_wick"] = df["high"] - df[["close", "open"]].max(axis=1)
    df["lower_wick"] = df[["close", "open"]].min(axis=1) - df["low"]
    df["wick_ratio"] = (df["upper_wick"] + df["lower_wick"]) / df["rng"].replace(0, 1)
    
    return df

def btc_trend():
    try:
        df = run_ta(ohlcv("BTCUSDT", Client.KLINE_INTERVAL_5MINUTE, 80).copy())
        row = df.iloc[-2]
        p, e5, e9, e21, m5 = row["close"], row["e5"], row["e9"], row["e21"], row["m5"]
        if p > e5 > e9 > e21 and m5 > 0.001: return "BULL"
        if p < e5 < e9 < e21 and m5 < -0.001: return "BEAR"
        return "SIDEWAYS"
    except: return "UNKNOWN"

def ks_check():
    k, now = _ks, time.time()
    if _prot["pause_until"] > now:
        return True, "PROT_PAUSE(5m)"

    if k["active"] and now >= k["resume"]:
        k["active"] = False; k["consec"] = 0
    if k["active"]: return True, k["reason"]
    day = now - (now % 86400)
    if day > k["day_reset"]: k["daily"] = 0.0; k["day_reset"] = day
    if k["daily"] <= DAILY_LOSS:
        k["active"] = True
        k["reason"] = f"daily({k['daily']:.2f})"
        k["resume"] = day + 86400
        return True, k["reason"]
    return False, ""

def ks_upd(pnl):
    _ks["daily"] += pnl

# ═══════════════════════════════════════════════════════
#  SIGNAL ENGINE — FILTER SUPER KETAT (IDE #5)
# ═══════════════════════════════════════════════════════
def signal(df, symbol=None):
    if df is None or len(df) < 55: return None, 0, [], 0.0

    row  = df.iloc[-2]
    prev = df.iloc[-3]
    curr = df.iloc[-1]

    p, e5, e9, e21, e50 = row["close"], row["e5"], row["e9"], row["e21"], row["e50"]
    rsi  = row["rsi"]; adx = row["adx"]; vr = row["vr"]
    mh   = row["mh"]; mh_p = prev["mh"]
    atr  = row["atr"]; body = row["br2"]
    m1   = row["m1"]
    wick_ratio = row["wick_ratio"]
    
    btc = _macro.get("btc", "UNKNOWN")
    
    # 🔥 FILTER SUPER KETAT (IDE #5)
    # 1. Volume harus 2x normal (naik dari 1.3)
    if vr <= 2.0:
        return None, 0, [], 0.0
    
    # 2. Body candle minimal 60% dari range (tidak boleh long wick)
    if body < 0.6:
        return None, 0, [], 0.0
    
    # 3. Wick ratio harus kecil (<0.3 artinya 70% body)
    if wick_ratio > 0.3:
        return None, 0, [], 0.0
    
    # 4. ADX minimal 30 (trend lebih kuat)
    if adx <= 30:
        return None, 0, [], 0.0
    
    # 5. Momentum 1 candle minimal 0.20% (naik dari 0.15%)
    if abs(m1) < 0.002:
        return None, 0, [], 0.0
    
    # 6. ATR minimal 0.25%
    atr_pct = atr / p if p > 0 else 0
    if atr_pct < 0.0025:
        return None, 0, [], 0.0
    
    # 🔥 KONFIRMASI EKSTRA: Candle harus closed di HIGH/LOW
    candle_close_at_high = abs(row["close"] - row["high"]) / row["atr"] < 0.2 if row["atr"] > 0 else False
    candle_close_at_low = abs(row["close"] - row["low"]) / row["atr"] < 0.2 if row["atr"] > 0 else False
    
    l_valid, s_valid = False, False
    
    # LONG (hanya ambil yang close di HIGH)
    if p > e5 > e9 > e21 and m1 > 0.002 and candle_close_at_high:
        if 45 <= rsi <= 70 and mh > mh_p:
            l_valid = True
    
    # SHORT (hanya ambil yang close di LOW)
    if p < e5 < e9 < e21 and m1 < -0.002 and candle_close_at_low:
        if 30 <= rsi <= 55 and mh < mh_p:
            s_valid = True
    
    # BTC FILTER lebih ketat
    if btc == "BEAR": l_valid = False
    if btc == "BULL": s_valid = False
    if btc == "SIDEWAYS": 
        l_valid = False
        s_valid = False
    
    if not l_valid and not s_valid:
        return None, 0, [], 0.0
    
    # 🔥 SKOR dengan bobot lebih tinggi
    score = 40
    if vr > 2.5: score += 15
    if adx > 35: score += 15
    if body > 0.75: score += 15  # Candle hampir full body
    if abs(m1) > 0.0035: score += 15
    if btc == "BULL" and l_valid: score += 15
    elif btc == "BEAR" and s_valid: score += 15
    if wick_ratio < 0.15: score += 10  # Wick sangat kecil
    
    if score < _prot["active_min_score"]:
        return None, 0, [], 0.0
    
    # COOLDOWN lebih lama (10 menit)
    if symbol in _cooldowns and time.time() < _cooldowns[symbol]:
        return None, 0, [], 0.0
    
    # Hitung expected value dengan TP/SL baru
    net_profit = FIXED_TP_PCT - FUTURES_FEE_PCT
    net_loss = FIXED_SL_PCT + FUTURES_FEE_PCT
    rr = net_profit / net_loss if net_loss > 0 else 0
    
    if l_valid:
        return "LONG", score, [f"ADX:{adx:.0f}", f"Mom:{m1*100:.2f}%", f"Body:{body:.0%}", f"RR:{rr:.2f}"], atr
    else:
        return "SHORT", score, [f"ADX:{adx:.0f}", f"Mom:{m1*100:.2f}%", f"Body:{body:.0%}", f"RR:{rr:.2f}"], atr

# ═══════════════════════════════════════════════════════
#  OPEN POSITION
# ═══════════════════════════════════════════════════════
def live_open(sym, direction, score, sigs, price, atr):
    with _lock:
        if sym in live_positions or len(live_positions) >= MAX_POSITIONS: return
        live_positions[sym] = {"_r": True}
    
    px_now = price_live(sym)
    if px_now > 0:
        slip = abs(px_now - price) / price
        if slip > SLIPPAGE_GUARD:
            with _lock: live_positions.pop(sym, None)
            return
        price = px_now
    
    try: q_val = qty(sym, price)
    except:
        with _lock: live_positions.pop(sym, None)
        return
    
    # FIXED TP/SL dengan value baru
    if direction == "LONG":
        sl_price = price * (1 - FIXED_SL_PCT)
        tp_price = price * (1 + FIXED_TP_PCT)
    else:
        sl_price = price * (1 + FIXED_SL_PCT)
        tp_price = price * (1 - FIXED_TP_PCT)
    
    # Hitung net setelah fee
    net_profit = FIXED_TP_PCT - FUTURES_FEE_PCT
    net_loss = FIXED_SL_PCT + FUTURES_FEE_PCT
    rr_net = net_profit / net_loss if net_loss > 0 else 0
    
    pos = {
        "side": direction, "entry": price, "qty": q_val,
        "open_time": time.time(), "score": score, "sigs": sigs, "atr": atr,
        "sl_price": sl_price, "tp_price": tp_price,
        "sl_pct": FIXED_SL_PCT, "tp_pct": FIXED_TP_PCT, "max_prof": 0.0, "stage": 0
    }
    with _lock: live_positions[sym] = pos
    
    d = "🟢" if direction == "LONG" else "🔴"
    print(f"\n  {d} [OPEN] {sym} {direction} @{price:.6g}")
    print(f"       SL:{FIXED_SL_PCT*100:.2f}% | TP:{FIXED_TP_PCT*100:.2f}%")
    print(f"       Net Loss:{net_loss*100:.2f}% | Net Profit:{net_profit*100:.2f}% | RR:1:{rr_net:.2f}")
    print(f"       Break-even WR:{net_loss/(net_profit+net_loss)*100:.0f}%")
    print(f"       Signal: {', '.join(sigs)} | Score: {score}")
    _stats["trades"] += 1

# ═══════════════════════════════════════════════════════
#  CLOSE POSITION & ADAPTIVE
# ═══════════════════════════════════════════════════════
def live_close(sym, reason, price=None):
    with _lock: pos = live_positions.pop(sym, None)
    if pos is None or pos.get("_r"): return
    
    if price is None: price = price_live(sym)
    if price == 0: return
    
    side, entry, q_val = pos["side"], pos["entry"], pos["qty"]
    gross_pnl = (price - entry) * q_val if side == "LONG" else (entry - price) * q_val
    total_fee = ((entry * q_val) + (price * q_val)) * FUTURES_FEE_PCT
    pnl = gross_pnl - total_fee
    
    pct   = (price - entry) / entry * 100 if side == "LONG" else (entry - price) / entry * 100
    hold  = time.time() - pos["open_time"]
    e = "🟢" if pnl >= 0 else "🔴"
    
    print(f"  {e} [CLOSE] {sym} {side} — {reason}")
    print(f"       {entry:.6g}→{price:.6g} ({pct:+.3f}%) hold:{hold:.0f}s | Net:{pnl:+.5f}U")
    
    _stats["pnl"]  += pnl
    ks_upd(pnl)
    
    _past_trades.append({
        "sym": sym, "score": pos["score"], "pnl": pnl
    })
    
    # ADAPTIVE LOGIC yang lebih agresif untuk profit
    if pnl >= 0:
        _stats["wins"] += 1
        _prot["consec_win"] += 1
        _prot["consec_loss"] = 0
        
        # 🔥 Naikin size lebih cepat saat profit streak
        _prot["size_multiplier"] = min(1.5, _prot["size_multiplier"] + 0.15)
        
        # 🔥 Turunkan threshold lebih agresif
        if _prot["consec_win"] >= 2:
            _prot["active_min_score"] = max(MIN_SCORE - 15, 70)
            print(f"  ✅ Win streak {_prot['consec_win']} → Score turun ke {_prot['active_min_score']} | Size:{_prot['size_multiplier']:.1f}x")
    else:
        _stats["losses"] += 1
        _prot["consec_loss"] += 1
        _prot["consec_win"] = 0
        
        # Kurangi size drastis saat loss
        _prot["size_multiplier"] = max(0.3, _prot["size_multiplier"] - 0.3)
        print(f"  ⚠️ Loss! Size multiplier: {_prot['size_multiplier']:.1f}x")
        
        # 🔥 Cooldown lebih lama untuk pair loss (15 menit)
        if "SL" in reason or "HardSL" in reason: 
            _cooldowns[sym] = time.time() + 900
            print(f"  ⏸️ {sym} cooldown 15 menit")
        
        # 🔥 Pause lebih lama setelah loss streak
        if _prot["consec_loss"] >= CONSEC_MAX:
            _prot["pause_until"] = time.time() + 600   # Pause 10 menit
            _prot["consec_loss"] = 0
            print("  🚨 2 Loss berturut-turut! Bot pause 10 menit.")
        elif _prot["consec_loss"] >= 1:
            _prot["active_min_score"] = min(_prot["active_min_score"] + 10, 95)
            print(f"  ⚠️ Loss → Min Score naik ke {_prot['active_min_score']}")
    
    trade_log.append({
        "sym": sym, "side": side, "entry": round(entry, 7), "exit": round(price, 7),
        "pnl": round(pnl, 5), "reason": reason, "hold": int(hold),
    })
    _hot_syms.appendleft(sym)
    _rescan_q.put(1)

# ═══════════════════════════════════════════════════════
#  MONITOR POSITIONS — OPTIMIZED FOR SMALLER TP
# ═══════════════════════════════════════════════════════
def monitor_positions():
    for sym in list(live_positions.keys()):
        pos = live_positions.get(sym)
        if pos is None or pos.get("_r"): continue
        
        px = price_live(sym)
        if px == 0: continue
        
        side, entry = pos["side"], pos["entry"]
        sl_px, tp_px = pos["sl_price"], pos["tp_price"]
        
        prof_pct = (px - entry) / entry if side == "LONG" else (entry - px) / entry
        pos["max_prof"] = max(pos["max_prof"], prof_pct)
        
        stage = pos["stage"]
        
        # 🔥 BREAK EVEN lebih cepat — di +0.05% (karena TP lebih kecil)
        if pos["max_prof"] >= 0.0005 and stage < 1:
            pos["stage"] = 1
            buffer = 0.0003  # 0.03% buffer
            if side == "LONG": 
                pos["sl_price"] = entry * (1 + buffer)
            else: 
                pos["sl_price"] = entry * (1 - buffer)
            print(f"  🛡️ {sym} Break-Even aktif (profit:{prof_pct:.3%})")
        
        # 🔥 LOCK PROFIT — di +0.15%
        if pos["max_prof"] >= 0.0015 and stage < 2:
            pos["stage"] = 2
            lock_pct = 0.0010  # Lock 0.10% dari puncak
            if side == "LONG":
                pos["sl_price"] = max(pos["sl_price"], px * (1 - lock_pct))
            else:
                pos["sl_price"] = min(pos["sl_price"], px * (1 + lock_pct))
            print(f"  🔒 {sym} Profit terkunci minimal 0.10%")
        
        # 🔥 TRAILING — di +0.25% (trail 0.05% saja karena TP 0.40%)
        if pos["max_prof"] >= 0.0025 and stage < 3:
            pos["stage"] = 3
            print(f"  📈 {sym} Trailing aktif (trail 0.05%)")
        
        if pos["stage"] == 3:
            trail_dist = 0.0005  # 0.05% trailing (lebih ketat)
            if side == "LONG":
                pos["sl_price"] = max(pos["sl_price"], px * (1 - trail_dist))
            else:
                pos["sl_price"] = min(pos["sl_price"], px * (1 + trail_dist))
        
        # Eksekusi exit
        if side == "LONG":
            if px <= sl_px:
                reason = "TrailSL" if stage >= 2 else ("BESL" if stage == 1 else "HardSL")
                live_close(sym, reason, px)
                continue
            if px >= tp_px:
                live_close(sym, "TakeProfit", px)
                continue
        else:
            if px >= sl_px:
                reason = "TrailSL" if stage >= 2 else ("BESL" if stage == 1 else "HardSL")
                live_close(sym, reason, px)
                continue
            if px <= tp_px:
                live_close(sym, "TakeProfit", px)
                continue

# ═══════════════════════════════════════════════════════
#  SCANNER & THREADS (sama seperti sebelumnya)
# ═══════════════════════════════════════════════════════
def scan_one(sym):
    try:
        time.sleep(SCAN_DELAY)
        df5 = run_ta(ohlcv(sym, Client.KLINE_INTERVAL_5MINUTE, 100).copy())
        if df5 is None: return None
        px  = df5["close"].iloc[-2]
        if px == 0: return None
        dir_, sc, sigs, atr_val = signal(df5, sym)
        if dir_ is None: return None
        px_live = price_live(sym)
        if px_live == 0: return None
        return (sym, dir_, sc, sigs, px_live, atr_val)
    except: return None

def scan_batch(syms):
    res = []
    fut = {_executor.submit(scan_one, s): s for s in syms[:BATCH_SIZE]}
    try:
        for f in as_completed(fut, timeout=5):
            if r := f.result(timeout=1): res.append(r)
    except: pass
    return res

def top_movers(syms, n=20):
    tk, ss = tickers_all(), set(syms)
    mv = [(s, abs(d["pct"])) for s, d in tk.items() if s in ss]
    return [s for s, _ in sorted(mv, key=lambda x: x[1], reverse=True)[:n]]

def print_stats():
    n    = _stats["wins"] + _stats["losses"]
    wr   = _stats["wins"] / n * 100 if n else 0
    pnl  = _stats["pnl"]
    
    gross_profit = sum(t["pnl"] for t in trade_log if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in trade_log if t["pnl"] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 0
    
    avg_win = gross_profit / _stats["wins"] if _stats["wins"] > 0 else 0
    avg_loss = gross_loss / _stats["losses"] if _stats["losses"] > 0 else 0
    ev = (avg_win * _stats["wins"] - avg_loss * _stats["losses"]) / n if n > 0 else 0
    
    # Hitung dengan parameter baru
    net_loss = FIXED_SL_PCT + FUTURES_FEE_PCT
    net_profit = FIXED_TP_PCT - FUTURES_FEE_PCT
    be_wr = net_loss / (net_profit + net_loss) * 100
    
    e    = "💚" if pnl >= 0 else "🔴"
    print(f"\n  {'─'*68}")
    print(f"    🔥 QUANT v23.0.0 [HIGH PROBABILITY SCALP]")
    print(f"    🎯 {n}T | WR:{wr:.1f}% | W:{_stats['wins']} L:{_stats['losses']}")
    print(f"    {e} PnL Net:{pnl:+.5f}U | Profit Factor:{pf:.2f} | EV:{ev:+.5f}U")
    print(f"    📊 Break-even WR:{be_wr:.0f}% | Current WR:{wr:.0f}% | Need:>{be_wr+5:.0f}%")
    if trade_log:
        print(f"    📋 Last 5:")
        for t in trade_log[-5:]:
            em = "🟢" if t["pnl"] > 0 else "🔴"
            print(f"       {em} {t['sym']:<16} {t['side']} {t['pnl']:+.5f}U {t['hold']}s — {t['reason']}")
    print(f"  {'─'*68}")

def t_monitor():
    while True:
        try:
            if live_positions: monitor_positions()
        except: pass
        time.sleep(MONITOR_INT)

def t_slot_filler(syms):
    scan_idx = 0
    n_bat    = math.ceil(len(syms) / BATCH_SIZE)
    while True:
        try:
            slots = MAX_POSITIONS - len(live_positions)
            if slots <= 0 or ks_check()[0]:
                time.sleep(SLOT_FILL_INT)
                continue
            
            hot  = [s for s in _hot_syms if s not in live_positions]
            mv   = [s for s in top_movers(syms, 20) if s not in live_positions]
            bs   = scan_idx * BATCH_SIZE
            reg  = [s for s in syms[bs:bs+BATCH_SIZE] if s not in live_positions and s not in mv]
            scan_idx = (scan_idx + 1) % n_bat
            
            scan_list = list(dict.fromkeys(hot[:3] + mv[:10] + reg[:10]))[:BATCH_SIZE]
            if not scan_list:
                time.sleep(SLOT_FILL_INT)
                continue
            
            res = scan_batch(scan_list)
            if res:
                res.sort(key=lambda x: x[2], reverse=True)
                for r in res[:slots]:
                    if len(live_positions) >= MAX_POSITIONS: break
                    sym, d, sc, sg, px, atr = r
                    live_open(sym, d, sc, sg, px, atr)
        except Exception as e:
            pass
        time.sleep(SLOT_FILL_INT)

def t_rescan(syms):
    while True:
        try:
            _rescan_q.get(timeout=5)
            time.sleep(0.1)
            slots = MAX_POSITIONS - len(live_positions)
            if slots <= 0 or ks_check()[0]: continue
            
            hot  = [s for s in _hot_syms if s not in live_positions]
            rest = [s for s in syms if s not in live_positions and s not in hot]
            res  = scan_batch((hot + rest)[:20])
            if res:
                res.sort(key=lambda x: x[2], reverse=True)
                for r in res[:slots]:
                    if len(live_positions) >= MAX_POSITIONS: break
                    sym, d, sc, sg, px, atr = r
                    live_open(sym, d, sc, sg, px, atr)
        except: pass

def t_macro():
    while True:
        try: 
            _macro["btc"] = btc_trend()
        except: pass
        time.sleep(10)

# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════
def run_bot():
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  🔥 QUANT v23.0.0 — HIGH PROBABILITY SCALP (IDE #5)           ║")
    print("║  ===========================================================  ║")
    print("║  🎯 SL:0.10% | TP:0.40% | Net RR:1:1.5                        ║")
    print("║  📊 Break-even WR: 40% | Target WR: 55%+                      ║")
    print("║  💰 Filter SUPER KETAT → Lebih sedikit trade, profit konsisten ║")
    print("║  🔥 Cuma ambil sinyal dengan skor > 85                        ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    
    syms = list(dict.fromkeys(SYMBOLS))
    print(f"\n  📋 {len(syms)} simbol terpantau | Leverage: {LEVERAGE}x")
    print(f"  🛡️ Max Loss harian: ${DAILY_LOSS} | Max consecutive loss: {CONSEC_MAX}")
    print(f"  💰 Fee round trip: {FUTURES_FEE_PCT*2*100:.2f}%")
    print(f"  🎯 Minimal Score: {MIN_SCORE} | Dynamic: {_prot['active_min_score']}-95")
    
    # Hitung dan tampilkan expected value
    net_loss = FIXED_SL_PCT + FUTURES_FEE_PCT
    net_profit = FIXED_TP_PCT - FUTURES_FEE_PCT
    ev_per_trade_55wr = (0.55 * net_profit) - (0.45 * net_loss)
    ev_per_trade_60wr = (0.60 * net_profit) - (0.40 * net_loss)
    
    print(f"\n  📈 EXPECTED VALUE (dengan leverage {LEVERAGE}x):")
    print(f"     Win rate 55% → EV: {ev_per_trade_55wr*LEVERAGE*100:.2f}% per trade")
    print(f"     Win rate 60% → EV: {ev_per_trade_60wr*LEVERAGE*100:.2f}% per trade")
    
    threading.Thread(target=t_monitor,         daemon=True).start()
    threading.Thread(target=t_slot_filler, args=(syms,), daemon=True).start()
    threading.Thread(target=t_rescan,      args=(syms,), daemon=True).start()
    threading.Thread(target=t_macro,                     daemon=True).start()
    
    time.sleep(2)
    tickers_all()
    
    cycle = 0
    while True:
        cycle += 1
        slots = MAX_POSITIONS - len(live_positions)
        print(f"\n{'═'*62}")
        
        n = _stats["wins"] + _stats["losses"]
        wr = _stats["wins"] / n * 100 if n > 0 else 0
        
        print(f"  #{cycle} {time.strftime('%H:%M:%S')}")
        print(f"  BTC:{_macro['btc']:<10} | MinScore:{_prot['active_min_score']:<3} | Slots:{len(live_positions)}/{MAX_POSITIONS}")
        print(f"  PnL:{_stats['pnl']:+.4f}U | WR:{wr:.0f}% | W:{_stats['wins']} L:{_stats['losses']}")
        print(f"  Size Mult:{_prot['size_multiplier']:.1f}x | Target WR:55%+")
        
        ks_status, ks_reason = ks_check()
        if ks_status: 
            print(f"  🚨 SYSTEM LOCK: {ks_reason}")
        elif slots == 0: 
            print(f"  ✅ Slots full — Monitoring {len(live_positions)} position(s)")
        else: 
            print(f"  🔍 {slots} slot kosong — Scanning with HIGH filter...")
        
        if cycle % 25 == 0: 
            print_stats()
        
        time.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    run_bot()